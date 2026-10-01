"""Head data for one wake word: every synthetic clip trimmed and placed at random in a 1.5 s window (a tenth held
out for validation, chosen by file), negatives from the shared negative windows; writes words/<w>-head-rp/train.csv
and val.csv.

    prep_word_windows.py <word> <src_dir> [<src_dir> ...] [neg:<dir> ...] [--path-map OLD=NEW ...]

A source written ``neg:<dir>`` is a set of sound-alike phrases: its clips are windowed the same way and labelled 0,
so the head learns to reject them as it learns to accept the word. The shared negative windows come from
datasets/train.csv and datasets/val.csv; ``--path-map OLD=NEW`` rewrites a path prefix in them when the lists were
written on another machine.
"""
import csv, glob, os, random, sys
import numpy as np, soundfile as sf
N = 24000
args = sys.argv[1:]
maps = [args[i + 1].split("=", 1) for i, a in enumerate(args) if a == "--path-map"]
args = [a for i, a in enumerate(args) if a != "--path-map" and (i == 0 or args[i - 1] != "--path-map")]
word, srcs = args[0], args[1:]
out = f"words/{word}-head-rp"


def clips(dirs):
    return sorted(f for s in dirs for ext in ("wav", "flac") for f in glob.glob(f"{s}/**/*.{ext}", recursive=True))


def windows(files, sub, label, seed):
    os.makedirs(f"{out}/{sub}", exist_ok=True)
    files = list(files); random.Random(seed).shuffle(files)
    rows = []
    for i, f in enumerate(files):
        try:
            w, sr = sf.read(f, dtype="float32", always_2d=True)
        except Exception:
            continue
        w = w.mean(1)
        if sr != 16000:
            import torch, torchaudio
            w = torchaudio.functional.resample(torch.from_numpy(w), sr, 16000).numpy()
        a = np.abs(w); nz = np.nonzero(a > 0.02 * a.max())[0]
        w = w[nz[0]:nz[-1] + 1] if len(nz) else w
        o = np.zeros(N, np.float32); w = w[:N]; s = random.Random(i).randint(0, N - len(w)); o[s:s + len(w)] = w  # random placement, as real audio has
        p = f"{out}/{sub}/{i:05d}.wav"; sf.write(p, o, 16000); rows.append((os.path.abspath(p), label, i % 10 == 0))
    return rows


pos = windows(clips([s for s in srcs if not s.startswith("neg:")]), "pos", 1, 0)
hard = windows(clips([s[4:] for s in srcs if s.startswith("neg:")]), "neg", 0, 1)


def remap(path):
    for old, new in maps:
        if path.startswith(old):
            return new + path[len(old):]
    return path


neg = lambda m: [(remap(r[0]), 0) for r in csv.reader(open(m)) if r and r[1] == "0"]
with open(f"{out}/train.csv", "w", newline="") as fh:
    csv.writer(fh).writerows([(p, y) for p, y, v in pos + hard if not v] + neg("datasets/train.csv"))
with open(f"{out}/val.csv", "w", newline="") as fh:
    csv.writer(fh).writerows([(p, y) for p, y, v in pos + hard if v] + neg("datasets/val.csv"))
print(word, len(pos), "positives and", len(hard), "sound-alike negatives from", len(srcs), "sources")
