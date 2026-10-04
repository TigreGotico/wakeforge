"""prep_clips.py <pool_dir> <librispeech_dir> <out_dir>

Writes <out_dir>/{val,calib,accept}/*.wav (16 kHz mono), disjoint:
  val     14 LibriSpeech + 3 AudioSet + 3 FMA     (validation against PyTorch)
  calib   40 LibriSpeech + 12 AudioSet + 12 FMA   (int8-static calibration)
  accept  18 LibriSpeech + 6 AudioSet + 6 FMA     (reduced-precision acceptance)
LibriSpeech: the last 2% of the sorted file list (train_pool.py's held-out split), cut to 10 s.
AudioSet (audioset_b) and FMA: the last 2% of each source by index (held out), only 10 s segments whose
every frame is forced non-speech.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR, EVAL_FRAC = 16000, 0.02
SPLITS = {"val": (14, 3, 3), "calib": (40, 12, 12), "accept": (18, 6, 6)}


def pick(items, counts):
    n = sum(counts)
    sel = [items[i] for i in np.linspace(0, len(items) - 1, n).astype(int)]
    assert len(set(map(str, sel))) == n, "not enough distinct items"
    out, k = [], 0
    for c in counts:
        out.append(sel[k:k + c])
        k += c
    return out


def segments(pool, name):
    sl = Path(pool) / f"{name}.slice.json"
    if sl.exists():
        j = json.loads(sl.read_text())
        raw, start, count, seg = j["raw"], j["start"], j["count"], j["seg"]
    else:
        raw, start, count, seg = str(Path(pool) / f"{name}_audio.raw"), 0, int((Path(pool) / f"{name}.done").read_text().strip()), 10 * SR
    forced = np.load(Path(pool) / f"{name}_forced.npy", mmap_mode="r")
    k = max(1, int(round(EVAL_FRAC * count)))
    idx = [i for i in range(count - k, count) if bool(np.asarray(forced[i]).all())]
    A = np.memmap(raw, np.int16, "r", offset=2 * seg * start, shape=(count, seg))
    return [(f"{name}_{i}", A, i) for i in idx]


def main():
    pool, libri, out = sys.argv[1:4]
    files = sorted(glob.glob(f"{libri}/**/*.flac", recursive=True))
    k = max(1, int(round(EVAL_FRAC * len(files))))
    lib = pick(files[-k:], [SPLITS[s][0] for s in SPLITS])
    aus = pick(segments(pool, "audioset_b"), [SPLITS[s][1] for s in SPLITS])
    fma = pick(segments(pool, "fma"), [SPLITS[s][2] for s in SPLITS])
    manifest = {}
    for j, s in enumerate(SPLITS):
        d = Path(out) / s
        d.mkdir(parents=True, exist_ok=True)
        manifest[s] = []
        for f in lib[j]:
            w, sr = sf.read(f, dtype="float32")
            assert sr == SR
            name = "libri_" + Path(f).stem
            sf.write(d / f"{name}.wav", w[:10 * SR], SR, subtype="FLOAT")
            manifest[s].append({"file": f"{name}.wav", "source": f, "seconds": round(min(len(w), 10 * SR) / SR, 2)})
        for name, A, i in aus[j] + fma[j]:
            w = np.asarray(A[i], np.float32) / 32767
            sf.write(d / f"{name}.wav", w, SR, subtype="FLOAT")
            manifest[s].append({"file": f"{name}.wav", "source": f"{name.rsplit('_', 1)[0]} segment {i}", "seconds": len(w) / SR})
    (Path(out) / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print({s: len(v) for s, v in manifest.items()})


if __name__ == "__main__":
    main()
