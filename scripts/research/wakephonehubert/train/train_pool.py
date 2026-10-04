"""train_pool.py <pool_dir> <out_dir> --sources fma,mswc_a,... [--side 0|128] [--epochs N] [--steps-per-epoch N] [--limit N]
                [--init-vad vad_head.pt] [--resume <run_dir>]
train_pool.py <pool_dir> <out_dir> --sources fma,mswc_a,... --resume <run_dir> --final-ipa-sources mswcx [--epochs N]

Sources are pool_teachers.py outputs of any name; each one's type (mswc, audioset, fma) comes from its progress file.
Within each batch kind, rows are drawn from that kind's sources in proportion to their training size.
Learning rate: linear warm-up to the peak, then cosine to 5% of it over --epochs. --resume loads the heads and their
optimiser states from <run_dir>/checkpoint.pt (written every epoch) and restarts that schedule over --epochs from the
resume point at half the original peaks, so a run can be extended any number of times with new sources.

Two independent heads on one frozen WakeHuBERT-tiny forward per batch (disjoint parameters, one optimiser each):
  vad    TapHead(1, 32)                 BCE to an active-speaker target
  ipa    TapHead(392, 192, side)        one layer over the wav2vec2 espeak teacher's symbol table: CTC (blank = pad id 0) to
                                        the teacher tokenizer's ids for the MSWC word, plus sparse KL to the teacher's top-8, delay 5
Batches cycle through three kinds:
  A  1 s MSWC clips (128), noise from fma/audioset at 5-30 dB SNR with p 0.5; targets from the clean clip's teachers
     (VAD: Silero on the clean clip; CTC where the clip has tokenizer targets)
  B  3 s crops (48) of audioset/fma segments starting on a 320 ms boundary, targets sliced to the crop; VAD is 0 with weight 1
     on forced non-speech frames and weight 0 elsewhere; CTC with an empty target when every frame of the crop is forced
  C  4 s synthetic mixtures (vad only): foreground (p 0.6, LibriSpeech or an MSWC word at a random offset) whose Silero
     probabilities, computed on the foreground alone, are the target; background babble (1-3 reverberant talkers, optionally
     band-limited 200-5000 Hz as a loudspeaker), music (FMA, vocals included) or noise (audioset without voice evidence)
Held out: the last --eval-frac of every audioset/fma source by index, every (1/--eval-frac)-th clip of every mswc
source (its clips are ordered by language, so a tail would hold out the last language only), and the last --eval-frac
of LibriSpeech's sorted file list and of the RIRs.
--init-vad warm-starts the VAD head (train_vad.py); the ipa head always starts from scratch.
--final-ipa-sources: CTC-only final IPA training on mswc-type sources that need only teacher-tokenizer ids
(pool_teachers.py --stages text). The VAD head is frozen and not run; batches are 85% clips from those
sources with batch-A noise mixing and 15% one-second crops of fully forced non-speech pool segments with empty targets;
no teacher KL; learning rate 0.3x the original IPA peak, warm-up then cosine to zero. PER on the --sources mswc
held-out split every epoch. Clips whose id is in that held-out split, or that have no targets, are not used.
Leakage guards: fma-type segments whose track is listed in --exclude-tracks are dropped from training and evaluation;
audioset-type video ids that also name files under --audioset-eval-dir are logged.
"""
import argparse
import glob
import json
import math
import multiprocessing as mp
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "pool"))
from pool_teachers import DROP_IDS, SEG, W2V, WINDOW_HOP, frame_targets, open_audio, silero_batch, silero_session, source_audio  # noqa: E402
from wph_model import TapHead, Trunk, load_trunk  # noqa: E402

SR, HOP = 16000, 320
CROP_F, MIX_N = 150, 4 * SR
BG_KINDS = ("babble", "tv_babble", "music", "noise")


class Src:
    def __init__(self, d, name, limit, eval_frac, exclude_tracks=frozenset()):
        d = Path(d)
        prog = json.loads((d / f"{name}_progress.json").read_text())
        n, total = prog["n"], math.ceil(prog["n"] / prog["chunk"])
        self.type = prog.get("type", name)
        need = {"silero": total, "ipa": total, "voice": total, "forced": 1} | ({"ids": 1} if self.type == "mswc" else {})
        missing = {k: (prog.get(k, 0), v) for k, v in need.items() if prog.get(k, 0) < v}
        if missing:
            raise SystemExit(f"{name}: teacher stages incomplete {missing}")
        self.n = min(n, limit) if limit else n
        self.name, self.seg = name, SEG[self.type]
        self.frames = self.seg // HOP
        raw, start, avail, _ = source_audio(d, name, self.seg)
        assert avail >= n, (name, avail, n)
        self.audio = open_audio(raw, start, n, self.seg)
        ld = lambda k: np.load(d / f"{name}_{k}.npy", mmap_mode="r")
        self.sil, self.tki, self.tkp = ld("silero"), ld("tk_idx"), ld("tk_p")
        self.forced = np.load(d / f"{name}_forced.npy")[:self.n]
        self.meta = []
        with open(d / f"{name}_meta.jsonl", encoding="utf-8") as f:
            for line in f:
                if len(self.meta) >= self.n:
                    break
                self.meta.append(json.loads(line))
        self.labels = None
        if self.type == "mswc":
            with open(d / f"{name}_ipa_ids.jsonl", encoding="utf-8") as f:
                self.labels = [json.loads(l)["ids"] for _, l in zip(range(self.n), f)]
            stride = max(2, int(round(1 / eval_frac)))
            ev = np.zeros(self.n, bool); ev[stride - 1::stride] = True
            self.train, self.eval = np.flatnonzero(~ev), np.flatnonzero(ev)
        else:
            k = max(1, int(round(eval_frac * self.n)))
            self.train, self.eval = np.arange(self.n - k), np.arange(self.n - k, self.n)
        self.excluded = 0
        if self.type == "fma" and exclude_tracks:
            bad = np.array([str(m.get("track")) in exclude_tracks for m in self.meta])
            self.excluded = int(bad.sum()); self.excluded_tracks = len({m.get("track") for m, b in zip(self.meta, bad) if b})
            self.train, self.eval = self.train[~bad[self.train]], self.eval[~bad[self.eval]]
        self.all_forced = self.forced.all(1)


class CtcSrc:
    """An mswc-type source with teacher-tokenizer ids only (no Silero, voice-evidence or wav2vec2 arrays)."""

    def __init__(self, d, name, limit, exclude_ids):
        d = Path(d)
        prog = json.loads((d / f"{name}_progress.json").read_text())
        self.type = prog.get("type", name)
        if self.type != "mswc" or not prog.get("ids"):
            raise SystemExit(f"{name}: needs type mswc and the ids stage (pool_teachers.py --stages text); progress {prog}")
        n = prog["n"]
        self.name, self.seg, self.n = name, SR, (min(n, limit) if limit else n)
        raw, start, avail, _ = source_audio(d, name, SR)
        assert avail >= n, (name, avail, n)
        self.audio = open_audio(raw, start, n, SR)
        with open(d / f"{name}_ipa_ids.jsonl", encoding="utf-8") as f:
            self.labels = [json.loads(l)["ids"] for _, l in zip(range(self.n), f)]
        with open(d / f"{name}_meta.jsonl", encoding="utf-8") as f:
            ids = [json.loads(l)["id"] for _, l in zip(range(self.n), f)]
        leak = np.array([i in exclude_ids for i in ids])
        has = np.array([bool(l) for l in self.labels])
        self.train = np.flatnonzero(has & ~leak)
        self.stats = {"n": self.n, "with_targets": int(has.sum()), "in_pool_heldout": int(leak.sum()), "used": len(self.train)}


def per(ref, hyp):
    """Edit distance of two id sequences and the reference length."""
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)], len(ref)


def load_wav(path, sr_want=SR):
    import soundfile as sf
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    w = w.mean(1)
    if sr != sr_want:
        from scipy.signal import resample_poly
        g = math.gcd(sr, sr_want); w = resample_poly(w, sr_want // g, sr // g).astype(np.float32)
    return w


class MixBank:
    """Sources for kind-C mixtures; split='train' or 'eval'."""

    def __init__(self, S, libri_dir, rir_dir, split, eval_frac):
        files = sorted(glob.glob(f"{libri_dir}/**/*.flac", recursive=True))
        rirs = sorted(glob.glob(f"{rir_dir}/**/*.wav", recursive=True))
        kl, kr = max(1, int(round(eval_frac * len(files)))), max(1, int(round(0.1 * len(rirs))))
        self.libri = files[:-kl] if split == "train" else files[-kl:]
        self.rir_paths = rirs[:-kr] if split == "train" else rirs[-kr:]
        self.rirs = None
        pick = lambda s: getattr(s, split)
        self.mswc = [(s.audio, pick(s)) for s in of(S, "mswc")]
        self.music = [(s.audio, pick(s)) for s in of(S, "fma")]
        self.noise = [(s.audio, pick(s)[s.all_forced[pick(s)]]) for s in of(S, "audioset")]
        self.noise = [x for x in self.noise if len(x[1])]

    def load_rirs(self):
        out = []
        for p in self.rir_paths:
            r = load_wav(p); r = r[int(np.argmax(np.abs(r))):][:SR]
            out.append((r / (np.sqrt((r ** 2).sum()) + 1e-9)).astype(np.float32))
        self.rirs = out

    def bg_kinds(self):
        return [k for k in BG_KINDS if {"babble": True, "tv_babble": True, "music": bool(self.music), "noise": bool(self.noise)}[k]]


def of(S, *types):
    return [s for s in S.values() if s.type in types]


def _pick(rng, lst):
    w = np.array([len(i) for _, i in lst], float)
    return lst[rng.choice(len(lst), p=w / w.sum())]


def _rms_db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-12)


def _at_db(x, db):
    return x * 10 ** ((db - _rms_db(x)) / 20)


def _libri(rng, B, n):
    import soundfile as sf
    p = B.libri[rng.integers(len(B.libri))]
    with sf.SoundFile(p) as f:
        s = rng.integers(0, max(1, f.frames - n + 1)); f.seek(int(s)); w = f.read(n, dtype="float32")
    return w if w.ndim == 1 else w.mean(1)


def _word(rng, B):
    A, idx = _pick(rng, B.mswc)
    return np.asarray(A[idx[rng.integers(len(idx))]], np.float32) / 32767


def _talker(rng, B):
    from scipy.signal import fftconvolve
    if not B.mswc or rng.random() < 0.7:
        s = _libri(rng, B, MIX_N)
    else:
        s = _word(rng, B)
    x = np.zeros(MIX_N, np.float32); off = rng.integers(0, MIX_N - len(s) + 1); x[off:off + len(s)] = s
    return fftconvolve(x, B.rirs[rng.integers(len(B.rirs))])[:MIX_N].astype(np.float32)


def _crop(rng, src):
    A, idx = _pick(rng, src)
    q = idx[rng.integers(len(idx))]; s = rng.integers(0, A.shape[1] - MIX_N + 1)
    return np.asarray(A[q, s:s + MIX_N], np.float32) / 32767


def mixture(rng, B, fg=None, bg=None):
    """One 4 s mixture: (mix, foreground-only signal or None, background kind or 'none')."""
    from scipy.signal import butter, sosfilt
    has_fg = rng.random() < 0.6 if fg is None else fg
    f = None
    if has_fg:
        s = _libri(rng, B, int(rng.uniform(1, 4) * SR)) if (not B.mswc or rng.random() < 0.5) else _word(rng, B)
        fg_db = rng.uniform(-35, -15)
        s = _at_db(s, fg_db) if np.abs(s).max() > 0 else s
        f = np.zeros(MIX_N, np.float32); off = rng.integers(0, MIX_N - len(s) + 1); f[off:off + len(s)] = s
    kinds = B.bg_kinds()
    if bg is None:
        bg = (["none"] + kinds)[rng.integers(len(kinds) + 1)] if has_fg else kinds[rng.integers(len(kinds))]
    x = np.zeros(MIX_N, np.float32)
    if bg in ("babble", "tv_babble"):
        for _ in range(rng.integers(1, 4)):
            t = _talker(rng, B)
            x += _at_db(t, (fg_db - rng.uniform(5, 20)) if has_fg else rng.uniform(-50, -30)) if np.abs(t).max() > 0 else t
        if bg == "tv_babble":
            x = sosfilt(butter(4, [200, 5000], "bandpass", fs=SR, output="sos"), x).astype(np.float32)
    elif bg in ("music", "noise"):
        t = _crop(rng, B.music if bg == "music" else B.noise)
        if np.abs(t).max() > 0:
            x = _at_db(t, (fg_db - rng.uniform(0, 15)) if has_fg else rng.uniform(-50, -20))
    mix = x + (f if f is not None else 0)
    pk = np.abs(mix).max()
    if pk > 0.99:
        mix, f = mix * 0.99 / pk, (f * 0.99 / pk if f is not None else None)
    return mix.astype(np.float32), f, bg


def mixture_batch(rng, B, sess, n, fgs=None, bgs=None, silero_on_mix=False):
    rows = [mixture(rng, B, None if fgs is None else fgs[i], None if bgs is None else bgs[i]) for i in range(n)]
    frames = MIX_N // HOP
    y = np.zeros((n, frames), np.float32)
    has = [i for i, r in enumerate(rows) if r[1] is not None]
    if has:
        p = silero_batch(sess, np.stack([rows[i][1] for i in has]))
        for i, pi in zip(has, p):
            y[i] = frame_targets(pi, frames)
    out = {"wav": np.stack([r[0] for r in rows]), "y": y, "bg": [r[2] for r in rows], "fg": [r[1] is not None for r in rows]}
    if silero_on_mix:
        out["silero_mix"] = np.stack([frame_targets(pi, frames) for pi in silero_batch(sess, out["wav"])])
    return out


def _mix_worker(seed, B, n, q):
    rng = np.random.default_rng(seed); sess = silero_session(); B.load_rirs()
    while True:
        b = mixture_batch(rng, B, sess, n)
        q.put({"wav": (np.clip(b["wav"], -1, 1) * 32767).astype(np.int16), "y": b["y"].astype(np.float16)})


def weighted_src(rng, srcs):
    w = np.array([len(s.train) for s in srcs], float)
    return srcs[rng.choice(len(srcs), p=w / w.sum())]


def batch_A(rng, S, nb):
    ms = of(S, "mswc"); rows = []
    for _ in range(nb):
        m = weighted_src(rng, ms); rows.append((m, m.train[rng.integers(len(m.train))]))
    wav = np.stack([np.asarray(m.audio[k], np.float32) / 32767 for m, k in rows])
    noise = of(S, "fma", "audioset")
    for r in range(nb):
        wav[r] = noisy(rng, wav[r], noise)
    lab = [m.labels[k] for m, k in rows]
    y = np.stack([np.asarray(m.sil[k], np.float32) for m, k in rows])
    return {"kind": "A", "wav": wav, "vad_y": y, "vad_w": np.ones_like(y), "tki": np.stack([np.asarray(m.tki[k]) for m, k in rows]),
            "tkp": np.stack([np.asarray(m.tkp[k], np.float32) for m, k in rows]),
            "ctc": lab, "ctc_use": np.array([len(l) > 0 for l in lab])}


def batch_B(rng, S, nb):
    srcs = of(S, "audioset", "fma")
    src0 = srcs[0]; T = src0.tki.shape[1]
    wav = np.zeros((nb, CROP_F * HOP), np.float32); vw = np.zeros((nb, CROP_F), np.float32)
    tki = np.zeros((nb, CROP_F, src0.tki.shape[2]), np.int16); tkp = np.zeros(tki.shape, np.float32)
    use = np.zeros(nb, bool)
    for r in range(nb):
        s = weighted_src(rng, srcs); q = s.train[rng.integers(len(s.train))]
        f0 = rng.integers(0, (s.frames - CROP_F) // WINDOW_HOP + 1) * WINDOW_HOP
        w = np.asarray(s.audio[q, f0 * HOP:(f0 + CROP_F) * HOP], np.float32) / 32767
        wav[r] = np.clip(w * 10 ** rng.uniform(-0.5, 0.2), -1, 1)
        fr = s.forced[q, f0:f0 + CROP_F]; vw[r] = fr; use[r] = fr.all()
        t = min(CROP_F, T - f0)
        tki[r, :t] = s.tki[q, f0:f0 + t]; tkp[r, :t] = s.tkp[q, f0:f0 + t]
    return {"kind": "B", "wav": wav, "vad_y": np.zeros_like(vw), "vad_w": vw, "tki": tki, "tkp": tkp,
            "ctc": [[] for _ in range(nb)], "ctc_use": use}


class Heads(nn.Module):
    def __init__(self, Vt, side):
        super().__init__()
        self.vad = TapHead(1, hidden=32)
        self.ipa = TapHead(Vt, hidden=192, side=side)


def greedy(row):
    hyp, prev = [], 0
    for t in row:
        if t != prev and t not in DROP_IDS:
            hyp.append(int(t))
        prev = t
    return hyp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool_dir"); ap.add_argument("out_dir")
    ap.add_argument("--sources", required=True, help="comma-separated pool source names, e.g. fma,mswc_a")
    ap.add_argument("--resume", default="", help="run dir whose checkpoint.pt to continue from")
    ap.add_argument("--final-ipa-sources", default="", help="comma-separated CTC-only mswc-type sources for the final IPA training")
    ap.add_argument("--exclude-tracks", default="", help="text file of FMA track ids kept out of fma-type sources (evaluation tracks)")
    ap.add_argument("--audioset-eval-dir", default="", help="directory of evaluation files named by AudioSet video id, to log overlap with the audioset sources")
    ap.add_argument("--lr-peak", type=float, default=2e-3); ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--side", type=int, default=0, choices=[0, 128])
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--steps-per-epoch", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--batch-a", type=int, default=128); ap.add_argument("--batch-b", type=int, default=48); ap.add_argument("--batch-c", type=int, default=64)
    ap.add_argument("--mix-workers", type=int, default=8); ap.add_argument("--mix-eval", type=int, default=400)
    ap.add_argument("--delay", type=int, default=5)
    ap.add_argument("--libri", default="work/data/LibriSpeech/train-clean-100")
    ap.add_argument("--rir", default="work/data/rir", help="directory of room impulse response wavs")
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--init-vad", default="", help="vad_head.pt state_dict (TapHead(1, 32), no side branch)")
    a = ap.parse_args()
    torch.set_num_threads(a.torch_threads)
    d, out = Path(a.pool_dir).resolve(), Path(a.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    excl = frozenset(Path(a.exclude_tracks).read_text().split()) if a.exclude_tracks else frozenset()
    S = {k: Src(d, k, a.limit, a.eval_frac, excl) for k in a.sources.split(",")}
    for s in of(S, "fma"):
        print(json.dumps({"exclude_tracks": a.exclude_tracks, "source": s.name, "listed_tracks": len(excl), "excluded_segments": s.excluded,
                          "excluded_tracks": getattr(s, "excluded_tracks", 0)}), flush=True)
    for s in of(S, "audioset"):
        files = [q for q in Path(a.audioset_eval_dir).rglob("*") if q.is_file()] if Path(a.audioset_eval_dir).is_dir() else []
        stems = {q.stem for q in files} | {q.stem[:11] for q in files}
        hit = sorted({m["id"] for m in s.meta if m.get("id") in stems})
        print(json.dumps({"audioset_eval_overlap": {"source": s.name, "eval_dir": a.audioset_eval_dir, "eval_files": len(files),
                          "video_ids_matching_eval_filenames": len(hit), "examples": hit[:10]}}), flush=True)
    if not of(S, "mswc") or not of(S, "audioset", "fma"):
        raise SystemExit(f"--sources needs at least one mswc-type and one audioset/fma-type source: {[(k, s.type) for k, s in S.items()]}")
    print(json.dumps({k: {"type": s.type, "n": s.n, "train": len(s.train), "eval": len(s.eval), "all_forced": int(s.all_forced.sum()),
                          "forced_frames": round(float(s.forced.mean()), 4)} for k, s in S.items()}), flush=True)

    if a.final_ipa_sources:
        if not a.resume:
            raise SystemExit("--final-ipa-sources needs --resume <run_dir>")
        heldout = {s.meta[k]["id"] for s in of(S, "mswc") for k in s.eval}
        X = {k: CtcSrc(d, k, a.limit, heldout) for k in a.final_ipa_sources.split(",")}
        print(json.dumps({"final_ipa_sources": {k: x.stats for k, x in X.items()}}), flush=True)
        final_ipa(a, S, X, out)
        return
    Btr = MixBank(S, a.libri, a.rir, "train", a.eval_frac); Bev = MixBank(S, a.libri, a.rir, "eval", a.eval_frac)
    ctx = mp.get_context("fork"); mq = ctx.Queue(maxsize=12)
    workers = [ctx.Process(target=_mix_worker, args=(1000 + i, Btr, a.batch_c, mq), daemon=True) for i in range(a.mix_workers)]
    for p in workers:
        p.start()
    try:
        run(a, S, Bev, mq, out)
    finally:
        for p in workers:
            p.terminate()
        for p in workers:
            p.join(5)


def noisy(rng, w, noise):
    """Batch-A augmentation of one 1 s clip: pool noise at 5-30 dB SNR with p 0.5, then random gain to a 0.5 peak."""
    if noise and rng.random() < 0.5:
        ns = weighted_src(rng, noise); q = ns.train[rng.integers(len(ns.train))]; s = rng.integers(0, ns.seg - SR + 1)
        v = np.asarray(ns.audio[q, s:s + SR], np.float32) / 32767
        snr = rng.uniform(5, 30); ps, pv = np.mean(w ** 2) + 1e-9, max(np.mean(v ** 2), 1e-10)
        w = w + v * np.sqrt(ps / (pv * 10 ** (snr / 10)))
    return w * 10 ** rng.uniform(-0.7, 0.3) / max(np.abs(w).max(), 1e-6) * 0.5


def batch_final(rng, X, S, nb, ns_frac):
    xs = list(X.values()); noise = of(S, "fma", "audioset")
    ns_srcs = [(s, s.train[s.all_forced[s.train]]) for s in noise]
    ns_srcs = [(s, i) for s, i in ns_srcs if len(i)]
    n_ns = int(round(nb * ns_frac)) if ns_srcs else 0
    wav = np.zeros((nb, SR), np.float32); tg = []
    for r in range(nb - n_ns):
        x = weighted_src(rng, xs); k = x.train[rng.integers(len(x.train))]
        wav[r] = noisy(rng, np.asarray(x.audio[k], np.float32) / 32767, noise); tg.append(x.labels[k])
    w = np.array([len(i) for _, i in ns_srcs], float)
    for r in range(nb - n_ns, nb):
        s, idx = ns_srcs[rng.choice(len(ns_srcs), p=w / w.sum())]
        q = idx[rng.integers(len(idx))]; st = rng.integers(0, s.seg - SR + 1)
        wav[r] = np.clip(np.asarray(s.audio[q, st:st + SR], np.float32) / 32767 * 10 ** rng.uniform(-1, 0.3), -1, 1); tg.append([])
    return wav, tg


def final_ipa(a, S, X, out):
    ck = torch.load(Path(a.resume) / "checkpoint.pt", map_location="cuda", weights_only=False)
    if ck["args"]["side"] != a.side:
        raise SystemExit(f"--side {a.side} differs from the resumed run's {ck['args']['side']}")
    from transformers import Wav2Vec2PhonemeCTCTokenizer
    tok = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(W2V)
    Vt = len(tok); symbols = tok.convert_ids_to_tokens(list(range(Vt)))
    trunk = Trunk(load_trunk()).cuda().eval()
    torch.manual_seed(0)
    H = Heads(Vt, a.side).cuda(); H.load_state_dict(ck["heads"]); H.eval()
    for prm in H.vad.parameters():
        prm.requires_grad_(False)
    H.ipa.train()
    params = list(H.ipa.parameters())
    opt = torch.optim.AdamW(params, 1e-3, weight_decay=1e-2); opt.load_state_dict(ck["opts"]["ipa"])
    peak = 0.3 * ck["orig_peaks"]["ipa"]
    for g in opt.param_groups:
        g["lr"] = g["initial_lr"] = peak
    total = a.epochs * a.steps_per_epoch; warm = max(1, min(200, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: (st + 1) / warm if st < warm else
                                              0.5 * (1 + math.cos(math.pi * min(1.0, (st - warm) / max(1, total - warm)))))
    ctc = nn.CTCLoss(zero_infinity=True, reduction="none")
    segment = {"final_ipa": a.final_ipa_sources, "start_epoch": ck["epochs_done"], "epochs": a.epochs, "peak": peak, "warmup_steps": warm,
               "nonspeech_frac": 0.15, "resume": a.resume, "pool_sources": a.sources}
    print(json.dumps({"schedule": segment}), flush=True)

    def eval_ipa():
        errs = tot = phones = secs = 0; by = {}
        with torch.no_grad():
            for s in S.values():
                idx = s.eval if s.type == "mswc" else s.eval[s.all_forced[s.eval]]
                bs = 256 if s.type == "mswc" else 16
                for i in range(0, len(idx), bs):
                    j = idx[i:i + bs]
                    best = H.ipa(*trunk(torch.from_numpy(np.asarray(s.audio[j], np.float32) / 32767).cuda())).argmax(-1).cpu().numpy()
                    for k, row in zip(j, best):
                        if s.type == "mswc":
                            if s.labels[k]:
                                e_, n_ = per(s.labels[k], greedy(row)); errs += e_; tot += n_
                                b = by.setdefault(s.name, [0, 0]); b[0] += e_; b[1] += n_
                        else:
                            phones += len(greedy(row)); secs += len(row) / 50
        return {"mswc_per": round(errs / max(tot, 1), 4), "mswc_per_by_source": {k: round(e_ / max(n_, 1), 4) for k, (e_, n_) in by.items()},
                "nonspeech_phones_per_s": round(phones / secs, 3) if secs else None}

    pq = queue.Queue(maxsize=8); stop = threading.Event()

    def producer():
        rng = np.random.default_rng(7)
        while not stop.is_set():
            b = batch_final(rng, X, S, a.batch_a, 0.15)
            while not stop.is_set():
                try:
                    pq.put(b, timeout=1); break
                except queue.Full:
                    pass

    threading.Thread(target=producer, daemon=True).start()
    H.ipa.eval(); r0 = eval_ipa(); H.ipa.train()
    print(json.dumps({"epoch": "before", **r0}), flush=True)
    hist = list(ck["history"]); t0 = time.time()
    for ep in range(a.epochs):
        te, tot_l, wait = time.time(), 0.0, 0.0
        for it in range(a.steps_per_epoch):
            tw = time.time(); wav, tg = pq.get(); wait += time.time() - tw
            with torch.no_grad():
                taps, o, mm = trunk(torch.from_numpy(np.clip(wav, -1, 1)).cuda())
            lp = F.log_softmax(H.ipa(taps, o, mm), -1).transpose(0, 1)
            tl = torch.tensor([len(t) for t in tg]); flat = torch.tensor([x for t in tg for x in t], dtype=torch.long)
            loss = (ctc(lp, flat.cuda(), torch.full((len(tg),), lp.shape[0], dtype=torch.long), tl) / tl.clamp_min(1).cuda()).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params, 5.0); opt.step(); sched.step()
            tot_l += float(loss.detach())
        dt = time.time() - te
        H.ipa.eval(); rep = {"epoch": ck["epochs_done"] + ep + 1, "final_ipa": a.final_ipa_sources, "lr": round(opt.param_groups[0]["lr"], 8),
                             "ctc_loss": round(tot_l / a.steps_per_epoch, 4), "steps_per_s": round(a.steps_per_epoch / dt, 2),
                             "clips_per_s": round(a.steps_per_epoch * a.batch_a / dt, 1), "data_wait_frac": round(wait / dt, 3), **eval_ipa(),
                             "minutes": round((time.time() - t0) / 60, 2)}
        H.ipa.train(); hist.append(rep); print(json.dumps(rep), flush=True)
        torch.save({"head": H.ipa.state_dict(), "symbols": symbols, "blank": 0, "dropped_ids": list(DROP_IDS), "delay": ck["args"]["delay"]}, out / "ipa_head.pt")
        torch.save(H.vad.state_dict(), out / "vad_head.pt")
        opts = dict(ck["opts"]); opts["ipa"] = opt.state_dict()
        tmp = out / "checkpoint.pt.tmp"
        torch.save({**ck, "heads": H.state_dict(), "opts": opts, "epochs_done": ck["epochs_done"] + ep + 1, "history": hist,
                    "segments": ck["segments"] + [segment]}, tmp)
        tmp.replace(out / "checkpoint.pt")
    stop.set()
    report = {"side": a.side, "ipa_symbols": Vt, "init": ck.get("init"), "epochs_total": ck["epochs_done"] + a.epochs,
              "schedule_segments": ck["segments"] + [segment], "final_ipa_sources": {k: x.stats for k, x in X.items()},
              "before": r0, "final": hist[-1], "history": hist,
              "mix_weights": {"vad": torch.softmax(H.vad.mix, 0).tolist(), "ipa": torch.softmax(H.ipa.mix, 0).tolist()}}
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("FINAL_IPA_DONE", flush=True)


def run(a, S, Bev, mq, out):
    t_mix = time.time()
    Bev.load_rirs(); sess = silero_session(); erng = np.random.default_rng(12345)
    kinds = Bev.bg_kinds(); n_fg = int(a.mix_eval * 0.4); n_bg = (a.mix_eval - n_fg) // len(kinds)
    plan_fg = [True] * n_fg + [False] * (n_bg * len(kinds)); plan_bg = [None] * n_fg + [k for k in kinds for _ in range(n_bg)]
    MV = {"wav": [], "y": [], "bg": [], "fg": [], "silero_mix": []}
    for i in range(0, len(plan_fg), 64):
        b = mixture_batch(erng, Bev, sess, len(plan_fg[i:i + 64]), plan_fg[i:i + 64], plan_bg[i:i + 64], silero_on_mix=True)
        for k in MV:
            MV[k] += list(b[k])
    MV = {k: np.array(v) for k, v in MV.items()}
    print(json.dumps({"mix_eval": len(MV["y"]), "with_foreground": int(MV["fg"].sum()), "background_only_per_kind": n_bg,
                      "seconds": round(time.time() - t_mix, 1)}), flush=True)

    from transformers import Wav2Vec2PhonemeCTCTokenizer
    tok = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(W2V)
    Vt = len(tok); symbols = tok.convert_ids_to_tokens(list(range(Vt)))
    print(json.dumps({"ipa_symbols": Vt, "mswc_clips_with_targets": {s.name: sum(bool(l) for l in s.labels) for s in of(S, "mswc")},
                      "mswc_clips": {s.name: s.n for s in of(S, "mswc")}}), flush=True)
    ck_resume = None
    if a.resume:
        ck_resume = torch.load(Path(a.resume) / "checkpoint.pt", map_location="cuda", weights_only=False)
        for k in ("side", "delay"):
            if ck_resume["args"][k] != getattr(a, k):
                raise SystemExit(f"--{k} {getattr(a, k)} differs from the resumed run's {ck_resume['args'][k]}")
        if a.init_vad:
            raise SystemExit("--init-vad does not combine with --resume")
    D = a.delay
    trunk = Trunk(load_trunk()).cuda().eval()
    torch.manual_seed(0)
    H = Heads(Vt, a.side).cuda()
    init = {"vad": a.init_vad or None, "ipa": None, "resume": a.resume or None}
    if ck_resume is not None:
        init.update(ck_resume.get("init", {}), resume=a.resume)

    def load_into(mod, sd, what):
        cur = mod.state_dict()
        bad = sorted(set(cur) ^ set(sd)) + [k for k in cur if k in sd and cur[k].shape != sd[k].shape]
        if bad:
            raise SystemExit(f"{what}: checkpoint does not match this head (side={a.side}); mismatched keys or shapes: {bad[:6]}")
        mod.load_state_dict(sd)

    if a.init_vad:
        load_into(H.vad, torch.load(a.init_vad, map_location="cuda"), f"--init-vad {a.init_vad}")
    groups = {"vad": list(H.vad.parameters()), "ipa": list(H.ipa.parameters())}
    total = a.epochs * a.steps_per_epoch
    warm = max(1, min(a.warmup_steps, total // 10))
    if ck_resume is None:
        orig_peaks = {"vad": a.lr_peak, "ipa": a.lr_peak}
        peaks = dict(orig_peaks)
    else:
        orig_peaks = ck_resume["orig_peaks"]; peaks = {k: 0.5 * v for k, v in orig_peaks.items()}
    opts = {k: torch.optim.AdamW(p, peaks[k], weight_decay=1e-2) for k, p in groups.items()}
    prev = {"epochs_done": 0, "history": [], "segments": []}
    if ck_resume is not None:
        H.load_state_dict(ck_resume["heads"])
        for k, o in opts.items():
            o.load_state_dict(ck_resume["opts"][k])
            for g in o.param_groups:
                g["lr"] = g["initial_lr"] = peaks[k]
        prev = {"epochs_done": ck_resume["epochs_done"], "history": ck_resume["history"], "segments": ck_resume["segments"]}

    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

    scheds = {k: torch.optim.lr_scheduler.LambdaLR(o, lr_factor) for k, o in opts.items()}
    segment = {"start_epoch": prev["epochs_done"], "epochs": a.epochs, "sources": a.sources, "peaks": peaks, "warmup_steps": warm, "resume": a.resume or None}
    print(json.dumps({"schedule": segment}), flush=True)
    ctc = nn.CTCLoss(zero_infinity=True, reduction="none")

    pq = queue.Queue(maxsize=6); stop = threading.Event()

    def producer():
        rng = np.random.default_rng(0); k = 0
        while not stop.is_set():
            b = batch_A(rng, S, a.batch_a) if k % 2 == 0 else batch_B(rng, S, a.batch_b)
            k += 1
            while not stop.is_set():
                try:
                    pq.put(b, timeout=1); break
                except queue.Full:
                    pass

    th = threading.Thread(target=producer, daemon=True); th.start()
    cuda = lambda x, dt=torch.float32: torch.from_numpy(np.asarray(x)).to("cuda", dt)

    def feats(wav):
        with torch.no_grad():
            return trunk(wav)

    def vad_loss(z, y, w):
        L = min(z.shape[1], y.shape[1])
        b = F.binary_cross_entropy_with_logits(z[:, :L], y[:, :L], reduction="none") * w[:, :L]
        return b.sum() / w[:, :L].sum().clamp_min(1.0)

    def teacher_kl(z, ti, tp):
        lq = F.log_softmax(z, -1); T = min(lq.shape[1] - D, ti.shape[1])
        return -(tp[:, :T] * torch.gather(lq[:, D:D + T], 2, ti[:, :T].long())).sum(-1).mean()

    def step(b):
        if b["kind"] == "C":
            wav = cuda(b["wav"]) / 32767
            taps, o, mm = feats(wav)
            y = cuda(b["y"]); lv = vad_loss(H.vad(taps, o, mm)[..., 0], y, torch.ones_like(y))
            return {"vad": lv}, lv
        taps, o, mm = feats(cuda(b["wav"]))
        lv = vad_loss(H.vad(taps, o, mm)[..., 0], cuda(b["vad_y"]), cuda(b["vad_w"]))
        z = H.ipa(taps, o, mm)
        lk = teacher_kl(z, cuda(b["tki"], torch.long), cuda(b["tkp"]))
        use = np.flatnonzero(b["ctc_use"]); lc = torch.zeros((), device="cuda")
        if len(use):
            lp = F.log_softmax(z[use], -1).transpose(0, 1)
            tg = [b["ctc"][i] for i in use]; tl = torch.tensor([len(t) for t in tg])
            flat = torch.tensor([x for t in tg for x in t], dtype=torch.long)
            lc = (ctc(lp, flat.cuda(), torch.full((len(use),), lp.shape[0], dtype=torch.long), tl) / tl.clamp_min(1).cuda()).mean()
        return {"vad": lv, "ctc": lc, "kl": lk}, lv + lc + lk

    def evaluate():
        r = {}
        H.eval()
        with torch.no_grad():
            pv = []
            for i in range(0, len(MV["y"]), 64):
                taps, o, mm = feats(cuda(MV["wav"][i:i + 64]))
                pv.append(torch.sigmoid(H.vad(taps, o, mm)[..., 0]).cpu().numpy())
            pv = np.concatenate(pv); L = min(pv.shape[1], MV["y"].shape[1]); pv, y, sm = pv[:, :L], MV["y"][:, :L], MV["silero_mix"][:, :L]
            r["mix_vad_agree@0.5"] = round(float(((pv >= .5) == (y >= .5)).mean()), 4)
            r["mix_silero_on_mix_agree@0.5"] = round(float(((sm >= .5) == (y >= .5)).mean()), 4)
            fg = MV["fg"]
            r["mix_vad_agree@0.5_with_foreground"] = round(float(((pv[fg] >= .5) == (y[fg] >= .5)).mean()), 4)
            r["background_only"] = {k: {"vad_mean": round(float(pv[(~fg) & (MV["bg"] == k)].mean()), 4),
                                        "silero_mean": round(float(sm[(~fg) & (MV["bg"] == k)].mean()), 4),
                                        "vad_frac@0.5": round(float((pv[(~fg) & (MV["bg"] == k)] >= .5).mean()), 4),
                                        "silero_frac@0.5": round(float((sm[(~fg) & (MV["bg"] == k)] >= .5).mean()), 4)}
                                    for k in BG_KINDS if ((~fg) & (MV["bg"] == k)).any()}
            errs = tot = agree = nagree = agree_nb = n_nb = 0; forced_p = []; mswc_agree = []
            phones = secs = nb_frames = 0
            per_src = {}
            for name, s in S.items():
                bs = 256 if s.type == "mswc" else 16
                for i in range(0, len(s.eval), bs):
                    j = s.eval[i:i + bs]
                    taps, o, mm = feats(cuda(np.asarray(s.audio[j], np.float32) / 32767))
                    v = torch.sigmoid(H.vad(taps, o, mm)[..., 0]).cpu().numpy()
                    L = min(v.shape[1], s.frames)
                    if s.type == "mswc":
                        sil = np.asarray(s.sil[j], np.float32)[:, :L]; mswc_agree.append(((v[:, :L] >= .5) == (sil >= .5)).ravel())
                    fr = s.forced[j][:, :L]
                    forced_p.append(v[:, :L][fr])
                    z = H.ipa(taps, o, mm)
                    best = z.argmax(-1).cpu().numpy()
                    if s.type == "mswc":
                        for k, row in zip(j, best):
                            if s.labels[k]:
                                e_, n_ = per(s.labels[k], greedy(row)); errs += e_; tot += n_
                                pe = per_src.setdefault(name, [0, 0]); pe[0] += e_; pe[1] += n_
                    for k, row in zip(j, best):
                        if s.all_forced[k]:
                            phones += len(greedy(row)); nb_frames += int((row != 0).sum()); secs += len(row) / 50
                    ti = cuda(np.asarray(s.tki[j]), torch.long); T = min(z.shape[1] - D, ti.shape[1])
                    hit = z[:, D:D + T].argmax(-1) == ti[:, :T, 0]; nb = ti[:, :T, 0] != 0
                    agree += int(hit.sum()); nagree += hit.numel(); agree_nb += int((hit & nb).sum()); n_nb += int(nb.sum())
            fp = np.concatenate(forced_p) if forced_p else np.zeros(0)
            r["pool_vad_mean_on_forced"] = round(float(fp.mean()), 4) if len(fp) else None
            r["pool_vad_agree@0.5_on_forced"] = round(float((fp < .5).mean()), 4) if len(fp) else None
            r["mswc_vad_agree@0.5_with_silero"] = round(float(np.concatenate(mswc_agree).mean()), 4)
            r["mswc_per"] = round(errs / max(tot, 1), 4)
            r["mswc_per_by_source"] = {k: round(e_ / max(n_, 1), 4) for k, (e_, n_) in per_src.items()}
            r["nonspeech_phones_per_s"] = round(phones / secs, 3) if secs else None
            r["nonspeech_nonblank_frames_per_s"] = round(nb_frames / secs, 3) if secs else None
            r["teacher_top1_agree"] = round(agree / max(nagree, 1), 4)
            r["teacher_top1_agree_nonblank"] = round(agree_nb / max(n_nb, 1), 4)
        H.train()
        return r

    hist = list(prev["history"]); t0 = time.time(); gstep = 0
    for ep in range(a.epochs):
        sums, counts, wait = {}, {}, 0.0; te = time.time()
        for it in range(a.steps_per_epoch):
            tw = time.time()
            if it % 3 == 2:
                c = mq.get(); b = {"kind": "C", "wav": c["wav"], "y": c["y"].astype(np.float32)}
            else:
                b = pq.get()
            wait += time.time() - tw
            parts, loss = step(b)
            for o_ in opts.values():
                o_.zero_grad(set_to_none=True)
            loss.backward()
            for k, p in groups.items():
                nn.utils.clip_grad_norm_(p, 5.0)
            for k in opts:
                if b["kind"] != "C" or k == "vad":
                    opts[k].step()
                scheds[k].step()
            gstep += 1
            for k, v in parts.items():
                key = f"{b['kind']}_{k}"; sums[key] = sums.get(key, 0.0) + float(v.detach()); counts[key] = counts.get(key, 0) + 1
        dt = time.time() - te
        rep = {"epoch": prev["epochs_done"] + ep + 1, "sources": a.sources, "lr": {k: round(o.param_groups[0]["lr"], 7) for k, o in opts.items()}, "train_loss": {k: round(sums[k] / counts[k], 4) for k in sorted(sums)},
               "steps_per_s": round(a.steps_per_epoch / dt, 2), "data_wait_frac": round(wait / dt, 3)}
        rep.update(evaluate()); rep["minutes"] = round((time.time() - t0) / 60, 2)
        hist.append(rep); print(json.dumps(rep), flush=True)
        torch.save(H.vad.state_dict(), out / "vad_head.pt")
        torch.save({"head": H.ipa.state_dict(), "symbols": symbols, "blank": 0, "dropped_ids": list(DROP_IDS), "delay": D}, out / "ipa_head.pt")
        tmp = out / "checkpoint.pt.tmp"
        torch.save({"heads": H.state_dict(), "opts": {k: o.state_dict() for k, o in opts.items()}, "scheds": {k: s_.state_dict() for k, s_ in scheds.items()},
                    "orig_peaks": orig_peaks, "init": init, "args": vars(a), "epochs_done": prev["epochs_done"] + ep + 1,
                    "history": hist, "segments": prev["segments"] + [segment]}, tmp)
        tmp.replace(out / "checkpoint.pt")
    stop.set()
    report = {"side": a.side, "ipa_symbols": Vt, "init": init, "delay": D, "epochs_total": prev["epochs_done"] + a.epochs,
              "schedule_segments": prev["segments"] + [segment], "steps_per_epoch": a.steps_per_epoch,
              "sources": {k: {"type": s.type, "n": s.n, "train": len(s.train), "eval": len(s.eval)} for k, s in S.items()},
              "params": {k: sum(p.numel() for p in g) for k, g in groups.items()},
              "mix_weights": {"vad": torch.softmax(H.vad.mix, 0).tolist(), "ipa": torch.softmax(H.ipa.mix, 0).tolist()},
              "final": hist[-1], "history": hist}
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("TRAIN_POOL_DONE", flush=True)


if __name__ == "__main__":
    main()
