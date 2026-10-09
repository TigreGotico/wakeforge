"""Wake-word heads on streamed WakeHuBERT-tiny features: data, training, calibration and scoring.

The streaming trunk (``stream_trunk.py``) carries its state for as long as the stream runs, so the frames at any
moment depend on everything heard before. The shipped heads were trained on isolated 1.5 s windows that start from
silence, and on streamed frames they lose recall and gain false activations. This module trains the same head, a GRU
over the last 75 frames scored every 80 ms block, on windows of streamed features of continuous synthetic streams,
and scores it on the same footing as the isolated models.

Subcommands (each audio-heavy one takes ``--part i --parts n`` so several cores share it):

    streams  training streams for a word (or negative-only streams with --word none), featurized: shards of
             features [n, T, 128] float16 and frame labels (1 wake word, 0 not, -1 unscored), as .npy files
    calib    the calibration stream (continuous negatives) and calibration positives, featurized
    negs     evaluation negatives as continuous streams: streamed features, and the plugin's isolated-window logits
             of published heads
    pos      evaluation positives, each after a lead-in: streamed features, and isolated-window logits
    fit      the window head on the shards of one word and the shared negative shards
    export   calibrate a trained head on the calibration stream and write the plugin's head ONNX
    score    recall at a false-activation rate and at the calibrated default, for shipped heads on isolated windows
             and on streamed frames, and for exported heads on streamed frames
"""
import argparse
import glob
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn

import importlib.util

_ST = Path(__file__).resolve().parent / "stream_trunk.py"
_spec = importlib.util.spec_from_file_location("stream_trunk", _ST)
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)

SR, HOP, BLOCK = 16000, 320, 1280
FPB = BLOCK // HOP
WINDOW = 24000
WINDOW_FRAMES = 75
DEBOUNCE_BLOCKS = 25
POS_FROM, POS_TO, IGNORE_TO = 2, 14, 30


# ------------------------------------------------------------------ audio
def files(*dirs, exts=(".wav", ".flac")):
    out = []
    for d in dirs:
        out += [p for p in sorted(Path(d).rglob("*")) if p.suffix in exts]
    return out


def read(p):
    w, sr = sf.read(str(p), dtype="float32", always_2d=True)
    w = w.mean(1)
    if sr != SR:
        import torchaudio
        w = torchaudio.functional.resample(torch.from_numpy(w), sr, SR).numpy()
    return w


def trim(w, rel=0.02):
    a = np.abs(w)
    nz = np.flatnonzero(a > rel * a.max()) if a.max() > 0 else []
    return w[nz[0]:nz[-1] + 1] if len(nz) else w


def split_by_file(paths, held_every=10):
    """(train, held) with every ``held_every``-th file, in sorted order, held out."""
    return [p for i, p in enumerate(paths) if i % held_every], [p for i, p in enumerate(paths) if not i % held_every]


def even_odd(paths):
    return paths[0::2], paths[1::2]


class Pools:
    """Background sources for training streams: speech, short negative windows, noise and room responses."""

    def __init__(self, speech, windows, noise, rirs, rng):
        self.speech, self.windows, self.noise, self.rng = speech, windows, noise, rng
        self.rirs = [r / (np.abs(r).max() + 1e-9) for r in (read(p)[:8000] for p in rirs)]

    def speech_run(self, n):
        out, got = [], 0
        while got < n:
            w = read(self.rng.choice(self.speech))
            out.append(w)
            got += len(w)
        return np.concatenate(out)[:n]

    def windows_run(self, n):
        out, got = [], 0
        while got < n:
            w = read(self.rng.choice(self.windows))
            out.append(w)
            got += len(w)
        return np.concatenate(out)[:n]

    def noise_run(self, n):
        w = read(self.rng.choice(self.noise))
        w = np.tile(w, n // max(len(w), 1) + 1)
        s = self.rng.randint(0, len(w) - n)
        return w[s:s + n]

    def babble(self, n):
        return sum(self.speech_run(n) for _ in range(self.rng.randint(2, 3)))

    def background(self, n):
        """``n`` samples of background made of segments of 3 to 12 s, each speech, short windows, noise, babble or
        near silence, at levels 0 to 30 dB apart."""
        out, got = [], 0
        while got < n:
            k = min(n - got, self.rng.randint(3 * SR, 12 * SR))
            kind = self.rng.choices(["speech", "windows", "noise", "babble", "quiet"], [3, 3, 2, 1, 1])[0]
            if kind == "quiet":
                seg = np.random.default_rng(self.rng.randint(0, 2 ** 31)).standard_normal(k).astype(np.float32) * 1e-4
            else:
                seg = getattr(self, {"speech": "speech_run", "windows": "windows_run", "noise": "noise_run",
                                     "babble": "babble"}[kind])(k)
                seg = seg / (np.sqrt(np.mean(seg ** 2)) + 1e-9) * 0.05 * 10 ** (-self.rng.uniform(0, 30) / 20)
            out.append(seg.astype(np.float32))
            got += k
        return np.concatenate(out)

    def word(self, w):
        """A trimmed wake-word clip with speed and room reverberation."""
        w = trim(w)
        if self.rng.random() < 0.5:
            f = self.rng.uniform(0.9, 1.1)
            w = np.interp(np.arange(0, len(w), f), np.arange(len(w)), w).astype(np.float32)
        if self.rirs and self.rng.random() < 0.5:
            h = self.rng.choice(self.rirs)
            L = len(w) + len(h) - 1
            r = np.fft.irfft(np.fft.rfft(w, L) * np.fft.rfft(h, L), L)[:len(w) + 1600]
            w = (r * np.sqrt(np.mean(w ** 2) / (np.mean(r ** 2) + 1e-12))).astype(np.float32)
        return w

    def device(self, y):
        """What a device adds to the whole stream: band limits, colour, level, occasional compression or clipping,
        and a noise floor."""
        rng = self.rng
        Fy = np.fft.rfft(y)
        f = np.fft.rfftfreq(len(y), 1 / SR)
        if rng.random() < 0.6:
            lo, hi = rng.uniform(60, 300), rng.uniform(3000, 7800)
            Fy = Fy / (1 + (lo / np.maximum(f, 1)) ** 4) / (1 + (f / hi) ** 8)
        if rng.random() < 0.5:
            Fy = Fy * np.interp(f, np.linspace(0, 8000, 8), 10 ** (np.array([rng.uniform(-6, 6) for _ in range(8)]) / 20))
        y = np.fft.irfft(Fy, len(y)).astype(np.float32)
        y = y / (np.abs(y).max() + 1e-9) * 10 ** (rng.uniform(-30, 0) / 20)
        if rng.random() < 0.2:
            y = np.sign(y) * np.abs(y) ** rng.uniform(0.5, 0.9)
        if rng.random() < 0.2:
            y = np.clip(y * 10 ** (rng.uniform(6, 18) / 20), -1, 1)
        if rng.random() < 0.5:
            n = np.random.default_rng(rng.randint(0, 2 ** 31)).standard_normal(len(y)).astype(np.float32)
            y = y + n * np.sqrt(np.mean(y ** 2)) * 10 ** (-rng.uniform(30, 55) / 20)
        return np.clip(y, -1, 1).astype(np.float32)


def labels_for(n_frames, spans):
    """Frame labels for wake words at sample spans ``(start, end)``: 1 from ``POS_FROM`` to ``POS_TO`` frames after the
    word ends, unscored (-1) from its middle until then and until ``IGNORE_TO`` frames after it, 0 elsewhere."""
    y = np.zeros(n_frames, np.int8)
    for s, e in spans:
        sf_, ef = s // HOP, e // HOP
        y[max(0, (sf_ + ef) // 2):min(n_frames, ef + IGNORE_TO + 1)] = -1
        y[max(0, ef + POS_FROM):min(n_frames, ef + POS_TO + 1)] = 1
    return y


def make_stream(pools, words, seconds, per_stream, snr=(0, 25)):
    """(audio, spans): background with ``per_stream`` (Poisson mean) wake words at least 3 s apart, each at an SNR
    above its local background, the whole stream through ``device``."""
    n = int(seconds * SR)
    bg = pools.background(n)
    k = np.random.default_rng(pools.rng.randint(0, 2 ** 31)).poisson(per_stream) if words else 0
    spans, y = [], bg.copy()
    slots = range(int(2 * SR), n - 3 * SR, int(3.5 * SR))
    starts = sorted(pools.rng.sample(slots, min(k, len(slots)))) if k else []
    for s in starts:
        w = pools.word(read(pools.rng.choice(words)))
        e = min(n, s + len(w))
        w = w[:e - s]
        local = np.sqrt(np.mean(bg[max(0, s - SR):e] ** 2)) + 1e-6
        gain = local * 10 ** (pools.rng.uniform(*snr) / 20) / (np.sqrt(np.mean(w ** 2)) + 1e-9)
        if pools.rng.random() < 0.15:
            gain = 0.05 * 10 ** (-pools.rng.uniform(0, 20) / 20) / (np.sqrt(np.mean(w ** 2)) + 1e-9)
        y[s:e] += w * gain
        spans.append((s, e))
    return pools.device(y), spans


def featurize(stream, wav):
    stream.reset()
    return stream.run(wav).astype(np.float16)


def cmd_streams(a):
    rng = random.Random(f"{a.word}:{a.seed}:{a.part}")
    pos = [] if a.word == "none" else split_by_file(files(*a.pos_dirs))[0]
    speech = files(*a.speech_dirs)
    noise = even_odd(files(a.noise_dir))[0]
    pools = Pools(speech, files(*a.window_dirs), noise, files(a.rir_dir) if a.rir_dir else [], rng)
    stream = st.OnnxStream(a.trunk)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    feats, labels, t0, k, n, n_pos = [], [], time.time(), 0, 0, 0

    def flush():
        nonlocal feats, labels, k
        if feats:
            stem = out / f"{a.word}-{a.part:02d}-{k:03d}"
            np.save(f"{stem}.feats.npy", np.stack(feats))
            np.save(f"{stem}.labels.npy", np.stack(labels))
            feats, labels, k = [], [], k + 1

    for i in range(a.part, a.n_streams, a.parts):
        wav, spans = make_stream(pools, pos, a.seconds, a.per_stream)
        f = featurize(stream, wav)
        feats.append(f)
        labels.append(labels_for(len(f), spans))
        n, n_pos = n + 1, n_pos + int((labels[-1] == 1).sum())
        if len(feats) == a.shard_streams:
            flush()
    flush()
    print(json.dumps({"word": a.word, "part": a.part, "streams": n, "seconds": round(time.time() - t0, 1),
                      "positive_frames": n_pos}), flush=True)


def calib_positive_stream(pools, clip):
    """A held-out wake word after 3 s of calibration background, then 1.5 s more of it."""
    n = int(4.5 * SR) + len(trim(clip))
    bg = pools.background(n)
    w = trim(clip)
    s = 3 * SR
    local = np.sqrt(np.mean(bg[s - SR:s + len(w)] ** 2)) + 1e-6
    snr = pools.rng.choice([5.0, 10.0, 20.0])
    y = bg.copy()
    y[s:s + len(w)] += w * local * 10 ** (snr / 20) / (np.sqrt(np.mean(w ** 2)) + 1e-9)
    y = y / max(1.0, np.abs(y).max())
    return y.astype(np.float32), s, s + len(w)


def cmd_calib(a):
    """Calibration negatives: ``--hours`` of continuous background from calibration-only sources; calibration
    positives: the held-out tenth of the word's clips, each inside calibration background."""
    rng = random.Random(f"calib:{a.seed}:{a.part}")
    noise = even_odd(files(a.noise_dir))[1]
    pools = Pools(files(*a.speech_dirs), files(*a.speech_dirs), noise, [], rng)
    stream = st.OnnxStream(a.trunk)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.word == "none":
        chunk = 600
        n = int(a.hours * 3600 / chunk)
        feats = [featurize(stream, pools.background(chunk * SR)) for i in range(a.part, n, a.parts)]
        np.savez(out / f"negs-{a.part:02d}.npz", feats=np.concatenate(feats))
        return
    held = split_by_file(files(*a.pos_dirs))[1]
    rows = []
    for i in range(a.part, len(held), a.parts):
        wav, s, e = calib_positive_stream(pools, read(held[i]))
        f = featurize(stream, wav)
        rows.append((f, s // BLOCK, (e + int(1.5 * SR)) // BLOCK))
    np.save(out / f"{a.word}-{a.part:02d}.npy", np.array(rows, dtype=object), allow_pickle=True)


def _sessions(published, head_paths):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    f = ort.InferenceSession(published, so, providers=["CPUExecutionProvider"])
    return f, {Path(p).stem: ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]) for p in head_paths}


CTX = 38 * BLOCK


def pieces(paths, piece):
    """A directory's files as one signal, cut into ``piece``-sample pieces (the last one block-aligned)."""
    acc, got = [], 0
    for p in paths:
        w = read(p)
        acc.append(w)
        got += len(w)
        if got >= piece:
            allw = np.concatenate(acc)
            while len(allw) >= piece:
                yield allw[:piece]
                allw = allw[piece:]
            acc, got = [allw], len(allw)
    rest = np.concatenate(acc) if acc else np.zeros(0, np.float32)
    rest = rest[:len(rest) // BLOCK * BLOCK]
    if len(rest):
        yield rest


def cmd_negs(a):
    """Each directory of ``--dirs`` is one continuous stream (its files in order), cut into 10-minute pieces. A piece
    is streamed after the last 3.04 s of the piece before it, longer than the trunk's 2.5 s receptive field, so its
    frames equal those of the whole stream streamed at once. Writes the streamed features and the isolated-window
    logits of every published head, per piece."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stream = st.OnnxStream(a.trunk)
    fsess, heads = _sessions(a.published_trunk, a.heads) if a.heads else (None, {})
    k = 0
    for d in a.dirs:
        prev = np.zeros(0, np.float32)
        for j, wav in enumerate(pieces(files(d), 600 * SR)):
            if k % a.parts == a.part:
                ctx = prev[-CTX:]
                stream.reset()
                feats = stream.run(np.concatenate([ctx, wav]))[len(ctx) // HOP:]
                rec = {"seconds": len(wav) / SR, "feats": feats.astype(np.float16)}
                rec.update({f"iso_{h}": v for h, v in _isolated_logits_ctx(fsess, heads, prev[-WINDOW:], wav).items()})
                np.savez(out / f"{Path(d).name}-{j:04d}.npz", **rec)
            prev = wav
            k += 1
    print("done", a.part, flush=True)


def _isolated_logits(fsess, heads, wav):
    return _isolated_logits_ctx(fsess, heads, np.zeros(0, np.float32), wav)


def _isolated_logits_ctx(fsess, heads, ctx, wav):
    """Per-block logits of the plugin's isolated windows: after each block of ``wav``, the last 1.5 s (``ctx`` before
    ``wav``, zeros before that) featurized alone and scored by every head."""
    if not heads:
        return {}
    full = np.concatenate([np.zeros(WINDOW - len(ctx), np.float32), ctx, wav])
    name = fsess.get_inputs()[0].name
    out = {k: [] for k in heads}
    nb = len(wav) // BLOCK
    for i in range(0, nb, 16):
        wins = np.stack([full[(j + 1) * BLOCK:(j + 1) * BLOCK + WINDOW] for j in range(i, min(nb, i + 16))])
        feats = fsess.run(None, {name: wins})[0]
        for k, h in heads.items():
            out[k].append(h.run(None, {h.get_inputs()[0].name: feats})[0].reshape(-1))
    return {k: np.concatenate(v) for k, v in out.items()}


def lead_in(kind, rng, lead_dirs):
    if kind == "quiet":
        return np.zeros(3 * SR, np.float32)
    w = read(rng.choice(files(*lead_dirs)))
    w = np.tile(w, 3 * SR // len(w) + 1)[:3 * SR]
    return (w / (np.sqrt(np.mean(w ** 2)) + 1e-9) * 0.03).astype(np.float32)


def cmd_pos(a):
    """Each test clip after a 3 s lead-in (``quiet``: zeros; ``speech``: read speech at a fixed level), then 1.5 s of
    zeros. Detection is read from the blocks from the clip's start to 1.5 s after it."""
    rng = random.Random(f"pos:{a.lead}")
    clips = files(*a.dirs)
    stream = st.OnnxStream(a.trunk)
    fsess, heads = _sessions(a.published_trunk, a.heads) if a.heads else (None, {})
    rows = []
    leads = [lead_in(a.lead, rng, a.lead_dirs) for _ in clips]
    for i in range(a.part, len(clips), a.parts):
        try:
            c = read(clips[i])
        except sf.LibsndfileError as e:
            print(f"skipped {clips[i]}: {e}", flush=True)
            continue
        c = c / max(1.0, np.abs(c).max())
        wav = np.concatenate([leads[i], c, np.zeros(int(1.5 * SR), np.float32)])
        wav = wav[:len(wav) // BLOCK * BLOCK]
        rec = {"file": str(clips[i]), "first_block": len(leads[i]) // BLOCK}
        stream.reset()
        rec["feats"] = stream.run(wav).astype(np.float16)
        if heads:
            rec.update({f"iso_{k}": v for k, v in _isolated_logits(fsess, heads, wav).items()})
        rows.append(rec)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{a.lead}-{a.part:02d}.npy", np.array(rows, dtype=object), allow_pickle=True)


# ------------------------------------------------------------------ heads
class WindowHead(nn.Module):
    """The shipped head: a GRU over a window of frames, the mean over time, a ReLU linear layer and one logit."""

    def __init__(self, dim=128, hidden=128, linear=128):
        super().__init__()
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.fc1 = nn.Linear(hidden, linear)
        self.fc2 = nn.Linear(linear, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc2(F.relu(self.fc1(out.mean(1)))).squeeze(-1)


class Affine(nn.Module):
    def __init__(self, head, scale, shift):
        super().__init__()
        self.head, self.scale, self.shift = head, scale, shift

    def forward(self, x):
        return self.scale * self.head(x) + self.shift


def shard_paths(shards, word):
    return sorted(glob.glob(f"{shards}/{word}-*.feats.npy"))


def load_shards(paths):
    """Memory-mapped features ``[n, T, D]`` and labels ``[n, T]`` of every shard."""
    return [(np.load(p, mmap_mode="r"), np.load(p.replace(".feats.npy", ".labels.npy"))) for p in paths]


def window_index(shards, frames=WINDOW_FRAMES):
    """(positives, negatives) as rows ``(shard, stream, end)``: every block end with a full window behind it,
    labelled by the frame before the end."""
    pos, neg = [], []
    for k, (_, lab) in enumerate(shards):
        ends = np.arange(-(-frames // FPB) * FPB, lab.shape[1] + 1, FPB)
        y = lab[:, ends - 1]
        for v, out in ((1, pos), (0, neg)):
            i, j = np.nonzero(y == v)
            out.append(np.stack([np.full(len(i), k), i, ends[j]], 1))
    return np.concatenate(pos), np.concatenate(neg)


def gather(shards, rows, frames=WINDOW_FRAMES):
    return torch.from_numpy(np.stack([shards[k][0][i, e - frames:e] for k, i, e in rows]).astype(np.float32))


def head_logits(head, x, batch=1024):
    with torch.no_grad():
        return torch.cat([head(x[i:i + batch]) for i in range(0, len(x), batch)]).numpy().astype(np.float64)


def head_window_logits(head, feats, frames=WINDOW_FRAMES, batch=1024):
    """Per-block logits of a torch window head over the last ``frames`` frames at every block end of ``feats``;
    blocks before a full window score -inf."""
    nb = len(feats) // FPB
    ends = np.arange(1, nb + 1) * FPB
    ok = np.flatnonzero(ends >= frames)
    out = np.full(nb, -np.inf)
    with torch.no_grad():
        for i in range(0, len(ok), batch):
            x = torch.from_numpy(np.stack([feats[e - frames:e] for e in ends[ok[i:i + batch]]]).astype(np.float32))
            out[ok[i:i + batch]] = head(x).numpy()
    return out


def calibration_scores(head, calib, word):
    """(calibration positives' best logit, calibration stream block logits, stream hours)."""
    negs = [np.load(p)["feats"] for p in sorted(glob.glob(f"{calib}/negs-*.npz"))]
    z = np.concatenate([head_window_logits(head, f) for f in negs])
    rows = load_rows(f"{calib}/{word}-*.npy")
    pos = np.array([head_window_logits(head, f)[s:e + 1].max() for f, s, e in rows])
    return pos, z, len(z) * BLOCK / SR / 3600


def cmd_fit(a):
    """Train the window head on windows of streamed features: every positive window and five negatives per positive
    each epoch, half of them the negatives the head scored highest last epoch; keep the epoch whose calibration
    positives most often beat every calibration stream block."""
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    shards = load_shards(shard_paths(a.shards, a.word) + shard_paths(a.shards, "none"))
    pos, neg = window_index(shards)
    pool = neg[rng.choice(len(neg), min(len(neg), a.neg_pool), replace=False)]
    n_neg = min(5 * len(pos), len(pool))
    hard = pool[rng.choice(len(pool), n_neg // 2, replace=False)]
    head = WindowHead()
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log, best = open(out.with_suffix(".jsonl"), "w"), (-1.0, 0)
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        rand = pool[rng.choice(len(pool), n_neg - len(hard), replace=False)]
        rows = np.concatenate([pos, hard, rand])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(hard) + len(rand))]).astype(np.float32)
        order = rng.permutation(len(rows))
        head.train()
        for i in range(0, len(order), a.batch):
            b = order[i:i + a.batch]
            loss = F.binary_cross_entropy_with_logits(head(gather(shards, rows[b])), torch.from_numpy(y[b]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        scores = np.concatenate([head_logits(head, gather(shards, pool[i:i + 4096])) for i in range(0, len(pool), 4096)])
        hard = pool[np.argsort(scores)[-(n_neg // 2):]]
        cpos, cz, _ = calibration_scores(head, a.calib, a.word)
        cal = float((cpos > cz.max()).mean())
        rec = {"epoch": ep, "seconds": round(time.time() - t0, 1), "loss": float(loss), "calib_recall": cal,
               "calib_recall_at_p999": float((cpos > np.quantile(cz, 0.999)).mean())}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        if (rec["calib_recall_at_p999"], cal) > best:
            best = (rec["calib_recall_at_p999"], cal)
            torch.save({"state": head.state_dict(), "word": a.word, "epoch": ep, "seed": a.seed,
                        "positives": int(len(pos)), "negatives": int(len(neg))}, out)


def debounced_events(z, thr):
    above = np.flatnonzero(np.asarray(z) >= thr)
    n, j = 0, 0
    while j < len(above):
        n += 1
        j = int(np.searchsorted(above, above[j] + DEBOUNCE_BLOCKS + 1))
    return n


def threshold_for_rate(streams, hours, rate):
    """Lowest threshold whose debounced detections over ``streams`` (lists of per-block logits) stay at or below
    ``rate`` per hour."""
    allz = np.concatenate(streams)
    lo, hi = float(allz.min()), float(allz.max()) + 1e-3
    want = rate * hours
    for _ in range(50):
        mid = (lo + hi) / 2
        if sum(debounced_events(z, mid) for z in streams) > want:
            lo = mid
        else:
            hi = mid
    return hi


def load_rows(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        rows += list(np.load(p, allow_pickle=True))
    return rows


def window_logits(sess, feats, frames=75, batch=512):
    """Per-block logits of a window head over the last ``frames`` streamed frames at the end of every block; blocks
    before ``frames`` frames exist score -inf."""
    name = sess.get_inputs()[0].name
    nb = len(feats) // FPB
    ends = np.arange(1, nb + 1) * FPB
    ok = np.flatnonzero(ends >= frames)
    out = np.full(nb, -np.inf)
    for i in range(0, len(ok), batch):
        e = ends[ok[i:i + batch]]
        x = np.stack([feats[k - frames:k] for k in e]).astype(np.float32)
        out[ok[i:i + batch]] = sess.run(None, {name: x})[0].reshape(-1)
    return out


def _by_dir(negs):
    by = {}
    for p in sorted(glob.glob(f"{negs}/*.npz")):
        by.setdefault(Path(p).name.rsplit("-", 1)[0], []).append(p)
    return by


def _rates(z_neg, hours, pos_sets, thr):
    fa = sum(debounced_events(z, thr) for z in z_neg) / hours
    return fa, {k: float(np.mean(v >= thr)) for k, v in pos_sets.items()}


def _report(z_neg, hours, pos_max, rate, default=None):
    thr = threshold_for_rate(z_neg, hours, rate)
    rec = {"threshold_at_rate": thr,
           "recall_at_rate": {k: round(float(np.mean(v >= thr)), 4) for k, v in pos_max.items()},
           "clips": {k: len(v) for k, v in pos_max.items()}}
    if default is not None:
        t = math.log(default / (1 - default))
        fa, rc = _rates(z_neg, hours, pos_max, t)
        rec.update({"default_threshold": default, "fa_per_hour_at_default": round(fa, 3),
                    "recall_at_default": {k: round(v, 4) for k, v in rc.items()}})
    return rec


TAIL = 19 * FPB


def _neg_logits(by, piece_logits):
    """Per directory, the per-block logits of its pieces in order, one piece in memory at a time: ``piece_logits``
    gets each piece and the streamed frames carried from the piece before (``TAIL`` frames, at least a window), and
    returns that piece's block logits."""
    out, seconds = [], 0.0
    for ps in by.values():
        zs, tail = [], None
        for p in ps:
            d = np.load(p)
            seconds += float(d["seconds"])
            zs.append(piece_logits(d, tail))
            tail = d["feats"][-TAIL:]
        out.append(np.concatenate(zs))
    return out, seconds / 3600


def _window_piece(sess):
    def run(d, tail):
        f = d["feats"] if tail is None else np.concatenate([tail, d["feats"]])
        z = window_logits(sess, f)
        return z if tail is None else z[len(tail) // FPB:]
    return run


def cmd_score(a):
    """Recall at ``--rate`` false activations per hour, and recall and false activations at the calibrated default,
    for shipped heads on isolated windows (``iso_<name>`` logits) and on the last 75 streamed frames. Negatives are
    each directory's pieces in order, scored continuously, one piece in memory at a time."""
    import onnx
    import onnxruntime as ort
    by = _by_dir(a.negs)
    pos = {}
    for spec in a.pos:
        name, pattern = spec.split("=", 1)
        pos[name] = load_rows(pattern)
    res = {}
    for spec in a.shipped:
        name, path = spec.split("=", 1)
        meta = {p.key: p.value for p in onnx.load(path).metadata_props}
        default = float(meta["default_threshold"]) if "default_threshold" in meta else None
        mine = {k: v for k, v in pos.items() if k.startswith(name)}
        z_iso, hours = _neg_logits(by, lambda d, tail: d[f"iso_{name}"])
        res["negative_hours"] = round(hours, 2)
        pmax = {k: np.array([r[f"iso_{name}"][r["first_block"]:].max() for r in rows]) for k, rows in mine.items()}
        res[f"{name}:isolated"] = _report(z_iso, hours, pmax, a.rate, default)
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        z_st, _ = _neg_logits(by, _window_piece(sess))
        pmax = {k: np.array([window_logits(sess, r["feats"])[r["first_block"]:].max() for r in rows])
                for k, rows in mine.items()}
        res[f"{name}:streamed_window"] = _report(z_st, hours, pmax, a.rate, default)
        np.savez(Path(a.out).with_suffix(f".{name}.npz"), **{f"iso_{i}": z for i, z in enumerate(z_iso)},
                 **{f"st_{i}": z for i, z in enumerate(z_st)})
    for spec in a.window:
        name, path = spec.split("=", 1)
        word = name.split(":")[0]
        meta = {p.key: p.value for p in onnx.load(path).metadata_props}
        default = float(meta["default_threshold"]) if "default_threshold" in meta else None
        mine = {k: v for k, v in pos.items() if k.startswith(word)}
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        z_st, hours = _neg_logits(by, _window_piece(sess))
        res["negative_hours"] = round(hours, 2)
        pmax = {k: np.array([window_logits(sess, r["feats"])[r["first_block"]:].max() for r in rows])
                for k, rows in mine.items()}
        res[f"{name}:streamed_window"] = _report(z_st, hours, pmax, a.rate, default)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


MIN_EVENTS = 20
CALIB_POS_PROB = 0.9
F2_PROBABILITIES = np.arange(5, 100) / 100


def stream_threshold(z, hours, rate):
    """Logit at which debounced detections on the calibration stream ``z`` come to ``rate`` per hour; below
    ``MIN_EVENTS`` detections the tail is taken as exponential through the thresholds for ``MIN_EVENTS`` and three
    times as many, and extended (the rule of ``head_bench.py export --calibrate``)."""
    want = rate * hours
    if want >= MIN_EVENTS:
        return threshold_for_rate([z], hours, rate), False
    hi, lo = threshold_for_rate([z], hours, MIN_EVENTS / hours), threshold_for_rate([z], hours, 3 * MIN_EVENTS / hours)
    return hi + math.log(MIN_EVENTS / want) * (hi - lo) / math.log(3), True


def f2_operating_point(pos, stream, hours, positives_per_hour=4.0):
    wanted, best = positives_per_hour * hours, None
    for prob in F2_PROBABILITIES:
        thr = math.log(prob / (1 - prob))
        recall = float((pos >= thr).mean())
        hits = recall * wanted
        precision = hits / (hits + debounced_events(stream, thr)) if hits else 0.0
        f2 = 5 * precision * recall / (4 * precision + recall) if hits else 0.0
        if best is None or f2 >= best[0]:
            best = (f2, float(prob), recall)
    return {"f2": best[0], "f2_threshold": best[1], "f2_recall": best[2]}


def cmd_export(a):
    """Calibrate a trained window head on the calibration stream and export it as the plugin's head ONNX: input
    ``features`` [batch, frames, 128], output ``logit``, the affine map inside. The stream's 1 FA/h logit maps to
    probability 0.5 and the median calibration positive to 0.9; the F2-optimal threshold becomes
    ``default_threshold``."""
    import onnx

    ck = torch.load(a.head, map_location="cpu")
    head = WindowHead()
    head.load_state_dict(ck["state"])
    head.eval()
    pos, z, hours = calibration_scores(head, a.calib, ck["word"])
    z_thr, extrapolated = stream_threshold(z, hours, a.rate)
    gap = float(np.median(pos)) - z_thr
    if gap <= 0:
        raise SystemExit(f"median calibration positive {np.median(pos):.3f} is below the {a.rate} FA/h logit {z_thr:.3f}")
    scale = math.log(CALIB_POS_PROB / (1 - CALIB_POS_PROB)) / gap
    shift = -scale * z_thr
    info = {"calib_a": scale, "calib_b": shift, "calib_hours": round(hours, 3), "calib_extrapolated": extrapolated,
            "calib_positives": len(pos), **f2_operating_point(scale * pos + shift, scale * z + shift, hours)}
    model = Affine(head, scale, shift).eval()
    with torch.no_grad():
        torch.onnx.export(model, torch.zeros(1, WINDOW_FRAMES, 128), a.out, input_names=["features"],
                          output_names=["logit"], dynamic_axes={"features": {0: "batch", 1: "frames"},
                                                                "logit": {0: "batch"}},
                          opset_version=17, dynamo=False)
    m = onnx.load(a.out)
    meta = {"wake_word": ck["word"].replace("_", " "), "pretrained_featurizer": "wakehubert-int8",
            "featurizer_features": "streamed", "window_frames": str(WINDOW_FRAMES), "calibrated": "1",
            "default_threshold": f"{info['f2_threshold']:.2f}", "training_data": a.training_data,
            "calib_a": f"{scale:.6f}", "calib_b": f"{shift:.6f}"}
    for k, v in meta.items():
        e = m.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(m, a.out)
    Path(a.out).with_suffix(".json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("streams")
    p.add_argument("--word", required=True)
    p.add_argument("--pos-dirs", nargs="*", default=[])
    p.add_argument("--speech-dirs", nargs="+", required=True)
    p.add_argument("--window-dirs", nargs="+", required=True)
    p.add_argument("--noise-dir", required=True)
    p.add_argument("--rir-dir", default=None)
    p.add_argument("--n-streams", type=int, default=3600)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--per-stream", type=float, default=3.0)
    p.add_argument("--shard-streams", type=int, default=200, help="streams per shard file")
    q = sub.add_parser("calib")
    q.add_argument("--word", required=True)
    q.add_argument("--pos-dirs", nargs="*", default=[])
    q.add_argument("--speech-dirs", nargs="+", required=True)
    q.add_argument("--noise-dir", required=True)
    q.add_argument("--hours", type=float, default=3.0)
    r = sub.add_parser("negs")
    r.add_argument("--dirs", nargs="+", required=True)
    r.add_argument("--published-trunk", default=None)
    r.add_argument("--heads", nargs="*", default=[])
    r.add_argument("--streamed", action="store_true")
    s = sub.add_parser("pos")
    s.add_argument("--dirs", nargs="+", required=True)
    s.add_argument("--lead", choices=["quiet", "speech"], default="quiet")
    s.add_argument("--lead-dirs", nargs="*", default=[])
    s.add_argument("--published-trunk", default=None)
    s.add_argument("--heads", nargs="*", default=[])
    for x in (p, q, r, s):
        x.add_argument("--trunk", required=True, help="stateful streaming trunk ONNX (int8)")
        x.add_argument("--out", required=True)
        x.add_argument("--part", type=int, default=0)
        x.add_argument("--parts", type=int, default=1)
        x.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("fit")
    t.add_argument("--shards", required=True)
    t.add_argument("--word", required=True)
    t.add_argument("--calib", required=True, help="output of calib: negs-*.npz and <word>-*.npy")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--neg-pool", type=int, default=200000, help="negative windows hard negatives are mined from")
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--seed", type=int, default=42)
    u = sub.add_parser("score")
    u.add_argument("--negs", required=True)
    u.add_argument("--pos", nargs="+", required=True, help="set=glob of pos rows; a set scores the heads whose name it starts with")
    u.add_argument("--shipped", nargs="*", default=[], help="name=head.onnx, a window head whose iso_<name> logits are in negs/pos")
    u.add_argument("--rate", type=float, default=1.0)
    u.add_argument("--out", default=None)
    v = sub.add_parser("export")
    v.add_argument("--head", required=True)
    v.add_argument("--calib", required=True)
    v.add_argument("--rate", type=float, default=1.0)
    v.add_argument("--training-data", required=True, help="the training data statement written into the metadata")
    v.add_argument("--out", required=True)
    u.add_argument("--window", nargs="*", default=[], help="name=head.onnx, a window head scored on streamed frames only")
    a = ap.parse_args(argv)
    {"streams": cmd_streams, "calib": cmd_calib, "negs": cmd_negs, "pos": cmd_pos, "fit": cmd_fit, "export": cmd_export,
     "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
