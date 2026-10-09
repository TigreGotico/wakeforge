"""Wake-word heads on streamed WakeHuBERT-tiny features: data, training, calibration and scoring.

The streaming trunk (``stream_trunk.py``) carries its state for as long as the stream runs, so the frames at any
moment depend on everything heard before. The shipped heads were trained on isolated 1.5 s windows that start from
silence, and on streamed frames they lose recall and gain false activations. This module trains the same head, a GRU
over the last 75 frames scored every 80 ms block, on windows of streamed features of continuous synthetic streams,
and scores it on the same footing as the isolated models.

``--head gru`` trains the stateful alternative for streaming: ``StreamGRUHead`` reads every frame once, carries its
hidden state across blocks for as long as the stream runs and gives a logit at the end of every 80 ms block. It is
trained on random crops of the shard streams (truncated backpropagation through time): each crop starts with a burn-in
whose loss is masked, so the state has settled when the scored blocks begin. A device runs the head from the moment it
starts listening, so no score starts from a state the device could not have. The calibration negatives run as one
continuous stream per file, from a zero state. Every positive (calibration row or evaluation row) starts from the state
left by the first 30 s (``WARM_FRAMES`` frames) of the negative features: the first calibration negatives file for
``fit`` and ``export``, the first piece of the first directory of ``--negs`` for ``score``.

Subcommands (each audio-heavy one takes ``--part i --parts n`` so several cores share it):

    streams  training streams for a word (or negative-only streams with --word none), featurized: shards of
             features [n, T, 128] float16 and frame labels (1 wake word, 0 not, -1 unscored), as .npy files
    calib    the calibration stream (continuous negatives) and calibration positives, featurized
    negs     evaluation negatives as continuous streams: streamed features, and the plugin's isolated-window logits
             of published heads
    pos      evaluation positives, each after a lead-in: streamed features, and isolated-window logits
    fit      the window head (or with --head gru the stateful GRU head) on the shards of one word and the shared
             negative shards
    export   calibrate a trained head on the calibration stream and write its head ONNX (the plugin format for a
             window head; the published plugin cannot run a GRU head)
    score    recall at a false-activation rate and at the calibrated default, for shipped heads on isolated windows
             and on streamed frames, and for exported window and GRU heads on streamed frames

The positive clips of a word come from ``positive_sources.json`` beside this file: per word, the directories (relative
to ``--data-root``, or a folder of a Hugging Face dataset snapshot) and the synthesis records that name each clip's
voice. ``streams`` trains on the voices that ``held_out`` keeps, ``calib`` scores the others, so no voice is on both
sides. ``streams`` and ``calib`` write ``trunk.json`` beside their outputs: the streaming trunk's sha256 and the
pretrained featurizer it was made from. ``fit`` refuses shards and calibration from different trunks, and ``export``
writes that featurizer into the head's metadata.
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import random
import re
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


SOURCES = Path(__file__).resolve().parent / "positive_sources.json"
HELD_EVERY = 10


def held_out(key, every=HELD_EVERY):
    """True for one key in ``every``, chosen by the key's sha256 alone."""
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % every == 0


def split_by_voice(pool):
    """(train, held) paths of ``pool`` rows ``(path, voice)``: a voice is held out by ``held_out`` of its name, so a
    file's side does not depend on the other files, their order or the sources they came from."""
    return ([p for p, v in pool if not held_out(v)], sorted(p for p, v in pool if held_out(v)))


def noise_halves(paths):
    """(training half, calibration half) of noise files, each file placed by ``held_out`` of its name."""
    return [p for p in paths if not held_out(Path(p).name, 2)], [p for p in paths if held_out(Path(p).name, 2)]


def resolve(spec, root):
    """A source path: a string relative to ``root``, or ``{"hf": repo, "revision": rev, "path": p}``, the path ``p``
    inside a Hugging Face dataset snapshot (read from the local cache; fetched only when it is not there)."""
    if isinstance(spec, str):
        return Path(root) / spec
    from huggingface_hub import snapshot_download
    snap = snapshot_download(spec["hf"], repo_type="dataset", revision=spec["revision"],
                             allow_patterns=[spec["path"], f"{spec['path']}/*"])
    return Path(snap) / spec["path"]


def voice_index(path):
    """``{file as the record names it: (voice, source clip)}`` from a synthesis record: a Hub ``metadata.jsonl``
    (``file_name``, ``voice`` or ``speaker_id``), a TTS ``synthesis.csv`` (``file``, ``options`` with ``voice`` or
    ``lang``) or a voice manifest (``file``, ``voice_or_seed``). A voice of the form ``mode; name`` keeps the name."""
    path = Path(path)
    with path.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()] if path.suffix == ".jsonl" else list(csv.DictReader(fh))
    out = {}
    for r in rows:
        opts = json.loads(r.get("options") or "{}")
        voice = next((str(v) for v in (r.get("voice"), r.get("speaker_id"), r.get("voice_or_seed"), opts.get("voice"),
                                       opts.get("lang")) if v), "")
        out[r.get("file") or r["file_name"]] = (voice.split(";")[-1].strip(), r.get("source_clip") or "")
    return out


def positive_pool(sources, root, exclude=None):
    """``(path, voice)`` of every clip of ``sources`` (one word's entries of ``positive_sources.json``).

    A source has ``dir`` and optionally ``metadata`` (a synthesis record, see ``voice_index``; a clip is found under
    the longest end of its path that the record names, the path taken from the record's directory when the clip
    lies under it), ``voice_pattern`` (a regex whose group 1 is the voice in the clip's path) and ``listed_only``
    (skip clips the record does not name). A clip with no voice gets its directory as voice, so unknown speakers of
    one directory stay together. Clips whose path, voice or source clip match ``exclude`` (the evaluation voices)
    are left out."""
    skip = re.compile(exclude) if exclude else None
    pool = []
    for src in sources:
        d = resolve(src["dir"], root)
        record = resolve(src["metadata"], root) if "metadata" in src else None
        index = voice_index(record) if record else {}
        pattern = re.compile(src["voice_pattern"]) if "voice_pattern" in src else None
        for p in files(d):
            rel = p.relative_to(d).as_posix()
            base = record.parent if record and p.is_relative_to(record.parent) else d
            named = p.relative_to(base).parts
            hit = next((index[k] for k in ("/".join(named[i:]) for i in range(len(named))) if k in index), None)
            if hit is None and src.get("listed_only"):
                continue
            voice, origin = hit or ("", "")
            if not voice and pattern and (m := pattern.search(rel)):
                voice = m.group(1)
            if skip and skip.search(f"{rel} {voice} {origin}"):
                continue
            pool.append((p, voice or f"dir:{p.parent.name}"))
    return pool


def word_pool(a):
    """The word's positive pool from ``--sources`` under ``--data-root``."""
    cfg = json.loads(Path(a.sources).read_text())
    if a.word not in cfg["words"]:
        raise SystemExit(f"{a.sources} has no positive sources for {a.word!r}")
    if not a.data_root:
        raise SystemExit("--data-root is needed to find the positive sources")
    return positive_pool(cfg["words"][a.word]["sources"], a.data_root, cfg.get("exclude"))


QUANT_OPS = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear", "QLinearConv", "QLinearMatMul",
             "ConvInteger", "MatMulInteger"}


def trunk_record(path):
    """What a streaming trunk was made from: its own sha256, and the pretrained featurizer file it was streamified
    from (``source_sha256`` and ``source_revision`` of its metadata) with the name the plugin knows it by."""
    import onnx
    m = onnx.load(str(path))
    meta = {p.key: p.value for p in m.metadata_props}
    int8 = any(n.op_type in QUANT_OPS for n in m.graph.node)
    return {"trunk_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "pretrained_featurizer": "wakehubert-int8" if int8 else "wakehubert",
            "featurizer_sha256": meta.get("source_sha256", ""), "featurizer_revision": meta.get("source_revision", "")}


def write_trunk_record(out, record):
    """Write ``trunk.json`` in ``out``; refuse when ``out`` already holds outputs of another trunk."""
    p = Path(out) / "trunk.json"
    if p.exists() and json.loads(p.read_text()) != record:
        raise SystemExit(f"{out} holds outputs of another trunk ({p}); write to a new directory")
    p.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")


def read_trunk_record(d):
    p = Path(d) / "trunk.json"
    if not p.exists():
        raise SystemExit(f"{d} has no trunk.json: build it with this script's streams or calib")
    return json.loads(p.read_text())


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
    pos = [] if a.word == "none" else split_by_voice(word_pool(a))[0]
    speech = files(*a.speech_dirs)
    noise = noise_halves(files(a.noise_dir))[0]
    pools = Pools(speech, files(*a.window_dirs), noise, files(a.rir_dir) if a.rir_dir else [], rng)
    stream = st.OnnxStream(a.trunk)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_trunk_record(out, trunk_record(a.trunk))
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
    positives: the clips of the word's held-out voices, each inside calibration background."""
    rng = random.Random(f"calib:{a.seed}:{a.part}")
    noise = noise_halves(files(a.noise_dir))[1]
    pools = Pools(files(*a.speech_dirs), files(*a.speech_dirs), noise, [], rng)
    stream = st.OnnxStream(a.trunk)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_trunk_record(out, trunk_record(a.trunk))
    if a.word == "none":
        chunk = 600
        n = int(a.hours * 3600 / chunk)
        feats = [featurize(stream, pools.background(chunk * SR)) for i in range(a.part, n, a.parts)]
        np.savez(out / f"negs-{a.part:02d}.npz", feats=np.concatenate(feats))
        return
    held = split_by_voice(word_pool(a))[1]
    if not held:
        raise SystemExit(f"no voice of {a.word!r} is held out: the calibration needs positives of unseen voices")
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
NEG_PIECE = 600 * SR


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
        for j, wav in enumerate(pieces(files(d), NEG_PIECE)):
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


class StreamGRUHead(nn.Module):
    """A GRU over the frames with a ReLU linear layer and one logit read at the end of every block of ``FPB`` frames.
    ``forward(x, h)`` returns the block logits ``[batch, frames // FPB]`` and the hidden state ``[1, batch, hidden]``
    after the last frame; ``h`` is carried by the caller and the head never resets it."""

    def __init__(self, dim=128, hidden=128, linear=128):
        super().__init__()
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.fc1 = nn.Linear(hidden, linear)
        self.fc2 = nn.Linear(linear, 1)

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        return self.fc2(F.relu(self.fc1(out[:, FPB - 1::FPB]))).squeeze(-1), h


class Affine(nn.Module):
    def __init__(self, head, scale, shift):
        super().__init__()
        self.head, self.scale, self.shift = head, scale, shift

    def forward(self, x):
        return self.scale * self.head(x) + self.shift


class AffineBlock(nn.Module):
    """A GRU head on one block with the affine calibration: ``(features, h)`` to ``(logit_calibrated, h_out)``."""

    def __init__(self, head, scale, shift):
        super().__init__()
        self.head, self.scale, self.shift = head, scale, shift

    def forward(self, x, h):
        z, h = self.head(x, h)
        return self.scale * z.reshape(-1) + self.shift, h


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


def hardest(rows, scores, k):
    """The ``k`` rows with the highest scores."""
    return rows[np.argsort(scores)[len(scores) - k:]]


def check_trunks(a):
    """The trunk record of the shards, refusing a calibration made by another trunk."""
    trunk = read_trunk_record(a.shards)
    if read_trunk_record(a.calib) != trunk:
        raise SystemExit(f"{a.shards} and {a.calib} were featurized by different trunks: "
                         f"{trunk} against {read_trunk_record(a.calib)}")
    return trunk


def epoch_record(ep, t0, loss, cpos, cz, hours, rate):
    """The log line of an epoch: recall of the calibration positives at the calibration stream's maxima, at its
    ``rate`` threshold and at its 99.9th percentile."""
    thr = threshold_for_rate([cz], hours, rate)
    cal = float((cpos > cz.max()).mean())
    return {"epoch": ep, "seconds": round(time.time() - t0, 1), "loss": float(loss), "calib_recall": cal,
            "calib_threshold_at_rate": thr, "calib_recall_at_rate": float((cpos >= thr).mean()),
            "calib_recall_at_p999": float((cpos > np.quantile(cz, 0.999)).mean())}


def cmd_fit(a):
    """Train the window head on windows of streamed features: every positive window and five negatives per positive
    each epoch, half of them the negatives the head scored highest last epoch; keep the epoch with the highest
    calibration recall at ``--select-rate`` debounced detections per hour of the calibration stream."""
    trunk = check_trunks(a)
    if a.head == "gru":
        return fit_gru(a, trunk)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    word_paths = shard_paths(a.shards, a.word)
    if not word_paths:
        raise SystemExit(f"{a.shards} holds no shard of {a.word!r}")
    shards = load_shards(word_paths + shard_paths(a.shards, "none"))
    pos, neg = window_index(shards)
    if not len(pos):
        raise SystemExit(f"{a.shards} holds no positive window of {a.word!r}")
    pool = neg[rng.choice(len(neg), min(len(neg), a.neg_pool), replace=False)]
    n_neg = min(5 * len(pos), len(pool))
    hard = pool[rng.choice(len(pool), n_neg // 2, replace=False)]
    head = WindowHead()
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = (-1.0, 0)
    with open(out.with_suffix(".jsonl"), "w") as log:
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
            hard = hardest(pool, scores, n_neg // 2)
            cpos, cz, hours = calibration_scores(head, a.calib, a.word)
            rec = epoch_record(ep, t0, loss.detach().item(), cpos, cz, hours, a.select_rate)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            if (rec["calib_recall_at_rate"], rec["calib_recall"]) > best:
                best = (rec["calib_recall_at_rate"], rec["calib_recall"])
                torch.save({"state": head.state_dict(), "word": a.word, "epoch": ep, "seed": a.seed, "trunk": trunk,
                            "select_rate": a.select_rate, "positives": int(len(pos)), "negatives": int(len(neg))}, out)


# ------------------------------------------------------------------ stateful GRU head
WARM_FRAMES = 30 * SR // HOP
MIN_BURN_BLOCKS = 2 * SR // BLOCK


def block_labels(lab):
    """The label of the frame at the end of each block of ``lab`` ``[..., T]``: ``[..., T // FPB]``."""
    return lab[..., FPB - 1:lab.shape[-1] // FPB * FPB:FPB]


def gru_loss(head, x, y, burn, pos_weight):
    """Mean binary cross-entropy of the head over crops ``x`` ``[batch, frames, D]`` with block labels ``y``
    ``[batch, blocks]``, from a zero state. Blocks of the first ``burn`` blocks and blocks labelled -1 do not count;
    positives weigh ``pos_weight``."""
    z, _ = head(x)
    keep = (y >= 0) & (torch.arange(y.shape[1]) >= burn)
    bce = F.binary_cross_entropy_with_logits(z, y.clamp(min=0).float(), reduction="none",
                                             pos_weight=torch.tensor(float(pos_weight)))
    return (bce * keep).sum() / keep.sum().clamp(min=1)


def gru_crops(shards, streams, pos_blocks, n, burn, scored, rng, pos_frac):
    """``n`` crops of ``burn + scored`` blocks: features ``[n, blocks * FPB, D]`` and block labels ``[n, blocks]``.
    A fraction ``pos_frac`` of the crops is placed so that a randomly drawn positive block (a row ``(shard, stream,
    block)`` of ``pos_blocks``) falls in the scored part; the others start anywhere in a random stream (a row
    ``(shard, stream)`` of ``streams``)."""
    total = burn + scored
    xs, ys = [], []
    for _ in range(n):
        if len(pos_blocks) and rng.random() < pos_frac:
            k, i, b = pos_blocks[rng.integers(len(pos_blocks))]
            start = b - burn - rng.integers(scored)
        else:
            k, i = streams[rng.integers(len(streams))]
            start = rng.integers(shards[k][1].shape[1] // FPB - total + 1)
        start = int(np.clip(start, 0, shards[k][1].shape[1] // FPB - total))
        xs.append(np.asarray(shards[k][0][i, start * FPB:(start + total) * FPB], np.float32))
        ys.append(block_labels(shards[k][1][i])[start:start + total])
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys).astype(np.int64))


def gru_logits(head, feats, h=None, chunk=6000):
    """(block logits of a torch GRU head over ``feats``, the state after them): ``feats`` run in pieces of ``chunk``
    frames with the state carried, from ``h`` (zeros when None). A trailing partial block is not scored."""
    n = len(feats) // FPB * FPB
    out = []
    with torch.no_grad():
        for i in range(0, n, chunk):
            z, h = head(torch.from_numpy(np.asarray(feats[i:min(i + chunk, n)], np.float32))[None], h)
            out.append(z[0].numpy())
    return np.concatenate(out).astype(np.float64), h


def warm_feats(feats):
    """The ``WARM_FRAMES`` frames that warm a state up before a positive."""
    if len(feats) < WARM_FRAMES:
        raise SystemExit(f"{len(feats)} negative frames cannot warm a state up: {WARM_FRAMES} are needed")
    return feats[:WARM_FRAMES]


def gru_calibration_scores(head, calib, word):
    """(calibration positives' best logit, calibration stream block logits, stream hours) of a GRU head. Each
    ``negs-*.npz`` runs as one stream from a zero state. Each positive row runs from the state the head holds after the
    first ``WARM_FRAMES`` frames (30 s) of the first ``negs-*.npz`` file, as a device does that has been listening."""
    paths = sorted(glob.glob(f"{calib}/negs-*.npz"))
    warm = gru_logits(head, warm_feats(np.load(paths[0])["feats"]))[1]
    z = np.concatenate([gru_logits(head, np.load(p)["feats"])[0] for p in paths])
    pos = np.array([gru_logits(head, f, warm)[0][s:e + 1].max() for f, s, e in load_rows(f"{calib}/{word}-*.npy")])
    return pos, z, len(z) * BLOCK / SR / 3600


def fit_gru(a, trunk):
    """Train the GRU head by truncated backpropagation through time on random crops of the shard streams. A crop is
    ``--burn-in-blocks`` blocks whose loss is masked, then ``--crop-blocks`` scored blocks; the block label is the
    label of the frame at the block end and -1 blocks are masked. Positive blocks are a few of every hundred, so half
    of the crops (``--pos-crops``) are placed around a positive block, and positives weigh ``--pos-weight`` in the loss.
    The kept epoch has the highest calibration recall at ``--select-rate``, as for the window head."""
    if a.burn_in_blocks < MIN_BURN_BLOCKS:
        raise SystemExit(f"--burn-in-blocks {a.burn_in_blocks} is under the {MIN_BURN_BLOCKS} blocks (2 s) a state needs")
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    word_paths = shard_paths(a.shards, a.word)
    if not word_paths:
        raise SystemExit(f"{a.shards} holds no shard of {a.word!r}")
    shards = load_shards(word_paths + shard_paths(a.shards, "none"))
    total = a.burn_in_blocks + a.crop_blocks
    streams = [(k, i) for k, (_, lab) in enumerate(shards) if lab.shape[1] // FPB >= total for i in range(len(lab))]
    if not streams:
        raise SystemExit(f"no shard stream holds {total} blocks ({total * BLOCK / SR:.1f} s)")
    pos_blocks = [(k, i, b) for k, i in streams
                  for b in np.flatnonzero(block_labels(shards[k][1][i]) == 1) if b >= a.burn_in_blocks]
    if not pos_blocks:
        raise SystemExit(f"{a.shards} holds no positive block of {a.word!r} after {a.burn_in_blocks} burn-in blocks")
    labels = [block_labels(shards[k][1][i]) for k, i in streams]
    n_pos, n_neg = (int(sum((y == v).sum() for y in labels)) for v in (1, 0))
    head = StreamGRUHead()
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = (-1.0, 0)
    with open(out.with_suffix(".jsonl"), "w") as log:
        for ep in range(1, a.epochs + 1):
            t0 = time.time()
            head.train()
            for done in range(0, a.crops, a.batch):
                x, y = gru_crops(shards, streams, pos_blocks, min(a.batch, a.crops - done), a.burn_in_blocks, a.crop_blocks,
                                 rng, a.pos_crops)
                loss = gru_loss(head, x, y, a.burn_in_blocks, a.pos_weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step()
            head.eval()
            cpos, cz, hours = gru_calibration_scores(head, a.calib, a.word)
            rec = epoch_record(ep, t0, loss.detach().item(), cpos, cz, hours, a.select_rate)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            if (rec["calib_recall_at_rate"], rec["calib_recall"]) > best:
                best = (rec["calib_recall_at_rate"], rec["calib_recall"])
                torch.save({"state": head.state_dict(), "word": a.word, "epoch": ep, "seed": a.seed, "trunk": trunk,
                            "select_rate": a.select_rate, "positives": n_pos, "negatives": n_neg, "arch": "stream_gru"}, out)


def debounced_events(z, thr):
    above = np.flatnonzero(np.asarray(z) >= thr)
    n, j = 0, 0
    while j < len(above):
        n += 1
        j = int(np.searchsorted(above, above[j] + DEBOUNCE_BLOCKS + 1))
    return n


def threshold_for_rate(streams, hours, rate):
    """Lowest threshold whose debounced detections over ``streams`` (lists of per-block logits) stay at or below
    ``rate`` per hour. Blocks scored -inf (before a full window) never fire and do not bound the search."""
    allz = np.concatenate(streams)
    allz = allz[np.isfinite(allz)]
    if not len(allz):
        raise SystemExit("no scored block: every stream is shorter than one window")
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


def gru_session_logits(sess, feats, h=None):
    """(block logits of an exported GRU head over ``feats``, the state after them): one block per run, the state
    carried from ``h`` (zeros when None). A trailing partial block is not scored."""
    feed, state = (i.name for i in sess.get_inputs())
    h = np.zeros((1, 1, 128), np.float32) if h is None else h
    z = np.empty(len(feats) // FPB)
    for b in range(len(z)):
        out, h = sess.run(None, {feed: np.asarray(feats[b * FPB:(b + 1) * FPB], np.float32)[None], state: h})
        z[b] = out[0]
    return z, h


def _gru_piece(sess):
    """``piece_logits`` for ``_neg_logits`` with the state carried from piece to piece: the piece that has no tail
    before it opens a directory and starts from a zero state."""
    h = None

    def run(d, tail):
        nonlocal h
        z, h = gru_session_logits(sess, d["feats"], h if tail is not None else None)
        return z
    return run


def cmd_score(a):
    """Recall at ``--rate`` false activations per hour, and recall and false activations at the calibrated default,
    for shipped heads on isolated windows (``iso_<name>`` logits) and on the last 75 streamed frames. Negatives are
    each directory's pieces in order, scored continuously, one piece in memory at a time. ``--gru`` heads carry their
    state across the pieces of a directory and start each directory from zero; their positives start from the state
    after the first ``WARM_FRAMES`` frames of the first piece of the first directory."""
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
    for spec in a.gru:
        name, path = spec.split("=", 1)
        word = name.split(":")[0]
        meta = {p.key: p.value for p in onnx.load(path).metadata_props}
        default = float(meta["default_threshold"]) if "default_threshold" in meta else None
        mine = {k: v for k, v in pos.items() if k.startswith(word)}
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        z_st, hours = _neg_logits(by, _gru_piece(sess))
        res["negative_hours"] = round(hours, 2)
        warm = gru_session_logits(sess, warm_feats(np.load(next(iter(by.values()))[0])["feats"]))[1]
        pmax = {k: np.array([gru_session_logits(sess, r["feats"], warm)[0][r["first_block"]:].max() for r in rows])
                for k, rows in mine.items()}
        res[f"{name}:streamed_gru"] = _report(z_st, hours, pmax, a.rate, default)
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


def roc_auc(pos, neg):
    """Probability that a positive scores above a negative, a tie counting half."""
    neg = np.sort(np.asarray(neg, np.float64))
    pos = np.asarray(pos, np.float64)
    below = np.searchsorted(neg, pos, "left") + np.searchsorted(neg, pos, "right")
    return float(below.sum() / (2 * len(pos) * len(neg)))


OUTPUT_NAME = "logit_calibrated"
TRAINER = "wakeforge scripts/research/microwakehubert/stream_heads.py"
PLUGIN_CALIBRATION = "identity; the affine calibration (calib_a, calib_b) is folded into the head graph"


def cmd_export(a):
    """Calibrate a trained head on the calibration stream and export it as the plugin's head ONNX: input
    ``features`` [batch, frames, 128], output ``logit_calibrated``, the affine map inside. A GRU head (checkpoint
    ``arch`` ``stream_gru``) takes one block, ``features`` [1, 4, 128] and the state ``h`` [1, 1, 128], and returns
    ``logit_calibrated`` [1] and the next state ``h_out`` [1, 1, 128]; its calibration positives start from a warmed-up
    state (see ``gru_calibration_scores``). The published plugin does not read ``h`` and cannot run a GRU export. The
    stream's ``--rate``
    FA/h logit maps to probability 0.5 and the median calibration positive to 0.9; the F2-optimal threshold becomes
    ``default_threshold``. The metadata names the pretrained featurizer the trunk was made from, and the keys the
    shipped calibrated heads carry."""
    import onnx
    from ww_trainer.version import __version__

    ck = torch.load(a.head, map_location="cpu")
    trunk = ck["trunk"]
    if read_trunk_record(a.calib) != trunk:
        raise SystemExit(f"{a.calib} was featurized by another trunk than the head's shards: "
                         f"{read_trunk_record(a.calib)} against {trunk}")
    if not (trunk["featurizer_sha256"] and trunk["featurizer_revision"]):
        raise SystemExit(f"the trunk {trunk['trunk_sha256']} records no source featurizer sha256 and revision; "
                         "streamify it with `stream_trunk.py streamify --source-revision` so the plugin loads the same featurizer")
    gru = ck.get("arch") == "stream_gru"
    head = StreamGRUHead() if gru else WindowHead()
    head.load_state_dict(ck["state"])
    head.eval()
    pos, z, hours = (gru_calibration_scores if gru else calibration_scores)(head, a.calib, ck["word"])
    z_thr, extrapolated = stream_threshold(z, hours, a.rate)
    gap = float(np.median(pos)) - z_thr
    if gap <= 0:
        raise SystemExit(f"median calibration positive {np.median(pos):.3f} is below the {a.rate} FA/h logit {z_thr:.3f}")
    scale = math.log(CALIB_POS_PROB / (1 - CALIB_POS_PROB)) / gap
    shift = -scale * z_thr
    scored = z[np.isfinite(z)]
    info = {"calib_a": scale, "calib_b": shift, "target_fa_per_hour": a.rate, "calib_hours": hours,
            "calib_events": debounced_events(z, z_thr), "calib_extrapolated": extrapolated,
            "calib_positives": len(pos), "roc_auc": roc_auc(pos, scored), "positives_per_hour": a.positives_per_hour,
            **f2_operating_point(scale * pos + shift, scale * z + shift, hours, a.positives_per_hour)}
    with torch.no_grad():
        if gru:
            torch.onnx.export(AffineBlock(head, scale, shift).eval(), (torch.zeros(1, FPB, 128), torch.zeros(1, 1, 128)),
                              a.out, input_names=["features", "h"], output_names=[OUTPUT_NAME, "h_out"],
                              opset_version=17, dynamo=False)
        else:
            torch.onnx.export(Affine(head, scale, shift).eval(), torch.zeros(1, WINDOW_FRAMES, 128), a.out,
                              input_names=["features"], output_names=[OUTPUT_NAME],
                              dynamic_axes={"features": {0: "batch", 1: "frames"}, OUTPUT_NAME: {0: "batch"}},
                              opset_version=17, dynamo=False)
    m = onnx.load(a.out)
    meta = {"wake_word": ck["word"].replace("_", " "), "pretrained_featurizer": trunk["pretrained_featurizer"],
            "featurizer_sha256": trunk["featurizer_sha256"], "featurizer_revision": trunk["featurizer_revision"],
            "stream_trunk_sha256": trunk["trunk_sha256"], "featurizer_features": "streamed",
            **({"frames_per_block": str(FPB), "state_shape": "1,1,128"} if gru else {"window_frames": str(WINDOW_FRAMES)}),
            "feature_dim": "128", "arch": "stream_gru" if gru else "gru", "trainer": TRAINER,
            "wakeforge_version": __version__, "license": "Apache-2.0", "plugin_calibration": PLUGIN_CALIBRATION,
            "calibrated": "1", "default_threshold": repr(info["f2_threshold"]), "training_data": a.training_data,
            **{k: repr(v) for k, v in info.items()}}
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
    for x in (p, q):
        x.add_argument("--sources", default=str(SOURCES), help="positive sources per word (JSON)")
        x.add_argument("--data-root", default=None, help="directory the source paths are relative to")
    for x in (p, q, r, s):
        x.add_argument("--trunk", required=True, help="stateful streaming trunk ONNX (int8)")
        x.add_argument("--out", required=True)
        x.add_argument("--part", type=int, default=0)
        x.add_argument("--parts", type=int, default=1)
        x.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("fit")
    t.add_argument("--shards", required=True)
    t.add_argument("--word", required=True)
    t.add_argument("--head", choices=["window", "gru"], default="window",
                   help="window: the head on isolated 1.5 s windows; gru: the stateful head for streaming")
    t.add_argument("--calib", required=True, help="output of calib: negs-*.npz and <word>-*.npy")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--neg-pool", type=int, default=200000, help="negative windows hard negatives are mined from")
    t.add_argument("--select-rate", type=float, default=1.0,
                   help="debounced detections per hour of calibration stream at which the kept epoch has the best recall")
    t.add_argument("--burn-in-blocks", type=int, default=40, help="gru: blocks of each crop run before the loss counts")
    t.add_argument("--crop-blocks", type=int, default=120, help="gru: scored blocks of each crop")
    t.add_argument("--crops", type=int, default=2000, help="gru: crops per epoch")
    t.add_argument("--pos-crops", type=float, default=0.5, help="gru: fraction of the crops placed around a positive block")
    t.add_argument("--pos-weight", type=float, default=10.0, help="gru: weight of a positive block in the loss")
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
    v.add_argument("--positives-per-hour", type=float, default=4.0,
                   help="real wake words per hour of audio, which weights recall against false activations in the F2 choice")
    v.add_argument("--training-data", required=True, help="the training data statement written into the metadata")
    v.add_argument("--out", required=True)
    u.add_argument("--gru", nargs="*", default=[], help="name=head.onnx, an exported GRU head scored on streamed frames only")
    u.add_argument("--window", nargs="*", default=[], help="name=head.onnx, a window head scored on streamed frames only")
    a = ap.parse_args(argv)
    {"streams": cmd_streams, "calib": cmd_calib, "negs": cmd_negs, "pos": cmd_pos, "fit": cmd_fit, "export": cmd_export,
     "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
