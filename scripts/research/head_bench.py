"""Wake-word head bench on frozen featurizers: features computed once, many heads trained at once.

    head_bench.py cache <featurizer.onnx> <cache_dir> [--metadata kws/train.csv --val kws/val.csv]
    bench.py fit <cache_dir> <out_dir> [--heads gru,gru-h256,...] [--epochs 20]
    bench.py export <out_dir> <head_name>        (ONNX head that kws_eval.py can score)

``cache`` runs the featurizer once over every 1.5 s training and validation window (the same windows
ww_trainer-train reads) and stores float16 features; every training positive is also stored as
--pos-aug augmented copies and a quarter of the negatives once (``augmenter``), since a head trained on
clean synthetic clips alone does not survive noise. ``fit`` trains every listed head on the same batches:
each epoch uses every positive and five times as many negatives, half the highest-scoring negatives
of the previous epoch (hard negatives) and half drawn at random; BCE; the kept checkpoint is the one with
the highest calibration recall. Heads share the batch order, so they differ only in architecture.

Checkpoints are chosen on a calibration set that no evaluation touches, because the synthetic validation
set is separable within a few epochs and its loss then keeps falling as the head over-fits: the
validation positives mixed with LibriSpeech dev-other babble (three talkers) at 10 and 5 dB and with
AudioSet noise at 5 dB, against one hour of dev-other speech and dev-other babble cut into windows at a
0.5 s hop. Calibration recall is the share of those positives above the highest-scoring negative window.
"""
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

SR, N = 16000, 24000


PATH_MAP = []  # (old prefix, new prefix) pairs from --path-map, for metadata written on another machine
CALIB_SPEECH = "data/LibriSpeech/dev-other"


def read_meta(path):
    rows = [r for r in csv.reader(open(path)) if r]
    files = []
    for r in rows:
        f = r[0]
        for old, new in PATH_MAP:
            if f.startswith(old):
                f = new + f[len(old):]
        files.append(f)
    return files, np.array([int(r[1]) for r in rows], np.int8)


def load_window(p):
    import soundfile as sf
    w, sr = sf.read(p, dtype="float32", always_2d=True)
    w = w.mean(1)
    assert sr == SR, (p, sr)
    out = np.zeros(N, np.float32); out[:min(N, len(w))] = w[:N]
    return out


def calibration_windows(val_files, val_labels, seconds=3600, seed=0):
    """(noisy positives [P, N], negative windows [M, N]) from sources the evaluation never uses."""
    import glob, soundfile as sf
    rng = random.Random(seed)
    other = sorted(glob.glob(f"{CALIB_SPEECH}/**/*.flac", recursive=True))
    noise = sorted(glob.glob("data/noise-audioset/*.wav"))

    def clip(p, n):
        w, sr = sf.read(p, dtype="float32", always_2d=True); w = w.mean(1)
        if len(w) < n:
            w = np.tile(w, n // len(w) + 1)
        s = rng.randint(0, len(w) - n); return w[s:s + n]

    def babble(n):
        return sum(clip(rng.choice(other), n) for _ in range(3)) / 3

    def mix(x, v, snr):
        pc, pn = np.mean(x ** 2), max(np.mean(v ** 2), 1e-10)
        y = x + v * np.sqrt(pc / (pn * 10 ** (snr / 10)))
        return (y / max(1.0, np.abs(y).max())).astype(np.float32)

    pos = [load_window(f) for f, l in zip(val_files, val_labels) if l == 1]
    noisy = [mix(x, babble(N), 10) for x in pos] + [mix(x, babble(N), 5) for x in pos] + \
            [mix(x, clip(rng.choice(noise), N), 5) for x in pos]
    neg, total = [], 0
    for p in rng.sample(other, len(other)):
        w, _ = sf.read(p, dtype="float32")
        neg += [w[i:i + N] for i in range(0, len(w) - N + 1, SR // 2)]; total += len(w)
        if total >= seconds * SR / 2:
            break
    for _ in range(int(seconds / 2 / 5)):
        b = babble(5 * SR); neg += [b[i:i + N] for i in range(0, len(b) - N + 1, SR // 2)]
    return np.stack(noisy), np.stack(neg).astype(np.float32)


TALK = "data/LibriSpeech/train-clean-100"
DEVICE = bool(int(__import__("os").environ.get("WF_DEVICE_AUG", "0")))


def augmenter(seed=0, pool=400):
    """Head-time augmentation of a 1.5 s window from training-only pools (never the evaluation's): gain,
    speed, room reverberation, and AudioSet noise or LibriSpeech train-clean-100 babble at 0-20 dB. The
    pools are ``pool`` crops read once into memory."""
    import glob, soundfile as sf
    rng = random.Random(seed)

    def crops(files, n, k):
        out = []
        for p in rng.sample(files, min(k, len(files))):
            try:
                w, sr = sf.read(p, dtype="float32", always_2d=True)
            except Exception:
                continue
            w = w.mean(1)
            if len(w) < n:
                w = np.tile(w, n // len(w) + 1)
            s = rng.randint(0, len(w) - n); out.append(w[s:s + n])
        return out

    noise = crops(sorted(glob.glob("data/noise-audioset/*.wav")), N, pool)
    talk = crops(sorted(glob.glob(f"{TALK}/**/*.flac", recursive=True)), N, pool)
    rirs = [r / (np.abs(r).max() + 1e-9) for r in crops(sorted(glob.glob("data/rir/**/*.wav", recursive=True)), 8000, 100)]

    def aug(x):
        y = x
        if rng.random() < 0.5:  # move the word within the window: real audio has it anywhere
            k = rng.randint(-SR * 2 // 5, SR * 2 // 5)
            y = np.roll(y, k)
            if k > 0: y[:k] = 0
            elif k < 0: y[k:] = 0
        if rng.random() < 0.5:  # speed by linear interpolation, kept at N samples
            f = rng.uniform(0.9, 1.1)
            y = np.interp(np.arange(N) * f, np.arange(N), y, right=0.0).astype(np.float32)
        if rirs and rng.random() < (0.6 if DEVICE else 0.3):
            h = rng.choice(rirs); L = N + len(h) - 1
            r = np.fft.irfft(np.fft.rfft(y, L) * np.fft.rfft(h, L), L)[:N]
            y = r * np.sqrt(np.mean(y ** 2) / (np.mean(r ** 2) + 1e-12))
        if rng.random() < 0.8:
            v = sum(rng.choice(talk) for _ in range(rng.randint(1, 3))) if rng.random() < 0.4 else rng.choice(noise)
            snr = rng.uniform(5 if rng.random() < 0.4 else 0, 20)
            pc, pn = np.mean(y ** 2), max(np.mean(v ** 2), 1e-10)
            y = y + v * np.sqrt(pc / (pn * 10 ** (snr / 10)))
        if DEVICE:
            return device(y)
        y = y * 10 ** (rng.uniform(-6, 6) / 20)
        return (y / max(1.0, np.abs(y).max())).astype(np.float32)

    def device(y):
        # what a real device adds: speech after the word, the microphone's band and colour, level-dependent
        # gain and compression, clipping when the input is too hot, and the microphone's own noise floor
        if talk and rng.random() < 0.3:
            v = rng.choice(talk); cut = rng.randint(N // 2, N - 1)
            y = y.copy(); y[cut:] += v[: N - cut] * np.sqrt(np.mean(y ** 2) / (np.mean(v ** 2) + 1e-10)) * rng.uniform(0.3, 1.0)
        F = np.fft.rfft(y); f = np.fft.rfftfreq(N, 1 / 16000)
        if rng.random() < 0.6:
            lo, hi = rng.uniform(60, 300), rng.uniform(3000, 7800)
            F = F / (1 + (lo / np.maximum(f, 1)) ** 4) / (1 + (f / hi) ** 8)
        if rng.random() < 0.5:
            bands = 10 ** (np.array([rng.uniform(-6, 6) for _ in range(8)]) / 20)
            F = F * np.interp(f, np.linspace(0, 8000, 8), bands)
        y = np.fft.irfft(F, N).astype(np.float32)
        y = y / (np.abs(y).max() + 1e-9) * 10 ** (rng.uniform(-30, 0) / 20)
        if rng.random() < 0.3:
            k = rng.uniform(0.4, 0.9); y = np.sign(y) * np.abs(y) ** k
        if rng.random() < 0.3:
            y = y * 10 ** (rng.uniform(6, 24) / 20)
            y = np.tanh(y) if rng.random() < 0.5 else np.clip(y, -1, 1)
        if rng.random() < 0.5:
            n = np.random.default_rng(rng.randint(0, 2 ** 31)).standard_normal(N).astype(np.float32)
            if rng.random() < 0.5:
                n = np.cumsum(n); n -= np.convolve(n, np.ones(64) / 64, "same")
            y = y + n / (np.std(n) + 1e-9) * np.sqrt(np.mean(y ** 2)) * 10 ** (-rng.uniform(25, 50) / 20)
        return np.clip(y, -1, 1).astype(np.float32)
    return aug


def cache(a):
    import onnxruntime as ort
    out = Path(a.cache_dir); out.mkdir(parents=True, exist_ok=True)
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sess = ort.InferenceSession(a.featurizer, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    vf, vl = read_meta(a.val)
    cpos, cneg = calibration_windows(vf, vl)
    splits = [("train", *read_meta(a.metadata), None), ("val", vf, vl, None),
              ("calib", [None] * (len(cpos) + len(cneg)), np.r_[np.ones(len(cpos)), np.zeros(len(cneg))].astype(np.int8),
               np.concatenate([cpos, cneg]))]
    if a.pos_aug:
        # every training positive also as --pos-aug augmented copies, and a quarter of the negatives once
        aug = augmenter()
        tf, tl = read_meta(a.metadata)
        extra = [(f, 1) for f, l in zip(tf, tl) if l == 1 for _ in range(a.pos_aug)]
        extra += [(f, 0) for f, l in zip(tf, tl) if l == 0][::4]
        splits[0] = ("train", list(tf) + [("aug", f) for f, _ in extra], np.r_[tl, [l for _, l in extra]].astype(np.int8), None)
    for split, files, labels, audio in splits:
        feats = None
        for i in range(0, len(files), 64):
            wins = audio[i:i + 64] if audio is not None else np.stack(
                [aug(load_window(f[1])) if isinstance(f, tuple) else load_window(f) for f in files[i:i + 64]])
            f = sess.run(None, {name: wins})[0]
            if f.shape[0] != len(wins):
                f = np.concatenate([sess.run(None, {name: w[None]})[0] for w in wins])
            if feats is None:
                feats = np.lib.format.open_memmap(out / f"{split}.npy", "w+", np.float16, (len(files), *f.shape[1:]))
            feats[i:i + len(wins)] = f.astype(np.float16)
        feats.flush(); np.save(out / f"{split}_labels.npy", labels)
        print(split, feats.shape, int(labels.sum()), "positives", flush=True)
    (out / "meta.json").write_text(json.dumps({"featurizer": a.featurizer, "metadata": a.metadata, "val": a.val}) + "\n")



def clip_subset(labels, copies, n, seed):
    """Row indices of ``n`` training positives chosen by clip, each with its ``copies`` augmented rows.

    ``cache`` writes the metadata rows first, positives leading, then every positive's augmented copies as one
    contiguous block in the same order, then a quarter of the negatives. A subset taken by row would keep an
    augmented copy of a clip whose original was dropped, which is a different clip count than the one reported.
    """
    labels = np.asarray(labels)
    n_orig = int(np.argmin(labels == 1)) if labels[0] == 1 else 0
    start = n_orig + int(np.argmax(labels[n_orig:] == 1))
    if labels.sum() != n_orig * (1 + copies) or not labels[start:start + n_orig * copies].all():
        raise SystemExit(f"cache layout does not match {copies} copies of {n_orig} leading positives; "
                         "pass the --pos-aug the cache was built with")
    if n > n_orig:
        raise SystemExit(f"--max-pos {n} exceeds the {n_orig} positive clips in the cache")
    keep = np.random.default_rng(seed).permutation(n_orig)[:n]
    rows = np.concatenate([keep] + [start + keep * copies + t for t in range(copies)])
    return np.sort(rows)


class GRUHead(nn.Module):
    """ww_trainer's GruClassifierHead: GRU, mean over time, ReLU linear, one logit."""

    def __init__(self, dim, hidden=128, linear=128, layers=1, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(dim, hidden, num_layers=layers, batch_first=True, bidirectional=bidirectional)
        self.fc1 = nn.Linear(hidden * (2 if bidirectional else 1), linear)
        self.fc2 = nn.Linear(linear, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc2(F.relu(self.fc1(out.mean(1)))).squeeze(-1)


def make_head(spec, dim):
    """gru | gru-h256 | gru-l2 | bigru, joined by '-' options."""
    kw = {}
    for part in spec.split("-")[1:]:
        if part.startswith("h"): kw["hidden"] = int(part[1:])
        elif part.startswith("l"): kw["layers"] = int(part[1:])
    return GRUHead(dim, bidirectional=spec.startswith("bigru"), **kw)


def fit(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = Path(a.cache_dir)
    X = torch.from_numpy(np.load(c / "train.npy", mmap_mode="r")[:]).to(dev)
    y = torch.from_numpy(np.load(c / "train_labels.npy").astype(np.float32)).to(dev)
    Xv = torch.from_numpy(np.load(c / "val.npy")[:]).to(dev)
    yv = torch.from_numpy(np.load(c / "val_labels.npy").astype(np.float32)).to(dev)
    Xc = torch.from_numpy(np.load(c / "calib.npy")[:]).to(dev)
    yc = torch.from_numpy(np.load(c / "calib_labels.npy").astype(np.float32)).to(dev)
    pos, neg = torch.nonzero(y == 1).squeeze(1), torch.nonzero(y == 0).squeeze(1)
    if a.max_pos:
        pos = torch.from_numpy(clip_subset(y.cpu().numpy(), a.pos_aug, a.max_pos, a.subset_seed)).to(dev)
        print(f"training on {a.max_pos} positive clips, {len(pos)} rows with their augmented copies", flush=True)
    g = torch.Generator(device="cpu").manual_seed(a.seed); torch.manual_seed(a.seed)
    specs = a.heads.split(",")
    heads = {s: make_head(s, X.shape[-1]).to(dev) for s in specs}
    opts = {s: torch.optim.Adam(h.parameters(), lr=a.lr) for s, h in heads.items()}
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    best = {s: (-1.0, 0.0) for s in specs}
    # five negatives per positive, half of them hard; capped by the negatives there are
    n_neg = min(5 * len(pos), len(neg)); n_hard = n_neg // 2
    hard = neg[torch.randperm(len(neg), generator=g)[:n_hard].to(dev)]
    log = open(out / "fit.jsonl", "a")
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        rand = neg[torch.randperm(len(neg), generator=g)[: n_neg - len(hard)].to(dev)]
        idx = torch.cat([pos, hard, rand]); idx = idx[torch.randperm(len(idx), generator=g).to(dev)]
        for h in heads.values(): h.train()
        for i in range(0, len(idx), a.batch_size):
            b = idx[i:i + a.batch_size]
            xb, yb = X[b].float(), y[b]
            for s, h in heads.items():
                loss = F.binary_cross_entropy_with_logits(h(xb), yb)
                opts[s].zero_grad(set_to_none=True); loss.backward(); opts[s].step()
        rec = {"epoch": ep, "seconds": round(time.time() - t0, 1)}
        with torch.no_grad():
            for h in heads.values(): h.eval()
            # hard negatives for the next epoch: the negatives the first head scores highest
            first = heads[specs[0]]
            sc = torch.cat([first(X[neg[i:i + 1024]].float()) for i in range(0, len(neg), 1024)])
            hard = neg[torch.topk(sc, n_hard).indices]
            for s, h in heads.items():
                lv = torch.cat([h(Xv[i:i + 1024].float()) for i in range(0, len(Xv), 1024)])
                vl = F.binary_cross_entropy_with_logits(lv, yv).item()
                p = torch.sigmoid(lv)
                lc = torch.cat([h(Xc[i:i + 1024].float()) for i in range(0, len(Xc), 1024)])
                cal = (lc[yc == 1] > lc[yc == 0].max()).float().mean().item()
                rec[s] = {"val_loss": round(vl, 4), "val_recall@0.5": round(((p >= 0.5) & (yv == 1)).sum().item() / max(1, (yv == 1).sum().item()), 4),
                          "val_fp@0.5": int(((p >= 0.5) & (yv == 0)).sum().item()), "calib_recall": round(cal, 4)}
                if (cal, -vl) > best[s]:
                    best[s] = (cal, -vl); torch.save({"spec": s, "dim": X.shape[-1], "state": h.state_dict(), "epoch": ep, "calib_recall": cal}, out / f"{s}.pt")
        log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)


def export(a):
    ck = torch.load(Path(a.out_dir) / f"{a.head}.pt", map_location="cpu")
    h = make_head(ck["spec"], ck["dim"]); h.load_state_dict(ck["state"]); h.eval()
    path = Path(a.out_dir) / f"{a.head}.onnx"
    torch.onnx.export(h, torch.randn(1, 75, ck["dim"]), str(path), input_names=["features"], output_names=["logit"],
                      dynamic_axes={"features": {0: "batch", 1: "frames"}, "logit": {0: "batch"}}, opset_version=17, dynamo=False)
    print(path)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cache"); p.add_argument("featurizer"); p.add_argument("cache_dir")
    p.add_argument("--metadata", default="kws/train.csv"); p.add_argument("--val", default="kws/val.csv")
    p.add_argument("--pos-aug", type=int, default=8, help="augmented copies of every training positive (0: none)")
    p.add_argument("--path-map", action="append", default=[], help="old=new prefix rewrite for metadata paths")
    p.add_argument("--aug-talk", default="data/LibriSpeech/train-clean-100", help="speech whose talkers become head-time babble")
    p.add_argument("--calib-speech", default="data/LibriSpeech/dev-other", help="LibriSpeech split used for calibration speech and babble")
    p = sub.add_parser("fit"); p.add_argument("cache_dir"); p.add_argument("out_dir")
    p.add_argument("--heads", default="gru"); p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--lr", type=float, default=5e-4); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-pos", type=int, default=0, help="train on this many positive clips, chosen by clip (0: all)")
    p.add_argument("--pos-aug", type=int, default=0, help="augmented copies per positive the cache was built with (needed by --max-pos)")
    p.add_argument("--subset-seed", type=int, default=0, help="which clips --max-pos keeps")
    p = sub.add_parser("export"); p.add_argument("out_dir"); p.add_argument("head")
    a = ap.parse_args()
    global CALIB_SPEECH, TALK
    if a.cmd == "cache":
        PATH_MAP.extend(tuple(m.split("=", 1)) for m in a.path_map)
        CALIB_SPEECH, TALK = a.calib_speech, a.aug_talk
    {"cache": cache, "fit": fit, "export": export}[a.cmd](a)


if __name__ == "__main__":
    main()
