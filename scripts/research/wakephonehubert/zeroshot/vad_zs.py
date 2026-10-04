"""Voice activity on LibriParty with vad_eval.py's reference and metric (speech-frame precision, recall and F1 pooled
over the 50 eval sessions on a 10 ms grid; thresholds at 0.5 and tuned on the 50 dev sessions over 0.05..0.95).

  extract   ONNX `features` (per 20 ms frame; the first 521 values, hubert, vad and ipa, are used) for every train, dev and eval session -> feats/vad/<split>.npy (fp16)
            with per-frame reference labels and session offsets
  zeroshot  the vad output used directly, raw and with a 5-frame median (trailing, i.e. causal, and centred), for both
            the PyTorch heads (bench/results/vad/raw-wakephonehubert.npz) and the shipped ONNX
  train     small causal heads on the frozen features, trained on the train sessions, model selection and threshold on
            dev, tested on eval: a per-frame linear layer and a two-layer causal convolution (kernel 5 -> 64 -> kernel 3),
            seeds 0 and 1
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import zslib  # noqa: E402

GRID = 160
LP_DEV_EVAL = zslib.BENCH / "data/libriparty/LibriParty/dataset"
LP_TRAIN = zslib.ZS / "data/libriparty/LibriParty/dataset"
FD = zslib.ZS / "feats/vad"
OD = zslib.ZS / "results/vad"
D = 521
THS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def session_dirs(split):
    root = LP_TRAIN if split == "train" else LP_DEV_EVAL
    return sorted((root / split).iterdir(), key=lambda p: int(p.name.split("_")[1]))


def reference(d, n):
    meta = json.loads((d / f"{d.name}.json").read_text())
    ref = np.zeros(n, bool)
    for k, segs in meta.items():
        if k.isdigit():
            for s in segs:
                ref[int(round(s["start"] * 100)):int(round(s["stop"] * 100))] = True
    return ref


def on_grid(scores, hop, n):
    idx = np.minimum((np.arange(n) * GRID) // hop, len(scores) - 1)
    return scores[idx]


def prf(ref, hyp):
    tp = int((ref & hyp).sum()); fp = int((~ref & hyp).sum()); fn = int((ref & ~hyp).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4)}


def metrics(dev, ev):
    """dev, ev: (reference on the 10 ms grid, score on the 10 ms grid)."""
    f1s = [prf(dev[0], dev[1] >= t)["f1"] for t in THS]
    t = float(THS[int(np.argmax(f1s))])
    return {"eval_at_0.5": prf(ev[0], ev[1] >= 0.5), "dev_tuned_threshold": t, "eval_at_dev_threshold": prf(ev[0], ev[1] >= t),
            "dev_f1_at_threshold": max(f1s)}


def _one(d):
    s = zslib.session("wakephonehubert_features_int8.onnx", threads=1)
    wav, sr = sf.read(d / f"{d.name}_mixture.wav", dtype="float32")
    assert sr == 16000
    f = zslib.run(s, wav, ("features",))["features"]
    ref = reference(d, len(wav) // GRID)
    lab = ref[::2][:len(f)]
    return d.name, f.astype(np.float16), np.pad(lab, (0, len(f) - len(lab))), ref


def extract():
    FD.mkdir(parents=True, exist_ok=True)
    for split in ("dev", "eval", "train"):
        if (FD / f"{split}.npz").exists():
            continue
        t0 = time.time()
        with Pool(8) as p:
            R = p.map(_one, session_dirs(split), chunksize=1)
        np.save(FD / f"{split}.npy", np.concatenate([r[1] for r in R]))
        np.savez_compressed(FD / f"{split}.npz", sessions=np.asarray([r[0] for r in R]), labels_20ms=np.concatenate([r[2] for r in R]),
                            offsets_20ms=np.cumsum([0] + [len(r[1]) for r in R]).astype(np.int64),
                            ref_10ms=np.concatenate([r[3] for r in R]), offsets_10ms=np.cumsum([0] + [len(r[3]) for r in R]).astype(np.int64))
        print(json.dumps({"split": split, "sessions": len(R), "frames": int(sum(len(r[1]) for r in R)),
                          "hours": round(sum(len(r[1]) for r in R) / 50 / 3600, 2), "seconds": round(time.time() - t0)}), flush=True)


def median5(x, causal):
    p = np.pad(x, (4, 0) if causal else (2, 2), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(p, 5), 1)


def grid_scores(native, off_native, hop, off_grid):
    return np.concatenate([on_grid(native[off_native[i]:off_native[i + 1]], hop, off_grid[i + 1] - off_grid[i]) for i in range(len(off_grid) - 1)])


def zeroshot():
    OD.mkdir(parents=True, exist_ok=True)
    res, raw = {}, {}
    pt = np.load(zslib.BENCH / "results/vad/raw-wakephonehubert.npz")
    M = {s: np.load(FD / f"{s}.npz") for s in ("dev", "eval")}
    for src in ("pytorch", "onnx"):
        for smooth in ("raw", "median5_causal", "median5_centred"):
            g = {}
            for s in ("dev", "eval"):
                if src == "pytorch":
                    nat, noff, ref, goff = pt[f"{s}_native_score"], pt[f"{s}_native_offsets"], pt[f"{s}_labels"], pt[f"{s}_offsets"]
                    assert (pt[f"{s}_sessions"] == M[s]["sessions"]).all()
                else:
                    nat = np.load(FD / f"{s}.npy", mmap_mode="r")[:, 128].astype(np.float32)
                    noff, ref, goff = M[s]["offsets_20ms"], M[s]["ref_10ms"], M[s]["offsets_10ms"]
                if smooth != "raw":
                    nat = np.concatenate([median5(nat[noff[i]:noff[i + 1]], smooth == "median5_causal") for i in range(len(noff) - 1)])
                g[s] = (ref, grid_scores(nat, noff, 320, goff))
                raw[f"{src}_{smooth}_{s}_score_10ms"] = g[s][1].astype(np.float32)
                raw[f"{s}_ref_10ms"] = ref
                raw[f"{s}_offsets_10ms"] = goff
            res[f"{src}_{smooth}"] = metrics(g["dev"], g["eval"])
            print(json.dumps({f"{src}_{smooth}": res[f"{src}_{smooth}"]}), flush=True)
    np.savez_compressed(OD / "zeroshot-raw.npz", **raw)
    (OD / "zeroshot.json").write_text(json.dumps(res, indent=1))


def train():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(3.0 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    dev = "cuda"
    X = {s: np.load(FD / f"{s}.npy", mmap_mode="r") for s in ("train", "dev", "eval")}
    M = {s: np.load(FD / f"{s}.npz") for s in ("train", "dev", "eval")}
    rng0 = np.random.default_rng(0)
    samp = np.sort(rng0.choice(len(X["train"]), 200000, replace=False))
    xs = X["train"][samp, :D].astype(np.float32)
    mu, sd = torch.tensor(xs.mean(0), device=dev), torch.tensor(xs.std(0) + 1e-5, device=dev)
    Xtr = np.ascontiguousarray(X["train"][:, :D])
    Ytr = M["train"]["labels_20ms"].astype(np.float32)
    off = M["train"]["offsets_20ms"]
    L = 500

    class Linear(nn.Module):
        def __init__(self):
            super().__init__(); self.o = nn.Conv1d(D, 1, 1)

        def forward(self, x):
            return self.o(x)[:, 0]

    class Conv2(nn.Module):
        def __init__(self, h=64):
            super().__init__(); self.c1 = nn.Conv1d(D, h, 5); self.c2 = nn.Conv1d(h, 1, 3)

        def forward(self, x):
            return self.c2(F.pad(F.relu(self.c1(F.pad(x, (4, 0)))), (2, 0)))[:, 0]

    def scores(model, s):
        out = []
        with torch.no_grad():
            o = M[s]["offsets_20ms"]
            for i in range(len(o) - 1):
                x = (torch.as_tensor(np.asarray(X[s][o[i]:o[i + 1], :D]), device=dev).float() - mu) / sd
                out.append(torch.sigmoid(model(x.T[None]))[0].cpu().numpy())
        return np.concatenate(out)

    def grid(s, sc):
        return M[s]["ref_10ms"], grid_scores(sc, M[s]["offsets_20ms"], 320, M[s]["offsets_10ms"])

    res, raw = {}, {}
    starts_ok = np.concatenate([np.arange(off[i], off[i + 1] - L) for i in range(len(off) - 1)])
    for name, cls in (("linear", Linear), ("conv2", Conv2)):
        for seed in (0, 1):
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            model = cls().to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            best = (-1, None, -1)
            t0 = time.time()
            for step in range(1, 4001):
                idx = rng.choice(starts_ok, 32)[:, None] + np.arange(L)[None]
                x = ((torch.as_tensor(Xtr[idx], device=dev).float() - mu) / sd).transpose(1, 2)
                loss = F.binary_cross_entropy_with_logits(model(x), torch.as_tensor(Ytr[idx], device=dev))
                opt.zero_grad(); loss.backward(); opt.step()
                if step % 500 == 0:
                    model.eval()
                    d = grid("dev", scores(model, "dev"))
                    f1 = max(prf(d[0], d[1] >= t)["f1"] for t in THS)
                    model.train()
                    if f1 > best[0]:
                        best = (f1, {k: v.clone() for k, v in model.state_dict().items()}, step)
                    print(json.dumps({"head": name, "seed": seed, "step": step, "loss": round(float(loss), 4), "dev_f1_best_threshold": f1}), flush=True)
            model.load_state_dict(best[1]); model.eval()
            sd_ = {s: scores(model, s) for s in ("dev", "eval")}
            r = metrics(grid("dev", sd_["dev"]), grid("eval", sd_["eval"]))
            r |= {"best_step": best[2], "params": sum(p.numel() for p in model.parameters()), "train_seconds": round(time.time() - t0)}
            res[f"{name}_seed{seed}"] = r
            for s in ("dev", "eval"):
                raw[f"{name}_seed{seed}_{s}_score_20ms"] = sd_[s].astype(np.float32)
            torch.save({"state": model.state_dict(), "mu": mu.cpu(), "sd": sd.cpu(), "kind": name}, OD / f"head-{name}-seed{seed}.pt")
            print(json.dumps({f"{name}_seed{seed}": r}), flush=True)
    for s in ("dev", "eval"):
        raw[f"{s}_offsets_20ms"] = M[s]["offsets_20ms"]; raw[f"{s}_sessions"] = M[s]["sessions"]
    np.savez_compressed(OD / "trained-raw.npz", **raw)
    (OD / "trained.json").write_text(json.dumps(res | {"train": {"sessions": len(off) - 1, "frames": int(off[-1])}}, indent=1))


def main():
    global LP_DEV_EVAL, LP_TRAIN, FD, OD
    ap = argparse.ArgumentParser()
    zslib.add_args(ap)
    ap.add_argument("--libriparty-train", default="", help="LibriParty dataset root holding the train sessions (default <zeroshot-dir>/data/libriparty/LibriParty/dataset)")
    ap.add_argument("cmd", choices=["extract", "zeroshot", "train"])
    a = ap.parse_args()
    zslib.configure(a)
    LP_DEV_EVAL = zslib.BENCH / "data/libriparty/LibriParty/dataset"
    LP_TRAIN = Path(a.libriparty_train) if a.libriparty_train else zslib.ZS / "data/libriparty/LibriParty/dataset"
    FD = zslib.ZS / "feats/vad"
    OD = zslib.ZS / "results/vad"
    {"extract": extract, "zeroshot": zeroshot, "train": train}[a.cmd]()


if __name__ == "__main__":
    main()
