"""train_pooled.py <task> <feat> [--seeds 0,1] [--bench-dir DIR] [--heads RUN_DIR]

Learned weighted sum + linear classifier on mean-pooled entries (SUPERB SID-style head) for ESC-50, language ID and speaker ID.

Same head and budget for every featurizer: per-entry standardisation (training-split statistics), learned projection for
entries whose width differs from the featurizer's, softmax-weighted sum, linear layer to the classes. Adam lr 1e-3,
batch 256, at most 50 epochs, early stop on dev accuracy with patience 5; the best-dev state is tested.
ESC-50: five folds; fold k is tested with fold k+1 (mod 5) as dev and the other three as training; mean over folds.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import featlib  # noqa: E402
from featlib import WeightedSum  # noqa: E402

B = Path("work/bench")


def load(task, feat):
    d = B / "feats" / task / feat
    info = json.loads((d / "info.json").read_text())
    done = np.load(d / "done.npy")
    X = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in info["dims"]}
    rows = json.loads((B / "feats" / task / "meta.json").read_text())["rows"]
    return info, X, rows, done


def run(info, X, y, tr, dv, te, seed, a, C, ids=None, npz=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda"
    big = sum(v.nbytes for v in X.values()) > 2 * 2**30
    yg = torch.as_tensor(y).to(dev)
    if big:
        def get(b):
            b = np.sort(b.cpu().numpy())
            return {k: torch.from_numpy(np.ascontiguousarray(v[b])).to(dev).float() for k, v in X.items()}, torch.as_tensor(b, device=dev)
    else:
        Xg = {k: torch.as_tensor(np.asarray(v)).to(dev) for k, v in X.items()}

        def get(b):
            return {k: v[b].float() for k, v in Xg.items()}, b
    samp = np.sort(rng.choice(tr, min(len(tr), 20000), replace=False))
    stats = {}
    for k, v in X.items():
        x = torch.from_numpy(np.ascontiguousarray(v[samp])).float()
        stats[k] = (x.mean(0), x.std(0))
    ws = WeightedSum(info["dims"], info["width"], stats).to(dev)
    lin = torch.nn.Linear(info["width"], C).to(dev)
    opt = torch.optim.Adam(list(ws.parameters()) + list(lin.parameters()), lr=a.lr)

    def acc(idx):
        ws.eval(); lin.eval()
        hits = 0
        with torch.no_grad():
            for s in range(0, len(idx), 4096):
                e, b = get(torch.as_tensor(idx[s:s + 4096], device=dev))
                hits += int((lin(ws(e)).argmax(-1) == yg[b]).sum())
        ws.train(); lin.train()
        return hits / len(idx)

    best, hist = (-1.0, -1, None), []
    for ep in range(a.epochs):
        perm = rng.permutation(tr)
        tot = 0.0
        for s in range(0, len(perm), a.batch):
            e, b = get(torch.as_tensor(perm[s:s + a.batch], device=dev))
            loss = F.cross_entropy(lin(ws(e)), yg[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        da = acc(dv)
        hist.append({"epoch": ep, "train_loss": round(tot / len(perm), 4), "dev_acc": round(da, 4)})
        if da > best[0]:
            best = (da, ep, ({k: v.clone() for k, v in ws.state_dict().items()}, {k: v.clone() for k, v in lin.state_dict().items()}))
        elif ep - best[1] >= a.patience:
            break
    ws.load_state_dict(best[2][0]); lin.load_state_dict(best[2][1])
    if npz is not None:
        ws.eval(); lin.eval()
        lg, order = [], []
        with torch.no_grad():
            for s in range(0, len(te), 4096):
                e, b = get(torch.as_tensor(te[s:s + 4096], device=dev))
                lg.append(lin(ws(e)).float().cpu().numpy()); order.append(b.cpu().numpy())
        order = np.concatenate(order)
        np.savez_compressed(npz, logits=np.concatenate(lg), labels=y[order], ids=np.asarray(ids)[order].astype(str), test_index=order,
                            layer_names=np.asarray(ws.names), layer_weights=torch.softmax(ws.w.detach(), 0).cpu().numpy(),
                            layer_logits=ws.w.detach().cpu().numpy(), lin_weight=lin.weight.detach().cpu().numpy(),
                            lin_bias=lin.bias.detach().cpu().numpy())
    return {"test_acc": acc(te), "best_dev_acc": best[0], "best_epoch": best[1], "epochs_run": len(hist), "weights": ws.weights(), "history": hist}


def main():
    global B
    ap = argparse.ArgumentParser()
    featlib.add_args(ap)
    ap.add_argument("task", choices=["esc50", "lid", "sid"]); ap.add_argument("feat")
    ap.add_argument("--seeds", default="0,1"); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--json-name", default="results.json")
    a = ap.parse_args()
    B = featlib.configure(a)
    torch.set_num_threads(4)
    info, X, rows, done = load(a.task, a.feat)
    assert done.all(), f"{(~done).sum()} rows not extracted"
    y = np.array([r["label"] for r in rows])
    C = int(y.max()) + 1
    t0 = time.time()
    out = B / "results" / a.task / a.feat
    out.mkdir(parents=True, exist_ok=True)
    ids = [r["key"] for r in rows]
    res = {"task": a.task, "featurizer": a.feat, "metric": "accuracy", "n_classes": C, "heads_from": info.get("heads_from"),
           "budget": {"optimizer": "Adam", "lr": a.lr, "batch": a.batch, "epochs_max": a.epochs, "patience": a.patience}, "seeds": {}}
    for seed in [int(s) for s in a.seeds.split(",")]:
        if a.task == "esc50":
            fold = np.array([r["fold"] for r in rows])
            folds = []
            for k in range(1, 6):
                dk = k % 5 + 1
                r = run(info, X, y, np.where((fold != k) & (fold != dk))[0], np.where(fold == dk)[0], np.where(fold == k)[0], seed, a, C, ids, out / f"preds-seed{seed}-fold{k}.npz")
                folds.append(r | {"test_fold": k, "dev_fold": dk})
                print(json.dumps({"seed": seed, "fold": k, "test_acc": r["test_acc"], "weights": r["weights"]}), flush=True)
            res["seeds"][str(seed)] = {"test_acc": float(np.mean([f["test_acc"] for f in folds])), "folds": folds,
                                      "weights_mean": {k: round(float(np.mean([f["weights"][k] for f in folds])), 4) for k in folds[0]["weights"]}}
        else:
            sp = np.array([r["split"] for r in rows])
            r = run(info, X, y, np.where(sp == "train")[0], np.where(sp == "dev")[0], np.where(sp == "test")[0], seed, a, C, ids, out / f"preds-seed{seed}.npz")
            res["seeds"][str(seed)] = r
            res["sizes"] = {s: int((sp == s).sum()) for s in ("train", "dev", "test")}
        print(json.dumps({"seed": seed, "test_acc": res["seeds"][str(seed)]["test_acc"]}), flush=True)
    res["train_seconds"] = round(time.time() - t0)
    (out / a.json_name).write_text(json.dumps(res, indent=1))
    print("POOLED_DONE", a.task, a.feat, {s: round(v["test_acc"], 4) for s, v in res["seeds"].items()}, flush=True)


if __name__ == "__main__":
    main()
