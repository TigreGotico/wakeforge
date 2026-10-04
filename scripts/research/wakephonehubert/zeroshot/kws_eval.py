"""Recall at fixed false-activation rates from the zero-shot keyword traces (kws_traces.py).

False activations are counted as recall_at_fa_mlops.py counts them: blocks at or above the threshold, merged into one
event while they are no more than 25 blocks (2 s) apart, summed over every negative stream and divided by its hours.
A positive file is detected when its highest block reaches the threshold. The operating threshold for a target rate
is the lowest one that keeps false activations at or below it (binary search over the observed scores, as there).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import zslib  # noqa: E402

GAP = 25


def per_file(z, key):
    x = z[key]
    return [x[a:a + n] for a, n in zip(z["first_block"], z["n_blocks"])]


def main():
    ap = argparse.ArgumentParser()
    zslib.add_args(ap)
    zslib.configure(ap.parse_args())
    OD = zslib.ZS / "results/wakeword"
    sets = json.loads((OD / "sets.json").read_text())
    NEG, POS = sets["negatives"], sets["positives"]
    Z = {s: dict(np.load(OD / f"trace-{s}.npz")) for s in NEG + list(POS.values())}
    hours = {s: float(Z[s]["n_blocks"].sum() * 0.08 / 3600) for s in NEG}
    H = sum(hours.values())
    res = {"negatives_hours": hours, "negatives_hours_total": H, "positives_files": {w: int(len(Z[p]["files"])) for w, p in POS.items()}}
    curves = {}
    for w, pset in POS.items():
        for sc in ("best_path", "forward"):
            key = f"{w}_{sc}"
            sep = np.full(GAP + 1, -np.inf, np.float32)
            neg = np.concatenate([np.concatenate([x, sep]) for s in NEG for x in per_file(Z[s], key)])
            pos_max = np.array([x.max() for x in per_file(Z[pset], key)])

            def fa(t):
                idx = np.flatnonzero(neg >= t)
                return 0.0 if idx.size == 0 else (1 + int(np.count_nonzero(np.diff(idx) > GAP))) / H

            cand = np.unique(np.concatenate([pos_max, neg[neg >= np.quantile(neg[np.isfinite(neg)], 0.99)]]))
            r = {}
            for target in (1.0, 0.5):
                lo, hi = 0, len(cand) - 1
                if fa(cand[hi]) > target:
                    r[f"at_{target}"] = {"threshold": None, "recall": 0.0, "note": "above target even at the top score"}
                    continue
                while lo < hi:
                    mid = (lo + hi) // 2
                    if fa(cand[mid]) <= target:
                        hi = mid
                    else:
                        lo = mid + 1
                t = float(cand[lo])
                r[f"at_{target}"] = {"threshold": t, "fa_per_h": fa(t), "recall": float((pos_max >= t).mean())}
            ths = np.unique(np.quantile(cand, np.linspace(0, 1, 400)))
            curves[key] = np.stack([ths, [fa(t) for t in ths], [(pos_max >= t).mean() for t in ths]])
            curves[f"{key}_positive_max"] = pos_max
            res[key] = r
            print(json.dumps({key: r}), flush=True)
    np.savez_compressed(OD / "curves.npz", **curves)
    (OD / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
