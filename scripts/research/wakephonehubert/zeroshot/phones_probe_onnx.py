"""Exports the phoneme probe trained by bench/pr.py for the model card and scores it through the int8 ONNX on LibriSpeech
test-clean.

    phones_probe_onnx.py [--probe-dir DIR] [--procs 6]
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import zslib  # noqa: E402

NAMES = [f"tap{i}" for i in range(9)] + ["out", "vad", "ipa"]


def export(seed, probe_dir):
    sd = torch.load(Path(probe_dir) / f"head-seed{seed}.pt", map_location="cpu")
    d = {"mix": sd["0.w"].numpy(), "names": np.asarray(NAMES), "lin_weight": sd["1.weight"].numpy(), "lin_bias": sd["1.bias"].numpy()}
    for n in NAMES:
        d[f"mu_{n}"] = sd[f"0.mu_{n}"].numpy(); d[f"sd_{n}"] = sd[f"0.sd_{n}"].numpy()
        if f"0.proj.{n}.weight" in sd:
            d[f"proj_{n}_weight"] = sd[f"0.proj.{n}.weight"].numpy(); d[f"proj_{n}_bias"] = sd[f"0.proj.{n}.bias"].numpy()
    return d


_S = {}


def work(row):
    path, _, ref = row
    if "s" not in _S:
        _S["s"] = zslib.session(threads=1)
    w = sf.read(path, dtype="float32")[0]
    o = zslib.run(_S["s"], w, ("layers", "hubert", "vad", "ipa"))
    e = {f"tap{i}": o["layers"][:, i] for i in range(9)} | {"out": o["hubert"], "vad": o["vad"], "ipa": o["ipa"]}
    hyps = []
    for h in HEADS:
        m = np.exp(h["mix"] - h["mix"].max()); m /= m.sum()
        acc = 0
        for i, n in enumerate(NAMES):
            x = (e[n] - h[f"mu_{n}"]) / h[f"sd_{n}"]
            if f"proj_{n}_weight" in h:
                x = x @ h[f"proj_{n}_weight"].T + h[f"proj_{n}_bias"]
            acc = acc + m[i] * x
        best = (acc @ h["lin_weight"].T + h["lin_bias"]).argmax(1)
        hyps.append([int(t) for i, t in enumerate(best) if t != 0 and (i == 0 or t != best[i - 1])])
    return hyps, [zslib.edit(h, ref) for h in hyps], len(ref)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    zslib.add_args(ap)
    ap.add_argument("--probe-dir", default="", help="pr.py train output of the wakephonehubert row (default <bench-dir>/results/pr/wakephonehubert)")
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    zslib.configure(a)
    HEADS = [export(s, a.probe_dir or zslib.BENCH / "results/pr/wakephonehubert") for s in (0, 1)]
    A = zslib.ZS / "card_assets/heads/phones_probe.npz"
    A.parent.mkdir(parents=True, exist_ok=True)
    (zslib.ZS / "results/phones").mkdir(parents=True, exist_ok=True)
    np.savez(A, **HEADS[0])
    rows = json.loads((zslib.BENCH / "data/pr_targets.json").read_text())["test"]
    with Pool(a.procs) as p:
        R = p.map(work, rows, chunksize=8)
    nref = sum(r[2] for r in R)
    res = {"set": "LibriSpeech test-clean", "utterances": len(rows), "phones_ref": nref, "path": "wakephonehubert_int8.onnx outputs, numpy probe",
           **{f"per_seed{s}": sum(r[1][j] for r in R) / nref for j, s in enumerate((0, 1))}}
    np.savez_compressed(zslib.ZS / "results/phones/probe-onnx-raw.npz", paths=np.asarray([r[0] for r in rows]),
                        **{f"hyp_seed{s}": np.concatenate([np.asarray(r[0][j], np.int16) for r in R]) for j, s in enumerate((0, 1))},
                        **{f"hyp_offsets_seed{s}": np.cumsum([0] + [len(r[0][j]) for r in R]).astype(np.int64) for j, s in enumerate((0, 1))},
                        **{f"edits_seed{s}": np.asarray([r[1][j] for r in R]) for j, s in enumerate((0, 1))})
    (zslib.ZS / "results/phones/probe-onnx.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
