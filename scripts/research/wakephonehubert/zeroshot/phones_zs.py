"""Zero-shot phone recognition: greedy CTC decoding of the ipa output on LibriSpeech test-clean, PER against the
benchmark's eSpeak targets (data/pr_targets.json, the same targets pr.py scores).

The ipa output is trained with a 5-frame (100 ms) delay, so the last phone of an utterance is emitted up to 100 ms
after its audio ends. "delay handled" appends 200 ms of zeros before decoding; "no padding" decodes the raw utterance.
Writes results/phones/raw.npz (per utterance: both hypotheses, the reference, per-frame top-8 ids and probabilities)
and results/phones/results.json.
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

PAD = int(0.2 * zslib.SR)
_S = {}


def work(row):
    path, _, ref = row
    if "s" not in _S:
        _S["s"] = zslib.session(threads=1)
    w = sf.read(path, dtype="float32")[0]
    n = len(w) // zslib.HOP
    p = zslib.run(_S["s"], np.concatenate([w, np.zeros(PAD, np.float32)]), ("ipa",))["ipa"]
    am = p.argmax(-1)
    top = np.argsort(-p, -1)[:, :8]
    return {"hyp_delay": zslib.greedy(am), "hyp_nopad": zslib.greedy(am[:n]), "ref": ref,
            "top_ids": top.astype(np.int16), "top_p": np.take_along_axis(p, top, -1).astype(np.float16), "frames_audio": n}


def decode_penalised(top_ids, top_p, off, pen):
    """Greedy decoding with the blank's log probability lowered by pen, from the stored per-frame top-8."""
    lp = np.log(np.clip(top_p.astype(np.float32), 1e-10, None)) - pen * (top_ids == 0)
    best = top_ids[np.arange(len(top_ids)), lp.argmax(1)]
    return [zslib.greedy(best[off[i]:off[i + 1]]) for i in range(len(off) - 1)]


def main():
    ap = argparse.ArgumentParser()
    zslib.add_args(ap)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    zslib.configure(a)
    T = json.loads((zslib.BENCH / "data/pr_targets.json").read_text())
    od = zslib.ZS / "results/phones"
    od.mkdir(parents=True, exist_ok=True)
    res = {"task": "phones", "decoding": "greedy CTC over ipa argmax, repeats collapsed, ids 0-3 dropped", "model": "wakephonehubert_int8.onnx"}
    raw = {}
    for split in ("dev", "test"):
        rows = T[split]
        t0 = time.time()
        with Pool(a.procs) as p:
            R = p.map(work, rows, chunksize=8)
        nref = sum(len(r["ref"]) for r in R)
        res[split] = {"set": {"dev": "LibriSpeech dev-clean", "test": "LibriSpeech test-clean"}[split], "utterances": len(rows), "phones_ref": nref,
                      "phones_hyp_delay": sum(len(r["hyp_delay"]) for r in R)}
        for k in ("hyp_delay", "hyp_nopad"):
            res[split][f"per_{k}"] = sum(zslib.edit(r[k], r["ref"]) for r in R) / nref
        cat = lambda k: np.concatenate([np.asarray(r[k], np.int16) for r in R])
        off = lambda k: np.cumsum([0] + [len(r[k]) for r in R]).astype(np.int64)
        raw[split] = dict(paths=np.asarray([r[0] for r in rows]), hyp_delay=cat("hyp_delay"), hyp_delay_offsets=off("hyp_delay"),
                          hyp_nopad=cat("hyp_nopad"), hyp_nopad_offsets=off("hyp_nopad"), ref=cat("ref"), ref_offsets=off("ref"),
                          top_ids=np.concatenate([r["top_ids"] for r in R]), top_p=np.concatenate([r["top_p"] for r in R]),
                          top_offsets=np.cumsum([0] + [len(r["top_ids"]) for r in R]).astype(np.int64),
                          frames_audio=np.asarray([r["frames_audio"] for r in R]))
        res[split]["seconds"] = round(time.time() - t0)
        print(json.dumps(res[split]), flush=True)
    refs = {s: [raw[s]["ref"][raw[s]["ref_offsets"][i]:raw[s]["ref_offsets"][i + 1]].tolist() for i in range(len(raw[s]["ref_offsets"]) - 1)] for s in raw}
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
    dev_per = {}
    for pen in grid:
        H = decode_penalised(raw["dev"]["top_ids"], raw["dev"]["top_p"], raw["dev"]["top_offsets"], pen)
        dev_per[pen] = sum(zslib.edit(h, r) for h, r in zip(H, refs["dev"])) / res["dev"]["phones_ref"]
        print(json.dumps({"blank_penalty": pen, "dev_per": dev_per[pen]}), flush=True)
    pen = min(dev_per, key=dev_per.get)
    H = decode_penalised(raw["test"]["top_ids"], raw["test"]["top_p"], raw["test"]["top_offsets"], pen)
    res["blank_penalty"] = {"grid_dev_per": {str(k): v for k, v in dev_per.items()}, "chosen_on_dev": pen,
                            "test_per": sum(zslib.edit(h, r) for h, r in zip(H, refs["test"])) / res["test"]["phones_ref"],
                            "test_phones_hyp": sum(len(h) for h in H)}
    for s in raw:
        np.savez_compressed(od / f"raw-{s}.npz", **raw[s])
    (od / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["blank_penalty"]), flush=True)


if __name__ == "__main__":
    main()
