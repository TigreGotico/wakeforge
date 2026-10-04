"""Zero-shot keyword spotting traces: plugin-faithful isolated 1.5 s windows (window_feats.py), one per 80 ms block,
each scored by the CTC keyword scorer in zslib over the ipa output. Writes results/wakeword/trace-<set>.npz with, per
keyword and per scorer, one float32 score per block, and the file index (file, first block, blocks).

    kws_traces.py --negative NAME=GLOB ... --positive WORD=GLOB ... [--every NAME=N ...] [--procs N]

Each --negative is a stream of speech or non-speech scored whole; each --positive is a folder of single utterances of WORD,
padded with 1 s of silence on both sides, stored as set pos_WORD. --every keeps every N-th file of a negative set. sets.json
names the sets for kws_eval.py.
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import zslib  # noqa: E402

SR, BLOCK, WIN = 16000, 1280, 24000
_S = {}


def work(task):
    f, pad, kws = task
    import soundfile as sf
    if "s" not in _S:
        _S["s"] = zslib.session("wakephonehubert_features_int8.onnx", threads=1)
    s = _S["s"]
    try:
        w, sr = sf.read(f, dtype="float32", always_2d=True)
    except Exception:
        return f, None
    w = w.mean(1)
    if sr != SR:
        import torch
        import torchaudio
        torch.set_num_threads(1)
        w = torchaudio.functional.resample(torch.from_numpy(w), sr, SR).numpy()
    if pad:
        w = np.concatenate([np.zeros(SR, np.float32), w, np.zeros(SR, np.float32)])
    buf = np.concatenate([np.zeros(WIN, np.float32), w])
    n = len(w) // BLOCK
    if n == 0:
        return f, None
    a, b = zslib.LAYOUT["ipa"]
    inp = s.get_inputs()[0].name
    out = {}
    for i in range(0, n, 64):
        wins = np.stack([buf[(j + 1) * BLOCK:(j + 1) * BLOCK + WIN] for j in range(i, min(n, i + 64))])
        ipa = s.run(["features"], {inp: wins})[0][:, :, a:b]
        for word, kw in kws.items():
            bp, fw = zslib.kws_scores(ipa, kw)
            out.setdefault(f"{word}_best_path", []).append(bp)
            out.setdefault(f"{word}_forward", []).append(fw)
    return f, {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    zslib.add_args(ap)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--negative", action="append", default=[], metavar="NAME=GLOB")
    ap.add_argument("--positive", action="append", default=[], metavar="WORD=GLOB")
    ap.add_argument("--every", action="append", default=[], metavar="NAME=N")
    a = ap.parse_args()
    zslib.configure(a)
    every = {s.split("=")[0]: int(s.split("=")[1]) for s in a.every}
    SETS = {n: (g, False, every.get(n, 1)) for n, g in (s.split("=", 1) for s in a.negative)}
    SETS |= {f"pos_{w}": (g, True, 1) for w, g in (s.split("=", 1) for s in a.positive)}
    WORDS = [s.split("=")[0] for s in a.positive]
    od = zslib.ZS / "results/wakeword"
    od.mkdir(parents=True, exist_ok=True)
    (od / "sets.json").write_text(json.dumps({"negatives": [s.split("=")[0] for s in a.negative], "positives": {w: f"pos_{w}" for w in WORDS}}))
    kws = {w: zslib.keyword_ids(w) for w in WORDS}
    voc = zslib.vocab()
    (od / "keywords.json").write_text(json.dumps({w: {"ids": v, "phones": [voc[i] for i in v]} for w, v in kws.items()}, ensure_ascii=False, indent=1))
    for name, (pat, pad, every) in SETS.items():
        if (od / f"trace-{name}.npz").exists():
            print("have", name, flush=True)
            continue
        files = sorted(glob.glob(pat, recursive=True))[::every]
        t0 = time.time()
        index, traces, pos = [], {}, 0
        with ProcessPoolExecutor(a.procs) as ex:
            for f, tr in ex.map(work, [(f, pad, kws) for f in files], chunksize=4):
                if tr is None:
                    continue
                m = len(next(iter(tr.values())))
                index.append((f, pos, m))
                pos += m
                for k, v in tr.items():
                    traces.setdefault(k, []).append(v)
        np.savez_compressed(od / f"trace-{name}.npz", files=np.asarray([i[0] for i in index]), first_block=np.asarray([i[1] for i in index]),
                            n_blocks=np.asarray([i[2] for i in index]), padded=np.asarray(pad), block_s=np.asarray(BLOCK / SR),
                            **{k: np.concatenate(v) for k, v in traces.items()})
        print(json.dumps({"set": name, "files": len(index), "blocks": pos, "hours": round(pos * BLOCK / SR / 3600, 3),
                          "seconds": round(time.time() - t0)}), flush=True)
    print("KWS_TRACES_DONE", flush=True)


if __name__ == "__main__":
    main()
