"""score_stream.py <featurizer.onnx> <out_dir> --heads NAME=head.onnx[:positive_set] ... --sets <name>=<glob> ... [--procs N]

Windowing is window_feats.py's: isolated 24,000-sample window per 1,280-sample block, zero-initialised buffer, same
files, order, resampling and 1 s padding for positive sets; features are rounded through float16 as the stored
windows are. Speech and nonspeech go through every head, a positive set only through the heads that name it.
Writes trace-<name>.part0of1.npy (keys speech, nonspeech, heldout) like score_windows.py, plus sets-index.json.
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from window_feats import BLOCK, SR, WIN

_S = {}


def _work(task):
    feat_path, heads, f, pad, names = task
    import onnxruntime as ort
    import soundfile as sf
    if "fz" not in _S:
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        _S["fz"] = ort.InferenceSession(feat_path, so, providers=["CPUExecutionProvider"])
        _S["hd"] = {}
    for n in names:
        if n not in _S["hd"]:
            so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
            s = ort.InferenceSession(heads[n], so, providers=["CPUExecutionProvider"])
            _S["hd"][n] = (s, s.get_inputs()[0].name)
    try:
        w, sr = sf.read(f, dtype="float32", always_2d=True)
    except Exception:
        return f, None
    w = w.mean(1)
    if sr != SR:
        import torch, torchaudio
        torch.set_num_threads(1)
        w = torchaudio.functional.resample(torch.from_numpy(w), sr, SR).numpy()
    if pad:
        w = np.concatenate([np.zeros(SR, np.float32), w, np.zeros(SR, np.float32)])
    buf = np.concatenate([np.zeros(WIN, np.float32), w])
    n = len(w) // BLOCK
    if n == 0:
        return f, None
    wins = np.stack([buf[(i + 1) * BLOCK:(i + 1) * BLOCK + WIN] for i in range(n)])
    fz = _S["fz"]
    inp, out = fz.get_inputs()[0].name, fz.get_outputs()[0].name
    z = {k: [] for k in names}
    for i in range(0, n, 32):
        x = fz.run([out], {inp: wins[i:i + 32]})[0].astype(np.float16).astype(np.float32)
        for k in names:
            s, hin = _S["hd"][k]
            z[k].append(s.run(None, {hin: x})[0].reshape(-1))
    return f, {k: (1 / (1 + np.exp(-np.concatenate(v)))).astype(np.float32) for k, v in z.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("featurizer"); ap.add_argument("out_dir")
    ap.add_argument("--heads", nargs="+", required=True); ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--limit", nargs="*", default=[], help="set=N: keep the first N files of that set")
    a = ap.parse_args()
    limits = {s.split("=")[0]: int(s.split("=")[1]) for s in a.limit}
    heads, pos = {}, {}
    for h in a.heads:
        name, rest = h.split("=", 1)
        path, _, ps = rest.partition(":")
        heads[name] = path; pos[name] = ps or None
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    res = {n: {} for n in heads}
    index = {}
    for spec in a.sets:
        sname, pat = spec.split("=", 1)
        if sname in ("speech", "nonspeech"):
            names = list(heads); key = sname
        else:
            names = [n for n in heads if pos[n] == sname]; key = "heldout"
        if not names:
            continue
        files = sorted(glob.glob(pat, recursive=True))
        if sname == "nonspeech":
            files = files[::10]
        if sname in limits:
            files = files[:limits[sname]]
        pad = sname not in ("speech", "nonspeech")
        acc = {n: [] for n in names}; kept = []
        with ProcessPoolExecutor(a.procs) as ex:
            for f, r in ex.map(_work, [(a.featurizer, heads, f, pad, names) for f in files], chunksize=4):
                if r is None:
                    continue
                kept.append(f)
                for n in names:
                    acc[n].append(r[n])
        for n in names:
            res[n][key] = acc[n]
        index[sname] = kept
        print("set", sname, len(kept), "files", sum(len(x) for x in acc[names[0]]), "blocks", "heads", len(names), flush=True)
    for n in heads:
        np.save(out / f"trace-{n}.part0of1.npy", np.array(res[n], dtype=object), allow_pickle=True)
    json.dump(index, open(out / "sets-index.json", "w"))
    print("STREAM_DONE", flush=True)


if __name__ == "__main__":
    main()
