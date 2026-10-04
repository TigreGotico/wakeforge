"""score_windows.py <windows_dir> <head.onnx> <name> <positive_set> [--scale 1.0] [--out DIR (default traces)]

Writes trace-<name>.part0of1.npy with keys speech, nonspeech, heldout: per file, the sigmoid of the head logit
times --scale at every 80 ms block, as the plugin trace does.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def load(d, name):
    idx = json.load(open(Path(d) / f"{name}.index.json"))
    mm = np.memmap(Path(d) / f"{name}.raw", np.float16, "r", shape=(idx["blocks"], idx["frames"], idx["dim"]))
    return idx, mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windows_dir"); ap.add_argument("head"); ap.add_argument("name"); ap.add_argument("positive_set")
    ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--out", default="traces")
    ap.add_argument("--dims", type=int, default=0, help="use only the first N feature channels (0 = all)")
    a = ap.parse_args()
    so = ort.SessionOptions(); so.intra_op_num_threads = 8
    hd = ort.InferenceSession(a.head, so, providers=["CPUExecutionProvider"]); hin = hd.get_inputs()[0].name
    out = {}
    for key, setname in (("speech", "speech"), ("nonspeech", "nonspeech"), ("heldout", a.positive_set)):
        idx, mm = load(a.windows_dir, setname)
        z = np.empty(idx["blocks"], np.float32)
        for i in range(0, idx["blocks"], 4096):
            x = np.asarray(mm[i:i + 4096], np.float32)
            if a.dims:
                x = x[..., :a.dims]
            z[i:i + len(x)] = np.concatenate([hd.run(None, {hin: x[j:j + 512]})[0].reshape(-1) for j in range(0, len(x), 512)])
        p = (1 / (1 + np.exp(-a.scale * z))).astype(np.float32)
        out[key] = [p[s:s + n] for _, s, n in idx["files"]]
        print(a.name, key, len(out[key]), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.save(Path(a.out) / f"trace-{a.name}.part0of1.npy", np.array(out, dtype=object), allow_pickle=True)
    print("SCORED", a.name, flush=True)


if __name__ == "__main__":
    main()
