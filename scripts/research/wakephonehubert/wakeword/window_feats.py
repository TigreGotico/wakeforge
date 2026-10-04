"""window_feats.py <featurizer.onnx> <out_dir> <set>=<dir-or-glob> ... [--procs N]

Reproduces OnnxWindowedWakeWord with isolated windows (the plugin default, no AGC): the stream is pushed in
1,280-sample (80 ms) blocks into a 24,000-sample buffer that starts as zeros, and after every block the whole buffer is
featurized on its own; the head later scores those 75 frames. Positives are padded with 1 s of silence each side.
Stores, per set, an fp16 memmap of [blocks, 75, F] plus an index of (file, first block, n blocks).
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SR, BLOCK, WIN = 16000, 1280, 24000


def _feats(args):
    feat_path, f, pad = args
    import onnxruntime as ort
    import soundfile as sf
    global _FZ
    if "_FZ" not in globals():
        so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.inter_op_num_threads = 1
        _FZ = ort.InferenceSession(feat_path, so, providers=["CPUExecutionProvider"])
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
    inp, out = _FZ.get_inputs()[0].name, _FZ.get_outputs()[0].name
    feats = np.concatenate([_FZ.run([out], {inp: wins[i:i + 32]})[0] for i in range(0, n, 32)])
    return f, feats.astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("featurizer"); ap.add_argument("out_dir"); ap.add_argument("sets", nargs="+")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for spec in a.sets:
        name, pat = spec.split("=", 1)
        if (out / f"{name}.index.json").exists():
            print("have", name, flush=True); continue
        if name == "nonspeech":
            files = sorted(glob.glob(pat, recursive=True))[::10]
        else:
            files = sorted(glob.glob(pat, recursive=True))
        pad = name not in ("speech", "nonspeech")
        raw = open(out / f"{name}.raw", "wb"); index, pos, dim = [], 0, None
        with ProcessPoolExecutor(a.procs) as ex:
            for f, feats in ex.map(_feats, [(a.featurizer, f, pad) for f in files], chunksize=4):
                if feats is None:
                    continue
                dim = feats.shape[-1]; raw.write(feats.tobytes()); index.append([f, pos, len(feats)]); pos += len(feats)
        raw.close()
        json.dump({"dim": dim, "frames": 75, "blocks": pos, "files": index}, open(out / f"{name}.index.json", "w"))
        print("set", name, len(index), "files", pos, "blocks", flush=True)
    print("WINDOWS_DONE", flush=True)


if __name__ == "__main__":
    main()
