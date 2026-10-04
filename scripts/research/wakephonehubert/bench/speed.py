"""speed.py [--rows ...] [--bench-dir DIR] [--heads RUN_DIR] [--efficientat DIR]

Parameter count, weight-file size and one-thread CPU latency per 80 ms of audio for every benchmark row.

Streaming setting: a model is called once per 80 ms hop. WakeHuBERT-tiny, WakePhoneHuBERT and log-mel run on a 2.5 s
window per call. HuBERT-base and wav2vec2-xlsr-53 have no streaming window of their own and are timed on the same 2.5 s
window. EfficientAT is timed on the 5 s window it is fed in this benchmark. Silero runs on its own 32 ms chunk (2.5 chunks
per 80 ms) and WebRTC on 10 ms frames (8 per 80 ms). Median over 30 calls after 5 warm-up calls, torch and onnxruntime
held to one thread. Absolute times carry whatever else the host is running.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import featlib  # noqa: E402
from featlib import W2V, Featurizer, param_count  # noqa: E402

B = Path("work/bench")
WIN = 40000


def timeit(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1000


def hub_size(repo, fname, rev=None):
    from huggingface_hub import HfApi
    i = HfApi().model_info(repo, revision=rev, files_metadata=True)
    return {s.rfilename: s.size for s in i.siblings}[fname]


def ort_sess(path):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def main():
    global B
    ap = argparse.ArgumentParser()
    featlib.add_args(ap)
    ap.add_argument("--rows", default="logmel,wakehubert,wakehubert-onnx-int8,wakephonehubert,hubert-base,wav2vec2-espeak,efficientat-mn10,silero,webrtc")
    a = ap.parse_args()
    B = featlib.configure(a)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((1, WIN)).astype(np.float32) * 0.1)
    out = json.loads((B / "results/speed.json").read_text()) if (B / "results/speed.json").exists() else {}
    for row in a.rows.split(","):
        r = {}
        try:
            if row in ("logmel", "wakehubert", "wakephonehubert"):
                f = Featurizer(row, device="cpu")
                m = f.trunk.m
                if row == "logmel":
                    r = {"params": sum(p.numel() for p in m.norm.parameters()), "file": "computed (trunk's CausalLogMel + BatchNorm)", "file_bytes": None,
                         "window": "2.5 s", "ms_per_80ms": timeit(lambda: m.norm(m.mel(x)))}
                elif row == "wakehubert":
                    r = {"params": param_count(f), "file": "TigreGotico/wakehubert-tiny model.safetensors",
                         "file_bytes": hub_size("TigreGotico/wakehubert-tiny", "model.safetensors", "3caa605725a932440442692f644b7217f54a29c3"),
                         "window": "2.5 s", "runtime": "pytorch fp32", "ms_per_80ms": timeit(lambda: f.trunk(x))}
                else:
                    heads = sum(p.numel() for h in (f.vad, f.ipa) for p in h.parameters())
                    hb = sum(os.path.getsize(Path(f.heads_from) / n) for n in ("vad_head.pt", "ipa_head.pt"))
                    r = {"params": param_count(f), "head_params": heads, "file": f"trunk safetensors + heads from {f.heads_from}",
                         "file_bytes": hub_size("TigreGotico/wakehubert-tiny", "model.safetensors", "3caa605725a932440442692f644b7217f54a29c3") + hb,
                         "window": "2.5 s", "runtime": "pytorch fp32", "ms_per_80ms": timeit(lambda: f.frames([x[0].numpy()]))}
            elif row == "wakehubert-onnx-int8":
                from huggingface_hub import hf_hub_download
                p = hf_hub_download("TigreGotico/wakehubert-tiny", "wakehubert_int8.onnx", revision="3caa605725a932440442692f644b7217f54a29c3")
                s = ort_sess(p)
                name = s.get_inputs()[0].name
                xn = x.numpy()
                r = {"params": None, "file": "TigreGotico/wakehubert-tiny wakehubert_int8.onnx (shipped)", "file_bytes": os.path.getsize(p),
                     "window": "2.5 s", "runtime": "onnxruntime int8", "ms_per_80ms": timeit(lambda: s.run(None, {name: xn}))}
            elif row in ("hubert-base", "wav2vec2-espeak"):
                f = Featurizer(row, device="cpu")
                fb = hub_size("facebook/hubert-base-ls960", "pytorch_model.bin") if row == "hubert-base" else hub_size(W2V[0], "model.safetensors", W2V[1])
                r = {"params": param_count(f), "file": "hub weights", "file_bytes": fb, "window": "2.5 s (no streaming window of its own)",
                     "runtime": "pytorch fp32", "ms_per_80ms": timeit(lambda: f.model(x, output_hidden_states=True), n=10, warm=2)}
            elif row == "efficientat-mn10":
                f = Featurizer(row, device="cpu")
                w = rng.standard_normal(5 * 16000).astype(np.float32) * 0.1
                r = {"params": param_count(f), "file": "EfficientAT/resources/mn10_as_mAP_471.pt",
                     "file_bytes": os.path.getsize(featlib.EFFICIENTAT_DIR / "resources/mn10_as_mAP_471.pt"), "window": "5 s (left-padded minimum used here)",
                     "runtime": "pytorch fp32", "ms_per_80ms": timeit(lambda: f.pooled([w]), n=10, warm=2)}
            elif row == "silero":
                from huggingface_hub import hf_hub_download
                from pool_teachers import CHUNK, CONTEXT, SILERO
                p = hf_hub_download(SILERO[0], SILERO[1], revision=SILERO[2])
                s = ort_sess(p)
                st = np.zeros((2, 1, 128), np.float32); inp = np.zeros((1, CHUNK + CONTEXT), np.float32)
                ms = timeit(lambda: s.run(None, {"input": inp, "state": st, "sr": np.array(16000, np.int64)}), n=200, warm=20)
                r = {"params": None, "file": "onnx-community/silero-vad onnx/model.onnx", "file_bytes": os.path.getsize(p), "window": "32 ms chunk, stateful",
                     "runtime": "onnxruntime fp32", "ms_per_80ms": ms * 80 / 32}
            elif row == "webrtc":
                import webrtcvad
                v = webrtcvad.Vad(3)
                fr = (rng.standard_normal(160) * 3000).astype(np.int16).tobytes()
                ms = timeit(lambda: v.is_speech(fr, 16000), n=2000, warm=100)
                r = {"params": None, "file": "py-webrtcvad (GMM, compiled in)", "file_bytes": None, "window": "10 ms frame", "runtime": "C", "ms_per_80ms": ms * 8}
        except SystemExit as e:
            r = {"pending": str(e)}
        if "ms_per_80ms" in r:
            r["ms_per_80ms"] = round(r["ms_per_80ms"], 4)
            r["real_time_factor_at_80ms_hop"] = round(r["ms_per_80ms"] / 80, 4)
        out[row] = r
        print(json.dumps({row: r}), flush=True)
    (B / "results").mkdir(parents=True, exist_ok=True)
    (B / "results/speed.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
