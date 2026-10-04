"""Validate a build_wakephonehubert.py output directory.

    validate_wakephonehubert.py --build DIR --int8 wakehubert_int8.onnx --heads RUN_DIR --clips CLIPS_DIR/val
                                [--latency-runs 300] [--window 24000]

Checks, on 4 random inputs and every .wav in --clips:
  hubert bit-identical to the published int8 featurizer (full model at default threads and 1 thread, and the
  features-only file's hubert slice);
  each head in the ONNX against the PyTorch head fed the ONNX's own int8 taps (max abs difference of the
  probabilities), for the final build and for the float32-heads reference build;
  each head in the ONNX against the PyTorch head fed the float trunk's taps (the trunk train_pool.py trains on):
  vad decisions at 0.5, ipa argmax per frame;
  the ONNX checker (full_check) on all files; file sizes; per-call latency on one thread for a --window-sample
  waveform, as the plugin's streaming runs it every 1,280 samples (80 ms).
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_wakephonehubert import ACTIVATION, activate, dct_matrix, load_head, read_audio  # noqa: E402
from wph_model import Trunk, load_trunk  # noqa: E402

BLOCK_MS = 80


def session(path, threads=0):
    o = ort.SessionOptions()
    if threads:
        o.intra_op_num_threads = threads
        o.inter_op_num_threads = 1
        o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def run(s, w):
    return dict(zip([o.name for o in s.get_outputs()], s.run(None, {"waveform": w})))


def u32(x):
    return np.ascontiguousarray(x).view(np.uint32)


def decisions(h, x):
    return x.argmax(-1) if ACTIVATION[h] == "softmax" else x >= 0.5


class Stat:
    def __init__(self):
        self.max_abs, self.agree, self.total = 0.0, 0, 0

    def add(self, h, a, b):
        t = min(a.shape[1], b.shape[1])
        a, b = a[:, :t], b[:, :t]
        self.max_abs = max(self.max_abs, float(np.abs(a - b).max()))
        da, db = decisions(h, a), decisions(h, b)
        self.agree += int((da == db).sum())
        self.total += int(da.size)

    def report(self, h):
        return {"max_abs_diff": self.max_abs, "decision_agreement": self.agree / max(self.total, 1), "decisions": self.total,
                "decision": "argmax per frame" if ACTIVATION[h] == "softmax" else "p >= 0.5 per frame"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--latency-runs", type=int, default=300)
    ap.add_argument("--window", type=int, default=24000)
    ap.add_argument("--prefix-of", default="", help="an earlier build's features file whose leading columns must equal this build's features bit for bit")
    a = ap.parse_args()
    b = Path(a.build)
    full_p, feat_p = b / "wakephonehubert_int8.onnx", b / "wakephonehubert_features_int8.onnx"
    ref_p = b / "build-intermediate" / "reference_float32_heads.onnx"
    cfg = json.loads((b / "config.json").read_text(encoding="utf-8"))
    HEAD_ORDER = tuple(cfg["heads"])
    rep = {"heads_dtype": cfg["heads_dtype"], "feature_dim": cfg["feature_dim"], "layout": cfg["output"]["layout"]}

    for p in (full_p, feat_p, ref_p, Path(a.int8)):
        onnx.checker.check_model(onnx.load(str(p)), full_check=True)
    rep["checker"] = "pass (full_check): " + ", ".join(p.name for p in (full_p, feat_p, ref_p, Path(a.int8)))
    rep["sizes_bytes"] = {p.name: p.stat().st_size for p in (Path(a.int8), full_p, feat_p, b / "config.json", b / "vocab.json")}
    m = onnx.load(str(full_p))
    rep["outputs"] = {o.name: [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim] for o in m.graph.output}
    rep["node_counts"] = {k: sum(n.op_type == k for n in m.graph.node) for k in ("Conv", "QuantizeLinear", "DequantizeLinear", "Relu", "Cast")}

    rng = np.random.default_rng(0)
    random_inputs = [("gauss_b2_40000", (rng.standard_normal((2, 40000)) * 0.1).astype(np.float32)),
                     ("uniform_b1_17123", rng.uniform(-0.5, 0.5, (1, 17123)).astype(np.float32)),
                     ("gauss_b1_96000", (rng.standard_normal((1, 96000)) * 0.02).astype(np.float32)),
                     ("short_b1_1000", (rng.standard_normal((1, 1000)) * 0.1).astype(np.float32))]
    files = sorted(glob.glob(f"{a.clips}/*.wav"))
    real_inputs = [(Path(f).stem, read_audio(f)[None]) for f in files]

    P, F, G, R = session(a.int8), session(full_p), session(feat_p), session(ref_p)
    P1, F1, G1 = session(a.int8, 1), session(full_p, 1), session(feat_p, 1)
    trunk = Trunk(load_trunk()).eval()
    heads = {h: load_head(Path(a.heads) / f"{h}_head.pt")[0] for h in HEAD_ORDER}
    dct = torch.from_numpy(dct_matrix())

    bit = []
    st = {k: {h: Stat() for h in HEAD_ORDER} for k in ("final_vs_torch_int8_taps", "reference_vs_torch_int8_taps",
                                                         "final_vs_torch_float_taps", "torch_int8_taps_vs_torch_float_taps",
                                                         "final_vs_reference")}
    vad_active = {}
    misc = {"features_vs_concat_max_abs": 0.0, "features_only_vs_full_max_abs": 0.0, "logmel_vs_torch_max_abs": 0.0,
            "mfcc_vs_torch_dct_max_abs": 0.0, "layers_min": np.inf, "frames_onnx_minus_float_trunk": set()}
    for kind, inputs in (("random", random_inputs), ("real", real_inputs)):
        for name, w in inputs:
            pub, pub1 = run(P, w)["features"], run(P1, w)["features"]
            r, r1, rr = run(F, w), run(F1, w), run(R, w)
            g, g1 = run(G, w)["features"], run(G1, w)["features"]
            nh = pub.shape[-1]
            bit.append({"input": name, "kind": kind, "samples": int(w.shape[1]), "batch": int(w.shape[0]), "frames": int(pub.shape[1]),
                        "hubert_bitwise_equal_default_threads": bool(r["hubert"].shape == pub.shape and np.array_equal(u32(r["hubert"]), u32(pub))),
                        "hubert_bitwise_equal_1_thread": bool(r1["hubert"].shape == pub1.shape and np.array_equal(u32(r1["hubert"]), u32(pub1))),
                        "features_only_hubert_slice_bitwise_equal": bool(np.array_equal(u32(g[..., :nh]), u32(pub)) and np.array_equal(u32(g1[..., :nh]), u32(pub1)))})
            misc["features_vs_concat_max_abs"] = max(misc["features_vs_concat_max_abs"], float(np.abs(r["features"] - np.concatenate([r["hubert"], *[r[h] for h in HEAD_ORDER]], -1)).max()))
            misc["features_only_vs_full_max_abs"] = max(misc["features_only_vs_full_max_abs"], float(np.abs(g - r["features"]).max()))
            misc["layers_min"] = min(misc["layers_min"], float(r["layers"].min()))
            with torch.no_grad():
                wt = torch.from_numpy(w)
                taps_f, out_f, mel2_f = trunk(wt)
                mel_log = trunk.m.mel(wt)
                mel_norm = trunk.m.norm(mel_log)
                t = r["hubert"].shape[1]
                misc["frames_onnx_minus_float_trunk"].add(int(t - out_f.shape[-1]))
                misc["logmel_vs_torch_max_abs"] = max(misc["logmel_vs_torch_max_abs"], float(np.abs(mel_norm[..., :2 * t].transpose(1, 2).numpy() - r["logmel"]).max()))
                mf = (mel_log[..., :2 * t].transpose(1, 2) @ dct).numpy()
                misc["mfcc_vs_torch_dct_max_abs"] = max(misc["mfcc_vs_torch_dct_max_abs"], float(np.abs(mf - r["mfcc"]).max()))
                taps_q = torch.from_numpy(np.ascontiguousarray(r["layers"].transpose(0, 2, 3, 1)))
                out_q = torch.from_numpy(np.ascontiguousarray(r["hubert"].transpose(0, 2, 1)))
                lmq = torch.from_numpy(r["logmel"]).transpose(1, 2)
                mel2_q = lmq.reshape(lmq.shape[0], lmq.shape[1], t, 2).permute(0, 1, 3, 2).reshape(lmq.shape[0], -1, t)
                for h in HEAD_ORDER:
                    pq = activate(h, heads[h](taps_q, out_q, mel2_q)).numpy()
                    pf = activate(h, heads[h](taps_f, out_f, mel2_f)).numpy()
                    st["final_vs_torch_int8_taps"][h].add(h, r[h], pq)
                    st["reference_vs_torch_int8_taps"][h].add(h, rr[h], pq)
                    if kind == "real":
                        st["final_vs_torch_float_taps"][h].add(h, r[h], pf)
                        st["torch_int8_taps_vs_torch_float_taps"][h].add(h, pq, pf)
                        st["final_vs_reference"][h].add(h, r[h], rr[h])
                if kind == "real":
                    src = name.split("_")[0]
                    vad_active.setdefault(src, []).append(float((r["vad"] >= 0.5).mean()))
    rep["bit_identity"] = bit
    rep["bit_identity_all_true"] = all(all(v for k, v in x.items() if isinstance(v, bool)) for x in bit)
    rep["heads"] = {k: {h: s.report(h) for h, s in d.items()} for k, d in st.items()}
    rep["vad_active_fraction_by_source"] = {k: {"clips": len(v), "mean": round(float(np.mean(v)), 4), "max": round(float(np.max(v)), 4), "min": round(float(np.min(v)), 4)} for k, v in vad_active.items()}
    misc["frames_onnx_minus_float_trunk"] = sorted(misc["frames_onnx_minus_float_trunk"])
    rep["misc"] = misc
    rep["real_clips"] = [{"file": n, "seconds": round(w.shape[1] / 16000, 2)} for n, w in real_inputs]

    if a.prefix_of:
        O = session(a.prefix_of)
        pre = []
        for name, w in random_inputs + real_inputs:
            new, old = run(G, w)["features"], run(O, w)["features"]
            d = new.shape[-1]
            pre.append({"input": name, "new_dim": int(d), "old_dim": int(old.shape[-1]),
                        "bitwise_equal": bool(new.shape[:2] == old.shape[:2] and np.array_equal(u32(new), u32(old[..., :d])))})
        pc = {"old_features_file": a.prefix_of, "inputs": pre, "all_equal": all(x["bitwise_equal"] for x in pre)}
        (b / "prefix_check.json").write_text(json.dumps(pc, indent=1))
        print("PREFIX_CHECK", json.dumps({k: pc[k] for k in ("old_features_file", "all_equal")}), sum(x["bitwise_equal"] for x in pre), len(pre))

    win = (np.random.default_rng(1).standard_normal((1, a.window)) * 0.1).astype(np.float32)
    lat = {}
    models = [("published wakehubert_int8", a.int8), ("wakephonehubert_features_int8", feat_p), ("wakephonehubert_int8 (all outputs)", full_p),
              ("float32-heads reference (all outputs)", ref_p)]
    for _ in range(2):
        for name, p in models:
            s = session(p, 1)
            for _ in range(20):
                s.run(None, {"waveform": win})
            ts = []
            for _ in range(a.latency_runs):
                t0 = time.perf_counter()
                s.run(None, {"waveform": win})
                ts.append((time.perf_counter() - t0) * 1000)
            ts = np.array(ts)
            lat.setdefault(name, []).append({"median_ms": round(float(np.median(ts)), 2), "p95_ms": round(float(np.percentile(ts, 95)), 2),
                                             "mean_ms": round(float(ts.mean()), 2), "fraction_of_80ms_block": round(float(np.median(ts)) / BLOCK_MS, 3)})
    rep[f"latency_{a.window}_sample_window_1_thread"] = lat
    rep["loadavg_after"] = os.getloadavg()
    rep["cpu"] = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    print(json.dumps(rep, indent=1))
    (b / "validation_report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
