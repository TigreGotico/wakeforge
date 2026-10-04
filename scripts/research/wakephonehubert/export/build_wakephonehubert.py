"""Build the WakePhoneHuBERT ONNX: the published WakeHuBERT-tiny int8 graph, untouched, plus its taps and the heads.

    build_wakephonehubert.py --int8 wakehubert_int8.onnx --heads RUN_DIR --out DIR --clips CLIPS_DIR
                             [--try-dtypes int8-static,int8-weights,float16] [--accept-tol 0.01] [--accept-agree 0.995]

RUN_DIR holds train_pool.py's vad_head.pt and ipa_head.pt. CLIPS_DIR holds prep_clips.py's calib/ and accept/
directories.

Outputs of wakephonehubert_int8.onnx: hubert, layers, logmel, mfcc, vad, ipa, features.
  vad    [B, T, 1]    sigmoid of TapHead(1, 32) (no side branch), as train_pool.py's BCE-with-logits target
  ipa    [B, T, 392]  softmax over the wav2vec2-xlsr-53-espeak-cv-ft tokenizer ids (the CTC's log_softmax, exponentiated);
                      id 0 <pad> is the CTC blank, ids 1-3 (<s>, </s>, <unk>) are never targets
  features = hubert | vad | ipa on the last axis (128 + 1 + 392 = 521). wakephonehubert_features_int8.onnx has only features.

Head precision is chosen per head. Each dtype in --try-dtypes (smallest first) is tried for one head at a time,
the others float32, and accepted when, on the accept/ clips (disjoint from the calibration clips), against the
float32 build: vad and ipa probabilities move by less than --accept-tol and decisions (vad at 0.5, ipa argmax per
frame) agree on at least --accept-agree. Each head keeps the smallest accepted dtype (float32 when none is). The
combined build is then checked again under the same rule.
  float16      head parameters stored in float16, cast to float32 at load, computed in float32;
  int8-weights Conv weights stored int8 (symmetric, per output channel) and dequantised at load, other head
               parameters in float16; activations and arithmetic float32;
  int8-static  onnxruntime static QDQ quantisation of that head's Conv nodes (int8 per-channel weights, uint8
               activations), calibrated on the calib/ clips run through the int8 trunk.

Tap location in the static QDQ graph: every ReLU is folded into a uint8
QuantizeLinear with zero point 0; tap 0 follows the stride-2 stem Conv, taps 1-8 follow the eight residual Adds, the
128-dim output is the 1x1 Conv reading tap 8; each tap gets a new DequantizeLinear on the existing QuantizeLinear,
so the published path is not rewired.
"""
import argparse
import glob
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wph_model import TRUNK_REPO, TRUNK_REVISION, TapHead  # noqa: E402

SR, HOP, N_MELS, N_MFCC, N_TAPS, TAP_DIM, OUT_DIM = 16000, 320, 64, 20, 9, 256, 128
HEAD_ORDER = ("vad", "ipa")
ACTIVATION = {"vad": "sigmoid", "ipa": "softmax"}
BASE_OUTPUTS = ["layers", "logmel", "mfcc"]
TAIL_INPUTS = [f"tap{i}" for i in range(N_TAPS)] + ["out", "mel_norm", "mel_log", "hubert"]


def dct_matrix(n_in=N_MELS, n_out=N_MFCC):
    """Orthonormal DCT-II as [n_in, n_out]: mfcc = logmel @ D."""
    n, k = np.arange(n_in), np.arange(n_out)[:, None]
    d = np.cos(math.pi * k * (2 * n + 1) / (2 * n_in)) * math.sqrt(2.0 / n_in)
    d[0] /= math.sqrt(2.0)
    return d.T.astype(np.float32)


def load_head(path):
    """train_pool.py checkpoint -> (TapHead in eval mode, side-branch width, metadata). vad_head.pt is a bare
    state_dict; ipa_head.pt wraps it under "head"."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["head"] if "head" in ck else ck
    side = sd["side.inp.weight"].shape[0] if "side.inp.weight" in sd else 0
    head = TapHead(sd["o.weight"].shape[0], hidden=sd["c1.weight"].shape[0], side=side)
    head.load_state_dict(sd, strict=True)
    meta = {k: v for k, v in ck.items() if k in ("symbols", "blank", "dropped_ids", "delay")} if "head" in ck else {}
    return head.eval(), side, meta


def output_names():
    return BASE_OUTPUTS + list(HEAD_ORDER) + ["features"]


def activate(name, z):
    a = ACTIVATION[name]
    return torch.softmax(z, -1) if a == "softmax" else torch.sigmoid(z)


class Tail(nn.Module):
    """Everything after the int8 trunk: stacking the taps, the log-mel views, MFCC, the heads, the join."""

    def __init__(self, heads):
        super().__init__()
        for n in HEAD_ORDER:
            setattr(self, n, heads[n])
        self.register_buffer("dct", torch.from_numpy(dct_matrix()))

    def forward(self, t0, t1, t2, t3, t4, t5, t6, t7, t8, out, mel_norm, mel_log, hubert):
        taps = torch.stack([t0, t1, t2, t3, t4, t5, t6, t7, t8], 1)
        b, t = out.shape[0], out.shape[2]
        mel = mel_norm[..., :2 * t]
        mel2 = mel.reshape(b, N_MELS, t, 2).permute(0, 1, 3, 2).reshape(b, 2 * N_MELS, t)
        ys = [activate(n, getattr(self, n)(taps, out, mel2)) for n in HEAD_ORDER]
        mfcc = torch.matmul(mel_log[..., :2 * t].transpose(1, 2), self.dct)
        return (taps.permute(0, 3, 1, 2), mel.transpose(1, 2), mfcc, *ys, torch.cat([hubert, *ys], -1))


def locate(g):
    """Return the tensor names of the 9 quantised taps, the quantised 128 output, the log-mel and the normalised
    log-mel, found structurally (see the module docstring)."""
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    init = {i.name: i for i in g.initializer}

    def only(xs, what):
        assert len(xs) == 1, f"{what}: expected 1, found {len(xs)}"
        return xs[0]

    def qdq_after(tensor, relu_folded):
        q = only([c for c in cons.get(tensor, []) if c.op_type == "QuantizeLinear"], f"QuantizeLinear after {tensor}")
        dq = only([c for c in cons[q.output[0]] if c.op_type == "DequantizeLinear"], f"DequantizeLinear after {q.name}")
        if relu_folded:
            zp = numpy_helper.to_array(init[q.input[2]])
            assert zp.dtype == np.uint8 and int(zp) == 0, f"{q.name}: zero point {zp.dtype} {zp} does not fold a ReLU"
        return q, dq

    assert not [n for n in g.node if n.op_type == "Relu"], "graph has Relu nodes; expected them folded into QDQ"
    log = only([n for n in g.node if n.op_type == "Log"], "Log")
    bn = only([c for c in cons[log.output[0]] if c.op_type == "BatchNormalization"], "BatchNormalization after Log")

    def descends_from(tensor, target, seen=None):
        seen = set() if seen is None else seen
        n = prod.get(tensor)
        if n is None or n.name in seen:
            return False
        if n is target:
            return True
        seen.add(n.name)
        return any(descends_from(i, target, seen) for i in n.input)

    def attr(n, k):
        return next((helper.get_attribute_value(a) for a in n.attribute if a.name == k), None)

    stem = only([n for n in g.node if n.op_type == "Conv" and attr(n, "strides") == [2] and descends_from(n.input[0], bn)],
                "stride-2 stem Conv")
    taps = [qdq_after(stem.output[0], True)]
    residual = {n.name for n in g.node if n.op_type == "Add"
                and all(prod.get(i) is not None and prod[i].op_type == "DequantizeLinear" for i in n.input)}
    visited = []
    for _ in range(8):
        add = only([c for c in cons[taps[-1][1].output[0]] if c.op_type == "Add"], f"residual Add reading {taps[-1][1].name}")
        visited.append(add.name)
        taps.append(qdq_after(add.output[0], True))
    assert set(visited) == residual and len(residual) == 8, (visited, residual)
    out_conv = only([c for c in cons[taps[-1][1].output[0]] if c.op_type == "Conv"], "output Conv reading tap 8")
    w = numpy_helper.to_array(init[prod[out_conv.input[1]].input[0]])
    assert w.shape[:1] == (OUT_DIM,) and w.shape[2:] == (1,), w.shape
    out = qdq_after(out_conv.output[0], False)
    return {"taps": [(q.output[0], q.input[1], q.input[2]) for q, _ in taps],
            "out": (out[0].output[0], out[0].input[1], out[0].input[2]),
            "mel_log": log.output[0], "mel_norm": bn.output[0],
            "residual_adds": visited, "stem": stem.name, "out_conv": out_conv.name}


def export_tail(heads, path):
    tail = Tail(heads).eval()
    b, t = 2, 37
    args = tuple([torch.rand(b, TAP_DIM, t) for _ in range(N_TAPS)] + [torch.randn(b, OUT_DIM, t), torch.randn(b, N_MELS, 2 * t + 1),
                                                                        torch.randn(b, N_MELS, 2 * t + 1), torch.randn(b, t, OUT_DIM)])
    dyn = {n: {0: "batch", 2: "frames"} for n in TAIL_INPUTS[:N_TAPS + 1]}
    dyn.update({"mel_norm": {0: "batch", 2: "mel_frames"}, "mel_log": {0: "batch", 2: "mel_frames"}, "hubert": {0: "batch", 1: "frames"}})
    for n in output_names():
        dyn[n] = {0: "batch", 1: "mel_frames_2x" if n in ("logmel", "mfcc") else "frames"}
    torch.onnx.export(tail, args, str(path), input_names=TAIL_INPUTS, output_names=output_names(), dynamic_axes=dyn,
                      opset_version=17, dynamo=False, do_constant_folding=True)
    return onnx.load(str(path))


def trunk_with_taps(trunk, loc):
    """The published graph with its output renamed to hubert and the tail's inputs exposed as extra outputs."""
    m1 = onnx.ModelProto()
    m1.CopyFrom(trunk)
    g = m1.graph
    assert [o.name for o in g.output] == ["features"], [o.name for o in g.output]
    for n in g.node:
        n.output[:] = ["hubert" if o == "features" else o for o in n.output]
    g.output[0].name = "hubert"
    srcs = [(f"wph_tap{i}", q) for i, q in enumerate(loc["taps"])] + [("wph_out", loc["out"])]
    for name, (qout, scale, zp) in srcs:
        g.node.append(helper.make_node("DequantizeLinear", [qout, scale, zp], [name], name=f"{name}_DequantizeLinear"))
    shp = {"wph_out": ["batch", OUT_DIM, "frames"], **{f"wph_tap{i}": ["batch", TAP_DIM, "frames"] for i in range(N_TAPS)}}
    for name, s in shp.items():
        g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, s))
    for name in (loc["mel_norm"], loc["mel_log"]):
        g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", N_MELS, "mel_frames"]))
    io = [(f"wph_tap{i}", f"tap{i}") for i in range(N_TAPS)] + [("wph_out", "out"), (loc["mel_norm"], "mel_norm"),
                                                               (loc["mel_log"], "mel_log"), ("hubert", "hubert")]
    return m1, io


def merge(trunk, tail, loc):
    m1, io = trunk_with_taps(trunk, loc)
    tail = onnx.compose.add_prefix(tail, "heads/")
    tail.ir_version = m1.ir_version
    names = output_names()
    merged = onnx.compose.merge_models(m1, tail, io_map=[(a, f"heads/{b}") for a, b in io],
                                       outputs=["hubert"] + [f"heads/{n}" for n in names])
    ren = {f"heads/{n}": n for n in names}
    for n in merged.graph.node:
        n.input[:] = [ren.get(i, i) for i in n.input]
        n.output[:] = [ren.get(o, o) for o in n.output]
    for o in merged.graph.output:
        o.name = ren.get(o.name, o.name)
    merged.producer_name = "build_wakephonehubert"
    return merged


def prune(model, outputs):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    keep = [o for o in g.output if o.name in outputs]
    del g.output[:]
    g.output.extend(keep)
    need, nodes = set(outputs), []
    for n in reversed(list(g.node)):
        if any(o in need for o in n.output):
            nodes.append(n)
            need.update(i for i in n.input if i)
    del g.node[:]
    g.node.extend(reversed(nodes))
    inits = [i for i in g.initializer if i.name in need]
    del g.initializer[:]
    g.initializer.extend(inits)
    vi = [v for v in g.value_info if v.name in need]
    del g.value_info[:]
    g.value_info.extend(vi)
    return m


def node_head(n):
    """Head a tail node belongs to, from the exporter's module-path node names ("/ipa/side/inp/Conv")."""
    parts = n.name.split("/")
    return parts[1] if len(parts) > 1 and parts[1] in HEAD_ORDER else None


def float16_heads(tail, names):
    """Store every float initializer used only by nodes of the heads in `names` in float16, cast back at load."""
    m = onnx.ModelProto()
    m.CopyFrom(tail)
    g = m.graph
    users = {}
    for n in g.node:
        for i in n.input:
            users.setdefault(i, set()).add(node_head(n))
    casts, keep, moved = [], [], {h: 0 for h in names}
    for init in g.initializer:
        u = users.get(init.name, set())
        if init.data_type == TensorProto.FLOAT and len(u) == 1 and next(iter(u)) in names and np.prod(init.dims) > 1:
            h = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), f"{init.name}_fp16")
            keep.append(h)
            casts.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=f"{init.name}_Cast"))
            moved[next(iter(u))] += 1
        else:
            keep.append(init)
    assert all(moved.values()), f"no float16 parameters found for some heads: {moved}"
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = casts + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    return m


def int8_weight_heads(tail, names):
    """Weight-only int8 for the heads in `names`: every Conv weight they use is stored as int8 with a symmetric
    per-output-channel scale and dequantised to float32 at load (DequantizeLinear on the initializer); activations
    and arithmetic stay float32. Their other float parameters (biases, tap mix) are stored in float16."""
    m = onnx.ModelProto()
    m.CopyFrom(tail)
    g = m.graph
    users, weight_of = {}, set()
    for n in g.node:
        for i in n.input:
            users.setdefault(i, set()).add(node_head(n))
        if n.op_type == "Conv" and node_head(n) in names:
            weight_of.add(n.input[1])
    new_nodes, keep, moved = [], [], {h: 0 for h in names}
    for init in g.initializer:
        u = users.get(init.name, set())
        if not (init.data_type == TensorProto.FLOAT and len(u) == 1 and next(iter(u)) in names):
            keep.append(init)
            continue
        w = numpy_helper.to_array(init)
        if init.name in weight_of and w.ndim == 3:
            scale = (np.abs(w).reshape(w.shape[0], -1).max(1) / 127.0).astype(np.float32)
            scale[scale == 0] = 1.0
            q = np.clip(np.round(w / scale[:, None, None]), -127, 127).astype(np.int8)
            keep += [numpy_helper.from_array(q, f"{init.name}_int8"), numpy_helper.from_array(scale, f"{init.name}_scale"),
                     numpy_helper.from_array(np.zeros(w.shape[0], np.int8), f"{init.name}_zp")]
            new_nodes.append(helper.make_node("DequantizeLinear", [f"{init.name}_int8", f"{init.name}_scale", f"{init.name}_zp"],
                                              [init.name], axis=0, name=f"{init.name}_DequantizeLinear"))
            moved[next(iter(u))] += 1
        elif w.size > 1:
            keep.append(numpy_helper.from_array(w.astype(np.float16), f"{init.name}_fp16"))
            new_nodes.append(helper.make_node("Cast", [f"{init.name}_fp16"], [init.name], to=TensorProto.FLOAT, name=f"{init.name}_Cast"))
        else:
            keep.append(init)
    assert all(moved.values()), f"no Conv weights found for some heads: {moved}"
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = new_nodes + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    return m


class TailInputs:
    """CalibrationDataReader over the tail inputs produced by the int8 trunk on real clips."""

    def __init__(self, trunk_path, io, clips):
        import onnxruntime as ort
        s = ort.InferenceSession(str(trunk_path), providers=["CPUExecutionProvider"])
        names = [a for a, _ in io]
        self.items = []
        for w in clips:
            r = dict(zip(names, s.run(names, {"waveform": w[None]})))
            self.items.append({b: r[a] for a, b in io})
        self.it = iter(self.items)

    def get_next(self):
        return next(self.it, None)

    def rewind(self):
        self.it = iter(self.items)


def int8_heads(tail_path, out_path, reader, names):
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    m = onnx.load(str(tail_path))
    nodes = [n.name for n in m.graph.node if n.op_type == "Conv" and node_head(n) in names]
    assert nodes, names
    pre = Path(out_path).with_suffix(".pre.onnx")
    quant_pre_process(str(tail_path), str(pre), skip_symbolic_shape=True)
    reader.rewind()
    quantize_static(str(pre), str(out_path), reader, quant_format=QuantFormat.QDQ,
                    op_types_to_quantize=["Conv"], nodes_to_quantize=nodes, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax)
    return onnx.load(str(out_path))


def read_audio(path):
    import soundfile as sf
    w, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == SR, (path, sr)
    return w.mean(1)


def clip_dir(d):
    files = sorted(glob.glob(f"{d}/*.wav"))
    assert files, d
    return [read_audio(f) for f in files]


def acceptance(ref_path, cand_path, clips, names, rule):
    import onnxruntime as ort
    a = ort.InferenceSession(str(ref_path), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(cand_path), providers=["CPUExecutionProvider"])
    acc = {h: {"max_abs_diff": 0.0, "agree": 0, "total": 0} for h in names}
    for w in clips:
        ra = dict(zip(names, a.run(list(names), {"waveform": w[None]})))
        rb = dict(zip(names, b.run(list(names), {"waveform": w[None]})))
        for h in names:
            r = acc[h]
            r["max_abs_diff"] = max(r["max_abs_diff"], float(np.abs(ra[h] - rb[h]).max()))
            da, db = (ra[h].argmax(-1), rb[h].argmax(-1)) if ACTIVATION[h] == "softmax" else (ra[h] >= 0.5, rb[h] >= 0.5)
            r["agree"] += int((da == db).sum())
            r["total"] += int(da.size)
    out = {}
    for h, r in acc.items():
        ag = r["agree"] / max(r["total"], 1)
        out[h] = {"max_abs_prob_diff": r["max_abs_diff"], "decision_agreement": ag, "decisions": r["total"],
                  "pass": bool(r["max_abs_diff"] < rule["tol"] and ag >= rule["agree"])}
    return out


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", required=True)
    ap.add_argument("--heads", required=True, help="train_pool.py run dir with vad_head.pt and ipa_head.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", required=True, help="prep_clips.py output: calib/ and accept/")
    ap.add_argument("--try-dtypes", default="int8-static,int8-weights,float16", help="reduced head precisions to try, smallest first")
    ap.add_argument("--accept-tol", type=float, default=0.01)
    ap.add_argument("--accept-agree", type=float, default=0.995)
    a = ap.parse_args()
    rule = {"tol": a.accept_tol, "agree": a.accept_agree}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "build-intermediate"
    work.mkdir(exist_ok=True)

    paths = {h: Path(a.heads) / f"{h}_head.pt" for h in HEAD_ORDER}
    loaded = {h: load_head(paths[h]) for h in HEAD_ORDER}
    heads = {h: v[0] for h, v in loaded.items()}
    sides = {h: v[1] for h, v in loaded.items()}
    ipa_meta = loaded["ipa"][2]
    vocab = list(ipa_meta["symbols"])
    n_out = {h: heads[h].o.out_channels for h in HEAD_ORDER}
    assert n_out == {"vad": 1, "ipa": len(vocab)}, (n_out, len(vocab))
    assert sides["vad"] == 0, sides

    trunk = onnx.load(a.int8)
    loc = locate(trunk.graph)
    print(json.dumps({"stem": loc["stem"], "residual_adds": loc["residual_adds"], "out_conv": loc["out_conv"],
                      "tap_quantize_outputs": [t[0] for t in loc["taps"]], "mel_log": loc["mel_log"], "mel_norm": loc["mel_norm"]}, indent=1))

    tail32_path = work / "tail_float32.onnx"
    tail32 = export_tail(heads, tail32_path)

    def write(tail, tag):
        full = merge(trunk, tail, loc)
        onnx.checker.check_model(full, full_check=True)
        fp = work / f"full_{tag}.onnx"
        onnx.save(full, str(fp))
        feat = prune(full, ["features"])
        onnx.checker.check_model(feat, full_check=True)
        onnx.save(feat, str(work / f"features_{tag}.onnx"))
        return full, feat, fp

    def build(choice, tag):
        q = [h for h in HEAD_ORDER if choice[h] == "int8-static"]
        tail = int8_heads(tail32_path, work / f"tail_{tag}_q.onnx", reader, q) if q else tail32
        wq = [h for h in HEAD_ORDER if choice[h] == "int8-weights"]
        if wq:
            tail = int8_weight_heads(tail, wq)
        f = [h for h in HEAD_ORDER if choice[h] == "float16"]
        if f:
            tail = float16_heads(tail, f)
        return write(tail, tag)

    builds = {"float32": write(tail32, "float32")}
    calib, accept = clip_dir(Path(a.clips) / "calib"), clip_dir(Path(a.clips) / "accept")
    m1, io = trunk_with_taps(trunk, loc)
    onnx.save(m1, str(work / "trunk_with_taps.onnx"))
    reader = TailInputs(work / "trunk_with_taps.onnx", io, calib)

    trials, chosen = {}, {}
    dtypes = [d for d in a.try_dtypes.split(",") if d]
    for h in HEAD_ORDER:
        chosen[h] = "float32"
        for d in dtypes:
            choice = {x: "float32" for x in HEAD_ORDER}
            choice[h] = d
            tag = f"{h}_{d}"
            _, _, fp = build(choice, tag)
            r = acceptance(builds["float32"][2], fp, accept, [h], rule)[h]
            trials[tag] = {"head": h, "dtype": d, **r, "full_file_bytes": fp.stat().st_size}
            print("TRIAL", tag, json.dumps(trials[tag]), flush=True)
            if r["pass"]:
                chosen[h] = d
                break
    final = build(chosen, "final") if any(v != "float32" for v in chosen.values()) else builds["float32"]
    final_check = acceptance(builds["float32"][2], final[2], accept, list(HEAD_ORDER), rule) if final is not builds["float32"] else None
    if final_check is not None:
        print("FINAL_CHECK", json.dumps(final_check), flush=True)
        assert all(r["pass"] for r in final_check.values()), "combined build fails the acceptance rule"
    full, feat, _ = final
    full_path, feat_path = out / "wakephonehubert_int8.onnx", out / "wakephonehubert_features_int8.onnx"
    onnx.save(full, str(full_path))
    onnx.save(feat, str(feat_path))
    onnx.save(builds["float32"][0], str(work / "reference_float32_heads.onnx"))

    hub_cfg = json.loads(Path(__import__("huggingface_hub").hf_hub_download(TRUNK_REPO, "config.json", revision=TRUNK_REVISION)).read_text())
    side_frames = {h: (2 * sum(heads[h].side.dil) if heads[h].side is not None else 0) for h in HEAD_ORDER}
    adapter_frames = max(heads[h].c1.kernel_size[0] - 1 + heads[h].c2.kernel_size[0] - 1 for h in HEAD_ORDER)
    rf = max(hub_cfg["receptive_field_samples"] + adapter_frames * HOP,
             max(side_frames.values()) * HOP + adapter_frames * HOP + 400)
    frames = {"frame_rate_hz": 50.0, "hop_samples": HOP}
    layout, pos = {"hubert": [0, OUT_DIM]}, OUT_DIM
    for h in HEAD_ORDER:
        layout[h] = [pos, pos + n_out[h]]
        pos += n_out[h]
    fdim = pos
    desc = {"vad": "active-speaker probability (sigmoid)",
            "ipa": "CTC posteriors (softmax) over the vocab ids; index 0 <pad> is the blank, indices 1-3 (<s>, </s>, <unk>) "
                   "are dropped when decoding; trained with a 5-frame (100 ms) teacher delay"}
    outputs = {
        "hubert": {"shape": ["batch", "frames", str(OUT_DIM)], **frames, "description": "WakeHuBERT-tiny int8 features, bit-identical to wakehubert_int8.onnx"},
        "layers": {"shape": ["batch", "frames", str(N_TAPS), str(TAP_DIM)], **frames, "description": "dequantised int8 trunk taps: stem output then the 8 block outputs, after ReLU"},
        "logmel": {"shape": ["batch", "mel_frames", str(N_MELS)], "frame_rate_hz": 100.0, "hop_samples": HOP // 2, "description": "BatchNorm-normalised log-mel, 2 frames per feature frame"},
        "mfcc": {"shape": ["batch", "mel_frames", str(N_MFCC)], "frame_rate_hz": 100.0, "hop_samples": HOP // 2, "description": "orthonormal DCT-II of the un-normalised log-mel"},
    }
    for h in HEAD_ORDER:
        outputs[h] = {"shape": ["batch", "frames", str(n_out[h])], **frames, "description": desc[h]}
    outputs["features"] = {"shape": ["batch", "frames", str(fdim)], **frames, "description": ", ".join(["hubert", *HEAD_ORDER]) + " joined on the last axis"}
    vocab_doc = {"symbols": vocab, "blank_index": int(ipa_meta["blank"]), "dropped_ids": list(ipa_meta["dropped_ids"]),
                 "size": len(vocab), "teacher_tokenizer": "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
                 "delay_frames": int(ipa_meta["delay"])}
    cfg = {
        "family": "WakePhoneHuBERT",
        "architecture": "frozen-int8-trunk+tap-heads(+mel-tcn side branch)",
        "input": hub_cfg["input"],
        "output": {"name": "features", "shape": ["batch", "frames", str(fdim)], **frames, "layout": layout},
        "outputs": outputs,
        "feature_dim": fdim,
        "streaming": True,
        "causal": True,
        "delay_ms": hub_cfg["delay_ms"],
        "receptive_field_samples": rf,
        "trunk": {"repo": TRUNK_REPO, "revision": TRUNK_REVISION, "file": Path(a.int8).name, "sha256": sha256(a.int8),
                  "tap_dim": TAP_DIM, "taps": N_TAPS, "n_mels": N_MELS, "n_mfcc": N_MFCC,
                  "receptive_field_samples": hub_cfg["receptive_field_samples"]},
        "heads": {h: {"hidden": heads[h].c1.out_channels, "side_branch_channels": sides[h], "n_out": n_out[h],
                      "activation": ACTIVATION[h], "params": sum(p.numel() for p in heads[h].parameters()),
                      "dtype": chosen[h], "source_checkpoint": str(paths[h]), "sha256": sha256(paths[h]),
                      "side_branch_lookback_frames": side_frames[h],
                      "mix_weights": torch.softmax(heads[h].mix, 0).tolist()} for h in HEAD_ORDER},
        "heads_dtype": chosen,
        "heads_dtype_trials": trials,
        "heads_dtype_final_check": final_check,
        "acceptance_rule": rule,
        "head_adapter_context_frames": adapter_frames,
        "vocab": vocab_doc,
        "files": {"int8": full_path.name, "features_int8": feat_path.name, "vocab": "vocab.json"},
        "license": hub_cfg["license"],
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "vocab.json").write_text(json.dumps(vocab_doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print("WROTE", json.dumps(chosen), full_path, full_path.stat().st_size, feat_path, feat_path.stat().st_size)


if __name__ == "__main__":
    main()
