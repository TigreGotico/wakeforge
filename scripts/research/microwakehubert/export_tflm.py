"""WakeHuBERT-tiny streamed trunk and a GRU head as int8 TensorFlow Lite Micro models for the ESP32-S3.

The log-mel front end runs in C (``firmware/main/logmel.c``) in float32; the rest of the trunk (stem, eight
dilated depthwise-separable blocks, output projection) is one int8 ``trunk.tflite`` that takes the normalised
log-mel frames of one 80 ms block and every block's state, and returns the block's four feature frames and the
next state. The state is carried as plain model inputs and outputs, laid out ``[1, frames, 1, channels]`` on
both sides, so the firmware copies (or requantizes) each output into the matching input between calls.

A dilated depthwise convolution is written as an undilated one whose kernel has zeros between the taps, so the
optimised esp-nn depthwise kernel runs it; the zeros cost at most 33 multiply-adds per channel and frame.

The head is a ready GRU head of the plugin (``models/wakehubert_<word>.onnx`` of OpenVoiceOS/wakehubert-wakewords),
run from a zero state over the last 75 streamed trunk frames every block, as the plugin runs it over a 1.5 s
window; the plugin featurizes that window on its own, so the head here sees features with more history than it was
trained on. TFLite has no GRU op, so the recurrence is unrolled into
FULLY_CONNECTED, LOGISTIC, TANH, MUL, ADD and SUB. Its output is the logit: an int8 probability saturates at
255/256. The decision is the plugin's rule for a ready model: probability ``sigmoid(calib_a * logit + calib_b)`` at or
above the head's ``default_threshold`` (all three from the head's metadata), one block suffices, then the stream is
reset and the next 2.0 s (25 blocks) are not scored.

    python export_tflm.py export --stream-onnx stream.onnx --head-onnx wakehubert_hey_jarvis.onnx \\
        --calib-dir dev-clean --noise-dir noise --positives-dir hey_jarvis --out-dir out --firmware-dir firmware/main
    python export_tflm.py check --out-dir out --stream-onnx stream.onnx --stream-int8 stream_int8.onnx \\
        --head-onnx wakehubert_hey_jarvis.onnx --speech-dir test-clean --noise-dir noise --positives-dir hey_jarvis \\
        --frontend-lib out/liblogmel.so
    python export_tflm.py audio --out-dir out --speech-dir test-clean --noise-dir noise \\
        --positives-dir hey_jarvis --frontend-lib out/liblogmel.so --firmware-dir firmware/main \\
        --head-onnx wakehubert_hey_jarvis.onnx
"""
import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
BLOCK = 1280
HOP = 320
MEL_HOP = 160
N_FFT = 400
CTX = N_FFT - MEL_HOP
MEL_FRAMES = BLOCK // MEL_HOP
FRAMES = BLOCK // HOP
WINDOW = 75
QUIET_BLOCKS = 25
DILATIONS = (1, 2, 4, 8, 1, 2, 4, 8)


def _onnx_weights(path):
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(path))
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    return m, inits


class TrunkWeights:
    """Float weights of the streamed trunk, read from the stateful fp32 ``stream.onnx`` (batch norms folded)."""

    def __init__(self, path):
        m, w = _onnx_weights(path)
        conv = {n.name: n for n in m.graph.node if n.op_type == "Conv"}
        bn = next(n for n in m.graph.node if n.op_type == "BatchNormalization")
        eps = next((a.f for a in bn.attribute if a.name == "epsilon"), 1e-5)
        g, b, mu, var = (w[k] for k in bn.input[1:])
        self.norm_scale = (g / np.sqrt(var + eps)).astype(np.float32)
        self.norm_shift = (b - mu * self.norm_scale).astype(np.float32)
        self.basis = w["mel.basis"][:, 0].astype(np.float32)
        self.fb = w["mel.fb"].astype(np.float32)
        log_eps = [n for n in m.graph.node if n.name == "/mel/Add_1"][0].input[1]
        const = {n.output[0]: n for n in m.graph.node if n.op_type == "Constant"}
        self.log_eps = float(onnx_const(const[log_eps]))
        self.window = np.sqrt(self.basis[0] ** 2 + self.basis[self.basis.shape[0] // 2] ** 2)

        def cw(name):
            n = conv[name]
            return w[n.input[1]], (w[n.input[2]] if len(n.input) > 2 else None)

        self.stem = cw("/stem/conv/Conv")
        self.blocks = []
        for i, d in enumerate(DILATIONS):
            dw, dwb = cw(f"/blocks.{i}/dw/Conv")
            dil = [a.ints[0] for a in conv[f"/blocks.{i}/dw/Conv"].attribute if a.name == "dilations"][0]
            if dil != d:
                raise ValueError(f"block {i}: dilation {dil}, expected {d}")
            self.blocks.append((dw, dwb, d, *cw(f"/blocks.{i}/pw/Conv")))
        self.out = cw("/out/Conv")
        self.ctx = [(b[0].shape[-1] - 1) * b[2] for b in self.blocks]
        self.channels = self.stem[0].shape[0]
        self.n_mels = self.fb.shape[0]
        self.feature_dim = self.out[0].shape[0]


def onnx_const(node):
    from onnx import numpy_helper

    return numpy_helper.to_array(node.attribute[0].t)


def logmel(tw, wav_ctx):
    """Normalised log-mel of ``[CTX + BLOCK]`` samples (state then block): ``[MEL_FRAMES, n_mels]`` float32."""
    idx = np.arange(MEL_FRAMES)[:, None] * MEL_HOP + np.arange(N_FFT)[None]
    spec = wav_ctx[idx] @ tw.basis.T
    nf = tw.basis.shape[0] // 2
    power = spec[:, :nf] ** 2 + spec[:, nf:] ** 2
    m = np.log(power @ tw.fb.T + tw.log_eps)
    return (m * tw.norm_scale + tw.norm_shift).astype(np.float32)


class Frontend:
    """The firmware's C log-mel (``logmel.c``) through ctypes, carrying its 240-sample state."""

    def __init__(self, lib, n_mels=64):
        self.lib = ctypes.CDLL(str(lib))
        self.n_mels = n_mels
        self.state = (ctypes.c_float * CTX)()
        self.lib.mwh_logmel_block.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                             ctypes.POINTER(ctypes.c_float)]
        self.lib.mwh_logmel_init()

    def reset(self):
        ctypes.memset(self.state, 0, CTX * 4)

    def push(self, block):
        block = np.ascontiguousarray(block, np.float32)
        out = np.zeros((MEL_FRAMES, self.n_mels), np.float32)
        self.lib.mwh_logmel_block(self.state, block.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                  out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return out


def expand_dilated(w, d):
    """Depthwise kernel ``[C, 1, k]`` at dilation ``d`` as an undilated ``[(k - 1) d + 1, C]`` kernel with zeros
    between the taps."""
    e = np.zeros(((w.shape[-1] - 1) * d + 1, w.shape[0]), np.float32)
    e[::d] = w[:, 0].T
    return e


def trunk_step(tw, mel, states):
    """The graph of ``build_trunk_fn`` in numpy, ``[T, C]`` tensors: (``[4, D]`` features, next states)."""
    x = np.concatenate([states[0], mel])
    nxt = [x[-2:]]
    w, b = tw.stem
    k = w.shape[-1]
    y = np.maximum(np.stack([np.einsum("oik,ki->o", w, x[2 * t:2 * t + k]) for t in range(FRAMES)]) + b, 0)
    for (dw, dwb, d, pw, pwb), s, c in zip(tw.blocks, states[1:], tw.ctx):
        xin = np.concatenate([s, y])
        nxt.append(xin[-c:])
        e = expand_dilated(dw, d)
        h = np.maximum(np.stack([(e * xin[t:t + len(e)]).sum(0) for t in range(FRAMES)]) + dwb, 0)
        y = np.maximum(h @ pw[:, :, 0].T + pwb + y, 0)
    return y @ tw.out[0][:, :, 0].T + tw.out[1], nxt


def zero_states(tw):
    return [np.zeros((2, tw.n_mels), np.float32), *(np.zeros((c, tw.channels), np.float32) for c in tw.ctx)]


def build_trunk_fn(tw):
    """A ``tf.function`` of the post-mel trunk with ``[1, T, 1, C]`` tensors throughout."""
    import tensorflow as tf

    def conv_w(w):
        return tf.constant(np.transpose(w, (2, 1, 0))[:, None].astype(np.float32))

    def dw_w(w, d):
        return tf.constant(expand_dilated(w, d)[:, None, :, None])

    stem_w, stem_b = conv_w(tw.stem[0]), tf.constant(tw.stem[1])
    blocks = [(dw_w(dw, d), tf.constant(dwb), conv_w(pw), tf.constant(pwb))
              for dw, dwb, d, pw, pwb in tw.blocks]
    out_w, out_b = conv_w(tw.out[0]), tf.constant(tw.out[1])
    specs = [tf.TensorSpec([1, MEL_FRAMES, 1, tw.n_mels], tf.float32, name="mel"),
             tf.TensorSpec([1, 2, 1, tw.n_mels], tf.float32, name="stem_state")]
    specs += [tf.TensorSpec([1, c, 1, tw.channels], tf.float32, name=f"block{i}_state")
              for i, c in enumerate(tw.ctx)]

    @tf.function(input_signature=specs)
    def trunk(mel, stem_state, *states):
        x = tf.concat([stem_state, mel], 1)
        nxt = {"stem_state_out": x[:, -2:]}
        y = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, stem_w, (1, 2, 1, 1), "VALID"), stem_b))
        for i, ((dw, dwb, pw, pwb), s, c) in enumerate(zip(blocks, states, tw.ctx)):
            xin = tf.concat([s, y], 1)
            nxt[f"block{i}_state_out"] = xin[:, -c:]
            h = tf.nn.relu(tf.nn.bias_add(tf.nn.depthwise_conv2d(xin, dw, (1, 1, 1, 1), "VALID"), dwb))
            y = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h, pw, (1, 1, 1, 1), "VALID"), pwb) + y)
        feats = tf.nn.bias_add(tf.nn.conv2d(y, out_w, (1, 1, 1, 1), "VALID"), out_b)
        return {"features": feats, **nxt}

    return trunk


class GruWeights:
    """A plugin GRU head (``gru.onnx``: ONNX GRU, linear_before_reset, mean over frames, fc1, ReLU, fc2)."""

    def __init__(self, path):
        m, w = _onnx_weights(path)
        g = next(n for n in m.graph.node if n.op_type == "GRU")
        if not any(a.name == "linear_before_reset" and a.i == 1 for a in g.attribute):
            raise ValueError("the head's GRU must use linear_before_reset=1 (PyTorch)")
        self.W, self.R = w[g.input[1]][0], w[g.input[2]][0]
        b = w[g.input[3]][0]
        h = self.R.shape[1]
        self.Wb, self.Rb = b[:3 * h], b[3 * h:]
        named = {k.split("/", 1)[-1]: v for k, v in w.items()}
        self.fc1 = (named["head.fc1.weight"], named["head.fc1.bias"])
        self.fc2 = (named["head.fc2.weight"], named["head.fc2.bias"])
        self.hidden = h
        meta = {p.key: p.value for p in m.metadata_props}
        self.calib_a = float(meta.get("calib_a", 1.0))
        self.calib_b = float(meta.get("calib_b", 0.0))
        self.threshold = float(meta.get("default_threshold", 0.5))

    def logit(self, feats):
        """Float reference over ``[T, D]``: the plugin head's logit (gate order z, r, h as in ONNX)."""
        h = np.zeros(self.hidden, np.float32)
        acc = np.zeros(self.hidden, np.float32)
        H = self.hidden
        for x in feats:
            gi = self.W @ x + self.Wb
            gh = self.R @ h + self.Rb
            z = 1 / (1 + np.exp(-(gi[:H] + gh[:H])))
            r = 1 / (1 + np.exp(-(gi[H:2 * H] + gh[H:2 * H])))
            n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
            h = n + z * (h - n)
            acc += h
        y = np.maximum(self.fc1[0] @ (acc / len(feats)) + self.fc1[1], 0)
        return float((self.fc2[0] @ y + self.fc2[1])[0])


def build_head_fn(gw, dim, window=WINDOW):
    import tensorflow as tf

    H = gw.hidden
    W, Wb = tf.constant(gw.W.T.astype(np.float32)), tf.constant(gw.Wb.astype(np.float32))
    Rz, Rr, Rn = (tf.constant(gw.R[i * H:(i + 1) * H].T.astype(np.float32)) for i in range(3))
    bz, br, bn = (tf.constant(gw.Rb[i * H:(i + 1) * H].astype(np.float32)) for i in range(3))
    f1w, f1b = tf.constant(gw.fc1[0].T.astype(np.float32)), tf.constant(gw.fc1[1].astype(np.float32))
    f2w, f2b = tf.constant(gw.fc2[0].T.astype(np.float32)), tf.constant(gw.fc2[1].astype(np.float32))

    @tf.function(input_signature=[tf.TensorSpec([window, dim], tf.float32, name="features")])
    def head(feats):
        gi = tf.matmul(feats, W) + Wb
        gz, gr, gn = gi[:, :H], gi[:, H:2 * H], gi[:, 2 * H:]
        h = tf.zeros([1, H])
        acc = tf.zeros([1, H])
        for t in range(window):
            if t == 0:
                hz, hr, hn = bz[None], br[None], bn[None]
            else:
                hz, hr, hn = tf.matmul(h, Rz) + bz, tf.matmul(h, Rr) + br, tf.matmul(h, Rn) + bn
            z = tf.sigmoid(gz[t:t + 1] + hz)
            r = tf.sigmoid(gr[t:t + 1] + hr)
            n = tf.tanh(gn[t:t + 1] + r * hn)
            h = n + z * (h - n) if t else n - z * n
            acc = acc + h
        y = tf.nn.relu(tf.matmul(acc * (1.0 / window), f1w) + f1b)
        return {"logit": tf.matmul(y, f2w) + f2b}

    return head


def _files(d, exts=(".flac", ".wav")):
    return sorted(p for p in Path(d).rglob("*") if p.suffix in exts)


def _read(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{path}: {sr} Hz, expected {SR}")
    return x.mean(1)


def mix(speech_files, noise_files, seconds, rng, snr=None, inserts=()):
    """``seconds`` of consecutive utterances with noise looped under them at an SNR from 0 to 20 dB (half of the
    signals clean when ``snr`` is None), and every ``(clip, at_seconds)`` of ``inserts`` placed over the speech
    after the mix is scaled; peak at most 0.9."""
    n = int(seconds * SR)
    parts, got = [], 0
    for i in rng.permutation(len(speech_files)):
        x = _read(speech_files[i])
        parts.append(x)
        got += len(x)
        if got >= n:
            break
    s = np.concatenate(parts)[:n]
    s = s * 0.3 / max(np.abs(s).max(), 1e-6)
    for clip, at in inserts:
        a = int(at * SR)
        c = clip[:n - a] * 0.6 / max(np.abs(clip).max(), 1e-6)
        s[a:a + len(c)] = c + 0.1 * s[a:a + len(c)]
    if snr is None and rng.random() < 0.5:
        y = s
    else:
        snr = rng.uniform(0, 20) if snr is None else snr
        z = _read(noise_files[rng.integers(len(noise_files))])
        z = np.tile(z, n // len(z) + 1)[:n]
        z *= np.sqrt(np.mean(s ** 2) / (np.mean(z ** 2) + 1e-12) / 10 ** (snr / 10))
        y = s + z
    return (y * 0.9 / max(np.abs(y).max(), 1e-6)).astype(np.float32)


class OnnxStream:
    """The stateful fp32 or int8 ``stream.onnx`` run block by block; also hands back the inputs it was fed."""

    def __init__(self, path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inputs = self.sess.get_inputs()
        self.reset()

    def reset(self):
        self.state = [np.zeros(i.shape, np.float32) for i in self.inputs[1:]]

    def push(self, block):
        fed = list(self.state)
        feats, *self.state = self.sess.run(None, dict(zip([i.name for i in self.inputs],
                                                          [block[None].astype(np.float32), *self.state])))
        return feats[0], fed


def _nchw_to_tflm(s):
    """``[1, C, T]`` ONNX state to ``[1, T, 1, C]``."""
    return np.transpose(s, (0, 2, 1))[:, :, None]


def calibration_samples(tw, stream, speech, noise, signals, seconds, every, seed):
    """Float trunk inputs (normalised log-mel and the states the fp32 stream carried) for int8 calibration."""
    rng = np.random.default_rng(seed)
    out, refs = [], []
    for _ in range(signals):
        wav = mix(speech, noise, seconds, rng)
        stream.reset()
        tail = np.zeros(CTX, np.float32)
        for i in range(len(wav) // BLOCK):
            blk = wav[i * BLOCK:(i + 1) * BLOCK]
            w = np.concatenate([tail, blk])
            tail = w[-CTX:]
            feats, fed = stream.push(blk)
            if i % every == 0:
                mel = logmel(tw, w)[None, :, None]
                out.append([mel, *(_nchw_to_tflm(s) for s in fed[1:])])
                refs.append(feats)
    return out, refs


def convert(fn, samples):
    """Full-integer int8 through a SavedModel, so the ``.tflite`` keeps a signature naming every input and output."""
    import tempfile

    import tensorflow as tf

    mod = tf.Module()
    mod.f = fn
    tmp = tempfile.mkdtemp(prefix="mwh-savedmodel-")
    tf.saved_model.save(mod, tmp, signatures=fn.get_concrete_function())
    conv = tf.lite.TFLiteConverter.from_saved_model(tmp)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    names = [t.name for t in fn.get_concrete_function().structured_input_signature[0]]
    conv.representative_dataset = lambda: ({n: v.astype(np.float32) for n, v in zip(names, sample)}
                                           for sample in samples)
    return conv.convert()


class Tflm:
    """A ``.tflite`` under TensorFlow Lite Micro's own kernels (the ``tflite-micro`` Python runtime), or under the
    TFLite interpreter with ``micro=False``; inputs and outputs addressed by name through the signature."""

    def __init__(self, path, micro=True):
        import tensorflow as tf

        it = tf.lite.Interpreter(model_path=str(path))
        sig = it.get_signature_list()["serving_default"]
        runner = it.get_signature_runner()
        in_idx = [d["index"] for d in it.get_input_details()]
        out_idx = [d["index"] for d in it.get_output_details()]
        ins, outs = runner.get_input_details(), runner.get_output_details()
        self.in_pos = {n: in_idx.index(d["index"]) for n, d in ins.items()}
        self.out_pos = {n: out_idx.index(d["index"]) for n, d in outs.items()}
        self.in_q = {n: (d["quantization"], d["shape"]) for n, d in ins.items()}
        self.out_q = {n: (d["quantization"], d["shape"]) for n, d in outs.items()}
        assert sorted(sig["inputs"]) == sorted(ins) and sorted(sig["outputs"]) == sorted(outs)
        self.micro = micro
        if micro:
            from tflite_micro.python.tflite_micro import runtime

            self.it = runtime.Interpreter.from_file(str(path), arena_size=4 << 20)
        else:
            self.it = tf.lite.Interpreter(model_path=str(path))
            self.it.allocate_tensors()

    def run(self, feeds):
        """``feeds``: name to int8 array. Returns name to int8 array."""
        if self.micro:
            for n, v in feeds.items():
                self.it.set_input(v.astype(np.int8), self.in_pos[n])
            self.it.invoke()
            return {n: self.it.get_output(p).copy() for n, p in self.out_pos.items()}
        dets = self.it.get_input_details()
        for n, v in feeds.items():
            self.it.set_tensor(dets[self.in_pos[n]]["index"], v.astype(np.int8))
        self.it.invoke()
        od = self.it.get_output_details()
        return {n: self.it.get_tensor(od[p]["index"]).copy() for n, p in self.out_pos.items()}


def q(x, qp):
    s, z = qp
    return np.clip(np.round(x / s) + z, -128, 127).astype(np.int8)


def dq(x, qp):
    s, z = qp
    return (x.astype(np.float32) - z) * s


def requant(x, src, dst):
    return x if src == dst else q(dq(x, src), dst)


class DeviceRun:
    """The firmware's loop on the host: C log-mel, int8 trunk with state carry, a ring of 75 frames, int8 head."""

    def __init__(self, out_dir, frontend, micro=True):
        out_dir = Path(out_dir)
        self.trunk = Tflm(out_dir / "trunk.tflite", micro)
        self.head = Tflm(out_dir / "head.tflite", micro)
        self.fe = frontend
        self.state_names = [n for n in self.trunk.in_q if n != "mel"]
        self.reset()

    def reset(self):
        self.fe.reset()
        self.state = {n: np.full(self.trunk.in_q[n][1], q(0.0, self.trunk.in_q[n][0]), np.int8)
                      for n in self.state_names}
        fq = self.head.in_q["features"]
        self.ring = np.full(fq[1], q(0.0, fq[0]), np.int8)

    def push(self, block):
        """One 80 ms block: (``[4, D]`` int8 features, the head's logit)."""
        mel = self.fe.push(block)[None, :, None]
        out = self.trunk.run({"mel": q(mel, self.trunk.in_q["mel"][0]), **self.state})
        for n in self.state_names:
            self.state[n] = requant(out[n + "_out"], self.trunk.out_q[n + "_out"][0], self.trunk.in_q[n][0])
        f = out["features"].reshape(FRAMES, -1)
        fh = requant(f, self.trunk.out_q["features"][0], self.head.in_q["features"][0])
        self.ring = np.concatenate([self.ring[FRAMES:], fh])
        p = self.head.run({"features": self.ring})["logit"]
        return f, float(dq(p, self.head.out_q["logit"][0]).ravel()[0])


def fnv1a(b):
    h = 2166136261
    for x in bytes(b):
        h = ((h ^ x) * 16777619) & 0xFFFFFFFF
    return h


def c_array(name, data, align=16, ctype="unsigned char"):
    data = bytes(data)
    rows = [", ".join(f"0x{v:02x}" for v in data[i:i + 16]) for i in range(0, len(data), 16)]
    return (f"alignas({align}) const {ctype} {name}[] = {{\n  " + ",\n  ".join(rows) +
            f"\n}};\nconst unsigned int {name}_len = {len(data)};\n")


def write_model_sources(out_dir, fw, gw):
    """``model_data.cc`` (both models as aligned arrays) and ``model_io.h`` (tensor positions and quantization)."""
    out_dir, fw = Path(out_dir), Path(fw)
    trunk = Tflm(out_dir / "trunk.tflite", micro=False)
    head = Tflm(out_dir / "head.tflite", micro=False)
    src = ["#include <cstdalign>", '#include "model_data.h"']
    for n in ("trunk", "head"):
        src.append(c_array(f"g_{n}_model", (out_dir / f"{n}.tflite").read_bytes()))
    (fw / "model_data.cc").write_text("\n".join(src))
    (fw / "model_data.h").write_text(
        "#pragma once\n" + "".join(f"extern const unsigned char g_{n}_model[];\nextern const unsigned int "
                                   f"g_{n}_model_len;\n" for n in ("trunk", "head")))
    states = [n for n in trunk.in_q if n != "mel"]
    order = ["stem_state", *sorted((n for n in states if n != "stem_state"), key=lambda s: int(s[5:-6]))]
    lines = ["#pragma once", "#include <stdint.h>", f"#define MWH_N_STATES {len(order)}",
             f"#define MWH_FRAMES {FRAMES}", f"#define MWH_FEATURE_DIM {trunk.out_q['features'][1][-1]}",
             f"#define MWH_WINDOW {head.in_q['features'][1][0]}",
             f"#define MWH_MEL_IN {trunk.in_pos['mel']}", f"#define MWH_FEATURES_OUT {trunk.out_pos['features']}",
             f"#define MWH_HEAD_IN {head.in_pos['features']}", f"#define MWH_HEAD_OUT {head.out_pos['logit']}",
             "typedef struct { int in; int out; int bytes; float in_scale; int in_zp; float out_scale; int out_zp; }"
             " mwh_state_io_t;", "static const mwh_state_io_t MWH_STATES[MWH_N_STATES] = {"]
    for n in order:
        (si, zi), shp = trunk.in_q[n]
        (so, zo), _ = trunk.out_q[n + "_out"]
        lines.append(f"  {{{trunk.in_pos[n]}, {trunk.out_pos[n + '_out']}, {int(np.prod(shp))}, {si!r}f, {zi}, "
                     f"{so!r}f, {zo}}},  /* {n} */")
    lines.append("};")
    (ms, mz), _ = trunk.in_q["mel"]
    (fs, fz), _ = trunk.out_q["features"]
    (hs, hz), _ = head.in_q["features"]
    (ps, pz), _ = head.out_q["logit"]
    lines += [f"#define MWH_MEL_SCALE {ms!r}f", f"#define MWH_MEL_ZP {mz}",
              f"#define MWH_FEAT_SCALE {fs!r}f", f"#define MWH_FEAT_ZP {fz}",
              f"#define MWH_HEAD_IN_SCALE {hs!r}f", f"#define MWH_HEAD_IN_ZP {hz}",
              f"#define MWH_LOGIT_SCALE {ps!r}f", f"#define MWH_LOGIT_ZP {pz}",
              f"#define MWH_CALIB_A {gw.calib_a!r}f", f"#define MWH_CALIB_B {gw.calib_b!r}f",
              f"#define MWH_THRESHOLD {gw.threshold!r}f", ""]
    (fw / "model_io.h").write_text("\n".join(lines))


def write_frontend_consts(tw, fw):
    def arr(name, a):
        a = np.asarray(a, np.float32).ravel()
        body = ",\n  ".join(", ".join(f"{v!r}f" for v in a[i:i + 8]) for i in range(0, len(a), 8))
        return f"static const float {name}[{len(a)}] = {{\n  {body}\n}};\n"

    nf = tw.basis.shape[0] // 2
    fb = tw.fb
    lo = [int(np.nonzero(r)[0].min()) if r.any() else 0 for r in fb]
    hi = [int(np.nonzero(r)[0].max()) + 1 if r.any() else 0 for r in fb]
    flat = np.concatenate([fb[i, lo[i]:hi[i]] for i in range(len(fb))])
    txt = ["#pragma once", f"#define MWH_N_FFT {N_FFT}", f"#define MWH_N_BINS {nf}", f"#define MWH_N_MELS {len(fb)}",
           f"#define MWH_MEL_HOP {MEL_HOP}", f"#define MWH_MEL_FRAMES {MEL_FRAMES}", f"#define MWH_CTX {CTX}",
           f"#define MWH_BLOCK {BLOCK}", f"#define MWH_LOG_EPS {tw.log_eps!r}f",
           arr("MWH_WINDOW_FN", tw.window), arr("MWH_FB", flat),
           f"static const short MWH_FB_LO[{len(lo)}] = {{{', '.join(map(str, lo))}}};",
           f"static const short MWH_FB_HI[{len(hi)}] = {{{', '.join(map(str, hi))}}};",
           arr("MWH_NORM_SCALE", tw.norm_scale), arr("MWH_NORM_SHIFT", tw.norm_shift)]
    (Path(fw) / "logmel_consts.h").write_text("\n".join(txt))


def cmd_export(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tw = TrunkWeights(a.stream_onnx)
    write_frontend_consts(tw, a.firmware_dir)
    speech, noise = _files(a.calib_dir), _files(a.noise_dir)
    stream = OnnxStream(a.stream_onnx)
    samples, refs = calibration_samples(tw, stream, speech, noise, a.calib_signals, a.calib_seconds, a.calib_every,
                                        1000)
    fn = build_trunk_fn(tw)
    names = ["mel", "stem_state", *(f"block{i}_state" for i in range(len(tw.ctx)))]
    float_err = max(float(np.abs(fn(*s)["features"].numpy()[0, :, 0] - r).max()) for s, r in zip(samples[:64], refs))
    (out / "trunk.tflite").write_bytes(convert(fn, samples))
    gw = GruWeights(a.head_onnx)
    head_samples = head_calibration(tw, gw, a, speech, noise)
    (out / "head.tflite").write_bytes(convert(build_head_fn(gw, tw.feature_dim), head_samples))
    write_model_sources(out, a.firmware_dir, gw)
    info = {"trunk_bytes": (out / "trunk.tflite").stat().st_size, "head_bytes": (out / "head.tflite").stat().st_size,
            "trunk_calibration_blocks": len(samples), "head_calibration_windows": len(head_samples),
            "inputs": names,
            "float_graph_vs_onnx_fp32_max_abs_64_blocks": float_err}
    (out / "export.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))


def float_features(stream, wav):
    stream.reset()
    return np.concatenate([stream.push(wav[i * BLOCK:(i + 1) * BLOCK])[0] for i in range(len(wav) // BLOCK)])


def head_calibration(tw, gw, a, speech, noise):
    """75-frame windows of fp32 streamed features over speech with and without the wake word."""
    rng = np.random.default_rng(2000)
    pos = _files(a.positives_dir)
    stream = OnnxStream(a.stream_onnx)
    out = []
    for k in range(a.head_signals):
        ins = [(_read(pos[rng.integers(len(pos))]), 2.0 + 3.0 * j) for j in range(2)] if k % 2 else []
        f = float_features(stream, mix(speech, noise, 9.0, rng, inserts=ins))
        for end in range(WINDOW, len(f) + 1, 8):
            out.append([f[end - WINDOW:end]])
    return out


def cmd_check(a):
    """Front end C vs numpy; int8 trunk under TFLM vs the fp32 and the ONNX int8 streams; head float vs int8."""
    tw = TrunkWeights(a.stream_onnx)
    gw = GruWeights(a.head_onnx)
    fe = Frontend(a.frontend_lib, tw.n_mels)
    speech, noise, pos = _files(a.speech_dir), _files(a.noise_dir), _files(a.positives_dir)
    rng = np.random.default_rng(a.seed)
    res = {"signals": a.signals, "seconds_each": a.seconds}
    cos_tflm, cos_onnx8, cos_onnx8_vs_fp = [], [], []
    tflm_vs_tflite_cos, head_rows = [], []
    fp, o8 = OnnxStream(a.stream_onnx), OnnxStream(a.stream_int8)
    dev, lite = DeviceRun(a.out_dir, fe, micro=True), DeviceRun(a.out_dir, Frontend(a.frontend_lib, tw.n_mels),
                                                             micro=False)
    for k in range(a.signals):
        ins = [(_read(pos[rng.integers(len(pos))]), 3.0)] if k % 2 else []
        wav = mix(speech, noise, a.seconds, rng, snr=a.snr_db, inserts=ins)
        fp.reset(), o8.reset(), dev.reset(), lite.reset()
        fps, f8s, devs, probs = [], [], [], []
        for i in range(len(wav) // BLOCK):
            blk = wav[i * BLOCK:(i + 1) * BLOCK]
            dev_feats, p = dev.push(blk)
            lf, _ = lite.push(blk)
            tflm_vs_tflite_cos.append(_cos(dq(lf, lite.trunk.out_q["features"][0]),
                                           dq(dev_feats, dev.trunk.out_q["features"][0])))
            fps.append(fp.push(blk)[0])
            f8s.append(o8.push(blk)[0])
            devs.append(dq(dev_feats, dev.trunk.out_q["features"][0]))
            probs.append(p)
        fpa, f8a, dva = np.concatenate(fps), np.concatenate(f8s), np.concatenate(devs)
        cos_tflm.append(_cos(dva, fpa))
        cos_onnx8.append(_cos(dva, f8a))
        cos_onnx8_vs_fp.append(_cos(f8a, fpa))
        ends = range(-(-WINDOW // FRAMES) * FRAMES, len(fpa) + 1, FRAMES)
        fl = np.array([gw.logit(fpa[e - WINDOW:e]) for e in ends])
        dl = np.array([probs[e // FRAMES - 1] for e in ends])
        head_rows.append({"wake_word_inserted": bool(ins), "float_logit_max": round(float(fl.max()), 3),
                          "device_logit_max": round(float(dl.max()), 3),
                          "device_vs_float_logit_abs_median": round(float(np.median(np.abs(dl - fl))), 3),
                          "device_vs_float_logit_abs_max": round(float(np.abs(dl - fl).max()), 3)})
    c = lambda xs: {"mean": float(np.mean(np.concatenate(xs))), "p01": float(np.quantile(np.concatenate(xs), 0.01)),
                    "min": float(np.min(np.concatenate(xs)))}
    res["frontend_c_vs_numpy_max_abs"] = frontend_error(tw, a.frontend_lib, speech, noise, a.seed)
    res["tflm_int8_vs_onnx_fp32_cosine"] = c(cos_tflm)
    res["tflm_int8_vs_onnx_int8_cosine"] = c(cos_onnx8)
    res["onnx_int8_vs_onnx_fp32_cosine"] = c(cos_onnx8_vs_fp)
    res["tflm_vs_tflite_interpreter_cosine"] = c(tflm_vs_tflite_cos)
    res["head"] = head_rows
    Path(a.out_dir, "check.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def _cos(a, b):
    return np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def frontend_error(tw, lib, speech, noise, seed):
    fe = Frontend(lib, tw.n_mels)
    wav = mix(speech, noise, 20.0, np.random.default_rng(seed + 7), snr=10.0)
    tail = np.zeros(CTX, np.float32)
    err = 0.0
    for i in range(len(wav) // BLOCK):
        blk = wav[i * BLOCK:(i + 1) * BLOCK]
        w = np.concatenate([tail, blk])
        tail = w[-CTX:]
        err = max(err, float(np.abs(fe.push(blk) - logmel(tw, w)).max()))
    return err


def cmd_audio(a):
    """The firmware's fixed test audio (int16, with the wake word at known times) and the host's expected trace."""
    speech, noise, pos = _files(a.speech_dir), _files(a.noise_dir), _files(a.positives_dir)
    rng = np.random.default_rng(a.seed)
    at = [float(t) for t in a.wake_at.split(",")]
    clips = [pos[i] for i in rng.choice(len(pos), len(at), replace=False)]
    wav = mix(speech, noise, a.seconds, rng, snr=a.snr_db, inserts=[(_read(c), t) for c, t in zip(clips, at)])
    pcm = np.round(wav * 32767).astype(np.int16)
    fw = Path(a.firmware_dir)
    (fw / "test_audio.cc").write_text('#include <cstdalign>\n#include "test_audio.h"\n' +
                                      c_array("g_test_audio", pcm.tobytes(), ctype="unsigned char"))
    (fw / "test_audio.h").write_text(f"#pragma once\n#define MWH_TEST_AUDIO_SAMPLES {len(pcm)}\n"
                                     "extern const unsigned char g_test_audio[];\n")
    sf.write(str(Path(a.out_dir) / "test_audio.wav"), pcm, SR)
    x = pcm.astype(np.float32) / 32768.0
    dev = DeviceRun(a.out_dir, Frontend(a.frontend_lib), micro=True)
    gw = GruWeights(a.head_onnx)
    rows, wakes, quiet = [], [], 0
    for i in range(len(x) // BLOCK):
        f, logit = dev.push(x[i * BLOCK:(i + 1) * BLOCK])
        p = 1 / (1 + np.exp(-(gw.calib_a * logit + gw.calib_b)))
        rows.append({"block": i, "fnv": f"{fnv1a(f.tobytes()):08x}", "logit": round(logit, 3), "p": round(p, 4)})
        if quiet > 0:
            quiet -= 1
        elif p >= gw.threshold:
            wakes.append({"block": i, "seconds": round((i + 1) * BLOCK / SR, 2), "p": round(p, 4)})
            dev.reset()
            quiet = QUIET_BLOCKS
    trace = {"wake_word_at_seconds": at, "clips": [c.name for c in clips], "threshold": gw.threshold,
             "calib_a": gw.calib_a, "calib_b": gw.calib_b, "wakes": wakes, "blocks": rows}
    Path(a.out_dir, "expected_trace.json").write_text(json.dumps(trace, indent=1) + "\n")
    print(json.dumps({k: v for k, v in trace.items() if k != "blocks"}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--stream-onnx", required=True, help="the stateful fp32 stream.onnx of stream_trunk.py export")
    e.add_argument("--head-onnx", required=True, help="a plugin GRU head gru.onnx")
    e.add_argument("--calib-dir", required=True)
    e.add_argument("--noise-dir", required=True)
    e.add_argument("--positives-dir", required=True, help="wake-word clips for the head's calibration windows")
    e.add_argument("--out-dir", required=True)
    e.add_argument("--firmware-dir", required=True, help="firmware/main, where the generated sources go")
    e.add_argument("--calib-signals", type=int, default=48)
    e.add_argument("--calib-seconds", type=float, default=8.0)
    e.add_argument("--calib-every", type=int, default=2)
    e.add_argument("--head-signals", type=int, default=24)
    c = sub.add_parser("check")
    for p in (c,):
        p.add_argument("--out-dir", required=True)
        p.add_argument("--stream-onnx", required=True)
        p.add_argument("--stream-int8", required=True, help="the int8 stream_int8.onnx of stream_trunk.py export")
        p.add_argument("--head-onnx", required=True)
        p.add_argument("--speech-dir", required=True)
        p.add_argument("--noise-dir", required=True)
        p.add_argument("--positives-dir", required=True)
        p.add_argument("--frontend-lib", required=True, help="logmel.c built as a shared library")
        p.add_argument("--signals", type=int, default=6)
        p.add_argument("--seconds", type=float, default=12.0)
        p.add_argument("--snr-db", type=float, default=10.0)
        p.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("audio")
    d.add_argument("--out-dir", required=True)
    d.add_argument("--speech-dir", required=True)
    d.add_argument("--noise-dir", required=True)
    d.add_argument("--positives-dir", required=True)
    d.add_argument("--frontend-lib", required=True)
    d.add_argument("--firmware-dir", required=True)
    d.add_argument("--seconds", type=float, default=12.0)
    d.add_argument("--wake-at", default="4.0,9.0", help="seconds at which wake-word clips are placed")
    d.add_argument("--snr-db", type=float, default=15.0)
    d.add_argument("--seed", type=int, default=3)
    d.add_argument("--head-onnx", required=True, help="the head the models were exported from (its calibration)")
    a = ap.parse_args(argv)
    {"export": cmd_export, "check": cmd_check, "audio": cmd_audio}[a.cmd](a)


if __name__ == "__main__":
    main()
