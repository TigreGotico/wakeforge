"""Streaming WakeHuBERT-tiny trunk: one audio block and a state in, that block's feature frames and the next state out.

The trunk (``MelTCNStudent`` in ``tinyhubert.py``, published as TigreGotico/wakehubert-tiny) is causal: feature frame
t depends only on samples before 320 (t + 1). Every layer that looks into the past keeps exactly the past it needs,
and nothing else is carried:

- the log-mel DFT, a 400-sample window at a 160-sample hop: the last 240 samples;
- the stride-2 stem, kernel 4 over normalised log-mel frames: the last 2 frames;
- each depthwise-separable block, kernel k and dilation d: its last (k - 1) d input frames.

A block is a whole number of 320-sample feature hops, so it holds an even number of log-mel frames and every stem
output starts on an even log-mel frame: the stem's phase is identical at every block boundary, and the 2-frame stem
buffer is the whole of its state. The zero state equals the zero padding the trunk applies on the left, so streaming
a signal from the zero state gives the frames of the trunk run once over the whole signal.

The plugin's isolated windows (1.5 s featurized from a zero start every block) differ from these frames because the
trunk's receptive field (2.5 s) is longer than the window; ``compare`` measures that difference per frame position.

    python stream_trunk.py export --weights model.safetensors --out-dir out --calib-dir LibriSpeech/dev-clean \\
        --noise-dir noise-audioset
    python stream_trunk.py compare --weights model.safetensors --stream-onnx out/stream.onnx \\
        --stream-int8 out/stream_int8.onnx --published-onnx wakehubert.onnx --published-int8 wakehubert_int8.onnx \\
        --speech-dir test-clean --noise-dir noise-audioset
    python stream_trunk.py bench --stream-int8 out/stream_int8.onnx --published-int8 wakehubert_int8.onnx
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn

SR = 16000
HOP = 320
BLOCK = 1280
WINDOW_FRAMES = 75
OPSET = 17

_TH = Path(__file__).resolve().parents[1] / "tinyhubert.py"


def _tinyhubert():
    spec = importlib.util.spec_from_file_location("tinyhubert", _TH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Mel(nn.Module):
    """``CausalLogMel`` over the carried samples and the new block, without padding."""

    def __init__(self, mel):
        super().__init__()
        self.register_buffer("basis", mel.basis)
        self.register_buffer("fb", mel.mel)
        self.ctx, self.hop, self.n_freq = mel.pad, mel.hop, mel.n_freq

    def forward(self, state, wav):
        w = torch.cat([state, wav], 1)
        spec = F.conv1d(w.unsqueeze(1), self.basis, stride=self.hop)
        power = spec[:, :self.n_freq].pow(2) + spec[:, self.n_freq:].pow(2)
        return torch.log(torch.matmul(self.fb, power) + 1e-6), w[:, -self.ctx:]


class _Stem(nn.Module):
    def __init__(self, stem, bn):
        super().__init__()
        self.conv, self.bn, self.ctx = stem.conv, bn, stem.pad

    def forward(self, state, m):
        s = torch.cat([state, m], 2)
        return F.relu(self.bn(self.conv(s))), s[..., -self.ctx:]


class _Block(nn.Module):
    def __init__(self, block):
        super().__init__()
        if not isinstance(block.dw, nn.Conv1d):
            raise ValueError("mixed-kernel blocks are not streamed")
        self.dw, self.bn1, self.pw, self.bn2, self.ctx = block.dw, block.bn1, block.pw, block.bn2, block.pad

    def forward(self, state, x):
        xin = torch.cat([state, x], 2)
        y = F.relu(self.bn1(self.dw(xin)))
        return F.relu(self.bn2(self.pw(y)) + x), xin[..., -self.ctx:]


class StreamingTrunk(nn.Module):
    """The trunk's own weights, called one block at a time.

    ``forward(waveform [B, block], *state)`` returns ``(features [B, block // 320, D], *next_state)``; the state is
    ``init_state()`` at the start of a stream and after a reset.
    """

    def __init__(self, trunk, block=BLOCK):
        super().__init__()
        if block <= 0 or block % HOP:
            raise ValueError(f"block {block}: must be a positive multiple of {HOP} samples")
        if getattr(trunk, "seq", None):
            raise ValueError("a trunk with sequence layers carries more than convolution buffers")
        if trunk.stem.conv.stride[0] * trunk.mel.hop != HOP:
            raise ValueError("the stem must take the log-mel hop to the 320-sample feature hop")
        self.block = block
        self.mel = _Mel(trunk.mel)
        self.norm = trunk.norm
        self.stem = _Stem(trunk.stem, trunk.stem_bn)
        self.blocks = nn.ModuleList(_Block(b) for b in trunk.blocks)
        self.out = trunk.out
        if min(self.mel.ctx, self.stem.ctx, *(b.ctx for b in self.blocks)) <= 0:
            raise ValueError("every streamed layer needs a past to carry")
        self.state_names = ("wav_state", "stem_state", *(f"block{i}_state" for i in range(len(self.blocks))))

    def state_shapes(self, batch=1):
        ch = self.stem.conv.out_channels
        return [(batch, self.mel.ctx), (batch, self.stem.conv.in_channels, self.stem.ctx),
                *((batch, ch, b.ctx) for b in self.blocks)]

    def init_state(self, batch=1):
        return [torch.zeros(s) for s in self.state_shapes(batch)]

    def forward(self, wav, *state):
        m, wav_state = self.mel(state[0], wav)
        x, stem_state = self.stem(state[1], self.norm(m))
        nxt = [wav_state, stem_state]
        for blk, s in zip(self.blocks, state[2:]):
            x, s = blk(s, x)
            nxt.append(s)
        return (self.out(x).transpose(1, 2), *nxt)

    @torch.no_grad()
    def stream(self, wav, state=None):
        """Features of every whole block of ``wav`` [B, N] and the state after the last one."""
        state = list(state) if state is not None else self.init_state(wav.shape[0])
        feats = []
        for i in range(wav.shape[1] // self.block):
            f, *state = self(wav[:, i * self.block:(i + 1) * self.block], *state)
            feats.append(f)
        return torch.cat(feats, 1), state


def load_trunk(weights, config=None):
    """The published trunk as ``MelTCNStudent`` from ``model.safetensors`` and the ``config.json`` beside it."""
    from safetensors.torch import load_file

    weights = Path(weights)
    args = json.loads(Path(config or weights.with_name("config.json")).read_text())["pytorch"]["init_args"]
    trunk = _tinyhubert().MelTCNStudent(**args, n_targets=3)
    missing, unexpected = trunk.load_state_dict(load_file(str(weights)), strict=False)
    if unexpected or any(not k.startswith("proj.") for k in missing):
        raise ValueError(f"weights do not match the trunk: missing {missing}, unexpected {unexpected}")
    return trunk.eval()


def macs_per_block(streamer):
    """Multiply-adds of one block, by layer: convolutions, the mel matmul, the DFT power and the batch norms.

    Batch norms after a convolution fold into it on export and cost nothing; the input batch norm follows a log
    and costs one multiply-add per value. ReLU, log and the residual adds are not multiply-adds and are not counted.
    """
    mel_frames = streamer.block // streamer.mel.hop
    frames = streamer.block // HOP
    n_bins, n_fft = streamer.mel.basis.shape[0], streamer.mel.basis.shape[2]
    n_mels = streamer.mel.fb.shape[0]
    st = streamer.stem.conv
    layers = {"dft": mel_frames * n_bins * n_fft,
              "power": mel_frames * n_bins,
              "mel_filterbank": mel_frames * n_mels * streamer.mel.n_freq,
              "input_norm": mel_frames * n_mels,
              "stem": frames * st.out_channels * st.in_channels * st.kernel_size[0]}
    for i, b in enumerate(streamer.blocks):
        layers[f"block{i}_dw"] = frames * b.dw.out_channels * b.dw.kernel_size[0]
        layers[f"block{i}_pw"] = frames * b.pw.out_channels * b.pw.in_channels
    layers["out"] = frames * streamer.out.out_channels * streamer.out.in_channels
    return layers


def export_onnx(streamer, path):
    """Stateful float32 graph: inputs ``waveform`` and every state, outputs ``features`` and every ``<state>_out``."""
    streamer = streamer.eval()
    names_out = ["features", *(f"{n}_out" for n in streamer.state_names)]
    with torch.no_grad():
        torch.onnx.export(streamer, (torch.zeros(1, streamer.block), *streamer.init_state()), str(path),
                          input_names=["waveform", *streamer.state_names], output_names=names_out,
                          opset_version=OPSET, dynamo=False)
    return path


class OnnxStream:
    """Runs a stateful export block by block, carrying its state."""

    def __init__(self, path, threads=1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        ins = self.sess.get_inputs()
        self.block = ins[0].shape[1] if isinstance(ins[0].shape[1], int) else BLOCK
        self.names = [i.name for i in ins]
        self.shapes = [i.shape for i in ins[1:]]
        self.reset()

    def reset(self):
        self.state = [np.zeros(s, np.float32) for s in self.shapes]

    def feeds(self, block):
        return dict(zip(self.names, [block[None].astype(np.float32), *self.state]))

    def push(self, block):
        feats, *self.state = self.sess.run(None, self.feeds(block))
        return feats[0]

    def run(self, wav):
        return np.concatenate([self.push(wav[i * self.block:(i + 1) * self.block])
                               for i in range(len(wav) // self.block)])


def quantize_int8(fp32, int8, feeds):
    """Static int8 with the published trunk's recipe: QDQ, per-channel int8 weights, uint8 activations, the log-mel
    front end and its input batch norm kept in float. ``feeds`` are the graph inputs of calibration blocks."""
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = Path(int8).with_suffix(".pre.onnx")
    quant_pre_process(str(fp32), str(pre))
    keep_float = [n.name for n in onnx.load(str(pre)).graph.node if "/mel/" in n.name or "/norm/" in n.name]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(feeds)

        def get_next(self):
            return next(self.it, None)

    quantize_static(str(pre), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8, nodes_to_exclude=keep_float)
    pre.unlink()
    return keep_float


def _state_name(pad_node, n):
    """``wav_state``, ``stem_state`` and ``block<i>_state`` for the trunk's own pads, ``state<n>`` otherwise."""
    parts = pad_node.split("/")
    if "mel" in parts:
        return "wav_state"
    if "stem" in parts:
        return "stem_state"
    block = next((p for p in parts if p.startswith("blocks.")), None)
    return f"block{block.split('.')[1]}_state" if block else f"state{n}"


def _pad_facts(model, block):
    """For every ``Pad``: its input's shape, its pads and its constant value, read by running the graph on one
    block, so pads computed inside the graph are read too."""
    import onnx
    import onnxruntime as ort

    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    pads = [n for n in probe.graph.node if n.op_type == "Pad"]
    names = sorted({x for n in pads for x in n.input[:3] if x})
    known = {o.name for o in probe.graph.output}
    inits = {i.name for i in probe.graph.initializer}
    probe.graph.output.extend(onnx.helper.make_empty_tensor_value_info(x) for x in names
                              if x not in known and x not in inits)
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    sess = ort.InferenceSession(probe.SerializeToString(), so, providers=["CPUExecutionProvider"])
    fetch = [x for x in names if x not in inits]
    vals = dict(zip(fetch, sess.run(fetch, {sess.get_inputs()[0].name: np.zeros((1, block), np.float32)})))
    vals.update({i.name: onnx.numpy_helper.to_array(i) for i in probe.graph.initializer if i.name in names})
    return {n.name: (list(vals[n.input[0]].shape), vals[n.input[1]].tolist(),
                     float(vals[n.input[2]]) if len(n.input) > 2 and n.input[2] else 0.0) for n in pads}


def streamify_onnx(src, dst, block=BLOCK, source_revision=""):
    """Rewrite a causal trunk ONNX file (float or int8, as exported, with left zero ``Pad`` nodes) into the stateful
    streaming graph, keeping every weight and quantization parameter as they are.

    Each ``Pad`` that adds k zeros on the left of the time axis becomes a ``Concat`` of a state input ``[1, C, k]``
    with its input, and the state output is the last k steps of that ``Concat``. Inputs: ``waveform`` [1, samples],
    samples a multiple of 320, then the states in graph order; outputs: ``features`` then ``<state>_out``. The
    metadata records the source file's sha256, its revision, the block and the state shapes. Returns the state
    shapes by name.
    """
    import hashlib

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(src))
    g = model.graph
    facts = _pad_facts(model, block)
    dims = g.input[0].type.tensor_type.shape.dim
    dims[0].Clear()
    dims[0].dim_value = 1
    nodes, states = [], {}
    for n in g.node:
        if n.op_type != "Pad":
            nodes.append(n)
            continue
        in_shape, pads, fill = facts[n.name]
        rank = len(pads) // 2
        k = pads[rank - 1]
        mode = next((x.s for x in n.attribute if x.name == "mode"), b"constant")
        if k <= 0 or any(p for j, p in enumerate(pads) if j != rank - 1) or fill or mode != b"constant":
            raise ValueError(f"{n.name}: only a left zero pad on the last axis streams, got pads {pads}")
        name = _state_name(n.name, len(states))
        shape = in_shape[:-1] + [k]
        g.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
        nodes.append(helper.make_node("Concat", [name, n.input[0]], [n.output[0]], axis=rank - 1, name=f"{n.name}_stream"))
        consts_in = [f"{name}_{x}" for x in ("starts", "ends", "axes")]
        g.initializer.extend(numpy_helper.from_array(np.array([v], np.int64), c)
                             for v, c in zip((-k, np.iinfo(np.int64).max, rank - 1), consts_in))
        nodes.append(helper.make_node("Slice", [n.output[0], *consts_in], [f"{name}_out"], name=f"{n.name}_state"))
        g.output.append(helper.make_tensor_value_info(f"{name}_out", TensorProto.FLOAT, shape))
        states[name] = shape
    if not states:
        raise ValueError(f"{src}: no left Pad to stream")
    del g.node[:]
    g.node.extend(nodes)
    dims[1].Clear()
    dims[1].dim_param = "samples"
    g.output[0].type.tensor_type.shape.dim[1].Clear()
    g.output[0].type.tensor_type.shape.dim[1].dim_param = "frames"
    meta = {"streaming": "stateful", "source_sha256": hashlib.sha256(Path(src).read_bytes()).hexdigest(),
            "source_revision": source_revision, "block_samples": str(block), "hop_samples": str(HOP),
            "states": json.dumps(states)}
    for k, v in meta.items():
        e = model.metadata_props.add()
        e.key, e.value = k, v
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    return states


def cosine(a, b):
    return np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def _audio(files):
    for f in files:
        x, sr = sf.read(str(f), dtype="float32", always_2d=True)
        if sr != SR:
            raise ValueError(f"{f}: {sr} Hz, expected {SR}")
        yield x.mean(1)


def _files(d, exts=(".flac", ".wav")):
    return sorted(p for p in Path(d).rglob("*") if p.suffix in exts)


def speech_plus_noise(speech_dir, noise_dir, seconds, snr_db, seed):
    """``seconds`` of consecutive utterances from ``speech_dir`` with a noise file looped under them at ``snr_db``,
    peak at most 0.9. ``snr_db`` None draws one SNR from 0 to 20 dB and leaves half the signals clean."""
    rng = np.random.default_rng(seed)
    speech = _files(speech_dir)
    n = int(seconds * SR)
    parts, got = [], 0
    for i in rng.permutation(len(speech)):
        x = next(_audio([speech[i]]))
        parts.append(x)
        got += len(x)
        if got >= n:
            break
    s = np.concatenate(parts)[:n]
    if snr_db is None and rng.random() < 0.5:
        return (s * 0.9 / max(np.abs(s).max(), 1e-6)).astype(np.float32)
    snr = rng.uniform(0, 20) if snr_db is None else snr_db
    noise_files = _files(noise_dir)
    z = next(_audio([noise_files[rng.integers(len(noise_files))]]))
    z = np.tile(z, n // len(z) + 1)[:n]
    z *= np.sqrt(np.mean(s ** 2) / (np.mean(z ** 2) + 1e-12) / 10 ** (snr / 10))
    y = s + z
    return (y * 0.9 / max(np.abs(y).max(), 1e-6)).astype(np.float32)


def cmd_streamify(a):
    states = streamify_onnx(a.src, a.out, a.block, a.source_revision)
    wav = speech_plus_noise(a.speech_dir, a.noise_dir, a.seconds, a.snr_db, seed=a.seed)
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    whole = ort.InferenceSession(a.src, so, providers=["CPUExecutionProvider"]).run(None, {"waveform": wav[None]})[0][0]
    got = OnnxStream(a.out).run(wav)
    d = np.abs(got - whole[:len(got)]).max(1)
    c = cosine(got, whole[:len(got)])
    res = {"states": states, "signal_seconds": len(wav) / SR, "frames": len(got),
           "max_abs_vs_whole": float(d.max()), "frames_within_1e-4": int((d <= 1e-4).sum()),
           "cosine_min": float(c.min()), "cosine_mean": float(c.mean()), "feature_abs_max": float(np.abs(whole).max())}
    print(json.dumps(res))


def cmd_export(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    streamer = StreamingTrunk(load_trunk(a.weights), a.block)
    fp32 = export_onnx(streamer, out / "stream.onnx")
    f = OnnxStream(fp32)
    feeds = []
    for k in range(a.calib_signals):
        wav = speech_plus_noise(a.calib_dir, a.noise_dir, a.calib_seconds, None, seed=1000 + k)
        f.reset()
        for i in range(len(wav) // f.block):
            blk = wav[i * f.block:(i + 1) * f.block]
            if i % a.calib_every == 0:
                feeds.append(f.feeds(blk))
            f.push(blk)
    keep = quantize_int8(fp32, out / "stream_int8.onnx", feeds)
    info = {"block_samples": a.block, "opset": OPSET, "state_inputs": dict(zip(streamer.state_names,
                                                                               streamer.state_shapes())),
            "calibration_blocks": len(feeds), "int8_float_nodes": len(keep),
            "fp32_bytes": fp32.stat().st_size, "int8_bytes": (out / "stream_int8.onnx").stat().st_size}
    (out / "stream.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))


def _isolated(sess, wav, block, window):
    """The plugin's isolated windows: after every block, the last ``window`` samples (zero-filled before the start)
    featurized on their own. Returns ``[n_blocks, window // 320, D]``."""
    name = sess.get_inputs()[0].name
    buf = np.concatenate([np.zeros(window, np.float32), wav])
    return np.stack([sess.run(None, {name: buf[None, i * block + block:i * block + block + window]})[0][0]
                     for i in range(len(wav) // block)])


def _per_position(iso, stream, block, window, start):
    """Cosine and relative L2 between isolated windows and the streamed frames at the same time, per position in the
    window, over every window that ends at or after ``start`` samples."""
    fpb, wf = block // HOP, window // HOP
    cos, rel = [], []
    for i in range(len(iso)):
        end = (i + 1) * fpb
        if (i + 1) * block < start:
            continue
        ref = stream[end - wf:end]
        cos.append(cosine(iso[i], ref))
        rel.append(np.linalg.norm(iso[i] - ref, axis=-1) / (np.linalg.norm(ref, axis=-1) + 1e-12))
    return np.array(cos), np.array(rel)


def cmd_compare(a):
    import onnxruntime as ort

    wav = speech_plus_noise(a.speech_dir, a.noise_dir, a.seconds, a.snr_db, seed=a.seed)
    trunk = load_trunk(a.weights)
    streamer = StreamingTrunk(trunk, a.block).eval()
    x = torch.from_numpy(wav)[None]
    with torch.no_grad():
        whole = trunk(x)[0].numpy()
        torch_stream = streamer.stream(x)[0][0].numpy()
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    pub = ort.InferenceSession(a.published_onnx, so, providers=["CPUExecutionProvider"])
    pub_whole = pub.run(None, {pub.get_inputs()[0].name: wav[None]})[0][0]
    onnx_stream = OnnxStream(a.stream_onnx).run(wav)
    int8_stream = OnnxStream(a.stream_int8).run(wav)
    n = len(torch_stream)
    res = {"signal_seconds": len(wav) / SR, "frames": n,
           "torch_stream_vs_torch_whole_max_abs": float(np.abs(torch_stream - whole[:n]).max()),
           "onnx_stream_vs_torch_whole_max_abs": float(np.abs(onnx_stream - whole[:n]).max()),
           "onnx_stream_vs_published_whole_max_abs": float(np.abs(onnx_stream - pub_whole[:n]).max()),
           "feature_abs_max": float(np.abs(whole).max())}
    c = cosine(onnx_stream, int8_stream)
    res["int8_vs_float_stream_cosine"] = {"mean": float(c.mean()), "p01": float(np.quantile(c, 0.01)),
                                          "min": float(c.min())}
    window = WINDOW_FRAMES * HOP
    start = a.receptive_seconds * SR
    for tag, path, stream in (("float", a.published_onnx, onnx_stream), ("int8", a.published_int8, int8_stream)):
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        iso = _isolated(sess, wav, a.block, window)
        cos, rel = _per_position(iso, stream, a.block, window, start)
        res[f"isolated_{tag}_vs_stream_{tag}"] = {
            "windows": len(cos),
            "cosine_mean_by_position": [round(float(v), 4) for v in cos.mean(0)],
            "cosine_p05_by_position": [round(float(v), 4) for v in np.quantile(cos, 0.05, axis=0)],
            "rel_l2_mean_by_position": [round(float(v), 4) for v in rel.mean(0)]}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("isolated")}, indent=1))


def cmd_bench(a):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    iso = ort.InferenceSession(a.published_int8, so, providers=["CPUExecutionProvider"])
    iso_name = iso.get_inputs()[0].name
    st = OnnxStream(a.stream_int8)
    rng = np.random.default_rng(0)
    window = (0.1 * rng.standard_normal(WINDOW_FRAMES * HOP)).astype(np.float32)
    block = window[-st.block:]
    for _ in range(20):
        st.push(block)
        iso.run(None, {iso_name: window[None]})
    t_st, t_iso = [], []
    for _ in range(a.rounds):
        t0 = time.perf_counter()
        st.push(block)
        t1 = time.perf_counter()
        iso.run(None, {iso_name: window[None]})
        t2 = time.perf_counter()
        t_st.append(t1 - t0)
        t_iso.append(t2 - t1)
    q = lambda t: {k: round(float(np.quantile(t, p)) * 1e3, 4) for k, p in (("p10_ms", .1), ("median_ms", .5),
                                                                              ("p90_ms", .9))}
    res = {"rounds": a.rounds, "stream_int8": q(t_st), "isolated_int8": q(t_iso),
           "median_ratio": float(np.median(t_iso) / np.median(t_st))}
    print(json.dumps(res))


def cmd_macs(a):
    tm = _tinyhubert()
    args = json.loads(Path(a.config).read_text())["pytorch"]["init_args"]
    streamer = StreamingTrunk(tm.MelTCNStudent(**args, n_targets=3), a.block)
    layers = macs_per_block(streamer)
    per_block = sum(layers.values())
    front = layers["dft"] + layers["power"] + layers["mel_filterbank"] + layers["input_norm"]
    blocks_per_s = SR / a.block
    iso = macs_per_block(StreamingTrunk(tm.MelTCNStudent(**args, n_targets=3), WINDOW_FRAMES * HOP))
    print(json.dumps({"per_block": per_block, "per_second": per_block * blocks_per_s,
                      "front_end_per_block": front, "after_front_end_per_block": per_block - front,
                      "isolated_window_per_block": sum(iso.values()),
                      "isolated_window_per_second": sum(iso.values()) * blocks_per_s, "layers": layers}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="float32 and int8 stateful ONNX")
    e.add_argument("--weights", required=True, help="model.safetensors, with config.json beside it")
    e.add_argument("--out-dir", required=True)
    e.add_argument("--calib-dir", required=True, help="speech for int8 calibration (16 kHz flac/wav, searched recursively)")
    e.add_argument("--noise-dir", required=True)
    e.add_argument("--calib-signals", type=int, default=48)
    e.add_argument("--calib-seconds", type=float, default=8.0)
    e.add_argument("--calib-every", type=int, default=2, help="calibrate on every n-th block of each signal")
    c = sub.add_parser("compare", help="streaming against whole-signal and against isolated windows")
    c.add_argument("--weights", required=True)
    c.add_argument("--stream-onnx", required=True)
    c.add_argument("--stream-int8", required=True)
    c.add_argument("--published-onnx", required=True)
    c.add_argument("--published-int8", required=True)
    c.add_argument("--speech-dir", required=True)
    c.add_argument("--noise-dir", required=True)
    c.add_argument("--seconds", type=float, default=60.0)
    c.add_argument("--snr-db", type=float, default=10.0)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--receptive-seconds", type=float, default=2.5,
                   help="compare isolated windows only once the stream has this much history")
    c.add_argument("--out", default="compare.json")
    b = sub.add_parser("bench", help="per-block time, streaming int8 against the isolated int8 window, one thread")
    b.add_argument("--stream-int8", required=True)
    b.add_argument("--published-int8", required=True)
    b.add_argument("--rounds", type=int, default=2000)
    m = sub.add_parser("macs", help="multiply-adds per block")
    m.add_argument("--config", required=True, help="the published config.json (one with pytorch.init_args)")
    s = sub.add_parser("streamify", help="rewrite a published trunk ONNX into the stateful streaming graph")
    s.add_argument("--src", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--source-revision", default="")
    s.add_argument("--speech-dir", required=True, help="speech for the parity check against the source graph")
    s.add_argument("--noise-dir", required=True)
    s.add_argument("--seconds", type=float, default=60.0)
    s.add_argument("--snr-db", type=float, default=10.0)
    s.add_argument("--seed", type=int, default=0)
    for p in (e, c, m, s):
        p.add_argument("--block", type=int, default=BLOCK)
    a = ap.parse_args(argv)
    {"streamify": cmd_streamify, "export": cmd_export, "compare": cmd_compare, "bench": cmd_bench, "macs": cmd_macs}[a.cmd](a)


if __name__ == "__main__":
    main()
