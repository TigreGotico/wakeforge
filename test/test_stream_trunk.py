"""Streaming WakeHuBERT-tiny trunk (scripts/research/microwakehubert/stream_trunk.py): block-by-block frames equal
the whole-signal causal trunk, the state keeps fixed shapes, a reset restarts the stream, the stateful ONNX export
round-trips, and each broken state carry (a short dilation buffer, a stem buffer off by one log-mel frame or dropped,
the waveform overlap dropped, the state reset every block) breaks the equality check.

The trunk has the published architecture with random weights and random batch-norm statistics, so nothing is
downloaded."""
import importlib.util
import types
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

_P = Path(__file__).resolve().parents[1] / "scripts" / "research" / "microwakehubert" / "stream_trunk.py"


def _load(source=None):
    if source is None:
        spec = importlib.util.spec_from_file_location("stream_trunk", _P)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    mod = types.ModuleType("stream_trunk_mutant")
    mod.__file__ = str(_P)
    exec(compile(source, str(_P), "exec"), mod.__dict__)
    return mod


st = _load()
SECONDS = 4


def _trunk(mod=st, seed=0):
    torch.manual_seed(seed)
    t = mod._tinyhubert().MelTCNStudent(channels=256, blocks=8, feature_dim=128, n_mels=64, kernel=5, n_targets=3)
    with torch.no_grad():
        for m in t.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.running_mean.normal_(0, 0.2)
                m.running_var.uniform_(0.5, 2.0)
                m.weight.uniform_(0.5, 1.5)
                m.bias.normal_(0, 0.1)
        t.norm.running_mean.fill_(-6.0)
        t.norm.running_var.fill_(16.0)
    return t.eval()


def _signal(seconds=SECONDS, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = seconds * st.SR
    t = torch.arange(n) / st.SR
    tone = torch.sin(2 * torch.pi * (200 + 300 * t) * t) * (0.5 + 0.5 * torch.sin(2 * torch.pi * 1.3 * t))
    return (0.3 * tone + 0.05 * torch.randn(n, generator=g))[None].float()


def _whole(trunk, wav):
    with torch.no_grad():
        return trunk(wav)


def _stream_error(mod, wav):
    trunk = _trunk(mod)
    whole = _whole(trunk, wav)
    got, _ = mod.StreamingTrunk(trunk).eval().stream(wav)
    return float((got - whole[:, :got.shape[1]]).abs().max() / whole.abs().max())


def test_stream_equals_whole_signal():
    wav = _signal()
    trunk = _trunk()
    whole = _whole(trunk, wav)
    got, _ = st.StreamingTrunk(trunk).eval().stream(wav)
    assert got.shape == whole.shape == (1, SECONDS * 50, 128)
    assert float((got - whole).abs().max() / whole.abs().max()) < 1e-5


def test_state_shapes_fixed():
    s = st.StreamingTrunk(_trunk()).eval()
    shapes = s.state_shapes()
    assert shapes[0] == (1, 240)
    assert shapes[1] == (1, 64, 2)
    assert [sh[2] for sh in shapes[2:]] == [4 * d for d in (1, 2, 4, 8, 1, 2, 4, 8)]
    wav, state = _signal(), s.init_state()
    with torch.no_grad():
        for i in range(wav.shape[1] // st.BLOCK):
            f, *state = s(wav[:, i * st.BLOCK:(i + 1) * st.BLOCK], *state)
            assert f.shape == (1, 4, 128)
            assert [tuple(x.shape) for x in state] == shapes


def test_reset_restarts_the_stream():
    trunk = _trunk()
    s = st.StreamingTrunk(trunk).eval()
    a, b = _signal(seed=1), _signal(seed=2)
    _, carried = s.stream(a)
    fresh = _whole(trunk, b)
    after_reset, _ = s.stream(b, s.init_state())
    without_reset, _ = s.stream(b, carried)
    scale = fresh.abs().max()
    assert float((after_reset - fresh).abs().max() / scale) < 1e-5
    assert float((without_reset - fresh).abs().max() / scale) > 1e-2


def test_block_must_be_whole_feature_hops():
    with pytest.raises(ValueError):
        st.StreamingTrunk(_trunk(), block=1440)
    s = st.StreamingTrunk(_trunk(), block=640).eval()
    wav = _signal()
    whole = _whole(_trunk(), wav)
    got, _ = s.stream(wav)
    assert float((got - whole).abs().max() / whole.abs().max()) < 1e-5


def test_onnx_round_trip(tmp_path):
    trunk = _trunk()
    s = st.StreamingTrunk(trunk).eval()
    path = st.export_onnx(s, tmp_path / "stream.onnx")
    g = onnx.load(str(path)).graph
    assert [i.name for i in g.input] == ["waveform", *s.state_names]
    assert [o.name for o in g.output] == ["features", *(f"{n}_out" for n in s.state_names)]
    dims = lambda v: tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
    assert [dims(i) for i in g.input[1:]] == s.state_shapes()
    assert [dims(o) for o in g.output[1:]] == s.state_shapes()
    wav = _signal()
    whole = _whole(trunk, wav)[0].numpy()
    got = st.OnnxStream(path).run(wav[0].numpy())
    assert got.shape == whole.shape
    assert float(np.abs(got - whole).max() / np.abs(whole).max()) < 1e-5


def test_int8_export_keeps_front_end_float_and_state(tmp_path):
    s = st.StreamingTrunk(_trunk()).eval()
    fp32 = st.export_onnx(s, tmp_path / "stream.onnx")
    f = st.OnnxStream(fp32)
    wav = _signal()[0].numpy()
    feeds = []
    for i in range(len(wav) // st.BLOCK):
        feeds.append(f.feeds(wav[i * st.BLOCK:(i + 1) * st.BLOCK]))
        f.push(wav[i * st.BLOCK:(i + 1) * st.BLOCK])
    keep = st.quantize_int8(fp32, tmp_path / "stream_int8.onnx", feeds)
    assert any("/mel/" in n for n in keep) and any("/norm/" in n for n in keep)
    g = onnx.load(str(tmp_path / "stream_int8.onnx")).graph
    assert sum(n.op_type == "QuantizeLinear" for n in g.node) > 10
    q = st.OnnxStream(tmp_path / "stream_int8.onnx")
    assert [tuple(sh) for sh in q.shapes] == s.state_shapes()
    a, b = st.OnnxStream(fp32).run(wav), q.run(wav)
    assert float(st.cosine(a, b).mean()) > 0.9


def test_macs_per_block():
    layers = st.macs_per_block(st.StreamingTrunk(_trunk()))
    assert layers["dft"] == 8 * 402 * 400
    assert layers["stem"] == 4 * 256 * 64 * 4
    assert all(layers[f"block{i}_dw"] == 4 * 256 * 5 and layers[f"block{i}_pw"] == 4 * 256 * 256 for i in range(8))
    assert sum(layers.values()) == 3924368


def _whole_onnx(trunk, path):
    with torch.no_grad():
        torch.onnx.export(trunk, torch.zeros(1, 16000), str(path), input_names=["waveform"], output_names=["features"],
                          dynamic_axes={"waveform": {0: "batch", 1: "samples"}, "features": {0: "batch", 1: "frames"}},
                          opset_version=17, dynamo=False)
    return path


def _streamify_error(mod, tmp_path):
    trunk = _trunk(mod)
    src = _whole_onnx(trunk, tmp_path / "whole.onnx")
    states = mod.streamify_onnx(src, tmp_path / "stream.onnx", source_revision="test")
    wav = _signal()
    whole = _whole(trunk, wav)[0].numpy()
    got = mod.OnnxStream(tmp_path / "stream.onnx").run(wav[0].numpy())
    return states, float(np.abs(got - whole[:len(got)]).max() / np.abs(whole).max())


def test_streamify_published_layout_equals_whole_signal(tmp_path):
    states, err = _streamify_error(st, tmp_path)
    assert list(states) == ["wav_state", "stem_state", *(f"block{i}_state" for i in range(8))]
    assert list(states.values()) == [[1, 1, 240], [1, 64, 2], *([1, 256, 4 * d] for d in (1, 2, 4, 8, 1, 2, 4, 8))]
    assert err < 1e-5
    meta = {p.key: p.value for p in onnx.load(str(tmp_path / "stream.onnx")).metadata_props}
    assert meta["source_sha256"] == __import__("hashlib").sha256((tmp_path / "whole.onnx").read_bytes()).hexdigest()
    assert meta["source_revision"] == "test" and meta["block_samples"] == "1280"


def test_streamified_graph_takes_any_whole_number_of_hops(tmp_path):
    trunk = _trunk()
    st.streamify_onnx(_whole_onnx(trunk, tmp_path / "whole.onnx"), tmp_path / "stream.onnx")
    s = st.OnnxStream(tmp_path / "stream.onnx")
    s.block = 640
    wav = _signal()
    whole = _whole(trunk, wav)[0].numpy()
    got = s.run(wav[0].numpy())
    assert float(np.abs(got - whole).max() / np.abs(whole).max()) < 1e-5


def _quantized_agreement(mod, tmp_path):
    """Streamify a static int8 QDQ graph (the published recipe, front end in float) and compare it streamed with the
    same int8 graph run once over the whole signal: (share of frames within 1e-4 of the feature scale, lowest
    per-frame cosine)."""
    import onnxruntime as ort

    trunk = _trunk(mod)
    src = _whole_onnx(trunk, tmp_path / "whole.onnx")
    calib = [{"waveform": _signal(2, seed=s)[0].numpy()[None]} for s in range(4)]
    mod.quantize_int8(src, tmp_path / "whole_int8.onnx", calib)
    mod.streamify_onnx(tmp_path / "whole_int8.onnx", tmp_path / "stream_int8.onnx")
    g = onnx.load(str(tmp_path / "stream_int8.onnx")).graph
    assert sum(n.op_type == "QuantizeLinear" for n in g.node) > 10
    wav = _signal(seed=5)[0].numpy()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    whole = ort.InferenceSession(str(tmp_path / "whole_int8.onnx"), so,
                                 providers=["CPUExecutionProvider"]).run(None, {"waveform": wav[None]})[0][0]
    got = mod.OnnxStream(tmp_path / "stream_int8.onnx").run(wav)
    whole = whole[:len(got)]
    close = np.abs(got - whole).max(1) <= 1e-4 * np.abs(whole).max()
    return float(close.mean()), float(st.cosine(got, whole).min())


def test_streamify_quantized_graph_equals_it_whole_but_for_rounding(tmp_path):
    close, cos = _quantized_agreement(st, tmp_path)
    assert close > 0.9
    assert cos > 0.98


STREAMIFY_MUTANTS = {
    "state sliced one step late": ("(-k, np.iinfo(np.int64).max, rank - 1)", "(-k - 1, -1, rank - 1)"),
    "pad kept in front of the state": ('[name, n.input[0]], [n.output[0]]', '[n.input[0], name], [n.output[0]]'),
}


@pytest.mark.parametrize("name", STREAMIFY_MUTANTS)
def test_broken_streamify_breaks_equality(name, tmp_path):
    anchor, replacement = STREAMIFY_MUTANTS[name]
    source = _P.read_text()
    assert source.count(anchor) == 1, f"{name}: anchor not unique in {_P.name}"
    mutant = _load(source.replace(anchor, replacement))
    try:
        _, err = _streamify_error(mutant, tmp_path)
    except Exception:
        err = None
    assert err is None or err > 1e-3
    try:
        close, cos = _quantized_agreement(mutant, tmp_path / "int8")
    except Exception:
        return
    assert close < 0.9 or cos < 0.98


MUTANTS = {
    "dilation buffer one frame short": ("xin[..., -self.ctx:]", "F.pad(xin[..., 1 - self.ctx:], (1, 0))"),
    "stem buffer off by one log-mel frame": (", s[..., -self.ctx:]", ", s[..., -self.ctx - 1:-1]"),
    "stem buffer dropped": (", s[..., -self.ctx:]", ", torch.zeros_like(state)"),
    "waveform overlap dropped": (", w[:, -self.ctx:]", ", torch.zeros_like(state)"),
    "state reset every block": ("f, *state = self(", "f, *_ = self("),
}


@pytest.mark.parametrize("name", MUTANTS)
def test_broken_state_carry_breaks_equality(name):
    anchor, replacement = MUTANTS[name]
    source = _P.read_text()
    assert source.count(anchor) == 1, f"{name}: anchor not unique in {_P.name}"
    mutant = _load(source.replace(anchor, replacement))
    assert _stream_error(st, _signal()) < 1e-5
    assert _stream_error(mutant, _signal()) > 1e-3
