"""TFLite Micro port of the streamed trunk (scripts/research/microwakehubert/export_tflm.py) without TensorFlow: the
numpy form of the exported graph (undilated depthwise kernels with zeros between the taps, state carried as
``[frames, channels]`` buffers) equals the stateful fp32 ONNX stream block by block; the firmware's C log-mel equals
the numpy log-mel; and each broken state carry (state reset every block, a buffer one frame short, the C front end's
overlap dropped) breaks the equality.

The trunk has the published architecture with random weights, so nothing is downloaded."""
import ctypes
import importlib.util
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

_DIR = Path(__file__).resolve().parents[1] / "scripts" / "research" / "microwakehubert"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


et = _load("export_tflm")
st = _load("stream_trunk")
from test_stream_trunk import _signal, _trunk  # noqa: E402


@pytest.fixture(scope="module")
def onnx_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("mwh") / "stream.onnx"
    st.export_onnx(st.StreamingTrunk(_trunk(st)), path)
    return path


@pytest.fixture(scope="module")
def tw(onnx_path):
    return et.TrunkWeights(onnx_path)


@pytest.fixture(scope="module")
def wav():
    return _signal()[0].numpy()


def _numpy_stream(tw, wav, reset_every_block=False, short=False):
    states = et.zero_states(tw)
    tail = np.zeros(et.CTX, np.float32)
    out = []
    for i in range(len(wav) // et.BLOCK):
        w = np.concatenate([tail, wav[i * et.BLOCK:(i + 1) * et.BLOCK]])
        tail = w[-et.CTX:]
        f, states = et.trunk_step(tw, et.logmel(tw, w), states)
        if short:
            states = [s if k == 0 else np.concatenate([np.zeros_like(s[:1]), s[:-1]]) for k, s in enumerate(states)]
        if reset_every_block:
            states = et.zero_states(tw)
        out.append(f)
    return np.concatenate(out)


def _onnx_stream(path, wav):
    s = et.OnnxStream(path)
    return np.concatenate([s.push(wav[i * et.BLOCK:(i + 1) * et.BLOCK])[0] for i in range(len(wav) // et.BLOCK)])


def test_expand_dilated_equals_dilated_convolution():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((6, 1, 5)).astype(np.float32)
    x = rng.standard_normal((50, 6)).astype(np.float32)
    for d in (1, 2, 4, 8):
        e = et.expand_dilated(w, d)
        assert e.shape == (4 * d + 1, 6)
        n = len(x) - 4 * d
        ref = np.stack([sum(w[:, 0, j] * x[t + j * d] for j in range(5)) for t in range(n)])
        got = np.stack([(e * x[t:t + len(e)]).sum(0) for t in range(n)])
        np.testing.assert_allclose(got, ref, atol=1e-5)


def test_weights_read_from_onnx(tw):
    assert tw.ctx == [4, 8, 16, 32, 4, 8, 16, 32]
    assert (tw.channels, tw.n_mels, tw.feature_dim) == (256, 64, 128)
    np.testing.assert_allclose(tw.window, np.hanning(et.N_FFT + 1)[:-1], atol=1e-6)


def test_numpy_graph_streams_like_onnx(tw, onnx_path, wav):
    ref = _onnx_stream(onnx_path, wav)
    got = _numpy_stream(tw, wav)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=2e-3 * np.abs(ref).max())


@pytest.mark.parametrize("broken", ["reset_every_block", "short"])
def test_broken_state_carry_breaks_equality(tw, onnx_path, wav, broken):
    ref = _onnx_stream(onnx_path, wav)
    got = _numpy_stream(tw, wav, **{broken: True})
    assert np.abs(got - ref).max() > 0.05 * np.abs(ref).max()


@pytest.fixture(scope="module")
def frontend(tw, tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc")
    assert cc, "a C compiler is required to test the firmware's log-mel"
    d = tmp_path_factory.mktemp("logmel")
    et.write_frontend_consts(tw, d)
    lib = d / "liblogmel.so"
    subprocess.run([cc, "-O2", "-shared", "-fPIC", f"-I{d}", "-o", str(lib), str(_DIR / "firmware/main/logmel.c"),
                    "-lm"], check=True, timeout=120)
    return lib


def _c_vs_numpy(tw, lib, wav, drop_overlap=False):
    fe = et.Frontend(lib, tw.n_mels)
    tail = np.zeros(et.CTX, np.float32)
    err = 0.0
    for i in range(len(wav) // et.BLOCK):
        blk = wav[i * et.BLOCK:(i + 1) * et.BLOCK]
        w = np.concatenate([tail, blk])
        tail = w[-et.CTX:]
        if drop_overlap:
            fe.reset()
        err = max(err, float(np.abs(fe.push(blk) - et.logmel(tw, w)).max()))
    return err


def test_c_logmel_equals_numpy(tw, frontend, wav):
    assert _c_vs_numpy(tw, frontend, wav) < 1e-2


def test_c_logmel_without_overlap_breaks_equality(tw, frontend, wav):
    assert _c_vs_numpy(tw, frontend, wav, drop_overlap=True) > 0.5


def test_c_logmel_signature_is_callable(frontend):
    lib = ctypes.CDLL(str(frontend))
    assert hasattr(lib, "mwh_logmel_block") and hasattr(lib, "mwh_logmel_init")
