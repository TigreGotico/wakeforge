"""scripts/research/wakephonehubert/wakeword: score_stream.py cuts the same plugin-faithful windows as window_feats.py.

Both scripts push a stream in 1,280-sample blocks into a 24,000-sample buffer that starts as zeros. The featurizer here is an
ONNX graph that reshapes a window to [75, 320] and the head one that averages it, so a window's score changes with every
sample the window holds and a shifted window cannot score the same.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
import soundfile as sf
from onnx import TensorProto as TP
from onnx import helper, numpy_helper

_DIR = Path(__file__).resolve().parent.parent / "scripts" / "research" / "wakephonehubert" / "wakeword"
sys.path.insert(0, str(_DIR))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


wf = _load("window_feats")
ss = _load("score_stream")

SR, BLOCK, WIN = 16000, 1280, 24000


def featurizer(path):
    g = helper.make_graph(
        [helper.make_node("Reshape", ["waveform", "shape"], ["features"])], "featurizer",
        [helper.make_tensor_value_info("waveform", TP.FLOAT, ["batch", WIN])],
        [helper.make_tensor_value_info("features", TP.FLOAT, ["batch", 75, 320])],
        [numpy_helper.from_array(np.array([-1, 75, 320], np.int64), "shape")])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), str(path))


def head(path):
    g = helper.make_graph(
        [helper.make_node("ReduceMean", ["x"], ["m"], axes=[1, 2], keepdims=0),
         helper.make_node("Reshape", ["m", "flat"], ["logit"])], "head",
        [helper.make_tensor_value_info("x", TP.FLOAT, ["batch", 75, 320])],
        [helper.make_tensor_value_info("logit", TP.FLOAT, ["batch"])],
        [numpy_helper.from_array(np.array([-1], np.int64), "flat")])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), str(path))


@pytest.fixture
def stream(tmp_path):
    featurizer(tmp_path / "feat.onnx")
    head(tmp_path / "head.onnx")
    rng = np.random.default_rng(0)
    n = int(1.3 * SR)
    wav = (np.linspace(-3, 3, n) + rng.standard_normal(n) * 0.2).astype(np.float32)
    sf.write(tmp_path / "clip.wav", wav, SR, subtype="FLOAT")
    vars(wf).pop("_FZ", None)
    ss._S.clear()
    return tmp_path


def window_scores(feats):
    x = feats.astype(np.float32)
    return 1 / (1 + np.exp(-x.mean(axis=(1, 2))))


@pytest.mark.parametrize("pad", [False, True])
def test_score_stream_scores_the_windows_window_feats_stores(stream, pad):
    f = str(stream / "clip.wav")
    _, feats = wf._feats((str(stream / "feat.onnx"), f, pad))
    _, scored = ss._work((str(stream / "feat.onnx"), {"h": str(stream / "head.onnx")}, f, pad, ["h"]))
    assert feats.dtype == np.float16
    n = (int(1.3 * SR) + (2 * SR if pad else 0)) // BLOCK
    assert len(feats) == n and scored["h"].shape == (n,)
    assert np.allclose(scored["h"], window_scores(feats), atol=1e-6)
    assert np.ptp(scored["h"]) > 0.05


def test_windows_are_isolated_and_start_from_zeros(stream):
    f = str(stream / "clip.wav")
    wav, _ = sf.read(f, dtype="float32")
    _, feats = wf._feats((str(stream / "feat.onnx"), f, False))
    buffer = np.zeros(WIN, np.float32)
    for i in range(len(feats)):
        buffer = np.concatenate([buffer, wav[i * BLOCK:(i + 1) * BLOCK]])[-WIN:]
        assert np.allclose(feats[i].astype(np.float32).reshape(-1), buffer, atol=3e-3), i
    assert not feats[0].astype(np.float32).reshape(-1)[:WIN - BLOCK].any()


def test_main_writes_index_and_traces(stream):
    out = stream / "out"
    argv = sys.argv
    sys.argv = ["score_stream.py", str(stream / "feat.onnx"), str(out), "--heads", f"h={stream / 'head.onnx'}:kw",
                "--sets", f"kw={stream / 'clip.wav'}", "--procs", "1"]
    try:
        ss.main()
    finally:
        sys.argv = argv
    trace = np.load(out / "trace-h.part0of1.npy", allow_pickle=True).item()
    assert set(trace) == {"heldout"} and len(trace["heldout"]) == 1
    assert json.loads((out / "sets-index.json").read_text())["kw"] == [str(stream / "clip.wav")]
    _, feats = wf._feats((str(stream / "feat.onnx"), str(stream / "clip.wav"), True))
    assert np.allclose(trace["heldout"][0], window_scores(feats), atol=1e-6)
