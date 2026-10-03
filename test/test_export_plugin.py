import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from click.testing import CliRunner

from ww_trainer.export_plugin import export_plugin, main

KW = dict(wake_word="hey jarvis", license="Apache-2.0", training_data="synthetic test speech",
          threshold=0.37)


class _Head(torch.nn.Module):
    def __init__(self, seed, dim=128, scale=1.0, probability=False):
        super().__init__()
        torch.manual_seed(seed)
        self.gru = torch.nn.GRU(dim, 8, batch_first=True)
        self.fc = torch.nn.Linear(8, 1)
        self.scale, self.probability = scale, probability

    def forward(self, x):
        out, _ = self.gru(x)
        z = self.fc(out[:, -1]).squeeze(-1) * self.scale
        return torch.sigmoid(z) if self.probability else z


def _head(path, seed, dim=128, wake_word="hey_jarvis", scale=1.0, output="logits",
          probability=False, featurizer="wakehubert", window=None):
    torch.onnx.export(_Head(seed, dim, scale, probability).eval(), torch.zeros(1, 75, dim), str(path),
                      input_names=["input_features"], output_names=[output],
                      dynamic_axes={"input_features": {0: "batch", 1: "frames"}, output: {0: "batch"}},
                      opset_version=18, dynamo=False)
    m = onnx.load(str(path))
    meta = {"wake_word": wake_word, "arch": "gru", "metric_f1": str(0.9 + seed / 100)}
    if featurizer:
        meta["pretrained_featurizer"] = featurizer
    if window:
        meta["window_frames"] = str(window)
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.save(m, str(path))
    return path


def _logit(path, x):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, {"input_features": x})[0].reshape(len(x)).astype(np.float64)


def _x(n=3, frames=75):
    return np.random.default_rng(1).normal(0, 1, (n, frames, 128)).astype(np.float32)


def _run(path, x):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, {"features": x})[0]


def test_single_head_has_plugin_io_and_metadata(tmp_path):
    h = _head(tmp_path / "h.onnx", 0)
    out = export_plugin([h], tmp_path / "plugin.onnx", **KW)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    (inp,), (o,) = sess.get_inputs(), sess.get_outputs()
    assert (inp.name, inp.shape[2]) == ("features", 128)
    assert o.name == "logit_calibrated"
    assert _run(out, np.zeros((1, 75, 128), np.float32)).shape == (1,)
    meta = sess.get_modelmeta().custom_metadata_map
    assert meta["wake_word"] == "hey jarvis"
    assert meta["pretrained_featurizer"] == "wakehubert"
    assert meta["window_frames"] == "75"
    assert meta["license"] == "Apache-2.0"
    assert meta["training_data"] == "synthetic test speech"
    assert float(meta["default_threshold"]) == 0.37
    assert meta["arch"] == "gru"


def test_calibrated_output_is_coef_logit_plus_intercept(tmp_path):
    h = _head(tmp_path / "h.onnx", 0)
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps({"coef": 1.7, "intercept": -0.6}))
    out = export_plugin([h], tmp_path / "plugin.onnx", calibration=cal, **KW)
    x = _x()
    np.testing.assert_allclose(_run(out, x), 1.7 * _logit(h, x) - 0.6, atol=1e-4)


def test_ensemble_outputs_calibrated_mean(tmp_path):
    a, b = _head(tmp_path / "a.onnx", 0), _head(tmp_path / "b.onnx", 1)
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps({"coef": 0.8, "intercept": 0.3}))
    out = export_plugin([a, b], tmp_path / "plugin.onnx", calibration=cal, **KW)
    x = _x()
    la, lb = _logit(a, x), _logit(b, x)
    assert np.max(np.abs(la - lb)) > 0.01
    np.testing.assert_allclose(_run(out, x), 0.8 * (la + lb) / 2 + 0.3, atol=1e-4)
    meta = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    assert "metric_f1" not in meta


def test_wrong_feature_dim_is_refused(tmp_path):
    h = _head(tmp_path / "h.onnx", 0, dim=64)
    with pytest.raises(ValueError, match="128"):
        export_plugin([h], tmp_path / "plugin.onnx", **KW)
    assert not (tmp_path / "plugin.onnx").exists()


@pytest.mark.parametrize("threshold", [0.0, 1.0, 1.5, -0.2])
def test_threshold_outside_unit_interval_is_refused(tmp_path, threshold):
    h = _head(tmp_path / "h.onnx", 0)
    with pytest.raises(ValueError, match="threshold"):
        export_plugin([h], tmp_path / "plugin.onnx", **{**KW, "threshold": threshold})
    assert not (tmp_path / "plugin.onnx").exists()


def test_cli_writes_the_file(tmp_path):
    h = _head(tmp_path / "h.onnx", 0)
    out = tmp_path / "plugin.onnx"
    res = CliRunner().invoke(main, [str(h), "--threshold", "0.4", "--training-data", "x", "-o", str(out)])
    assert res.exit_code == 0, res.output
    meta = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    assert meta["wake_word"] == "hey jarvis"


def test_conflicting_featurizers_are_refused(tmp_path):
    a = _head(tmp_path / "a.onnx", 0)
    b = _head(tmp_path / "b.onnx", 1, featurizer="mfcc")
    with pytest.raises(ValueError, match="wakehubert"):
        export_plugin([a, b], tmp_path / "plugin.onnx", **KW)


def _featurizer_of(path):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.get_modelmeta().custom_metadata_map["pretrained_featurizer"]


def test_int8_featurizer_is_kept(tmp_path):
    h = _head(tmp_path / "h.onnx", 0, featurizer="wakehubert-int8")
    out = export_plugin([h], tmp_path / "plugin.onnx", **KW)
    assert _featurizer_of(out) == "wakehubert-int8"


def test_int8_ensemble_is_kept(tmp_path):
    a = _head(tmp_path / "a.onnx", 0, featurizer="wakehubert-int8")
    b = _head(tmp_path / "b.onnx", 1, featurizer="wakehubert-int8")
    out = export_plugin([a, b], tmp_path / "plugin.onnx", **KW)
    assert _featurizer_of(out) == "wakehubert-int8"


def test_float32_and_int8_heads_are_refused_together(tmp_path):
    a = _head(tmp_path / "a.onnx", 0, featurizer="wakehubert")
    b = _head(tmp_path / "b.onnx", 1, featurizer="wakehubert-int8")
    with pytest.raises(ValueError, match="disagree.*wakehubert.*wakehubert-int8"):
        export_plugin([a, b], tmp_path / "plugin.onnx", **KW)


def test_unknown_int8_featurizer_is_refused(tmp_path):
    h = _head(tmp_path / "h.onnx", 0, featurizer="mfcc-int8")
    with pytest.raises(ValueError, match="mfcc-int8"):
        export_plugin([h], tmp_path / "plugin.onnx", **KW)


def test_missing_featurizer_is_refused(tmp_path):
    h = _head(tmp_path / "h.onnx", 0, featurizer=None)
    with pytest.raises(ValueError, match="wakehubert"):
        export_plugin([h], tmp_path / "plugin.onnx", **KW)


def test_probability_output_is_refused(tmp_path):
    h = _head(tmp_path / "h.onnx", 0, output="probability", probability=True)
    with pytest.raises(ValueError, match="logits"):
        export_plugin([h], tmp_path / "plugin.onnx", **KW)


def test_window_frames_comes_from_the_heads(tmp_path):
    h = _head(tmp_path / "h.onnx", 0, window=50)
    out = export_plugin([h], tmp_path / "plugin.onnx", **KW)
    meta = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    assert meta["window_frames"] == "50"


def test_window_frames_mismatch_is_refused(tmp_path):
    h = _head(tmp_path / "h.onnx", 0, window=50)
    with pytest.raises(ValueError, match="window-frames"):
        export_plugin([h], tmp_path / "plugin.onnx", window_frames=75, **KW)
    a, b = _head(tmp_path / "a.onnx", 0, window=50), _head(tmp_path / "b.onnx", 1, window=75)
    with pytest.raises(ValueError, match="disagree"):
        export_plugin([a, b], tmp_path / "plugin.onnx", **KW)


def test_sum_instead_of_mean_is_refused_on_large_logits(tmp_path, monkeypatch):
    from onnx import helper
    a = _head(tmp_path / "a.onnx", 0, scale=3000.0)
    b = _head(tmp_path / "b.onnx", 1, scale=3000.0)
    assert np.min(np.abs(_logit(a, _x()))) > 20
    real = helper.make_node
    monkeypatch.setattr(helper, "make_node",
                        lambda op, *args, **kw: real("Sum" if op == "Mean" else op, *args, **kw))
    with pytest.raises(ValueError, match="differs"):
        export_plugin([a, b], tmp_path / "plugin.onnx", **KW)
