"""Tests for ww_trainer/inference.py — ONNX-only inference."""
import numpy as np
import pytest
import torch
import torch.nn as nn

# Use the legacy TorchScript-based ONNX exporter directly to avoid the new
# torch.export-based path which requires onnxscript (not installed in CI).
from torch.onnx._internal.torchscript_exporter.utils import export as _ts_onnx_export


def _export_trivial_extractor(tmp_path) -> str:
    """Export a trivial linear model as extractor ONNX for testing."""
    class TrivialExtractor(nn.Module):
        def forward(self, x):  # x: [B, T]
            # Return [B, T//160, 13] — simulate MFCC-like output
            B, T = x.shape
            frames = max(1, T // 160)
            return torch.zeros(B, frames, 13)

    model = TrivialExtractor().eval()
    path = str(tmp_path / "trivial_extractor.onnx")
    dummy = torch.zeros(1, 16000)
    _ts_onnx_export(
        model, (dummy,), path,
        input_names=["input_values"],
        output_names=["features"],
        dynamic_axes={"input_values": {0: "B", 1: "T"}, "features": {0: "B", 1: "T"}},
        opset_version=14,
    )
    return path


def _export_trivial_head(tmp_path) -> str:
    """Export a trivial linear model as head ONNX for testing."""
    class TrivialHead(nn.Module):
        def forward(self, x):  # x: [B, T, F]
            return x.mean(dim=(1, 2))  # [B]

    model = TrivialHead().eval()
    path = str(tmp_path / "trivial_head.onnx")
    dummy = torch.zeros(1, 10, 13)
    _ts_onnx_export(
        model, (dummy,), path,
        input_names=["input_features"],
        output_names=["logits"],
        dynamic_axes={"input_features": {0: "B", 1: "T"}},
        opset_version=14,
    )
    return path


@pytest.fixture
def onnx_sessions(tmp_path):
    ext_path = _export_trivial_extractor(tmp_path)
    head_path = _export_trivial_head(tmp_path)
    return ext_path, head_path


def test_import_no_torch():
    """OnnxWakeWordInferencer must be importable (onnxruntime-only module)."""
    from ww_trainer.inference import OnnxWakeWordInferencer
    assert OnnxWakeWordInferencer is not None


def test_no_torch_in_module_source():
    """inference.py must not contain 'import torch' at module level."""
    import ww_trainer.inference as inf_mod
    import inspect
    source = inspect.getsource(inf_mod)
    # Allow torch in comments/strings but not as a top-level import statement
    lines = [l.strip() for l in source.splitlines()]
    top_level_torch_imports = [
        l for l in lines
        if l.startswith("import torch") or l.startswith("from torch")
    ]
    assert len(top_level_torch_imports) == 0, (
        f"Found torch imports in inference.py: {top_level_torch_imports}"
    )


def test_infer_returns_probability(onnx_sessions):
    from ww_trainer.inference import OnnxWakeWordInferencer
    ext_path, head_path = onnx_sessions
    inferencer = OnnxWakeWordInferencer(ext_path, head_path, sample_rate=16000, device="cpu")
    audio = np.zeros(16000, dtype=np.float32)
    prob = inferencer.infer(audio)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_infer_batch(onnx_sessions):
    from ww_trainer.inference import OnnxWakeWordInferencer
    ext_path, head_path = onnx_sessions
    inferencer = OnnxWakeWordInferencer(ext_path, head_path, sample_rate=16000, device="cpu")
    batch = np.zeros((3, 16000), dtype=np.float32)
    probs = inferencer.infer_batch(batch)
    assert probs.shape == (3,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_infer_streaming(onnx_sessions):
    from ww_trainer.inference import OnnxWakeWordInferencer
    ext_path, head_path = onnx_sessions
    inferencer = OnnxWakeWordInferencer(ext_path, head_path, sample_rate=16000, device="cpu")
    chunk = np.zeros(4000, dtype=np.float32)
    cache = None
    for _ in range(5):
        prob, cache = inferencer.infer_streaming(chunk, cache)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0
        assert cache.ndim == 2


def test_infer_rejects_2d_input(onnx_sessions):
    from ww_trainer.inference import OnnxWakeWordInferencer
    ext_path, head_path = onnx_sessions
    inferencer = OnnxWakeWordInferencer(ext_path, head_path, sample_rate=16000, device="cpu")
    with pytest.raises(ValueError):
        inferencer.infer(np.zeros((2, 16000), dtype=np.float32))
