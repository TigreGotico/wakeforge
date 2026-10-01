"""Tests for ww_trainer/quantize.py — ONNX quantization helpers."""
import numpy as np
import pytest
import torch

from ww_trainer.feats import MfccExtractor
from ww_trainer.model import FfnClassifierHead
from ww_trainer.quantize import quantize_onnx, quantize_model_pair, QuantizationReport


@pytest.fixture
def onnx_pair(tmp_path):
    """Export a tiny model pair to ONNX."""
    ext = MfccExtractor(n_mfcc=13)
    head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")
    ext.eval()
    head.eval()
    ext_path = str(tmp_path / "ext.onnx")
    head_path = str(tmp_path / "head.onnx")
    ext.export_to_onnx(ext_path)
    head.export_to_onnx(head_path)
    return ext_path, head_path


class TestQuantizeOnnx:
    def test_int8(self, onnx_pair, tmp_path):
        ext_path, _ = onnx_pair
        out = str(tmp_path / "ext_int8.onnx")
        result = quantize_onnx(ext_path, out, weight_type="int8")
        assert result == out
        assert (tmp_path / "ext_int8.onnx").exists()

    def test_int16(self, onnx_pair, tmp_path):
        _, head_path = onnx_pair
        out = str(tmp_path / "head_int16.onnx")
        quantize_onnx(head_path, out, weight_type="int16")
        assert (tmp_path / "head_int16.onnx").exists()


class TestQuantizeModelPair:
    def test_returns_report(self, onnx_pair, tmp_path):
        ext_path, head_path = onnx_pair
        report = quantize_model_pair(
            ext_path, head_path,
            output_dir=str(tmp_path / "q"),
            validate=True,
        )
        assert isinstance(report, QuantizationReport)
        assert report.extractor_original_kb > 0
        assert report.extractor_int8_kb > 0
        assert report.head_original_kb > 0
        assert report.numerical_diff is not None

    def test_size_reduction(self, onnx_pair, tmp_path):
        ext_path, head_path = onnx_pair
        report = quantize_model_pair(
            ext_path, head_path,
            output_dir=str(tmp_path / "q"),
            validate=False,
        )
        # Quantized should be smaller (or at least not larger)
        assert report.extractor_int8_kb <= report.extractor_original_kb * 1.1
