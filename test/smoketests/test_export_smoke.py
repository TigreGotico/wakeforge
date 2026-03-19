"""Smoke tests: ONNX export + inference for key pipelines, plus C export and calibration."""
import os
import pytest
import numpy as np
import torch
import onnx
import onnxruntime as ort

from ww_trainer.feats import MfccExtractor
from ww_trainer.model import FfnClassifierHead, GruClassifierHead, BaseWakeModel


class TestOnnxExportHead:
    """Export classifier heads to ONNX and verify they load."""

    @pytest.mark.parametrize("hidden_dim", [16, 64])
    def test_ffn_export(self, tmp_path, hidden_dim: int) -> None:
        head = FfnClassifierHead(input_size=13, hidden_dim=hidden_dim, device="cpu")
        out = str(tmp_path / f"ffn_{hidden_dim}.onnx")
        head.export_to_onnx(out)
        assert os.path.exists(out)
        onnx.checker.check_model(onnx.load(out))

    def test_gru_export(self, tmp_path) -> None:
        head = GruClassifierHead(input_size=13, hidden_dim=16, device="cpu")
        out = str(tmp_path / "gru.onnx")
        head.export_to_onnx(out)
        assert os.path.exists(out)

    def test_ffn_export_quantize(self, tmp_path) -> None:
        head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")
        out = str(tmp_path / "ffn_q.onnx")
        head.export_to_onnx(out, quantize=True)
        int8_path = str(tmp_path / "ffn_q_int8.onnx")
        assert os.path.exists(int8_path)

    def test_ffn_export_with_metadata(self, tmp_path) -> None:
        head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")
        out = str(tmp_path / "ffn_meta.onnx")
        head.export_to_onnx(out, metadata={"wake_word": "test", "arch": "ffn"})
        model = onnx.load(out)
        props = {p.key: p.value for p in model.metadata_props}
        assert props["wake_word"] == "test"


class TestOnnxExportExtractor:
    """Export feature extractors to ONNX."""

    def test_mfcc_export(self, tmp_path) -> None:
        ext = MfccExtractor(n_mfcc=13)
        ext.cpu()
        out = str(tmp_path / "mfcc.onnx")
        ext.export_to_onnx(out)
        assert os.path.exists(out)

    def test_mfcc_export_with_metadata(self, tmp_path) -> None:
        ext = MfccExtractor(n_mfcc=13)
        ext.cpu()
        out = str(tmp_path / "mfcc_meta.onnx")
        ext.export_to_onnx(out, metadata={"type": "mfcc"})
        model = onnx.load(out)
        props = {p.key: p.value for p in model.metadata_props}
        assert props["type"] == "mfcc"


class TestOnnxExportFullModel:
    """Export full model (head + featurizer) to ONNX."""

    def test_export_featurizer(self, tmp_path) -> None:
        ext = MfccExtractor(n_mfcc=13)
        ext.cpu()
        head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")
        model = BaseWakeModel(ext, head, device="cpu")
        out = str(tmp_path / "model.onnx")
        model.export_to_onnx(out, export_featurizer=True, metadata={"test": "true"})
        assert os.path.exists(out)
        feat_out = str(tmp_path / "model_featurizer.onnx")
        assert os.path.exists(feat_out)


class TestOnnxInference:
    """ONNX export → OnnxWakeWordInferencer round-trip."""

    def test_infer_roundtrip(self, tmp_path) -> None:
        ext = MfccExtractor(n_mfcc=13)
        ext.cpu()
        head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")
        model = BaseWakeModel(ext, head, device="cpu")

        ext_path = str(tmp_path / "ext.onnx")
        head_path = str(tmp_path / "head.onnx")
        ext.export_to_onnx(ext_path)
        head.export_to_onnx(head_path)

        from ww_trainer.inference import OnnxWakeWordInferencer
        inf = OnnxWakeWordInferencer(ext_path, head_path)

        audio = np.random.randn(16000).astype(np.float32)
        prob = inf.infer(audio)
        assert 0 <= prob <= 1

    def test_batch_infer(self, tmp_path) -> None:
        ext = MfccExtractor(n_mfcc=13)
        ext.cpu()
        head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")

        ext_path = str(tmp_path / "ext.onnx")
        head_path = str(tmp_path / "head.onnx")
        ext.export_to_onnx(ext_path)
        head.export_to_onnx(head_path)

        from ww_trainer.inference import OnnxWakeWordInferencer
        inf = OnnxWakeWordInferencer(ext_path, head_path)

        audio_batch = np.random.randn(4, 16000).astype(np.float32)
        probs = inf.infer_batch(audio_batch)
        assert probs.shape == (4,)
        assert all(0 <= p <= 1 for p in probs)

    def test_streaming_infer(self, tmp_path) -> None:
        ext = MfccExtractor(n_mfcc=13)
        ext.cpu()
        head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")

        ext_path = str(tmp_path / "ext.onnx")
        head_path = str(tmp_path / "head.onnx")
        ext.export_to_onnx(ext_path)
        head.export_to_onnx(head_path)

        from ww_trainer.inference import OnnxWakeWordInferencer
        inf = OnnxWakeWordInferencer(ext_path, head_path)

        # Stream 3 chunks
        cache = None
        for _ in range(3):
            chunk = np.random.randn(4000).astype(np.float32)
            prob, cache = inf.infer_streaming(chunk, cache)
            assert 0 <= prob <= 1


class TestCExport:
    """C header export for ESP32."""

    def test_export(self, tmp_path) -> None:
        from ww_trainer.export_c import export_to_c_header
        head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")
        out = str(tmp_path / "ww.h")
        export_to_c_header(head, out, model_name="hey_test", wake_word="hey test")
        content = open(out).read()
        assert "hey_test_infer" in content
        assert "int8_t" in content


class TestCalibration:
    """Platt scaling calibration."""

    def test_fit_apply(self) -> None:
        from ww_trainer.calibration import fit_platt_scaling, apply_platt_scaling
        logits = np.array([3.0, 2.0, -1.0, -3.0, 1.0, -0.5])
        labels = np.array([1, 1, 0, 0, 1, 0])
        params = fit_platt_scaling(logits, labels)
        probs = apply_platt_scaling(logits, params)
        # Positive logits should have higher calibrated probability
        assert probs[0] > probs[3]
        assert all(0 <= p <= 1 for p in probs)
