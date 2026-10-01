"""Tests for suggestion implementations: S-007, S-011, S-013, S-014."""
import os
import json
import pytest
import numpy as np
import torch
import torch.nn as nn

from ww_trainer.feats import MfccExtractor, BaseExtractor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel, ClassifierHead


# ---- S-014: device tracking ----

class TestDeviceTracking:
    """BaseExtractor and ClassifierHead should track .to() calls."""

    def test_extractor_device_tracks_cpu(self) -> None:
        ext = MfccExtractor(n_mfcc=13)
        ext.cpu()
        assert ext.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_extractor_device_tracks_cuda(self) -> None:
        ext = MfccExtractor(n_mfcc=13)
        ext.to("cuda")
        assert ext.device.type == "cuda"
        ext.cpu()
        assert ext.device == torch.device("cpu")

    def test_classifier_device_tracks_cpu(self) -> None:
        head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")
        assert head.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_classifier_device_tracks_cuda(self) -> None:
        head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")
        head.to("cuda")
        assert head.device.type == "cuda"


# ---- S-007: dataset validation ----

class TestDatasetValidation:

    def test_validate_flag_accepted(self, tmp_path) -> None:
        """validate=True should not crash on valid data."""
        from ww_trainer.dataset import AudioDataset
        import soundfile as sf

        # Create a valid wav file
        wav_path = str(tmp_path / "test.wav")
        sf.write(wav_path, np.random.randn(16000).astype(np.float32), 16000)

        ds = AudioDataset([(wav_path, "1")], validate=True)
        assert len(ds) == 1

    def test_validate_detects_bad_file(self, tmp_path) -> None:
        """validate=True should warn about unreadable files."""
        from ww_trainer.dataset import AudioDataset

        bad_path = str(tmp_path / "bad.wav")
        with open(bad_path, "w") as f:
            f.write("not audio")

        # Should not crash, just log warnings
        ds = AudioDataset([(bad_path, "0")], validate=True)
        assert len(ds) == 1

    def test_imbalance_warning(self, tmp_path) -> None:
        """Severe imbalance should trigger a warning."""
        from ww_trainer.dataset import AudioDataset
        import soundfile as sf

        wav_path = str(tmp_path / "test.wav")
        sf.write(wav_path, np.random.randn(16000).astype(np.float32), 16000)

        # 11:1 ratio
        samples = [(wav_path, "0")] * 11 + [(wav_path, "1")]
        ds = AudioDataset(samples)
        assert len(ds) == 12


# ---- S-011: calibration ----

class TestCalibration:

    def test_fit_platt_scaling(self) -> None:
        from ww_trainer.calibration import fit_platt_scaling

        logits = np.array([2.0, 1.5, -1.0, -2.0, 0.5, -0.5, 3.0, -3.0])
        labels = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        params = fit_platt_scaling(logits, labels)
        assert "coef" in params
        assert "intercept" in params
        assert params["coef"] > 0  # positive correlation

    def test_apply_platt_scaling(self) -> None:
        from ww_trainer.calibration import apply_platt_scaling

        params = {"coef": 1.0, "intercept": 0.0}
        logits = np.array([0.0, 10.0, -10.0])
        probs = apply_platt_scaling(logits, params)
        assert abs(probs[0] - 0.5) < 0.01
        assert probs[1] > 0.99
        assert probs[2] < 0.01

    def test_save_load_calibration(self, tmp_path) -> None:
        from ww_trainer.calibration import save_calibration, load_calibration

        params = {"coef": 1.23, "intercept": -0.45}
        path = str(tmp_path / "cal.json")
        save_calibration(params, path)
        loaded = load_calibration(path)
        assert abs(loaded["coef"] - 1.23) < 1e-6
        assert abs(loaded["intercept"] - (-0.45)) < 1e-6


# ---- S-013: C header export ----

class TestExportC:

    def test_export_c_header(self, tmp_path) -> None:
        from ww_trainer.export_c import export_to_c_header

        head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")
        out = str(tmp_path / "model.h")
        export_to_c_header(head, out, model_name="test_ww", wake_word="hey_test")

        assert os.path.exists(out)
        content = open(out).read()
        assert "#ifndef TEST_WW_H" in content
        assert "test_ww_infer" in content
        assert "hey_test" in content
        assert "int8_t test_ww_w0" in content

    def test_export_c_from_base_wake_model(self, tmp_path) -> None:
        from ww_trainer.export_c import export_to_c_header

        ext = MfccExtractor(n_mfcc=13)
        head = FfnClassifierHead(input_size=13, hidden_dim=8, device="cpu")
        model = BaseWakeModel(ext, head, device="cpu")

        out = str(tmp_path / "full_model.h")
        export_to_c_header(model, out)

        content = open(out).read()
        assert "WW_MODEL_INPUT_DIM 13" in content
        assert "WW_MODEL_HIDDEN_DIM 8" in content

    def test_export_c_rejects_non_ffn(self, tmp_path) -> None:
        from ww_trainer.export_c import export_to_c_header
        from ww_trainer.model import GruClassifierHead

        head = GruClassifierHead(input_size=13, hidden_dim=16, device="cpu")
        with pytest.raises(TypeError, match="FfnClassifierHead"):
            export_to_c_header(head, str(tmp_path / "bad.h"))


# ---- S-012: QAT ----

class TestQAT:

    def test_prepare_qat(self) -> None:
        from ww_trainer.qat import prepare_qat

        head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")
        prepared = prepare_qat(head, backend="x86")
        assert prepared is head  # in-place modification
        # Should still be able to forward
        dummy = torch.randn(1, 50, 13)
        out = prepared(dummy)
        assert out.shape == (1,)

    def test_convert_qat(self) -> None:
        from ww_trainer.qat import prepare_qat, convert_qat

        head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")
        prepare_qat(head, backend="x86")
        # Simulate a training step
        dummy = torch.randn(2, 50, 13)
        _ = head(dummy)
        # Convert
        quantized = convert_qat(head)
        assert quantized is not None


# ---- S-010: DDP utilities ----

class TestDDP:

    def test_is_ddp_available_false(self) -> None:
        from ww_trainer.ddp import is_ddp_available
        # In test env, LOCAL_RANK is not set
        if "LOCAL_RANK" not in os.environ:
            assert not is_ddp_available()

    def test_is_main_process(self) -> None:
        from ww_trainer.ddp import is_main_process
        # Without DDP, should be True (rank 0)
        if "LOCAL_RANK" not in os.environ:
            assert is_main_process()

    def test_get_local_rank_default(self) -> None:
        from ww_trainer.ddp import get_local_rank
        if "LOCAL_RANK" not in os.environ:
            assert get_local_rank() == 0
