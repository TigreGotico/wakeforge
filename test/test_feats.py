"""Tests for ww_trainer/feats.py extractor classes."""
import math
import pytest
import torch
import numpy as np

from ww_trainer.feats import (
    MfccExtractor,
    OnnxFeatureExtractor,
    FilterbankExtractor,
    SincNetExtractor,
    DeltaExtractor,
    GammatoneExtractor,
    ensure_wav_list,
    SlidingFeatureCacheTensor,
)


class TestMfccExtractor:
    def test_forward_shape(self):
        """MfccExtractor output must be [B, T, n_mfcc]."""
        model = MfccExtractor(sr=16000, n_mfcc=40, n_mels=40, n_fft=400, hop_length=160)
        wav = torch.randn(1, 16000)
        out = model(wav)
        assert out.ndim == 3
        assert out.shape[0] == 1  # batch
        assert out.shape[2] == 40  # n_mfcc

    def test_forward_list_input(self):
        """MfccExtractor accepts list of 1-D tensors."""
        model = MfccExtractor(sr=16000, n_mfcc=13, n_mels=40)
        wavs = [torch.randn(16000), torch.randn(8000)]
        out = model(wavs)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 13

    def test_feature_dim_property(self):
        model = MfccExtractor(n_mfcc=20)
        assert model.feature_dim == 20

    def test_export_to_onnx(self, tmp_path):
        """MfccExtractor.export_to_onnx() produces a loadable ONNX file."""
        model = MfccExtractor(sr=16000, n_mfcc=13, n_mels=40, n_fft=400, hop_length=160)
        out_path = str(tmp_path / "mfcc_test.onnx")
        model.export_to_onnx(out_path)
        assert (tmp_path / "mfcc_test.onnx").exists()

    def test_onnx_loaded_by_onnx_extractor(self, tmp_path):
        """ONNX exported by MfccExtractor loads correctly into OnnxFeatureExtractor."""
        model = MfccExtractor(sr=16000, n_mfcc=13, n_mels=40, n_fft=400, hop_length=160)
        out_path = str(tmp_path / "mfcc_test.onnx")
        model.export_to_onnx(out_path)
        loader = OnnxFeatureExtractor(out_path, sample_rate=16000, device="cpu")
        wav = torch.randn(1, 16000)
        out = loader(wav)
        assert out.ndim == 3
        assert out.shape[0] == 1
        assert out.shape[2] == 13


class TestOnnxFeatureExtractorFeatureDim:
    def test_feature_dim_from_exported_mfcc(self, tmp_path):
        model = MfccExtractor(sr=16000, n_mfcc=20, n_mels=40, n_fft=400, hop_length=160)
        out_path = str(tmp_path / "mfcc20.onnx")
        model.export_to_onnx(out_path)
        loader = OnnxFeatureExtractor(out_path, sample_rate=16000, device="cpu")
        assert loader.feature_dim == 20


class TestFilterbankExtractor:
    def test_forward_shape(self):
        model = FilterbankExtractor(sr=16000, n_mels=80)
        wav = torch.randn(1, 16000)
        out = model(wav)
        assert out.ndim == 3
        assert out.shape[0] == 1
        assert out.shape[2] == 80

    def test_feature_dim(self):
        model = FilterbankExtractor(n_mels=40)
        assert model.feature_dim == 40

    def test_export_onnx(self, tmp_path):
        model = FilterbankExtractor(sr=16000, n_mels=40, n_fft=400, hop_length=160)
        out_path = str(tmp_path / "fbank.onnx")
        model.export_to_onnx(out_path)
        assert (tmp_path / "fbank.onnx").exists()


class TestSincNetExtractor:
    def test_forward_shape(self):
        model = SincNetExtractor(sr=16000, n_filters=40)
        wav = torch.randn(1, 16000)
        out = model(wav)
        assert out.ndim == 3
        assert out.shape[0] == 1
        assert out.shape[2] == 40

    def test_feature_dim(self):
        model = SincNetExtractor(n_filters=32)
        assert model.feature_dim == 32

    def test_learnable_params(self):
        model = SincNetExtractor(n_filters=20)
        params = list(model.parameters())
        assert len(params) >= 2  # low_hz_ and band_hz_

    def test_export_onnx(self, tmp_path):
        model = SincNetExtractor(sr=16000, n_filters=20, kernel_size=51)
        out_path = str(tmp_path / "sincnet.onnx")
        model.export_to_onnx(out_path)
        assert (tmp_path / "sincnet.onnx").exists()


class TestDeltaExtractor:
    def test_forward_shape(self):
        base = MfccExtractor(sr=16000, n_mfcc=13)
        model = DeltaExtractor(base)
        wav = torch.randn(1, 16000)
        out = model(wav)
        assert out.ndim == 3
        assert out.shape[0] == 1
        assert out.shape[2] == 39  # 13 * 3

    def test_feature_dim(self):
        base = MfccExtractor(n_mfcc=20)
        model = DeltaExtractor(base)
        assert model.feature_dim == 60  # 20 * 3

    def test_delta_width_2(self):
        base = FilterbankExtractor(sr=16000, n_mels=40)
        model = DeltaExtractor(base, delta_width=2)
        wav = torch.randn(2, 8000)
        out = model(wav)
        assert out.shape[-1] == 120  # 40 * 3

    def test_export_to_onnx(self, tmp_path):
        base = MfccExtractor(sr=16000, n_mfcc=13, n_mels=40, n_fft=400, hop_length=160)
        model = DeltaExtractor(base)
        out_path = str(tmp_path / "delta_mfcc.onnx")
        model.export_to_onnx(out_path)
        assert (tmp_path / "delta_mfcc.onnx").exists()

    def test_onnx_output_matches_pytorch(self, tmp_path):
        import numpy as np
        import onnxruntime as ort
        base = MfccExtractor(sr=16000, n_mfcc=13, n_mels=40, n_fft=400, hop_length=160)
        model = DeltaExtractor(base)
        model.eval()
        out_path = str(tmp_path / "delta_mfcc.onnx")
        model.export_to_onnx(out_path)
        wav = torch.randn(1, 16000)
        with torch.no_grad():
            pt_out = model(wav).numpy()
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {"input_values": wav.numpy()})[0]
        np.testing.assert_allclose(pt_out, onnx_out, rtol=1e-3, atol=1e-4)


class TestGammatoneExtractor:
    def test_forward_shape(self):
        model = GammatoneExtractor(sr=16000, n_filters=32)
        wav = torch.randn(1, 16000)
        out = model(wav)
        assert out.ndim == 3
        assert out.shape[0] == 1
        assert out.shape[2] == 32

    def test_feature_dim(self):
        model = GammatoneExtractor(n_filters=64)
        assert model.feature_dim == 64

    def test_list_input(self):
        model = GammatoneExtractor(sr=16000, n_filters=16)
        wavs = [torch.randn(8000), torch.randn(4000)]
        out = model(wavs)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 16

    def test_export_to_onnx(self, tmp_path):
        model = GammatoneExtractor(sr=16000, n_filters=16, frame_len=400, hop_length=160)
        out_path = str(tmp_path / "gammatone.onnx")
        model.export_to_onnx(out_path)
        assert (tmp_path / "gammatone.onnx").exists()


class TestSlidingFeatureCacheTensor:
    def test_cache_grows(self):
        cache = SlidingFeatureCacheTensor(feature_dim=8, window_size=10)
        feats = torch.ones(3, 8)
        out = cache(feats)
        assert out.shape == (3, 8)

    def test_cache_saturates_at_window(self):
        cache = SlidingFeatureCacheTensor(feature_dim=8, window_size=5)
        for _ in range(4):
            cache(torch.ones(2, 8))
        out = cache(torch.ones(2, 8))
        assert out.shape[0] == 5
