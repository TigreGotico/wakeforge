"""Extended tests for ww_trainer/feats.py — MfccExtractor variants, OnnxFeatureExtractor fallback."""
import numpy as np
import pytest
import torch

from ww_trainer.feats import MfccExtractor, OnnxFeatureExtractor, ensure_wav_list


# ---- MfccExtractor with different n_mfcc values ----

@pytest.mark.parametrize("n_mfcc", [13, 20, 40])
def test_mfcc_extractor_output_dim(n_mfcc):
    model = MfccExtractor(sr=16000, n_mfcc=n_mfcc, n_mels=40, n_fft=400, hop_length=160)
    wav = torch.randn(1, 16000)
    out = model(wav)
    assert out.ndim == 3
    assert out.shape[2] == n_mfcc, f"Expected last dim={n_mfcc}, got {out.shape}"


@pytest.mark.parametrize("n_mfcc", [13, 20, 40])
def test_mfcc_feature_dim_property(n_mfcc):
    model = MfccExtractor(n_mfcc=n_mfcc)
    assert model.feature_dim == n_mfcc


def test_mfcc_extractor_batch_of_two():
    model = MfccExtractor(sr=16000, n_mfcc=13, n_mels=40, n_fft=400, hop_length=160)
    wav = torch.randn(2, 16000)
    out = model(wav)
    assert out.shape[0] == 2
    assert out.shape[2] == 13


def test_mfcc_extractor_no_nan():
    model = MfccExtractor(sr=16000, n_mfcc=13, n_mels=40)
    wav = torch.randn(1, 8000)
    out = model(wav)
    assert not torch.isnan(out).any()


# ---- OnnxFeatureExtractor.feature_dim fallback (dynamic axis) ----

def test_onnx_feature_dim_fallback_via_dummy_inference(tmp_path):
    """Verify OnnxFeatureExtractor.feature_dim works even for dynamic output shapes."""
    # Export MfccExtractor with dynamic axes (simulates dynamic shape scenario)
    model = MfccExtractor(sr=16000, n_mfcc=13, n_mels=40, n_fft=400, hop_length=160)
    onnx_path = str(tmp_path / "mfcc_dynamic.onnx")
    model.export_to_onnx(onnx_path)

    # Manually patch the output shape to simulate a dynamic last dimension
    import onnx
    onnx_model = onnx.load(onnx_path)
    # Make the last dim of the output dynamic (None)
    output = onnx_model.graph.output[0]
    dim = output.type.tensor_type.shape.dim
    if len(dim) >= 3:
        dim[2].ClearField("dim_value")
        dim[2].dim_param = "feat_dim"  # marks as dynamic
    onnx.save(onnx_model, onnx_path)

    loader = OnnxFeatureExtractor(onnx_path, sample_rate=16000, device="cpu")
    # feature_dim should fall back to dummy inference and return 13
    fd = loader.feature_dim
    assert fd == 13


# ---- ensure_wav_list edge cases ----

def test_ensure_wav_list_2d_tensor():
    """2D tensor [B, T] → list of B 1D tensors."""
    wav = torch.zeros(4, 16000)
    result = ensure_wav_list(wav)
    assert isinstance(result, list)
    assert len(result) == 4
    for w in result:
        assert w.ndim == 1
        assert w.shape[0] == 16000


def test_ensure_wav_list_1d_tensor():
    wav = torch.zeros(16000)
    result = ensure_wav_list(wav)
    assert len(result) == 1
    assert result[0].shape == (16000,)


def test_ensure_wav_list_list_passthrough():
    wavs = [torch.zeros(8000), torch.zeros(4000)]
    result = ensure_wav_list(wavs)
    assert result is wavs  # same object returned


def test_ensure_wav_list_raises_for_3d_tensor():
    wav = torch.zeros(2, 3, 16000)
    with pytest.raises(ValueError):
        ensure_wav_list(wav)


def test_ensure_wav_list_raises_for_invalid_type():
    with pytest.raises(TypeError):
        ensure_wav_list("not_a_tensor")


def test_ensure_wav_list_raises_for_numpy():
    import numpy as np
    wav = np.zeros(16000)
    with pytest.raises(TypeError):
        ensure_wav_list(wav)
