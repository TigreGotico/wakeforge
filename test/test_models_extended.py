"""Extended tests for ww_trainer/model.py — ONNX export, infer(), checkpoint round-trip."""
import numpy as np
import pytest
import torch

from ww_trainer.feats import MfccExtractor
from ww_trainer.model import (
    FfnClassifierHead,
    GruClassifierHead,
    CnnClassifierHead,
    BaseWakeModel,
)


INPUT_SIZE = 13  # n_mfcc — small for fast tests


def _make_mfcc_ffn_model():
    extractor = MfccExtractor(sr=16000, n_mfcc=INPUT_SIZE, n_mels=40, n_fft=400, hop_length=160)
    head = FfnClassifierHead(input_size=INPUT_SIZE, hidden_dim=16, device="cpu")
    return BaseWakeModel(feature_extractor=extractor, classifier=head, device="cpu")


def _make_mfcc_gru_model():
    extractor = MfccExtractor(sr=16000, n_mfcc=INPUT_SIZE, n_mels=40, n_fft=400, hop_length=160)
    head = GruClassifierHead(input_size=INPUT_SIZE, hidden_dim=16, device="cpu")
    return BaseWakeModel(feature_extractor=extractor, classifier=head, device="cpu")


# ---- FfnClassifierHead.export_to_onnx ----

def test_ffn_export_to_onnx(tmp_path):
    head = FfnClassifierHead(input_size=INPUT_SIZE, hidden_dim=16, device="cpu")
    out = str(tmp_path / "ffn.onnx")
    head.export_to_onnx(out)
    assert (tmp_path / "ffn.onnx").exists()


def test_ffn_onnx_is_loadable(tmp_path):
    import onnxruntime as ort
    head = FfnClassifierHead(input_size=INPUT_SIZE, hidden_dim=16, device="cpu")
    out = str(tmp_path / "ffn.onnx")
    head.export_to_onnx(out)
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    # Run a dummy inference
    dummy = np.zeros((1, 10, INPUT_SIZE), dtype=np.float32)
    result = sess.run(None, {"input_features": dummy})
    assert result[0].shape == (1,)


# ---- GruClassifierHead.export_to_onnx ----

def test_gru_export_to_onnx(tmp_path):
    head = GruClassifierHead(input_size=INPUT_SIZE, hidden_dim=16, device="cpu")
    out = str(tmp_path / "gru.onnx")
    head.export_to_onnx(out)
    assert (tmp_path / "gru.onnx").exists()


def test_gru_onnx_is_loadable(tmp_path):
    import onnxruntime as ort
    head = GruClassifierHead(input_size=INPUT_SIZE, hidden_dim=16, device="cpu")
    out = str(tmp_path / "gru.onnx")
    head.export_to_onnx(out)
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    dummy = np.zeros((1, 10, INPUT_SIZE), dtype=np.float32)
    result = sess.run(None, {"input_features": dummy})
    assert result[0].shape == (1,)


# ---- CnnClassifierHead.forward with [B, F, T] input ----

def test_cnn_forward_bft_input():
    """CnnClassifierHead accepts [B, T, F] — consistent with BaseExtractor output contract."""
    head = CnnClassifierHead(input_size=INPUT_SIZE, device="cpu")
    head.eval()
    # [B=2, T=50, F=INPUT_SIZE]
    feats = torch.randn(2, 50, INPUT_SIZE)
    with torch.no_grad():
        out = head(feats)
    assert out.shape == (2,)
    assert not torch.isnan(out).any()


def test_cnn_export_to_onnx(tmp_path):
    head = CnnClassifierHead(input_size=INPUT_SIZE, device="cpu")
    out = str(tmp_path / "cnn.onnx")
    head.export_to_onnx(out)
    assert (tmp_path / "cnn.onnx").exists()


# ---- BaseWakeModel.infer ----

def test_base_wake_model_infer_returns_float():
    model = _make_mfcc_ffn_model()
    model.eval()
    audio = np.zeros(8000, dtype=np.float32)
    prob = model.infer(audio)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_base_wake_model_infer_rejects_2d():
    model = _make_mfcc_ffn_model()
    with pytest.raises(ValueError, match="1-D"):
        model.infer(np.zeros((2, 8000), dtype=np.float32))


def test_base_wake_model_infer_sine_wave():
    model = _make_mfcc_ffn_model()
    model.eval()
    t = np.linspace(0, 0.5, 8000)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    prob = model.infer(audio)
    assert 0.0 <= prob <= 1.0


# ---- BaseWakeModel save/load checkpoint round-trip ----

def test_model_checkpoint_round_trip(tmp_path):
    model = _make_mfcc_ffn_model()
    # Perturb weights to a known value
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.5)
    ckpt_path = str(tmp_path / "model.pt")
    model.save_checkpoint(ckpt_path)

    model2 = _make_mfcc_ffn_model()
    model2.load_checkpoint(ckpt_path)

    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-5)


def test_model_checkpoint_file_created(tmp_path):
    model = _make_mfcc_ffn_model()
    ckpt_path = str(tmp_path / "weights.pt")
    model.save_checkpoint(ckpt_path)
    assert (tmp_path / "weights.pt").exists()


# ---- BaseWakeModel forward with GRU head ----

def test_base_wake_model_gru_forward():
    model = _make_mfcc_gru_model()
    model.eval()
    wav = torch.randn(2, 8000)
    with torch.no_grad():
        out = model(wav)
    assert out.shape == (2,)
    assert not torch.isnan(out).any()


# ---- BaseWakeModel embed ----

def test_base_wake_model_embed_shape():
    model = _make_mfcc_ffn_model()
    model.eval()
    wav = torch.randn(3, 8000)
    with torch.no_grad():
        emb = model.embed(wav)
    assert emb.shape[0] == 3
    assert emb.ndim == 2
