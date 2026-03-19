"""Tests for streaming inference (BaseWakeModel.forward_streaming and OnnxWakeWordInferencer.infer_streaming)."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from ww_trainer.feats import MfccExtractor, SlidingFeatureCacheTensor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel


@pytest.fixture
def mfcc_ffn_model():
    extractor = MfccExtractor(sr=16000, n_mfcc=13, n_mels=40, n_fft=400, hop_length=160)
    head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")
    model = BaseWakeModel(feature_extractor=extractor, classifier=head, sample_rate=16000, device="cpu")
    model.eval()
    return model


def test_forward_streaming_returns_float(mfcc_ffn_model):
    cache = SlidingFeatureCacheTensor(feature_dim=13, window_size=50)
    chunk = torch.randn(4000)  # 0.25s chunk
    prob = mfcc_ffn_model.forward_streaming(chunk, cache)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_forward_streaming_cache_grows(mfcc_ffn_model):
    cache = SlidingFeatureCacheTensor(feature_dim=13, window_size=50)
    chunk = torch.randn(4000)
    prev_len = cache.current_len
    mfcc_ffn_model.forward_streaming(chunk, cache)
    assert cache.current_len > prev_len


def test_forward_streaming_cache_saturates(mfcc_ffn_model):
    cache = SlidingFeatureCacheTensor(feature_dim=13, window_size=50)
    chunk = torch.randn(4000)
    # Run many chunks until cache saturates
    for _ in range(20):
        mfcc_ffn_model.forward_streaming(chunk, cache)
    assert cache.current_len == cache.window_size


def test_forward_streaming_10_calls(mfcc_ffn_model):
    cache = SlidingFeatureCacheTensor(feature_dim=13, window_size=50)
    chunk = torch.randn(4000)
    probs = []
    for _ in range(10):
        prob = mfcc_ffn_model.forward_streaming(chunk, cache)
        probs.append(prob)
    assert len(probs) == 10
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_onnx_inferencer_streaming(tmp_path):
    """OnnxWakeWordInferencer.infer_streaming works over multiple chunks."""
    extractor = MfccExtractor(sr=16000, n_mfcc=13, n_mels=40, n_fft=400, hop_length=160)
    head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")

    ext_path = str(tmp_path / "ext.onnx")
    head_path = str(tmp_path / "head.onnx")
    extractor.export_to_onnx(ext_path)
    head.export_to_onnx(head_path)

    from ww_trainer.inference import OnnxWakeWordInferencer
    inferencer = OnnxWakeWordInferencer(ext_path, head_path, sample_rate=16000, device="cpu")

    chunk = np.zeros(4000, dtype=np.float32)
    cache = None
    for _ in range(5):
        prob, cache = inferencer.infer_streaming(chunk, cache)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0
        assert cache.ndim == 2
