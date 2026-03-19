"""Tests for classifier head forward passes."""
import pytest
import torch


INPUT_SIZE = 768


@pytest.fixture
def feats():
    torch.manual_seed(0)
    return torch.randn(2, 50, INPUT_SIZE)


def test_ffn_forward(feats):
    from ww_trainer.model import FfnClassifierHead
    head = FfnClassifierHead(input_size=INPUT_SIZE, device="cpu")
    head.eval()
    with torch.no_grad():
        out = head(feats)
    assert out.shape == (2,), f"Expected (2,), got {out.shape}"
    assert not torch.isnan(out).any()


def test_ffn_embed(feats):
    from ww_trainer.model import FfnClassifierHead
    head = FfnClassifierHead(input_size=INPUT_SIZE, hidden_dim=64, device="cpu")
    head.eval()
    with torch.no_grad():
        emb = head.embed(feats)
    assert emb.shape == (2, 64)


def test_cnn_forward(feats):
    from ww_trainer.model import CnnClassifierHead
    # CNN expects [B, F, T] — transpose from [B, T, F]
    head = CnnClassifierHead(input_size=INPUT_SIZE, device="cpu")
    head.eval()
    with torch.no_grad():
        out = head(feats)  # head handles [B, T, F] internally
    assert out.shape == (2,), f"Expected (2,), got {out.shape}"
    assert not torch.isnan(out).any()


def test_cnn_embed(feats):
    from ww_trainer.model import CnnClassifierHead
    head = CnnClassifierHead(input_size=INPUT_SIZE, linear_dim=64, device="cpu")
    head.eval()
    with torch.no_grad():
        emb = head.embed(feats)  # head handles [B, T, F] internally
    assert emb.shape == (2, 64)


def test_gru_forward(feats):
    from ww_trainer.model import GruClassifierHead
    head = GruClassifierHead(input_size=INPUT_SIZE, device="cpu")
    head.eval()
    with torch.no_grad():
        out = head(feats)
    assert out.shape == (2,), f"Expected (2,), got {out.shape}"
    assert not torch.isnan(out).any()


def test_gru_embed(feats):
    from ww_trainer.model import GruClassifierHead
    head = GruClassifierHead(input_size=INPUT_SIZE, linear_dim=64, device="cpu")
    head.eval()
    with torch.no_grad():
        emb = head.embed(feats)
    assert emb.shape == (2, 64)


def test_gru_transpose_detection():
    """GRU auto-detects and transposes [B, F, T] to [B, T, F]."""
    from ww_trainer.model import GruClassifierHead
    head = GruClassifierHead(input_size=INPUT_SIZE, device="cpu")
    head.eval()
    feats_bft = torch.randn(2, INPUT_SIZE, 50)  # [B, F, T]
    with torch.no_grad():
        out = head(feats_bft)
    assert out.shape == (2,)


def test_gru_ambiguous_shape_raises():
    """GRU raises ValueError when D1 == D2 == input_size."""
    from ww_trainer.model import GruClassifierHead
    head = GruClassifierHead(input_size=50, device="cpu")
    ambiguous = torch.randn(2, 50, 50)  # both dims equal input_size=50
    with pytest.raises(ValueError, match="Ambiguous"):
        head(ambiguous)
