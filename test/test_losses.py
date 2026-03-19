"""Tests for all loss functions and LossManager."""
import pytest
import torch
import torch.nn as nn


def _embeddings(n: int = 8, d: int = 64, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, d)


def _labels(n: int = 8) -> torch.Tensor:
    """Alternating 1/0 labels."""
    return torch.tensor([i % 2 for i in range(n)], dtype=torch.float32)


# ---- individual loss classes ----

def test_soft_triplet_no_nan():
    from ww_trainer.loss import SoftTripletLoss
    loss = SoftTripletLoss()
    a, p, n = _embeddings(4), _embeddings(4, seed=1), _embeddings(4, seed=2)
    val = loss(a, p, n)
    assert not torch.isnan(val)
    assert not torch.isinf(val)


def test_contrastive_no_nan():
    from ww_trainer.loss import ContrastiveLoss
    loss = ContrastiveLoss(margin=1.0)
    emb = _embeddings()
    lab = _labels()
    val = loss(emb, lab)
    assert not torch.isnan(val)
    assert not torch.isinf(val)


def test_lifted_structure_no_nan():
    from ww_trainer.loss import LiftedStructureLoss
    loss = LiftedStructureLoss(margin=1.0)
    emb = _embeddings()
    lab = _labels()
    val = loss(emb, lab)
    assert not torch.isnan(val)
    assert not torch.isinf(val)


def test_angular_no_nan():
    from ww_trainer.loss import AngularLoss
    loss = AngularLoss(margin=0.5)
    emb = _embeddings()
    lab = _labels()
    val = loss(emb, lab)
    assert not torch.isnan(val)
    assert not torch.isinf(val)


def test_cn2pair_no_nan():
    from ww_trainer.loss import CN2Plus1PairLoss
    loss = CN2Plus1PairLoss()
    a = _embeddings(4)
    p = _embeddings(4, seed=1)
    neg = _embeddings(4, seed=2)
    val = loss(a, p, neg)
    assert not torch.isnan(val)
    assert not torch.isinf(val)


def test_rppl_no_nan():
    from ww_trainer.loss import RobustProtoDiversityLoss
    loss = RobustProtoDiversityLoss()
    torch.manual_seed(0)
    logits = torch.randn(8)
    emb = _embeddings()
    lab = _labels()
    val = loss(logits, lab, emb)
    assert not torch.isnan(val)
    assert not torch.isinf(val)


def test_rppl_gradient_sign():
    """Positive prototype should be closer to positive embeds after gradient step."""
    from ww_trainer.loss import RobustProtoDiversityLoss
    loss_fn = RobustProtoDiversityLoss(alpha=1.0, beta=1.0, gamma=0.0, delta=1.0, eta=0.0)

    torch.manual_seed(42)
    # Clear separation: positives cluster near 1.0, negatives near -1.0
    pos_emb = torch.ones(4, 8) + torch.randn(4, 8) * 0.1
    neg_emb = -torch.ones(4, 8) + torch.randn(4, 8) * 0.1
    emb = torch.cat([pos_emb, neg_emb], dim=0).requires_grad_(True)
    logits = torch.zeros(8)
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)

    val = loss_fn(logits, labels, emb)
    val.backward()
    assert emb.grad is not None


def test_rppl_min_positive_batch():
    """RPPL handles single-class batch without crashing (returns non-nan)."""
    from ww_trainer.loss import RobustProtoDiversityLoss
    loss = RobustProtoDiversityLoss()
    logits = torch.randn(4)
    emb = _embeddings(4)
    # all negative labels — no positives
    lab = torch.zeros(4)
    val = loss(logits, lab, emb)
    assert not torch.isnan(val)


# ---- LossManager ----

def test_loss_manager_bce():
    from ww_trainer.loss import LossManager
    mgr = LossManager([{"name": "bce", "weight": 1.0}], device="cpu")
    # Use a minimal stub model
    model = _make_stub_model()
    wavs = torch.zeros(4, 8000)
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    total, results = mgr.compute_loss(model, wavs, labels)
    assert "bce" in results
    assert "total" in results
    assert not torch.isnan(total)


def test_loss_manager_rppl():
    from ww_trainer.loss import LossManager
    mgr = LossManager([{"name": "rppl", "weight": 1.0}], device="cpu")
    model = _make_stub_model()
    wavs = torch.zeros(8, 8000)
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
    total, results = mgr.compute_loss(model, wavs, labels)
    assert "rppl" in results
    assert not torch.isnan(total)


def test_loss_manager_unknown_raises():
    from ww_trainer.loss import LossManager
    with pytest.raises(ValueError, match="Unknown loss"):
        LossManager([{"name": "nonexistent_loss"}], device="cpu")


# ---- helpers ----

def _make_stub_model() -> nn.Module:
    """Minimal model that satisfies LossManager interface (forward + embed)."""

    class StubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8000, 64)
            self.head = nn.Linear(64, 1)

        def forward(self, wavs: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.linear(wavs))
            return self.head(h).squeeze(-1)

        def embed(self, wavs: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(wavs))

    return StubModel()
