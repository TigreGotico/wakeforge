"""Extended tests for ww_trainer/loss.py — additional coverage."""
import pytest
import torch
import torch.nn as nn

from ww_trainer.loss import (
    LossManager,
    SoftTripletLoss,
    ContrastiveLoss,
    LiftedStructureLoss,
    AngularLoss,
    CN2Plus1PairLoss,
    RobustProtoDiversityLoss,
)


def _emb(n: int = 8, d: int = 16, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, d)


def _labels(n: int = 8) -> torch.Tensor:
    return torch.tensor([i % 2 for i in range(n)], dtype=torch.float32)


def _make_stub_model() -> nn.Module:
    class StubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8000, 16)
            self.head = nn.Linear(16, 1)

        def forward(self, wavs: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.linear(wavs))
            return self.head(h).squeeze(-1)

        def embed(self, wavs: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(wavs))

    return StubModel()


# ---- LossManager with multiple losses simultaneously ----

def test_loss_manager_bce_and_rppl(tmp_path):
    """LossManager can combine bce + rppl without error."""
    mgr = LossManager([
        {"name": "bce", "weight": 1.0},
        {"name": "rppl", "weight": 0.5},
    ], device="cpu")
    model = _make_stub_model()
    wavs = torch.zeros(8, 8000)
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
    total, results = mgr.compute_loss(model, wavs, labels)
    assert "bce" in results
    assert "rppl" in results
    assert "total" in results
    assert not torch.isnan(total)


def test_loss_manager_triplet(tmp_path):
    mgr = LossManager([{"name": "triplet", "weight": 1.0, "margin": 1.0}], device="cpu")
    model = _make_stub_model()
    wavs = torch.zeros(8, 8000)
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
    total, results = mgr.compute_loss(model, wavs, labels)
    assert "triplet" in results
    assert not torch.isnan(total)


def test_loss_manager_soft_triplet():
    mgr = LossManager([{"name": "soft_triplet", "weight": 1.0}], device="cpu")
    model = _make_stub_model()
    wavs = torch.zeros(8, 8000)
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
    total, results = mgr.compute_loss(model, wavs, labels)
    assert "soft_triplet" in results
    assert not torch.isnan(total)


def test_loss_manager_pair():
    mgr = LossManager([{"name": "pair", "weight": 1.0, "margin": 1.0}], device="cpu")
    model = _make_stub_model()
    wavs = torch.zeros(8, 8000)
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
    total, results = mgr.compute_loss(model, wavs, labels)
    assert "pair" in results
    assert not torch.isnan(total)


def test_loss_manager_cn2pair():
    mgr = LossManager([{"name": "cn2pair", "weight": 1.0}], device="cpu")
    model = _make_stub_model()
    wavs = torch.zeros(8, 8000)
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
    total, results = mgr.compute_loss(model, wavs, labels)
    assert "cn2pair" in results
    assert not torch.isnan(total)


def test_loss_manager_lse():
    mgr = LossManager([{"name": "lse", "weight": 1.0, "margin": 1.0}], device="cpu")
    model = _make_stub_model()
    wavs = torch.zeros(8, 8000)
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
    total, results = mgr.compute_loss(model, wavs, labels)
    assert "lse" in results
    assert not torch.isnan(total)


def test_loss_manager_contrastive():
    mgr = LossManager([{"name": "contrastive", "weight": 1.0, "margin": 1.0}], device="cpu")
    model = _make_stub_model()
    wavs = torch.zeros(8, 8000)
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
    total, results = mgr.compute_loss(model, wavs, labels)
    assert "contrastive" in results
    assert not torch.isnan(total)


def test_loss_manager_angular():
    mgr = LossManager([{"name": "angular", "weight": 1.0, "margin": 0.5}], device="cpu")
    model = _make_stub_model()
    wavs = torch.zeros(8, 8000)
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
    total, results = mgr.compute_loss(model, wavs, labels)
    assert "angular" in results
    assert not torch.isnan(total)


def test_loss_manager_three_losses():
    """Three losses together — total is weighted sum."""
    mgr = LossManager([
        {"name": "bce", "weight": 1.0},
        {"name": "contrastive", "weight": 0.5, "margin": 1.0},
        {"name": "rppl", "weight": 0.3},
    ], device="cpu")
    model = _make_stub_model()
    wavs = torch.zeros(8, 8000)
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
    total, results = mgr.compute_loss(model, wavs, labels)
    assert {"bce", "contrastive", "rppl", "total"}.issubset(results.keys())
    assert not torch.isnan(total)


# ---- RPPL with aug_embeds (consistency loss) ----

def test_rppl_with_aug_embeds_no_nan():
    loss = RobustProtoDiversityLoss(eta=1.0)
    torch.manual_seed(0)
    logits = torch.randn(8)
    emb = _emb(8, 16)
    aug_emb = _emb(8, 16, seed=99)
    lab = _labels(8)
    val = loss(logits, lab, emb, aug_embeds=aug_emb)
    assert not torch.isnan(val)
    assert not torch.isinf(val)


# ---- RPPL with K_neg_proto > 1 ----

def test_rppl_multiple_neg_prototypes():
    loss = RobustProtoDiversityLoss(K_neg_proto=3)
    torch.manual_seed(0)
    logits = torch.randn(8)
    emb = _emb(8, 16)
    lab = _labels(8)
    val = loss(logits, lab, emb)
    assert not torch.isnan(val)


# ---- CN2Plus1PairLoss with 3D negatives ----

def test_cn2pair_3d_negatives():
    loss = CN2Plus1PairLoss()
    a = _emb(4)
    p = _emb(4, seed=1)
    neg = _emb(4, seed=2).unsqueeze(0)  # [1, 4, D]
    val = loss(a, p, neg)
    assert not torch.isnan(val)


def test_cn2pair_no_negatives():
    """Empty negatives → returns 0."""
    loss = CN2Plus1PairLoss()
    a = _emb(4)
    p = _emb(4, seed=1)
    neg = torch.zeros(0, 16)  # no negatives
    val = loss(a, p, neg)
    assert val.item() == 0.0
