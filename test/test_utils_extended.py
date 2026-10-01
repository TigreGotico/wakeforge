"""Extended tests for ww_trainer/utils.py — compute_metric_statistics, sample_semihard_triplets."""
import pytest
import torch

from ww_trainer.utils import compute_metric_statistics, sample_semihard_triplets, sample_triplets


def _emb(n: int = 8, d: int = 16, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, d)


def _labels(n: int = 8) -> torch.Tensor:
    return torch.tensor([i % 2 for i in range(n)], dtype=torch.float32)


# ---- compute_metric_statistics ----

def test_compute_metric_statistics_returns_three_floats():
    emb = _emb(8)
    lab = _labels(8).long()
    ap, an, vf = compute_metric_statistics(lab, emb, margin=1.0)
    assert isinstance(ap, float)
    assert isinstance(an, float)
    assert isinstance(vf, float)


def test_compute_metric_statistics_all_negative_labels():
    """All negative labels → no valid anchors → returns (0, 0, 0)."""
    emb = _emb(4)
    lab = torch.zeros(4, dtype=torch.long)
    ap, an, vf = compute_metric_statistics(lab, emb)
    assert ap == 0.0
    assert an == 0.0
    assert vf == 0.0


def test_compute_metric_statistics_ap_nonneg():
    emb = _emb(8)
    lab = _labels(8).long()
    ap, an, vf = compute_metric_statistics(lab, emb)
    assert ap >= 0.0
    assert an >= 0.0


# ---- sample_semihard_triplets ----

def test_sample_semihard_triplets_validity():
    import random
    random.seed(0)  # mining shuffles anchors via the global RNG
    emb = _emb(8)
    lab = _labels(8).long()
    a, p, n, frac = sample_semihard_triplets(lab, emb, margin=1.0)
    assert a is not None, "expected semihard triplets for a balanced, seeded batch"
    B = 8
    assert a.max() < B and p.max() < B and n.max() < B
    for ai, pi in zip(a.tolist(), p.tolist()):
        assert lab[ai] == lab[pi]
    for ai, ni in zip(a.tolist(), n.tolist()):
        assert lab[ai] != lab[ni]


def test_sample_semihard_triplets_all_negative():
    """All-negative batch → no anchors → returns None."""
    emb = _emb(4)
    lab = torch.zeros(4, dtype=torch.long)
    a, p, n, frac = sample_semihard_triplets(lab, emb, margin=1.0)
    assert a is None


# ---- sample_triplets with different mining_type ----

def test_sample_triplets_hard_mining():
    import random
    random.seed(0)
    emb = _emb(8)
    lab = _labels(8).long()
    a, p, n, frac = sample_triplets(lab, emb, margin=1.0, mining_type="hard")
    assert a is not None


def test_sample_triplets_random_mining():
    import random
    random.seed(0)
    emb = _emb(8)
    lab = _labels(8).long()
    a, p, n, frac = sample_triplets(lab, emb, margin=1.0, mining_type="random")
    assert a is not None


def test_sample_triplets_semihard_mining():
    emb = _emb(8)
    lab = _labels(8).long()
    result = sample_triplets(lab, emb, margin=1.0, mining_type="semihard")
    # Should return a 4-tuple always
    assert len(result) == 4
