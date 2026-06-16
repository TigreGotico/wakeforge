"""Tests for metric utility functions."""
import pytest
import torch


def _emb(n: int = 8, d: int = 16, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, d)


def _labels(n: int = 8) -> torch.Tensor:
    return torch.tensor([i % 2 for i in range(n)], dtype=torch.float32)


# ---- pairwise_distance ----

def test_pairwise_distance_shape():
    from ww_trainer.utils import pairwise_distance
    emb = _emb(6)
    d = pairwise_distance(emb)
    assert d.shape == (6, 6)


def test_pairwise_distance_zero_diagonal():
    from ww_trainer.utils import pairwise_distance
    emb = _emb(6)
    d = pairwise_distance(emb)
    diag = torch.diag(d)
    assert torch.allclose(diag, torch.zeros(6), atol=1e-4)


def test_pairwise_distance_symmetric():
    from ww_trainer.utils import pairwise_distance
    emb = _emb(6)
    d = pairwise_distance(emb)
    assert torch.allclose(d, d.t(), atol=1e-5)


def test_pairwise_distance_nonnegative():
    from ww_trainer.utils import pairwise_distance
    emb = _emb(6)
    d = pairwise_distance(emb)
    assert (d >= 0).all()


# ---- sample_triplets ----

def test_sample_triplets_validity():
    import random
    from ww_trainer.utils import sample_triplets
    random.seed(0)  # sample_triplets shuffles anchors via the global RNG
    emb = _emb(8)
    lab = _labels(8).long()
    a, p, n, frac = sample_triplets(lab, emb, margin=1.0)
    # With 4 positives + 4 negatives and seeded RNG, valid triplets always exist.
    assert a is not None, "expected valid triplets for a balanced, seeded batch"
    B = 8
    assert a.max() < B and p.max() < B and n.max() < B
    # anchor and positive must have same label
    for ai, pi in zip(a.tolist(), p.tolist()):
        assert lab[ai] == lab[pi]
    # anchor and negative must have different labels
    for ai, ni in zip(a.tolist(), n.tolist()):
        assert lab[ai] != lab[ni]
    # anchor != positive
    assert not torch.all(a == p)


def test_sample_triplets_no_valid_anchors():
    """All-negative batch returns None."""
    from ww_trainer.utils import sample_triplets
    emb = _emb(4)
    lab = torch.zeros(4, dtype=torch.long)  # no positives → no anchors
    a, p, n, frac = sample_triplets(lab, emb, margin=1.0)
    assert a is None


# ---- triplet_violation_fraction ----

def test_violation_fraction_zero_when_well_separated():
    """Perfect separation → 0 violations."""
    from ww_trainer.utils import triplet_violation_fraction
    # Construct embeddings where AP distance << AN distance
    pos = torch.zeros(4, 8)
    neg = torch.ones(4, 8) * 10.0
    emb = torch.cat([pos, neg])
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
    _, _, frac = triplet_violation_fraction(labels, emb, margin=1.0)
    assert frac == 0.0


def test_violation_fraction_nonzero_when_collapsed():
    """Collapsed embeddings → high violation rate."""
    from ww_trainer.utils import triplet_violation_fraction
    emb = torch.zeros(8, 8)  # all same → AP == AN == 0 → always violating
    labels = _labels(8).long()
    _, _, frac = triplet_violation_fraction(labels, emb, margin=1.0)
    assert frac > 0.0


# ---- get_hard_pair_distances ----

def test_get_hard_pair_distances_shape():
    from ww_trainer.utils import get_hard_pair_distances
    emb = _emb(8)
    lab = _labels(8).long()
    d_ap, d_an = get_hard_pair_distances(lab, emb)
    assert d_ap.shape == (8,)
    assert d_an.shape == (8,)
    assert (d_ap >= 0).all()
    assert (d_an >= 0).all()


# ---- pairwise_cosine_similarity ----

def test_pairwise_cosine_similarity_normalized():
    from ww_trainer.utils import pairwise_cosine_similarity
    import torch.nn.functional as F
    emb = F.normalize(_emb(6), p=2, dim=1)
    sim = pairwise_cosine_similarity(emb)
    assert sim.shape == (6, 6)
    # Diagonal should be ~1.0 for normalized vectors
    diag = torch.diag(sim)
    assert torch.allclose(diag, torch.ones(6), atol=1e-5)
