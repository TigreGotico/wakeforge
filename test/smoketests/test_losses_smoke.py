"""Smoke tests: every loss registered in LossManager computes without error."""
import pytest
import torch
import torch.nn as nn

from ww_trainer.feats import MfccExtractor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel


def _make_model(hidden_dim: int = 32) -> BaseWakeModel:
    ext = MfccExtractor(n_mfcc=13)
    head = FfnClassifierHead(input_size=13, hidden_dim=hidden_dim, device="cpu")
    return BaseWakeModel(ext, head, device="cpu")


def _dummy_batch() -> tuple:
    wavs = [torch.randn(16000) for _ in range(8)]
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.long)
    return wavs, labels


# Classification losses — only need logits and labels
CLASSIFICATION_LOSSES = [
    "bce",
    "focal",
    "label_smoothing_bce",
]

# Metric losses — need embeddings and pairs/triplets in batch
METRIC_LOSSES = [
    "contrastive",
    "angular",
    "lse",
    "ntxent",
    "supcon",
    "multi_similarity",
]

# Losses that need embed_dim
EMBED_DIM_LOSSES = [
    "arcface",
    "center",
    "proxy_nca",
]

# Triplet-based losses
TRIPLET_LOSSES = [
    "triplet",
    "soft_triplet",
]

# Special losses
SPECIAL_LOSSES = [
    "pair",
    "cn2pair",
    "rppl",
]


@pytest.mark.parametrize("loss_name", CLASSIFICATION_LOSSES)
def test_classification_loss(loss_name: str) -> None:
    from ww_trainer.loss import LossManager
    model = _make_model()
    wavs, labels = _dummy_batch()
    mgr = LossManager([{"name": loss_name, "weight": 1.0}], device="cpu")
    total, parts = mgr.compute_loss(model, wavs, labels)
    assert not torch.isnan(total)
    assert loss_name in parts


@pytest.mark.parametrize("loss_name", METRIC_LOSSES)
def test_metric_loss(loss_name: str) -> None:
    from ww_trainer.loss import LossManager
    model = _make_model()
    wavs, labels = _dummy_batch()
    mgr = LossManager([{"name": loss_name, "weight": 1.0}], device="cpu")
    total, parts = mgr.compute_loss(model, wavs, labels)
    assert not torch.isnan(total)


@pytest.mark.parametrize("loss_name", EMBED_DIM_LOSSES)
def test_embed_dim_loss(loss_name: str) -> None:
    from ww_trainer.loss import LossManager
    model = _make_model(hidden_dim=32)
    wavs, labels = _dummy_batch()
    mgr = LossManager(
        [{"name": loss_name, "weight": 1.0, "embed_dim": 32}],
        device="cpu",
    )
    total, parts = mgr.compute_loss(model, wavs, labels)
    assert not torch.isnan(total)


@pytest.mark.parametrize("loss_name", TRIPLET_LOSSES)
def test_triplet_loss(loss_name: str) -> None:
    from ww_trainer.loss import LossManager
    model = _make_model()
    wavs, labels = _dummy_batch()
    mgr = LossManager([{"name": loss_name, "weight": 1.0}], device="cpu")
    total, parts = mgr.compute_loss(model, wavs, labels)
    assert not torch.isnan(total)


@pytest.mark.parametrize("loss_name", SPECIAL_LOSSES)
def test_special_loss(loss_name: str) -> None:
    from ww_trainer.loss import LossManager
    model = _make_model()
    wavs, labels = _dummy_batch()
    mgr = LossManager([{"name": loss_name, "weight": 1.0}], device="cpu")
    total, parts = mgr.compute_loss(model, wavs, labels)
    assert not torch.isnan(total)


def test_size_aware_loss() -> None:
    from ww_trainer.loss import LossManager
    model = _make_model()
    wavs, labels = _dummy_batch()
    mgr = LossManager(
        [{"name": "size_aware", "weight": 1.0, "param_budget": 1024}],
        device="cpu",
    )
    total, parts = mgr.compute_loss(model, wavs, labels)
    assert not torch.isnan(total)
    assert "size_aware" in parts


def test_multi_loss_combination() -> None:
    """Test combining BCE + contrastive losses."""
    from ww_trainer.loss import LossManager
    model = _make_model()
    wavs, labels = _dummy_batch()
    mgr = LossManager(
        [
            {"name": "bce", "weight": 1.0},
            {"name": "contrastive", "weight": 0.5},
        ],
        device="cpu",
    )
    total, parts = mgr.compute_loss(model, wavs, labels)
    assert not torch.isnan(total)
    assert "bce" in parts
    assert "contrastive" in parts
