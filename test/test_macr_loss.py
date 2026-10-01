"""Tests for MarginAwareContrastiveLoss (MACR, Zhang et al., Interspeech 2026).

Expected values are worked out by hand from 2-D embeddings whose cosines are
known exactly, never read back from the loss.
"""
import math

import pytest
import torch
import torch.nn as nn

from ww_trainer.loss import LossManager, MarginAwareContrastiveLoss

INV_SQRT2 = 1.0 / math.sqrt(2.0)

# Wake: (1, 0) and (0, 1) -> cosine 0, so the pull term is 1 - 0 = 1.
# Non-wake: (1, 1)/sqrt(2) has cosine 1/sqrt(2) with both wake clips;
#           (-1, 0) has cosine -1 and 0, below any positive margin.
EMBEDS = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
LABELS = torch.tensor([1, 1, 0, 0])
PULL = 1.0
EXCESS = INV_SQRT2 - 0.4  # the only positive hinge, once per wake anchor


def test_value_without_hard_negatives():
    loss = MarginAwareContrastiveLoss(margin=0.4)(EMBEDS, LABELS)
    # Each wake anchor averages [EXCESS, 0] over the two negatives.
    assert loss.item() == pytest.approx(PULL + EXCESS / 2, abs=1e-6)


def test_hard_mask_scales_only_the_hard_pairs():
    hard = torch.tensor([False, False, True, False])
    loss = MarginAwareContrastiveLoss(margin=0.4, hard_weight=2.0)(EMBEDS, LABELS, hard_mask=hard)
    assert loss.item() == pytest.approx(PULL + 2.0 * EXCESS / 2, abs=1e-6)


def test_logits_mark_confident_negatives_as_hard():
    logits = torch.tensor([4.0, 4.0, 3.0, -3.0])  # sigmoid(3) = 0.95 >= 0.5
    loss = MarginAwareContrastiveLoss(margin=0.4, hard_weight=3.0)(EMBEDS, LABELS, logits=logits)
    assert loss.item() == pytest.approx(PULL + 3.0 * EXCESS / 2, abs=1e-6)


def test_negatives_below_margin_cost_nothing_regardless_of_their_own_spread():
    """MACR's push term looks only at each wake/non-wake cosine. Whether the
    non-wake clips sit apart or on top of each other does not matter, since
    MACR never pulls non-wake clips towards each other or anchors on them."""
    wake = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    spread = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    together = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    labels = torch.tensor([1, 1, 0, 0])
    macr = MarginAwareContrastiveLoss(margin=0.4)

    macr_spread = macr(torch.cat([wake, spread]), labels)
    macr_together = macr(torch.cat([wake, together]), labels)
    assert macr_spread.item() == pytest.approx(0.0, abs=1e-6)
    assert macr_together.item() == pytest.approx(0.0, abs=1e-6)


def test_gradient_pushes_imposter_away_from_wake():
    wake = torch.tensor([[1.0, 0.0], [1.0, 0.2]])
    imposter = torch.tensor([[1.0, 1.0]])
    embeds = torch.cat([wake, imposter]).requires_grad_(True)
    MarginAwareContrastiveLoss(margin=0.4)(embeds, torch.tensor([1, 1, 0])).backward()
    moved = embeds[2:3] - 0.1 * embeds.grad[2:3]
    before = nn.functional.cosine_similarity(imposter, wake[0:1]).item()
    after = nn.functional.cosine_similarity(moved, wake[0:1]).item()
    assert after < before


def test_batch_without_wake_is_zero_with_grad():
    embeds = torch.randn(4, 8, requires_grad=True)
    loss = MarginAwareContrastiveLoss()(embeds, torch.zeros(4, dtype=torch.long))
    assert loss.item() == 0.0
    loss.backward()
    assert embeds.grad is not None


class _FixedModel(nn.Module):
    """Returns fixed embeddings and logits so the manager's value is known."""

    def __init__(self, embeds: torch.Tensor, logits: torch.Tensor) -> None:
        super().__init__()
        self.embeds = nn.Parameter(embeds.clone())
        self.logits = nn.Parameter(logits.clone())

    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        return self.logits

    def embed(self, wavs: torch.Tensor) -> torch.Tensor:
        return self.embeds


def test_loss_manager_dispatches_macr_with_logit_hardness():
    mgr = LossManager([{"name": "macr", "weight": 0.5, "hard_weight": 2.0}], device="cpu")
    model = _FixedModel(EMBEDS, torch.tensor([4.0, 4.0, 3.0, -3.0]))
    total, results = mgr.compute_loss(model, torch.zeros(4, 16), LABELS.float())
    expected = PULL + 2.0 * EXCESS / 2
    assert results["macr"] == pytest.approx(expected, abs=1e-6)
    assert total.item() == pytest.approx(0.5 * expected, abs=1e-6)
    total.backward()
    assert model.embeds.grad is not None
