"""Smoke tests for dynamic negative weight scheduling."""
import math

import torch
import torch.nn as nn

from ww_trainer.loss import compute_neg_weight_schedule, LossManager


class TestComputeNegWeightSchedule:
    """Tests for the standalone schedule function."""

    def test_linear_start(self) -> None:
        assert compute_neg_weight_schedule(0, 100, 100.0, "linear") == 1.0

    def test_linear_end(self) -> None:
        assert compute_neg_weight_schedule(99, 100, 100.0, "linear") == 100.0

    def test_linear_midpoint(self) -> None:
        val = compute_neg_weight_schedule(49, 99, 100.0, "linear")
        assert 49.0 < val < 51.0

    def test_cosine_start(self) -> None:
        assert compute_neg_weight_schedule(0, 100, 100.0, "cosine") == 1.0

    def test_cosine_end(self) -> None:
        val = compute_neg_weight_schedule(99, 100, 100.0, "cosine")
        assert abs(val - 100.0) < 0.01

    def test_cosine_midpoint(self) -> None:
        val = compute_neg_weight_schedule(50, 101, 100.0, "cosine")
        expected = 1.0 + 99.0 * 0.5 * (1.0 - math.cos(math.pi * 50 / 100))
        assert abs(val - expected) < 0.01

    def test_single_step(self) -> None:
        assert compute_neg_weight_schedule(0, 1, 100.0, "linear") == 100.0

    def test_invalid_schedule(self) -> None:
        try:
            compute_neg_weight_schedule(0, 10, 100.0, "exponential")
            assert False, "Expected ValueError"
        except ValueError:
            pass


class TestLossManagerNegWeight:
    """Tests for LossManager neg weight integration."""

    def test_update_neg_weight_linear(self) -> None:
        lm = LossManager(
            [{"name": "bce", "weight": 1.0}],
            neg_weight_schedule="linear",
            max_neg_weight=50.0,
        )
        w = lm.update_neg_weight(0, 100)
        assert w == 1.0
        w = lm.update_neg_weight(99, 100)
        assert w == 50.0

    def test_no_schedule(self) -> None:
        lm = LossManager([{"name": "bce", "weight": 1.0}])
        w = lm.update_neg_weight(50, 100)
        assert w == 1.0  # unchanged

    def test_adjust_max_neg_weight(self) -> None:
        lm = LossManager(
            [{"name": "bce", "weight": 1.0}],
            neg_weight_schedule="linear",
            max_neg_weight=50.0,
        )
        lm.adjust_max_neg_weight(2.0)
        assert lm.max_neg_weight == 100.0
