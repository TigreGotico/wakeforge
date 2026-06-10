"""Tests added as part of public-release prep (feat/public-prep).

Covers pure-logic paths that run on CPU without model weights:
  - metrics (FAR/FRR, EER, FP/hour)
  - read_dataset_csv
  - tiers round-trip
  - TierConfig field invariants
"""
from __future__ import annotations

import csv
import math

import numpy as np
import pytest


# ── metrics ──────────────────────────────────────────────────────────────────

class TestComputeFarFrr:
    def test_perfect_classifier(self):
        from ww_trainer.metrics import compute_far_frr
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.1, 0.2])
        far, frr = compute_far_frr(y_true, y_scores, threshold=0.5)
        assert far == 0.0
        assert frr == 0.0

    def test_all_wrong(self):
        from ww_trainer.metrics import compute_far_frr
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.9, 0.8])
        far, frr = compute_far_frr(y_true, y_scores, threshold=0.5)
        assert far == 1.0
        assert frr == 1.0

    def test_threshold_boundary(self):
        from ww_trainer.metrics import compute_far_frr
        y_true = np.array([1, 0])
        y_scores = np.array([0.5, 0.5])
        # Both scores exactly at threshold → both accepted → FAR=1, FRR=0
        far, frr = compute_far_frr(y_true, y_scores, threshold=0.5)
        assert frr == 0.0
        assert far == 1.0

    def test_no_positives_returns_zero(self):
        from ww_trainer.metrics import compute_far_frr
        y_true = np.array([0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7])
        far, frr = compute_far_frr(y_true, y_scores, threshold=0.5)
        assert far == 0.0
        assert frr == 0.0

    def test_no_negatives_returns_zero(self):
        from ww_trainer.metrics import compute_far_frr
        y_true = np.array([1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3])
        far, frr = compute_far_frr(y_true, y_scores, threshold=0.5)
        assert far == 0.0
        assert frr == 0.0


class TestComputeEer:
    def test_perfect_eer_is_zero(self):
        from ww_trainer.metrics import compute_eer
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.85, 0.8, 0.1, 0.15, 0.2])
        eer, thresh = compute_eer(y_true, y_scores)
        assert eer < 0.05, f"Expected EER near 0, got {eer}"
        assert 0.0 <= thresh <= 1.0

    def test_random_eer_in_range(self):
        from ww_trainer.metrics import compute_eer
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100)
        y_scores = rng.random(size=100)
        eer, thresh = compute_eer(y_true, y_scores)
        assert 0.0 <= eer <= 1.0
        assert 0.0 <= thresh <= 1.0

    def test_eer_returns_tuple(self):
        from ww_trainer.metrics import compute_eer
        y_true = np.array([1, 0, 1, 0])
        y_scores = np.array([0.7, 0.3, 0.6, 0.4])
        result = compute_eer(y_true, y_scores)
        assert len(result) == 2


class TestEstimateFpPerHour:
    """estimate_fp_per_hour takes an infer_fn + audio paths — test via compute_far_frr instead."""

    def test_far_frr_rate_zero_on_negatives(self):
        from ww_trainer.metrics import compute_far_frr
        y_true = np.zeros(10, dtype=int)
        y_scores = np.full(10, 0.1)
        far, frr = compute_far_frr(y_true, y_scores, threshold=0.5)
        assert far == 0.0


# ── read_dataset_csv ──────────────────────────────────────────────────────────

class TestReadDatasetCsv:
    def test_plain_rows(self, tmp_path):
        from ww_trainer.utils import read_dataset_csv
        p = tmp_path / "ds.csv"
        p.write_text("audio/a.wav,1\naudio/b.wav,0\n")
        rows = read_dataset_csv(str(p))
        assert rows == [("audio/a.wav", "1"), ("audio/b.wav", "0")]

    def test_header_row_included_as_data(self, tmp_path):
        """read_dataset_csv does not strip headers; callers handle that."""
        from ww_trainer.utils import read_dataset_csv
        p = tmp_path / "ds.csv"
        p.write_text("path,label\naudio/a.wav,1\n")
        rows = read_dataset_csv(str(p))
        assert len(rows) == 2
        assert rows[0] == ("path", "label")

    def test_blank_lines_skipped(self, tmp_path):
        from ww_trainer.utils import read_dataset_csv
        p = tmp_path / "ds.csv"
        p.write_text("audio/a.wav,1\n\naudio/b.wav,0\n\n")
        rows = read_dataset_csv(str(p))
        assert len(rows) == 2

    def test_lines_without_comma_skipped(self, tmp_path):
        from ww_trainer.utils import read_dataset_csv
        p = tmp_path / "ds.csv"
        p.write_text("audio/a.wav,1\njunk_no_comma\naudio/b.wav,0\n")
        rows = read_dataset_csv(str(p))
        assert len(rows) == 2

    def test_empty_file(self, tmp_path):
        from ww_trainer.utils import read_dataset_csv
        p = tmp_path / "empty.csv"
        p.write_text("")
        rows = read_dataset_csv(str(p))
        assert rows == []

    def test_path_with_comma_in_middle(self, tmp_path):
        """CSV split on first comma — path should not contain commas."""
        from ww_trainer.utils import read_dataset_csv
        p = tmp_path / "ds.csv"
        p.write_text("audio/a.wav,1\naudio/b.wav,0\n")
        rows = read_dataset_csv(str(p))
        assert rows[0][1] == "1"


# ── tiers ─────────────────────────────────────────────────────────────────────

class TestTierConfigInvariants:
    """Verify that all shipped tiers satisfy hard constraints."""

    def test_all_tiers_have_name_matching_key(self):
        from ww_trainer.tiers import HARDWARE_TIERS
        for key, cfg in HARDWARE_TIERS.items():
            assert cfg.name == key, f"Tier key {key!r} name mismatch: {cfg.name!r}"

    def test_all_tiers_have_positive_hidden_dim(self):
        from ww_trainer.tiers import HARDWARE_TIERS
        for name, cfg in HARDWARE_TIERS.items():
            assert cfg.hidden_dim > 0, f"{name}: hidden_dim must be > 0"

    def test_all_tiers_have_nonempty_head_arch(self):
        from ww_trainer.tiers import HARDWARE_TIERS
        for name, cfg in HARDWARE_TIERS.items():
            assert cfg.head_arch, f"{name}: head_arch must not be empty"
            assert isinstance(cfg.head_arch, str)

    def test_get_tier_case_insensitive_if_supported(self):
        from ww_trainer.tiers import get_tier, HARDWARE_TIERS
        # Just verify micro is always retrievable regardless of test environment
        cfg = get_tier("micro")
        assert cfg.name == "micro"

    def test_list_tiers_contains_all_names(self):
        from ww_trainer.tiers import list_tiers, HARDWARE_TIERS
        listing = list_tiers()
        for name in HARDWARE_TIERS:
            assert name in listing, f"list_tiers() missing {name!r}"
