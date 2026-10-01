"""Tests for ww_trainer/metrics.py — detection evaluation metrics."""
import numpy as np
import pytest

from ww_trainer.metrics import (
    compute_far_frr,
    compute_eer,
    det_curve,
    find_optimal_threshold,
    classification_report,
    DetectionReport,
)


class TestComputeFarFrr:
    def test_perfect_predictions(self):
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.1, 0.2])
        far, frr = compute_far_frr(y_true, y_scores, 0.5)
        assert far == 0.0
        assert frr == 0.0

    def test_all_false_accepts(self):
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.9, 0.9, 0.9, 0.9])
        far, frr = compute_far_frr(y_true, y_scores, 0.5)
        assert far == 1.0  # both negatives accepted
        assert frr == 0.0  # no false rejects

    def test_all_false_rejects(self):
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.1, 0.1, 0.1, 0.1])
        far, frr = compute_far_frr(y_true, y_scores, 0.5)
        assert frr == 1.0  # all positives rejected
        assert far == 0.0  # no false accepts

    def test_no_positives(self):
        far, frr = compute_far_frr(np.array([0, 0]), np.array([0.5, 0.5]), 0.5)
        assert far == 0.0 and frr == 0.0


class TestComputeEer:
    def test_perfect_separation(self):
        y_true = np.array([1]*50 + [0]*50)
        y_scores = np.array([0.9]*50 + [0.1]*50)
        eer, thresh = compute_eer(y_true, y_scores)
        assert eer < 0.05
        assert 0.1 < thresh < 0.9

    def test_random_scores(self):
        np.random.seed(42)
        y_true = np.array([1]*100 + [0]*100)
        y_scores = np.random.rand(200)
        eer, thresh = compute_eer(y_true, y_scores)
        # Random scores → EER near 50%
        assert 0.3 < eer < 0.7


class TestDetCurve:
    def test_returns_arrays(self):
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.1, 0.2])
        fars, frrs, thresholds = det_curve(y_true, y_scores, n_thresholds=100)
        assert len(fars) == 100
        assert len(frrs) == 100
        assert len(thresholds) == 100
        assert all(0 <= f <= 1 for f in fars)
        assert all(0 <= f <= 1 for f in frrs)


class TestFindOptimalThreshold:
    def test_f1_criterion(self):
        y_true = np.array([1]*50 + [0]*50)
        y_scores = np.array([0.8]*50 + [0.2]*50)
        thresh, f1 = find_optimal_threshold(y_true, y_scores, criterion="f1")
        assert 0.2 < thresh < 0.8
        assert f1 > 0.9

    def test_eer_criterion(self):
        y_true = np.array([1]*50 + [0]*50)
        y_scores = np.array([0.8]*50 + [0.2]*50)
        thresh, eer = find_optimal_threshold(y_true, y_scores, criterion="eer")
        assert eer < 0.1

    def test_far_constraint(self):
        y_true = np.array([1]*100 + [0]*100)
        y_scores = np.concatenate([np.random.uniform(0.6, 1.0, 100),
                                    np.random.uniform(0.0, 0.4, 100)])
        thresh, recall = find_optimal_threshold(
            y_true, y_scores, criterion="far", max_far=0.01
        )
        far, _ = compute_far_frr(y_true, y_scores, thresh)
        assert far <= 0.02  # small tolerance


class TestClassificationReport:
    def test_returns_dataclass(self):
        y_true = np.array([1]*50 + [0]*50)
        y_scores = np.array([0.9]*50 + [0.1]*50)
        report = classification_report(y_true, y_scores)
        assert isinstance(report, DetectionReport)
        assert report.eer < 0.1
        assert report.f1 > 0.9
        assert report.n_positive == 50
        assert report.n_negative == 50

    def test_custom_threshold(self):
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.1, 0.2])
        report = classification_report(y_true, y_scores, threshold=0.5)
        assert report.threshold == 0.5
        assert report.accuracy > 0.99

    def test_far_at_frr_targets(self):
        y_true = np.array([1]*100 + [0]*100)
        y_scores = np.array([0.9]*100 + [0.1]*100)
        report = classification_report(y_true, y_scores, frr_targets=(0.05, 0.10))
        assert 0.05 in report.far_at_frr
        assert 0.10 in report.far_at_frr

    def test_auc_near_one_for_separable(self):
        y_true = np.array([1]*100 + [0]*100)
        y_scores = np.array([0.95]*100 + [0.05]*100)
        report = classification_report(y_true, y_scores)
        assert report.auc > 0.9
