"""Standalone evaluation metrics for wake word detection.

Computes detection-specific metrics: EER, FAR/FRR at threshold,
optimal threshold selection, and DET curve data.

Usage::

    from ww_trainer.metrics import (
        compute_eer,
        compute_far_frr,
        find_optimal_threshold,
        det_curve,
        classification_report,
    )

    report = classification_report(y_true, y_scores)
    print(f"EER: {report['eer']:.4f} at threshold {report['eer_threshold']:.4f}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DetectionReport:
    """Complete detection evaluation report.

    Attributes:
        eer: Equal Error Rate (where FAR == FRR).
        eer_threshold: Threshold at EER.
        far: False Accept Rate at the operating threshold.
        frr: False Reject Rate at the operating threshold.
        threshold: Operating threshold (optimized or user-specified).
        accuracy: Classification accuracy at threshold.
        precision: Precision at threshold.
        recall: Recall (= 1 - FRR) at threshold.
        f1: F1 score at threshold.
        auc: Area under ROC curve.
        n_positive: Number of positive samples.
        n_negative: Number of negative samples.
        far_at_frr: Dict mapping FRR targets to achieved FAR values.
    """
    eer: float = 0.0
    eer_threshold: float = 0.5
    far: float = 0.0
    frr: float = 0.0
    threshold: float = 0.5
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc: float = 0.0
    n_positive: int = 0
    n_negative: int = 0
    far_at_frr: dict = field(default_factory=dict)


def compute_far_frr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    """Compute False Accept Rate and False Reject Rate at a threshold.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_scores: Predicted probabilities in [0, 1].
        threshold: Decision threshold.

    Returns:
        (FAR, FRR) tuple. FAR = false positives / total negatives,
        FRR = false negatives / total positives.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    preds = (y_scores >= threshold).astype(int)

    positives = y_true == 1
    negatives = y_true == 0
    n_pos = positives.sum()
    n_neg = negatives.sum()

    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.0

    false_accepts = ((preds == 1) & negatives).sum()
    false_rejects = ((preds == 0) & positives).sum()

    far = float(false_accepts / n_neg)
    frr = float(false_rejects / n_pos)
    return far, frr


def compute_eer(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_thresholds: int = 1000,
) -> tuple[float, float]:
    """Compute Equal Error Rate (EER) — the threshold where FAR == FRR.

    Args:
        y_true: Binary labels.
        y_scores: Predicted probabilities.
        n_thresholds: Number of thresholds to evaluate.

    Returns:
        (EER value, threshold at EER).
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    min_diff = float("inf")
    eer = 0.0
    eer_thresh = 0.5

    for t in thresholds:
        far, frr = compute_far_frr(y_true, y_scores, t)
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2.0
            eer_thresh = float(t)

    return eer, eer_thresh


def det_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_thresholds: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Detection Error Tradeoff (DET) curve data.

    Args:
        y_true: Binary labels.
        y_scores: Predicted probabilities.
        n_thresholds: Number of threshold points.

    Returns:
        (far_array, frr_array, thresholds) — arrays for plotting.
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    fars = np.zeros(n_thresholds)
    frrs = np.zeros(n_thresholds)

    for i, t in enumerate(thresholds):
        fars[i], frrs[i] = compute_far_frr(y_true, y_scores, t)

    return fars, frrs, thresholds


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    criterion: str = "f1",
    n_thresholds: int = 1000,
    max_far: Optional[float] = None,
) -> tuple[float, float]:
    """Find the optimal decision threshold.

    Args:
        y_true: Binary labels.
        y_scores: Predicted probabilities.
        criterion: Optimization criterion — ``"f1"``, ``"eer"``, or ``"far"``.
        n_thresholds: Number of thresholds to evaluate.
        max_far: When criterion is ``"far"``, the maximum acceptable FAR.
                 Returns the lowest threshold achieving FAR <= max_far.

    Returns:
        (optimal_threshold, metric_value).
    """
    if criterion == "eer":
        eer, thresh = compute_eer(y_true, y_scores, n_thresholds)
        return thresh, eer

    thresholds = np.linspace(0, 1, n_thresholds)
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    best_thresh = 0.5
    best_metric = -1.0

    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()

        if criterion == "f1":
            prec = tp / (tp + fp + 1e-10)
            rec = tp / (tp + fn + 1e-10)
            metric = 2 * prec * rec / (prec + rec + 1e-10)
        elif criterion == "far":
            n_neg = (y_true == 0).sum()
            far = fp / (n_neg + 1e-10)
            if max_far is not None and far > max_far:
                continue
            metric = tp / (tp + fn + 1e-10)  # maximize recall at FAR constraint
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        if metric > best_metric:
            best_metric = metric
            best_thresh = float(t)

    return best_thresh, best_metric


def classification_report(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None,
    frr_targets: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> DetectionReport:
    """Generate a complete detection evaluation report.

    If no threshold is provided, the optimal F1 threshold is used.

    Args:
        y_true: Binary ground truth labels.
        y_scores: Predicted probabilities in [0, 1].
        threshold: Decision threshold. If None, auto-optimized for F1.
        frr_targets: FRR levels at which to report FAR.

    Returns:
        :class:`DetectionReport` with all metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    # EER
    eer, eer_thresh = compute_eer(y_true, y_scores)

    # Optimal threshold
    if threshold is None:
        threshold, _ = find_optimal_threshold(y_true, y_scores, criterion="f1")

    # FAR/FRR at operating threshold
    far, frr = compute_far_frr(y_true, y_scores, threshold)

    # Classification metrics
    preds = (y_scores >= threshold).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()
    tn = ((preds == 0) & (y_true == 0)).sum()

    accuracy = (tp + tn) / (len(y_true) + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    # AUC (trapezoidal)
    fars_curve, frrs_curve, _ = det_curve(y_true, y_scores, 500)
    tprs = 1.0 - frrs_curve
    sorted_idx = np.argsort(fars_curve)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trapz(tprs[sorted_idx], fars_curve[sorted_idx]))

    # FAR at specific FRR targets
    far_at_frr = {}
    for target_frr in frr_targets:
        # Find threshold giving FRR closest to target
        best_far = 1.0
        for i in range(len(frrs_curve)):
            if frrs_curve[i] <= target_frr:
                best_far = min(best_far, fars_curve[i])
        far_at_frr[target_frr] = best_far

    return DetectionReport(
        eer=eer,
        eer_threshold=eer_thresh,
        far=far,
        frr=frr,
        threshold=threshold,
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        auc=auc,
        n_positive=int((y_true == 1).sum()),
        n_negative=int((y_true == 0).sum()),
        far_at_frr=far_at_frr,
    )
