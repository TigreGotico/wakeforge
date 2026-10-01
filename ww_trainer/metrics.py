"""Standalone evaluation metrics for wake word detection.

Computes detection-specific metrics: EER, FAR/FRR at threshold,
optimal threshold selection, DET curve data, and FP/hour ambient validation.

Usage::

    from ww_trainer.metrics import (
        compute_eer,
        compute_far_frr,
        find_optimal_threshold,
        det_curve,
        classification_report,
        estimate_fp_per_hour,
        frr_at_fa_per_hour,
    )

    report = classification_report(y_true, y_scores)
    print(f"EER: {report['eer']:.4f} at threshold {report['eer_threshold']:.4f}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


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


def area_under_det(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_thresholds: int = 1001,
) -> float:
    """Area Under the Detection Error Tradeoff curve (AUT).

    Integrates FNR as a function of FPR via the trapezoidal rule. Lower is
    better (0 = perfect separation). Mirrors the AUT metric defined in
    ``livekit/livekit-wakeword``'s evaluation pipeline.

    Args:
        y_true: Binary labels.
        y_scores: Predicted probabilities.
        n_thresholds: Number of threshold points on the DET curve.

    Returns:
        Scalar AUT value.
    """
    fars, frrs, _ = det_curve(y_true, y_scores, n_thresholds)
    order = np.argsort(fars)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(_trapz(frrs[order], fars[order]))


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


def stream_scores(
    infer_fn: "Callable[[np.ndarray], float]",
    audio: np.ndarray,
    window_sec: float = 1.5,
    stride_sec: float = 0.75,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Score every sliding window of a long recording.

    Args:
        infer_fn: Callable that takes a 1-D float32 numpy array and returns
            a probability in [0, 1].
        audio: 1-D float32 audio at ``sample_rate``.
        window_sec: Sliding window duration in seconds.
        stride_sec: Stride between windows in seconds.
        sample_rate: Audio sample rate.

    Returns:
        One score per window, in time order; empty when the audio is shorter
        than one window.
    """
    window = int(window_sec * sample_rate)
    stride = int(stride_sec * sample_rate)
    starts = range(0, len(audio) - window + 1, stride)
    return np.array([infer_fn(audio[s:s + window]) for s in starts], dtype=float)


def count_activations(
    scores: np.ndarray,
    threshold: float,
    stride_sec: float,
    refractory_sec: float = 0.5,
) -> int:
    """Count event-level activations in a stream of window scores.

    A run of consecutive windows at or above ``threshold`` is one
    activation. After an activation starts, a new one cannot start for
    ``refractory_sec``. This is the event-level protocol of the streaming
    keyword-spotting literature (Zhang et al., Interspeech 2026, with a
    500 ms refractory window), and it matches a detector that fires once
    per utterance rather than once per overlapping window.

    Args:
        scores: Window scores in time order (see :func:`stream_scores`).
        threshold: Detection threshold.
        stride_sec: Time between consecutive windows.
        refractory_sec: Dead time after each activation.

    Returns:
        Number of activations.
    """
    above = (np.asarray(scores) >= threshold).astype(np.int8)
    rising = np.flatnonzero(np.diff(above, prepend=0) == 1)
    gap = int(np.ceil(refractory_sec / stride_sec - 1e-9))
    count = 0
    next_allowed = 0
    for idx in rising:
        if idx >= next_allowed:
            count += 1
            next_allowed = idx + gap
    return count


@dataclass
class OperatingPoint:
    """A detector operating point calibrated to a false-activation budget.

    Attributes:
        threshold: Locked detection threshold.
        fa_per_hour: Event-level false activations per hour at ``threshold``
            on the calibration streams.
        frr: False reject rate of the positive clips at ``threshold``.
        target_fa_per_hour: The budget the threshold was calibrated to.
    """
    threshold: float
    fa_per_hour: float
    frr: float
    target_fa_per_hour: float


def threshold_for_fa_per_hour(
    negative_streams: List[np.ndarray],
    negative_hours: float,
    target_fa_per_hour: float,
    stride_sec: float = 0.75,
    refractory_sec: float = 0.5,
) -> float:
    """Lowest threshold whose event-level FA/hour stays within a budget.

    Event counts are not monotone in the threshold: raising it can split one
    long activation into two. Candidates are therefore scanned from the
    highest score down, and the result is the lowest candidate at which it
    and every higher candidate keep FA/hour at or below the budget.

    Args:
        negative_streams: Window scores per negative recording.
        negative_hours: Total duration of those recordings in hours.
        target_fa_per_hour: False activations per hour allowed.
        stride_sec: Time between consecutive windows.
        refractory_sec: Dead time after each activation.

    Returns:
        The threshold. When even the highest score exceeds the budget, a
        threshold just above it, so nothing fires.
    """
    all_scores = np.concatenate([np.asarray(s, dtype=float) for s in negative_streams])
    if all_scores.size == 0:
        return 0.0
    budget = target_fa_per_hour * negative_hours
    best = float(np.nextafter(all_scores.max(), np.inf))
    for candidate in np.unique(all_scores)[::-1]:
        events = sum(count_activations(s, candidate, stride_sec, refractory_sec)
                     for s in negative_streams)
        if events > budget:
            break
        best = float(candidate)
    return best


def frr_at_fa_per_hour(
    positive_scores: np.ndarray,
    negative_streams: List[np.ndarray],
    negative_hours: float,
    target_fa_per_hour: float,
    stride_sec: float = 0.75,
    refractory_sec: float = 0.5,
) -> OperatingPoint:
    """Calibrate a threshold to a FA/hour budget, then measure FRR there.

    This is the operating point wake-word work reports (FRR at 1 or 0.5
    false activations per hour), not an F1-optimal threshold. Calibrate on a
    validation set and apply the returned threshold, unchanged, to the test
    set.

    Args:
        positive_scores: One score per positive clip.
        negative_streams: Window scores per negative recording.
        negative_hours: Total duration of the negative recordings in hours.
        target_fa_per_hour: False activations per hour allowed.
        stride_sec: Time between consecutive windows.
        refractory_sec: Dead time after each activation.

    Returns:
        :class:`OperatingPoint` with the locked threshold, the achieved
        FA/hour on the negatives and the FRR on the positives.
    """
    threshold = threshold_for_fa_per_hour(
        negative_streams, negative_hours, target_fa_per_hour, stride_sec, refractory_sec,
    )
    events = sum(count_activations(s, threshold, stride_sec, refractory_sec)
                 for s in negative_streams)
    positive_scores = np.asarray(positive_scores, dtype=float)
    frr = float((positive_scores < threshold).mean()) if positive_scores.size else 0.0
    return OperatingPoint(
        threshold=threshold,
        fa_per_hour=events / negative_hours if negative_hours > 0 else 0.0,
        frr=frr,
        target_fa_per_hour=target_fa_per_hour,
    )


def estimate_fp_per_hour(
    infer_fn: "Callable[[np.ndarray], float]",
    ambient_audio_paths: List[Union[str, Path]],
    threshold: float = 0.5,
    window_sec: float = 1.5,
    stride_sec: float = 0.75,
    sample_rate: int = 16000,
    refractory_sec: Optional[float] = None,
) -> float:
    """Estimate false positives per hour on ambient (non-wake) audio.

    Slides a window over long-form ambient audio files, counts activations
    above threshold, and extrapolates to FP/hour.

    By default every window at or above the threshold counts, as
    ``ww-benchmarks`` counts it, so one false trigger that spans two
    overlapping windows counts twice. Pass ``refractory_sec`` to count
    event-level activations with :func:`count_activations` instead.

    Inspired by openWakeWord and micro-wake-word which validate with
    separate ambient sets measuring FP/hour.

    Args:
        infer_fn: Callable that takes a 1-D float32 numpy array and returns
            a probability in [0, 1]. Compatible with ``model.infer`` or
            ``OnnxWakeWordInferencer.infer``.
        ambient_audio_paths: Paths to long-form non-wake audio files.
        threshold: Detection threshold.
        window_sec: Sliding window duration in seconds.
        stride_sec: Stride between windows in seconds.
        sample_rate: Audio sample rate.
        refractory_sec: When set, count event-level activations with this
            dead time instead of every window above the threshold.

    Returns:
        Estimated false positives per hour. Returns 0.0 if no audio is provided
        or total duration is zero.
    """
    import soundfile as sf

    total_fp = 0
    total_duration_sec = 0.0

    for path in ambient_audio_paths:
        path = str(path)
        try:
            audio, sr = sf.read(path, dtype="float32")
        except Exception as exc:
            logger.warning("Failed to read ambient file %s: %s", path, exc)
            continue

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != sample_rate:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            except ImportError:
                logger.warning("librosa not available for resampling %s (sr=%d)", path, sr)
                continue

        total_duration_sec += len(audio) / sample_rate

        scores = stream_scores(infer_fn, audio, window_sec, stride_sec, sample_rate)
        if refractory_sec is None:
            total_fp += int((scores >= threshold).sum())
        else:
            total_fp += count_activations(scores, threshold, stride_sec, refractory_sec)

    if total_duration_sec <= 0:
        return 0.0

    total_hours = total_duration_sec / 3600.0
    return total_fp / total_hours


def compute_detection_metrics(
    infer_fn: "Callable[[np.ndarray], float]",
    test_audio_paths: List[Union[str, Path]],
    test_labels: List[int],
    thresholds: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
) -> dict:
    """Compute ROC, DET, and recall@FPR curves from audio files.

    Args:
        infer_fn: Callable returning probability for a 1-D float32 array.
        test_audio_paths: Paths to test audio files.
        test_labels: Binary labels (0 or 1) for each file.
        thresholds: Array of thresholds to evaluate. Defaults to 500 points in [0, 1].
        sample_rate: Audio sample rate.

    Returns:
        Dict with keys: ``"y_scores"``, ``"y_true"``, ``"report"``
        (a :class:`DetectionReport`).
    """
    import soundfile as sf

    scores: List[float] = []
    valid_labels: List[int] = []

    for path, label in zip(test_audio_paths, test_labels):
        try:
            audio, sr = sf.read(str(path), dtype="float32")
        except Exception:
            continue

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            except ImportError:
                continue

        prob = infer_fn(audio)
        scores.append(prob)
        valid_labels.append(label)

    y_scores = np.array(scores)
    y_true = np.array(valid_labels)
    report = classification_report(y_true, y_scores)

    return {"y_scores": y_scores, "y_true": y_true, "report": report}
