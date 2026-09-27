"""Tests for event-level false activations and FRR at a FA/hour budget.

Expected counts are worked out by hand from the window geometry.
"""
import numpy as np
import pytest
import soundfile as sf

from ww_trainer.metrics import (
    OperatingPoint,
    count_activations,
    estimate_fp_per_hour,
    frr_at_fa_per_hour,
    threshold_for_fa_per_hour,
)

SR = 16000


def test_run_of_windows_is_one_activation():
    scores = np.array([0, 0.9, 0.9, 0, 0.9, 0, 0, 0, 0.9])
    # Rising edges at windows 1, 4, 8 -> t = 0.1, 0.4, 0.8 s.
    assert count_activations(scores, 0.5, stride_sec=0.1, refractory_sec=0.0) == 3
    # 0.4 s falls inside the 0.5 s dead time after 0.1 s; 0.8 s does not.
    assert count_activations(scores, 0.5, stride_sec=0.1, refractory_sec=0.5) == 2


@pytest.fixture
def bursts_wav(tmp_path):
    """60 s of silence with three 0.5 s bursts at 10.0, 30.0 and 50.0 s."""
    audio = np.zeros(60 * SR, dtype=np.float32)
    for start in (10.0, 30.0, 50.0):
        audio[int(start * SR):int((start + 0.5) * SR)] = 0.9
    path = tmp_path / "bursts.wav"
    sf.write(str(path), audio, SR)
    return str(path)


def _loud(chunk: np.ndarray) -> float:
    return float(np.abs(chunk).max())


def test_estimate_fp_per_hour_counts_windows_by_default(bursts_wav):
    # 1.5 s windows every 0.75 s. Burst at 10.0 s: windows at 9.0 and 9.75.
    # Burst at 30.0 s: 29.25 and 30.0. Burst at 50.0 s: 48.75, 49.5, 50.25.
    fph = estimate_fp_per_hour(_loud, [bursts_wav], threshold=0.5)
    assert fph == pytest.approx(7 / (60 / 3600))


def test_estimate_fp_per_hour_event_level(bursts_wav):
    fph = estimate_fp_per_hour(_loud, [bursts_wav], threshold=0.5, refractory_sec=0.5)
    assert fph == pytest.approx(3 / (60 / 3600))


def test_threshold_for_budget_is_the_lowest_safe_candidate():
    stream = [np.array([0.2, 0.95, 0.2, 0.7, 0.2, 0.6])]
    # One hour of audio: 0.95 -> 1 event, 0.7 -> 2, 0.6 -> 3.
    kw = dict(negative_hours=1.0, stride_sec=1.0, refractory_sec=0.0)
    assert threshold_for_fa_per_hour(stream, target_fa_per_hour=1.0, **kw) == 0.95
    assert threshold_for_fa_per_hour(stream, target_fa_per_hour=2.0, **kw) == 0.7
    assert threshold_for_fa_per_hour(stream, target_fa_per_hour=0.5, **kw) > 0.95


def test_threshold_is_safe_for_every_higher_candidate():
    # At 0.9 the stream holds two events; at 0.5 the run merges into one.
    # A budget of one event must not pick 0.5, because 0.9 breaks it.
    stream = [np.array([0.9, 0.5, 0.9])]
    thr = threshold_for_fa_per_hour(stream, 1.0, 1.0, stride_sec=1.0, refractory_sec=0.0)
    assert thr > 0.9


def test_frr_at_fa_per_hour():
    stream = [np.array([0.2, 0.95, 0.2, 0.7, 0.2, 0.6])]
    positives = np.array([0.99, 0.8, 0.96, 0.5])
    op = frr_at_fa_per_hour(positives, stream, negative_hours=1.0, target_fa_per_hour=1.0,
                            stride_sec=1.0, refractory_sec=0.0)
    assert op == OperatingPoint(threshold=0.95, fa_per_hour=1.0, frr=0.5, target_fa_per_hour=1.0)


def test_empty_stream_has_no_activations():
    assert count_activations(np.array([]), 0.5, stride_sec=0.1) == 0


def test_refractory_boundary_is_inclusive():
    # Rising edges at 0.1 s and 0.6 s: exactly one refractory period apart.
    scores = np.array([0, 0.9, 0, 0, 0, 0, 0.9])
    assert count_activations(scores, 0.5, stride_sec=0.1, refractory_sec=0.5) == 2
