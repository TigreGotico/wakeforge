"""Smoke tests for FP/hour ambient validation metrics."""
import numpy as np
import soundfile as sf

from ww_trainer.metrics import estimate_fp_per_hour


def _always_trigger(audio: np.ndarray) -> float:
    """Mock infer function that always returns 1.0."""
    return 1.0


def _never_trigger(audio: np.ndarray) -> float:
    """Mock infer function that always returns 0.0."""
    return 0.0


class TestEstimateFpPerHour:
    """Tests for estimate_fp_per_hour."""

    def test_no_files_returns_zero(self) -> None:
        result = estimate_fp_per_hour(_always_trigger, [], threshold=0.5)
        assert result == 0.0

    def test_always_trigger_high_fp_rate(self, tmp_path) -> None:
        # 60 seconds of silence = many windows
        audio = np.zeros(16000 * 60, dtype=np.float32)
        path = tmp_path / "ambient.wav"
        sf.write(str(path), audio, 16000)

        fp_per_hour = estimate_fp_per_hour(
            _always_trigger, [path], threshold=0.5,
            window_sec=1.5, stride_sec=0.75,
        )
        # 60s at stride=0.75s -> ~78 windows -> all trigger
        # Extrapolated to 1 hour: 78 * 60 = ~4680
        assert fp_per_hour > 100

    def test_never_trigger_zero_fp(self, tmp_path) -> None:
        audio = np.zeros(16000 * 10, dtype=np.float32)
        path = tmp_path / "ambient.wav"
        sf.write(str(path), audio, 16000)

        fp_per_hour = estimate_fp_per_hour(
            _never_trigger, [path], threshold=0.5,
        )
        assert fp_per_hour == 0.0

    def test_threshold_controls_detection(self, tmp_path) -> None:
        audio = np.zeros(16000 * 10, dtype=np.float32)
        path = tmp_path / "ambient.wav"
        sf.write(str(path), audio, 16000)

        def half_prob(audio: np.ndarray) -> float:
            return 0.3

        # threshold=0.5 -> no FP
        assert estimate_fp_per_hour(half_prob, [path], threshold=0.5) == 0.0
        # threshold=0.2 -> all windows trigger
        assert estimate_fp_per_hour(half_prob, [path], threshold=0.2) > 0
