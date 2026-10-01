"""Smoke tests for PredictionSmoother."""
from ww_trainer.inference import PredictionSmoother


class TestPredictionSmoother:
    """Tests for EMA, mean, max smoothing + debounce."""

    def test_ema_responds_to_input(self) -> None:
        s = PredictionSmoother(method="ema", ema_alpha=0.5)
        val = s.update(1.0)
        assert val == 0.5  # 0.5 * 1.0 + 0.5 * 0.0
        val = s.update(1.0)
        assert val == 0.75

    def test_mean_rolling_average(self) -> None:
        s = PredictionSmoother(method="mean", window_size=3)
        s.update(0.0)
        s.update(0.0)
        val = s.update(1.0)
        assert abs(val - 1.0 / 3.0) < 0.01

    def test_max_rolling_window(self) -> None:
        s = PredictionSmoother(method="max", window_size=3)
        s.update(0.1)
        s.update(0.9)
        val = s.update(0.2)
        assert val == 0.9  # max of window

    def test_trigger_with_patience(self) -> None:
        s = PredictionSmoother(method="ema", ema_alpha=1.0, threshold=0.5,
                               patience=3, debounce_sec=0, frame_rate_hz=10)
        s.update(0.8)
        assert not s.is_triggered()
        s.update(0.8)
        assert not s.is_triggered()
        s.update(0.8)
        assert s.is_triggered()

    def test_debounce_suppresses_retrigger(self) -> None:
        s = PredictionSmoother(method="ema", ema_alpha=1.0, threshold=0.5,
                               patience=1, debounce_sec=1.0, frame_rate_hz=10)
        s.update(0.8)
        assert s.is_triggered()
        # Immediate retrigger should be suppressed
        s.update(0.8)
        assert not s.is_triggered()

    def test_debounce_allows_after_cooldown(self) -> None:
        s = PredictionSmoother(method="ema", ema_alpha=1.0, threshold=0.5,
                               patience=1, debounce_sec=0.5, frame_rate_hz=10)
        s.update(0.8)
        assert s.is_triggered()
        # Feed 5 frames to pass debounce (0.5s * 10hz = 5 frames)
        for _ in range(5):
            s.update(0.8)
        assert s.is_triggered()

    def test_reset_clears_state(self) -> None:
        s = PredictionSmoother(method="ema", ema_alpha=0.5)
        s.update(1.0)
        s.reset()
        val = s.update(0.0)
        assert val == 0.0

    def test_below_threshold_resets_patience(self) -> None:
        s = PredictionSmoother(method="ema", ema_alpha=1.0, threshold=0.5,
                               patience=3, debounce_sec=0, frame_rate_hz=10)
        s.update(0.8)
        s.update(0.8)
        s.update(0.1)  # drops below threshold, resets consecutive count
        s.update(0.8)
        assert not s.is_triggered()  # only 1 consecutive, need 3

    def test_invalid_method(self) -> None:
        try:
            PredictionSmoother(method="invalid")
            assert False, "Expected ValueError"
        except ValueError:
            pass
