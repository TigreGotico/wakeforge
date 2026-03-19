"""Smoke tests: data replacement produces correct sizes and balance."""
import random

import pytest


class TestDataReplacement:
    """Verify epoch resampling logic extracted from trainer."""

    @staticmethod
    def _apply_replacement(
        epoch_data: list,
        wakes: list,
        nonwakes: list,
        replacement_ratio: float,
        balanced: bool,
    ) -> list:
        """Replicate the replacement logic from WakeWordTrainer.train()."""
        if replacement_ratio > 0 and len(epoch_data) > 0:
            n_replace = int(len(epoch_data) * replacement_ratio)
            if n_replace > 0:
                keep = random.sample(epoch_data, len(epoch_data) - n_replace)
                if balanced:
                    half = n_replace // 2
                    rep_wake = random.choices(wakes, k=half) if wakes else []
                    rep_nonwake = random.choices(nonwakes, k=n_replace - half) if nonwakes else []
                    replacements = rep_wake + rep_nonwake
                else:
                    full_pool = wakes + nonwakes
                    replacements = random.choices(full_pool, k=n_replace)
                epoch_data = keep + replacements
        return epoch_data

    def test_replacement_preserves_size(self) -> None:
        wakes = [("w.wav", "1")] * 10
        nonwakes = [("n.wav", "0")] * 20
        epoch_data = wakes + nonwakes
        result = self._apply_replacement(epoch_data, wakes, nonwakes, 0.4, balanced=False)
        assert len(result) == len(epoch_data)

    def test_balanced_replacement_has_both_classes(self) -> None:
        wakes = [("w.wav", "1")] * 10
        nonwakes = [("n.wav", "0")] * 20
        epoch_data = wakes + nonwakes
        random.seed(42)
        result = self._apply_replacement(epoch_data, wakes, nonwakes, 0.5, balanced=True)
        labels = [x[1] for x in result]
        assert "1" in labels
        assert "0" in labels

    def test_zero_ratio_no_change(self) -> None:
        data = [("a.wav", "1"), ("b.wav", "0")]
        result = self._apply_replacement(data, [data[0]], [data[1]], 0.0, balanced=False)
        assert result == data

    def test_replacement_ratio_one(self) -> None:
        """Full replacement should still produce same-length output."""
        wakes = [("w.wav", "1")] * 5
        nonwakes = [("n.wav", "0")] * 5
        epoch_data = wakes + nonwakes
        result = self._apply_replacement(epoch_data, wakes, nonwakes, 1.0, balanced=True)
        assert len(result) == len(epoch_data)
