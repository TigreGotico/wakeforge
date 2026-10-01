"""Smoke tests: fitness score computation with known inputs."""
import pytest

from ww_trainer.evaluation import compute_fitness_score


class TestFitnessScore:
    """Verify compute_fitness_score with deterministic inputs."""

    def test_perfect_score(self) -> None:
        score = compute_fitness_score(
            f1=1.0, fp_rate=0.0, fn_rate=0.0,
            param_count=50_000, param_budget=100_000,
        )
        assert score == pytest.approx(1.0)

    def test_fp_penalty(self) -> None:
        """High FP rate should reduce score significantly (0.8 weight)."""
        score = compute_fitness_score(
            f1=0.5, fp_rate=0.5, fn_rate=0.0,
            param_count=50_000, param_budget=100_000,
        )
        # detection_score = 1 - 0.8*0.5 - 0.2*0 = 0.6, size_penalty = 1.0
        assert score == pytest.approx(0.6)

    def test_fn_penalty(self) -> None:
        """High FN rate should reduce score less than FP (0.2 weight)."""
        score = compute_fitness_score(
            f1=0.5, fp_rate=0.0, fn_rate=0.5,
            param_count=50_000, param_budget=100_000,
        )
        # detection_score = 1 - 0 - 0.2*0.5 = 0.9, size_penalty = 1.0
        assert score == pytest.approx(0.9)

    def test_size_penalty(self) -> None:
        """Exceeding param budget should reduce score."""
        score = compute_fitness_score(
            f1=1.0, fp_rate=0.0, fn_rate=0.0,
            param_count=200_000, param_budget=100_000,
        )
        # detection_score = 1.0, size_penalty = max(0, 1 - 0.1*(2-1)) = 0.9
        assert score == pytest.approx(0.9)

    def test_massive_oversize_clamps_to_zero(self) -> None:
        """Extreme oversize should clamp penalty to 0."""
        score = compute_fitness_score(
            f1=1.0, fp_rate=0.0, fn_rate=0.0,
            param_count=2_000_000, param_budget=100_000,
        )
        assert score == pytest.approx(0.0)

    def test_within_budget_no_size_penalty(self) -> None:
        score = compute_fitness_score(
            f1=0.8, fp_rate=0.1, fn_rate=0.1,
            param_count=80_000, param_budget=100_000,
        )
        # detection_score = 1 - 0.08 - 0.02 = 0.9, size_penalty = 1.0
        assert score == pytest.approx(0.9)

    def test_custom_weights(self) -> None:
        score = compute_fitness_score(
            f1=0.5, fp_rate=0.5, fn_rate=0.5,
            param_count=50_000, param_budget=100_000,
            fp_weight=0.5, fn_weight=0.5, size_weight=0.0,
        )
        # detection_score = 1 - 0.25 - 0.25 = 0.5
        assert score == pytest.approx(0.5)
