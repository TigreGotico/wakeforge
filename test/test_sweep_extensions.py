"""Unit tests for wakegp-inspired sweep extensions."""
import io
import math
import time
import pytest
from unittest.mock import patch, MagicMock, mock_open


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_CSV = "/fake/path.wav,1\n/fake/path2.wav,0\n"
_FAKE_ENTRIES = [("/fake/path.wav", "1"), ("/fake/path2.wav", "0")]


def _patch_deme_io():
    """Return a context-manager stack that makes _run_deme not touch the FS."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with patch("builtins.open", mock_open(read_data=_FAKE_CSV)):
            with patch("os.path.isfile", return_value=True):
                with patch("ww_trainer.sweep.Path.mkdir"):
                    yield

    return _ctx()


# ---------------------------------------------------------------------------
# Fitness-transform tests
# ---------------------------------------------------------------------------

class TestFitnessTransforms:
    def test_identity(self):
        from ww_trainer.sweep import _apply_fitness_fn
        assert _apply_fitness_fn(0.8, "f1") == 0.8

    def test_exp_f1_greater_than_f1(self):
        from ww_trainer.sweep import _apply_fitness_fn
        f1 = 0.8
        assert _apply_fitness_fn(f1, "exp_f1") > f1

    def test_double_exp_greater_than_exp(self):
        from ww_trainer.sweep import _apply_fitness_fn
        f1 = 0.8
        assert _apply_fitness_fn(f1, "double_exp_f1") > _apply_fitness_fn(f1, "exp_f1")

    def test_all_positive(self):
        from ww_trainer.sweep import _apply_fitness_fn
        for fn in ("f1", "exp_f1", "double_exp_f1"):
            assert _apply_fitness_fn(0.8, fn) > 0


# ---------------------------------------------------------------------------
# Timeout / target-F1 early-stop tests
# ---------------------------------------------------------------------------

class TestTimeoutAndTargetF1:
    def test_timeout_respected(self):
        """With a tiny timeout, search should return quickly."""
        from ww_trainer.sweep import run_genetic_search
        with patch("builtins.open", mock_open(read_data=_FAKE_CSV)):
            with patch("os.path.isfile", return_value=True):
                with patch("ww_trainer.sweep.Path.mkdir"):
                    with patch("ww_trainer.sweep._evaluate_config", return_value=0.5):
                        start = time.monotonic()
                        result = run_genetic_search(
                            metadata_csv="dummy.csv",
                            population_size=3,
                            generations=1000,
                            timeout_minutes=0.001,  # ~60 ms
                        )
                        elapsed = time.monotonic() - start
        assert elapsed < 10, "Timeout not respected"
        assert "best_config" in result

    def test_target_f1_stops_early(self):
        """With target_f1=0.0, should stop after generation 0."""
        from ww_trainer.sweep import run_genetic_search
        with patch("builtins.open", mock_open(read_data=_FAKE_CSV)):
            with patch("os.path.isfile", return_value=True):
                with patch("ww_trainer.sweep.Path.mkdir"):
                    with patch("ww_trainer.sweep._evaluate_config", return_value=0.5):
                        result = run_genetic_search(
                            metadata_csv="dummy.csv",
                            population_size=3,
                            generations=100,
                            target_f1=0.0,
                        )
        # Should have stopped very early (target met immediately)
        assert len(result["history"]) <= 2


# ---------------------------------------------------------------------------
# Two-stage search tests
# ---------------------------------------------------------------------------

class TestTwoStageSearch:
    def test_result_structure(self):
        from ww_trainer.sweep import run_two_stage_genetic_search
        with patch("builtins.open", mock_open(read_data=_FAKE_CSV)):
            with patch("os.path.isfile", return_value=True):
                with patch("ww_trainer.sweep.Path.mkdir"):
                    with patch("ww_trainer.sweep._evaluate_config", return_value=0.7):
                        result = run_two_stage_genetic_search(
                            metadata_csv="dummy.csv",
                            population_size=2,
                            generations=1,
                            stage2_population=2,
                            stage2_generations=1,
                        )
        assert "stage1" in result
        assert "stage2" in result
        assert "best_config" in result
        assert "best_score" in result
        assert "history" in result
        stages = {h["stage"] for h in result["history"]}
        assert 1 in stages
        assert 2 in stages

    def test_best_score_is_raw_f1(self):
        from ww_trainer.sweep import run_two_stage_genetic_search
        with patch("builtins.open", mock_open(read_data=_FAKE_CSV)):
            with patch("os.path.isfile", return_value=True):
                with patch("ww_trainer.sweep.Path.mkdir"):
                    with patch("ww_trainer.sweep._evaluate_config", return_value=0.85):
                        result = run_two_stage_genetic_search(
                            metadata_csv="dummy.csv",
                            population_size=2,
                            generations=1,
                            stage2_population=2,
                            stage2_generations=1,
                            fitness_fn="exp_f1",
                        )
        # best_score should be raw F1, not the exp-transformed value
        assert result["best_score"] <= 1.0


# ---------------------------------------------------------------------------
# Island model (n_demes) tests
# ---------------------------------------------------------------------------

class TestDemes:
    def test_n_demes_2_returns_valid_result(self):
        from ww_trainer.sweep import run_genetic_search
        with patch("builtins.open", mock_open(read_data=_FAKE_CSV)):
            with patch("os.path.isfile", return_value=True):
                with patch("ww_trainer.sweep.Path.mkdir"):
                    with patch("ww_trainer.sweep._evaluate_config", return_value=0.6):
                        result = run_genetic_search(
                            metadata_csv="dummy.csv",
                            population_size=2,
                            generations=1,
                            n_demes=2,
                        )
        assert "best_config" in result
        assert "best_score" in result
        assert isinstance(result["best_score"], float)
