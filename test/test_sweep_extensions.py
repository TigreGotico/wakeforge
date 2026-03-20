"""Unit tests for wakegp-inspired sweep extensions."""
import math
import time
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_CSV = "/fake/path.wav,1\n/fake/path2.wav,0\n"

_COMMON_PATCHES = (
    ("builtins.open", mock_open(read_data=_FAKE_CSV)),
    ("os.path.isfile", True),
    ("ww_trainer.sweep.Path.mkdir", None),
    ("ww_trainer.sweep._evaluate_config", 0.5),
)


def _run_genetic(**kwargs):
    """Run run_genetic_search with all heavy I/O mocked out."""
    from ww_trainer.sweep import run_genetic_search
    with patch("builtins.open", mock_open(read_data=_FAKE_CSV)):
        with patch("os.path.isfile", return_value=True):
            with patch("ww_trainer.sweep.Path.mkdir"):
                with patch("ww_trainer.sweep._evaluate_config", return_value=kwargs.pop("_score", 0.5)):
                    return run_genetic_search(metadata_csv="dummy.csv", **kwargs)


def _run_two_stage(**kwargs):
    """Run run_two_stage_genetic_search with all heavy I/O mocked out."""
    from ww_trainer.sweep import run_two_stage_genetic_search
    with patch("builtins.open", mock_open(read_data=_FAKE_CSV)):
        with patch("os.path.isfile", return_value=True):
            with patch("ww_trainer.sweep.Path.mkdir"):
                with patch("ww_trainer.sweep._evaluate_config", return_value=kwargs.pop("_score", 0.7)):
                    return run_two_stage_genetic_search(metadata_csv="dummy.csv", **kwargs)


# ---------------------------------------------------------------------------
# Fitness-transform tests
# ---------------------------------------------------------------------------

class TestFitnessTransforms:
    def test_identity(self):
        from ww_trainer.sweep import _apply_fitness_fn
        assert _apply_fitness_fn(0.8, "f1") == 0.8

    def test_f1_zero(self):
        """Identity transform at boundary f1=0.0 must return 0.0."""
        from ww_trainer.sweep import _apply_fitness_fn
        assert _apply_fitness_fn(0.0, "f1") == 0.0

    def test_f1_one(self):
        """Identity transform at boundary f1=1.0 must return 1.0."""
        from ww_trainer.sweep import _apply_fitness_fn
        assert _apply_fitness_fn(1.0, "f1") == 1.0

    def test_exp_f1_zero(self):
        """exp_f1 at f1=0.0 must equal math.exp(0.0) == 1.0."""
        from ww_trainer.sweep import _apply_fitness_fn
        assert _apply_fitness_fn(0.0, "exp_f1") == pytest.approx(math.exp(0.0))

    def test_exp_f1_one(self):
        """exp_f1 at f1=1.0 must equal math.exp(1.0)."""
        from ww_trainer.sweep import _apply_fitness_fn
        assert _apply_fitness_fn(1.0, "exp_f1") == pytest.approx(math.exp(1.0))

    def test_double_exp_f1_zero(self):
        """double_exp_f1 at f1=0.0 must equal exp(exp(0)-1) = exp(0) = 1.0."""
        from ww_trainer.sweep import _apply_fitness_fn
        expected = math.exp(math.exp(0.0) - 1)
        assert _apply_fitness_fn(0.0, "double_exp_f1") == pytest.approx(expected)

    def test_unknown_fitness_fn_falls_back_to_identity(self):
        """Unknown fitness_fn name must fall through to the identity (f1) branch."""
        from ww_trainer.sweep import _apply_fitness_fn
        assert _apply_fitness_fn(0.75, "not_a_real_fn") == 0.75

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
# _run_deme direct invocation tests
# ---------------------------------------------------------------------------

class TestRunDemeDirect:
    """Test _run_deme called directly (bypasses run_genetic_search wrapper)."""

    def _call_deme(self, seed_population=None, score=0.6):
        """Call _run_deme with minimal configuration and mocked I/O."""
        from ww_trainer.sweep import _run_deme, _build_search_space
        space = _build_search_space(full=False)
        with patch("builtins.open", mock_open(read_data=_FAKE_CSV)):
            with patch("os.path.isfile", return_value=True):
                with patch("ww_trainer.sweep.Path.mkdir"):
                    with patch("ww_trainer.sweep._evaluate_config", return_value=score):
                        return _run_deme(
                            seed=42,
                            metadata_csv="dummy.csv",
                            population_size=2,
                            generations=1,
                            output_dir=Path("/tmp/deme_test"),
                            featurizer_type="mfcc",
                            device="cpu",
                            epochs_per_trial=1,
                            search_space=space,
                            mutation_rate=0.3,
                            elite_frac=0.5,
                            timeout_minutes=None,
                            target_f1=None,
                            fitness_fn="f1",
                            seed_population=seed_population,
                            full=False,
                        )

    def test_returns_expected_keys(self):
        result = self._call_deme()
        assert {"best_config", "best_score", "all_results", "history"} <= result.keys()

    def test_history_contains_elapsed_seconds(self):
        """Each history entry must have an elapsed_seconds field (sweep.py:506-511)."""
        result = self._call_deme()
        assert len(result["history"]) >= 1
        for entry in result["history"]:
            assert "elapsed_seconds" in entry
            assert isinstance(entry["elapsed_seconds"], float)
            assert entry["elapsed_seconds"] >= 0.0

    def test_with_seed_population(self):
        """Seeded population is used as initial generation (sweep.py:472-477)."""
        from ww_trainer.sweep import _build_search_space
        space = _build_search_space(full=False)
        seed_pop = [
            {"arch": "ffn", "hidden_dim": 64, "lr": 1e-3, "batch_size": 32, "dropout": 0.1},
            {"arch": "gru", "hidden_dim": 128, "lr": 5e-4, "batch_size": 16, "dropout": 0.2},
        ]
        result = self._call_deme(seed_population=seed_pop)
        assert result["best_score"] == pytest.approx(0.6)
        assert len(result["all_results"]) >= 2

    def test_best_score_equals_mocked_f1(self):
        """best_score must equal raw F1 returned by _evaluate_config, not a transform."""
        result = self._call_deme(score=0.77)
        assert result["best_score"] == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# run_genetic_search parameter coverage
# ---------------------------------------------------------------------------

class TestGeneticSearchParams:
    def test_seed_population_accepted(self):
        """run_genetic_search passes seed_population through to _run_deme."""
        from ww_trainer.sweep import _build_search_space
        space = _build_search_space(full=False)
        seed_pop = [{"arch": "ffn", "hidden_dim": 64, "lr": 1e-3, "batch_size": 32, "dropout": 0.0}]
        result = _run_genetic(
            population_size=2,
            generations=1,
            seed_population=seed_pop,
            _score=0.65,
        )
        assert result["best_score"] == pytest.approx(0.65)

    def test_fitness_fn_exp_f1(self):
        """fitness_fn='exp_f1' must not affect best_score (raw F1 invariant)."""
        result = _run_genetic(
            population_size=2,
            generations=1,
            fitness_fn="exp_f1",
            _score=0.72,
        )
        assert result["best_score"] <= 1.0
        assert result["best_score"] == pytest.approx(0.72)

    def test_fitness_fn_double_exp_f1(self):
        """fitness_fn='double_exp_f1' must not affect best_score (raw F1 invariant)."""
        result = _run_genetic(
            population_size=2,
            generations=1,
            fitness_fn="double_exp_f1",
            _score=0.68,
        )
        assert result["best_score"] <= 1.0
        assert result["best_score"] == pytest.approx(0.68)

    def test_history_contains_elapsed_seconds(self):
        """History entries from run_genetic_search must include elapsed_seconds."""
        result = _run_genetic(population_size=2, generations=2)
        for entry in result["history"]:
            assert "elapsed_seconds" in entry

    def test_best_score_le_one_for_all_fitness_fns(self):
        """best_score (raw F1) must be <= 1.0 regardless of fitness_fn."""
        for fn in ("f1", "exp_f1", "double_exp_f1"):
            result = _run_genetic(population_size=2, generations=1, fitness_fn=fn, _score=0.9)
            assert result["best_score"] <= 1.0, f"Failed for fitness_fn={fn!r}"


# ---------------------------------------------------------------------------
# Timeout / target-F1 early-stop tests
# ---------------------------------------------------------------------------

class TestTimeoutAndTargetF1:
    def test_timeout_respected(self):
        """With a tiny timeout, search should return quickly."""
        start = time.monotonic()
        result = _run_genetic(population_size=3, generations=1000, timeout_minutes=0.001)
        elapsed = time.monotonic() - start
        assert elapsed < 10, "Timeout not respected"
        assert "best_config" in result

    def test_target_f1_stops_early(self):
        """With target_f1=0.0, should stop after generation 0."""
        result = _run_genetic(population_size=3, generations=100, target_f1=0.0)
        # Target is immediately met; at most 1 generation is completed
        assert len(result["history"]) <= 2


# ---------------------------------------------------------------------------
# Two-stage search tests
# ---------------------------------------------------------------------------

class TestTwoStageSearch:
    def test_result_structure(self):
        result = _run_two_stage(
            population_size=2, generations=1,
            stage2_population=2, stage2_generations=1,
        )
        assert "stage1" in result
        assert "stage2" in result
        assert "best_config" in result
        assert "best_score" in result
        assert "history" in result

    def test_history_has_stage_field_for_both_stages(self):
        """Every history entry must carry a 'stage' key (1 or 2) — sweep.py:774-778."""
        result = _run_two_stage(
            population_size=2, generations=1,
            stage2_population=2, stage2_generations=1,
        )
        stages = {h["stage"] for h in result["history"]}
        assert 1 in stages
        assert 2 in stages

    def test_best_score_is_raw_f1(self):
        """best_score must equal raw F1, not the transformed fitness value."""
        result = _run_two_stage(
            population_size=2, generations=1,
            stage2_population=2, stage2_generations=1,
            fitness_fn="exp_f1", _score=0.85,
        )
        assert result["best_score"] <= 1.0
        assert result["best_score"] == pytest.approx(0.85)

    def test_top_k_seed_larger_than_stage1_results(self):
        """top_k_seed > actual stage1 results must not raise; stage2 seeds from all results."""
        # population_size=2 × generations=1 → 2 all_results entries; top_k_seed=100 > 2
        result = _run_two_stage(
            population_size=2, generations=1,
            stage2_population=2, stage2_generations=1,
            top_k_seed=100,
        )
        assert "best_config" in result
        assert "best_score" in result

    def test_n_demes_2_two_stage(self):
        """run_two_stage_genetic_search with n_demes=2 must complete and return valid result."""
        result = _run_two_stage(
            population_size=2, generations=1,
            stage2_population=2, stage2_generations=1,
            n_demes=2,
        )
        assert isinstance(result["best_score"], float)
        assert result["best_score"] <= 1.0

    def test_best_score_le_one_for_all_fitness_fns(self):
        """best_score is raw F1 (<= 1.0) regardless of fitness_fn in two-stage search."""
        for fn in ("f1", "exp_f1", "double_exp_f1"):
            result = _run_two_stage(
                population_size=2, generations=1,
                stage2_population=2, stage2_generations=1,
                fitness_fn=fn, _score=0.9,
            )
            assert result["best_score"] <= 1.0, f"Failed for fitness_fn={fn!r}"


# ---------------------------------------------------------------------------
# Island model (n_demes) tests
# ---------------------------------------------------------------------------

class TestDemes:
    def test_n_demes_2_returns_valid_result(self):
        result = _run_genetic(population_size=2, generations=1, n_demes=2, _score=0.6)
        assert "best_config" in result
        assert "best_score" in result
        assert isinstance(result["best_score"], float)
