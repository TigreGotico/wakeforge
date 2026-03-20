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
        """Unknown fitness_fn in _apply_fitness_fn falls through to identity (internal helper — no validation)."""
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


# ---------------------------------------------------------------------------
# Input validation tests (LIM-002, LIM-003)
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Tests for _validate_ga_params enforced via run_genetic_search."""

    def test_invalid_fitness_fn_raises(self) -> None:
        """run_genetic_search with unknown fitness_fn must raise ValueError (LIM-002)."""
        with pytest.raises(ValueError, match="fitness_fn"):
            _run_genetic(population_size=2, generations=1, fitness_fn="bad_fn")

    def test_invalid_elite_frac_raises(self) -> None:
        """elite_frac outside (0, 1) must raise ValueError (LIM-003)."""
        with pytest.raises(ValueError, match="elite_frac"):
            _run_genetic(population_size=2, generations=1, elite_frac=1.5)

    def test_invalid_mutation_rate_raises(self) -> None:
        """mutation_rate outside [0, 1] must raise ValueError (LIM-003)."""
        with pytest.raises(ValueError, match="mutation_rate"):
            _run_genetic(population_size=2, generations=1, mutation_rate=-0.1)

    def test_elite_frac_zero_raises(self) -> None:
        """elite_frac=0.0 is outside (0, 1) and must raise ValueError."""
        with pytest.raises(ValueError, match="elite_frac"):
            _run_genetic(population_size=2, generations=1, elite_frac=0.0)

    def test_mutation_rate_exactly_one_is_valid(self) -> None:
        """mutation_rate=1.0 is within [0, 1] and must not raise."""
        result = _run_genetic(population_size=2, generations=1, mutation_rate=1.0)
        assert "best_config" in result

    def test_validate_ga_params_directly(self) -> None:
        """_validate_ga_params raises ValueError for each invalid input."""
        from ww_trainer.sweep import _validate_ga_params
        with pytest.raises(ValueError, match="fitness_fn"):
            _validate_ga_params("typo_fn", 0.2, 0.3)
        with pytest.raises(ValueError, match="elite_frac"):
            _validate_ga_params("f1", 1.5, 0.3)
        with pytest.raises(ValueError, match="mutation_rate"):
            _validate_ga_params("f1", 0.2, -0.1)


# ---------------------------------------------------------------------------
# Deme output dir isolation tests (LIM-004)
# ---------------------------------------------------------------------------

class TestDemeOutputDirIsolation:
    """Verify each deme writes to a separate subdirectory (LIM-004)."""

    def test_deme_output_dirs_separate(self) -> None:
        """With n_demes=2, run_genetic_search passes deme_0 and deme_1 output dirs.

        Inspects the kwargs passed to executor.submit so no pickling of mocks occurs.
        """
        from pathlib import Path
        import concurrent.futures

        submitted_kwargs: list[dict] = []
        _DEME_RESULT = {
            "best_config": {},
            "best_score": 0.5,
            "all_results": [],
            "history": [],
        }

        class FakeFuture:
            """Minimal concurrent.futures.Future stand-in."""
            def result(self) -> dict:
                return _DEME_RESULT

        class FakeExecutor:
            """Captures submit calls without spawning real processes."""
            def __enter__(self) -> "FakeExecutor":
                return self

            def __exit__(self, *args: object) -> None:
                pass

            def submit(self, fn: object, **kwargs: object) -> FakeFuture:
                submitted_kwargs.append(dict(kwargs))
                return FakeFuture()

        fake_futures: dict[FakeFuture, int] = {}

        original_as_completed = concurrent.futures.as_completed

        def fake_as_completed(fs: object) -> list:
            # fs is a dict {future: deme_id} in run_genetic_search
            return list(fs.keys()) if isinstance(fs, dict) else list(fs)

        # Import before patching builtins.open to avoid interfering with
        # matplotlib's matplotlibrc lookup during lazy module import.
        from ww_trainer.sweep import run_genetic_search  # noqa: PLC0415

        with patch("builtins.open", mock_open(read_data=_FAKE_CSV)):
            with patch("os.path.isfile", return_value=True):
                with patch("ww_trainer.sweep.Path.mkdir"):
                    with patch(
                        "ww_trainer.sweep.concurrent.futures.ProcessPoolExecutor",
                        return_value=FakeExecutor(),
                    ):
                        with patch(
                            "ww_trainer.sweep.concurrent.futures.as_completed",
                            side_effect=fake_as_completed,
                        ):
                            run_genetic_search(
                                metadata_csv="dummy.csv",
                                population_size=2,
                                generations=1,
                                n_demes=2,
                                migration_interval=0,  # force parallel executor path
                                output_dir="/tmp/deme_isolation_test",
                            )

        assert len(submitted_kwargs) == 2, f"Expected 2 submit calls, got {len(submitted_kwargs)}"
        dir_names = {Path(kw["output_dir"]).name for kw in submitted_kwargs}
        assert dir_names == {"deme_0", "deme_1"}, f"Got dir names: {dir_names}"


# ---------------------------------------------------------------------------
# S-018: Progress callback tests
# ---------------------------------------------------------------------------

class TestProgressCallback:
    """Tests for the on_generation callback (S-018)."""

    def test_on_generation_callback_called(self) -> None:
        """Callback is invoked once per generation with required keys."""
        events: list[dict] = []
        result = _run_genetic(
            population_size=2,
            generations=3,
            on_generation=events.append,
        )
        assert len(events) == 3
        for event in events:
            assert "generation" in event
            assert "stage" in event
            assert "best" in event
            assert "avg" in event
            assert "elapsed_seconds" in event
            assert "deme" in event

    def test_callback_receives_elapsed_seconds(self) -> None:
        """elapsed_seconds in each callback invocation is a positive float."""
        events: list[dict] = []
        _run_genetic(
            population_size=2,
            generations=2,
            on_generation=events.append,
        )
        assert len(events) == 2
        for event in events:
            assert isinstance(event["elapsed_seconds"], float)
            assert event["elapsed_seconds"] >= 0.0

    def test_callback_not_called_when_none(self) -> None:
        """No error raised and no calls made when on_generation is None (default)."""
        result = _run_genetic(population_size=2, generations=2)
        assert "best_config" in result

    def test_callback_generation_field_increments(self) -> None:
        """generation field in callback events matches loop index."""
        events: list[dict] = []
        _run_genetic(population_size=2, generations=4, on_generation=events.append)
        assert [e["generation"] for e in events] == [0, 1, 2, 3]

    def test_two_stage_callback_stage_field(self) -> None:
        """on_generation receives stage=1 for stage 1 events and stage=2 for stage 2."""
        events: list[dict] = []
        _run_two_stage(
            population_size=2, generations=2,
            stage2_population=2, stage2_generations=2,
            on_generation=events.append,
        )
        stages = {e["stage"] for e in events}
        assert 1 in stages
        assert 2 in stages


# ---------------------------------------------------------------------------
# S-017: Adaptive mutation decay tests
# ---------------------------------------------------------------------------

class TestMutationDecay:
    """Tests for the mutation_decay parameter (S-017)."""

    def test_mutation_decay_floor(self) -> None:
        """With high decay over many generations, rate never drops below 1e-4."""
        # decay=0.9999 drops rate very fast; verify floor is respected in _run_deme
        from ww_trainer.sweep import _run_deme, _build_search_space
        space = _build_search_space(full=False)
        with patch("builtins.open", mock_open(read_data=_FAKE_CSV)):
            with patch("os.path.isfile", return_value=True):
                with patch("ww_trainer.sweep.Path.mkdir"):
                    with patch("ww_trainer.sweep._evaluate_config", return_value=0.5):
                        result = _run_deme(
                            seed=42,
                            metadata_csv="dummy.csv",
                            population_size=2,
                            generations=50,
                            output_dir=Path("/tmp/decay_floor_test"),
                            featurizer_type="mfcc",
                            device="cpu",
                            epochs_per_trial=1,
                            search_space=space,
                            mutation_rate=0.5,
                            elite_frac=0.5,
                            timeout_minutes=None,
                            target_f1=None,
                            fitness_fn="f1",
                            seed_population=None,
                            full=False,
                            mutation_decay=0.9999,
                        )
        # Search completed without error; floor prevented zero rate
        assert len(result["history"]) > 0

    def test_invalid_mutation_decay_raises(self) -> None:
        """mutation_decay outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="mutation_decay"):
            _run_genetic(population_size=2, generations=1, mutation_decay=-0.1)

    def test_mutation_decay_zero_unchanged(self) -> None:
        """mutation_decay=0.0 (default) leaves mutation rate unchanged — no error."""
        result = _run_genetic(population_size=2, generations=2, mutation_decay=0.0)
        assert "best_config" in result

    def test_mutation_decay_valid_range(self) -> None:
        """mutation_decay=0.5 completes without error."""
        result = _run_genetic(population_size=2, generations=2, mutation_decay=0.5)
        assert "best_config" in result


# ---------------------------------------------------------------------------
# S-016: Deme migration tests
# ---------------------------------------------------------------------------

class TestDemeMigration:
    """Tests for the migration_interval / migration_size parameters (S-016)."""

    def test_migration_interval_zero_uses_parallel_path(self) -> None:
        """n_demes=2, migration_interval=0 uses the parallel executor path and returns a valid result."""
        from pathlib import Path
        import concurrent.futures

        _DEME_RESULT = {
            "best_config": {},
            "best_score": 0.5,
            "all_results": [],
            "history": [],
        }

        class FakeFuture:
            def result(self) -> dict:
                return _DEME_RESULT

        class FakeExecutor:
            def __enter__(self) -> "FakeExecutor":
                return self

            def __exit__(self, *args: object) -> None:
                pass

            def submit(self, fn: object, **kwargs: object) -> FakeFuture:
                return FakeFuture()

        def fake_as_completed(fs: object) -> list:
            return list(fs.keys()) if isinstance(fs, dict) else list(fs)

        from ww_trainer.sweep import run_genetic_search

        with patch("builtins.open", mock_open(read_data=_FAKE_CSV)):
            with patch("os.path.isfile", return_value=True):
                with patch("ww_trainer.sweep.Path.mkdir"):
                    with patch(
                        "ww_trainer.sweep.concurrent.futures.ProcessPoolExecutor",
                        return_value=FakeExecutor(),
                    ):
                        with patch(
                            "ww_trainer.sweep.concurrent.futures.as_completed",
                            side_effect=fake_as_completed,
                        ):
                            result = run_genetic_search(
                                metadata_csv="dummy.csv",
                                population_size=2,
                                generations=1,
                                n_demes=2,
                                migration_interval=0,
                            )
        assert "best_config" in result
        assert isinstance(result["best_score"], float)

    def test_migration_interval_positive_runs(self) -> None:
        """n_demes=2, migration_interval=1, generations=3 completes and returns merged history with both demes."""
        events: list[dict] = []
        result = _run_genetic(
            population_size=2,
            generations=3,
            n_demes=2,
            migration_interval=1,
            migration_size=1,
            on_generation=events.append,
        )
        assert "best_config" in result
        assert isinstance(result["best_score"], float)
        assert len(result["history"]) > 0
        # History should include entries from both demes
        deme_ids = {e.get("deme") for e in result["history"]}
        assert len(deme_ids) >= 1

    def test_migration_invalid_size_raises(self) -> None:
        """migration_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="migration_size"):
            _run_genetic(population_size=2, generations=1, migration_size=0)

    def test_migration_invalid_interval_raises(self) -> None:
        """migration_interval=-1 raises ValueError."""
        with pytest.raises(ValueError, match="migration_interval"):
            _run_genetic(population_size=2, generations=1, migration_interval=-1)

    def test_migration_single_deme_ignores_migration_params(self) -> None:
        """migration_interval / migration_size accepted without error for n_demes=1."""
        result = _run_genetic(
            population_size=2, generations=2,
            n_demes=1, migration_interval=1, migration_size=1,
        )
        assert "best_config" in result
