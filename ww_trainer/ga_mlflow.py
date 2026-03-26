"""MLflow integration for genetic hyperparameter searches.

Provides :class:`GeneticMLflowLogger` — a thin wrapper that opens a parent
MLflow run for an entire genetic search, logs per-trial and per-generation
metrics as steps, and uploads plots + result JSON when the search ends.

Usage::

    from ww_trainer.ga_mlflow import GeneticMLflowLogger

    mlf = GeneticMLflowLogger(
        mlflow_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        experiment_name="hey_mycroft",
        run_name="micro_genetic · 360m",
        search_space=SEARCH_SPACE,
        search_params={"stage1_pop": 16, "fp_weight": 0.6, ...},
    )

    # Inside the generation loop:
    mlf.log_trial(result_dict)          # after each trial
    mlf.log_generation(stage, gen,      # after each generation
                       gen_results, global_gen)

    # At the end:
    mlf.finalize(all_results, final_results,
                 plots_dir=PLOTS_DIR, results_json=ALL_RESULTS_JSON)
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GeneticMLflowLogger:
    """Manages a single MLflow parent run for a genetic hyperparameter search.

    All MLflow calls are wrapped in try/except so a tracking-server outage
    never aborts a long-running search.

    Args:
        mlflow_uri:      MLflow tracking URI. Pass ``None`` to disable entirely.
        experiment_name: Wake-word / project name (e.g. ``"hey_mycroft"``).
                         The MLflow experiment is set to ``"ww: <name>"``.
        run_name:        Human-readable name for the parent run.
        search_space:    Dict mapping param name → list of candidate values.
                         Used to log ``n_combos`` and ``search_keys`` params.
        search_params:   Flat dict of scalar search configuration values
                         (pop sizes, epochs, budgets, weights, …) to log as
                         MLflow params at run start.
    """

    def __init__(
        self,
        mlflow_uri: str | None,
        experiment_name: str,
        run_name: str,
        search_space: dict[str, list] | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> None:
        self._mlf = None
        if not mlflow_uri:
            return
        try:
            import mlflow as _m
            _m.set_tracking_uri(mlflow_uri)
            _m.set_experiment(f"ww: {experiment_name.replace('_', ' ')}")
            run = _m.start_run(
                run_name=run_name,
                tags={
                    "study":      "micro_genetic",
                    "experiment": experiment_name,
                    "script":     "train_micro_genetic",
                },
            )
            params: dict[str, Any] = dict(search_params or {})
            if search_space:
                params["search_keys"] = " ".join(search_space.keys())
                try:
                    params["n_combos"] = math.prod(len(v) for v in search_space.values())
                except Exception:
                    pass
            _m.log_params({k: str(v) for k, v in params.items()})
            self._mlf = _m
            logger.info("[MLflow] genetic search run: %s", run.info.run_id)
        except Exception as exc:
            logger.warning("[MLflow] setup failed: %s", exc)

    # ── Per-trial ──────────────────────────────────────────────────────────────

    def log_trial(self, result: dict) -> None:
        """Log one trial's metrics (step = trial id).

        Expected keys in *result*: ``fitness``, ``f1``, ``fpr``, ``fnr``,
        ``n_params``, ``tid``.
        """
        if self._mlf is None:
            return
        try:
            self._mlf.log_metrics(
                {
                    "trial/fitness":  result["fitness"],
                    "trial/f1":       result["f1"],
                    "trial/fpr":      result["fpr"],
                    "trial/fnr":      result["fnr"],
                    "trial/n_params": result.get("n_params", 0),
                },
                step=result["tid"],
            )
        except Exception:
            pass

    # ── Per-generation ─────────────────────────────────────────────────────────

    def log_generation(
        self,
        stage: int,
        gen: int,
        gen_results: list[dict],
        global_gen: int,
    ) -> None:
        """Log per-generation summary metrics (step = cumulative generation index).

        Args:
            stage:       Stage number (1, 2, …) — used as metric prefix ``s<N>/``.
            gen:         Generation index within the stage (0-based).
            gen_results: Results produced by this generation.
            global_gen:  Cumulative generation counter across all stages (used
                         as the MLflow step so all stages share one x-axis).
        """
        if self._mlf is None or not gen_results:
            return
        try:
            best     = max(gen_results, key=lambda r: r["fitness"])
            avg_fit  = sum(r["fitness"] for r in gen_results) / len(gen_results)
            zero_fit = sum(1 for r in gen_results if r["fitness"] == 0.0)
            self._mlf.log_metrics(
                {
                    f"s{stage}/best_fitness":   best["fitness"],
                    f"s{stage}/avg_fitness":    avg_fit,
                    f"s{stage}/best_f1":        best["f1"],
                    f"s{stage}/best_fpr":       best["fpr"],
                    f"s{stage}/best_fnr":       best["fnr"],
                    f"s{stage}/n_evaluated":    len(gen_results),
                    f"s{stage}/n_zero_fitness": zero_fit,
                },
                step=global_gen,
            )
        except Exception:
            pass

    # ── Finalize ───────────────────────────────────────────────────────────────

    def finalize(
        self,
        all_results: list[dict],
        final_results: list[dict],
        plots_dir: Path | None = None,
        results_json: Path | None = None,
    ) -> None:
        """Log summary metrics, upload artifacts, close the parent run.

        Args:
            all_results:    All trial results (stages 1 + 2).
            final_results:  Stage-3 final model results (may be empty).
            plots_dir:      Directory of PNG plots to upload as artifacts.
            results_json:   Path to ``all_results.json`` to upload.
        """
        if self._mlf is None:
            return
        try:
            if plots_dir:
                for p in Path(plots_dir).glob("*.png"):
                    self._mlf.log_artifact(str(p), artifact_path="plots")
            if results_json and Path(results_json).exists():
                self._mlf.log_artifact(str(results_json), artifact_path="results")

            if all_results:
                best = max(all_results, key=lambda r: r["fitness"])
                self._mlf.log_metrics(
                    {
                        "summary/best_fitness": best["fitness"],
                        "summary/best_f1":      best["f1"],
                        "summary/best_fpr":     best["fpr"],
                        "summary/best_fnr":     best["fnr"],
                        "summary/n_trials":     len(all_results),
                        "summary/n_nonzero":    sum(
                            1 for r in all_results if r["fitness"] > 0
                        ),
                    }
                )
                self._mlf.log_params(
                    {f"best_search/{k}": str(v) for k, v in best["config"].items()}
                )

            if final_results:
                best_f = max(final_results, key=lambda r: r["fitness"])
                self._mlf.log_metrics(
                    {
                        "final/fitness":  best_f["fitness"],
                        "final/f1":       best_f["f1"],
                        "final/fpr":      best_f["fpr"],
                        "final/fnr":      best_f["fnr"],
                        "final/n_params": best_f.get("n_params", 0),
                    }
                )
                self._mlf.log_params(
                    {f"best_final/{k}": str(v) for k, v in best_f["config"].items()}
                )

            self._mlf.end_run()
            logger.info("[MLflow] genetic search run closed")
        except Exception as exc:
            logger.warning("[MLflow] finalize failed: %s", exc)
