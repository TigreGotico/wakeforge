"""Optuna hyperparameter sweep for ww-trainer.

Usage:
    from ww_trainer.sweep import run_sweep
    run_sweep(metadata_csv="dataset.csv", n_trials=50, output_dir="sweep_results/")

Or from CLI:
    python -m ww_trainer.sweep --metadata dataset.csv --trials 50
"""
from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_VALID_FITNESS_FNS = frozenset({"f1", "exp_f1", "double_exp_f1"})


def _validate_ga_params(
    fitness_fn: str,
    elite_frac: float,
    mutation_rate: float,
) -> None:
    """Validate genetic algorithm parameters, raising ValueError on bad inputs.

    Args:
        fitness_fn: Selection pressure transform name.
        elite_frac: Fraction of population kept as elite; must be in (0, 1).
        mutation_rate: Per-gene mutation probability; must be in [0, 1].

    Raises:
        ValueError: If any parameter is invalid.
    """
    if fitness_fn not in _VALID_FITNESS_FNS:
        raise ValueError(
            f"Unknown fitness_fn {fitness_fn!r}. "
            f"Valid values: {sorted(_VALID_FITNESS_FNS)}"
        )
    if not (0.0 < elite_frac < 1.0):
        raise ValueError(
            f"elite_frac must be in (0, 1), got {elite_frac!r}"
        )
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError(
            f"mutation_rate must be in [0, 1], got {mutation_rate!r}"
        )


def _apply_fitness_fn(f1: float, fitness_fn: str) -> float:
    """Apply fitness transform for selection pressure. Reported scores remain raw F1.

    Args:
        f1: Raw F1 score in [0, 1].
        fitness_fn: Transform name — ``"f1"`` (identity), ``"exp_f1"``,
                    or ``"double_exp_f1"``.

    Returns:
        Transformed fitness value used only for GA selection comparisons.
    """
    if fitness_fn == "exp_f1":
        return math.exp(f1)
    if fitness_fn == "double_exp_f1":
        return math.exp(math.exp(f1) - 1)
    return f1  # "f1" identity


def run_sweep(
    metadata_csv: str,
    n_trials: int = 20,
    output_dir: str = "sweep_results",
    featurizer: Optional[str] = None,
    featurizer_type: str = "mfcc",
    sample_rate: int = 16000,
    device: str = "auto",
    epochs_per_trial: int = 5,
    study_name: str = "ww_trainer_sweep",
    storage: Optional[str] = None,
    full: bool = False,
) -> None:
    """Run an Optuna hyperparameter sweep over ww-trainer configurations.

    Args:
        metadata_csv: Path to dataset CSV (path,label per line).
        n_trials: Number of Optuna trials to run.
        output_dir: Directory to save per-trial checkpoints.
        featurizer: Path to ONNX extractor file (for featurizer_type='onnx').
        featurizer_type: Extractor type (ignored when ``full=True``).
        sample_rate: Audio sample rate.
        device: 'cpu', 'cuda', or 'auto'.
        epochs_per_trial: Training epochs per trial (keep short for sweep).
        study_name: Optuna study name.
        storage: Optuna storage URL (e.g. 'sqlite:///sweep.db'). None = in-memory.
        full: If True, also search over featurizer, classifier, and loss.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna is required for hyperparameter sweeps. "
            "Install with: pip install optuna"
        )

    import random

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_csv, "r") as f:
        entries = [tuple(line.strip().split(",", 1)) for line in f if line.strip()]
    random.shuffle(entries)
    entries = [e for e in entries if os.path.isfile(e[0])]
    split = int(len(entries) * 0.8)
    train_data, val_data = entries[:split], entries[split:]

    full_space = _build_search_space(full=True) if full else {}

    def objective(trial: "optuna.Trial") -> float:
        config: dict = {}

        if full:
            config["featurizer_type"] = trial.suggest_categorical(
                "featurizer_type", full_space["featurizer_type"])
            config["arch"] = trial.suggest_categorical("arch", full_space["arch"])
            config["loss"] = trial.suggest_categorical("loss", full_space["loss"])
            config["n_features"] = trial.suggest_categorical(
                "n_features", full_space["n_features"])
        else:
            config["arch"] = trial.suggest_categorical("arch", ["ffn", "gru", "cnn"])

        config["hidden_dim"] = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        config["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        config["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])
        config["dropout"] = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)

        return _evaluate_config(
            config, train_data, val_data, featurizer,
            featurizer_type, device, epochs_per_trial, out_dir, trial.number,
        )

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Best trial: %s", study.best_trial)
    logger.info("Best params: %s", study.best_params)
    logger.info("Best value (F1): %.4f", study.best_value)

    # Save best params
    import json
    best_path = out_dir / "best_params.json"
    with open(best_path, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)
    logger.info("Best params saved to %s", best_path)


def _build_search_space(full: bool = False) -> dict:
    """Default hyperparameter search space.

    Args:
        full: If True, include featurizer_type, classifier arch, and loss
              in the search space.  If False, only tune hyperparameters
              (arch, hidden_dim, lr, batch_size, dropout).

    Returns:
        Dict mapping parameter names to lists of candidate values.
    """
    space = {
        "arch": ["ffn", "gru", "cnn"],
        "hidden_dim": [64, 128, 256],
        "lr": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "batch_size": [16, 32, 64],
        "dropout": [0.0, 0.1, 0.2, 0.3],
    }
    if full:
        space.update({
            "featurizer_type": [
                "mfcc", "filterbank", "sincnet", "gammatone",
            ],
            "arch": [
                "ffn", "gru", "cnn",
                "bcresnet", "tcresnet", "dscnn", "matchboxnet",
                "res15", "conformer", "crnn",
            ],
            "loss": [
                "bce", "focal", "label_smoothing_bce",
                "supcon", "arcface", "ntxent",
            ],
            "n_features": [13, 40, 64, 80],
        })
    return space


# Map featurizer_type to kwargs for WakeWordTrainer
_FEAT_KWARGS = {
    "mfcc": lambda n: {"featurizer_type": "mfcc", "n_mfcc": n},
    "filterbank": lambda n: {"featurizer_type": "filterbank", "n_mels": n},
    "sincnet": lambda n: {"featurizer_type": "sincnet", "n_filters": n},
    "gammatone": lambda n: {"featurizer_type": "gammatone", "n_filters": n},
}


def _evaluate_config(
    config: dict,
    train_data: list,
    val_data: list,
    featurizer: Optional[str],
    featurizer_type: str,
    device: str,
    epochs: int,
    output_dir: Path,
    trial_id: int,
) -> float:
    """Train one configuration and return F1 score.

    Supports both simple configs (arch/hidden_dim/lr/batch_size/dropout)
    and full configs (featurizer_type/arch/loss/n_features).
    """
    from ww_trainer.trainer import WakeWordTrainer

    trial_dir = output_dir / f"trial_{trial_id}"
    trial_dir.mkdir(exist_ok=True)

    # Determine featurizer type and kwargs
    ft = config.get("featurizer_type", featurizer_type)
    n_feat = config.get("n_features", 40)
    feat_builder = _FEAT_KWARGS.get(ft)
    if feat_builder:
        kwargs = feat_builder(n_feat)
    else:
        kwargs = {"featurizer_type": ft}

    kwargs["hidden_dim"] = config.get("hidden_dim", 128)
    kwargs["dropout"] = config.get("dropout", 0.1)

    # Determine loss config
    loss_name = config.get("loss", "bce")
    losses_cfg = [{"name": loss_name, "weight": 1.0}]
    # Losses that need embed_dim
    if loss_name in ("arcface", "center", "proxy_nca"):
        losses_cfg[0]["embed_dim"] = kwargs["hidden_dim"]

    arch = config.get("arch", "ffn")

    trainer = WakeWordTrainer(
        arch=arch,
        featurizer=featurizer or "",
        feature_dim=None,
        device=device,
        losses_cfg=losses_cfg,
        **kwargs,
    )

    try:
        best_f1 = trainer.train(
            train_data=train_data,
            test_data=val_data,
            epochs=epochs,
            batch_size=config.get("batch_size", 32),
            lr=config.get("lr", 1e-3),
            output_dir=trial_dir,
            save_best="f1",
            metrics_log=str(trial_dir / "metrics.csv"),
            tsne_every=0, pca_every=0, umap_every=0,
        )
        return float(best_f1) if best_f1 is not None else 0.0
    except Exception as exc:
        logger.warning("Trial %d failed: %s", trial_id, exc)
        return 0.0


def run_grid_search(
    metadata_csv: str,
    output_dir: str = "grid_results",
    featurizer: Optional[str] = None,
    featurizer_type: str = "mfcc",
    device: str = "auto",
    epochs_per_trial: int = 5,
    search_space: Optional[dict] = None,
    full: bool = False,
) -> dict:
    """Exhaustive grid search over all hyperparameter combinations.

    Evaluates every combination in the search space.  Best for small
    spaces (< 100 combinations).

    Args:
        metadata_csv: Dataset CSV path.
        output_dir: Directory for results.
        featurizer: ONNX extractor path.
        featurizer_type: Extractor type.
        device: Device.
        epochs_per_trial: Epochs per configuration.
        search_space: Dict mapping param names to lists of values.
                      Defaults to a standard KWS grid.

    Returns:
        Dict with ``best_config``, ``best_score``, ``all_results``.
    """
    import itertools
    import json
    import random

    space = search_space or _build_search_space(full=full)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(metadata_csv) as f:
        entries = [tuple(line.strip().split(",", 1)) for line in f if line.strip()]
    random.shuffle(entries)
    entries = [e for e in entries if os.path.isfile(e[0])]
    split = int(len(entries) * 0.8)
    train_data, val_data = entries[:split], entries[split:]

    # Generate all combinations
    keys = list(space.keys())
    combos = list(itertools.product(*[space[k] for k in keys]))
    logger.info("Grid search: %d combinations", len(combos))

    results = []
    best_score = -1.0
    best_config = {}

    for i, values in enumerate(combos):
        config = dict(zip(keys, values))
        logger.info("Trial %d/%d: %s", i + 1, len(combos), config)

        score = _evaluate_config(
            config, train_data, val_data, featurizer, featurizer_type,
            device, epochs_per_trial, out_dir, i,
        )
        results.append({"config": config, "score": score})

        if score > best_score:
            best_score = score
            best_config = config

    out = {"best_config": best_config, "best_score": best_score, "all_results": results}
    with open(out_dir / "grid_results.json", "w") as f:
        json.dump(out, f, indent=2)

    logger.info("Grid search complete. Best: %.4f — %s", best_score, best_config)
    return out


def run_random_search(
    metadata_csv: str,
    n_trials: int = 50,
    output_dir: str = "random_results",
    featurizer: Optional[str] = None,
    featurizer_type: str = "mfcc",
    device: str = "auto",
    epochs_per_trial: int = 5,
    search_space: Optional[dict] = None,
    full: bool = False,
) -> dict:
    """Random search: sample configurations uniformly from the search space.

    More efficient than grid search for high-dimensional spaces.
    Bergstra & Bengio (2012) showed random search finds good configs
    faster than grid search when not all hyperparameters matter equally.

    Args:
        metadata_csv: Dataset CSV path.
        n_trials: Number of random configurations to evaluate.
        output_dir: Directory for results.
        featurizer: ONNX extractor path.
        featurizer_type: Extractor type.
        device: Device.
        epochs_per_trial: Epochs per configuration.
        search_space: Dict mapping param names to lists of values.

    Returns:
        Dict with ``best_config``, ``best_score``, ``all_results``.
    """
    import json
    import random

    space = search_space or _build_search_space(full=full)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_csv) as f:
        entries = [tuple(line.strip().split(",", 1)) for line in f if line.strip()]
    random.shuffle(entries)
    entries = [e for e in entries if os.path.isfile(e[0])]
    split = int(len(entries) * 0.8)
    train_data, val_data = entries[:split], entries[split:]

    results = []
    best_score = -1.0
    best_config = {}

    for i in range(n_trials):
        config = {k: random.choice(v) for k, v in space.items()}
        logger.info("Trial %d/%d: %s", i + 1, n_trials, config)

        score = _evaluate_config(
            config, train_data, val_data, featurizer, featurizer_type,
            device, epochs_per_trial, out_dir, i,
        )
        results.append({"config": config, "score": score})

        if score > best_score:
            best_score = score
            best_config = config

    out = {"best_config": best_config, "best_score": best_score, "all_results": results}
    with open(out_dir / "random_results.json", "w") as f:
        json.dump(out, f, indent=2)

    logger.info("Random search complete. Best: %.4f — %s", best_score, best_config)
    return out


def _run_deme(
    seed: int,
    metadata_csv: str,
    population_size: int,
    generations: int,
    output_dir: Path,
    featurizer_type: str,
    device: str,
    epochs_per_trial: int,
    search_space: dict,
    mutation_rate: float,
    elite_frac: float,
    timeout_minutes: Optional[float],
    target_f1: Optional[float],
    fitness_fn: str,
    seed_population: Optional[list],
    full: bool,
) -> dict:
    """Run one GA deme (island).  Top-level so it is picklable for multiprocessing.

    Args:
        seed: Random seed for this deme.
        metadata_csv: Dataset CSV path.
        population_size: Number of individuals per generation.
        generations: Maximum number of generations.
        output_dir: Directory for per-trial results.
        featurizer_type: Extractor type.
        device: Torch device.
        epochs_per_trial: Training epochs per candidate.
        search_space: Dict mapping param names to lists of values.
        mutation_rate: Probability of mutating each gene.
        elite_frac: Fraction of population kept as elite.
        timeout_minutes: Stop after this many minutes if not None.
        target_f1: Stop once raw F1 >= this value if not None.
        fitness_fn: Fitness transform — ``"f1"``, ``"exp_f1"``, or ``"double_exp_f1"``.
        seed_population: Optional pre-seeded list of configs (for stage 2).
        full: Whether this is a full-space search (used for logging only).

    Returns:
        Dict with ``best_config``, ``best_score`` (raw F1), ``all_results``,
        ``history`` (list of dicts with ``generation``, ``best``, ``avg``,
        ``elapsed_seconds``).
    """
    import json
    import random

    random.seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_csv) as f:
        entries = [tuple(line.strip().split(",", 1)) for line in f if line.strip()]
    random.shuffle(entries)
    entries = [e for e in entries if os.path.isfile(e[0])]
    split = int(len(entries) * 0.8)
    train_data, val_data = entries[:split], entries[split:]

    keys = list(search_space.keys())
    n_elite = max(1, int(population_size * elite_frac))
    trial_id = 0
    start_time = time.monotonic()

    def _random_individual() -> dict:
        return {k: random.choice(v) for k, v in search_space.items()}

    def _crossover(parent1: dict, parent2: dict) -> dict:
        """Uniform crossover: each gene from random parent."""
        return {k: random.choice([parent1[k], parent2[k]]) for k in keys}

    def _mutate(individual: dict) -> dict:
        """Randomly replace genes with probability mutation_rate."""
        mutant = individual.copy()
        for k in keys:
            if random.random() < mutation_rate:
                mutant[k] = random.choice(search_space[k])
        return mutant

    # Initialise population
    if seed_population:
        population = list(seed_population[:population_size])
        while len(population) < population_size:
            population.append(_random_individual())
    else:
        population = [_random_individual() for _ in range(population_size)]

    best_score = -1.0
    best_config: dict = {}
    history: list = []
    all_results: list = []

    for gen in range(generations):
        raw_scores: list[float] = []
        fit_scores: list[float] = []

        for ind in population:
            raw = _evaluate_config(
                ind, train_data, val_data, None, featurizer_type,
                device, epochs_per_trial, out_dir, trial_id,
            )
            fit = _apply_fitness_fn(raw, fitness_fn)
            raw_scores.append(raw)
            fit_scores.append(fit)
            all_results.append({"config": ind, "score": raw})
            trial_id += 1

            if raw > best_score:
                best_score = raw
                best_config = ind.copy()

        elapsed = time.monotonic() - start_time
        gen_best_raw = max(raw_scores)
        gen_avg_raw = sum(raw_scores) / len(raw_scores)
        history.append({
            "generation": gen,
            "best": gen_best_raw,
            "avg": gen_avg_raw,
            "elapsed_seconds": elapsed,
        })
        logger.info("Gen %d: best=%.4f avg=%.4f elapsed=%.1fs",
                    gen, gen_best_raw, gen_avg_raw, elapsed)

        # Early-stop checks
        if timeout_minutes is not None and elapsed > timeout_minutes * 60:
            logger.info("Timeout reached after gen %d (%.1fs)", gen, elapsed)
            break
        if target_f1 is not None and best_score >= target_f1:
            logger.info("Target F1 %.4f reached after gen %d", target_f1, gen)
            break

        # Selection by transformed fitness
        ranked = sorted(zip(fit_scores, population), key=lambda x: -x[0])
        elite = [ind for _, ind in ranked[:n_elite]]

        next_pop = list(elite)
        while len(next_pop) < population_size:
            p1, p2 = random.choices(elite, k=2)
            child = _crossover(p1, p2)
            child = _mutate(child)
            next_pop.append(child)
        population = next_pop

    return {
        "best_config": best_config,
        "best_score": best_score,
        "all_results": all_results,
        "history": history,
    }


def run_genetic_search(
    metadata_csv: str,
    population_size: int = 20,
    generations: int = 10,
    output_dir: str = "genetic_results",
    featurizer: Optional[str] = None,
    featurizer_type: str = "mfcc",
    device: str = "auto",
    epochs_per_trial: int = 5,
    search_space: Optional[dict] = None,
    mutation_rate: float = 0.3,
    elite_frac: float = 0.2,
    full: bool = False,
    n_demes: int = 1,
    timeout_minutes: Optional[float] = None,
    target_f1: Optional[float] = None,
    fitness_fn: str = "f1",
    seed_population: Optional[list] = None,
) -> dict:
    """Genetic algorithm hyperparameter search.

    Evolves a population of configurations through selection, crossover,
    and mutation.  Supports island-model parallelism (``n_demes > 1``),
    early stopping (``timeout_minutes``, ``target_f1``), and configurable
    selection pressure via ``fitness_fn``.

    Algorithm:
    1. Initialise random population (or use ``seed_population``).
    2. Evaluate fitness (F1 score); apply ``fitness_fn`` transform for selection.
    3. Select elite (top performers by transformed fitness).
    4. Crossover: combine two parents' genes.
    5. Mutation: randomly perturb genes.
    6. Repeat for ``generations`` or until an early-stop condition triggers.

    Args:
        metadata_csv: Dataset CSV path.
        population_size: Number of individuals per generation.
        generations: Maximum number of generations.
        output_dir: Directory for results.
        featurizer: ONNX extractor path (unused when ``featurizer_type`` is set).
        featurizer_type: Extractor type.
        device: Device.
        epochs_per_trial: Epochs per configuration.
        search_space: Dict mapping param names to lists of values.
        mutation_rate: Probability of mutating each gene.
        elite_frac: Fraction of population to keep as elite.
        full: Expand search space to include featurizer/arch/loss.
        n_demes: Number of independent parallel island populations.  Each
                 deme runs in its own process.  The best result is returned.
        timeout_minutes: Stop after this many wall-clock minutes if not None.
        target_f1: Stop once best raw F1 >= this value if not None.
        fitness_fn: Selection pressure transform.  One of ``"f1"``
                    (identity), ``"exp_f1"``, or ``"double_exp_f1"``.
        seed_population: Optional list of pre-built configs to seed the
                         initial population (used by two-stage search).

    Returns:
        Dict with ``best_config``, ``best_score`` (raw F1), ``all_results``,
        ``history``.
    """
    _validate_ga_params(fitness_fn, elite_frac, mutation_rate)

    import json

    space = search_space or _build_search_space(full=full)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import random
    base_seed = random.randint(0, 2**31)

    deme_kwargs = dict(
        metadata_csv=metadata_csv,
        population_size=population_size,
        generations=generations,
        output_dir=out_dir,
        featurizer_type=featurizer_type,
        device=device,
        epochs_per_trial=epochs_per_trial,
        search_space=space,
        mutation_rate=mutation_rate,
        elite_frac=elite_frac,
        timeout_minutes=timeout_minutes,
        target_f1=target_f1,
        fitness_fn=fitness_fn,
        seed_population=seed_population,
        full=full,
    )

    if n_demes <= 1:
        result = _run_deme(seed=base_seed, **deme_kwargs)
    else:
        futures_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_demes) as executor:
            futs = {
                executor.submit(
                    _run_deme,
                    seed=base_seed ^ deme_id,
                    **{**deme_kwargs, "output_dir": out_dir / f"deme_{deme_id}"},
                ): deme_id
                for deme_id in range(n_demes)
            }
            for fut in concurrent.futures.as_completed(futs):
                deme_id = futs[fut]
                deme_result = fut.result()
                # Tag history entries with deme index
                for entry in deme_result["history"]:
                    entry["deme"] = deme_id
                futures_results.append(deme_result)

        # Pick the winning deme
        result = max(futures_results, key=lambda r: r["best_score"])
        result["history"] = sorted(
            [entry for r in futures_results for entry in r["history"]],
            key=lambda e: (e.get("deme", 0), e["generation"]),
        )

    with open(out_dir / "genetic_results.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Genetic search complete. Best: %.4f — %s",
                result["best_score"], result["best_config"])
    return result


def run_two_stage_genetic_search(
    metadata_csv: str,
    population_size: int = 20,
    generations: int = 10,
    stage2_population: int = 10,
    stage2_generations: int = 5,
    stage2_elite_frac: float = 0.5,
    stage2_mutation_rate: float = 0.05,
    top_k_seed: int = 5,
    output_dir: str = "sweep_results",
    featurizer: Optional[str] = None,
    featurizer_type: str = "mfcc",
    device: str = "auto",
    epochs_per_trial: int = 5,
    full: bool = False,
    n_demes: int = 1,
    timeout_minutes: Optional[float] = None,
    target_f1: Optional[float] = None,
    fitness_fn: str = "f1",
) -> dict:
    """Two-stage genetic search: broad exploration then focused refinement.

    Stage 1 runs a standard GA over the full search space to find promising
    regions.  Stage 2 seeds a smaller, tighter GA with the top-K configs
    from stage 1 and uses lower mutation rate and higher elite fraction to
    converge on the best region found.

    Args:
        metadata_csv: Dataset CSV path.
        population_size: Stage 1 population size.
        generations: Stage 1 generation count.
        stage2_population: Stage 2 population size.
        stage2_generations: Stage 2 generation count.
        stage2_elite_frac: Elite fraction for stage 2.
        stage2_mutation_rate: Mutation rate for stage 2 (lower = more focused).
        top_k_seed: Number of best stage-1 configs used to seed stage 2.
        output_dir: Base output directory.
        featurizer: ONNX extractor path.
        featurizer_type: Extractor type.
        device: Torch device.
        epochs_per_trial: Training epochs per candidate.
        full: Expand search space.
        n_demes: Parallel island count (applied to both stages).
        timeout_minutes: Wall-clock timeout (applied per stage).
        target_f1: Early-stop F1 target (applied per stage).
        fitness_fn: Selection pressure transform for both stages.

    Returns:
        Dict with ``best_config``, ``best_score`` (raw F1), ``stage1``,
        ``stage2``, ``history`` (merged; each entry has a ``stage`` field).
    """
    import json
    from pathlib import Path

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        metadata_csv=metadata_csv,
        featurizer=featurizer,
        featurizer_type=featurizer_type,
        device=device,
        epochs_per_trial=epochs_per_trial,
        full=full,
        n_demes=n_demes,
        timeout_minutes=timeout_minutes,
        target_f1=target_f1,
        fitness_fn=fitness_fn,
    )

    # Stage 1 — broad search
    stage1 = run_genetic_search(
        population_size=population_size,
        generations=generations,
        output_dir=str(out_dir / "stage1"),
        **common,
    )

    # Extract top-K elites from stage 1
    sorted_results = sorted(stage1["all_results"], key=lambda r: r["score"], reverse=True)
    top_k = [r["config"] for r in sorted_results[:top_k_seed]]

    # Seed stage 2: top-K + mutated variants fill to stage2_population
    import random
    space = _build_search_space(full=full)
    keys = list(space.keys())

    def _mutate_stage2(cfg: dict) -> dict:
        """Mutate a config using stage2_mutation_rate."""
        mutant = cfg.copy()
        for k in keys:
            if random.random() < stage2_mutation_rate:
                mutant[k] = random.choice(space[k])
        return mutant

    seeded: list = list(top_k)
    while len(seeded) < stage2_population:
        parent = random.choice(top_k)
        seeded.append(_mutate_stage2(parent))

    # Stage 2 — focused refinement
    stage2 = run_genetic_search(
        population_size=stage2_population,
        generations=stage2_generations,
        output_dir=str(out_dir / "stage2"),
        elite_frac=stage2_elite_frac,
        mutation_rate=stage2_mutation_rate,
        seed_population=seeded,
        **common,
    )

    # Merge histories with stage tags
    merged_history = []
    for entry in stage1["history"]:
        merged_history.append({**entry, "stage": 1})
    for entry in stage2["history"]:
        merged_history.append({**entry, "stage": 2})

    # Best overall
    if stage2["best_score"] >= stage1["best_score"]:
        best_config = stage2["best_config"]
        best_score = stage2["best_score"]
    else:
        best_config = stage1["best_config"]
        best_score = stage1["best_score"]

    result = {
        "best_config": best_config,
        "best_score": best_score,
        "stage1": stage1,
        "stage2": stage2,
        "history": merged_history,
    }
    with open(out_dir / "two_stage_results.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Two-stage search complete. Best: %.4f — %s", best_score, best_config)
    return result


def _build_micro_search_space(tier_name: str) -> dict:
    """Build a search space constrained to ESP32-viable architectures.

    Only includes MFCC extractor with low feature counts and FFN heads
    with small hidden dimensions that fit within the tier's param budget.

    Args:
        tier_name: One of ``esp32_nano``, ``esp32_sweet``, ``esp32_max``.

    Returns:
        Dict mapping parameter names to lists of candidate values.

    Raises:
        ValueError: If tier_name is not an ESP32 tier.
    """
    from ww_trainer.tiers import get_tier
    tier = get_tier(tier_name)
    if tier.max_params is None:
        raise ValueError(f"Tier {tier_name!r} has no param budget — not an ESP32 tier")

    # Constrained spaces per budget
    if tier.max_size_kb is not None and tier.max_size_kb <= 1.0:
        hidden_dims = [8, 12, 16]
        n_mfccs = [10, 13]
        dropouts = [0.0, 0.1]
    elif tier.max_size_kb is not None and tier.max_size_kb <= 10.0:
        hidden_dims = [16, 32, 48, 64]
        n_mfccs = [13, 20]
        dropouts = [0.0, 0.1, 0.2]
    else:
        hidden_dims = [32, 64, 96, 128]
        n_mfccs = [13, 20, 26]
        dropouts = [0.0, 0.1, 0.2, 0.3]

    return {
        "arch": ["ffn"],
        "featurizer_type": ["mfcc"],
        "n_features": n_mfccs,
        "hidden_dim": hidden_dims,
        "lr": [5e-4, 1e-3, 5e-3],
        "batch_size": [16, 32],
        "dropout": dropouts,
    }


def _estimate_ffn_params(n_mfcc: int, hidden_dim: int) -> int:
    """Estimate parameter count for MFCC + FFN architecture.

    Architecture: Linear(n_mfcc, hidden_dim) + Linear(hidden_dim, 1).

    Args:
        n_mfcc: Number of MFCC coefficients (input features).
        hidden_dim: Hidden layer dimension.

    Returns:
        Total trainable parameter count.
    """
    return (n_mfcc * hidden_dim + hidden_dim) + (hidden_dim * 1 + 1)


def run_micro_search(
    metadata_csv: str,
    tier_name: str = "esp32_sweet",
    population_size: int = 16,
    generations: int = 8,
    output_dir: str = "micro_search_results",
    device: str = "auto",
    epochs_per_trial: int = 5,
    mutation_rate: float = 0.3,
    elite_frac: float = 0.25,
    accuracy_weight: float = 0.7,
    size_weight: float = 0.3,
) -> dict:
    """Genetic search with composite fitness for ESP32-constrained models.

    Fitness = accuracy_weight * f1 + size_weight * (1 - params/budget).
    Individuals exceeding the tier's param budget get fitness = -1 (rejected).

    Args:
        metadata_csv: Dataset CSV path.
        tier_name: ESP32 tier name (determines param budget and search space).
        population_size: Number of individuals per generation.
        generations: Number of generations.
        output_dir: Directory for results.
        device: Torch device.
        epochs_per_trial: Training epochs per trial.
        mutation_rate: Probability of mutating each gene.
        elite_frac: Fraction of population kept as elite.
        accuracy_weight: Weight for F1 score in fitness.
        size_weight: Weight for compactness in fitness.

    Returns:
        Dict with ``best_config``, ``best_score``, ``best_fitness``, ``history``.
    """
    import json
    import random

    from ww_trainer.tiers import get_tier

    tier = get_tier(tier_name)
    if tier.max_params is None:
        raise ValueError(f"Tier {tier_name!r} has no param budget")

    budget = tier.max_params
    space = _build_micro_search_space(tier_name)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_csv) as f:
        entries = [tuple(line.strip().split(",", 1)) for line in f if line.strip()]
    random.shuffle(entries)
    entries = [e for e in entries if os.path.isfile(e[0])]
    split = int(len(entries) * 0.8)
    train_data, val_data = entries[:split], entries[split:]

    keys = list(space.keys())
    n_elite = max(1, int(population_size * elite_frac))
    trial_id = 0

    def _random_individual() -> dict:
        return {k: random.choice(v) for k, v in space.items()}

    def _crossover(p1: dict, p2: dict) -> dict:
        return {k: random.choice([p1[k], p2[k]]) for k in keys}

    def _mutate(ind: dict) -> dict:
        m = ind.copy()
        for k in keys:
            if random.random() < mutation_rate:
                m[k] = random.choice(space[k])
        return m

    def _fitness(config: dict, f1: float) -> float:
        """Composite fitness: accuracy + compactness. Hard-reject over-budget."""
        n_params = _estimate_ffn_params(config["n_features"], config["hidden_dim"])
        if n_params > budget:
            return -1.0
        compactness = 1.0 - n_params / budget
        return accuracy_weight * f1 + size_weight * compactness

    population = [_random_individual() for _ in range(population_size)]
    best_overall_fitness = -2.0
    best_overall_config: dict = {}
    best_overall_f1 = 0.0
    history = []

    for gen in range(generations):
        fitnesses = []
        for ind in population:
            # Hard rejection: skip training if over budget
            est = _estimate_ffn_params(ind["n_features"], ind["hidden_dim"])
            if est > budget:
                fitnesses.append(-1.0)
                trial_id += 1
                continue

            f1 = _evaluate_config(
                ind, train_data, val_data, None, "mfcc",
                device, epochs_per_trial, out_dir, trial_id,
            )
            fit = _fitness(ind, f1)
            fitnesses.append(fit)
            trial_id += 1

            if fit > best_overall_fitness:
                best_overall_fitness = fit
                best_overall_config = ind.copy()
                best_overall_f1 = f1

        gen_best = max(fitnesses)
        gen_avg = sum(f for f in fitnesses if f >= 0) / max(1, sum(1 for f in fitnesses if f >= 0))
        history.append({"generation": gen, "best_fitness": gen_best, "avg_fitness": gen_avg})
        logger.info("Gen %d: best_fitness=%.4f avg=%.4f (overall=%.4f, f1=%.4f)",
                     gen, gen_best, gen_avg, best_overall_fitness, best_overall_f1)

        # Selection
        ranked = sorted(zip(fitnesses, population), key=lambda x: -x[0])
        elite = [ind for _, ind in ranked[:n_elite]]

        next_pop = list(elite)
        while len(next_pop) < population_size:
            p1, p2 = random.choices(elite, k=2)
            child = _crossover(p1, p2)
            child = _mutate(child)
            next_pop.append(child)
        population = next_pop

    out = {
        "tier": tier_name,
        "param_budget": budget,
        "best_config": best_overall_config,
        "best_f1": best_overall_f1,
        "best_fitness": best_overall_fitness,
        "history": history,
    }
    with open(out_dir / "micro_search_results.json", "w") as f:
        json.dump(out, f, indent=2)

    logger.info("Micro search complete. Best fitness=%.4f f1=%.4f config=%s",
                best_overall_fitness, best_overall_f1, best_overall_config)
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optuna sweep for ww-trainer")
    parser.add_argument("--metadata", required=True, help="Dataset CSV path")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--output-dir", default="sweep_results", help="Output directory")
    parser.add_argument("--featurizer", default=None, help="ONNX extractor path")
    parser.add_argument("--featurizer-type", default="mfcc",
                        choices=["mfcc", "onnx", "hubert", "wav2vec2"])
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per trial")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--storage", default=None, help="Optuna storage URL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_sweep(
        metadata_csv=args.metadata,
        n_trials=args.trials,
        output_dir=args.output_dir,
        featurizer=args.featurizer,
        featurizer_type=args.featurizer_type,
        epochs_per_trial=args.epochs,
        device=args.device,
        storage=args.storage,
    )
