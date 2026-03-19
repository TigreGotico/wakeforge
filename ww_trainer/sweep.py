"""Optuna hyperparameter sweep for ww-trainer.

Usage:
    from ww_trainer.sweep import run_sweep
    run_sweep(metadata_csv="dataset.csv", n_trials=50, output_dir="sweep_results/")

Or from CLI:
    python -m ww_trainer.sweep --metadata dataset.csv --trials 50
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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
) -> dict:
    """Genetic algorithm hyperparameter search.

    Evolves a population of configurations through selection, crossover,
    and mutation.  Good for complex search spaces where Bayesian
    optimization struggles.

    Algorithm:
    1. Initialize random population
    2. Evaluate fitness (F1 score)
    3. Select elite (top performers)
    4. Crossover: combine two parents' genes
    5. Mutation: randomly perturb genes
    6. Repeat for N generations

    Args:
        metadata_csv: Dataset CSV path.
        population_size: Number of individuals per generation.
        generations: Number of generations to evolve.
        output_dir: Directory for results.
        featurizer: ONNX extractor path.
        featurizer_type: Extractor type.
        device: Device.
        epochs_per_trial: Epochs per configuration.
        search_space: Dict mapping param names to lists of values.
        mutation_rate: Probability of mutating each gene.
        elite_frac: Fraction of population to keep as elite.

    Returns:
        Dict with ``best_config``, ``best_score``, ``history``.
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

    keys = list(space.keys())
    n_elite = max(1, int(population_size * elite_frac))
    trial_id = 0

    def _random_individual() -> dict:
        return {k: random.choice(v) for k, v in space.items()}

    def _crossover(parent1: dict, parent2: dict) -> dict:
        """Uniform crossover: each gene from random parent."""
        child = {}
        for k in keys:
            child[k] = random.choice([parent1[k], parent2[k]])
        return child

    def _mutate(individual: dict) -> dict:
        """Randomly replace genes with probability mutation_rate."""
        mutant = individual.copy()
        for k in keys:
            if random.random() < mutation_rate:
                mutant[k] = random.choice(space[k])
        return mutant

    # Initialize population
    population = [_random_individual() for _ in range(population_size)]

    best_overall_score = -1.0
    best_overall_config = {}
    history = []

    for gen in range(generations):
        # Evaluate
        scores = []
        for ind in population:
            score = _evaluate_config(
                ind, train_data, val_data, featurizer, featurizer_type,
                device, epochs_per_trial, out_dir, trial_id,
            )
            scores.append(score)
            trial_id += 1

            if score > best_overall_score:
                best_overall_score = score
                best_overall_config = ind.copy()

        gen_best = max(scores)
        gen_avg = sum(scores) / len(scores)
        history.append({"generation": gen, "best": gen_best, "avg": gen_avg})
        logger.info("Gen %d: best=%.4f avg=%.4f (overall best=%.4f)",
                     gen, gen_best, gen_avg, best_overall_score)

        # Selection: rank by score, keep elite
        ranked = sorted(zip(scores, population), key=lambda x: -x[0])
        elite = [ind for _, ind in ranked[:n_elite]]

        # Build next generation
        next_pop = list(elite)  # elitism
        while len(next_pop) < population_size:
            p1, p2 = random.choices(elite, k=2)
            child = _crossover(p1, p2)
            child = _mutate(child)
            next_pop.append(child)
        population = next_pop

    out = {
        "best_config": best_overall_config,
        "best_score": best_overall_score,
        "history": history,
    }
    with open(out_dir / "genetic_results.json", "w") as f:
        json.dump(out, f, indent=2)

    logger.info("Genetic search complete. Best: %.4f — %s",
                best_overall_score, best_overall_config)
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
