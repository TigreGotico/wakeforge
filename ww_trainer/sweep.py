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
) -> None:
    """Run an Optuna hyperparameter sweep over ww-trainer configurations.

    Args:
        metadata_csv: Path to dataset CSV (path,label per line).
        n_trials: Number of Optuna trials to run.
        output_dir: Directory to save per-trial checkpoints.
        featurizer: Path to ONNX extractor file (for featurizer_type='onnx').
        featurizer_type: Extractor type — 'mfcc', 'onnx', 'hubert', 'wav2vec2'.
        sample_rate: Audio sample rate.
        device: 'cpu', 'cuda', or 'auto'.
        epochs_per_trial: Training epochs per trial (keep short for sweep).
        study_name: Optuna study name.
        storage: Optuna storage URL (e.g. 'sqlite:///sweep.db'). None = in-memory.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna is required for hyperparameter sweeps. "
            "Install with: pip install optuna"
        )

    import random

    from ww_trainer.trainer import WakeWordTrainer

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data once (shared across trials)
    with open(metadata_csv, "r") as f:
        entries = [tuple(line.strip().split(",", 1)) for line in f if line.strip()]
    random.shuffle(entries)
    entries = [e for e in entries if os.path.isfile(e[0])]
    split = int(len(entries) * 0.8)
    train_data, val_data = entries[:split], entries[split:]

    def objective(trial: "optuna.Trial") -> float:
        arch = trial.suggest_categorical("arch", ["ffn", "gru", "cnn"])
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)

        losses_cfg = [{"name": "bce", "weight": 1.0}]

        trial_dir = out_dir / f"trial_{trial.number}"
        trial_dir.mkdir(exist_ok=True)

        kwargs = dict(
            hidden_dim=hidden_dim,
            dropout=dropout,
            featurizer_type=featurizer_type,
        )
        if featurizer_type == "mfcc":
            kwargs["n_mfcc"] = 40

        trainer = WakeWordTrainer(
            arch=arch,
            featurizer=featurizer or "",
            feature_dim=None,
            device=device,
            losses_cfg=losses_cfg,
            **kwargs,
        )

        # Minimal train — returns best val F1
        try:
            best_f1 = trainer.train(
                train_data=train_data,
                test_data=val_data,
                epochs=epochs_per_trial,
                batch_size=batch_size,
                lr=lr,
                output_dir=trial_dir,
                save_best="f1",
                metrics_log=str(trial_dir / "metrics.csv"),
                tsne_every=0,
                pca_every=0,
                umap_every=0,
            )
        except Exception as exc:
            logger.warning("Trial %d failed: %s", trial.number, exc)
            return 0.0

        return float(best_f1) if best_f1 is not None else 0.0

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
