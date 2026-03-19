"""Training loop extracted from ``WakeWordTrainer``.

The public entry point is :func:`training_loop`.  It is intentionally
decoupled from ``WakeWordTrainer`` so it can be tested independently and
so ``trainer.py`` stays ≤400 lines.
"""
from __future__ import annotations

import csv
import logging
import os.path
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ww_trainer.checkpoint import save_intermediate_checkpoint
from ww_trainer.mining import load_mining_cache, save_mining_cache
from ww_trainer.dataset import AudioDataset, collate_fn
from ww_trainer.evaluation import (
    evaluate_model,
    log_metrics_csv,
    compute_fitness_score,
    compute_readiness,
)
from ww_trainer.loss import LossManager
from ww_trainer.mining import mine_hard_negatives
from ww_trainer.visualization import (
    log_confidence_histogram,
    log_embeddings_stats,
    log_pca,
    log_tsne,
    log_umap,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_epoch_data(
        wakes: List[Tuple[str, str]],
        nonwakes: List[Tuple[str, str]],
        hard_negatives: List[Tuple[str, str]],
        easy_negatives: List[Tuple[str, str]],
        hard_ratio: float,
        easy_ratio: float,
        random_ratio: float,
        replacement_ratio: float,
        balanced_replacement: bool,
) -> Tuple[List[Tuple[str, str]], int, int, int]:
    """Sample epoch data from wake/nonwake pools.

    Returns:
        (epoch_data, n_hard, n_easy, n_random)
    """
    n_wake = len(wakes)
    selected_hard = random.sample(hard_negatives, min(int(n_wake * hard_ratio), len(hard_negatives))) \
        if hard_negatives else []
    selected_easy = random.sample(easy_negatives, min(int(n_wake * easy_ratio), len(easy_negatives))) \
        if easy_negatives else []
    selected_random = random.sample(nonwakes, min(int(n_wake * random_ratio), len(nonwakes))) \
        if nonwakes else []

    epoch_data: List[Tuple[str, str]] = wakes + selected_hard + selected_easy + selected_random

    if replacement_ratio > 0 and epoch_data:
        n_replace = int(len(epoch_data) * replacement_ratio)
        if n_replace > 0:
            keep = random.sample(epoch_data, len(epoch_data) - n_replace)
            if balanced_replacement:
                half = n_replace // 2
                replacements = (random.choices(wakes, k=half) if wakes else []) + \
                               (random.choices(nonwakes, k=n_replace - half) if nonwakes else [])
            else:
                replacements = random.choices(wakes + nonwakes, k=n_replace)
            epoch_data = keep + replacements

    random.shuffle(epoch_data)
    return epoch_data, len(selected_hard), len(selected_easy), len(selected_random)


def _run_batch_loop(
        model: torch.nn.Module,
        device: torch.device,
        losses_cfg: List[Dict[str, Any]],
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_manager: LossManager,
        scaler: Optional[torch.cuda.amp.GradScaler],
        effective_amp: bool,
        accumulate_grad_batches: int,
        ep: int,
        total_steps: int,
) -> Tuple[float, Dict[str, float]]:
    """Run one training epoch. Returns (avg_loss, loss_breakdown)."""
    model.train()
    total_loss = 0.0
    loss_breakdown: Dict[str, float] = {cfg["name"]: 0.0 for cfg in losses_cfg}
    optimizer.zero_grad()

    for batch_idx, (wavs, labels, _) in enumerate(tqdm(loader, desc="Training", leave=False)):
        global_step = ep * len(loader) + batch_idx
        loss_manager.update_neg_weight(global_step, total_steps)

        with torch.amp.autocast(device_type=device.type, enabled=effective_amp):
            loss, loss_dict = loss_manager.compute_loss(model, wavs, labels, loader.dataset)

        loss_scaled = loss / max(1, accumulate_grad_batches)
        if scaler is not None:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if (batch_idx + 1) % accumulate_grad_batches == 0 or (batch_idx + 1) == len(loader):
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss_dict["total"]
        for k, v in loss_dict.items():
            if k != "total":
                loss_breakdown[k] += v

    n = max(1, len(loader))
    return total_loss / n, {k: v / n for k, v in loss_breakdown.items()}


def _update_best_checkpoints(
        trainer,
        ep: int,
        avg_loss: float,
        prec: float,
        rec: float,
        f1: float,
        best_metrics: Dict[str, float],
        optimizer: torch.optim.Optimizer,
        output_dir: Path,
        save_best: bool,
) -> List[str]:
    """Save best-metric checkpoints; return list of updated metric names."""
    updated: List[str] = []
    if not save_best:
        ckpt = output_dir / f"ep{ep + 1}.pt"
        save_intermediate_checkpoint(
            trainer.model, trainer.wake_word, trainer.arch,
            ep, best_metrics, optimizer, ckpt,
            trainer.export_onnx, trainer.mlflow,
        )
        logger.info("Saved checkpoint: %s", ckpt)
        return updated

    spec = [
        ("loss",      avg_loss, lambda a, b: a <= b, "best_loss.pt"),
        ("precision", prec,     lambda a, b: a >= b, "best_precision.pt"),
        ("recall",    rec,      lambda a, b: a >= b, "best_recall.pt"),
        ("f1",        f1,       lambda a, b: a >= b, "best_f1.pt"),
    ]
    for key, val, is_better, filename in spec:
        if is_better(val, best_metrics[key]):
            best_metrics[key] = val
            save_intermediate_checkpoint(
                trainer.model, trainer.wake_word, trainer.arch,
                ep, best_metrics, optimizer, output_dir / filename,
                trainer.export_onnx, trainer.mlflow,
            )
            updated.append(f"{key.capitalize()}={val:.4f}")

    return updated


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def training_loop(
        trainer,
        output_dir: "str | Path",
        train_data: List[Tuple[str, str]],
        test_data: List[Tuple[str, str]],
        epochs: int = 30,
        batch_size: int = 8,
        lr: float = 1e-3,
        neg_threshold: float = 0.5,
        mine_fraction: float = 0.2,
        mining_type: str = "semihard",
        patience: int = 2,
        save_best: bool = True,
        metrics_log: str = "metrics_log.csv",
        pca_every: int = 0,
        tsne_every: int = 0,
        umap_every: int = 0,
        blend_ratio: float = 0.7,
        base_hard: float = 0.5,
        max_hard: float = 5.0,
        base_easy: float = 1.5,
        min_easy: float = 0.2,
        base_random: float = 0.1,
        total_ratio: float = 5,
        use_amp: bool = False,
        accumulate_grad_batches: int = 1,
        resume: Optional[str] = None,
        neg_weight_schedule: Optional[str] = None,
        max_neg_weight: float = 100.0,
        target_fpr: Optional[float] = None,
        ambient_dir: Optional[str] = None,
        spec_augment: bool = False,
        spec_augment_kwargs: Optional[Dict[str, Any]] = None,
        replacement_ratio: float = 0.0,
        balanced_replacement: bool = True,
        fitness_checkpoint: bool = False,
        fitness_param_budget: int = 100_000,
) -> float:
    """Run the full training loop for *trainer*.

    This function contains all epoch-level logic previously inside
    ``WakeWordTrainer.train()``.  ``WakeWordTrainer.train()`` now delegates
    here directly so that this function can be used and tested independently.

    Args:
        trainer: A :class:`~ww_trainer.trainer.WakeWordTrainer` instance.
        output_dir: Directory for checkpoints, plots, and logs.
        train_data: List of ``(path, label)`` tuples for training.
        test_data: List of ``(path, label)`` tuples for evaluation.
        epochs: Maximum number of training epochs.
        batch_size: DataLoader batch size.
        lr: Initial learning rate (Adam).
        neg_threshold: Confidence threshold used for hard-negative mining.
        mine_fraction: Fraction of non-wake data to scan each epoch.
        mining_type: Triplet mining strategy (``"semihard"``, ``"hard"``).
        patience: Early-stopping patience (epochs without new hard negatives).
        save_best: If True, save best-per-metric checkpoints.
        metrics_log: Filename for the per-epoch metrics CSV.
        pca_every: Save PCA plot every N epochs (0 = disabled).
        tsne_every: Save t-SNE plot every N epochs (0 = disabled).
        umap_every: Save UMAP plot every N epochs (0 = disabled).
        blend_ratio: Controls how fast adaptive mixing shifts toward hard negatives.
        base_hard: Minimum hard-negative ratio.
        max_hard: Maximum hard-negative ratio.
        base_easy: Initial easy-negative ratio.
        min_easy: Minimum easy-negative ratio.
        base_random: Minimum random-negative ratio.
        total_ratio: Approximate total neg:wake ratio.
        use_amp: Enable automatic mixed precision (AMP).
        accumulate_grad_batches: Gradient accumulation steps.
        resume: Path to a checkpoint to resume from.
        neg_weight_schedule: BCE negative weight schedule (``"linear"``, etc.).
        max_neg_weight: Maximum negative weight for the schedule.
        target_fpr: If set, double ``max_neg_weight`` whenever FPR exceeds this.
        ambient_dir: Directory of ambient audio for FP/hour estimation.
        spec_augment: Enable spectrogram augmentation.
        spec_augment_kwargs: Kwargs for :class:`~ww_trainer.augment.SpectrogramAugment`.
        replacement_ratio: Fraction of epoch data to replace each epoch.
        balanced_replacement: Balance wake/non-wake when replacing.
        fitness_checkpoint: Save a ``best_fitness.pt`` checkpoint.
        fitness_param_budget: Parameter budget for fitness penalty.

    Returns:
        Best F1 score achieved during training.
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    effective_amp = use_amp or trainer.use_amp
    scaler = trainer.scaler if effective_amp else None
    if effective_amp and scaler is None:
        scaler = torch.amp.GradScaler(enabled=True)

    hardness_cache: Dict[str, float] = {}
    if resume:
        cache_path = str(Path(resume).parent / "hardneg_cache.pt")
        hardness_cache = load_mining_cache(cache_path)
        if hardness_cache:
            logger.info("[Mining] Loaded %d cached hardness scores", len(hardness_cache))

    params = filter(lambda p: p.requires_grad, trainer.model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    initial_lr = lr
    eta_min = lr * (0.05 if len(train_data) > 5000 else 0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    loss_manager = LossManager(
        loss_configs=trainer.losses_cfg, mining_type=mining_type, device=trainer.device,
        neg_weight_schedule=neg_weight_schedule, max_neg_weight=max_neg_weight,
    )
    if spec_augment:
        from ww_trainer.augment import SpectrogramAugment
        loss_manager.set_spec_augment(SpectrogramAugment(**(spec_augment_kwargs or {})))

    wakes = [x for x in train_data if x[1] == "1" and os.path.isfile(x[0])]
    nonwakes = [x for x in train_data if x[1] == "0" and os.path.isfile(x[0])]
    logger.info("Total wake-word samples: %d", len(wakes))
    logger.info("Total not-wake-word samples: %d", len(nonwakes))

    best_metrics = {"loss": float("inf"), "precision": 0.0, "recall": 0.0, "f1": 0.0}
    best_fitness = -1.0
    epochs_no_new = 0
    hard_negatives: List[Tuple[str, str]] = []
    easy_negatives: List[Tuple[str, str]] = []
    readiness_ema = 0.1

    ep = 0
    for ep in range(epochs):
        logger.info("=== Epoch %d/%d ===", ep + 1, epochs)

        # Progressive unfreezing
        if trainer.unfreeze_at_epoch is not None and ep == trainer.unfreeze_at_epoch:
            trainer._unfreeze()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, trainer.model.parameters()),
                lr=optimizer.param_groups[0]["lr"],
            )

        current_lr = optimizer.param_groups[0]["lr"]
        lr_factor = current_lr / initial_lr
        progress = (ep + 1) / max(1, epochs)
        adaptive_phase = blend_ratio * progress + (1 - blend_ratio) * (1 - lr_factor)

        if ep > 0:
            stats = log_embeddings_stats(trainer.model, test_data, ep + 1, trainer.device,
                                         128, trainer.mlflow)
            readiness = compute_readiness(stats)
            readiness_ema = 0.9 * readiness_ema + 0.1 * readiness
        if trainer.mlflow:
            trainer.mlflow.log_metrics({"readiness": readiness_ema}, step=ep + 1)

        adaptive_phase = (adaptive_phase + readiness_ema) / 2.0
        hard_ratio = base_hard + (max_hard - base_hard) * adaptive_phase
        easy_ratio = base_easy - (base_easy - min_easy) * adaptive_phase
        random_ratio = max(base_random, total_ratio - (hard_ratio + easy_ratio))
        logger.info("[Adaptive] readiness=%.3f -> hard=%.2f easy=%.2f rand=%.2f",
                    readiness_ema, hard_ratio, easy_ratio, random_ratio)
        if trainer.mlflow:
            trainer.mlflow.log_metrics({
                "learning-rate": current_lr, "hard-ratio": hard_ratio,
                "easy-ratio": easy_ratio, "random-ratio": random_ratio,
            }, step=ep + 1)

        epoch_data, n_hard, n_easy, n_rand = _build_epoch_data(
            wakes, nonwakes, hard_negatives, easy_negatives,
            hard_ratio, easy_ratio, random_ratio,
            replacement_ratio, balanced_replacement,
        )
        logger.info("Epoch data: total=%d wake=%d hard=%d easy=%d rand=%d",
                    len(epoch_data), len(wakes), n_hard, n_easy, n_rand)

        loader = DataLoader(
            AudioDataset(epoch_data, device=trainer.device.type, **trainer.augment_opts),
            batch_size=batch_size, shuffle=True,
            collate_fn=lambda b: collate_fn(b, trainer.device),
        )
        total_steps = epochs * max(1, len(loader))
        avg_loss, loss_breakdown = _run_batch_loop(
            trainer.model, trainer.device, trainer.losses_cfg,
            loader, optimizer, loss_manager, scaler,
            effective_amp, accumulate_grad_batches, ep, total_steps,
        )

        logger.info("Average total loss: %.4f", avg_loss)
        metrics = {"total_loss": avg_loss, **loss_breakdown}
        if trainer.mlflow:
            trainer.mlflow.log_metrics(metrics, step=ep + 1)

        acc, prec, rec, f1, auc, fp_paths, fn_paths, paths_all, targets, preds, probs, det_report = \
            evaluate_model(trainer.model, test_data, trainer.device,
                           batch_size=batch_size, threshold=0.4,
                           epoch=ep + 1, output_dir=output_dir, mlflow=trainer.mlflow)
        logger.info("Loss=%.4f Epoch %d: Acc=%.3f Prec=%.3f Rec=%.3f F1=%.3f AUC=%.3f EER=%.4f",
                    avg_loss, ep + 1, acc, prec, rec, f1, auc, det_report.eer)

        if fitness_checkpoint:
            param_count = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            fitness = compute_fitness_score(
                f1=f1, fp_rate=max(0.0, 1.0 - prec),
                fn_rate=max(0.0, 1.0 - rec),
                param_count=param_count, param_budget=fitness_param_budget,
            )
            logger.info("Fitness score: %.4f", fitness)
            if trainer.mlflow:
                trainer.mlflow.log_metrics({"fitness": fitness}, step=ep + 1)
            if fitness > best_fitness:
                best_fitness = fitness
                save_intermediate_checkpoint(
                    trainer.model, trainer.wake_word, trainer.arch, ep,
                    {**best_metrics, "fitness": fitness}, optimizer,
                    output_dir / "best_fitness.pt",
                    trainer.export_onnx, trainer.mlflow,
                )
                logger.info("Updated best fitness: %.4f", fitness)

        if target_fpr is not None and neg_weight_schedule is not None:
            current_fpr = max(0.0, 1.0 - prec)
            if current_fpr > target_fpr:
                loss_manager.adjust_max_neg_weight(2.0)
                logger.info("[NegWeight] FPR %.4f > target %.4f — doubled max_neg_weight to %.1f",
                            current_fpr, target_fpr, loss_manager.max_neg_weight)

        if ambient_dir is not None:
            from ww_trainer.metrics import estimate_fp_per_hour
            ambient_paths = sorted(Path(ambient_dir).rglob("*.wav"))
            if ambient_paths:
                fp_per_hour = estimate_fp_per_hour(trainer.model.infer, ambient_paths, threshold=0.5)
                logger.info("[Ambient] FP/hour: %.2f (%d files)", fp_per_hour, len(ambient_paths))
                if trainer.mlflow:
                    trainer.mlflow.log_metrics({"fp_per_hour": fp_per_hour}, step=ep + 1)

        if metrics_log:
            log_metrics_csv(str(output_dir / metrics_log), ep + 1, avg_loss, acc, prec, rec, f1, auc)

        log_confidence_histogram(targets, probs, ep + 1,
                                 output_dir / "viz" / "confidence", mlflow=trainer.mlflow)
        if pca_every and (ep + 1) % pca_every == 0:
            log_pca(trainer.model, test_data, output_dir / "viz" / "pca",
                    ep + 1, trainer.device, mlflow=trainer.mlflow)
        if tsne_every and (ep + 1) % tsne_every == 0:
            log_tsne(trainer.model, test_data, output_dir / "viz" / "tsne",
                     ep + 1, trainer.device, mlflow=trainer.mlflow)
        if umap_every and (ep + 1) % umap_every == 0:
            log_umap(trainer.model, test_data, output_dir / "viz" / "umap",
                     ep + 1, trainer.device, mlflow=trainer.mlflow)

        if trainer.mlflow:
            try:
                trainer.mlflow.log_metrics(
                    {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc},
                    step=ep + 1,
                )
                _log_fp_fn_artifacts(trainer.mlflow, ep, paths_all, targets, preds, probs, output_dir)
            except Exception as exc:
                logger.error("Failed to log metrics/artifacts to MLflow: %s", exc)

        scheduler.step()

        updated = _update_best_checkpoints(
            trainer, ep, avg_loss, prec, rec, f1, best_metrics, optimizer, output_dir, save_best,
        )
        if updated:
            logger.info("Updated best model(s): %s", ", ".join(updated))

        if ep == epochs - 1:
            break

        # Hard-negative mining
        if not hasattr(trainer, "hardness_cache"):
            trainer.hardness_cache = hardness_cache
        new_hards, easy_negatives, trainer.hardness_cache = mine_hard_negatives(
            model=trainer.model, nonwakes=nonwakes, device=trainer.device,
            hardness_cache=trainer.hardness_cache,
            neg_threshold=neg_threshold, dataset_fraction=mine_fraction,
            max_cache_size=3 * max(1, len(wakes)),
            wake_cache=getattr(trainer, "_wake_cache", []),
        )
        if new_hards:
            merged = {x[0]: x for x in new_hards}
            max_samples = max(1, len(wakes)) * 3
            hard_negatives = list(merged.values())[-max_samples:]
            epochs_no_new = 0
            logger.info("  -> Found %d false positives", len(hard_negatives))
        elif mine_fraction > 0:
            epochs_no_new += 1
            logger.info("  -> No new hard negatives (%d/%d)", epochs_no_new, patience)
            if epochs_no_new >= patience:
                logger.info("Early stopping — no new hard negatives.")
                break

    # Final save
    try:
        ckpt = output_dir / "final_model.pt"
        save_intermediate_checkpoint(
            trainer.model, trainer.wake_word, trainer.arch,
            ep, best_metrics, optimizer, ckpt,
            trainer.export_onnx, trainer.mlflow,
        )
        save_mining_cache(getattr(trainer, "hardness_cache", {}), str(output_dir / "hardneg_cache.pt"))
        logger.info("Training complete. Saved to %s", ckpt)
    except Exception as exc:
        logger.error("Failed to save final model: %s", exc)

    if trainer.mlflow:
        try:
            trainer.mlflow.end_run()
        except Exception:
            pass

    return best_metrics.get("f1", 0.0)


def _log_fp_fn_artifacts(mlflow, ep: int, paths_all, targets, preds, probs, output_dir: Path) -> None:
    """Write FP/FN CSV artifacts and log them to MLflow."""
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for kind, flag_pred, flag_true in [("fp", 1, 0), ("fn", 0, 1)]:
        csv_path = artifacts_dir / f"{kind}_epoch_{ep + 1}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true_label", "predicted_label", "probability", "file_path"])
            for path, true, pred, prob in zip(paths_all, targets, preds, probs):
                if pred == flag_pred and true == flag_true:
                    writer.writerow([true, pred, f"{prob:.4f}", path])
        mlflow.log_artifact(str(csv_path), artifact_path=f"false_{'positives' if kind == 'fp' else 'negatives'}")
