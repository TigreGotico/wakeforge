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
from ww_trainer.metrics import find_optimal_threshold
from ww_trainer.loss import LossManager
from ww_trainer.mining import mine_hard_negatives
from ww_trainer.visualization import (
    log_confidence_histogram,
    log_embeddings_stats,
    log_pca,
    log_tsne,
    log_umap,
    plot_confusion_matrix,
    plot_rppl_dashboard,
    plot_threshold_sensitivity,
    plot_training_curves,
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
        use_mixup: bool = False,
        mixup_alpha: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Run one training epoch. Returns (avg_loss, loss_breakdown)."""
    model.train()
    total_loss = 0.0
    loss_breakdown: Dict[str, float] = {cfg["name"]: 0.0 for cfg in losses_cfg}
    optimizer.zero_grad()

    batch_bar = tqdm(loader, desc=f"  ep{ep+1} train", unit="batch", leave=False, position=1)
    for batch_idx, batch in enumerate(batch_bar):
        wavs, labels, _, text_token_ids = batch if len(batch) == 4 else (*batch, None)
        global_step = ep * len(loader) + batch_idx
        loss_manager.update_neg_weight(global_step, total_steps)

        if use_mixup and len(wavs) > 1:
            from ww_trainer.augment import Mixup
            wavs, labels = Mixup.mix_batch(wavs, labels, beta_param=mixup_alpha)

        with torch.amp.autocast(device_type=device.type, enabled=effective_amp):
            loss, loss_dict = loss_manager.compute_loss(
                model, wavs, labels, loader.dataset, text_token_ids=text_token_ids
            )

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
                loss_breakdown[k] = loss_breakdown.get(k, 0.0) + v
        batch_bar.set_postfix(loss=f"{loss_dict['total']:.4f}")

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
        patience: int = 5,
        save_best: bool = True,
        metrics_log: str = "metrics_log.csv",
        pca_every: int = 0,
        tsne_every: int = 0,
        umap_every: int = 0,
        rppl_every: int = 0,
        blend_ratio: float = 0.7,
        base_hard: float = 1.5,   # was 0.5 — hard negatives should dominate
        max_hard: float = 5.0,
        base_easy: float = 0.5,   # was 1.5 — easy negatives provide less signal
        min_easy: float = 0.1,    # was 0.2
        base_random: float = 0.1,
        total_ratio: float = 5,
        use_amp: bool = False,
        accumulate_grad_batches: int = 1,
        resume: Optional[str] = None,
        neg_weight_schedule: Optional[str] = "linear",   # was None — now on by default
        max_neg_weight: float = 100.0,
        target_fpr: Optional[float] = None,
        target_fp_per_hour: Optional[float] = None,
        ambient_dir: Optional[str] = None,
        spec_augment: bool = True,    # was False — enable by default
        spec_augment_kwargs: Optional[Dict[str, Any]] = None,
        replacement_ratio: float = 0.0,
        balanced_replacement: bool = True,
        fitness_checkpoint: bool = False,
        fitness_param_budget: int = 100_000,
        use_mixup: bool = True,     # waveform Mixup — improves boundary robustness
        mixup_alpha: float = 1.0,   # Beta(alpha, alpha) mixing coefficient
        feature_cache=None,         # SharedWaveformCache or FeatureCache instance
        aug_prob: float = 0.7,      # waveform aug probability per sample (bg_noise/music/rir/pitch/speed)
        aug_warmup_epochs: int = 3, # ramp aug_prob from 0 → aug_prob linearly over this many epochs
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
        rppl_every: Save RPPL dashboard plot every N epochs (0 = end-only; auto-enabled when loss=rppl).
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
        target_fp_per_hour: If set, double ``max_neg_weight`` whenever the
            ambient FP/hour estimate exceeds this. Requires ``ambient_dir``.
            Ported from ``livekit/livekit-wakeword``; mirrors ``target_fpr``
            but uses an absolute trigger-rate target rather than a per-batch
            negative-rate fraction.
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
    # Warmup + cosine annealing: linear warmup for first 2 epochs, then cosine decay
    warmup_epochs = min(2, max(0, epochs - 1))
    eta_min = lr * 0.01   # decay to 1% of initial LR (previously 5-20%, too aggressive)

    def _lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return (ep + 1) / max(1, warmup_epochs)  # linear warmup
        cos_ep = ep - warmup_epochs
        cos_total = max(1, epochs - warmup_epochs)
        import math
        cos_val = 0.5 * (1.0 + math.cos(math.pi * cos_ep / cos_total))
        return (eta_min / lr) + (1.0 - eta_min / lr) * cos_val

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # Resume: restore optimizer state, epoch counter, and best-metric tracking so
    # training genuinely continues instead of silently re-running from scratch.
    # (The trainer ctor already restored model weights; here we also recover the
    # optimizer/LR schedule/best metrics that live in the sibling ``.ts`` file.)
    start_epoch = 0
    resumed_metrics: dict = {}
    if resume:
        loaded_epoch, resumed_metrics = trainer.load_checkpoint(resume, optimizer)
        if resumed_metrics:
            # ``loaded_epoch`` is the last completed epoch → continue from the next.
            start_epoch = loaded_epoch + 1
            # Fast-forward the LR schedule so warmup/cosine resumes at the right point.
            for _ in range(start_epoch):
                scheduler.step()
            logger.info(
                "[Resume] Continuing from epoch %d/%d (best F1 so far: %.4f)",
                start_epoch + 1, epochs, resumed_metrics.get("f1", 0.0),
            )

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

    # Log dataset + model stats as run params once
    if trainer.mlflow:
        try:
            param_count = sum(p.numel() for p in trainer.model.parameters())
            imbalance_ratio = len(nonwakes) / max(1, len(wakes))
            trainer.mlflow.log_params({
                "n_train_wake":    len(wakes),
                "n_train_nonwake": len(nonwakes),
                "n_test":          len(test_data),
                "imbalance_ratio": round(imbalance_ratio, 2),
                "param_count":     param_count,
                "epochs":          epochs,
                "lr":              lr,
                "batch_size":      batch_size,
                "mine_fraction":   mine_fraction,
                "max_neg_weight":    max_neg_weight,
                "use_mixup":         use_mixup,
                "mixup_alpha":       mixup_alpha,
                "spec_augment":      spec_augment,
                "aug_prob":          aug_prob,
                "aug_warmup_epochs": aug_warmup_epochs,
            })
        except Exception as exc:
            logger.warning("Failed to log run params: %s", exc)

    best_metrics = {"loss": float("inf"), "precision": 0.0, "recall": 0.0, "f1": 0.0}
    # Carry forward best metrics from a resumed run so a mediocre first epoch
    # can't overwrite a previously-saved better checkpoint.
    for _k in best_metrics:
        if _k in resumed_metrics:
            best_metrics[_k] = resumed_metrics[_k]
    best_fitness = -1.0
    epochs_no_new = 0
    hard_negatives: List[Tuple[str, str]] = []
    easy_negatives: List[Tuple[str, str]] = []
    readiness_ema = 0.5      # start at genuine uncertainty, not pessimistic 0.1
    current_threshold = 0.5  # updated each epoch via find_optimal_threshold
    _epoch_history: List[dict] = []  # accumulates per-epoch metrics for summary plots

    ep = start_epoch
    epoch_bar = tqdm(range(start_epoch, epochs), desc="Epochs", unit="ep",
                     position=0, initial=start_epoch, total=epochs)
    for ep in epoch_bar:
        epoch_bar.set_description(f"Epoch {ep+1}/{epochs}")
        logger.info("=== Epoch %d/%d ===", ep + 1, epochs)
        loss_manager.step_epoch(ep)

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

        # Update readiness every 3 epochs on a balanced sample of train data
        # (not test — test is imbalanced and inflates readiness via majority-class dominance).
        # Skip epoch 0 so the untrained model doesn't poison the EMA.
        if ep > 0 and ep % 3 == 0:
            # Sample a class-balanced subset so readiness isn't skewed by neg/pos ratio
            _r_wakes    = random.sample(wakes,    min(200, len(wakes)))
            _r_nonwakes = random.sample(nonwakes, min(200, len(nonwakes)))
            _readiness_data = _r_wakes + _r_nonwakes
            stats = log_embeddings_stats(trainer.model, _readiness_data, ep + 1,
                                         trainer.device, 128, trainer.mlflow)
            readiness = compute_readiness(stats)
            ema_alpha = 0.3 if ep < epochs // 2 else 0.15
            readiness_ema = (1 - ema_alpha) * readiness_ema + ema_alpha * readiness
        if trainer.mlflow:
            trainer.mlflow_log("log_metrics", {"readiness": readiness_ema}, step=ep + 1)

        adaptive_phase = (adaptive_phase + readiness_ema) / 2.0
        hard_ratio = base_hard + (max_hard - base_hard) * adaptive_phase
        easy_ratio = base_easy - (base_easy - min_easy) * adaptive_phase
        random_ratio = max(base_random, total_ratio - (hard_ratio + easy_ratio))

        # When the hard pool is empty (early epochs), redirect that budget to random negatives
        # so the total neg:wake ratio stays at total_ratio instead of dropping.
        if not hard_negatives:
            random_ratio = max(random_ratio, hard_ratio + easy_ratio + base_random)
            hard_ratio = 0.0
            easy_ratio = 0.0

        logger.info("[Adaptive] readiness=%.3f -> hard=%.2f easy=%.2f rand=%.2f",
                    readiness_ema, hard_ratio, easy_ratio, random_ratio)
        if trainer.mlflow:
            trainer.mlflow_log("log_metrics", {
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

        # Ramp aug_prob linearly from 0 → target over aug_warmup_epochs,
        # then hold at target. Warmup lets the model learn clean signal first.
        has_aug_files = any(
            trainer.augment_opts.get(k)
            for k in ("bg_noise_folder", "music_folder", "rir_folder",
                      "mic_noise_folder", "bg_speech_folder")
        )
        if has_aug_files and aug_prob > 0:
            current_aug_prob = aug_prob * min(1.0, (ep + 1) / max(1, aug_warmup_epochs))
        else:
            current_aug_prob = 0.0
        if trainer.mlflow:
            trainer.mlflow_log("log_metrics", {"aug_prob": current_aug_prob}, step=ep + 1)

        loader = DataLoader(
            AudioDataset(epoch_data, device=trainer.device.type,
                         aug_prob=current_aug_prob,
                         feature_cache=feature_cache if current_aug_prob == 0.0 else None,
                         **trainer.augment_opts),
            batch_size=batch_size, shuffle=True,
            collate_fn=lambda b: collate_fn(b, trainer.device),
        )
        total_steps = epochs * max(1, len(loader))
        avg_loss, loss_breakdown = _run_batch_loop(
            trainer.model, trainer.device, trainer.losses_cfg,
            loader, optimizer, loss_manager, scaler,
            effective_amp, accumulate_grad_batches, ep, total_steps,
            use_mixup=use_mixup, mixup_alpha=mixup_alpha,
        )

        logger.info("Average total loss: %.4f", avg_loss)
        metrics = {"total_loss": avg_loss, **loss_breakdown}
        if trainer.mlflow:
            trainer.mlflow_log("log_metrics", metrics, step=ep + 1)

        tqdm.write(f"  [ep {ep+1}] evaluating …")
        acc, prec, rec, f1, auc, fp_paths, fn_paths, paths_all, targets, preds, probs, det_report = \
            evaluate_model(trainer.model, test_data, trainer.device,
                           batch_size=batch_size, threshold=current_threshold,
                           epoch=ep + 1, output_dir=output_dir, mlflow=trainer.mlflow,
                           feature_cache=feature_cache)
        tqdm.write(
            f"  ep {ep+1:>3}/{epochs}  loss={avg_loss:.4f}  "
            f"F1={f1:.4f}  prec={prec:.3f}  rec={rec:.3f}  "
            f"AUC={auc:.4f}  EER={det_report.eer:.4f}  "
            f"hard={len(hard_negatives)}  thr={current_threshold:.3f}"
        )
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", f1=f"{f1:.4f}", eer=f"{det_report.eer:.4f}")

        if len(set(targets)) > 1:
            current_threshold, _ = find_optimal_threshold(
                np.array(targets), np.array(probs), criterion="f1"
            )
            logger.info("[Threshold] Optimal F1 threshold for next epoch: %.4f", current_threshold)
            if trainer.mlflow:
                trainer.mlflow_log("log_metrics", {"optimal_threshold": current_threshold}, step=ep + 1)

        if fitness_checkpoint:
            param_count = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            fitness = compute_fitness_score(
                f1=f1, fp_rate=max(0.0, 1.0 - prec),
                fn_rate=max(0.0, 1.0 - rec),
                param_count=param_count, param_budget=fitness_param_budget,
            )
            logger.info("Fitness score: %.4f", fitness)
            if trainer.mlflow:
                trainer.mlflow_log("log_metrics", {"fitness": fitness}, step=ep + 1)
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
                fp_per_hour = estimate_fp_per_hour(trainer.model.infer, ambient_paths, threshold=current_threshold)
                logger.info("[Ambient] FP/hour: %.2f (%d files)", fp_per_hour, len(ambient_paths))
                if trainer.mlflow:
                    trainer.mlflow_log("log_metrics", {"fp_per_hour": fp_per_hour}, step=ep + 1)
                if (target_fp_per_hour is not None
                        and neg_weight_schedule is not None
                        and fp_per_hour > target_fp_per_hour):
                    loss_manager.adjust_max_neg_weight(2.0)
                    logger.info("[NegWeight] FP/h %.2f > target %.2f — doubled max_neg_weight to %.1f",
                                fp_per_hour, target_fp_per_hour, loss_manager.max_neg_weight)

        if metrics_log:
            log_metrics_csv(str(output_dir / metrics_log), ep + 1, avg_loss, acc, prec, rec, f1, auc)

        log_confidence_histogram(targets, probs, ep + 1,
                                 output_dir / "viz" / "confidence", mlflow=trainer.mlflow)
        _embed_metrics_this_epoch: dict = {}
        if pca_every and (ep + 1) % pca_every == 0:
            _, _em = log_pca(trainer.model, test_data, output_dir / "viz" / "pca",
                             ep + 1, trainer.device, mlflow=trainer.mlflow)
            _embed_metrics_this_epoch.update(_em)
        if tsne_every and (ep + 1) % tsne_every == 0:
            _, _em = log_tsne(trainer.model, test_data, output_dir / "viz" / "tsne",
                              ep + 1, trainer.device, mlflow=trainer.mlflow)
            _embed_metrics_this_epoch.update(_em)
        if umap_every and (ep + 1) % umap_every == 0:
            _, _em = log_umap(trainer.model, test_data, output_dir / "viz" / "umap",
                              ep + 1, trainer.device, mlflow=trainer.mlflow)
            _embed_metrics_this_epoch.update(_em)

        # Derived imbalance metrics
        n_pos_test = sum(1 for t in targets if t == 1)
        n_neg_test = sum(1 for t in targets if t == 0)
        fp_count = sum(1 for t, p in zip(targets, preds) if p == 1 and t == 0)
        fn_count = sum(1 for t, p in zip(targets, preds) if p == 0 and t == 1)
        fpr = fp_count / max(1, n_neg_test)
        fnr = fn_count / max(1, n_pos_test)
        current_neg_weight = getattr(loss_manager, "_current_neg_weight", None) or max_neg_weight

        epoch_record: Dict[str, Any] = {
            "epoch": ep + 1, "loss": avg_loss, "f1": f1, "auc": auc,
            "precision": prec, "recall": rec,
            "n_hard_negatives": len(hard_negatives),
        }
        # Accumulate RPPL sub-components from loss_breakdown
        for rppl_key in ("rppl_bce", "rppl_proto", "rppl_div", "rppl_center", "rppl_cons"):
            if rppl_key in loss_breakdown:
                epoch_record[rppl_key] = loss_breakdown[rppl_key]
        # Accumulate embedding metrics logged this epoch by PCA/t-SNE
        for embed_key in ("embed_fisher_ratio", "embed_silhouette", "embed_centroid_dist"):
            if embed_key in _embed_metrics_this_epoch:
                epoch_record[embed_key] = _embed_metrics_this_epoch[embed_key]
        _epoch_history.append(epoch_record)

        # Log RPPL-specific scalars to MLflow
        if trainer.mlflow and any(k.startswith("rppl_") for k in loss_breakdown):
            try:
                rppl_extras: Dict[str, float] = {}
                # Warmup geo_scale
                rppl_crit = next(
                    (e["criterion"] for e in loss_manager.losses
                     if e.get("name") == "rppl" and hasattr(e.get("criterion"), "_epoch")),
                    None,
                )
                if rppl_crit is not None:
                    geo_scale = min(1.0, rppl_crit._epoch / max(1, rppl_crit.warmup_epochs))
                    rppl_extras["rppl_geo_scale"] = geo_scale
                    ema_norm = float(rppl_crit.proto_w_ema.norm().item())
                    rppl_extras["rppl_proto_ema_norm"] = ema_norm
                    epoch_record["rppl_proto_ema_norm"] = ema_norm
                trainer.mlflow_log("log_metrics", rppl_extras, step=ep + 1)
            except Exception as exc:
                logger.debug("RPPL scalar logging failed: %s", exc)

        if trainer.mlflow:
            trainer.mlflow_log("log_metrics", {
                    "accuracy": acc, "precision": prec, "recall": rec,
                    "f1": f1, "auc": auc,
                    "fpr": fpr, "fnr": fnr,
                    "eer": det_report.eer,
                    "n_hard_negatives": len(hard_negatives),
                    "n_easy_negatives": len(easy_negatives),
                    "neg_weight": current_neg_weight,
                    "n_fp": fp_count, "n_fn": fn_count,
                }, step=ep + 1)
            if trainer.mlflow:  # still active after log attempt
                try:
                    _log_fp_fn_artifacts(trainer.mlflow, ep, paths_all, targets, preds, probs, output_dir)
                except Exception as exc:
                    logger.warning("Failed to log artifacts to MLflow: %s", exc)

        # Confusion matrix + threshold sensitivity — update every 10 epochs and at the end
        if output_dir and ((ep + 1) % 10 == 0 or ep == epochs - 1):
            viz_dir = output_dir / "viz"
            viz_dir.mkdir(parents=True, exist_ok=True)
            plot_confusion_matrix(targets, preds, ep + 1, viz_dir, trainer.mlflow)
            if len(set(targets)) > 1:
                plot_threshold_sensitivity(targets, probs, viz_dir, trainer.mlflow)

        # RPPL dashboard — periodic snapshot
        _is_rppl = any(cfg.get("name") == "rppl" for cfg in trainer.losses_cfg)
        _effective_rppl_every = rppl_every if rppl_every > 0 else (5 if _is_rppl else 0)
        if output_dir and _effective_rppl_every and (ep + 1) % _effective_rppl_every == 0:
            _rppl_crit = next(
                (e["criterion"] for e in loss_manager.losses if e.get("name") == "rppl"),
                None,
            )
            _warmup = getattr(_rppl_crit, "warmup_epochs", 5) if _rppl_crit else 5
            plot_rppl_dashboard(
                _epoch_history, output_dir / "viz", trainer.mlflow, warmup_epochs=_warmup
            )

        scheduler.step()

        updated = _update_best_checkpoints(
            trainer, ep, avg_loss, prec, rec, f1, best_metrics, optimizer, output_dir, save_best,
        )
        if updated:
            logger.info("Updated best model(s): %s", ", ".join(updated))

        if ep == epochs - 1:
            break

        # Hard-negative mining
        tqdm.write(f"  [ep {ep+1}] mining hard negatives (fraction={mine_fraction}) …")
        if not hasattr(trainer, "hardness_cache"):
            trainer.hardness_cache = hardness_cache
        prev_hard_paths = {x[0] for x in hard_negatives}
        new_hards, easy_negatives, trainer.hardness_cache = mine_hard_negatives(
            model=trainer.model, nonwakes=nonwakes, device=trainer.device,
            hardness_cache=trainer.hardness_cache,
            neg_threshold=neg_threshold, dataset_fraction=mine_fraction,
            max_cache_size=3 * max(1, len(wakes)),
            wake_cache=getattr(trainer, "_wake_cache", []),
            feature_cache=feature_cache,
        )
        if new_hards:
            max_samples = max(1, len(wakes)) * 3
            hard_negatives = new_hards[:max_samples]
            new_paths = {x[0] for x in hard_negatives} - prev_hard_paths
            if new_paths:
                epochs_no_new = 0
            else:
                epochs_no_new += 1
            logger.info("  -> %d hard negatives (%d new)", len(hard_negatives), len(new_paths))
        elif mine_fraction > 0:
            hard_negatives = []
            epochs_no_new += 1
            logger.info("  -> No hard negatives found (%d/%d)", epochs_no_new, patience)

        if mine_fraction > 0 and epochs_no_new >= patience:
            logger.info("Early stopping — no new hard negatives for %d epochs.", patience)
            break

    # Stage 2: fit OCSVM on positive embeddings if head supports it
    if hasattr(trainer.model.classifier, "fit_ocsvm"):
        try:
            logger.info("OCSVMHead detected — fitting OCSVM on training positives.")
            _ocsvm_loader = DataLoader(
                AudioDataset(train_data, device=trainer.device.type, aug_prob=0.0),
                batch_size=batch_size, shuffle=False,
                collate_fn=lambda b: collate_fn(b, trainer.device),
            )
            trainer.model.classifier.fit_ocsvm(_ocsvm_loader)
        except Exception as exc:
            logger.warning("fit_ocsvm() failed (skipping): %s", exc)

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

    # End-of-training summary artifacts
    if output_dir:
        viz_dir = output_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        plot_training_curves(_epoch_history, viz_dir, trainer.mlflow)
        if any(cfg.get("name") == "rppl" for cfg in trainer.losses_cfg):
            _rppl_crit = next(
                (e["criterion"] for e in loss_manager.losses if e.get("name") == "rppl"),
                None,
            )
            _warmup = getattr(_rppl_crit, "warmup_epochs", 5) if _rppl_crit else 5
            plot_rppl_dashboard(
                _epoch_history, viz_dir, trainer.mlflow, warmup_epochs=_warmup
            )

    if trainer.mlflow:
        trainer.mlflow_log("log_metrics", {
            "best_f1":        best_metrics.get("f1", 0.0),
            "best_precision": best_metrics.get("precision", 0.0),
            "best_recall":    best_metrics.get("recall", 0.0),
            "best_loss":      best_metrics.get("loss", 0.0),
        })
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
