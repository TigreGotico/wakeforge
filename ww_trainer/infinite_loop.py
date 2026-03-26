"""Infinite training loop with streaming hard-negative mining.

Unlike the standard ``loop.py`` which trains for a fixed number of epochs
over a fixed dataset, this module treats training as an open-ended search:

  - The NWW *pool* can be arbitrarily large (millions of files).
  - Each epoch a random ``scan_size`` subset is inferred → only the hard
    negatives (conf ≥ neg_threshold) survive into training.
  - Optionally, ``vc_per_epoch`` new voice-cloned positives are synthesised
    at the start of every epoch via chatterbox-onnx (CPU-only, ~1-3s each).
  - Stopping is goal-based, not epoch-count-based.

Usage
-----
    from ww_trainer.infinite_loop import StoppingGoal, infinite_training_loop

    goal = StoppingGoal(target_f1=0.95, target_eer=0.05)
    best_f1 = infinite_training_loop(trainer, output_dir, wakes, nww_pool,
                                      test_data, goal=goal)
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader

from ww_trainer.dataset import AudioDataset, collate_fn
from ww_trainer.evaluation import evaluate_model, log_metrics_csv, compute_readiness
from ww_trainer.loss import LossManager
from ww_trainer.metrics import find_optimal_threshold
from ww_trainer.mining import mine_hard_negatives, save_mining_cache, load_mining_cache
from ww_trainer.checkpoint import save_intermediate_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stopping goal
# ---------------------------------------------------------------------------

@dataclass
class StoppingGoal:
    """Multi-metric stopping criteria for infinite training.

    Training continues until **all** enabled targets are met AND the model
    has not improved for ``patience_after`` epochs, OR until a hard plateau
    (no new hard negatives for ``hard_plateau`` epochs) is detected, OR the
    optional ``max_epochs`` ceiling is reached.

    Set a target to ``None`` to disable that criterion.
    """
    target_f1:       Optional[float] = 0.90   # minimum F1 required
    target_eer:      Optional[float] = 0.10   # maximum EER required
    target_far_frr1: Optional[float] = None   # FAR at FRR=1%
    target_far_frr5: Optional[float] = None   # FAR at FRR=5%

    patience_after:  int   = 10    # epochs of no improvement after goals are met
    hard_plateau:    int   = 20    # stop if no new hard negatives for this many epochs
    min_epochs:      int   = 20    # never stop before this
    max_epochs:      Optional[int] = None    # hard ceiling (None = unlimited)

    def goals_met(self, metrics: dict) -> bool:
        """Return True if all enabled targets are satisfied."""
        checks = [
            (self.target_f1,       metrics.get("f1", 0.0),       lambda v, t: v >= t),
            (self.target_eer,      metrics.get("eer", 1.0),      lambda v, t: v <= t),
            (self.target_far_frr1, metrics.get("far_frr1", 1.0), lambda v, t: v <= t),
            (self.target_far_frr5, metrics.get("far_frr5", 1.0), lambda v, t: v <= t),
        ]
        return all(check(val, tgt) for tgt, val, check in checks if tgt is not None)

    def summary(self) -> str:
        parts = []
        if self.target_f1       is not None: parts.append(f"F1≥{self.target_f1}")
        if self.target_eer      is not None: parts.append(f"EER≤{self.target_eer}")
        if self.target_far_frr1 is not None: parts.append(f"FAR@FRR1%≤{self.target_far_frr1}")
        if self.target_far_frr5 is not None: parts.append(f"FAR@FRR5%≤{self.target_far_frr5}")
        return " | ".join(parts) or "no goals set"


# ---------------------------------------------------------------------------
# VC synthesis helper
# ---------------------------------------------------------------------------

_vc_backend_cache = None  # module-level singleton — loaded once


def _get_vc_backend(vc_backend: str, vc_device: str):
    """Return a cached VC backend instance, loading on first call."""
    global _vc_backend_cache
    if _vc_backend_cache is None:
        from ww_trainer.vc_helpers import load_vc_backend
        try:
            _vc_backend_cache = load_vc_backend(backend=vc_backend, device=vc_device)
            logger.info("[VC] Backend loaded: %s  sr=%d",
                        _vc_backend_cache.name, _vc_backend_cache.sample_rate)
        except Exception as exc:
            logger.warning("[VC] Failed to load backend %r: %s — VC synthesis disabled", vc_backend, exc)
            _vc_backend_cache = None
    return _vc_backend_cache


def _synthesise_positives(
    source_paths: List[Path],
    donor_paths: List[Path],
    out_dir: Path,
    n: int,
    seed: int,
    vc_backend: str = "auto",
    vc_device: str = "auto",
) -> List[Tuple[str, str]]:
    """Generate up to *n* VC positives by cloning existing wake-word clips into donor voices.

    Each output = one real wake-word audio clip voice-converted to a random donor's timbre.
    No text synthesis — chatterbox is used in VC mode only.

    Returns list of (path, "1") pairs for newly created files.
    Failures are silently skipped — the caller continues with fewer samples.
    """
    if not source_paths:
        logger.warning("[VC] No source wake-word clips available — skipping VC synthesis")
        return []

    backend = _get_vc_backend(vc_backend, vc_device)
    if backend is None:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    donors = rng.sample(donor_paths, min(n, len(donor_paths)))

    results = []
    for i, donor in enumerate(donors):
        out_path = out_dir / f"vc_ep_{seed:04d}_{i:04d}.wav"
        if out_path.exists():
            results.append((str(out_path), "1"))
            continue
        # Pick a random source wake-word clip to voice-convert
        source = rng.choice(source_paths)
        try:
            backend.vc(source, donor, out_path)
            results.append((str(out_path), "1"))
        except Exception as exc:
            logger.debug("[VC] Failed for donor %s / source %s: %s",
                         donor.name, source.name, exc)

    logger.info("[VC] Generated %d/%d VC positives this epoch", len(results), n)
    return results


# ---------------------------------------------------------------------------
# Epoch data builder
# ---------------------------------------------------------------------------

def _build_epoch_data(
    wakes: List[Tuple[str, str]],
    hard_negatives: List[Tuple[str, str]],
    easy_negatives: List[Tuple[str, str]],
    n_neg_target: int,
    hard_ratio: float = 0.6,
    easy_ratio: float = 0.2,
) -> List[Tuple[str, str]]:
    """Build one epoch's training set from positives + mined negatives."""
    n_hard = int(n_neg_target * hard_ratio)
    n_easy = int(n_neg_target * easy_ratio)
    n_rand = n_neg_target - n_hard - n_easy

    selected: List[Tuple[str, str]] = []
    selected += hard_negatives[:n_hard]

    remaining_easy = [x for x in easy_negatives if x not in hard_negatives[:n_hard]]
    selected += remaining_easy[:n_easy]

    # Fill random from easy pool if hard pool was too small
    if len(selected) < n_hard + n_easy:
        deficit = (n_hard + n_easy) - len(selected)
        n_rand += deficit

    all_negs = hard_negatives + easy_negatives
    rand_pool = [x for x in all_negs if x not in selected]
    if rand_pool and n_rand > 0:
        selected += random.sample(rand_pool, min(n_rand, len(rand_pool)))

    data = list(wakes) + selected
    random.shuffle(data)
    return data


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def infinite_training_loop(
    trainer,
    output_dir: "str | Path",
    wakes: List[Tuple[str, str]],
    nww_pool: List[Tuple[str, str]],
    test_data: List[Tuple[str, str]],
    *,
    goal: Optional[StoppingGoal] = None,
    scan_size: int = 5000,
    neg_threshold: float = 0.5,
    hard_ratio: float = 0.6,
    easy_ratio: float = 0.2,
    neg_multiplier: float = 3.0,       # target neg count = pos count × this
    lr: float = 5e-4,
    batch_size: int = 16,
    aug_prob: float = 0.7,
    aug_warmup_epochs: int = 3,
    use_mixup: bool = True,
    mixup_alpha: float = 1.0,
    spec_augment: bool = True,
    neg_weight_schedule: str = "linear",
    vc_per_epoch: int = 0,             # how many VC positives to generate per epoch
    vc_backend: str = "auto",          # "auto", "chatterbox-onnx", or "chatterbox"
    vc_device: str = "auto",           # PyTorch device for torch backend
    vc_out_dir: Optional["str | Path"] = None,
    eval_every: int = 1,               # evaluate every N epochs
    readiness_every: int = 3,
    cache_file: Optional[str] = None,  # path to persist mining cache across runs
    resume_cache: bool = True,
    **kwargs: Any,
) -> float:
    """Open-ended training loop; returns best F1 achieved."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if goal is None:
        goal = StoppingGoal()

    logger.info("=== Infinite Training Mode ===")
    logger.info("NWW pool: %d files | scan/epoch: %d", len(nww_pool), scan_size)
    logger.info("Stopping goals: %s", goal.summary())
    logger.info("Hard plateau: %d epochs | patience_after: %d | min_epochs: %d",
                goal.hard_plateau, goal.patience_after, goal.min_epochs)

    # ── Setup ────────────────────────────────────────────────────────────────
    model = trainer.model
    device = trainer.device

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=lr * 0.01,
    )
    loss_manager = LossManager(trainer.losses_cfg or [{"name": "bce", "weight": 1.0}])

    # Mining cache
    hardness_cache: Dict[str, float] = {}
    _cache_path = cache_file or str(output_dir / "mining_cache.pt")
    if resume_cache:
        hardness_cache = load_mining_cache(_cache_path)
        if hardness_cache:
            logger.info("[Cache] Loaded %d cached scores from %s", len(hardness_cache), _cache_path)

    # VC pools: NWW clips as voice donors, wake-word clips as conversion sources
    nww_paths = [Path(p) for p, _ in nww_pool]
    wake_paths = [Path(p) for p, _ in wakes]
    vc_out = Path(vc_out_dir) if vc_out_dir else output_dir / "vc_epoch_positives"

    # State
    best_f1 = 0.0
    best_metrics: dict = {}
    ep = 0
    epochs_no_new_hard = 0
    epochs_no_improvement = 0
    goals_first_met_at: Optional[int] = None
    prev_hard_paths: set = set()
    readiness = 0.5
    start_time = time.time()

    # Log goals to MLflow
    if trainer.mlflow:
        try:
            trainer.mlflow.log_params({
                "mode":             "infinite",
                "scan_size":        scan_size,
                "neg_threshold":    neg_threshold,
                "target_f1":        goal.target_f1,
                "target_eer":       goal.target_eer,
                "hard_plateau":     goal.hard_plateau,
                "patience_after":   goal.patience_after,
                "vc_per_epoch":     vc_per_epoch,
                "neg_multiplier":   neg_multiplier,
            })
        except Exception:
            pass

    # ── Main loop ────────────────────────────────────────────────────────────
    while True:
        ep_start = time.time()
        logger.info("=== Epoch %d ===", ep + 1)

        # -- 1. Optional VC synthesis ------------------------------------------
        epoch_wakes = list(wakes)
        if vc_per_epoch > 0 and nww_paths and wake_paths:
            vc_samples = _synthesise_positives(
                wake_paths, nww_paths, vc_out, vc_per_epoch,
                seed=ep, vc_backend=vc_backend, vc_device=vc_device,
            )
            epoch_wakes += vc_samples
            if vc_samples:
                logger.info("[VC] +%d positives this epoch (total pos: %d)",
                            len(vc_samples), len(epoch_wakes))

        # -- 2. Hard-negative mining from NWW pool -----------------------------
        # Use the best available wake prototype from any active loss criterion
        # (RPPL EMA > ArcFace center > CenterLoss center > ProxyNCA proxy).
        # Falls back to batch-mean of wake embeddings when none is available.
        loss_proto = trainer.loss_manager.get_wake_prototype()

        hard_negatives, easy_negatives, hardness_cache = mine_hard_negatives(
            model=model,
            nonwakes=nww_pool,
            device=device,
            hardness_cache=hardness_cache,
            neg_threshold=neg_threshold,
            dataset_fraction=min(1.0, scan_size / max(1, len(nww_pool))),
            cache_decay=0.7,
            max_cache_size=min(100_000, len(nww_pool)),
            wake_cache=epoch_wakes,
            rppl_proto=loss_proto,
        )

        # Persist cache
        save_mining_cache(hardness_cache, _cache_path)

        # Track new hard negatives for plateau detection
        current_hard_paths = {p for p, _ in hard_negatives}
        new_paths = current_hard_paths - prev_hard_paths
        if new_paths:
            epochs_no_new_hard = 0
        else:
            epochs_no_new_hard += 1
        prev_hard_paths = current_hard_paths

        logger.info("[HardNeg] Hard: %d | New: %d | No-new streak: %d/%d",
                    len(hard_negatives), len(new_paths),
                    epochs_no_new_hard, goal.hard_plateau)

        # -- 3. Build epoch training data --------------------------------------
        n_neg_target = int(len(epoch_wakes) * neg_multiplier)
        if not hard_negatives and not easy_negatives:
            # No hard negatives yet — use random subset of pool
            n_neg_target = max(n_neg_target, len(epoch_wakes) * 3)
            rand_negs = random.sample(nww_pool, min(n_neg_target, len(nww_pool)))
            epoch_data = list(epoch_wakes) + rand_negs
        else:
            epoch_data = _build_epoch_data(
                epoch_wakes, hard_negatives, easy_negatives,
                n_neg_target, hard_ratio, easy_ratio,
            )
        random.shuffle(epoch_data)

        pos_count = sum(1 for _, l in epoch_data if l == "1")
        neg_count = sum(1 for _, l in epoch_data if l == "0")
        logger.info("[Data] Epoch %d: %d samples (%d pos / %d neg)",
                    ep + 1, len(epoch_data), pos_count, neg_count)

        # -- 4. Augmentation schedule -----------------------------------------
        current_aug_prob = aug_prob * min(1.0, (ep + 1) / max(1, aug_warmup_epochs))

        # -- 5. Neg-weight schedule -------------------------------------------
        if neg_weight_schedule == "linear":
            ratio = neg_count / max(1, pos_count)
            neg_weight = max(1.0, min(ratio * 0.5, 5.0))
        elif neg_weight_schedule == "sqrt":
            ratio = neg_count / max(1, pos_count)
            neg_weight = max(1.0, ratio ** 0.5)
        else:
            neg_weight = 1.0

        # -- 6. Readiness ------------------------------------------------------
        if ep % readiness_every == 0 and hasattr(model, "embed"):
            pos_sample = random.sample(epoch_wakes, min(200, len(epoch_wakes)))
            neg_sample = random.sample(
                hard_negatives or easy_negatives or nww_pool,
                min(200, len(hard_negatives or easy_negatives or nww_pool)),
            )
            if pos_sample and neg_sample:
                try:
                    import numpy as np
                    def _get_embeds(samples):
                        wavs = []
                        for item in samples:
                            p = item[0] if isinstance(item, (list, tuple)) else item
                            wav, _ = torchaudio.load(p)
                            wavs.append(wav.squeeze(0))
                        with torch.no_grad():
                            return model.embed(wavs).cpu().numpy()
                    pos_emb = _get_embeds(pos_sample)
                    neg_emb = _get_embeds(neg_sample)
                    all_emb = np.concatenate([pos_emb, neg_emb], axis=0)
                    intra_pos_var = float(pos_emb.var(axis=0).mean()) if len(pos_emb) > 1 else 1.0
                    embed_var_total = float(all_emb.var(axis=0).mean())
                    readiness = compute_readiness({
                        "intra_pos_var": intra_pos_var,
                        "embed_var_total": embed_var_total,
                    })
                except Exception as _re:
                    logger.debug("Readiness computation skipped: %s", _re)
                    readiness = 0.0
                logger.info("[Readiness] %.3f", readiness)
                if trainer.mlflow:
                    try:
                        trainer.mlflow.log_metric("readiness", readiness, step=ep)
                    except Exception:
                        pass

        # -- 7. Unfreeze layers if scheduled ----------------------------------
        if (trainer.unfreeze_at_epoch is not None
                and ep == trainer.unfreeze_at_epoch):
            trainer._unfreeze()

        # -- 8. Train one epoch -----------------------------------------------
        dataset = AudioDataset(
            epoch_data,
            device=device.type,
            aug_prob=current_aug_prob,
            feature_cache=None,
            **trainer.augment_opts,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, device),
            num_workers=0,
            pin_memory=False,
        )

        model.train()
        total_loss = 0.0
        n_batches = 0

        for wavs, labels, paths in loader:
            labels_f = labels.float()
            neg_mask = labels_f == 0
            pos_mask = labels_f == 1
            weights = torch.ones_like(labels_f)
            weights[neg_mask] = neg_weight

            # Mixup
            if use_mixup and random.random() < 0.5:
                import numpy as np
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                idx = torch.randperm(len(wavs))
                mixed = [lam * w + (1 - lam) * wavs[idx[i]]
                         for i, w in enumerate(wavs)]
                mixed_labels = lam * labels_f + (1 - lam) * labels_f[idx]
                loss = loss_manager.compute(
                    model, mixed, mixed_labels, weights=weights,
                )
            else:
                loss = loss_manager.compute(
                    model, wavs, labels_f, weights=weights,
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)
        ep_elapsed = time.time() - ep_start
        logger.info("[Train] Epoch %d loss=%.4f  time=%.0fs", ep + 1, avg_loss, ep_elapsed)

        if trainer.mlflow:
            try:
                trainer.mlflow.log_metrics({
                    "train_loss":       avg_loss,
                    "aug_prob":         current_aug_prob,
                    "neg_weight":       neg_weight,
                    "hard_neg_count":   len(hard_negatives),
                    "epoch_size":       len(epoch_data),
                    "pos_count":        pos_count,
                    "neg_count":        neg_count,
                    "new_hard_paths":   len(new_paths),
                    "epochs_no_new_hard": epochs_no_new_hard,
                }, step=ep)
            except Exception:
                pass

        # -- 9. Evaluation ----------------------------------------------------
        if ep % eval_every == 0:
            threshold = 0.5
            metrics_tuple = evaluate_model(
                model, test_data, device,
                batch_size=batch_size * 4,
                threshold=threshold,
                epoch=ep,
                output_dir=output_dir,
                mlflow=trainer.mlflow,
            )
            # evaluate_model returns (f1, eer, auc, far, frr, prec, rec, threshold, n_fp, n_fn)
            if isinstance(metrics_tuple, (tuple, list)) and len(metrics_tuple) >= 2:
                f1_val  = float(metrics_tuple[0])
                eer_val = float(metrics_tuple[1])
                auc_val = float(metrics_tuple[2]) if len(metrics_tuple) > 2 else 0.0
            else:
                f1_val = eer_val = auc_val = 0.0

            # Try to get FAR@FRR metrics if available
            metrics_dict = {
                "f1":    f1_val,
                "eer":   eer_val,
                "auc":   auc_val,
            }
            if len(metrics_tuple) >= 10:
                metrics_dict["far"] = float(metrics_tuple[3])
                metrics_dict["frr"] = float(metrics_tuple[4])

            logger.info("[Eval] Epoch %d  F1=%.4f  EER=%.4f  AUC=%.4f",
                        ep + 1, f1_val, eer_val, auc_val)

            # Best model checkpoint
            if f1_val > best_f1:
                best_f1 = f1_val
                best_metrics = metrics_dict.copy()
                epochs_no_improvement = 0
                save_intermediate_checkpoint(
                    model, trainer.wake_word, trainer.arch,
                    ep, best_metrics, optimizer,
                    output_dir / "best_f1.pt",
                    trainer.export_onnx, trainer.mlflow,
                )
                logger.info("[Best] New best F1=%.4f at epoch %d", best_f1, ep + 1)
            else:
                epochs_no_improvement += 1

            # Log CSV
            log_metrics_csv(
                output_dir / "metrics.csv", ep + 1,
                metrics_dict.get("loss", 0.0),
                0.0,  # acc not computed in infinite loop
                0.0,  # prec not tracked here — use MLflow for full metrics
                0.0,
                metrics_dict.get("f1",  0.0),
                metrics_dict.get("auc", 0.0),
            )

            # Goals check
            if goal.goals_met(metrics_dict):
                if goals_first_met_at is None:
                    goals_first_met_at = ep
                    logger.info("[Goal] All goals met at epoch %d! "
                                "Patience window: %d epochs", ep + 1, goal.patience_after)
            else:
                # Goals no longer met (regression) — reset patience window
                goals_first_met_at = None

        ep += 1

        # -- 10. Stopping logic -----------------------------------------------
        stop_reason = None

        if goal.max_epochs is not None and ep >= goal.max_epochs:
            stop_reason = f"max_epochs={goal.max_epochs} reached"

        elif ep >= goal.min_epochs:
            if epochs_no_new_hard >= goal.hard_plateau:
                stop_reason = (f"hard plateau: no new hard negatives for "
                               f"{epochs_no_new_hard} epochs")

            elif (goals_first_met_at is not None
                  and epochs_no_improvement >= goal.patience_after):
                stop_reason = (f"goals met (since epoch {goals_first_met_at + 1}) + "
                               f"no improvement for {epochs_no_improvement} epochs")

        if stop_reason:
            logger.info("[Stop] %s", stop_reason)
            logger.info("[Done] Best F1=%.4f after %d epochs  "
                        "(total time %.1fh)",
                        best_f1, ep, (time.time() - start_time) / 3600)
            break

    # Final checkpoint
    save_intermediate_checkpoint(
        model, trainer.wake_word, trainer.arch,
        ep, best_metrics, optimizer,
        output_dir / "final.pt",
        trainer.export_onnx, trainer.mlflow,
    )

    if trainer.mlflow:
        try:
            trainer.mlflow.log_metrics({
                "best_f1":    best_f1,
                "total_epochs": ep,
                "total_hours":  (time.time() - start_time) / 3600,
            })
            trainer.mlflow.end_run()
        except Exception:
            pass

    return best_f1
