"""Standalone evaluation and metric-logging helpers.

Extracted from ``WakeWordTrainer`` so they can be used without instantiating
the full trainer.
"""
import csv
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ww_trainer.dataset import AudioDataset, collate_fn
from ww_trainer.metrics import DetectionReport, classification_report as _metrics_report
from ww_trainer.visualization import plot_roc, plot_pr, plot_det

logger = logging.getLogger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    dataset: List[Tuple[str, str]],
    device: torch.device,
    batch_size: int = 128,
    threshold: float = 0.5,
    epoch: int = 0,
    output_dir: Optional[Path] = None,
    aug_prob: float = 0,
    mlflow=None,
) -> tuple:
    """Run evaluation on *dataset* and return metrics.

    Args:
        model: The wake-word model (must support ``model(wavs)``).
        dataset: List of ``(path, label)`` tuples.
        device: Torch device for inference.
        batch_size: DataLoader batch size.
        threshold: Classification threshold.
        epoch: Current epoch (used for logging).
        output_dir: If set, ROC/PR/DET plots are saved here.
        aug_prob: Augmentation probability for the dataset.
        mlflow: Optional mlflow module for metric logging.

    Returns:
        Tuple of ``(acc, prec, rec, f1, auc, fp_paths, fn_paths,
        paths_all, targets, preds, probs)``.
    """
    if not dataset:
        return 0.0, 0.0, 0.0, 0.0, 0.0, [], [], [], [], [], []

    loader = DataLoader(
        AudioDataset(dataset, aug_prob=aug_prob),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, device),
    )

    preds, probs, targets = [], [], []
    paths_all: List[str] = []
    model.eval()
    with torch.no_grad():
        for wavs, labels, paths in tqdm(loader, desc="Evaluating", leave=False):
            logits = model(wavs)
            prob = torch.sigmoid(logits).cpu().numpy().flatten()
            pred = (prob > threshold).astype(int)
            preds.extend(pred.tolist())
            probs.extend(prob.tolist())
            targets.extend(labels.cpu().int().numpy().tolist())
            paths_all.extend(paths)

    acc = accuracy_score(targets, preds) if len(targets) > 0 else 0.0
    prec = precision_score(targets, preds, zero_division=0) if len(targets) > 0 else 0.0
    rec = recall_score(targets, preds, zero_division=0) if len(targets) > 0 else 0.0
    f1 = f1_score(targets, preds, zero_division=0) if len(targets) > 0 else 0.0
    try:
        auc = roc_auc_score(targets, probs) if len(set(targets)) > 1 else 0.0
    except Exception:
        auc = 0.0

    fp_paths = [p for p, t, pr in zip(paths_all, targets, preds) if pr == 1 and t == 0]
    fn_paths = [p for p, t, pr in zip(paths_all, targets, preds) if pr == 0 and t == 1]

    if mlflow:
        try:
            tgt_arr = np.array(targets)
            prob_arr = np.array(probs)
            pos_mean = np.mean(prob_arr[tgt_arr == 1]) if np.any(tgt_arr == 1) else 0
            neg_mean = np.mean(prob_arr[tgt_arr == 0]) if np.any(tgt_arr == 0) else 0
            separation = pos_mean - neg_mean
            mlflow.log_metrics({
                "mean_conf_wake": pos_mean,
                "mean_conf_nonwake": neg_mean,
                "mean_conf_gap": separation,
            }, step=epoch)
        except Exception as e:
            logger.error("Failed to log confidence stats to MLflow: %s", e)

    if output_dir is not None:
        plot_dir = Path(output_dir) / "roc_pr_det"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_roc(targets, probs, epoch, plot_dir, auc, mlflow)
        plot_pr(targets, probs, epoch, plot_dir, mlflow)
        plot_det(targets, probs, epoch, plot_dir, mlflow)

    # Build DetectionReport from collected targets/probs
    report = _build_detection_report(targets, probs, threshold)

    return acc, prec, rec, f1, auc, fp_paths, fn_paths, paths_all, targets, preds, probs, report


def _build_detection_report(
    targets: list,
    probs: list,
    threshold: float = 0.5,
) -> DetectionReport:
    """Build a DetectionReport from raw targets and probabilities."""
    if not targets:
        return DetectionReport()
    y_true = np.array(targets)
    y_scores = np.array(probs)
    return _metrics_report(y_true, y_scores, threshold=threshold)


def evaluate_detection(
    model: torch.nn.Module,
    dataset: List[Tuple[str, str]],
    device: torch.device,
    batch_size: int = 128,
    threshold: Optional[float] = None,
) -> DetectionReport:
    """Evaluate model and return a DetectionReport with EER, FAR/FRR, DET data.

    This is the recommended evaluation function for production use.
    If no threshold is provided, the optimal F1 threshold is auto-selected.

    Args:
        model: Wake word model.
        dataset: List of ``(path, label)`` tuples.
        device: Torch device.
        batch_size: Batch size for inference.
        threshold: Decision threshold (None = auto-optimize for F1).

    Returns:
        :class:`~ww_trainer.metrics.DetectionReport` with all metrics.
    """
    result = evaluate_model(model, dataset, device, batch_size=batch_size,
                            threshold=threshold or 0.5)
    return result[-1]  # last element is the DetectionReport


def compute_fitness_score(
    f1: float,
    fp_rate: float,
    fn_rate: float,
    param_count: int,
    param_budget: int,
    fp_weight: float = 0.8,
    fn_weight: float = 0.2,
    size_weight: float = 0.1,
) -> float:
    """Compute a composite training fitness score.

    Penalizes false positives 4x more than false negatives (by default), with
    a model-size penalty for exceeding ``param_budget``.

    Args:
        f1: F1 score (unused in formula but kept for API consistency).
        fp_rate: False positive rate (0–1).
        fn_rate: False negative rate (0–1).
        param_count: Number of trainable model parameters.
        param_budget: Target parameter budget.
        fp_weight: Weight for FP penalty (default 0.8).
        fn_weight: Weight for FN penalty (default 0.2).
        size_weight: Weight for size penalty (default 0.1).

    Returns:
        Fitness score in range [0, 1].  Higher is better.
    """
    detection_score = 1.0 - fp_weight * fp_rate - fn_weight * fn_rate
    size_penalty = max(0.0, 1.0 - size_weight * max(0.0, param_count / max(1, param_budget) - 1.0))
    return max(0.0, detection_score * size_penalty)


def log_metrics_csv(
    path: str,
    epoch: int,
    loss: float,
    acc: float,
    prec: float,
    rec: float,
    f1: float,
    auc: float,
) -> None:
    """Append one row of per-epoch metrics to a CSV file.

    Creates the file with a header row if it does not yet exist.

    Args:
        path: Filesystem path to the CSV file.
        epoch: Epoch number.
        loss: Average training loss.
        acc: Accuracy.
        prec: Precision.
        rec: Recall.
        f1: F1 score.
        auc: ROC AUC.
    """
    new = not Path(path).exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["epoch", "loss", "accuracy", "precision", "recall", "f1", "auc"])
        writer.writerow([epoch, loss, acc, prec, rec, f1, auc])
