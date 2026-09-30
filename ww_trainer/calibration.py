"""Post-training confidence calibration via Platt scaling.

Fits a logistic regression on validation logits to produce calibrated
probabilities. Calibration parameters (slope + intercept) can be saved
alongside the model and applied at inference time.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def fit_platt_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Fit Platt scaling (logistic regression) on validation logits.

    Args:
        logits: 1-D array of raw model logits from validation set.
        labels: 1-D array of binary labels (0 or 1).

    Returns:
        Dict with ``coef`` (float) and ``intercept`` (float).
    """
    from sklearn.linear_model import LogisticRegression

    logits = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
    labels = np.asarray(labels, dtype=np.int64).ravel()

    cal = LogisticRegression(solver="lbfgs", max_iter=1000)
    cal.fit(logits, labels)

    params = {
        "coef": float(cal.coef_[0][0]),
        "intercept": float(cal.intercept_[0]),
    }
    logger.info("Platt scaling fit: coef=%.4f, intercept=%.4f", params["coef"], params["intercept"])
    return params


def apply_platt_scaling(
    logits: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Apply Platt scaling to raw logits.

    Args:
        logits: 1-D array of raw logits.
        params: Dict with ``coef`` and ``intercept`` from :func:`fit_platt_scaling`.

    Returns:
        1-D array of calibrated probabilities in [0, 1].
    """
    logits = np.asarray(logits, dtype=np.float64)
    z = params["coef"] * logits + params["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def save_calibration(params: dict, path: str) -> None:
    """Save calibration parameters to JSON.

    Args:
        params: Dict with ``coef`` and ``intercept``.
        path: Output JSON file path.
    """
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info("Calibration params saved to %s", path)


def load_calibration(path: str) -> dict:
    """Load calibration parameters from JSON.

    Args:
        path: Path to calibration JSON file.

    Returns:
        Dict with ``coef`` and ``intercept``.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    with open(path) as f:
        return json.load(f)


def calibrate_model(
    model: "torch.nn.Module",
    val_data: list,
    output_dir: str,
    device: str = "cpu",
    batch_size: int = 32,
    feature_store=None,
) -> dict:
    """End-to-end calibration: collect logits from val set, fit, save.

    Args:
        model: Trained ``BaseWakeModel``.
        val_data: List of ``(path, label)`` tuples.
        output_dir: Directory to save ``calibration.json``.
        device: Torch device.
        batch_size: Batch size for logit collection.
        feature_store: Optional :class:`~ww_trainer.feature_store.FeatureStore`
            of the model's frozen featurizer; stored features are reused.

    Returns:
        Calibration parameters dict.
    """
    import torch
    from ww_trainer.dataset import AudioDataset, collate_fn
    from ww_trainer.model import batch_forward
    from torch.utils.data import DataLoader

    dataset = AudioDataset(val_data, aug_prob=0.0, feature_store=feature_store)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, device))

    all_logits = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            _, labels, _, _ = batch
            logits = batch_forward(model, batch)
            all_logits.extend(logits.cpu().numpy().ravel())
            all_labels.extend(labels.cpu().numpy().ravel())

    params = fit_platt_scaling(np.array(all_logits), np.array(all_labels))

    out_path = str(Path(output_dir) / "calibration.json")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_calibration(params, out_path)

    return params

