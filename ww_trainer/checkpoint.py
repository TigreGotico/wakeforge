"""Checkpoint save/load helpers extracted from WakeWordTrainer."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
        model,
        epoch: int,
        metrics: dict,
        optimizer: Optional[torch.optim.Optimizer],
        out_path: "Path | str",
) -> None:
    """Save model weights and trainer state to disk.

    Args:
        model: The BaseWakeModel instance.
        epoch: Current epoch number.
        metrics: Dict of best metric values.
        optimizer: Optimizer whose state to save (or None).
        out_path: Output path (without or with suffix; .pt is used for weights, .ts for state).
    """
    out_path = Path(out_path) if isinstance(out_path, str) else out_path
    ckpt_dir = out_path.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_ckpt_path = out_path.with_suffix(".pt")
    model.save_checkpoint(str(model_ckpt_path))

    trainer_state = {
        "epoch": epoch,
        "metrics": metrics,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else {},
    }
    torch.save(trainer_state, out_path.with_suffix(".ts"))
    logger.info("[Checkpoint] Saved model + trainer state at epoch %d -> %s", epoch, model_ckpt_path)


def save_intermediate_checkpoint(
        model,
        wake_word: str,
        arch: str,
        epoch: int,
        metrics: dict,
        optimizer: Optional[torch.optim.Optimizer],
        model_file: Path,
        export_onnx: bool = False,
        mlflow=None,
) -> None:
    """Save a training checkpoint with optional ONNX export and MLflow logging.

    Args:
        model: The BaseWakeModel instance.
        wake_word: Wake word label (stored in ONNX metadata).
        arch: Architecture name (stored in ONNX metadata).
        epoch: Current epoch.
        metrics: Current best metrics dict.
        optimizer: Optimizer for state dict.
        model_file: Destination path for the ``.pt`` file.
        export_onnx: If True, also export the model to ONNX.
        mlflow: Optional mlflow module for artifact logging.
    """
    model_file = Path(model_file)
    onnx_path = model_file.with_suffix(".onnx")
    save_checkpoint(model, epoch, metrics or {}, optimizer, model_file)

    if export_onnx:
        logger.info("Exporting model to onnx: %s", onnx_path)
        meta: dict = {
            "wake_word": wake_word,
            "arch": arch,
            "epoch": epoch,
            "featurizer": model.feature_extractor.__class__.__name__,
        }
        if metrics:
            meta.update({f"metric_{k}": str(v) for k, v in metrics.items()})
        model.export_to_onnx(onnx_path, metadata=meta)

        # Export featurizer ONNX so the full inference pipeline is captured
        feat_onnx_path = model_file.with_name(model_file.stem + "_featurizer.onnx")
        try:
            if hasattr(model, "feature_extractor") and hasattr(model.feature_extractor, "export_to_onnx"):
                model.feature_extractor.export_to_onnx(str(feat_onnx_path))
                logger.info("Exported featurizer to onnx: %s", feat_onnx_path)
        except Exception as exc:
            logger.warning("Featurizer ONNX export failed (non-fatal): %s", exc)
            feat_onnx_path = None
    else:
        feat_onnx_path = None

    if mlflow is not None:
        stem = model_file.stem  # e.g. "best_f1"
        artifact_path = f"models/{stem}"
        if export_onnx:
            for path in [onnx_path, feat_onnx_path]:
                if path and Path(path).exists():
                    try:
                        mlflow.log_artifact(str(path), artifact_path=artifact_path)
                    except Exception as exc:
                        logger.error("Failed to log onnx to MLflow: %s", exc)
        try:
            mlflow.log_artifact(str(model_file), artifact_path=artifact_path)
        except Exception as exc:
            logger.error("Failed to log checkpoint to MLflow: %s", exc)


def load_checkpoint(
        model,
        path: "Path | str",
        device,
        optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, dict]:
    """Load model weights and trainer state from disk.

    Args:
        model: The BaseWakeModel instance to restore weights into.
        path: Path to the .pt model weights file.
        device: torch device for map_location.
        optimizer: If provided, its state is restored from the .ts file.

    Returns:
        (start_epoch, metrics)
    """
    model_path = Path(path) if isinstance(path, str) else path
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if hasattr(model, "load_checkpoint"):
        model.load_checkpoint(str(model_path))
    else:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)

    start_epoch = 0
    metrics: dict = {}
    trainer_state_path = model_path.with_suffix(".ts")
    if trainer_state_path.exists():
        state = torch.load(trainer_state_path, map_location=device)
        start_epoch = state.get("epoch", 0)
        metrics = state.get("metrics", {})
        if optimizer is not None and "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])
        logger.info("[Resume] Loaded checkpoint from epoch %d", start_epoch)
    else:
        logger.info("[Resume] Loaded model weights only (trainer state missing)")

    return start_epoch, metrics
