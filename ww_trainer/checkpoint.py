"""Checkpoint save/load helpers extracted from WakeWordTrainer."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch


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
    print(f"[Checkpoint] Saved model + trainer state at epoch {epoch} -> {model_ckpt_path}")


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
        print(f"[Resume] Loaded checkpoint from epoch {start_epoch}")
    else:
        print("[Resume] Loaded model weights only (trainer state missing)")

    return start_epoch, metrics
