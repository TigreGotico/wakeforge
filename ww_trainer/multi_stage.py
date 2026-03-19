"""Multi-stage training with checkpoint averaging.

Inspired by openWakeWord's cascaded training (lr 1e-4 -> 1e-5 -> 1e-6)
with top-K checkpoint averaging for the final model.

Usage::

    from ww_trainer.multi_stage import run_multi_stage_training, average_checkpoints

    stages = [
        {"epochs": 30, "lr": 1e-4, "max_neg_weight": 100},
        {"epochs": 5, "lr": 1e-5},
        {"epochs": 5, "lr": 1e-6},
    ]
    run_multi_stage_training(trainer, stages, train_data, test_data, output_dir)
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


def average_checkpoints(
    checkpoint_paths: List[Union[str, Path]],
    top_k: Optional[int] = None,
    device: str = "cpu",
) -> OrderedDict:
    """Average model weights from multiple checkpoint files.

    Loads N state dicts and computes the element-wise mean of all
    parameters.  Used after multi-stage training to produce a
    smoother final model.

    Args:
        checkpoint_paths: Paths to ``.pt`` model checkpoint files.
        top_k: If provided, only average the top K checkpoints
            (assumes list is ordered by quality, best first).
        device: Device for loading checkpoints.

    Returns:
        Averaged state dict (``OrderedDict``).

    Raises:
        ValueError: If no checkpoint paths are provided.
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided.")

    if top_k is not None and top_k > 0:
        checkpoint_paths = checkpoint_paths[:top_k]

    state_dicts: List[OrderedDict] = []
    for path in checkpoint_paths:
        state = torch.load(str(path), map_location=device)
        # Handle both raw state dicts and wrapped checkpoints
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        state_dicts.append(OrderedDict(state))

    if len(state_dicts) == 1:
        return state_dicts[0]

    avg_state: OrderedDict = OrderedDict()
    keys = state_dicts[0].keys()
    n = len(state_dicts)

    for key in keys:
        tensors = [sd[key].float() for sd in state_dicts]
        avg_state[key] = (sum(tensors) / n).to(tensors[0].dtype)

    logger.info("Averaged %d checkpoints", n)
    return avg_state


def parse_stage_spec(spec: str) -> List[Dict[str, Any]]:
    """Parse a CLI stage specification string.

    Format: ``"epochs:lr,epochs:lr,..."``
    Example: ``"30:1e-4,5:1e-5,5:1e-6"``

    Args:
        spec: Comma-separated ``epochs:lr`` pairs.

    Returns:
        List of stage config dicts with ``"epochs"`` and ``"lr"`` keys.
    """
    stages: List[Dict[str, Any]] = []
    for part in spec.split(","):
        part = part.strip()
        if ":" not in part:
            raise ValueError(f"Invalid stage spec: {part!r}. Expected 'epochs:lr'.")
        epochs_str, lr_str = part.split(":", 1)
        stages.append({"epochs": int(epochs_str), "lr": float(lr_str)})
    return stages


def run_multi_stage_training(
    trainer: Any,
    stages: List[Dict[str, Any]],
    train_data: List[Tuple[str, str]],
    test_data: List[Tuple[str, str]],
    output_dir: Union[str, Path],
    **shared_kwargs: Any,
) -> float:
    """Run cascaded multi-stage training with checkpoint averaging.

    Each stage resumes from the previous stage's best F1 checkpoint.
    After all stages, averages the top-K checkpoints by F1.

    Args:
        trainer: A ``WakeWordTrainer`` instance.
        stages: List of stage configs. Each dict may contain:
            ``"epochs"``, ``"lr"``, ``"max_neg_weight"``, and any
            other kwargs accepted by ``trainer.train()``.
        train_data: Training data as ``[(path, label), ...]``.
        test_data: Test data.
        output_dir: Base output directory.
        **shared_kwargs: Kwargs passed to every stage's ``train()`` call.

    Returns:
        Best F1 score across all stages.
    """
    output_dir = Path(output_dir)
    best_f1 = 0.0
    best_ckpt: Optional[Path] = None
    all_best_ckpts: List[Path] = []

    for i, stage in enumerate(stages):
        stage_dir = output_dir / f"stage_{i}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Merge shared kwargs with stage-specific overrides
        kwargs = {**shared_kwargs}
        kwargs.update(stage)
        epochs = kwargs.pop("epochs", 30)
        lr = kwargs.pop("lr", 1e-4)

        # Resume from previous stage's best checkpoint
        resume = None
        if best_ckpt is not None and best_ckpt.exists():
            resume = str(best_ckpt)

        logger.info("=== Multi-Stage %d/%d: epochs=%d, lr=%g ===", i + 1, len(stages), epochs, lr)

        f1 = trainer.train(
            output_dir=stage_dir,
            train_data=train_data,
            test_data=test_data,
            epochs=epochs,
            lr=lr,
            resume=resume,
            **kwargs,
        )

        stage_best = stage_dir / "best_f1.pt"
        if stage_best.exists():
            all_best_ckpts.append(stage_best)
            best_ckpt = stage_best

        if f1 > best_f1:
            best_f1 = f1

    # Average top checkpoints
    if len(all_best_ckpts) > 1:
        avg_state = average_checkpoints(all_best_ckpts)
        avg_path = output_dir / "averaged_model.pt"
        torch.save(avg_state, avg_path)
        logger.info("Saved averaged model from %d stages to %s", len(all_best_ckpts), avg_path)

        # Also load into trainer's model
        trainer.model.load_state_dict(avg_state, strict=False)
        final_path = output_dir / "final_model.pt"
        trainer.model.save_checkpoint(str(final_path))
        logger.info("Saved final averaged model to %s", final_path)

    return best_f1
