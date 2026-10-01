"""Distributed Data Parallel (DDP) utilities for multi-GPU training.

Launch with ``torchrun``:

.. code-block:: bash

    torchrun --nproc_per_node=2 -m ww_trainer.cli train --ddp ...

This module provides helpers to:
- Initialize the process group
- Wrap a model in DDP
- Create a DistributedSampler for the DataLoader
- Gate logging/checkpointing to rank 0 only
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


def is_ddp_available() -> bool:
    """Check if DDP environment variables are set (i.e., launched via torchrun)."""
    return "LOCAL_RANK" in os.environ


def get_local_rank() -> int:
    """Return the local GPU rank from environment, or 0 if not in DDP mode."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Return the world size from environment, or 1 if not in DDP mode."""
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    """Return True if this is rank 0 (or if not in DDP mode)."""
    return get_local_rank() == 0


def setup_ddp(backend: str = "nccl") -> int:
    """Initialize the DDP process group.

    Args:
        backend: Communication backend (``"nccl"`` for GPU, ``"gloo"`` for CPU).

    Returns:
        Local rank.
    """
    local_rank = get_local_rank()
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(local_rank)
    logger.info("DDP initialized: rank=%d, world_size=%d", local_rank, get_world_size())
    return local_rank


def cleanup_ddp() -> None:
    """Destroy the DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(model: nn.Module, local_rank: int) -> nn.Module:
    """Wrap a model in DistributedDataParallel.

    Args:
        model: The model to wrap.
        local_rank: Local GPU rank.

    Returns:
        DDP-wrapped model.
    """
    model = model.to(local_rank)
    return DDP(model, device_ids=[local_rank])


def create_distributed_loader(
    dataset: "torch.utils.data.Dataset",
    batch_size: int,
    collate_fn: Optional[callable] = None,
    num_workers: int = 0,
) -> tuple:
    """Create a DataLoader with DistributedSampler.

    Args:
        dataset: The dataset.
        batch_size: Per-GPU batch size.
        collate_fn: Optional collate function.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of ``(DataLoader, DistributedSampler)``.
    """
    sampler = DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, sampler
