"""Reproducibility utilities for deterministic training.

Usage::

    from ww_trainer.reproducibility import set_seed, get_seed_info

    set_seed(42)  # Sets Python, NumPy, PyTorch, CUDA seeds
    print(get_seed_info())  # Returns dict of environment info
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducible training.

    Sets seeds for Python ``random``, ``numpy``, ``torch``, and CUDA.
    Optionally enables deterministic CuDNN algorithms (slower but reproducible).

    Args:
        seed: Random seed value.
        deterministic: If True, enable deterministic CuDNN and disable
                       benchmark mode. Slows training but ensures
                       bit-exact reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass  # Not all operations support deterministic mode


def get_seed_info() -> dict:
    """Return environment info relevant to reproducibility.

    Returns:
        Dict with Python, NumPy, PyTorch versions, CUDA availability,
        CuDNN status, and deterministic flag.
    """
    import sys
    info = {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_enabled": torch.backends.cudnn.enabled if torch.cuda.is_available() else False,
        "cudnn_deterministic": torch.backends.cudnn.deterministic if torch.cuda.is_available() else None,
        "cudnn_benchmark": torch.backends.cudnn.benchmark if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info
