"""Quantization-Aware Training (QAT) utilities for MCU deployment.

Wraps PyTorch's ``torch.ao.quantization`` to prepare FFN heads for
int8 inference with minimal accuracy loss. Most beneficial for ESP32
tiers where post-training quantization can degrade accuracy.

Usage:
    model = prepare_qat(model)      # before training
    # ... train normally ...
    model = convert_qat(model)      # after training
    model.export_to_onnx("qat.onnx")
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def prepare_qat(
    model: nn.Module,
    backend: str = "x86",
) -> nn.Module:
    """Prepare a model for quantization-aware training.

    Inserts fake quantization observers into the model so that forward
    passes simulate int8 arithmetic. The model remains in float32 and
    can be trained normally.

    Args:
        model: The model to prepare (typically a ``BaseWakeModel`` or ``FfnClassifierHead``).
        backend: Quantization backend (``"x86"``, ``"fbgemm"``, ``"qnnpack"``).

    Returns:
        The prepared model (modified in-place).
    """
    model.train()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    torch.ao.quantization.prepare_qat(model, inplace=True)
    logger.info("QAT prepared with backend=%s", backend)
    return model


def convert_qat(model: nn.Module) -> nn.Module:
    """Convert a QAT-prepared model to a quantized model.

    Replaces fake quantization with real int8 operations. Call this
    after training is complete, before export.

    Args:
        model: QAT-prepared model (from :func:`prepare_qat`).

    Returns:
        Quantized model.
    """
    model.eval()
    quantized = torch.ao.quantization.convert(model, inplace=False)
    n_params = sum(p.numel() for p in quantized.parameters())
    logger.info("QAT conversion complete: %d params", n_params)
    return quantized
