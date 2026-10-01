"""ONNX quantization helpers for wake word models.

Provides systematic INT8/INT16 quantization for both extractors and
classifier heads, with optional accuracy validation.

Usage::

    from ww_trainer.quantize import quantize_onnx, quantize_model_pair

    # Quantize a single ONNX file
    quantize_onnx("model.onnx", "model_int8.onnx", weight_type="int8")

    # Quantize both extractor + head and measure size reduction
    report = quantize_model_pair(
        extractor_path="extractor.onnx",
        head_path="head.onnx",
        output_dir="quantized/",
    )
    print(report)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class QuantizationReport:
    """Report from quantizing a model pair.

    Attributes:
        extractor_original_kb: Original extractor ONNX size in KB.
        extractor_int8_kb: INT8 quantized extractor size in KB.
        head_original_kb: Original head ONNX size in KB.
        head_int8_kb: INT8 quantized head size in KB.
        size_reduction_pct: Overall size reduction percentage.
        numerical_diff: Max absolute difference between original and quantized outputs.
    """
    extractor_original_kb: float = 0.0
    extractor_int8_kb: float = 0.0
    head_original_kb: float = 0.0
    head_int8_kb: float = 0.0
    size_reduction_pct: float = 0.0
    numerical_diff: Optional[float] = None


def quantize_onnx(
    input_path: str,
    output_path: str,
    weight_type: str = "int8",
    op_types: Optional[list[str]] = None,
) -> str:
    """Quantize a single ONNX model file.

    Args:
        input_path: Path to original ONNX file.
        output_path: Path for quantized output.
        weight_type: ``"int8"`` or ``"int16"``.
        op_types: Operator types to quantize (default: MatMul, Gemm, Conv).

    Returns:
        Path to the quantized file.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    qt = QuantType.QInt8 if weight_type == "int8" else QuantType.QInt16
    ops = op_types or ["MatMul", "Gemm", "Conv"]

    quantize_dynamic(
        input_path, output_path,
        op_types_to_quantize=ops,
        weight_type=qt,
    )
    return output_path


def quantize_model_pair(
    extractor_path: str,
    head_path: str,
    output_dir: str = "quantized",
    weight_type: str = "int8",
    validate: bool = True,
    sample_rate: int = 16000,
) -> QuantizationReport:
    """Quantize both extractor and head ONNX files.

    Optionally validates by comparing outputs of original vs quantized
    models on random audio input.

    Args:
        extractor_path: Path to extractor ONNX.
        head_path: Path to head ONNX.
        output_dir: Directory for quantized files.
        weight_type: ``"int8"`` or ``"int16"``.
        validate: If True, run inference comparison.
        sample_rate: Audio sample rate for validation.

    Returns:
        :class:`QuantizationReport` with size and accuracy info.
    """
    import onnxruntime as ort

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    suffix = f"_{weight_type}"
    ext_out = str(out / (Path(extractor_path).stem + suffix + ".onnx"))
    head_out = str(out / (Path(head_path).stem + suffix + ".onnx"))

    quantize_onnx(extractor_path, ext_out, weight_type)
    quantize_onnx(head_path, head_out, weight_type)

    ext_orig_kb = os.path.getsize(extractor_path) / 1024
    ext_q_kb = os.path.getsize(ext_out) / 1024
    head_orig_kb = os.path.getsize(head_path) / 1024
    head_q_kb = os.path.getsize(head_out) / 1024

    total_orig = ext_orig_kb + head_orig_kb
    total_q = ext_q_kb + head_q_kb
    reduction = (1.0 - total_q / total_orig) * 100 if total_orig > 0 else 0.0

    report = QuantizationReport(
        extractor_original_kb=ext_orig_kb,
        extractor_int8_kb=ext_q_kb,
        head_original_kb=head_orig_kb,
        head_int8_kb=head_q_kb,
        size_reduction_pct=reduction,
    )

    if validate:
        # Compare original vs quantized on random input
        audio = np.random.randn(1, sample_rate).astype(np.float32)

        sess_orig_ext = ort.InferenceSession(extractor_path, providers=["CPUExecutionProvider"])
        sess_q_ext = ort.InferenceSession(ext_out, providers=["CPUExecutionProvider"])

        in_name = sess_orig_ext.get_inputs()[0].name
        out_name = sess_orig_ext.get_outputs()[0].name

        feats_orig = sess_orig_ext.run([out_name], {in_name: audio})[0]
        feats_q = sess_q_ext.run([out_name], {in_name: audio})[0]

        sess_orig_head = ort.InferenceSession(head_path, providers=["CPUExecutionProvider"])
        sess_q_head = ort.InferenceSession(head_out, providers=["CPUExecutionProvider"])

        h_in = sess_orig_head.get_inputs()[0].name
        h_out = sess_orig_head.get_outputs()[0].name

        logit_orig = sess_orig_head.run([h_out], {h_in: feats_orig})[0]
        logit_q = sess_q_head.run([h_out], {h_in: feats_q})[0]

        report.numerical_diff = float(np.max(np.abs(logit_orig - logit_q)))

    return report
