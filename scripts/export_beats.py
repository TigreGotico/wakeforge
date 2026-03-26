#!/usr/bin/env python3
"""Export BEATs audio SSL encoder to ONNX for use as OnnxFeatureExtractor.

Exports microsoft/beats-iter3-plus (and optionally other BEATs checkpoints)
to ONNX format compatible with ww_trainer.feats.OnnxFeatureExtractor.

The HuggingFace AutoFeatureExtractor preprocessor is not ONNX-traceable, so
this script wraps the model in an ONNX-safe normalisation step (mean/std per
sample) that matches what the processor does for 16 kHz mono float32 audio.

Usage::

    # Default model (microsoft/beats-iter3-plus)
    .venv/bin/python scripts/export_beats.py

    # Custom checkpoint, quantized
    .venv/bin/python scripts/export_beats.py \\
        --model microsoft/beats-iter3 \\
        --output beats-iter3.onnx \\
        --quantize

    # Verify the exported model
    .venv/bin/python scripts/export_beats.py --verify beats-iter3-plus.onnx
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class _BeatsOnnxWrapper(nn.Module):
    """ONNX-traceable wrapper around a BEATs encoder.

    Replaces the HuggingFace feature extractor (not traceable) with an
    inline per-sample mean/std normalisation, which is what the processor
    does for 16 kHz mono audio.

    Input:  [B, T]  raw float32 waveform, 16 kHz, mono
    Output: [B, T', hidden_size]  contextual frame embeddings
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        # input_values: [B, T]
        mean = input_values.mean(dim=-1, keepdim=True)
        std = input_values.std(dim=-1, keepdim=True).clamp(min=1e-5)
        normed = (input_values - mean) / std
        out = self.model(normed)
        return out.last_hidden_state  # [B, T', hidden_size]


def export(
    model_name: str,
    output_path: str,
    quantize: bool = False,
    opset: int = 18,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
) -> None:
    logger.info("Loading %s ...", model_name)
    try:
        from transformers import AutoModel
    except ImportError:
        raise SystemExit("Install transformers: pip install transformers")

    base_model = AutoModel.from_pretrained(model_name).eval()
    for p in base_model.parameters():
        p.requires_grad = False

    wrapper = _BeatsOnnxWrapper(base_model).eval()
    hidden_size = base_model.config.hidden_size
    logger.info("Hidden size: %d", hidden_size)

    dummy = torch.zeros(1, int(sample_rate * duration_s))

    out_path = str(output_path)
    logger.info("Tracing ONNX export → %s", out_path)
    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=["input_values"],
        output_names=["features"],
        dynamic_axes={
            "input_values": {0: "batch_size", 1: "time"},
            "features": {0: "batch_size", 1: "frames"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
        training=torch.onnx.TrainingMode.EVAL,
        verbose=False,
        external_data=False,
    )

    import onnx
    model_proto = onnx.load(out_path)
    onnx.checker.check_model(model_proto)
    logger.info("ONNX model validated ✓")

    # Embed metadata so OnnxFeatureExtractor knows the feature dim
    try:
        from ww_trainer.utils import embed_onnx_metadata
        embed_onnx_metadata(out_path, {
            "model_name": model_name,
            "feature_dim": str(hidden_size),
            "sample_rate": str(sample_rate),
            "extractor_type": "beats",
        })
    except Exception as exc:
        logger.warning("Could not embed metadata: %s", exc)

    if quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = str(Path(out_path).with_stem(Path(out_path).stem + "_int8"))
        quantize_dynamic(out_path, int8_path,
                         op_types_to_quantize=["MatMul", "Gemm"],
                         weight_type=QuantType.QInt8)
        logger.info("INT8 quantized model → %s", int8_path)

    size_mb = Path(out_path).stat().st_size / 1e6
    logger.info("Export complete: %s (%.1f MB)", out_path, size_mb)


def verify(onnx_path: str, sample_rate: int = 16000, duration_s: float = 1.0) -> None:
    import numpy as np
    from ww_trainer.feats import OnnxFeatureExtractor

    logger.info("Verifying %s with OnnxFeatureExtractor ...", onnx_path)
    extractor = OnnxFeatureExtractor(onnx_path, sample_rate=sample_rate)
    wav = torch.zeros(1, int(sample_rate * duration_s))
    feats = extractor(wav)
    logger.info("Output shape: %s  (feature_dim=%d)", tuple(feats.shape), extractor.feature_dim)
    logger.info("Verification passed ✓")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BEATs encoder to ONNX")
    parser.add_argument("--model", default="microsoft/beats-iter3-plus",
                        help="HuggingFace model ID (default: microsoft/beats-iter3-plus)")
    parser.add_argument("--output", default=None,
                        help="Output ONNX path (default: <model-slug>.onnx)")
    parser.add_argument("--quantize", action="store_true",
                        help="Also export an INT8 quantized version")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--verify", metavar="ONNX_PATH", default=None,
                        help="Only verify an already-exported ONNX file")
    args = parser.parse_args()

    if args.verify:
        verify(args.verify)
        return

    output = args.output or (args.model.split("/")[-1] + ".onnx")
    export(
        model_name=args.model,
        output_path=output,
        quantize=args.quantize,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
