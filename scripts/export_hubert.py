#!/usr/bin/env python3
"""Export a HuBERT encoder to ONNX for use as OnnxFeatureExtractor.

Uses the Hugging Face `optimum` library for reliable ONNX export of transformer
models. The resulting ONNX file is compatible with `ww_trainer.feats.OnnxFeatureExtractor`
and should be used for **both training and inference** — this guarantees feature parity.

Usage::

    # Install export dependencies
    pip install optimum[exporters] transformers

    # Export HuBERT base (768-dim, ~95M params)
    .venv/bin/python scripts/export_hubert.py \\
        --model voidful/hubert-tiny-v2 \\
        --output hubert-tiny-v2.onnx

    # Export with INT8 quantization
    .venv/bin/python scripts/export_hubert.py \\
        --model facebook/hubert-base-ls960 \\
        --output hubert-base.onnx \\
        --quantize

    # Verify an already-exported file
    .venv/bin/python scripts/export_hubert.py --verify hubert-tiny-v2.onnx

After export, use for training::

    from ww_trainer.quickstart import train_from_wakeword
    result = train_from_wakeword(
        "hey jarvis", "./hey_jarvis",
        featurizer_type="onnx",
        featurizer="hubert-tiny-v2.onnx",
        tier="ssl_small",
    )

Published ONNX variants: https://huggingface.co/TigreGotico/onnx-feature-extractors
"""

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Well-known HuBERT checkpoints
KNOWN_MODELS = {
    "tiny":   "voidful/hubert-tiny-v2",
    "base":   "facebook/hubert-base-ls960",
    "large":  "facebook/hubert-large-ll60k",
}


def export(model_name: str, output_path: str, quantize: bool = False, opset: int = 17) -> None:
    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        raise SystemExit(
            "Install optimum for ONNX export:\n"
            "  pip install 'optimum[exporters]' transformers"
        )

    logger.info("Exporting %s → %s", model_name, output_path)
    with tempfile.TemporaryDirectory() as tmp:
        main_export(
            model_name_or_path=model_name,
            output=tmp,
            task="feature-extraction",
            opset=opset,
            optimize=None,
            no_post_process=True,
        )
        # optimum writes model.onnx inside the temp dir
        src = next(Path(tmp).glob("*.onnx"))
        shutil.copy(src, output_path)

    import onnx
    onnx.checker.check_model(onnx.load(output_path))
    logger.info("Validated ✓  size=%.1f MB", Path(output_path).stat().st_size / 1e6)

    if quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = str(Path(output_path).with_stem(Path(output_path).stem + "_int8"))
        quantize_dynamic(output_path, int8_path,
                         op_types_to_quantize=["MatMul", "Gemm"],
                         weight_type=QuantType.QInt8)
        logger.info("INT8 quantized → %s  (%.1f MB)", int8_path,
                    Path(int8_path).stat().st_size / 1e6)


def verify(onnx_path: str, sample_rate: int = 16000) -> None:
    import torch
    from ww_trainer.feats import OnnxFeatureExtractor
    ext = OnnxFeatureExtractor(onnx_path, sample_rate=sample_rate)
    wav = torch.zeros(1, sample_rate)
    feats = ext(wav)
    logger.info("output shape=%s  feature_dim=%d  ✓", tuple(feats.shape), ext.feature_dim)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HuBERT encoder to ONNX")
    parser.add_argument("--model", default="voidful/hubert-tiny-v2",
                        help="HF model ID or shorthand: tiny / base / large")
    parser.add_argument("--output", default=None, help="Output .onnx path")
    parser.add_argument("--quantize", action="store_true", help="Also export INT8 variant")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset (default 17)")
    parser.add_argument("--verify", metavar="ONNX_PATH", default=None,
                        help="Only verify an existing ONNX file")
    args = parser.parse_args()

    if args.verify:
        verify(args.verify)
        return

    model_name = KNOWN_MODELS.get(args.model, args.model)
    output = args.output or (model_name.split("/")[-1] + ".onnx")
    export(model_name, output, quantize=args.quantize, opset=args.opset)


if __name__ == "__main__":
    main()
