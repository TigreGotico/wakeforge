#!/usr/bin/env python3
"""Export a Wav2Vec2 encoder to ONNX for use as OnnxFeatureExtractor.

Uses the Hugging Face `optimum` library for reliable ONNX export of transformer
models. The resulting ONNX file is compatible with `ww_trainer.feats.OnnxFeatureExtractor`
and should be used for **both training and inference** — this guarantees feature parity.

Usage::

    # Install export dependencies
    pip install optimum[exporters] transformers

    # Export Wav2Vec2 base (768-dim)
    .venv/bin/python scripts/export_wav2vec2.py \\
        --model facebook/wav2vec2-base \\
        --output wav2vec2-base.onnx

    # Export with INT8 quantization
    .venv/bin/python scripts/export_wav2vec2.py \\
        --model facebook/wav2vec2-base \\
        --output wav2vec2-base.onnx \\
        --quantize

    # Verify an already-exported file
    .venv/bin/python scripts/export_wav2vec2.py --verify wav2vec2-base.onnx

After export, use for training::

    from ww_trainer.quickstart import train_from_wakeword
    result = train_from_wakeword(
        "hey jarvis", "./hey_jarvis",
        featurizer_type="onnx",
        featurizer="wav2vec2-base.onnx",
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

# Well-known Wav2Vec2 checkpoints
KNOWN_MODELS = {
    "tiny":   "patrickvonplaten/tiny-wav2vec2-no-tokenizer",
    "base":   "facebook/wav2vec2-base",
    "large":  "facebook/wav2vec2-large",
    "large-960h": "facebook/wav2vec2-large-960h",
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
    parser = argparse.ArgumentParser(description="Export Wav2Vec2 encoder to ONNX")
    parser.add_argument("--model", default="facebook/wav2vec2-base",
                        help="HF model ID or shorthand: tiny / base / large / large-960h")
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
