#!/usr/bin/env python3
"""ONNX-only inference — zero PyTorch dependency at runtime.

This example shows how to:
  1. Export extractor + head to ONNX (requires PyTorch)
  2. Run inference using only numpy + onnxruntime (no PyTorch needed)

The OnnxWakeWordInferencer is designed for edge deployment where
installing PyTorch is impractical.
"""
import tempfile
from pathlib import Path

import numpy as np
import torch

from ww_trainer.feats import MfccExtractor
from ww_trainer.model import FfnClassifierHead


def main() -> None:
    # ========================================================
    # STEP 1: Export (requires PyTorch — done once on dev machine)
    # ========================================================
    extractor = MfccExtractor(sr=16000, n_mfcc=40)
    head = FfnClassifierHead(input_size=40, hidden_dim=128, device="cpu")

    tmpdir = tempfile.mkdtemp()
    ext_path = str(Path(tmpdir) / "extractor.onnx")
    head_path = str(Path(tmpdir) / "head.onnx")

    extractor.eval()
    head.eval()
    extractor.export_to_onnx(ext_path)
    head.export_to_onnx(head_path)
    print(f"Exported to {tmpdir}/")

    # ========================================================
    # STEP 2: Inference (numpy + onnxruntime ONLY — no torch!)
    # ========================================================
    from ww_trainer.inference import OnnxWakeWordInferencer

    inferencer = OnnxWakeWordInferencer(
        extractor_path=ext_path,
        head_path=head_path,
        sample_rate=16000,
        device="cpu",
    )

    # --- Single inference ---
    audio = np.random.randn(16000).astype(np.float32)
    prob = inferencer.infer(audio)
    print(f"Single inference: prob={prob:.4f}")

    # --- Batch inference ---
    batch = np.random.randn(4, 16000).astype(np.float32)
    probs = inferencer.infer_batch(batch)
    print(f"Batch inference: probs={probs}")

    # --- Streaming inference ---
    chunk_size = 1600  # 100ms chunks
    cache = None
    for i in range(10):
        chunk = np.random.randn(chunk_size).astype(np.float32)
        prob, cache = inferencer.infer_streaming(chunk, cache)
        print(f"  Chunk {i}: prob={prob:.4f}, cache_shape={cache.shape}")

    print("\nNo torch import needed for inference!")


if __name__ == "__main__":
    main()
