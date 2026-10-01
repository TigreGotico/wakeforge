#!/usr/bin/env python3
"""Streaming wake word detection with rolling feature cache.

Demonstrates real-time wake word detection by processing audio in small
chunks (e.g. 100ms). Uses SlidingFeatureCacheTensor to maintain a rolling
window of features for the classifier.

Two paths shown:
  1. PyTorch streaming (BaseWakeModel.forward_streaming)
  2. ONNX streaming (OnnxWakeWordInferencer.infer_streaming)
"""
import tempfile
from pathlib import Path

import numpy as np
import torch

from ww_trainer.feats import MfccExtractor, SlidingFeatureCacheTensor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel


def main() -> None:
    device = "cpu"

    # --- Build model ---
    extractor = MfccExtractor(sr=16000, n_mfcc=40)
    head = FfnClassifierHead(input_size=40, hidden_dim=128, device=device)
    model = BaseWakeModel(extractor, head, device=device)
    model.eval()

    # ========================================================
    # PATH 1: PyTorch streaming
    # ========================================================
    print("=== PyTorch Streaming ===")
    cache = SlidingFeatureCacheTensor(
        feature_dim=extractor.feature_dim,
        window_size=50,  # ~500ms of context at 10ms hop
    )

    chunk_samples = 1600  # 100ms at 16 kHz
    for i in range(10):
        chunk = torch.randn(chunk_samples)
        with torch.no_grad():
            prob = model.forward_streaming(chunk, cache)
        print(f"  Chunk {i}: prob={prob:.4f}")

    # ========================================================
    # PATH 2: ONNX streaming (no torch at inference time)
    # ========================================================
    print("\n=== ONNX Streaming ===")
    tmpdir = tempfile.mkdtemp()
    ext_path = str(Path(tmpdir) / "ext.onnx")
    head_path = str(Path(tmpdir) / "head.onnx")
    extractor.export_to_onnx(ext_path)
    head.export_to_onnx(head_path)

    from ww_trainer.inference import OnnxWakeWordInferencer

    onnx_inf = OnnxWakeWordInferencer(ext_path, head_path, device="cpu")
    cache_np = None
    for i in range(10):
        chunk = np.random.randn(chunk_samples).astype(np.float32)
        prob, cache_np = onnx_inf.infer_streaming(chunk, cache_np)
        print(f"  Chunk {i}: prob={prob:.4f}, cache={cache_np.shape}")


if __name__ == "__main__":
    main()
