#!/usr/bin/env python3
"""Complete end-to-end pipeline: build, train, export ONNX, inference.

Demonstrates the full lifecycle in one script:
  1. Build model (MfccExtractor + GRU head)
  2. Train on synthetic data (mini loop with LossManager)
  3. Export extractor and head to ONNX
  4. Run inference with OnnxWakeWordInferencer (no PyTorch needed)
  5. Streaming inference with rolling feature cache
"""
import tempfile
from pathlib import Path

import numpy as np
import torch

from ww_trainer.feats import MfccExtractor
from ww_trainer.loss import LossManager
from ww_trainer.model import GruClassifierHead, BaseWakeModel


def main() -> None:
    tmpdir = tempfile.mkdtemp()

    # ============================
    # STEP 1: Build model
    # ============================
    print("=== Step 1: Build model ===")
    ext = MfccExtractor(sr=16000, n_mfcc=40)
    head = GruClassifierHead(input_size=40, hidden_dim=128, device="cpu")
    model = BaseWakeModel(ext, head, device="cpu")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: MFCC-40 + GRU-128, {n_params:,} params")

    # ============================
    # STEP 2: Train (synthetic)
    # ============================
    print("\n=== Step 2: Train (5 steps on synthetic data) ===")
    losses_cfg = [
        {"name": "bce", "weight": 1.0},
        {"name": "focal", "weight": 0.3, "alpha": 0.25, "gamma": 2.0},
    ]
    lm = LossManager(losses_cfg, device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for step in range(5):
        wavs = torch.randn(8, 16000)
        labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)
        optimizer.zero_grad()
        loss, breakdown = lm.compute_loss(model, wavs, labels)
        loss.backward()
        optimizer.step()
        print(f"  Step {step}: loss={breakdown['total']:.4f}")

    # ============================
    # STEP 3: Export to ONNX
    # ============================
    print("\n=== Step 3: Export to ONNX ===")
    ext_path = str(Path(tmpdir) / "extractor.onnx")
    head_path = str(Path(tmpdir) / "head.onnx")

    model.eval()
    ext.export_to_onnx(ext_path)
    head.export_to_onnx(head_path)
    print(f"Exported to: {tmpdir}/")

    # ============================
    # STEP 4: ONNX inference
    # ============================
    print("\n=== Step 4: ONNX inference (no PyTorch needed) ===")
    from ww_trainer.inference import OnnxWakeWordInferencer

    inferencer = OnnxWakeWordInferencer(
        extractor_path=ext_path,
        head_path=head_path,
        sample_rate=16000,
        device="cpu",
    )

    # Single inference
    audio = np.random.randn(16000).astype(np.float32)
    prob = inferencer.infer(audio)
    print(f"Single inference: prob={prob:.4f}")

    # Batch inference
    batch = np.random.randn(4, 16000).astype(np.float32)
    probs = inferencer.infer_batch(batch)
    print(f"Batch inference: probs={[f'{p:.4f}' for p in probs]}")

    # ============================
    # STEP 5: Streaming inference
    # ============================
    print("\n=== Step 5: Streaming inference ===")
    chunk_size = 1600  # 100ms chunks
    cache = None
    for i in range(10):
        chunk = np.random.randn(chunk_size).astype(np.float32)
        prob, cache = inferencer.infer_streaming(chunk, cache)
        print(f"  Chunk {i}: prob={prob:.4f}, cache_frames={cache.shape[0]}")

    print(f"\nPipeline complete. ONNX files in: {tmpdir}")


if __name__ == "__main__":
    main()
