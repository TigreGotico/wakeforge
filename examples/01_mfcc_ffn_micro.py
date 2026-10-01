#!/usr/bin/env python3
"""Micro tier: MFCC extractor + FFN classifier head.

The smallest architecture (~50K params), suitable for MCUs and RPi Zero.
Demonstrates the fundamental ww-trainer workflow:
  1. Build extractor + head
  2. Combine into BaseWakeModel
  3. Forward pass on synthetic audio
  4. Export both components to ONNX
"""
import tempfile
from pathlib import Path

import torch

from ww_trainer.feats import MfccExtractor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. Build components ---
    extractor = MfccExtractor(sr=16000, n_mfcc=40)
    head = FfnClassifierHead(input_size=extractor.feature_dim,
                             hidden_dim=128, dropout=0.2, device=device)
    model = BaseWakeModel(extractor, head, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Micro model: {total_params:,} parameters on {device}")

    # --- 2. Forward pass on synthetic 1-second audio ---
    audio = torch.randn(1, 16000, device=device)  # 1 second at 16 kHz
    with torch.no_grad():
        logit = model(audio)
        prob = torch.sigmoid(logit).item()
    print(f"Wake word probability: {prob:.4f}")

    # --- 3. Embedding extraction ---
    with torch.no_grad():
        embedding = model.embed(audio)
    print(f"Embedding shape: {embedding.shape}")  # [1, hidden_dim]

    # --- 4. Export to ONNX ---
    with tempfile.TemporaryDirectory() as tmpdir:
        ext_path = str(Path(tmpdir) / "mfcc_extractor.onnx")
        head_path = str(Path(tmpdir) / "ffn_head.onnx")

        extractor.export_to_onnx(ext_path)
        head.export_to_onnx(head_path)

        ext_size = Path(ext_path).stat().st_size / 1024
        head_size = Path(head_path).stat().st_size / 1024
        print(f"ONNX sizes: extractor={ext_size:.1f} KB, head={head_size:.1f} KB")

    print("Done. This model runs on MCUs and RPi Zero.")


if __name__ == "__main__":
    main()
