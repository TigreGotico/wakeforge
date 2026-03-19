#!/usr/bin/env python3
"""Gammatone extractor + FFN classifier head.

The gammatone filterbank models the human auditory system using
filters spaced on the ERB (Equivalent Rectangular Bandwidth) scale.
More robust to noise than mel-based features for certain applications.
"""
import torch

from ww_trainer.feats import GammatoneExtractor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = GammatoneExtractor(
        sr=16000,
        n_filters=64,
        f_min=50.0,
        f_max=8000.0,
        frame_len=400,     # 25ms window
        hop_length=160,    # 10ms hop
        order=4,           # 4th-order gammatone (standard)
    )
    head = FfnClassifierHead(
        input_size=extractor.feature_dim,  # 64
        hidden_dim=128,
        device=device,
    )
    model = BaseWakeModel(extractor, head, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Gammatone-64 + FFN: {total_params:,} parameters")

    # --- Forward pass ---
    audio = torch.randn(1, 16000, device=device)
    with torch.no_grad():
        feats = extractor(audio)
        prob = torch.sigmoid(model(audio)).item()
    print(f"Feature shape: {feats.shape}")
    print(f"Probability: {prob:.4f}")


if __name__ == "__main__":
    main()
