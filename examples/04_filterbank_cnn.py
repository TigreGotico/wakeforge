#!/usr/bin/env python3
"""FilterBank extractor + CNN classifier head.

FilterbankExtractor produces log-mel spectrograms without the DCT step
of MFCC, preserving more spectral detail. The CNN head applies 1D
convolutions along the time axis for pattern detection.
"""
import torch

from ww_trainer.feats import FilterbankExtractor
from ww_trainer.model import CnnClassifierHead, BaseWakeModel


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = FilterbankExtractor(
        sr=16000,
        n_mels=80,       # 80 mel bins — richer than 40 MFCC
        n_fft=400,
        hop_length=160,   # 10ms hop → 100 frames/sec
    )
    head = CnnClassifierHead(
        input_size=extractor.feature_dim,  # 80
        conv_dim=256,
        linear_dim=128,
        kernel_size=3,
        device=device,
    )
    model = BaseWakeModel(extractor, head, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"FilterBank-80 + CNN: {total_params:,} parameters")

    # --- Feature inspection ---
    audio = torch.randn(1, 16000, device=device)
    with torch.no_grad():
        feats = extractor(audio)
    print(f"Feature shape: {feats.shape}")
    # [1, T, 80] where T ≈ 100 (1 sec / 10ms hop)

    # --- Inference ---
    with torch.no_grad():
        logit = model(audio)
        prob = torch.sigmoid(logit).item()
    print(f"Probability: {prob:.4f}")


if __name__ == "__main__":
    main()
