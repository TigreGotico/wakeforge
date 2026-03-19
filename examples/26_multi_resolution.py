#!/usr/bin/env python3
"""Multi-resolution features: combine two extractors at different time scales.

Fine extractor (small hop) captures details; coarse extractor (large hop)
captures broad temporal patterns. Features are concatenated per frame.
"""
import torch
from ww_trainer.feats import MfccExtractor, FilterbankExtractor, MultiResolutionExtractor
from ww_trainer.model import GruClassifierHead, BaseWakeModel


def main() -> None:
    # Fine: MFCC at 5ms hop → 200 frames/sec
    fine = MfccExtractor(n_mfcc=13, hop_length=80)
    # Coarse: FilterBank at 20ms hop → 50 frames/sec
    coarse = FilterbankExtractor(n_mels=40, hop_length=320)

    ext = MultiResolutionExtractor(fine, coarse)
    print(f"Fine: {fine.feature_dim}d @ 5ms hop")
    print(f"Coarse: {coarse.feature_dim}d @ 20ms hop")
    print(f"Combined: {ext.feature_dim}d (coarse interpolated to fine resolution)")

    audio = torch.randn(1, 16000)
    with torch.no_grad():
        feats = ext(audio)
    print(f"Output shape: {feats.shape}")  # [1, ~200, 53]

    head = GruClassifierHead(input_size=ext.feature_dim, hidden_dim=64, device="cpu")
    model = BaseWakeModel(ext, head, device="cpu")
    params = sum(p.numel() for p in model.parameters())
    print(f"MultiRes + GRU: {params:,} params")


if __name__ == "__main__":
    main()
