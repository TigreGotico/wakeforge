#!/usr/bin/env python3
"""Delta-wrapped MFCC extractor + FFN head.

DeltaExtractor wraps any base extractor and appends first-order (velocity)
and second-order (acceleration) delta features. This triples the feature
dimension, giving the classifier temporal dynamics information.

MFCC-13 + deltas → 39-dim features (classic ASR configuration).
"""
import torch

from ww_trainer.feats import MfccExtractor, DeltaExtractor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Base MFCC with 13 coefficients ---
    base = MfccExtractor(sr=16000, n_mfcc=13)
    print(f"Base MFCC feature_dim: {base.feature_dim}")  # 13

    # --- Wrap with deltas → 13 * 3 = 39 ---
    extractor = DeltaExtractor(base, delta_width=2)
    print(f"Delta MFCC feature_dim: {extractor.feature_dim}")  # 39

    head = FfnClassifierHead(
        input_size=extractor.feature_dim,
        hidden_dim=128,
        device=device,
    )
    model = BaseWakeModel(extractor, head, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Delta-MFCC-13 + FFN: {total_params:,} parameters")

    # --- Compare features ---
    audio = torch.randn(1, 16000, device=device)
    with torch.no_grad():
        base_feats = base(audio)
        delta_feats = extractor(audio)
    print(f"Base features:  {base_feats.shape}")   # [1, T, 13]
    print(f"Delta features: {delta_feats.shape}")   # [1, T, 39]

    # --- Also works with FilterBank ---
    from ww_trainer.feats import FilterbankExtractor

    fb_base = FilterbankExtractor(sr=16000, n_mels=40)
    fb_delta = DeltaExtractor(fb_base)
    print(f"\nFilterBank-40 + deltas: feature_dim={fb_delta.feature_dim}")  # 120


if __name__ == "__main__":
    main()
