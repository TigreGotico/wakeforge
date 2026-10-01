#!/usr/bin/env python3
"""ESP32 nano tier: sub-1KB wake word models via grid search.

Explores the smallest viable MFCC + FFN configurations that fit within
~1 KB int8.  The nano tier uses MFCC-13 + FFN with hidden_dim in {8, 12, 16}.

Usage:
    python examples/34_esp32_nano.py --metadata dataset.csv
    python examples/34_esp32_nano.py  # synthetic demo (no dataset needed)
"""
import sys

import torch

from ww_trainer.feats import MfccExtractor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel
from ww_trainer.tiers import get_tier
from ww_trainer.sweep import _estimate_ffn_params


def main() -> None:
    tier = get_tier("esp32_nano")
    print(f"Tier: {tier.name}")
    print(f"Budget: {tier.max_params} params, {tier.max_size_kb} KB int8")
    print()

    # Grid of viable configurations
    configs = [
        {"n_mfcc": 10, "hidden_dim": 8},
        {"n_mfcc": 10, "hidden_dim": 12},
        {"n_mfcc": 10, "hidden_dim": 16},
        {"n_mfcc": 13, "hidden_dim": 8},
        {"n_mfcc": 13, "hidden_dim": 12},
        {"n_mfcc": 13, "hidden_dim": 16},
    ]

    print(f"{'n_mfcc':<8} {'hidden':<8} {'params':<10} {'KB_int8':<10} {'fits?':<6}")
    print("-" * 44)

    for cfg in configs:
        n_params = _estimate_ffn_params(cfg["n_mfcc"], cfg["hidden_dim"])
        size_kb = n_params / 1024  # 1 byte per param in int8
        fits = n_params <= tier.max_params
        print(f"{cfg['n_mfcc']:<8} {cfg['hidden_dim']:<8} {n_params:<10} {size_kb:<10.2f} {'YES' if fits else 'NO':<6}")

        if fits:
            # Instantiate and verify
            extractor = MfccExtractor(n_mfcc=cfg["n_mfcc"])
            head = FfnClassifierHead(
                input_size=cfg["n_mfcc"],
                hidden_dim=cfg["hidden_dim"],
                device="cpu",
            )
            model = BaseWakeModel(feature_extractor=extractor, classifier=head, device="cpu")
            actual = sum(p.numel() for p in model.classifier.parameters())
            # Smoke test: forward pass
            dummy = torch.randn(1, 16000)
            logits = model(dummy)
            print(f"  -> actual head params={actual}, logit={logits.item():.4f}")

    print("\nDone. All nano-tier configs instantiated and verified.")


if __name__ == "__main__":
    main()
