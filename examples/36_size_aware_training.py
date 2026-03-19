#!/usr/bin/env python3
"""SizeAwareLoss demo: train a tiny model with sparsity + size penalties.

Shows how SizeAwareLoss wraps BCE and adds L1 sparsity pressure plus a
parameter-count penalty to encourage smaller weights during training.

Usage:
    python examples/36_size_aware_training.py
"""
import torch
import torch.nn as nn

from ww_trainer.feats import MfccExtractor
from ww_trainer.loss import SizeAwareLoss
from ww_trainer.model import FfnClassifierHead, BaseWakeModel
from ww_trainer.tiers import get_tier


def main() -> None:
    tier = get_tier("esp32_nano")
    print(f"Tier: {tier.name} | Budget: {tier.max_params} params\n")

    # Build a tiny model
    extractor = MfccExtractor(n_mfcc=13)
    head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")
    model = BaseWakeModel(feature_extractor=extractor, classifier=head, device="cpu")

    n_params = sum(p.numel() for p in model.classifier.parameters())
    print(f"Model params: {n_params} (budget: {tier.max_params})")

    # Setup SizeAwareLoss
    base_loss = nn.BCEWithLogitsLoss()
    loss_fn = SizeAwareLoss(
        base_loss=base_loss,
        l1_weight=1e-5,
        size_weight=0.1,
        param_budget=tier.max_params,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Synthetic training loop
    print("\nTraining with SizeAwareLoss (5 epochs, synthetic data):")
    for epoch in range(5):
        # Synthetic batch: 8 samples of 1-second audio
        wavs = torch.randn(8, 16000)
        labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)

        optimizer.zero_grad()
        logits = model(wavs)
        loss = loss_fn(logits, labels, model.classifier)
        loss.backward()
        optimizer.step()

        # Report sparsity: fraction of near-zero weights
        with torch.no_grad():
            all_params = torch.cat([p.flatten() for p in model.classifier.parameters()])
            sparsity = (all_params.abs() < 1e-3).float().mean().item()

        print(f"  Epoch {epoch + 1}: loss={loss.item():.4f}, sparsity={sparsity:.2%}")

    print("\nDone. SizeAwareLoss encourages sparse, compact models.")


if __name__ == "__main__":
    main()
