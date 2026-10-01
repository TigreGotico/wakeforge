#!/usr/bin/env python3
"""Keyword Transformer (KWT) head (Berg et al. 2021).

Vision Transformer adapted for spectrograms. Patches along time,
positional embedding, transformer encoder, CLS token classification.
Highest accuracy but heavier than CNN/RNN approaches.
"""
import torch
from ww_trainer.feats import FilterbankExtractor
from ww_trainer.model import KWTHead, BaseWakeModel


def main() -> None:
    extractor = FilterbankExtractor(n_mels=40)

    # Small KWT
    head = KWTHead(input_size=40, patch_len=5, d_model=64,
                   n_heads=4, n_layers=4, dim_ff=128, device="cpu")
    model = BaseWakeModel(extractor, head, device="cpu")

    params = sum(p.numel() for p in head.parameters())
    print(f"KWT (d=64, 4 layers): {params:,} head params")

    with torch.no_grad():
        prob = torch.sigmoid(model(torch.randn(1, 16000))).item()
        emb = model.embed(torch.randn(1, 16000))
    print(f"Probability: {prob:.4f}")
    print(f"Embedding: {emb.shape}")

    # Larger KWT
    head_large = KWTHead(input_size=40, patch_len=5, d_model=128,
                         n_heads=8, n_layers=6, dim_ff=256, device="cpu")
    params_large = sum(p.numel() for p in head_large.parameters())
    print(f"\nKWT-large (d=128, 6 layers): {params_large:,} head params")


if __name__ == "__main__":
    main()
