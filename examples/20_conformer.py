#!/usr/bin/env python3
"""Conformer classifier head (Gulati et al. 2020).

Convolution-augmented transformer: self-attention + depthwise convolutions.
Uses AttentionPooling instead of mean-pooling for better accuracy.
"""
import torch
from ww_trainer.feats import FilterbankExtractor
from ww_trainer.model import ConformerHead, BaseWakeModel


def main() -> None:
    extractor = FilterbankExtractor(n_mels=80)

    head = ConformerHead(input_size=80, d_model=64, n_heads=4,
                         n_layers=4, conv_kernel=15, dim_ff=128, device="cpu")
    model = BaseWakeModel(extractor, head, device="cpu")

    params = sum(p.numel() for p in head.parameters())
    print(f"Conformer (d=64, 4 layers): {params:,} head params")

    with torch.no_grad():
        prob = torch.sigmoid(model(torch.randn(1, 16000))).item()
        emb = model.embed(torch.randn(1, 16000))
    print(f"Probability: {prob:.4f}")
    print(f"Embedding: {emb.shape}")


if __name__ == "__main__":
    main()
