#!/usr/bin/env python3
"""Res15 classifier head (Tang & Lin, Interspeech 2018).

1D dilated residual network with exponentially increasing dilation.
6 residual blocks with dilations [1, 2, 4, 8, 16, 32].
"""
import torch
from ww_trainer.feats import MfccExtractor
from ww_trainer.model import Res15Head, BaseWakeModel


def main() -> None:
    extractor = MfccExtractor(n_mfcc=40)
    head = Res15Head(input_size=40, channels=45, device="cpu")
    model = BaseWakeModel(extractor, head, device="cpu")

    params = sum(p.numel() for p in head.parameters())
    print(f"Res15 (45 channels): {params:,} head params")

    with torch.no_grad():
        prob = torch.sigmoid(model(torch.randn(1, 16000))).item()
        emb = model.embed(torch.randn(1, 16000))
    print(f"Probability: {prob:.4f}")
    print(f"Embedding: {emb.shape}")


if __name__ == "__main__":
    main()
