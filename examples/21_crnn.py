#!/usr/bin/env python3
"""CRNN (CNN + RNN) classifier head.

2D CNN frontend extracts local patterns, GRU processes the temporal
sequence. A production-proven hybrid architecture for KWS.
"""
import torch
from ww_trainer.feats import MfccExtractor
from ww_trainer.model import CRNNHead, BaseWakeModel


def main() -> None:
    extractor = MfccExtractor(n_mfcc=40)

    head = CRNNHead(input_size=40, conv_channels=32, gru_hidden=64,
                    gru_layers=1, device="cpu")
    model = BaseWakeModel(extractor, head, device="cpu")

    params = sum(p.numel() for p in head.parameters())
    print(f"CRNN (32ch, GRU-64): {params:,} head params")

    with torch.no_grad():
        prob = torch.sigmoid(model(torch.randn(1, 16000))).item()
        emb = model.embed(torch.randn(1, 16000))
    print(f"Probability: {prob:.4f}")
    print(f"Embedding: {emb.shape}")


if __name__ == "__main__":
    main()
