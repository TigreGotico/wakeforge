#!/usr/bin/env python3
"""DS-CNN classifier head (Zhang et al., "Hello Edge", 2017).

The ARM/Google benchmark standard for microcontroller KWS.
Depthwise-separable 2D convolutions. Sizes: S (~20K), M (~80K), L (~250K).
"""
import torch
from ww_trainer.feats import MfccExtractor
from ww_trainer.model import DSCNNHead, BaseWakeModel


def main() -> None:
    extractor = MfccExtractor(n_mfcc=40)

    for size in ["S", "M", "L"]:
        head = DSCNNHead(input_size=40, size=size, device="cpu")
        model = BaseWakeModel(extractor, head, device="cpu")
        params = sum(p.numel() for p in head.parameters())
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.randn(1, 16000))).item()
        print(f"DS-CNN-{size}: {params:,} head params, prob={prob:.4f}")


if __name__ == "__main__":
    main()
