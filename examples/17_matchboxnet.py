#!/usr/bin/env python3
"""MatchboxNet classifier head (Majumdar & Ginsburg, NVIDIA 2020).

1D time-channel separable convolutions. Architecture: MatchboxNet-B×R×C.
"""
import torch
from ww_trainer.feats import FilterbankExtractor
from ww_trainer.model import MatchboxNetHead, BaseWakeModel


def main() -> None:
    extractor = FilterbankExtractor(n_mels=40)

    configs = [(3, 1, 64, "3x1x64"), (3, 2, 64, "3x2x64"), (6, 2, 64, "6x2x64")]
    for B, R, C, name in configs:
        head = MatchboxNetHead(input_size=40, B=B, R=R, C=C, device="cpu")
        params = sum(p.numel() for p in head.parameters())
        model = BaseWakeModel(extractor, head, device="cpu")
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.randn(1, 16000))).item()
        print(f"MatchboxNet-{name}: {params:,} head params, prob={prob:.4f}")


if __name__ == "__main__":
    main()
