#!/usr/bin/env python3
"""TC-ResNet classifier head (Choi et al., Interspeech 2019).

Purely 1D temporal convolutions with residual connections.
Two variants: TC-ResNet8 (3 blocks) and TC-ResNet14 (6 blocks).
"""
import torch
from ww_trainer.feats import MfccExtractor
from ww_trainer.model import TCResNetHead, BaseWakeModel


def main() -> None:
    extractor = MfccExtractor(n_mfcc=40)

    for variant in [8, 14]:
        head = TCResNetHead(input_size=40, variant=variant, channels=64, device="cpu")
        model = BaseWakeModel(extractor, head, device="cpu")
        params = sum(p.numel() for p in head.parameters())
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.randn(1, 16000))).item()
        print(f"TC-ResNet{variant}: {params:,} head params, prob={prob:.4f}")


if __name__ == "__main__":
    main()
