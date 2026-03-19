#!/usr/bin/env python3
"""Classical extractors: PLP, PNCC, and CQT.

- PLP (Hermansky 1990): Bark-scale + equal-loudness + cube-root compression
- PNCC (Kim & Stern 2016): noise-robust via power normalization
- CQT: constant-Q (logarithmic frequency) transform
"""
import torch
from ww_trainer.feats import PLPExtractor, PNCCExtractor, CQTExtractor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel


def main() -> None:
    audio = torch.randn(1, 16000)

    for name, ext in [
        ("PLP-13", PLPExtractor(n_plp=13)),
        ("PNCC-13", PNCCExtractor(n_pncc=13)),
        ("CQT-84", CQTExtractor(n_bins=12, n_octaves=7)),
    ]:
        head = FfnClassifierHead(input_size=ext.feature_dim, hidden_dim=64, device="cpu")
        model = BaseWakeModel(ext, head, device="cpu")
        params = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            feats = ext(audio)
            prob = torch.sigmoid(model(audio)).item()
        print(f"{name}: dim={ext.feature_dim}, shape={feats.shape}, "
              f"params={params:,}, prob={prob:.4f}")


if __name__ == "__main__":
    main()
