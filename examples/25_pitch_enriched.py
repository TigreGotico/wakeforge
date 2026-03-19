#!/usr/bin/env python3
"""Pitch-enriched features via autocorrelation-based F0 estimation.

PitchExtractor wraps any base extractor and appends 3 per-frame signals:
normalized F0, voicing probability, and F0 delta (pitch dynamics).
"""
import torch
from ww_trainer.feats import MfccExtractor, PitchExtractor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel


def main() -> None:
    base = MfccExtractor(n_mfcc=13)
    ext = PitchExtractor(base, f0_min=50, f0_max=600)
    print(f"Base: {base.feature_dim} dims → With pitch: {ext.feature_dim} dims")

    # Pure tone at 200 Hz
    t = torch.linspace(0, 1, 16000)
    tone_200 = torch.sin(2 * torch.pi * 200 * t).unsqueeze(0) * 0.5
    tone_400 = torch.sin(2 * torch.pi * 400 * t).unsqueeze(0) * 0.5

    with torch.no_grad():
        out_200 = ext(tone_200)
        out_400 = ext(tone_400)
        f0_200 = out_200[:, :, 13].mean().item()   # normalized F0
        f0_400 = out_400[:, :, 13].mean().item()
        voicing_200 = out_200[:, :, 14].mean().item()  # voicing prob

    print(f"200 Hz tone — F0 (norm): {f0_200:.3f}, voicing: {voicing_200:.3f}")
    print(f"400 Hz tone — F0 (norm): {f0_400:.3f}")
    print(f"Higher pitch → higher normalized F0: {f0_400 > f0_200}")

    # Use in a model
    head = FfnClassifierHead(input_size=ext.feature_dim, hidden_dim=64, device="cpu")
    model = BaseWakeModel(ext, head, device="cpu")
    params = sum(p.numel() for p in model.parameters())
    print(f"\nMFCC+Pitch + FFN: {params:,} params")


if __name__ == "__main__":
    main()
