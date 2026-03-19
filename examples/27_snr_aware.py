#!/usr/bin/env python3
"""SNR-aware features: per-frame signal-to-noise ratio estimation.

SNRAwareExtractor wraps any base extractor and appends 2 per-frame
signals: normalized SNR estimate and noise floor level. Lets the
classifier weight clean frames more heavily.
"""
import torch
from ww_trainer.feats import MfccExtractor, SNRAwareExtractor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel


def main() -> None:
    base = MfccExtractor(n_mfcc=13)
    ext = SNRAwareExtractor(base, noise_percentile=10.0)
    print(f"Base: {base.feature_dim} dims → With SNR: {ext.feature_dim} dims")

    # Compare clean vs noisy
    t = torch.linspace(0, 1, 16000)
    clean = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0) * 0.5
    noisy = clean + torch.randn(1, 16000) * 2.0

    with torch.no_grad():
        out_clean = ext(clean)
        out_noisy = ext(noisy)
        snr_clean = out_clean[:, :, 13].std().item()
        snr_noisy = out_noisy[:, :, 13].std().item()

    print(f"Clean SNR std: {snr_clean:.4f}")
    print(f"Noisy SNR std: {snr_noisy:.4f}")

    # Full stack: MFCC + SNR + VAD + Pitch
    from ww_trainer.feats import VoiceActivityExtractor, PitchExtractor
    full = SNRAwareExtractor(PitchExtractor(VoiceActivityExtractor(base)))
    print(f"\nFull enrichment stack: {full.feature_dim} dims")
    print(f"  MFCC: 13 + VAD: 4 + Pitch: 3 + SNR: 2 = {full.feature_dim}")

    head = FfnClassifierHead(input_size=full.feature_dim, hidden_dim=64, device="cpu")
    model = BaseWakeModel(full, head, device="cpu")
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.randn(1, 16000))).item()
    print(f"Fully enriched model prob: {prob:.4f}")


if __name__ == "__main__":
    main()
