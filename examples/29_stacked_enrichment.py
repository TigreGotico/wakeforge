#!/usr/bin/env python3
"""Stacked feature enrichment: MFCC + Delta + VAD + Pitch + SNR.

Demonstrates stacking ALL enrichment wrappers to build a 22+ dimension
feature vector from 13 base MFCC coefficients. Compares a plain MFCC
model against the fully enriched version.

Feature breakdown:
  MFCC base:   13 dims
  Delta wrap:   13*3 = 39 dims (static + delta + delta-delta)
  VAD wrap:     +4 dims (energy, ZCR, spectral flatness, VAD prob)
  Pitch wrap:   +3 dims (F0 norm, voicing prob, F0 delta)
  SNR wrap:     +2 dims (SNR estimate, noise floor)
  Total:        48 dims (or 22 dims without Delta)
"""
import torch

from ww_trainer.feats import (
    MfccExtractor,
    DeltaExtractor,
    VoiceActivityExtractor,
    PitchExtractor,
    SNRAwareExtractor,
)
from ww_trainer.model import GruClassifierHead, BaseWakeModel


def main() -> None:
    # Base: MFCC-13
    base = MfccExtractor(n_mfcc=13)
    print(f"Base MFCC: {base.feature_dim} dims")

    # Stack enrichment wrappers (without delta)
    vad = VoiceActivityExtractor(base)
    print(f"  + VAD:   {vad.feature_dim} dims (+4)")

    pitch = PitchExtractor(vad)
    print(f"  + Pitch: {pitch.feature_dim} dims (+3)")

    snr = SNRAwareExtractor(pitch)
    print(f"  + SNR:   {snr.feature_dim} dims (+2)")
    print(f"  Total without delta: {snr.feature_dim} dims")

    # Full stack with delta (wraps base first, then enrichments on top)
    delta = DeltaExtractor(base)
    print(f"\nWith Delta: {delta.feature_dim} dims (13*3)")

    full = SNRAwareExtractor(PitchExtractor(VoiceActivityExtractor(delta)))
    print(f"Full stack (Delta+VAD+Pitch+SNR): {full.feature_dim} dims")

    # Build models for comparison
    audio = torch.randn(2, 16000)  # batch of 2 synthetic waveforms

    # Plain model
    plain_head = GruClassifierHead(input_size=base.feature_dim, hidden_dim=64, device="cpu")
    plain_model = BaseWakeModel(base, plain_head, device="cpu")

    # Enriched model (without delta for speed)
    enriched_head = GruClassifierHead(input_size=snr.feature_dim, hidden_dim=64, device="cpu")
    enriched_model = BaseWakeModel(snr, enriched_head, device="cpu")

    # Run inference
    with torch.no_grad():
        plain_feats = base(audio)
        enriched_feats = snr(audio)
        print(f"\nPlain features shape:    {plain_feats.shape}")
        print(f"Enriched features shape: {enriched_feats.shape}")

        plain_prob = torch.sigmoid(plain_model(audio))
        enriched_prob = torch.sigmoid(enriched_model(audio))
        print(f"\nPlain model probs:    {plain_prob.tolist()}")
        print(f"Enriched model probs: {enriched_prob.tolist()}")

    # Show feature channel breakdown
    print(f"\nFeature channel map (enriched, no delta):")
    print(f"  [0:13]  MFCC coefficients")
    print(f"  [13:17] VAD: energy, ZCR, spectral flatness, VAD prob")
    print(f"  [17:20] Pitch: F0 norm, voicing prob, F0 delta")
    print(f"  [20:22] SNR: SNR estimate, noise floor")


if __name__ == "__main__":
    main()
