#!/usr/bin/env python3
"""Voice Activity Detection enriched features.

VoiceActivityExtractor wraps any base extractor and appends 4 per-frame
VAD signals: RMS energy, zero-crossing rate, spectral flatness, and a
combined voice activity probability. This lets the classifier focus on
speech regions without learning VAD from scratch.
"""
import torch
from ww_trainer.feats import MfccExtractor, VoiceActivityExtractor
from ww_trainer.model import GruClassifierHead, BaseWakeModel


def main() -> None:
    # Base: MFCC-13 → 13 dims
    base = MfccExtractor(n_mfcc=13)

    # With VAD: 13 + 4 = 17 dims
    ext = VoiceActivityExtractor(base)
    print(f"Base feature_dim: {base.feature_dim}")
    print(f"VAD-enriched feature_dim: {ext.feature_dim}")

    head = GruClassifierHead(input_size=ext.feature_dim, hidden_dim=64, device="cpu")
    model = BaseWakeModel(ext, head, device="cpu")

    # Compare silence vs speech-like signal
    silence = torch.zeros(1, 16000) + 1e-4
    t = torch.linspace(0, 1, 16000)
    speech = torch.sin(2 * torch.pi * 200 * t).unsqueeze(0) * 0.5

    with torch.no_grad():
        out_silence = ext(silence)
        out_speech = ext(speech)
        vad_silence = out_silence[:, :, -1].mean().item()  # vad_prob
        vad_speech = out_speech[:, :, -1].mean().item()

    print(f"\nVAD probability — silence: {vad_silence:.4f}, speech: {vad_speech:.4f}")
    print(f"VAD correctly distinguishes speech from silence: {vad_speech > vad_silence}")


if __name__ == "__main__":
    main()
