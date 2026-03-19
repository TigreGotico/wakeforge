#!/usr/bin/env python3
"""SincNet extractor + GRU classifier head.

SincNet uses learnable sinc-based bandpass filters instead of fixed mel
filterbanks. The filter cutoff frequencies are learned during training,
letting the model discover task-optimal frequency bands.

Good for: noisy environments, non-speech wake words, multi-language.
"""
import torch

from ww_trainer.feats import SincNetExtractor
from ww_trainer.model import GruClassifierHead, BaseWakeModel


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = SincNetExtractor(
        sr=16000,
        n_filters=80,       # 80 learnable bandpass filters
        kernel_size=251,     # ~16ms window at 16 kHz
        stride=160,          # 10ms hop
        min_freq=50.0,
        min_band=50.0,
    )
    head = GruClassifierHead(
        input_size=extractor.feature_dim,  # 80
        hidden_dim=128,
        gru_n_layers=1,
        device=device,
    )
    model = BaseWakeModel(extractor, head, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"SincNet-80 + GRU: {total_params:,} parameters")

    # --- Inspect learned filter frequencies ---
    low_hz = extractor.low_hz_.data.cpu()
    band_hz = extractor.band_hz_.data.cpu()
    print(f"Filter freq range: {low_hz.min():.0f} Hz – {(low_hz + band_hz).max():.0f} Hz")

    # --- Forward pass ---
    audio = torch.randn(2, 16000, device=device)
    with torch.no_grad():
        feats = extractor(audio)
        probs = torch.sigmoid(model(audio))
    print(f"Feature shape: {feats.shape}")
    print(f"Probabilities: {probs.tolist()}")


if __name__ == "__main__":
    main()
