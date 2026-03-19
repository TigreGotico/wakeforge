#!/usr/bin/env python3
"""Markov-based feature extractors: temporal modeling without neural networks.

Two novel extractors that use classical Markov models on quantized audio:

1. MarkovTransitionExtractor: Appends per-frame transition probability
   vectors from a trained Markov chain. Captures "what usually follows
   this audio pattern" as features.

2. HMMStateExtractor: Appends Hidden Markov Model state posteriors
   (forward algorithm). Models phoneme-like temporal structure.

Both are trainable on wake word audio and stackable with other wrappers.
"""
import torch

from ww_trainer.feats import (
    MfccExtractor,
    MarkovTransitionExtractor,
    HMMStateExtractor,
    VoiceActivityExtractor,
)
from ww_trainer.model import FfnClassifierHead, BaseWakeModel


def main() -> None:
    base = MfccExtractor(n_mfcc=13)

    # ============================================================
    # 1. Markov Transition Extractor
    # ============================================================
    print("=== MarkovTransitionExtractor ===")
    markov = MarkovTransitionExtractor(base, n_codes=16, order=2)
    print(f"Before training: feature_dim={markov.feature_dim} (13 base + 16 transition probs)")

    # Train on synthetic "wake word" audio
    wake_audio = [torch.randn(16000) * 0.5 for _ in range(10)]
    markov.fit(wake_audio)
    print(f"Trained on {len(wake_audio)} samples")
    print(f"Codebook shape: {markov._codebook.shape}")
    print(f"Transition matrix: {markov._transition_matrix.shape}")

    # Forward pass
    out = markov(torch.randn(1, 16000))
    print(f"Output shape: {out.shape}")
    # Last 16 dims are transition probabilities
    trans_probs = out[0, 50, 13:]
    print(f"Frame 50 transition probs sum: {trans_probs.sum():.4f} (should be ~1.0)")

    # ============================================================
    # 2. HMM State Extractor
    # ============================================================
    print("\n=== HMMStateExtractor ===")
    hmm_ext = HMMStateExtractor(base, n_states=4, n_codes=16)
    print(f"feature_dim={hmm_ext.feature_dim} (13 base + 4 state posteriors)")

    # Train with markovonnx
    try:
        hmm_ext.fit(wake_audio, n_iter=3)
        print("HMM trained via Baum-Welch")
    except ImportError:
        print("markovonnx not installed — using uniform priors")

    out = hmm_ext(torch.randn(1, 16000))
    posteriors = out[0, 50, 13:]
    print(f"Frame 50 state posteriors: {posteriors.tolist()}")
    print(f"Posteriors sum: {posteriors.sum():.4f} (should be ~1.0)")

    # ============================================================
    # 3. Stack: MFCC + VAD + Markov
    # ============================================================
    print("\n=== Stacked: MFCC + VAD + Markov ===")
    vad = VoiceActivityExtractor(base)          # 13 + 4 = 17
    full = MarkovTransitionExtractor(vad, n_codes=8, order=1)  # 17 + 8 = 25
    full.fit(wake_audio)
    print(f"Full stack feature_dim: {full.feature_dim}")

    head = FfnClassifierHead(input_size=full.feature_dim, hidden_dim=32, device="cpu")
    model = BaseWakeModel(full, head, device="cpu")
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}")

    with torch.no_grad():
        prob = torch.sigmoid(model(torch.randn(1, 16000))).item()
    print(f"Wake probability: {prob:.4f}")


if __name__ == "__main__":
    main()
