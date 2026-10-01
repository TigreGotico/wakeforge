#!/usr/bin/env python3
"""LEAF: Learnable Audio Frontend (Zeghidour et al., ICLR 2021, Google).

Replaces fixed mel filterbanks with fully learnable components:
- Gabor convolution (learnable center freq + bandwidth)
- Squared modulus (energy)
- Gaussian low-pass pooling (learnable smoothing)
- Per-Channel Energy Normalization (PCEN)

All components train end-to-end with the classifier.
"""
import torch
from ww_trainer.feats import LEAFExtractor
from ww_trainer.model import FfnClassifierHead, GruClassifierHead, BaseWakeModel


def main() -> None:
    # --- LEAF + FFN (simplest) ---
    leaf = LEAFExtractor(sr=16000, n_filters=40)
    head = FfnClassifierHead(input_size=40, hidden_dim=128, device="cpu")
    model = BaseWakeModel(leaf, head, device="cpu")

    ext_params = sum(p.numel() for p in leaf.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LEAF-40 + FFN: {ext_params:,} extractor + {total_params - ext_params:,} head = {total_params:,} total")

    # Inspect learned filter frequencies
    print(f"Center freqs: {leaf.center_hz.data[:5].tolist()} ... {leaf.center_hz.data[-3:].tolist()} Hz")
    print(f"Bandwidths:   {leaf.bandwidth_hz.data[:5].tolist()} ... Hz")

    audio = torch.randn(1, 16000)
    with torch.no_grad():
        feats = leaf(audio)
        prob = torch.sigmoid(model(audio)).item()
    print(f"Feature shape: {feats.shape}")
    print(f"Probability: {prob:.4f}")

    # --- LEAF + GRU (temporal modeling) ---
    gru_head = GruClassifierHead(input_size=40, hidden_dim=64, device="cpu")
    model2 = BaseWakeModel(leaf, gru_head, device="cpu")
    params2 = sum(p.numel() for p in model2.parameters())
    print(f"\nLEAF-40 + GRU: {params2:,} total params")

    # --- Verify gradients flow through LEAF ---
    audio_grad = torch.randn(1, 16000)
    out = model(audio_grad)
    out.sum().backward()
    print(f"LEAF center_hz grad norm: {leaf.center_hz.grad.norm():.6f}")
    print(f"LEAF bandwidth grad norm: {leaf.bandwidth_hz.grad.norm():.6f}")
    print("Gradients flow end-to-end through LEAF!")


if __name__ == "__main__":
    main()
