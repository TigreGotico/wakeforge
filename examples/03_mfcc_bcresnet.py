#!/usr/bin/env python3
"""MFCC extractor + BC-ResNet classifier head.

BC-ResNet (Kim et al., Interspeech 2021) uses broadcasted residual learning
with 2D convolutions on mel-spectrograms. The tau parameter controls width:

  tau=1   → ~6K params   (ultra-tiny)
  tau=2   → ~20K params  (tiny)
  tau=8   → ~280K params (full)

Pair with MfccExtractor or FilterbankExtractor for mel-spectrogram features.
"""
import tempfile
from pathlib import Path

import torch

from ww_trainer.feats import MfccExtractor, FilterbankExtractor
from ww_trainer.model import BCResNetHead, BaseWakeModel


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Compare tau values ---
    print("BC-ResNet scaling (with MFCC-40 extractor):")
    print(f"{'tau':<6} {'head params':>12} {'total params':>14}")
    print("-" * 34)

    extractor = MfccExtractor(sr=16000, n_mfcc=40)
    ext_params = sum(p.numel() for p in extractor.parameters())

    for tau in [1, 1.5, 2, 3, 6, 8]:
        head = BCResNetHead(input_size=40, tau=tau, device=device)
        head_params = sum(p.numel() for p in head.parameters())
        print(f"{tau:<6} {head_params:>12,} {ext_params + head_params:>14,}")

    # --- Full model with tau=2 ---
    head = BCResNetHead(input_size=40, tau=2, device=device)
    model = BaseWakeModel(extractor, head, device=device)

    audio = torch.randn(2, 16000, device=device)
    with torch.no_grad():
        probs = torch.sigmoid(model(audio))
    print(f"\nBC-ResNet-2 probabilities: {probs.tolist()}")

    # --- With FilterbankExtractor (80 mels, richer features) ---
    fb_ext = FilterbankExtractor(sr=16000, n_mels=80)
    fb_head = BCResNetHead(input_size=80, tau=2, device=device)
    fb_model = BaseWakeModel(fb_ext, fb_head, device=device)
    fb_params = sum(p.numel() for p in fb_model.parameters())
    print(f"FilterBank-80 + BC-ResNet-2: {fb_params:,} parameters")

    # --- ONNX export ---
    with tempfile.TemporaryDirectory() as tmpdir:
        head_path = str(Path(tmpdir) / "bcresnet_head.onnx")
        head.eval()
        head.export_to_onnx(head_path)
        print(f"ONNX head size: {Path(head_path).stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
