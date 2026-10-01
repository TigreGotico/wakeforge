#!/usr/bin/env python3
"""Large tier: HuBERT/Wav2Vec2 extractor + GRU classifier.

Transformer-based extractors (HuBERT, Wav2Vec2, Wav2Vec2Bert) produce
rich 768/1024-dim features but are too large for edge devices. The
workflow is:

  1. Train with transformer extractor (requires `transformers` library)
  2. Export extractor to ONNX
  3. Load ONNX extractor via OnnxFeatureExtractor for inference
  4. Optionally distill into a smaller CnnLstmExtractor

This example requires: pip install transformers

Note: Downloads ~100MB+ model weights on first run.
"""
import sys
import tempfile
from pathlib import Path

import torch

# Check if transformers is available
try:
    import transformers  # noqa: F401
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def demo_hubert() -> None:
    """HuBERT extractor demo."""
    from ww_trainer.feats import HubertExtractor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load HuBERT (tiny variant for demo) ---
    extractor = HubertExtractor(
        model_name="voidful/hubert-tiny-v2",
        device=device,
    )
    print(f"HuBERT feature_dim: {extractor.feature_dim}")

    # --- Feature extraction ---
    audio = torch.randn(1, 16000, device=device)
    with torch.no_grad():
        feats = extractor(audio)
    print(f"Feature shape: {feats.shape}")  # [1, ~49, feature_dim]

    # --- Export to ONNX ---
    tmpdir = tempfile.mkdtemp()
    onnx_path = str(Path(tmpdir) / "hubert_tiny.onnx")
    extractor.eval()
    extractor.export_to_onnx(onnx_path)
    print(f"ONNX size: {Path(onnx_path).stat().st_size / 1024 / 1024:.1f} MB")

    # --- Load back via OnnxFeatureExtractor ---
    from ww_trainer.feats import OnnxFeatureExtractor

    onnx_ext = OnnxFeatureExtractor(onnx_path, device="cpu")
    print(f"ONNX feature_dim: {onnx_ext.feature_dim}")

    # --- Pair with GRU head ---
    from ww_trainer.model import GruClassifierHead, BaseWakeModel

    head = GruClassifierHead(
        input_size=onnx_ext.feature_dim,
        hidden_dim=256,
        bidirectional=True,
        gru_n_layers=2,
        device="cpu",
    )
    # Note: for training, use the PyTorch HubertExtractor.
    # For inference/deployment, use the ONNX version.
    head_params = sum(p.numel() for p in head.parameters())
    print(f"GRU head: {head_params:,} parameters")


def demo_wav2vec2() -> None:
    """Wav2Vec2 extractor demo."""
    from ww_trainer.feats import Wav2Vec2Extractor

    device = "cpu"
    extractor = Wav2Vec2Extractor(
        model_name="patrickvonplaten/tiny-wav2vec2-no-tokenizer",
        device=device,
    )
    print(f"\nWav2Vec2 feature_dim: {extractor.feature_dim}")

    audio = torch.randn(1, 16000, device=device)
    with torch.no_grad():
        feats = extractor(audio)
    print(f"Feature shape: {feats.shape}")


def main() -> None:
    if not HAS_TRANSFORMERS:
        print("This example requires the `transformers` library.")
        print("Install with: uv pip install transformers")
        sys.exit(1)

    demo_hubert()
    demo_wav2vec2()

    print("\nFor production, export the transformer to ONNX and use")
    print("OnnxFeatureExtractor + a lightweight classifier head.")


if __name__ == "__main__":
    main()
