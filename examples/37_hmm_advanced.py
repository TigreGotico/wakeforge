#!/usr/bin/env python3
"""Advanced Hybrid Wake Word Model: MFCC + HMM + GRU.

This example demonstrates training an HMM feature extractor on wake-word
audio and using its latent state posteriors as input to a neural head.

This 'Hybrid Architecture' combines classical sequential modeling (HMM)
with deep temporal modeling (GRU).
"""
import torch
from pathlib import Path

from ww_trainer.feats import MfccExtractor, HMMStateExtractor
from ww_trainer.model import GruClassifierHead, BaseWakeModel


def main() -> None:
    print("=== Hybrid HMM + GRU Wake Word Architecture ===")
    
    # 1. Setup Extractors
    base = MfccExtractor(n_mfcc=13)
    # 8-state HMM representing the acoustic sub-components of the wake word
    hmm = HMMStateExtractor(base, n_states=8, n_codes=32)
    
    # 2. Train HMM (Unsupervised)
    print("Training HMM on synthetic patterns...")
    # Generate repetitive patterns to simulate a structured word
    # Frame 0-25: Pattern A, 25-50: Pattern B, 50-75: Pattern C, 75-100: Pattern D
    t = torch.linspace(0, 1, 16000)
    pattern_a = torch.sin(2 * 3.14 * 440 * t) # 440Hz
    pattern_b = torch.sin(2 * 3.14 * 880 * t) # 880Hz
    fake_samples = [pattern_a, pattern_b, pattern_a, pattern_b]
    
    try:
        hmm.fit(fake_samples, n_iter=5)
        print("HMM parameters fitted.")
    except ImportError:
        print("markovonnx not installed - skipping fit.")

    # 3. Setup Classifier Head
    # Input size = 13 (MFCC) + 8 (HMM Posteriors) = 21
    print(f"Total features per frame: {hmm.feature_dim}")
    head = GruClassifierHead(
        input_size=hmm.feature_dim,
        hidden_dim=32,
        num_layers=1,
        dropout=0.1
    )
    
    # 4. Assemble Full Model
    model = BaseWakeModel(hmm, head)
    print(f"Hybrid Model params: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Export Full Hybrid Pipeline to ONNX
    # This exports: Wav -> MFCC -> VQ -> HMM -> GRU
    onnx_path = "hybrid_hmm_gru.onnx"
    print(f"Exporting full pipeline to {onnx_path}...")
    model.export_to_onnx(onnx_path, export_featurizer=True)
    
    if Path(onnx_path).exists():
        print("Export successful.")
        # Cleanup
        Path(onnx_path).unlink()
        Path(onnx_path.replace(".onnx", "_featurizer.onnx")).unlink()

    # 6. Inference Example
    with torch.no_grad():
        dummy_wav = torch.randn(1, 16000)
        # Raw features (MFCC + HMM)
        feats = hmm(dummy_wav)
        print(f"HMM Posterior shape: {feats[..., 13:].shape}")
        
        # Classifier probability
        prob = torch.sigmoid(model(dummy_wav)).item()
        print(f"Final wake word probability: {prob:.4f}")

if __name__ == "__main__":
    main()
