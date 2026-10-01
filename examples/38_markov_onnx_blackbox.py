#!/usr/bin/env python3
"""End-to-End: Using a Markov ONNX Featurizer as a Blackbox.

This example demonstrates the full lifecycle:
1. Load a pre-trained Markov featurizer (exported as ONNX).
2. Wrap it in `OnnxFeatureExtractor`.
3. Use it to train a standard neural head (Ffn or Gru).

This allows benchmarking different deep learning heads against the same
frozen, classical sequential features.
"""
import torch
import os
from pathlib import Path

from ww_trainer.feats import OnnxFeatureExtractor, MfccExtractor, MarkovTransitionExtractor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel


def main() -> None:
    print("=== Blackbox Markov ONNX End-to-End Example ===")
    
    # --- PHASE 1: Create and Export the Featurizer (Simulated) ---
    # In a real scenario, you would use `scripts/train_markov_featurizer.py`
    onnx_path = "markov_blackbox.onnx"
    print(f"Creating mock featurizer: {onnx_path}...")
    
    base = MfccExtractor(n_mfcc=13)
    markov = MarkovTransitionExtractor(base, n_codes=16, order=1)
    # Force to CPU for reliable ONNX export
    markov.to("cpu")
    markov.device = torch.device("cpu")
    base.to("cpu")
    base.device = torch.device("cpu")
    
    # Fit on some random noise to satisfy the pipeline
    markov.fit([torch.randn(16000) for _ in range(5)])
    markov.export_to_onnx(onnx_path)
    
    # --- PHASE 2: Load as Blackbox ---
    print("\nLoading featurizer as blackbox ONNX...")
    # This only requires ONNX Runtime, no markovonnx or fitting logic needed here.
    blackbox_ext = OnnxFeatureExtractor(onnx_path)
    print(f"Detected feature dimension: {blackbox_ext.feature_dim}")
    # (13 MFCC + 16 Markov Probs = 29)
    
    # --- PHASE 3: Standard Training Pipeline ---
    print("\nBuilding classifier head for blackbox features...")
    head = FfnClassifierHead(
        input_size=blackbox_ext.feature_dim,
        hidden_dim=64
    )
    
    # This model can now be trained normally with WakeWordTrainer
    model = BaseWakeModel(blackbox_ext, head)
    
    # Test a forward pass
    dummy_wav = torch.randn(1, 16000)
    with torch.no_grad():
        # Extractor runs in ONNX Runtime, Head runs in PyTorch
        logit = model(dummy_wav)
        prob = torch.sigmoid(logit).item()
        
    print(f"Inference successful.")
    print(f"Input: Waveform [1, 16000]")
    print(f"Blackbox features: {blackbox_ext.sess.get_outputs()[0].name}")
    print(f"Wake probability: {prob:.4f}")

    # Cleanup
    if Path(onnx_path).exists():
        Path(onnx_path).unlink()

if __name__ == "__main__":
    main()
