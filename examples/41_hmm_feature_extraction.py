#!/usr/bin/env python3
"""Example: Classical HMM Feature Extraction.

This script demonstrates how to train a Hidden Markov Model (HMM) on 
wake-word audio and use it to extract "state posteriors" as features.

Unlike neural networks, HMMs are statistically transparent and highly 
efficient for temporal modeling on microcontrollers.
"""
import torch
import numpy as np

from ww_trainer.feats import MfccExtractor, HMMStateExtractor

def main() -> None:
    print("=== Classical HMM Feature Extraction ===")
    
    # 1. Setup Base Extractor
    # HMMs usually work best on top of spectral features like MFCC
    base = MfccExtractor(n_mfcc=13)
    
    # 2. Setup HMM Extractor
    # n_states: Number of acoustic units to model (e.g. 8 sounds in the word)
    # n_codes: Vocabulary size for vector quantization (e.g. 32 clusters)
    hmm_ext = HMMStateExtractor(base, n_states=8, n_codes=32)
    print(f"HMM Feature Dimension: {hmm_ext.feature_dim} (13 MFCC + 8 States)")
    
    # 3. Simulate Training Data
    # In a real scenario, use actual wake-word audio samples
    print("\nSimulating training data...")
    fake_audio = [torch.randn(16000) for _ in range(5)]
    
    # 4. Train HMM (Baum-Welch)
    # This clusters audio frames and learns transition/emission probabilities
    try:
        hmm_ext.fit(fake_audio, n_iter=10)
        print("HMM training complete.")
    except ImportError:
        print("Error: markovonnx is required for HMM training.")
        return
        
    # 5. Extract Features
    # The output contains the original MFCCs plus 8 probability channels
    print("\nExtracting features from a new sample...")
    test_wav = torch.randn(1, 16000)
    features = hmm_ext(test_wav) # [Batch, Time, Features]
    
    print(f"Output shape: {features.shape}")
    
    # Analyze state posteriors (last 8 dims)
    posteriors = features[0, :, 13:]
    frame_idx = 50
    print(f"State distribution at frame {frame_idx}:")
    for i, p in enumerate(posteriors[frame_idx]):
        print(f"  State {i}: {p:.4f}")
        
    print(f"\nSum of posteriors: {posteriors[frame_idx].sum():.4f}")
    
    # 6. Export for Deployment
    # The entire pipeline (MFCC -> VQ -> HMM) is exported as one ONNX graph
    onnx_path = "hmm_featurizer.onnx"
    print(f"\nExporting full pipeline to {onnx_path}...")
    hmm_ext.export_to_onnx(onnx_path)
    print("Success!")
    
    import os
    if os.path.exists(onnx_path):
        os.remove(onnx_path)

if __name__ == "__main__":
    main()
