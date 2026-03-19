#!/usr/bin/env python3
"""Example: Multi-ONNX Wake Word Inference.

This example demonstrates a "production-style" pipeline that uses only
ONNX files and NumPy, with no PyTorch dependency for inference.

The pipeline consists of:
1. Base Extractor (ONNX) - e.g., MFCC
2. VAD model (ONNX) - e.g., Silero VAD
3. Classifier Head (ONNX) - e.g., GRU

The `OnnxWakeWordInferencer` handles loading all three models and 
performing the necessary concatenation and alignment.
"""
import numpy as np
import os
from pathlib import Path
import torch

from ww_trainer.feats import MfccExtractor, SileroVadWrapper
from ww_trainer.model import FfnClassifierHead, BaseWakeModel
from ww_trainer.inference import OnnxWakeWordInferencer

def main() -> None:
    print("=== Multi-ONNX production Pipeline (No PyTorch for Inference) ===")
    
    # --- PHASE 1: Export all components (Simulated) ---
    base_path = "base_mfcc.onnx"
    vad_path = "silero_vad.onnx"
    head_path = "classifier_head.onnx"
    
    print("Exporting components...")
    
    # 1. Base Extractor
    base = MfccExtractor(n_mfcc=13)
    base.to("cpu")
    base.export_to_onnx(base_path)
    
    # 2. Classifier Head (trained on 13 MFCC + 1 VAD = 14 inputs)
    head = FfnClassifierHead(input_size=14, hidden_dim=32)
    head.to("cpu")
    head.export_to_onnx(head_path)
    
    # 3. Silero VAD (We'll use the PyTorch Hub version to "pretend" we have the ONNX)
    # In a real scenario, you'd download the official silero_vad.onnx
    model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
    # We can't export it easily as I noted, but for this demo, let's assume it exists.
    # If it doesn't exist, the production inferencer will fail.
    # To make this demo runnable, I'll just check if the user has it, or skip.
    
    if not os.path.exists(vad_path):
        print(f"\n[Note] Official '{vad_path}' not found in local dir.")
        print("To run this for real, download 'silero_vad.onnx' from Silero.")
        # We can't proceed with the full multi-onnx test without the real VAD ONNX.
        # But we've verified the logic in inference.py and feats.py.
    else:
        # --- PHASE 2: Production Inference (NumPy + ORT only) ---
        print("\nLoading Production Inferencer...")
        # No PyTorch used here!
        inferencer = OnnxWakeWordInferencer(
            extractor_path=base_path,
            head_path=head_path,
            vad_path=vad_path
        )
        
        print("Running production inference on random noise...")
        dummy_audio = np.random.randn(16000).astype(np.float32)
        prob = inferencer.infer(dummy_audio)
        print(f"Wake probability: {prob:.4f}")

    # Cleanup
    for p in [base_path, head_path]:
        if Path(p).exists():
            Path(p).unlink()

if __name__ == "__main__":
    main()
