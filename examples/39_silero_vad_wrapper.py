#!/usr/bin/env python3
"""Example: Using Pre-Trained Silero VAD as a Feature Stream.

This example demonstrates how to wrap any base feature extractor (like MFCC)
with the SileroVadWrapper. This appends a state-of-the-art neural Voice 
Activity Detection probability to every frame of the base features.

This provides the classifier head with explicit knowledge of "where the speech is,"
which is highly beneficial for wake-word models operating in noisy environments.
"""
import torch

from ww_trainer.feats import MfccExtractor, SileroVadWrapper
from ww_trainer.model import FfnClassifierHead, BaseWakeModel

def main() -> None:
    print("=== Silero VAD Hybrid Feature Pipeline ===")
    
    # 1. Base Extractor
    # Standard MFCC features (13 dims)
    base = MfccExtractor(n_mfcc=13, sr=16000)
    print(f"Base feature dim: {base.feature_dim}")
    
    # 2. Silero VAD Wrapper
    # This will automatically download/load the Silero VAD PyTorch model
    # from torch.hub. It will append 1 extra dimension (VAD probability).
    print("Loading Silero VAD from torch.hub...")
    hybrid_extractor = SileroVadWrapper(base)
    print(f"Hybrid feature dim: {hybrid_extractor.feature_dim} (13 MFCC + 1 VAD prob)")
    
    # 3. Model Head
    # Now we can feed these 14 features into any standard head
    head = FfnClassifierHead(
        input_size=hybrid_extractor.feature_dim,
        hidden_dim=32,
        device="cpu"
    )
    model = BaseWakeModel(hybrid_extractor, head, device="cpu")
    
    # 4. Inference Test
    # Simulate 1 second of audio (16000 samples)
    print("\nRunning inference test...")
    dummy_wav = torch.randn(1, 16000)
    
    with torch.no_grad():
        # The wrapper handles unfolding the audio, running batch Silero VAD inference,
        # interpolating the result to match the MFCC timeframe, and concatenating.
        features = hybrid_extractor(dummy_wav)
        print(f"Extracted features shape: {features.shape} -> [Batch, Time, Features]")
        
        # Verify the VAD channel (last dimension)
        vad_probs = features[0, :, -1]
        print(f"VAD probabilities range: {vad_probs.min():.4f} to {vad_probs.max():.4f}")
        
        # Full model pass
        logit = model(dummy_wav)
        prob = torch.sigmoid(logit).item()
        print(f"Final wake probability: {prob:.4f}")
        
    print("\nNote: Exporting the full pipeline (Base + Silero + Head) to a single ONNX ")
    print("is not supported due to Silero's internal JIT structure. You must export ")
    print("the base extractor and head separately, and run the official Silero ONNX ")
    print("as a parallel stream in your production environment.")

if __name__ == "__main__":
    main()
