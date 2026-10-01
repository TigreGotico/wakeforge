#!/usr/bin/env python3
"""Train and Export a Markov Feature Extractor to ONNX.

This script takes a folder of wake-word samples, fits a MarkovTransitionExtractor,
and exports the full pipeline (Base + Markov) to a single ONNX file.

The resulting ONNX file can then be used as a "blackbox" featurizer in the 
standard ww-trainer pipeline via the `onnx` featurizer type.
"""
import os
import torch
import click
from pathlib import Path
from typing import List

from ww_trainer.feats import MfccExtractor, MarkovTransitionExtractor, FilterbankExtractor
from ww_trainer.utils import embed_onnx_metadata

def load_audio_list(folder: str) -> List[torch.Tensor]:
    """Load all .wav files from a folder as a list of 1D tensors."""
    import torchaudio
    p = Path(folder)
    wavs = []
    for ext in [".wav", ".flac", ".mp3"]:
        for f in p.rglob(f"*{ext}"):
            try:
                wav, sr = torchaudio.load(str(f))
                if sr != 16000:
                    wav = torchaudio.transforms.Resample(sr, 16000)(wav)
                wavs.append(wav.mean(dim=0))
            except Exception as e:
                print(f"Error loading {f}: {e}")
    return wavs

@click.command()
@click.option("--wake-folder", required=True, help="Folder with wake word samples for fitting.")
@click.option("--out", default="markov_featurizer.onnx", help="Output ONNX filename.")
@click.option("--base", type=click.Choice(["mfcc", "fb"]), default="mfcc", help="Base extractor type.")
@click.option("--n-codes", default=32, help="Number of VQ tokens.")
@click.option("--order", default=2, help="Markov chain order.")
@click.option("--n-mfcc", default=13, help="Number of coefficients if using MFCC.")
@click.option("--wake-name", default="unknown", help="Wake word name for metadata.")
def main(wake_folder, out, base, n_codes, order, n_mfcc, wake_name):
    print(f"Loading samples from {wake_folder}...")
    audio = load_audio_list(wake_folder)
    if not audio:
        print("No audio found!")
        return
    print(f"Loaded {len(audio)} samples.")

    # 1. Setup Base Extractor
    if base == "mfcc":
        base_ext = MfccExtractor(n_mfcc=n_mfcc)
    else:
        base_ext = FilterbankExtractor(n_mels=n_mfcc)

    # 2. Setup Markov Wrapper
    print(f"Initializing MarkovTransitionExtractor (order={order}, n_codes={n_codes})...")
    markov = MarkovTransitionExtractor(base_ext, n_codes=n_codes, order=order)
    # Force to CPU for reliable ONNX export
    markov.to("cpu")
    markov.device = torch.device("cpu")
    base_ext.to("cpu")
    base_ext.device = torch.device("cpu")

    # 3. Fit Model
    print("Fitting Markov chain (K-means + Transition Counting)...")
    markov.fit(audio)
    print("Fit complete.")

    # 4. Export to ONNX
    print(f"Exporting full pipeline to {out}...")
    metadata = {
        "wake_word": wake_name,
        "type": "markov_featurizer",
        "order": order,
        "n_codes": n_codes,
        "base_type": base
    }
    markov.export_to_onnx(out, metadata=metadata)
    
    if os.path.exists(out):
        print(f"Success! Model size: {os.path.getsize(out) / 1024:.1f} KB")
        print(f"You can now use this featurizer in ww-trainer with:")
        print(f"  --featurizer {out} --featurizer-type onnx")

if __name__ == "__main__":
    main()
