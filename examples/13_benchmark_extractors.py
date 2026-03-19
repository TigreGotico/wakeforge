#!/usr/bin/env python3
"""Benchmark all feature extractors — latency, throughput, and parameter counts.

Useful for choosing the right extractor for your hardware budget.
Runs each extractor on synthetic audio and reports timing.
"""
import time

import torch

from ww_trainer.feats import (
    MfccExtractor,
    FilterbankExtractor,
    SincNetExtractor,
    GammatoneExtractor,
    DeltaExtractor,
)


def _bench_extractor(name: str, ext: torch.nn.Module, audio: torch.Tensor,
                     n_runs: int = 20, warmup: int = 3) -> None:
    """Benchmark a single extractor."""
    params = sum(p.numel() for p in ext.parameters())

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            ext(audio)

    # Timed runs
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            out = ext(audio)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_runs * 1000  # ms

    print(f"  {name:<25} {params:>10,} params  {out.shape[1]:>4} frames  "
          f"{out.shape[2]:>4}d  {elapsed:>7.2f} ms")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio = torch.randn(1, 16000, device=device)  # 1 second

    print(f"Benchmarking extractors on {device} (1s audio, 20 runs):\n")
    print(f"  {'Extractor':<25} {'Params':>10}        {'T':>4}    {'F':>4}  {'Latency':>7}")
    print("  " + "-" * 70)

    extractors = [
        ("MFCC-13", MfccExtractor(n_mfcc=13)),
        ("MFCC-40", MfccExtractor(n_mfcc=40)),
        ("FilterBank-40", FilterbankExtractor(n_mels=40)),
        ("FilterBank-80", FilterbankExtractor(n_mels=80)),
        ("SincNet-80", SincNetExtractor(n_filters=80)),
        ("Gammatone-64", GammatoneExtractor(n_filters=64)),
        ("Delta-MFCC-13", DeltaExtractor(
            MfccExtractor(n_mfcc=13))),
        ("Delta-FilterBank-40", DeltaExtractor(
            FilterbankExtractor(n_mels=40))),
    ]

    for name, ext in extractors:
        _bench_extractor(name, ext, audio)

    print("\nNote: HuBERT/Wav2Vec2/Wav2Vec2Bert require `transformers` and are much larger.")
    print("Export them to ONNX first, then use OnnxFeatureExtractor for inference benchmarks.")


if __name__ == "__main__":
    main()
