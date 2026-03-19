#!/usr/bin/env python3
"""Genetic search over full architecture space: extractors + heads + losses.

Demonstrates run_genetic_search with full=True, which searches over
featurizer_type (mfcc/filterbank/sincnet/gammatone), classifier arch
(ffn/gru/cnn/bcresnet/tcresnet/...), and loss function (bce/focal/
arcface/supcon/...) in addition to standard hyperparameters.

Crossover mixes architectures from two parents — e.g. one parent's
extractor with another's classifier head — enabling discovery of
novel combinations that would not emerge from grid or random search.
"""
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ww_trainer.sweep import run_genetic_search, _build_search_space


def _make_csv(tmpdir: str, n: int = 40) -> str:
    """Generate a synthetic metadata CSV for demo."""
    csv_path = Path(tmpdir) / "metadata.csv"
    rows = []
    for i in range(n):
        t = np.linspace(0, 0.5, 8000)
        wav = np.sin(2 * np.pi * (440 + i * 10) * t).astype(np.float32)
        p = Path(tmpdir) / f"audio_{i}.wav"
        sf.write(str(p), wav, 16000)
        label = "1" if i < n // 2 else "0"
        rows.append(f"{p},{label}")
    csv_path.write_text("\n".join(rows))
    return str(csv_path)


def main() -> None:
    # Show what full=True adds to the search space
    basic_space = _build_search_space(full=False)
    full_space = _build_search_space(full=True)
    print("Basic search space keys:", list(basic_space.keys()))
    print("Full search space keys:", list(full_space.keys()))
    print(f"\nExtractors searched: {full_space['featurizer_type']}")
    print(f"Architectures searched: {full_space['arch']}")
    print(f"Losses searched: {full_space['loss']}")
    print(f"Feature dims searched: {full_space['n_features']}")

    # Demonstrate crossover concept
    parent_a = {"featurizer_type": "mfcc", "arch": "gru", "loss": "bce"}
    parent_b = {"featurizer_type": "sincnet", "arch": "bcresnet", "loss": "arcface"}
    print(f"\nCrossover example:")
    print(f"  Parent A: {parent_a}")
    print(f"  Parent B: {parent_b}")
    print(f"  Possible child: featurizer=sincnet (from B), arch=gru (from A), loss=arcface (from B)")

    # Run a tiny genetic search with synthetic data
    tmpdir = tempfile.mkdtemp()
    csv_path = _make_csv(tmpdir)

    print("\nRunning genetic search (full=True, 2 generations, population=4)...")
    print("In production, use generations=20+, population_size=20+.\n")

    result = run_genetic_search(
        metadata_csv=csv_path,
        population_size=4,
        generations=2,
        output_dir=str(Path(tmpdir) / "genetic_results"),
        device="cpu",
        epochs_per_trial=1,
        mutation_rate=0.3,
        elite_frac=0.5,
        full=True,
    )

    print(f"\nBest config: {result['best_config']}")
    print(f"Best score: {result['best_score']:.4f}")
    for gen in result["history"]:
        print(f"  Gen {gen['generation']}: best={gen['best']:.4f} avg={gen['avg']:.4f}")


if __name__ == "__main__":
    main()
