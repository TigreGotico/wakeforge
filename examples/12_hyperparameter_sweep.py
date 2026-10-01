#!/usr/bin/env python3
"""Hyperparameter sweep with Optuna.

Automatically searches over architecture (FFN/GRU/CNN), hidden dimensions,
learning rate, batch size, and dropout to find optimal settings for your
wake word dataset.

Requires: pip install optuna
"""
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


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
    try:
        import optuna  # noqa: F401
    except ImportError:
        print("This example requires optuna: uv pip install optuna")
        return

    from ww_trainer.sweep import run_sweep

    tmpdir = tempfile.mkdtemp()
    csv_path = _make_csv(tmpdir)

    print("Running hyperparameter sweep (3 trials, 2 epochs each)...")
    print("In practice, use n_trials=50+ and epochs=10+ for real results.\n")

    run_sweep(
        metadata_csv=csv_path,
        n_trials=3,
        output_dir=str(Path(tmpdir) / "sweep_results"),
        featurizer_type="mfcc",
        epochs_per_trial=2,
        device="cpu",
        study_name="demo_sweep",
    )

    # Check results
    results_dir = Path(tmpdir) / "sweep_results"
    best_params = results_dir / "best_params.json"
    if best_params.exists():
        print(f"\nBest params saved to: {best_params}")
        print(best_params.read_text())


if __name__ == "__main__":
    main()
