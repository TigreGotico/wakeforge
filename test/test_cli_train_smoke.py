"""End-to-end smoke test for the ``ww_trainer-train`` CLI.

The CLI ``train`` command was previously untested, which let two integration
bugs survive: ``wake_word`` passed both explicitly and via ``**opts``, and a
``feature_cache`` collision between the CLI wiring and the training loop. This
test runs the whole command on a tiny synthetic dataset so that path stays
exercised.
"""
import csv
import wave

import numpy as np
from click.testing import CliRunner

from ww_trainer.cli import train


def _write_wav(path, freq, sr=16000, dur=0.5):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    wav = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(wav.tobytes())


def _make_csv(tmp_path, name, rows):
    p = tmp_path / name
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return str(p)


def test_cli_train_micro_end_to_end(tmp_path):
    # Two separable classes: positives at 440 Hz, negatives at 110 Hz.
    rows_train, rows_test = [], []
    for i in range(8):
        pos = tmp_path / f"pos_{i}.wav"
        neg = tmp_path / f"neg_{i}.wav"
        _write_wav(pos, 440 + i)
        _write_wav(neg, 110 + i)
        (rows_train if i < 6 else rows_test).append((str(pos), "1"))
        (rows_train if i < 6 else rows_test).append((str(neg), "0"))

    train_csv = _make_csv(tmp_path, "train.csv", rows_train)
    test_csv = _make_csv(tmp_path, "test.csv", rows_test)
    out_dir = tmp_path / "model"

    result = CliRunner().invoke(train, [
        "--wake-word", "hey_test",
        "--metadata", train_csv,
        "--test-metadata", test_csv,
        "--tier", "micro",
        "--epochs", "1",
        "--batch-size", "4",
        "--device", "cpu",
        "--seed", "42",
        "--output-dir", str(out_dir),
        "--no-feature-cache",
    ])

    # Surface the traceback in the assertion message if it crashed.
    assert result.exit_code == 0, f"CLI train failed:\n{result.output}\n{result.exception}"
    assert out_dir.exists()
    assert list(out_dir.glob("*.pt")), "no checkpoint was written"


def test_cli_train_with_feature_cache(tmp_path):
    """Exercise the feature-cache wiring (the path that previously collided)."""
    rows = []
    for i in range(8):
        pos = tmp_path / f"p{i}.wav"
        neg = tmp_path / f"n{i}.wav"
        _write_wav(pos, 440 + i)
        _write_wav(neg, 110 + i)
        rows += [(str(pos), "1"), (str(neg), "0")]
    train_csv = _make_csv(tmp_path, "train.csv", rows)
    out_dir = tmp_path / "model"

    result = CliRunner().invoke(train, [
        "--wake-word", "hey_test",
        "--metadata", train_csv,
        "--tier", "micro",
        "--epochs", "1",
        "--batch-size", "4",
        "--device", "cpu",
        "--split", "0.75",
        "--output-dir", str(out_dir),
        "--feature-cache-dir", str(tmp_path / "fcache"),
    ])
    assert result.exit_code == 0, f"CLI train (feature cache) failed:\n{result.output}\n{result.exception}"
