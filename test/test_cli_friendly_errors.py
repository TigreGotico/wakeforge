"""The CLIs report bad input in one line, never as a traceback."""
import csv
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from click.testing import CliRunner
from torch.onnx._internal.torchscript_exporter.utils import export as _ts_onnx_export

from ww_trainer.cli import train

ROOT = Path(__file__).resolve().parent.parent


ENTRY_POINTS = {
    "ww_trainer.inference": "cli_main",
    "ww_trainer.datagen": "cli_main",
    "ww_trainer.quickstart": "cli_main",
    "ww_trainer.cli": "train",
}


def _run(module: str, *args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1"}
    code = f"from {module} import {ENTRY_POINTS[module]} as main; main()"
    return subprocess.run([sys.executable, "-c", code, *args], capture_output=True,
                          text=True, env=env, timeout=timeout)


def _assert_friendly(proc: subprocess.CompletedProcess, *needles: str,
                     code: "int | None" = None) -> None:
    out = proc.stdout + proc.stderr
    if code is None:
        assert proc.returncode != 0, out
    else:
        assert proc.returncode == code, out
    assert "Traceback" not in out, out
    errors = [l for l in out.splitlines() if l.startswith("Error:") or ": error:" in l]
    assert errors, out
    for needle in needles:
        assert any(needle in l for l in errors), out


def _write_wav(path, freq=440, sr=16000, dur=0.5, channels=1):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    wav = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    wav = np.repeat(wav[:, None], channels, axis=1).reshape(-1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(wav.tobytes())


class _DiffExtractor(nn.Module):
    """Per-10 ms mean absolute first difference: depends on the sample rate."""

    def forward(self, x):
        d = (x[:, 1:] - x[:, :-1]).abs().unsqueeze(1)
        return F.avg_pool1d(d, 160, 160).transpose(1, 2)


class _MeanHead(nn.Module):
    def forward(self, x):
        return x.mean(dim=(1, 2)) * 40


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("onnx")
    ext, head = str(d / "extractor.onnx"), str(d / "head.onnx")
    _ts_onnx_export(
        _DiffExtractor().eval(), (torch.zeros(1, 16000),), ext,
        input_names=["input_values"], output_names=["features"],
        dynamic_axes={"input_values": {0: "B", 1: "T"}, "features": {0: "B", 1: "T"}},
        opset_version=14)
    _ts_onnx_export(
        _MeanHead().eval(), (torch.zeros(1, 10, 1),), head,
        input_names=["input_features"], output_names=["logits"],
        dynamic_axes={"input_features": {0: "B", 1: "T"}}, opset_version=14)
    return ext, head


def _score(proc: subprocess.CompletedProcess) -> float:
    line = next(l for l in proc.stdout.splitlines() if l.startswith("score="))
    return float(line.split()[0].split("=")[1])


@pytest.mark.parametrize("sr,channels", [(24000, 1), (24000, 2), (16000, 2)])
def test_infer_converts_to_mono_16k(tmp_path, models, sr, channels):
    ext, head = models
    reference = tmp_path / "ref.wav"
    other = tmp_path / "other.wav"
    _write_wav(reference, sr=16000)
    _write_wav(other, sr=sr, channels=channels)
    args = ("--featurizer", ext, "--model", head, "--device", "cpu")
    want = _score(_run("ww_trainer.inference", *args, "--audio", str(reference)))
    got_proc = _run("ww_trainer.inference", *args, "--audio", str(other))
    assert "Traceback" not in got_proc.stderr, got_proc.stderr
    assert abs(_score(got_proc) - want) < 0.02


def test_infer_missing_audio(tmp_path, models):
    ext, head = models
    missing = str(tmp_path / "nope.wav")
    _assert_friendly(_run("ww_trainer.inference", "--featurizer", ext, "--model", head,
                          "--audio", missing), missing, code=2)


def test_infer_missing_model(tmp_path, models):
    ext, _ = models
    wav = tmp_path / "a.wav"
    _write_wav(wav)
    _assert_friendly(_run("ww_trainer.inference", "--featurizer", ext,
                          "--model", "/nonexistent/x.onnx", "--audio", str(wav)),
                     "/nonexistent/x.onnx", code=2)


@pytest.mark.parametrize("flag", ["--model", "--featurizer"])
def test_infer_corrupt_onnx(tmp_path, models, flag):
    ext, head = models
    wav = tmp_path / "a.wav"
    _write_wav(wav)
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(os.urandom(2048))
    args = {"--featurizer": ext, "--model": head}
    args[flag] = str(bad)
    proc = _run("ww_trainer.inference", "--featurizer", args["--featurizer"],
                "--model", args["--model"], "--audio", str(wav), "--device", "cpu")
    _assert_friendly(proc, str(bad), code=2)


def test_infer_swapped_featurizer_and_head(tmp_path, models):
    ext, head = models
    wav = tmp_path / "a.wav"
    _write_wav(wav)
    _assert_friendly(_run("ww_trainer.inference", "--featurizer", head, "--model", ext,
                          "--audio", str(wav), "--device", "cpu"), "swapped", code=2)


def test_infer_missing_featurizer_keeps_value_error(tmp_path, models):
    _, head = models
    wav = tmp_path / "a.wav"
    _write_wav(wav)
    proc = _run("ww_trainer.inference", "--model", head, "--audio", str(wav),
                "--device", "cpu")
    assert proc.returncode != 0
    assert "does not record a pretrained featurizer" in proc.stderr


def test_train_missing_metadata(tmp_path):
    missing = "./nope/metadata.csv"
    proc = _run("ww_trainer.cli", "--wake-word", "hey_test", "--metadata", missing,
                "--tier", "micro", "--device", "cpu", "--output-dir", str(tmp_path / "o"))
    _assert_friendly(proc, missing)


def test_train_missing_test_metadata(tmp_path):
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("")
    missing = "./nope/test.csv"
    proc = _run("ww_trainer.cli", "--wake-word", "hey_test", "--metadata", str(train_csv),
                "--test-metadata", missing, "--tier", "micro", "--device", "cpu",
                "--output-dir", str(tmp_path / "o"))
    _assert_friendly(proc, missing)


@pytest.mark.parametrize("module", ["ww_trainer.datagen", "ww_trainer.quickstart"])
@pytest.mark.parametrize("phrase", ["", "   "])
def test_empty_wake_word_refused(tmp_path, module, phrase):
    out = tmp_path / "out"
    proc = _run(module, "--wake-word", phrase, "--output-dir", str(out))
    _assert_friendly(proc, "--wake-word")
    assert not out.exists()


def _dataset(tmp_path):
    rows = []
    for i in range(8):
        pos, neg = tmp_path / f"p{i}.wav", tmp_path / f"n{i}.wav"
        _write_wav(pos, 440 + i)
        _write_wav(neg, 110 + i)
        rows += [(str(pos), "1"), (str(neg), "0")]
    csv_path = tmp_path / "train.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return str(csv_path)


def _train_args(csv_path, out_dir, *extra):
    return ["--wake-word", "hey_test", "--metadata", csv_path, "--tier", "micro",
            "--epochs", "1", "--batch-size", "4", "--device", "cpu", "--split", "0.75",
            "--output-dir", str(out_dir), "--no-feature-cache", *extra]


def _log_rows(out_dir):
    with open(Path(out_dir) / "metrics_log.csv") as f:
        return list(csv.reader(f))[1:]


def test_rerun_into_used_output_dir_refused_then_overwritten(tmp_path):
    csv_path = _dataset(tmp_path)
    out = tmp_path / "model"
    first = CliRunner().invoke(train, _train_args(csv_path, out))
    assert first.exit_code == 0, f"{first.output}\n{first.exception}"
    assert len(_log_rows(out)) == 1

    again = CliRunner().invoke(train, _train_args(csv_path, out))
    assert again.exit_code != 0
    assert isinstance(again.exception, SystemExit)
    assert str(out / "ep1.pt") in again.output
    assert "--resume" in again.output and "--overwrite" in again.output
    assert len(_log_rows(out)) == 1

    clean = CliRunner().invoke(train, _train_args(csv_path, out, "--overwrite"))
    assert clean.exit_code == 0, f"{clean.output}\n{clean.exception}"
    assert len(_log_rows(out)) == 1


def test_resume_with_no_epochs_left(tmp_path):
    csv_path = _dataset(tmp_path)
    out = tmp_path / "model"
    first = CliRunner().invoke(train, _train_args(csv_path, out))
    assert first.exit_code == 0, f"{first.output}\n{first.exception}"

    done = CliRunner().invoke(train, _train_args(csv_path, out, "--resume", str(out / "ep1.pt")))
    assert done.exit_code != 0
    assert "no epochs left to train (checkpoint is at epoch 1 of 1)" in done.output
    assert "Training complete" not in done.output


def test_overwrite_moves_previous_run_aside(tmp_path):
    csv_path = _dataset(tmp_path)
    out = tmp_path / "model"
    out.mkdir()
    for n in range(1, 6):
        (out / f"ep{n}.pt").write_bytes(b"old")
    (out / "metrics_log.csv").write_text("epoch\n1\n")

    result = CliRunner().invoke(train, _train_args(csv_path, out, "--overwrite", "--epochs", "2"))
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert sorted(p.name for p in out.glob("ep*.pt")) == ["ep1.pt", "ep2.pt"]
    assert all(p.stat().st_size > 3 for p in out.glob("ep*.pt"))
    (archive,) = [p for p in out.iterdir() if p.name.startswith(".old-")]
    assert sorted(p.name for p in archive.glob("ep*.pt")) == [f"ep{n}.pt" for n in range(1, 6)]
    assert (archive / "metrics_log.csv").read_text() == "epoch\n1\n"
    assert len(_log_rows(out)) == 2


def test_rerun_guard_sees_multi_stage_checkpoints(tmp_path):
    csv_path = _dataset(tmp_path)
    out = tmp_path / "model"
    (out / "stage_0").mkdir(parents=True)
    (out / "stage_0" / "best_f1.pt").write_bytes(b"old")
    result = CliRunner().invoke(
        train, _train_args(csv_path, out, "--training-stages", "1:1e-3"))
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    assert str(out / "stage_0" / "best_f1.pt") in result.output
    assert "--overwrite" in result.output


def test_overwrite_archives_everything_the_old_run_wrote(tmp_path):
    csv_path = _dataset(tmp_path)
    out = tmp_path / "model"
    for rel in ("stage_0/best_f1.pt", "viz/pca_1.png", "artifacts/fp.csv", "model.onnx",
                "final_model.pt", "metrics_log.csv"):
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old")
    earlier = out / ".old-20200101T000000Z"
    earlier.mkdir()
    (earlier / "keep.pt").write_bytes(b"older")

    result = CliRunner().invoke(train, _train_args(csv_path, out, "--overwrite"))
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    for rel in ("stage_0/best_f1.pt", "viz/pca_1.png", "artifacts/fp.csv", "model.onnx"):
        assert not (out / rel).exists(), rel
    (archive,) = [p for p in out.iterdir()
                  if p.name.startswith(".old-") and p.name != earlier.name]
    for rel in ("stage_0/best_f1.pt", "viz/pca_1.png", "artifacts/fp.csv", "model.onnx",
                "final_model.pt", "metrics_log.csv"):
        assert (archive / rel).read_bytes() == b"old", rel
    assert (earlier / "keep.pt").is_file()


def test_overwrite_keeps_inputs_that_live_in_the_output_dir(tmp_path):
    csv_path = _dataset(tmp_path)
    out = tmp_path
    (out / "ep1.pt").write_bytes(b"old")
    (out / "metrics_log.csv").write_text("epoch\n1\n")

    result = CliRunner().invoke(train, _train_args(csv_path, out, "--overwrite"))
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert Path(csv_path).is_file()
    assert (out / "p0.wav").is_file()
    (archive,) = [p for p in out.iterdir() if p.name.startswith(".old-")]
    assert (archive / "ep1.pt").read_bytes() == b"old"


def test_overwrite_keeps_audio_a_relative_csv_points_at(tmp_path, monkeypatch):
    out = tmp_path / "model"
    clips = out / "clips"
    clips.mkdir(parents=True)
    rows = []
    for i in range(8):
        _write_wav(clips / f"p{i}.wav", 440 + i)
        _write_wav(clips / f"n{i}.wav", 110 + i)
        rows += [f"clips/p{i}.wav,1", f"clips/n{i}.wav,0"]
    csv_path = out / "train.csv"
    csv_path.write_text("\n".join(rows) + "\n")
    (out / "ep1.pt").write_bytes(b"old")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = CliRunner().invoke(train, _train_args(str(csv_path), out, "--overwrite"))
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert len(list(clips.glob("*.wav"))) == 16
    assert csv_path.is_file()
    (archive,) = [p for p in out.iterdir() if p.name.startswith(".old-")]
    assert (archive / "ep1.pt").read_bytes() == b"old"
