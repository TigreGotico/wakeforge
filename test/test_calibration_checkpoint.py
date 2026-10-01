"""Checkpoint selection on calibration audio (``--calib-speech``).

The calibration metric is recall at zero false accepts: the share of noisy
validation wake clips scoring above the highest window of held-out speech or
babble. Runs use the stand-in pretrained featurizer of ``fake_hub``
(``conftest.py``) and built-in MFCC, on CPU.
"""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from click.testing import CliRunner

import ww_trainer.loop as loop
from ww_trainer.calibration_audio import CalibrationSet, build_calibration_set, calib_recall
from ww_trainer.cli import train
from ww_trainer.feature_store import FeatureStore
from ww_trainer.trainer import WakeWordTrainer

SR = 16000
TRAIN_KW = dict(epochs=2, batch_size=4, lr=1e-3, mine_fraction=0.0, pca_every=0,
                save_best=False, metrics_log="metrics.csv")


@pytest.fixture
def clips(tmp_path):
    """4 positives and 4 negatives of different lengths (0.5 to 1.1 s)."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(4):
        n = int(SR * (0.5 + 0.2 * i))
        t = np.arange(n) / SR
        pos, neg = tmp_path / f"pos_{i}.wav", tmp_path / f"neg_{i}.wav"
        sf.write(pos, (0.3 * np.sin(2 * np.pi * (440 + 40 * i) * t)).astype(np.float32), SR)
        sf.write(neg, (0.1 * rng.standard_normal(n)).astype(np.float32), SR)
        rows += [(str(pos), "1"), (str(neg), "0")]
    return rows


@pytest.fixture
def noise_dir(tmp_path):
    d = tmp_path / "noise"
    d.mkdir()
    sf.write(d / "n.wav", (0.05 * np.random.default_rng(1).standard_normal(SR)).astype(np.float32), SR)
    return str(d)


@pytest.fixture
def speech_dir(tmp_path):
    """A LibriSpeech-style tree: speaker/chapter/utterance.flac, 2 to 3 s each."""
    rng = np.random.default_rng(2)
    root = tmp_path / "calib-speech"
    for spk in ("11", "22"):
        d = root / spk / "100"
        d.mkdir(parents=True)
        for k in range(2):
            n = int(SR * (2 + k))
            t = np.arange(n) / SR
            w = 0.2 * np.sin(2 * np.pi * (150 + 30 * k) * t) * (1 + np.sin(2 * np.pi * 3 * t))
            sf.write(d / f"{spk}-100-{k:04d}.flac", (w + 0.02 * rng.standard_normal(n)).astype(np.float32), SR)
    return str(root)


def _trainer(featurizer="wakehubert", **kwargs):
    return WakeWordTrainer(arch="gru", featurizer=None, featurizer_type=featurizer, device="cpu",
                           seed=0, losses_cfg=[{"name": "bce", "weight": 1.0}], **kwargs)


def _metrics(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _state(path):
    return torch.load(path, map_location="cpu")


def _same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------------------
# 1. The metric
# ---------------------------------------------------------------------------

def test_calib_recall_counts_positives_above_the_highest_negative():
    assert calib_recall([3.0, 2.0, 1.0], [0.5, 1.5]) == pytest.approx(2 / 3)
    assert calib_recall([0.9, 0.8, 0.95], [0.1, 0.2, 0.3]) == 1.0
    assert calib_recall([1.0, 2.0], [1.0]) == 0.5  # a tie with the top negative is a false accept


def test_one_loud_negative_zeroes_calib_recall():
    pos = [0.9, 0.8, 0.95, 0.97]
    quiet = [0.01] * 999
    assert calib_recall(pos, quiet) == 1.0
    assert calib_recall(pos, quiet + [0.99]) == 0.0


def test_calib_recall_needs_negatives():
    assert calib_recall([], [0.1]) == 0.0
    with pytest.raises(ValueError):
        calib_recall([0.5], [])


# ---------------------------------------------------------------------------
# The calibration set
# ---------------------------------------------------------------------------

def test_calibration_set_layout(clips, speech_dir, noise_dir):
    wakes = [c for c in clips if c[1] == "1"]
    calib = build_calibration_set(wakes, speech_dir, noise_dir, seconds=10)
    lengths = [int(sf.info(p).frames) for p, _ in wakes]
    # babble at 10 dB, babble at 5 dB and noise at 5 dB, each at its clip's length
    assert [len(p) for p in calib.positives] == lengths * 3
    window = int(np.median(lengths))
    assert calib.negatives and all(len(w) == window for w in calib.negatives)
    # half speech (5 s read whole), half babble (one 5 s segment), windows every 0.5 s
    speech_windows = len(calib.negatives) - (5 * SR - window) // (SR // 2) - 1
    assert speech_windows > 0
    without_noise = build_calibration_set(wakes, speech_dir, None, seconds=10)
    assert len(without_noise.positives) == 2 * len(wakes)


def test_calibration_set_is_reproducible(clips, speech_dir, noise_dir):
    wakes = [c for c in clips if c[1] == "1"]
    a = build_calibration_set(wakes, speech_dir, noise_dir, seconds=10)
    b = build_calibration_set(wakes, speech_dir, noise_dir, seconds=10)
    assert all(torch.equal(x, y) for x, y in zip(a.positives + a.negatives, b.positives + b.negatives))


# ---------------------------------------------------------------------------
# 2. A training run with calibration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("featurizer", ["wakehubert", "mfcc"])
def test_training_run_writes_best_calib_and_logs_calib_recall(fake_hub, clips, speech_dir, noise_dir,
                                                               tmp_path, featurizer):
    trainer = _trainer(featurizer, bg_noise_folder=noise_dir)
    out = tmp_path / "out"
    trainer.train(output_dir=out, train_data=clips, test_data=clips,
                  calib_speech=speech_dir, calib_seconds=10, **TRAIN_KW)
    assert (out / "best_calib.pt").exists() and (out / "best_calib.ts").exists()
    rows = _metrics(out / "metrics.csv")
    assert [int(r["epoch"]) for r in rows] == [1, 2]
    for r in rows:
        assert 0.0 <= float(r["calib_recall"]) <= 1.0
        assert float(r["val_loss"]) > 0.0
    saved = _state(out / "best_calib.ts")["metrics"]
    assert saved["calib_recall"] == max(float(r["calib_recall"]) for r in rows)
    assert _same_weights(_state(out / "final_model.pt"), _state(out / "best_calib.pt"))


def test_calibration_audio_stays_out_of_the_feature_store(fake_hub, clips, speech_dir, tmp_path):
    trainer = _trainer()
    trainer.train(output_dir=tmp_path / "out", train_data=clips, test_data=clips,
                  calib_speech=speech_dir, calib_seconds=10, **TRAIN_KW)
    store = FeatureStore.for_extractor(trainer.model.feature_extractor)
    assert len(store) > 0
    assert {path for path, _ in store._memory} <= {p for p, _ in clips}


def test_calibration_audio_stays_out_of_a_prefilled_disk_store(fake_hub, clips, speech_dir, tmp_path):
    trainer = _trainer()
    cache = tmp_path / "feature-cache"
    store = FeatureStore.for_extractor(trainer.model.feature_extractor, str(cache))
    trainer.train(output_dir=tmp_path / "out", train_data=clips, test_data=clips,
                  feature_store=store, feature_cache_workers=1,
                  calib_speech=speech_dir, calib_seconds=10, **TRAIN_KW)
    assert (tmp_path / "out" / "best_calib.pt").exists()
    clip_files = {store._disk_path(p, 0) for p, _ in clips}
    assert set(cache.glob("*.npy")) == clip_files
    assert {path for path, _ in store._memory} <= {p for p, _ in clips}


def test_cli_calib_speech(fake_hub, clips, speech_dir, tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("".join(f"{p},{l}\n" for p, l in clips))
    out = tmp_path / "model"
    result = CliRunner().invoke(train, [
        "--wake-word", "hey_test", "--metadata", str(csv_path), "--test-metadata", str(csv_path),
        "--featurizer-type", "wakehubert", "--epochs", "2", "--batch-size", "4", "--device", "cpu",
        "--output-dir", str(out), "--no-feature-cache", "--mine-sample", "0", "--pca-every", "0",
        "--calib-speech", speech_dir, "--calib-seconds", "10",
    ])
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert (out / "best_calib.pt").exists()
    assert "calib_recall" in _metrics(out / "metrics_log.csv")[0]
    meta = json.loads((out / "hey_test_meta.json").read_text())
    assert meta["calib_speech"] == speech_dir and meta["calib_seconds"] == 10


# ---------------------------------------------------------------------------
# 3. Selection
# ---------------------------------------------------------------------------

def _scripted(monkeypatch, recalls, losses):
    """Make epoch k score calib_recall ``recalls[k]`` with validation loss ``losses[k]``."""
    calls = {"scores": 0, "loss": 0}

    def scores(*args, **kwargs):
        r = recalls[calls["scores"]]
        calls["scores"] += 1
        return np.array([1.0] * int(r * 4) + [-1.0] * (4 - int(r * 4))), np.array([0.0])

    def loss(targets, probs):
        v = losses[calls["loss"]]
        calls["loss"] += 1
        return v

    monkeypatch.setattr(loop, "calibration_scores", scores)
    monkeypatch.setattr(loop, "validation_loss", loss)
    return calls


@pytest.mark.parametrize("recalls,losses,best_epoch", [
    ([1.0, 0.5], [0.5, 0.1], 0),    # later epoch: worse recall, better loss
    ([0.5, 1.0], [0.1, 0.5], 1),    # later epoch: better recall, worse loss
    ([0.75, 0.75], [0.5, 0.3], 1),  # tie on recall: the lower validation loss wins
    ([0.75, 0.75], [0.3, 0.5], 0),
])
def test_best_calib_keeps_the_best_epoch_and_becomes_the_final_model(
        fake_hub, clips, speech_dir, tmp_path, monkeypatch, recalls, losses, best_epoch):
    calls = _scripted(monkeypatch, recalls, losses)
    out = tmp_path / "out"
    _trainer().train(output_dir=out, train_data=clips, test_data=clips,
                     calib_speech=speech_dir, calib_seconds=10, **TRAIN_KW)
    assert calls["scores"] == 2
    assert [float(r["calib_recall"]) for r in _metrics(out / "metrics.csv")] == recalls
    assert _state(out / "best_calib.ts")["epoch"] == best_epoch
    best = _state(out / "best_calib.pt")
    assert _same_weights(best, _state(out / f"ep{best_epoch + 1}.pt"))
    other = _state(out / f"ep{2 - best_epoch}.pt")
    assert not _same_weights(best, other)
    final = _state(out / "final_model.pt")
    assert _same_weights(final, best)
    final_state = _state(out / "final_model.ts")
    assert final_state["epoch"] == best_epoch
    assert final_state["metrics"]["calib_recall"] == recalls[best_epoch]


# ---------------------------------------------------------------------------
# 4. Without the flag
# ---------------------------------------------------------------------------

def test_without_calib_speech_no_calibration_work_happens(fake_hub, clips, tmp_path, monkeypatch):
    calls = []

    def spy(*args, **kwargs):
        calls.append(args)
        return CalibrationSet([torch.zeros(SR)], [torch.zeros(SR)])

    monkeypatch.setattr(loop, "build_calibration_set", spy)
    monkeypatch.setattr(loop, "calibration_scores",
                        lambda *a, **k: calls.append(a) or (np.ones(1), np.zeros(1)))
    out = tmp_path / "out"
    _trainer().train(output_dir=out, train_data=clips, test_data=clips, **TRAIN_KW)
    assert calls == []
    assert not (out / "best_calib.pt").exists()
    with open(out / "metrics.csv", newline="") as f:
        assert next(csv.reader(f)) == ["epoch", "loss", "accuracy", "precision", "recall", "f1", "auc"]
    assert _same_weights(_state(out / "final_model.pt"), _state(out / "ep2.pt"))


# ---------------------------------------------------------------------------
# Multi-stage training
# ---------------------------------------------------------------------------

STAGE_CASES = [
    ([1.0, 0.5, 0.75, 0.5], [0.4, 0.4, 0.4, 0.4], 0, 0),   # stage 0, epoch 0 wins
    ([1.0, 0.5, 0.5, 1.0], [0.4, 0.4, 0.4, 0.3], 1, 1),    # tie on recall, stage 1 has the lower loss
]


@pytest.mark.parametrize("save_best", [True, False])
@pytest.mark.parametrize("recalls,losses,best_stage,best_epoch", STAGE_CASES)
def test_multi_stage_chains_and_finishes_on_best_calib(fake_hub, clips, speech_dir, tmp_path, monkeypatch,
                                                       save_best, recalls, losses, best_stage, best_epoch):
    from ww_trainer.multi_stage import run_multi_stage_training
    _scripted(monkeypatch, recalls, losses)
    trainer = _trainer()
    starts = []
    train = trainer.train

    def recording_train(**kwargs):
        starts.append({k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()})
        return train(**kwargs)

    monkeypatch.setattr(trainer, "train", recording_train)
    out = tmp_path / "out"
    kw = {**TRAIN_KW, "save_best": save_best}
    kw.pop("epochs")
    kw.pop("lr")
    run_multi_stage_training(trainer, [{"epochs": 2, "lr": 1e-3}, {"epochs": 2, "lr": 1e-3}], clips, clips,
                             out, calib_speech=speech_dir, calib_seconds=10, **kw)

    stage0 = _state(out / "stage_0" / "best_calib.pt")
    assert _state(out / "stage_0" / "best_calib.ts")["epoch"] == 0
    assert _same_weights(starts[1], stage0)

    expected = _state(out / f"stage_{best_stage}" / "best_calib.pt")
    assert _state(out / f"stage_{best_stage}" / "best_calib.ts")["epoch"] == best_epoch
    final = _state(out / "final_model.pt")
    assert _same_weights(final, expected)
    final_ts = _state(out / "final_model.ts")
    assert final_ts["epoch"] == best_epoch
    assert final_ts["metrics"]["calib_recall"] == recalls[2 * best_stage + best_epoch]
    assert _same_weights(trainer.model.state_dict(), expected)
    assert not (out / "averaged_model.pt").exists()
    import onnx
    meta = {p.key: p.value for p in onnx.load(str(out / "final_model.onnx")).metadata_props}
    assert float(meta["metric_calib_recall"]) == recalls[2 * best_stage + best_epoch]


def test_cli_multi_stage_final_model_is_best_calib(fake_hub, clips, speech_dir, tmp_path, monkeypatch):
    recalls = [0.5, 1.0, 0.75, 0.5]
    _scripted(monkeypatch, recalls, [0.4] * 4)
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("".join(f"{p},{l}\n" for p, l in clips))
    out = tmp_path / "model"
    result = CliRunner().invoke(train, [
        "--wake-word", "hey_test", "--metadata", str(csv_path), "--test-metadata", str(csv_path),
        "--featurizer-type", "wakehubert", "--batch-size", "4", "--device", "cpu", "--save-best",
        "--output-dir", str(out), "--no-feature-cache", "--mine-sample", "0", "--pca-every", "0",
        "--training-stages", "2:1e-3,2:1e-3", "--calib-speech", speech_dir, "--calib-seconds", "10",
    ])
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert _same_weights(_state(out / "final_model.pt"), _state(out / "stage_0" / "best_calib.pt"))
    assert _state(out / "stage_0" / "best_calib.ts")["epoch"] == 1
