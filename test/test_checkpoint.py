"""Tests for ww_trainer/checkpoint.py — save/load checkpoint helpers."""
import pytest
import torch

from ww_trainer.feats import MfccExtractor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel
from ww_trainer.checkpoint import save_checkpoint, load_checkpoint


def _make_model():
    extractor = MfccExtractor(sr=16000, n_mfcc=13, n_mels=40, n_fft=400, hop_length=160)
    head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")
    return BaseWakeModel(feature_extractor=extractor, classifier=head, device="cpu")


# ---- save_checkpoint ----

def test_save_checkpoint_creates_pt_file(tmp_path):
    model = _make_model()
    out_path = tmp_path / "ckpt"
    save_checkpoint(model, epoch=1, metrics={"auc": 0.9}, optimizer=None, out_path=out_path)
    assert (tmp_path / "ckpt.pt").exists()


def test_save_checkpoint_creates_ts_file(tmp_path):
    model = _make_model()
    out_path = tmp_path / "ckpt"
    save_checkpoint(model, epoch=2, metrics={"f1": 0.8}, optimizer=None, out_path=out_path)
    assert (tmp_path / "ckpt.ts").exists()


def test_save_checkpoint_with_optimizer(tmp_path):
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    out_path = tmp_path / "ckpt"
    # Should not raise
    save_checkpoint(model, epoch=3, metrics={}, optimizer=optimizer, out_path=out_path)
    assert (tmp_path / "ckpt.pt").exists()


def test_save_checkpoint_creates_parent_dirs(tmp_path):
    model = _make_model()
    nested = tmp_path / "deep" / "nested" / "ckpt"
    save_checkpoint(model, epoch=1, metrics={}, optimizer=None, out_path=nested)
    assert (tmp_path / "deep" / "nested" / "ckpt.pt").exists()


# ---- load_checkpoint ----

def test_load_checkpoint_returns_epoch(tmp_path):
    model = _make_model()
    out_path = tmp_path / "ckpt"
    save_checkpoint(model, epoch=7, metrics={"auc": 0.95}, optimizer=None, out_path=out_path)

    model2 = _make_model()
    epoch, metrics = load_checkpoint(model2, path=out_path.with_suffix(".pt"), device="cpu")
    assert epoch == 7


def test_load_checkpoint_returns_metrics(tmp_path):
    model = _make_model()
    out_path = tmp_path / "ckpt"
    save_checkpoint(model, epoch=5, metrics={"auc": 0.88, "f1": 0.77}, optimizer=None, out_path=out_path)

    model2 = _make_model()
    epoch, metrics = load_checkpoint(model2, path=out_path.with_suffix(".pt"), device="cpu")
    assert abs(metrics.get("auc", 0) - 0.88) < 1e-6
    assert abs(metrics.get("f1", 0) - 0.77) < 1e-6


def test_load_checkpoint_raises_for_missing(tmp_path):
    model = _make_model()
    missing = tmp_path / "does_not_exist.pt"
    with pytest.raises(FileNotFoundError):
        load_checkpoint(model, path=missing, device="cpu")


def test_round_trip_epoch(tmp_path):
    """save then load returns same epoch."""
    model = _make_model()
    out_path = tmp_path / "ckpt"
    for ep in [1, 5, 42]:
        save_checkpoint(model, epoch=ep, metrics={}, optimizer=None, out_path=out_path)
        model2 = _make_model()
        loaded_epoch, _ = load_checkpoint(model2, path=out_path.with_suffix(".pt"), device="cpu")
        assert loaded_epoch == ep


def test_load_checkpoint_restores_weights(tmp_path):
    """Loaded weights match saved model."""
    model = _make_model()
    # perturb weights
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.23)
    out_path = tmp_path / "ckpt"
    save_checkpoint(model, epoch=1, metrics={}, optimizer=None, out_path=out_path)

    model2 = _make_model()
    load_checkpoint(model2, path=out_path.with_suffix(".pt"), device="cpu")

    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_load_checkpoint_with_optimizer(tmp_path):
    """load_checkpoint restores optimizer state when .ts exists."""
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Take a step to populate optimizer state
    loss = model(torch.zeros(1, 8000)).sum()
    loss.backward()
    optimizer.step()

    out_path = tmp_path / "ckpt"
    save_checkpoint(model, epoch=3, metrics={}, optimizer=optimizer, out_path=out_path)

    model2 = _make_model()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    epoch, _ = load_checkpoint(model2, path=out_path.with_suffix(".pt"), device="cpu", optimizer=optimizer2)
    assert epoch == 3
