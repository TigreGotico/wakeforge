"""Tests for AudioDataset and collate_fn."""
import numpy as np
import pytest
import soundfile as sf
import torch


def _make_samples(tmp_path, n: int = 6, sample_rate: int = 16000) -> list:
    """Return list of (path, label) tuples with synthetic sine WAVs."""
    samples = []
    for i in range(n):
        freq = 440 * (i + 1)
        t = np.linspace(0, 0.5, int(sample_rate * 0.5))
        wav = np.sin(2 * np.pi * freq * t).astype(np.float32)
        p = tmp_path / f"audio_{i}.wav"
        sf.write(str(p), wav, sample_rate)
        label = str(i % 2)  # alternating 0/1
        samples.append((str(p), label))
    return samples


def test_dataset_len(tmp_path):
    from ww_trainer.dataset import AudioDataset
    samples = _make_samples(tmp_path, n=6)
    ds = AudioDataset(samples, aug_prob=0.0)
    assert len(ds) == 6


def test_dataset_getitem(tmp_path):
    from ww_trainer.dataset import AudioDataset
    samples = _make_samples(tmp_path, n=4)
    ds = AudioDataset(samples, aug_prob=0.0)
    wav, label, path, kw_ids = ds[0]
    assert isinstance(wav, torch.Tensor)
    assert wav.ndim == 1
    assert isinstance(label, int)
    assert label in (0, 1)
    assert isinstance(path, str)


def test_dataset_getitem_all_labels(tmp_path):
    from ww_trainer.dataset import AudioDataset
    samples = _make_samples(tmp_path, n=6)
    ds = AudioDataset(samples, aug_prob=0.0)
    labels_seen = set()
    for i in range(len(ds)):
        _, label, _, _ = ds[i]
        labels_seen.add(label)
    assert labels_seen == {0, 1}


def test_collate_fn_pads_to_max_length(tmp_path):
    from ww_trainer.dataset import AudioDataset, collate_fn
    samples = _make_samples(tmp_path, n=4, sample_rate=16000)
    ds = AudioDataset(samples, aug_prob=0.0)
    batch = [ds[i] for i in range(4)]
    padded, labels_t, paths, kw_ids = collate_fn(batch, device="cpu")
    max_len = max(ds[i][0].shape[-1] for i in range(4))
    assert padded.shape == (4, max_len)
    assert labels_t.shape == (4,)
    assert len(paths) == 4
    assert kw_ids is None  # no keyword IDs in 3-tuple batch


def test_collate_fn_different_lengths(tmp_path):
    """Tensors of varying length are padded to the longest."""
    from ww_trainer.dataset import collate_fn
    wav_short = torch.zeros(800)
    wav_long = torch.zeros(1600)
    batch = [(wav_short, 0, "a"), (wav_long, 1, "b")]
    padded, labels_t, _, kw_ids = collate_fn(batch, device="cpu")
    assert padded.shape == (2, 1600)
    assert padded[0, 800:].sum() == 0.0  # padding is zero
    assert kw_ids is None


def test_mix_background_peak():
    """_mix_background output peak must be ≤ 1.0."""
    from ww_trainer.dataset import _mix_background
    rng = np.random.default_rng(0)
    clean = rng.uniform(-0.5, 0.5, 8000).astype(np.float32)
    bg = rng.uniform(-1.0, 1.0, 8000).astype(np.float32)
    for snr in [0, 5, 10, 20]:
        mixed = _mix_background(clean, bg, snr)
        assert np.max(np.abs(mixed)) <= 1.0 + 1e-6, f"Peak exceeded 1.0 at SNR={snr}"


def test_apply_reverb_length():
    """_apply_reverb output length matches input length."""
    from ww_trainer.dataset import _apply_reverb
    wav = np.random.randn(8000).astype(np.float32)
    rir = np.random.randn(200).astype(np.float32)
    out = _apply_reverb(wav, rir)
    assert len(out) == len(wav)


def test_get_augmented_returns_tensor(tmp_path):
    from ww_trainer.dataset import AudioDataset
    samples = _make_samples(tmp_path, n=2)
    ds = AudioDataset(samples, aug_prob=1.0)
    wav_t = torch.randn(8000)
    aug = ds.get_augmented(wav_t)
    assert isinstance(aug, torch.Tensor)
    assert aug.ndim == 1
