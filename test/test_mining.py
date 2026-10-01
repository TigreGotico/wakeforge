"""Tests for ww_trainer/mining.py — hard-negative mining helpers."""
import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn as nn

from ww_trainer.mining import mine_hard_negatives, save_mining_cache, load_mining_cache


# ---- helpers ----

def _make_wav_files(tmp_path, n: int = 4, sample_rate: int = 16000) -> list:
    """Return list of (path, "0") tuples with synthetic sine WAVs."""
    samples = []
    for i in range(n):
        t = np.linspace(0, 0.5, int(sample_rate * 0.5))
        wav = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        p = tmp_path / f"neg_{i}.wav"
        sf.write(str(p), wav, sample_rate)
        samples.append((str(p), "0"))
    return samples


def _make_stub_model():
    """Minimal model satisfying mine_hard_negatives interface."""

    class StubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8000, 16)

        def forward(self, wavs: torch.Tensor) -> torch.Tensor:
            # Accept [B, T] or shorter; pool to fixed size
            b = wavs.shape[0]
            x = wavs[:, :8000] if wavs.shape[1] >= 8000 else torch.nn.functional.pad(wavs, (0, 8000 - wavs.shape[1]))
            return self.linear(x).mean(dim=-1)

        def embed(self, wavs: torch.Tensor) -> torch.Tensor:
            b = wavs.shape[0]
            x = wavs[:, :8000] if wavs.shape[1] >= 8000 else torch.nn.functional.pad(wavs, (0, 8000 - wavs.shape[1]))
            return self.linear(x)

    return StubModel()


# ---- save_mining_cache / load_mining_cache ----

def test_save_mining_cache_creates_file(tmp_path):
    cache = {"a.wav": 0.8, "b.wav": 0.2}
    path = str(tmp_path / "cache.pt")
    save_mining_cache(cache, path)
    assert (tmp_path / "cache.pt").exists()


def test_load_mining_cache_missing_returns_empty(tmp_path):
    path = str(tmp_path / "nonexistent.pt")
    result = load_mining_cache(path)
    assert result == {}


def test_load_mining_cache_round_trip(tmp_path):
    cache = {"x.wav": 0.9, "y.wav": 0.1, "z.wav": 0.5}
    path = str(tmp_path / "cache.pt")
    save_mining_cache(cache, path)
    loaded = load_mining_cache(path)
    assert set(loaded.keys()) == set(cache.keys())
    for k in cache:
        assert abs(loaded[k] - cache[k]) < 1e-5


def test_load_mining_cache_empty_dict(tmp_path):
    cache = {}
    path = str(tmp_path / "empty.pt")
    save_mining_cache(cache, path)
    loaded = load_mining_cache(path)
    assert loaded == {}


# ---- mine_hard_negatives ----

def test_mine_hard_negatives_disabled_returns_all_as_easy(tmp_path):
    """dataset_fraction=0.0 → disabled path → all samples in easy, none in hard."""
    nonwakes = _make_wav_files(tmp_path, n=4)
    model = _make_stub_model()
    hard, easy, cache = mine_hard_negatives(
        model, nonwakes, device="cpu",
        dataset_fraction=0.0,
    )
    assert hard == []
    assert easy == nonwakes


def test_mine_hard_negatives_empty_nonwakes(tmp_path):
    """Empty nonwake list → all empty returns."""
    model = _make_stub_model()
    hard, easy, cache = mine_hard_negatives(
        model, [], device="cpu", dataset_fraction=0.5,
    )
    assert hard == []
    assert easy == []
    assert isinstance(cache, dict)


def test_mine_hard_negatives_returns_tuples(tmp_path):
    """Returned hard/easy items are (path, label) tuples."""
    nonwakes = _make_wav_files(tmp_path, n=6)
    model = _make_stub_model()
    hard, easy, cache = mine_hard_negatives(
        model, nonwakes, device="cpu",
        dataset_fraction=0.5,
        use_embedding_mining=False,
    )
    for pair in hard + easy:
        assert isinstance(pair, tuple) and len(pair) == 2
        assert pair[1] == "0"


def test_mine_hard_negatives_cache_updated(tmp_path):
    """Returned cache dict has keys for processed paths."""
    nonwakes = _make_wav_files(tmp_path, n=4)
    model = _make_stub_model()
    hard, easy, cache = mine_hard_negatives(
        model, nonwakes, device="cpu",
        dataset_fraction=1.0,
        use_embedding_mining=False,
    )
    assert isinstance(cache, dict)
    assert len(cache) > 0


def test_mine_hard_negatives_with_existing_cache(tmp_path):
    """Passing a pre-populated hardness_cache works without error."""
    nonwakes = _make_wav_files(tmp_path, n=4)
    model = _make_stub_model()
    existing_cache = {str(nonwakes[0][0]): 0.7}
    hard, easy, updated = mine_hard_negatives(
        model, nonwakes, device="cpu",
        hardness_cache=existing_cache,
        dataset_fraction=0.5,
        use_embedding_mining=False,
    )
    assert isinstance(updated, dict)
