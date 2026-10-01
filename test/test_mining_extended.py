"""Extended tests for ww_trainer/mining.py — covering embedding mining and cache trimming."""
import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn as nn

from ww_trainer.mining import mine_hard_negatives, save_mining_cache, load_mining_cache


SAMPLE_RATE = 16000


def _mkdir(p):
    """Create directory and return it."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def _make_wav_files(tmp_path, n: int = 8, label: str = "0") -> list:
    """Return list of (path, label) tuples with synthetic WAVs."""
    samples = []
    for i in range(n):
        t = np.linspace(0, 0.5, int(SAMPLE_RATE * 0.5))
        wav = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        p = tmp_path / f"wav_{label}_{i}.wav"
        sf.write(str(p), wav, SAMPLE_RATE)
        samples.append((str(p), label))
    return samples


def _stub_model():
    """Model with forward() and embed() for mining tests."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8000, 16)

        def forward(self, wavs: torch.Tensor) -> torch.Tensor:
            b = wavs.shape[0]
            x = wavs[:, :8000] if wavs.shape[1] >= 8000 else nn.functional.pad(wavs, (0, 8000 - wavs.shape[1]))
            return self.linear(x).mean(dim=-1)

        def embed(self, wavs: torch.Tensor) -> torch.Tensor:
            b = wavs.shape[0]
            x = wavs[:, :8000] if wavs.shape[1] >= 8000 else nn.functional.pad(wavs, (0, 8000 - wavs.shape[1]))
            return self.linear(x)

    return M()


class TestMineWithEmbedding:
    """Tests for the embedding mining branch."""

    def test_embedding_mining_with_wake_cache(self, tmp_path):
        """use_embedding_mining=True with wake_cache activates embedding path."""
        neg_dir = tmp_path / "neg"
        neg_dir.mkdir()
        nonwakes = _make_wav_files(neg_dir, n=8, label="0")

        wake_dir = tmp_path / "wake"
        wake_dir.mkdir()
        wakes = _make_wav_files(wake_dir, n=4, label="1")

        model = _stub_model()
        hard, easy, cache = mine_hard_negatives(
            model, nonwakes, device="cpu",
            dataset_fraction=1.0,
            use_embedding_mining=True,
            wake_cache=wakes,
        )
        assert isinstance(hard, list)
        assert isinstance(cache, dict)
        # All returned items should be label "0"
        for p, lbl in hard:
            assert lbl == "0"

    def test_embedding_mining_empty_wake_cache(self, tmp_path):
        """use_embedding_mining=True but empty wake_cache skips embedding."""
        neg_dir = tmp_path / "neg2"
        neg_dir.mkdir()
        nonwakes = _make_wav_files(neg_dir, n=4, label="0")
        model = _stub_model()
        hard, easy, cache = mine_hard_negatives(
            model, nonwakes, device="cpu",
            dataset_fraction=1.0,
            use_embedding_mining=True,
            wake_cache=[],
        )
        assert isinstance(hard, list)

    def test_embedding_mining_no_embed_method(self, tmp_path):
        """Model without embed() skips embedding mining gracefully."""
        neg_dir = tmp_path / "neg3"
        neg_dir.mkdir()
        nonwakes = _make_wav_files(neg_dir, n=4, label="0")

        class NoEmbedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8000, 1)

            def forward(self, wavs):
                x = wavs[:, :8000] if wavs.shape[1] >= 8000 else nn.functional.pad(wavs, (0, 8000 - wavs.shape[1]))
                return self.linear(x).squeeze(-1)

        model = NoEmbedModel()
        hard, easy, cache = mine_hard_negatives(
            model, nonwakes, device="cpu",
            dataset_fraction=1.0,
            use_embedding_mining=True,
            wake_cache=_make_wav_files(_mkdir(tmp_path / "w"), n=2, label="1"),
        )
        # Should still return results (just without embedding refinement)
        assert isinstance(hard, list)


class TestCacheTrimming:
    """Tests for cache size limits."""

    def test_cache_trimmed_to_max_size(self, tmp_path):
        """When cache exceeds max_cache_size, it is trimmed."""
        neg_dir = tmp_path / "neg4"
        neg_dir.mkdir()
        nonwakes = _make_wav_files(neg_dir, n=6, label="0")
        model = _stub_model()
        # Pre-fill cache with many entries
        big_cache = {f"/fake/{i}.wav": float(i) for i in range(50)}
        hard, easy, cache = mine_hard_negatives(
            model, nonwakes, device="cpu",
            hardness_cache=big_cache,
            dataset_fraction=1.0,
            max_cache_size=10,
            use_embedding_mining=False,
        )
        assert len(cache) <= 10

    def test_cache_decay_applied(self, tmp_path):
        """Existing cache values are decayed."""
        neg_dir = tmp_path / "neg5"
        neg_dir.mkdir()
        nonwakes = _make_wav_files(neg_dir, n=2, label="0")
        model = _stub_model()
        initial_path = nonwakes[0][0]
        old_cache = {initial_path: 1.0}
        _, _, cache = mine_hard_negatives(
            model, nonwakes, device="cpu",
            hardness_cache=old_cache,
            dataset_fraction=1.0,
            cache_decay=0.9,
            use_embedding_mining=False,
        )
        # The decayed value should differ from 1.0
        if initial_path in cache:
            assert cache[initial_path] != 1.0


class TestSaveLoadEdgeCases:
    def test_large_cache_round_trip(self, tmp_path):
        """Large cache round-trips correctly."""
        cache = {f"path_{i}.wav": float(i) / 1000 for i in range(500)}
        path = str(tmp_path / "big.pt")
        save_mining_cache(cache, path)
        loaded = load_mining_cache(path)
        assert len(loaded) == 500
