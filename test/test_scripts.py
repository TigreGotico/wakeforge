"""Tests for scripts/dataset_generation/ utilities.

Covers pure-Python and numpy-only helpers that don't require TTS engines,
VAD plugins, or network access.  Heavy CLI entrypoints are tested via
``--help`` subprocess invocations only.
"""
from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# ---------------------------------------------------------------------------
# Helpers to import script modules without executing their __main__ blocks
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts" / "dataset_generation"


def _load_script(name: str):
    """Load a script file as a module without running its if __name__ == '__main__' guard."""
    path = SCRIPTS_DIR / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 01_adversarial_gen.py — GraphemeAugmenter (pure Python)
# ---------------------------------------------------------------------------

class TestGraphemeAugmenter:
    """Tests for the pure-Python GraphemeAugmenter."""

    @pytest.fixture(scope="class")
    def aug(self):
        mod = _load_script("01_adversarial_gen.py")
        return mod.GraphemeAugmenter(min_edits=1, max_edits=2)

    def test_levenshtein_identity(self, aug):
        assert aug._levenshtein_distance("hey", "hey") == 0

    def test_levenshtein_known(self, aug):
        assert aug._levenshtein_distance("hey", "he") == 1
        assert aug._levenshtein_distance("hey", "hew") == 1

    def test_char_class(self, aug):
        assert aug._get_char_class("a") == "vowel"
        assert aug._get_char_class("b") == "consonant"
        assert aug._get_char_class("1") is None

    def test_generate_confusables_count(self, aug):
        results = aug.generate_confusables("hey", n_samples=5)
        # May return fewer if search space is exhausted, but never more
        assert len(results) <= 5
        assert all(isinstance(r, str) for r in results)

    def test_generate_confusables_no_original(self, aug):
        results = aug.generate_confusables("mycroft", n_samples=20)
        assert "mycroft" not in results

    def test_generate_confusables_zero(self, aug):
        assert aug.generate_confusables("hey", n_samples=0) == []

    def test_generate_confusables_levenshtein_bounds(self, aug):
        results = aug.generate_confusables("hey", n_samples=10)
        for r in results:
            d = aug._levenshtein_distance("hey", r)
            assert aug.min_edits <= d <= aug.max_edits, f"{r!r} has distance {d}"

    def test_generate_grapheme_samples(self):
        mod = _load_script("01_adversarial_gen.py")
        aug = mod.GraphemeAugmenter(min_edits=1, max_edits=1)
        results = mod.generate_grapheme_samples(aug, "hey", n_samples=5)
        assert isinstance(results, list)
        assert all(isinstance(r, str) for r in results)


# ---------------------------------------------------------------------------
# 03_training_aug.py — audio utility functions
# ---------------------------------------------------------------------------

class TestTrainingAugUtils:
    """Tests for audio utility functions in 03_training_aug.py."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load_script("03_training_aug.py")

    @pytest.fixture
    def sine_wav(self):
        t = np.linspace(0, 0.5, 8000, dtype=np.float32)
        return (np.sin(2 * math.pi * 440 * t) * 0.5).astype(np.float32)

    def test_load_audio_mono(self, mod, tmp_path, sine_wav):
        p = tmp_path / "test.wav"
        sf.write(str(p), sine_wav, 16000)
        loaded = mod.load_audio_mono(str(p), sr=16000)
        assert loaded is not None
        assert loaded.dtype == np.float32
        assert loaded.shape == sine_wav.shape

    def test_load_audio_mono_missing(self, mod):
        result = mod.load_audio_mono("/nonexistent/path.wav", sr=16000)
        assert result is None

    def test_save_wav(self, mod, tmp_path, sine_wav):
        p = tmp_path / "sub" / "out.wav"
        mod.save_wav(str(p), sine_wav, sr=16000)
        assert p.exists()
        data, sr = sf.read(str(p))
        assert sr == 16000
        np.testing.assert_allclose(data, sine_wav, atol=1e-4)

    def test_mix_audio_shape(self, mod, sine_wav):
        bg = np.random.randn(16000).astype(np.float32) * 0.1
        mixed = mod.mix_audio(sine_wav, bg, snr_db=20.0)
        assert mixed.shape == sine_wav.shape
        assert mixed.dtype == np.float32

    def test_mix_audio_no_clip(self, mod, sine_wav):
        bg = np.random.randn(16000).astype(np.float32) * 10.0  # loud bg
        mixed = mod.mix_audio(sine_wav, bg, snr_db=0.0)
        assert np.max(np.abs(mixed)) <= 1.0 + 1e-6

    def test_apply_reverb_shape(self, mod, sine_wav):
        rir = np.array([1.0, 0.5, 0.25, 0.1], dtype=np.float32)
        reverbed = mod.apply_reverb(sine_wav, rir)
        assert reverbed.shape == sine_wav.shape
        assert reverbed.dtype == np.float32

    def test_collect_audio_files_empty(self, mod, tmp_path):
        result = mod.collect_audio_files(str(tmp_path))
        assert result == []

    def test_collect_audio_files_finds_wavs(self, mod, tmp_path, sine_wav):
        for name in ["a.wav", "b.wav"]:
            sf.write(str(tmp_path / name), sine_wav, 16000)
        result = mod.collect_audio_files(str(tmp_path))
        assert len(result) == 2

    def test_collect_audio_files_none(self, mod):
        assert mod.collect_audio_files(None) == []
