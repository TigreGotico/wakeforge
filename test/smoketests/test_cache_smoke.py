"""Smoke tests: FeatureCache put/get/invalidation with synthetic features."""
import numpy as np
import pytest
import soundfile as sf

from ww_trainer.cache import FeatureCache, make_extractor_params_hash


@pytest.fixture()
def cache(tmp_path) -> FeatureCache:
    """Create a FeatureCache in a temp directory."""
    params_hash = make_extractor_params_hash("MFCC", 13, 16000)
    return FeatureCache(str(tmp_path / "cache"), "MFCC", params_hash)


@pytest.fixture()
def wav_file(tmp_path) -> str:
    """Create a dummy WAV file."""
    path = str(tmp_path / "test.wav")
    sf.write(path, np.random.randn(16000).astype(np.float32), 16000)
    return path


class TestFeatureCache:
    """Cache put/get/invalidation."""

    def test_miss_returns_none(self, cache: FeatureCache, wav_file: str) -> None:
        assert cache.get(wav_file) is None

    def test_put_get_roundtrip(self, cache: FeatureCache, wav_file: str) -> None:
        features = np.random.randn(13, 50).astype(np.float32)
        cache.put(wav_file, features)
        cached = cache.get(wav_file)
        assert cached is not None
        np.testing.assert_array_equal(cached, features)

    def test_size(self, cache: FeatureCache, wav_file: str) -> None:
        assert cache.size() == 0
        cache.put(wav_file, np.zeros(10))
        assert cache.size() == 1

    def test_clear(self, cache: FeatureCache, wav_file: str) -> None:
        cache.put(wav_file, np.zeros(10))
        removed = cache.clear()
        assert removed == 1
        assert cache.size() == 0

    def test_different_extractor_invalidates(self, tmp_path, wav_file: str) -> None:
        """Changing extractor params should miss existing cache entries."""
        hash_a = make_extractor_params_hash("MFCC", 13, 16000)
        hash_b = make_extractor_params_hash("HuBERT", 768, 16000)
        cache_a = FeatureCache(str(tmp_path / "cache"), "MFCC", hash_a)
        cache_b = FeatureCache(str(tmp_path / "cache"), "HuBERT", hash_b)

        features = np.random.randn(10).astype(np.float32)
        cache_a.put(wav_file, features)
        assert cache_a.get(wav_file) is not None
        assert cache_b.get(wav_file) is None  # different extractor → miss
