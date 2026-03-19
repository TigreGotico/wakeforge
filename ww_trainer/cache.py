"""Feature vectorization cache for avoiding redundant feature extraction.

Caches extracted features keyed by MD5 hash of (file content + extractor identity),
so changing the extractor or audio file auto-invalidates stale entries.
"""
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class FeatureCache:
    """Disk-backed cache for extracted audio features.

    Features are stored as ``.npy`` files keyed by a content hash that includes
    the audio file bytes, extractor class name, feature dimension, and sample rate.

    Args:
        cache_dir: Directory for cached ``.npy`` files.
        extractor_name: Name of the feature extractor (e.g. ``"HuBERT"``).
        extractor_params_hash: Hash string summarizing extractor config
            (feature_dim, sample_rate, etc.).
    """

    def __init__(
        self,
        cache_dir: str,
        extractor_name: str,
        extractor_params_hash: str,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.extractor_name = extractor_name
        self.extractor_params_hash = extractor_params_hash

    def _content_hash(self, audio_path: str) -> str:
        """Compute MD5 hash of file content + extractor identity.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Hex digest string.
        """
        h = hashlib.md5(usedforsecurity=False)
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        h.update(self.extractor_name.encode())
        h.update(self.extractor_params_hash.encode())
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        """Return the ``.npy`` path for a given hash key.

        Args:
            key: Content hash string.

        Returns:
            Path to the cached file.
        """
        return self.cache_dir / f"{key}.npy"

    def get(self, audio_path: str) -> Optional[np.ndarray]:
        """Return cached features if available, else ``None``.

        Args:
            audio_path: Path to the source audio file.

        Returns:
            Cached feature array, or ``None`` on miss.
        """
        key = self._content_hash(audio_path)
        p = self._cache_path(key)
        if p.exists():
            try:
                return np.load(p)
            except Exception:
                logger.warning("Corrupt cache entry %s — removing", p)
                p.unlink(missing_ok=True)
        return None

    def put(self, audio_path: str, features: np.ndarray) -> None:
        """Store features in the cache.

        Args:
            audio_path: Path to the source audio file.
            features: Extracted feature array.
        """
        key = self._content_hash(audio_path)
        np.save(self._cache_path(key), features)

    def clear(self) -> int:
        """Remove all cached entries.

        Returns:
            Number of files removed.
        """
        count = 0
        for p in self.cache_dir.glob("*.npy"):
            p.unlink()
            count += 1
        logger.info("FeatureCache: cleared %d entries from %s", count, self.cache_dir)
        return count

    def size(self) -> int:
        """Return the number of cached entries.

        Returns:
            Count of ``.npy`` files in cache directory.
        """
        return sum(1 for _ in self.cache_dir.glob("*.npy"))


def make_extractor_params_hash(
    extractor_name: str,
    feature_dim: int,
    sample_rate: int,
) -> str:
    """Build a deterministic hash string from extractor configuration.

    Args:
        extractor_name: Class name of the feature extractor.
        feature_dim: Output feature dimensionality.
        sample_rate: Audio sample rate.

    Returns:
        Hex digest string.
    """
    h = hashlib.md5(usedforsecurity=False)
    h.update(f"{extractor_name}:{feature_dim}:{sample_rate}".encode())
    return h.hexdigest()
