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


class SharedWaveformCache:
    """In-process shared-memory waveform cache for parallel training runs.

    Pre-loads all audio files as float32 tensors into ``torch`` shared memory
    (``tensor.share_memory_()``).  When the cache object is passed to multiple
    ``torch.multiprocessing``-spawned processes, the OS shares the physical
    memory pages — no duplication.

    Compatible with :class:`FeatureCache` interface: ``.get()`` / ``.put()``.

    Typical use::

        cache = SharedWaveformCache()
        cache.preload(all_paths, sr=16000, max_duration=2.0, n_workers=4)
        # Pass cache to each parallel training process
        trainer.train(..., feature_cache=cache)

    Args:
        sr: Target sample rate (default 16000).
        max_duration: Waveforms longer than this are truncated (seconds).
            Keeps memory bounded; wake-word clips are typically ≤ 2 s.
    """

    def __init__(self, sr: int = 16000, max_duration: float = 2.0) -> None:
        self.sr = sr
        self.max_len = int(sr * max_duration)
        # path → 1-D float32 shared tensor
        self._store: dict[str, "torch.Tensor"] = {}
        self._hits = 0
        self._misses = 0

    def preload(
        self,
        paths: list[str],
        n_workers: int = 1,
        show_progress: bool = True,
    ) -> None:
        """Load all audio files into shared memory.

        Args:
            paths: List of audio file paths (duplicates are deduped).
            n_workers: Thread-pool workers for parallel I/O.
            show_progress: Show tqdm progress bar.
        """
        import concurrent.futures
        import soundfile as sf
        import torch
        from tqdm import tqdm

        unique = list(dict.fromkeys(paths))  # dedup, preserve order
        logger.info("SharedWaveformCache: pre-loading %d files …", len(unique))

        def _load(path: str):
            try:
                data, orig_sr = sf.read(path, dtype="float32", always_2d=False)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if orig_sr != self.sr:
                    import librosa
                    data = librosa.resample(data, orig_sr=orig_sr, target_sr=self.sr)
                if len(data) > self.max_len:
                    data = data[: self.max_len]
                t = torch.from_numpy(data.copy())
                t.share_memory_()
                return path, t
            except Exception as exc:
                logger.debug("SharedWaveformCache: skip %s — %s", path, exc)
                return path, None

        loaded = 0
        if n_workers <= 1:
            it = tqdm(unique, desc="Preloading audio", disable=not show_progress)
            for path in it:
                _, tensor = _load(path)
                if tensor is not None:
                    self._store[path] = tensor
                    loaded += 1
        else:
            failed = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_load, p): p for p in unique}
                it = tqdm(concurrent.futures.as_completed(futures),
                          total=len(futures), desc="Preloading audio", disable=not show_progress)
                for fut in it:
                    path, tensor = fut.result()
                    if tensor is not None:
                        self._store[path] = tensor
                        loaded += 1
                    else:
                        failed.append(path)

            # Retry failures sequentially
            if failed:
                logger.warning("SharedWaveformCache: %d files failed — retrying sequentially", len(failed))
                for path in failed:
                    _, tensor = _load(path)
                    if tensor is not None:
                        self._store[path] = tensor
                        loaded += 1

        total_mb = sum(t.numel() * 4 for t in self._store.values()) / 1e6
        skipped = len(unique) - loaded
        if skipped:
            logger.warning("SharedWaveformCache: %d files could not be loaded and will be read from disk", skipped)
        logger.info(
            "SharedWaveformCache: loaded %d/%d files — %.1f MB in shared memory",
            loaded, len(unique), total_mb,
        )

    def get(self, audio_path: str) -> Optional[np.ndarray]:
        """Return cached waveform as numpy array, or ``None`` on miss.

        Compatible with :class:`FeatureCache`.
        """
        t = self._store.get(audio_path)
        if t is not None:
            self._hits += 1
            return t.numpy()
        self._misses += 1
        return None

    def put(self, audio_path: str, features: np.ndarray) -> None:
        """Store a waveform in shared memory (called on first cache miss).

        Compatible with :class:`FeatureCache`.
        """
        import torch
        t = torch.from_numpy(np.array(features, dtype=np.float32))
        t.share_memory_()
        self._store[audio_path] = t

    def stats(self) -> dict:
        """Return hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "entries": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, total),
        }

    def __len__(self) -> int:
        return len(self._store)


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
