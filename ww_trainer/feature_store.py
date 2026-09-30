"""Features of a fixed featurizer, computed once per clip and reused.

A fixed featurizer (a pretrained ONNX extractor, or any extractor with no
parameters) gives the same features for the same audio every time, and for
pretrained featurizers it is the expensive part of training. A
:class:`FeatureStore` keeps those features keyed by clip and variant:

* variant ``0`` is the clip as recorded, used by evaluation, mining and every
  un-augmented training draw;
* variants ``1..K`` are augmented renditions of the clip. When a training draw
  is augmented, :class:`~ww_trainer.dataset.AudioDataset` picks one of the
  ``K`` variants and computes it (waveform augmentation, then the featurizer)
  only the first time it is drawn. With ``K = 0`` augmented draws are computed
  on the fly and never stored, which keeps augmentation continuous.

Stores are shared by every model built on the same featurizer file in one
process (the heads of a sweep or genetic search reuse one another's features),
and a ``cache_dir`` persists ONNX features to disk across processes and runs.
All live stores in a process share one memory budget,
:attr:`FeatureStore.max_bytes`: a store that needs room takes it from the
stores used least recently, so a store left over from an earlier run never
blocks a later one. :meth:`FeatureStore.close` frees a store and
:meth:`FeatureStore.close_all` frees every one.

:func:`store_for` decides whether a model gets a store: by default only ONNX
featurizers do, since built-in extractors (MFCC, filterbank, ...) are cheap
and training on them keeps waveform-domain augmentation (waveform Mixup,
RPPL's waveform-augmented view); those use a store only when asked.
"""
from __future__ import annotations

import hashlib
import logging
import itertools
import os
import weakref
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ww_trainer.feats import BaseExtractor, OnnxFeatureExtractor

logger = logging.getLogger(__name__)

def is_fixed(extractor: BaseExtractor) -> bool:
    """True when *extractor* has no parameters at all, so its features never change.

    An extractor with parameters is never fixed, even while frozen: it may be
    unfrozen later (``unfreeze_at_epoch``) and must then run on every batch.
    """
    return next(extractor.parameters(), None) is None


def _file_identity(path: str) -> str:
    st = os.stat(path)
    return f"{os.path.realpath(path)}:{st.st_size}:{st.st_mtime_ns}"


def extractor_identity(extractor: BaseExtractor) -> Optional[str]:
    """Identity of an ONNX featurizer's graph file, or ``None`` for other extractors.

    Only an ONNX graph file identifies an extractor's output exactly; features
    of other extractors are kept per model and never written to disk.
    """
    if type(extractor) is OnnxFeatureExtractor:
        return f"onnx:{_file_identity(extractor.model_path)}:{extractor.sample_rate}"
    return None


def store_for(extractor: BaseExtractor, requested: Optional[bool] = None,
              cache_dir: Optional[str] = None,
              unfreeze_at_epoch: Optional[int] = None) -> Optional["FeatureStore"]:
    """The :class:`FeatureStore` a model trains from, or ``None`` to featurize every batch.

    Args:
        extractor: The model's extractor.
        requested: ``None`` uses a store for ONNX featurizers only; ``True``
            for any fixed extractor; ``False`` never.
        cache_dir: Directory where ONNX features persist.
        unfreeze_at_epoch: Progressive unfreezing retrains the extractor, so
            no store is used when it is set.
    """
    if requested is False or unfreeze_at_epoch is not None or not is_fixed(extractor):
        return None
    if requested is None and extractor_identity(extractor) is None:
        return None
    return FeatureStore.for_extractor(extractor, cache_dir)


class FeatureStore:
    """Features ``[T, D]`` of one frozen extractor, keyed by clip and variant.

    Args:
        extractor: The frozen extractor whose features are stored.
        cache_dir: Optional directory for ``.npy`` copies, so later runs and
            other processes reuse them. Only ONNX extractors are persisted.

    Features are kept in memory while all live stores together hold less than
    :attr:`max_bytes`. When a new feature does not fit, the stores used least
    recently are cleared to make room; if it still does not fit, it is
    computed (and written to ``cache_dir``) but not kept.
    """

    max_bytes: int = 4 * 1024 ** 3
    _shared: Dict[Tuple[str, Optional[str]], "FeatureStore"] = {}
    _live: "weakref.WeakSet[FeatureStore]" = weakref.WeakSet()
    _clock = itertools.count()

    def __init__(self, extractor: BaseExtractor, cache_dir: Optional[str] = None) -> None:
        self.extractor = extractor
        self.identity = extractor_identity(extractor)
        self.cache_dir = Path(cache_dir) if cache_dir and self.identity else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory: Dict[Tuple[str, int], torch.Tensor] = {}
        self.nbytes = 0
        self._last_used = next(FeatureStore._clock)
        FeatureStore._live.add(self)
        self.hits = 0
        self.misses = 0

    @classmethod
    def for_extractor(cls, extractor: BaseExtractor,
                      cache_dir: Optional[str] = None) -> "FeatureStore":
        """The store for *extractor*, shared with every model on the same ONNX file."""
        identity = extractor_identity(extractor)
        if identity is None:
            return cls(extractor, cache_dir)
        key = (identity, str(cache_dir) if cache_dir else None)
        if key not in cls._shared:
            cls._shared[key] = cls(extractor, cache_dir)
        return cls._shared[key]

    def _disk_path(self, path: str, variant: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        h = hashlib.md5(usedforsecurity=False)
        h.update(f"{self.identity}|{_file_identity(path)}|{variant}".encode())
        return self.cache_dir / f"{h.hexdigest()}.npy"

    @classmethod
    def used_bytes(cls) -> int:
        """Bytes held in memory by all live stores."""
        return sum(store.nbytes for store in list(cls._live))

    @classmethod
    def close_all(cls) -> None:
        """Clear every live store and forget the shared ones."""
        for store in list(cls._live):
            store.close()
        cls._shared.clear()

    def close(self) -> None:
        """Drop this store's features from memory (files in ``cache_dir`` stay).

        The store stays usable and refills as features are computed again.
        """
        self._memory.clear()
        self.nbytes = 0

    def get(self, path: str, variant: int = 0) -> Optional[torch.Tensor]:
        """Stored features for *path* and *variant*, or ``None``."""
        self._last_used = next(FeatureStore._clock)
        feats = self._memory.get((path, variant))
        if feats is None:
            disk = self._disk_path(path, variant)
            if disk is not None and disk.exists():
                try:
                    feats = torch.from_numpy(np.load(disk))
                except (OSError, ValueError):
                    logger.warning("Corrupt feature cache entry %s — removing", disk)
                    disk.unlink(missing_ok=True)
                else:
                    self._remember(path, variant, feats)
        if feats is None:
            self.misses += 1
        else:
            self.hits += 1
        return feats

    def put(self, path: str, variant: int, feats: torch.Tensor) -> None:
        """Store *feats* for *path* and *variant*."""
        self._remember(path, variant, feats)
        disk = self._disk_path(path, variant)
        if disk is not None:
            np.save(disk, feats.numpy())

    def _remember(self, path: str, variant: int, feats: torch.Tensor) -> None:
        size = feats.numel() * feats.element_size()
        self._last_used = next(FeatureStore._clock)
        used = FeatureStore.used_bytes()
        if used + size > FeatureStore.max_bytes:
            stale = sorted((s for s in list(FeatureStore._live) if s is not self and s.nbytes),
                           key=lambda s: s._last_used)
            for store in stale:
                used -= store.nbytes
                store.close()
                if used + size <= FeatureStore.max_bytes:
                    break
        if used + size <= FeatureStore.max_bytes:
            self._memory[(path, variant)] = feats
            self.nbytes += size

    @torch.no_grad()
    def featurize(self, wav: torch.Tensor) -> torch.Tensor:
        """Features ``[T, D]`` (float32, CPU) of one 1-D waveform."""
        return self.extractor([wav.to(self.extractor.device)])[0].float().cpu()

    def __len__(self) -> int:
        return len(self._memory)
