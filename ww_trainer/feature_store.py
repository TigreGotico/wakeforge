"""Features of a frozen featurizer, computed once per clip and reused.

A frozen featurizer (a pretrained ONNX extractor, or any extractor with no
trainable parameters) gives the same features for the same audio every time,
and for pretrained featurizers it is the expensive part of training. A
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
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ww_trainer.feats import BaseExtractor, OnnxFeatureExtractor

logger = logging.getLogger(__name__)

DEFAULT_MAX_BYTES = 4 * 1024 ** 3


def is_frozen(extractor: BaseExtractor) -> bool:
    """True when *extractor* has no trainable parameters, so its features can be reused."""
    return not any(p.requires_grad for p in extractor.parameters())


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


class FeatureStore:
    """Features ``[T, D]`` of one frozen extractor, keyed by clip and variant.

    Args:
        extractor: The frozen extractor whose features are stored.
        cache_dir: Optional directory for ``.npy`` copies, so later runs and
            other processes reuse them. Only ONNX extractors are persisted.
        max_bytes: Memory budget; beyond it new features are still computed
            (and written to ``cache_dir``) but no longer kept in memory.
    """

    _shared: Dict[Tuple[str, Optional[str]], "FeatureStore"] = {}

    def __init__(self, extractor: BaseExtractor, cache_dir: Optional[str] = None,
                 max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.extractor = extractor
        self.identity = extractor_identity(extractor)
        self.cache_dir = Path(cache_dir) if cache_dir and self.identity else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._memory: Dict[Tuple[str, int], torch.Tensor] = {}
        self._bytes = 0
        self.hits = 0
        self.misses = 0

    @classmethod
    def for_extractor(cls, extractor: BaseExtractor, cache_dir: Optional[str] = None,
                      max_bytes: int = DEFAULT_MAX_BYTES) -> "FeatureStore":
        """The store for *extractor*, shared with every model on the same ONNX file."""
        identity = extractor_identity(extractor)
        if identity is None:
            return cls(extractor, cache_dir, max_bytes)
        key = (identity, str(cache_dir) if cache_dir else None)
        if key not in cls._shared:
            cls._shared[key] = cls(extractor, cache_dir, max_bytes)
        return cls._shared[key]

    def _disk_path(self, path: str, variant: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        h = hashlib.md5(usedforsecurity=False)
        h.update(f"{self.identity}|{_file_identity(path)}|{variant}".encode())
        return self.cache_dir / f"{h.hexdigest()}.npy"

    def get(self, path: str, variant: int = 0) -> Optional[torch.Tensor]:
        """Stored features for *path* and *variant*, or ``None``."""
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
        if self._bytes + size <= self.max_bytes:
            self._memory[(path, variant)] = feats
            self._bytes += size

    @torch.no_grad()
    def featurize(self, wav: torch.Tensor) -> torch.Tensor:
        """Features ``[T, D]`` (float32, CPU) of one 1-D waveform."""
        return self.extractor([wav.to(self.extractor.device)])[0].float().cpu()

    def __len__(self) -> int:
        return len(self._memory)
