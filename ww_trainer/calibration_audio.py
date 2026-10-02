"""Checkpoint selection on calibration audio: recall at zero false accepts.

The validation split is synthetic like the training set, so its F1 saturates
within a few epochs and stops telling checkpoints apart. A calibration set
built from real speech separates them:

* positives are every validation wake clip mixed with three-talker babble at
  10 dB, again at 5 dB, and with background noise at 5 dB;
* negatives are windows of held-out speech (a LibriSpeech-style folder) and of
  three-talker babble cut from it, at a 0.5 s hop, each window as long as the
  median wake clip.

:func:`calib_recall` is the share of positives scoring above the highest
negative, i.e. the recall the model reaches at a threshold with no false
accept on the negatives. The training loop keeps the checkpoint with the best
``(calib_recall, -validation loss)``.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from ww_trainer.augment import _collect_audio_files, _load_audio_mono, mix_background
from ww_trainer.dataset import AudioDataset, collate_fn
from ww_trainer.model import batch_forward

logger = logging.getLogger(__name__)

HOP_SECONDS = 0.5
BABBLE_TALKERS = 3
BABBLE_SECONDS = 5


def calib_recall(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    """Share of *pos_scores* strictly above the highest of *neg_scores*."""
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    if neg.size == 0:
        raise ValueError("calib_recall needs at least one negative score")
    if pos.size == 0:
        return 0.0
    return float(np.mean(pos > neg.max()))


@dataclass
class CalibrationSet:
    """Calibration waveforms: noisy wake clips and fixed-length negative windows.

    ``negatives`` are views into the loaded speech, so the windows share its
    memory. Features of a frozen featurizer are computed once on first use and
    kept here, never in the :class:`~ww_trainer.feature_store.FeatureStore`.
    """

    positives: List[torch.Tensor]
    negatives: List[torch.Tensor]
    keyword_ids: Optional[list] = None
    _features: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = field(default=None, repr=False)

    def inputs(self, feature_store=None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """``(positives, negatives)`` as the model takes them: waveforms, or stored-path features."""
        if feature_store is None:
            return self.positives, self.negatives
        if self._features is None:
            self._features = ([feature_store.featurize(w) for w in self.positives],
                              [feature_store.featurize(w) for w in self.negatives])
        return self._features


def build_calibration_set(
        wake_samples: List[tuple],
        speech_dir: str,
        noise_dir: Optional[str] = None,
        seconds: float = 3600.0,
        sample_rate: int = 16000,
        seed: int = 0,
) -> CalibrationSet:
    """Calibration audio from *wake_samples* and the speech under *speech_dir*.

    Args:
        wake_samples: Validation wake clips, ``(path, "1"[, keyword_ids])``.
        speech_dir: Folder of held-out speech (flac/wav, searched recursively).
        noise_dir: Background-noise folder; without one the noise-mixed
            positives are left out.
        seconds: Negative audio in total, half speech and half babble.
        sample_rate: Sample rate of the model.
        seed: Seed of every random choice, so a run is reproducible.
    """
    if not wake_samples:
        raise ValueError("calibration needs wake clips in the validation data")
    speech_files = [str(p) for p in _collect_audio_files(speech_dir)]
    if not speech_files:
        raise ValueError(f"no audio files under calibration speech folder {speech_dir}")
    noise_files = [str(p) for p in _collect_audio_files(noise_dir)] if noise_dir else []
    rng = random.Random(seed)

    @lru_cache(maxsize=256)
    def load(path: str) -> np.ndarray:
        return _load_audio_mono(path, sample_rate)

    def crop(path: str, n: int) -> np.ndarray:
        w = load(path)
        if len(w) < n:
            w = np.tile(w, n // max(1, len(w)) + 1)
        start = rng.randint(0, len(w) - n)
        return w[start:start + n]

    def babble(n: int) -> np.ndarray:
        return sum(crop(rng.choice(speech_files), n) for _ in range(BABBLE_TALKERS)) / BABBLE_TALKERS

    loader = AudioDataset(wake_samples, sample_rate=sample_rate)
    clean = [loader._waveform(i, False, False, False).numpy().astype(np.float32)
             for i in range(len(wake_samples))]
    positives = [mix_background(x, babble(len(x)), 10.0) for x in clean]
    positives += [mix_background(x, babble(len(x)), 5.0) for x in clean]
    if noise_files:
        positives += [mix_background(x, crop(rng.choice(noise_files), len(x)), 5.0) for x in clean]
    else:
        logger.warning("[Calibration] no --bg-noise-folder: positives are babble-mixed only")

    window = int(np.median([len(x) for x in clean]))
    hop = int(HOP_SECONDS * sample_rate)
    negatives: List[torch.Tensor] = []
    total = 0
    for path in rng.sample(speech_files, len(speech_files)):
        w = torch.from_numpy(_load_audio_mono(path, sample_rate))
        negatives += [w[i:i + window] for i in range(0, len(w) - window + 1, hop)]
        total += len(w)
        if total >= seconds * sample_rate / 2:
            break
    for _ in range(int(seconds / 2 / BABBLE_SECONDS)):
        b = torch.from_numpy(babble(max(window, BABBLE_SECONDS * sample_rate)).astype(np.float32))
        negatives += [b[i:i + window] for i in range(0, len(b) - window + 1, hop)]
    if not negatives:
        raise ValueError(f"calibration speech under {speech_dir} gave no {window}-sample window")

    kw = wake_samples[0][2] if len(wake_samples[0]) > 2 else None
    logger.info("[Calibration] %d positives, %d negative windows of %.2f s",
                len(positives), len(negatives), window / sample_rate)
    return CalibrationSet([torch.from_numpy(p) for p in positives], negatives, kw)


@torch.no_grad()
def calibration_scores(model, calib: CalibrationSet, device, batch_size: int = 64,
                       feature_store=None) -> Tuple[np.ndarray, np.ndarray]:
    """Logits of the calibration positives and negatives, scored like validation clips."""
    model.eval()

    def score(items: List[torch.Tensor]) -> np.ndarray:
        out = []
        for i in range(0, len(items), batch_size):
            batch = collate_fn([(x, 0, "", calib.keyword_ids) for x in items[i:i + batch_size]],
                               device)
            out.append(batch_forward(model, batch).float().cpu().numpy().ravel())
        return np.concatenate(out) if out else np.zeros(0)

    pos, neg = calib.inputs(feature_store)
    return score(pos), score(neg)
