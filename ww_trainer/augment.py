"""Composable audio augmentation transforms for wake word training.

Each transform is a callable ``np.ndarray → np.ndarray``.
Combine them with :class:`AugmentationPipeline` to build
randomized augmentation chains.

Usage::

    from ww_trainer.augment import (
        AugmentationPipeline, MixBackground, ApplyReverb,
        PitchShift, SpeedPerturb, GaussianNoise, VolumePerturb,
        SpecAugment, TimeShift, Normalize,
    )

    pipeline = AugmentationPipeline([
        (MixBackground("/path/to/noise/"), 0.6),
        (ApplyReverb("/path/to/rirs/"), 0.3),
        (PitchShift(), 0.3),
        (SpeedPerturb(), 0.3),
        (GaussianNoise(), 0.5),
        (VolumePerturb(), 0.5),
        (TimeShift(), 0.3),
        (Normalize(), 1.0),
    ])

    augmented_wav = pipeline(wav_np)  # np.ndarray → np.ndarray
"""
from __future__ import annotations

import abc
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _collect_audio_files(base_folder: str) -> list[Path]:
    """Collect all valid audio files from a folder recursively.

    Args:
        base_folder: Directory to scan.

    Returns:
        Sorted list of Path objects for audio files.
    """
    exts = [".wav", ".flac", ".mp3", ".m4a", ".ogg"]
    p = Path(base_folder)
    if not p.exists():
        logger.warning("Augmentation folder not found: %s", base_folder)
        return []
    files: list[Path] = []
    for ext in exts:
        files.extend(p.rglob(f"*{ext}"))
    return sorted(files)


def _load_audio_mono(path: str | Path, sr: int = 16000) -> np.ndarray:
    """Load audio file as mono float32 at target sample rate.

    Args:
        path: Audio file path.
        sr: Target sample rate.

    Returns:
        1-D float32 numpy array.
    """
    import soundfile as sf

    wav, orig_sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if orig_sr != sr:
        import librosa
        wav = librosa.resample(wav.astype(np.float32), orig_sr=orig_sr, target_sr=sr)
    return wav.astype(np.float32)


class AudioTransform(abc.ABC):
    """Base class for audio augmentation transforms."""

    @abc.abstractmethod
    def __call__(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Apply transform to audio waveform.

        Args:
            wav: 1-D float32 numpy array.
            sr: Sample rate.

        Returns:
            Augmented 1-D float32 numpy array.
        """
        ...


class MixBackground(AudioTransform):
    """Mix background audio at a random SNR.

    Args:
        folder: Path to folder of background audio files.
        snr_min: Minimum SNR in dB.
        snr_max: Maximum SNR in dB.
        sr: Expected sample rate.
    """

    def __init__(self, folder: str, snr_min: float = 0.0, snr_max: float = 20.0,
                 sr: int = 16000) -> None:
        self.files = _collect_audio_files(folder)
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.sr = sr

    def __call__(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        if not self.files:
            return wav
        bg_path = random.choice(self.files)
        bg = _load_audio_mono(bg_path, sr)
        snr = random.uniform(self.snr_min, self.snr_max)
        return mix_background(wav, bg, snr)


class ApplyReverb(AudioTransform):
    """Convolve with a random Room Impulse Response.

    Args:
        folder: Path to folder of RIR audio files.
        attenuation: Gain factor after convolution (default 0.5).
    """

    def __init__(self, folder: str, attenuation: float = 0.5) -> None:
        self.files = _collect_audio_files(folder)
        self.attenuation = attenuation

    def __call__(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        if not self.files:
            return wav
        rir_path = random.choice(self.files)
        rir = _load_audio_mono(rir_path, sr)
        return apply_reverb(wav, rir, self.attenuation)


class PitchShift(AudioTransform):
    """Random pitch shift in semitones.

    Args:
        min_steps: Minimum semitones (default -1).
        max_steps: Maximum semitones (default 1).
    """

    def __init__(self, min_steps: float = -1.0, max_steps: float = 1.0) -> None:
        self.min_steps = min_steps
        self.max_steps = max_steps

    def __call__(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        import librosa
        n_steps = random.uniform(self.min_steps, self.max_steps)
        return librosa.effects.pitch_shift(wav, sr=sr, n_steps=n_steps).astype(np.float32)


class SpeedPerturb(AudioTransform):
    """Random speed perturbation (time stretch without pitch change).

    Args:
        min_rate: Minimum speed factor (default 0.95).
        max_rate: Maximum speed factor (default 1.05).
    """

    def __init__(self, min_rate: float = 0.95, max_rate: float = 1.05) -> None:
        self.min_rate = min_rate
        self.max_rate = max_rate

    def __call__(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        import librosa
        rate = random.uniform(self.min_rate, self.max_rate)
        return librosa.effects.time_stretch(wav, rate=rate).astype(np.float32)


class GaussianNoise(AudioTransform):
    """Add Gaussian white noise at a random SNR.

    Args:
        snr_min: Minimum SNR in dB (default 10).
        snr_max: Maximum SNR in dB (default 40).
    """

    def __init__(self, snr_min: float = 10.0, snr_max: float = 40.0) -> None:
        self.snr_min = snr_min
        self.snr_max = snr_max

    def __call__(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        snr_db = random.uniform(self.snr_min, self.snr_max)
        rms_signal = np.sqrt(np.mean(wav ** 2) + 1e-9)
        rms_noise = rms_signal / (10 ** (snr_db / 20.0))
        noise = np.random.randn(len(wav)).astype(np.float32) * rms_noise
        return (wav + noise).astype(np.float32)


class VolumePerturb(AudioTransform):
    """Random volume scaling.

    Args:
        min_gain_db: Minimum gain in dB (default -6).
        max_gain_db: Maximum gain in dB (default 6).
    """

    def __init__(self, min_gain_db: float = -6.0, max_gain_db: float = 6.0) -> None:
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

    def __call__(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        gain_db = random.uniform(self.min_gain_db, self.max_gain_db)
        gain = 10 ** (gain_db / 20.0)
        return (wav * gain).astype(np.float32)


class TimeShift(AudioTransform):
    """Randomly shift audio forward/backward in time with zero padding.

    Args:
        max_shift_frac: Maximum shift as fraction of audio length (default 0.1 = 10%).
    """

    def __init__(self, max_shift_frac: float = 0.1) -> None:
        self.max_shift_frac = max_shift_frac

    def __call__(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        shift = int(random.uniform(-self.max_shift_frac, self.max_shift_frac) * len(wav))
        if shift > 0:
            return np.concatenate([np.zeros(shift, dtype=np.float32), wav[:-shift]])
        elif shift < 0:
            return np.concatenate([wav[-shift:], np.zeros(-shift, dtype=np.float32)])
        return wav


class SpecAugment(AudioTransform):
    """SpecAugment-style augmentation applied in waveform domain.

    Randomly masks time segments with silence. For frequency masking,
    apply after feature extraction instead.

    Args:
        max_mask_frac: Maximum fraction of audio to mask (default 0.1).
        n_masks: Number of masks to apply (default 2).
    """

    def __init__(self, max_mask_frac: float = 0.1, n_masks: int = 2) -> None:
        self.max_mask_frac = max_mask_frac
        self.n_masks = n_masks

    def __call__(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        result = wav.copy()
        T = len(wav)
        max_len = int(T * self.max_mask_frac)
        for _ in range(self.n_masks):
            mask_len = random.randint(1, max(1, max_len))
            start = random.randint(0, max(0, T - mask_len))
            result[start:start + mask_len] = 0.0
        return result


class Normalize(AudioTransform):
    """Peak-normalize audio to [-1, 1].

    Args:
        target_peak: Target peak amplitude (default 1.0).
    """

    def __init__(self, target_peak: float = 1.0) -> None:
        self.target_peak = target_peak

    def __call__(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        peak = np.max(np.abs(wav))
        if peak > 1e-9:
            return (wav / peak * self.target_peak).astype(np.float32)
        return wav


class AugmentationPipeline(AudioTransform):
    """Chain multiple transforms, each applied with an independent probability.

    Args:
        transforms: List of ``(transform, probability)`` tuples.

    Example::

        pipeline = AugmentationPipeline([
            (GaussianNoise(snr_min=10), 0.5),
            (PitchShift(), 0.3),
            (Normalize(), 1.0),
        ])
        augmented = pipeline(wav_np)
    """

    def __init__(self, transforms: list[tuple[AudioTransform, float]]) -> None:
        self.transforms = transforms

    def __call__(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        for transform, prob in self.transforms:
            if random.random() < prob:
                wav = transform(wav, sr)
        return wav


# --- Standalone utility functions (used by dataset.py and transforms) ---


def mix_background(clean: np.ndarray, bg: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix clean audio with background at specified SNR.

    Args:
        clean: Clean audio waveform.
        bg: Background audio (auto-tiled if shorter).
        snr_db: Signal-to-noise ratio in dB.

    Returns:
        Mixed audio, peak-normalized to avoid clipping.
    """
    clean_len = len(clean)
    if len(bg) < clean_len:
        bg = np.tile(bg, int(np.ceil(clean_len / len(bg))))
    start = random.randint(0, len(bg) - clean_len)
    bg = bg[start:start + clean_len]

    rms_clean = np.sqrt(np.mean(clean ** 2) + 1e-9)
    rms_bg = np.sqrt(np.mean(bg ** 2) + 1e-9)
    desired_bg_rms = rms_clean / (10 ** (snr_db / 20.0))
    if rms_bg > 0:
        bg = bg * (desired_bg_rms / rms_bg)
    mixed = clean + bg
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def apply_reverb(wav: np.ndarray, rir: np.ndarray,
                 attenuation: float = 0.5) -> np.ndarray:
    """Apply room impulse response convolution.

    Args:
        wav: Input waveform.
        rir: Room impulse response.
        attenuation: Gain factor after convolution.

    Returns:
        Reverberated audio with matched RMS.
    """
    out = np.convolve(wav, rir)[:len(wav)]
    rms_wav = np.sqrt(np.mean(wav ** 2) + 1e-9)
    rms_out = np.sqrt(np.mean(out ** 2) + 1e-9)
    out = out * (rms_wav / rms_out) * attenuation
    return out.astype(np.float32)
