import logging
import random
import tempfile
from typing import Optional, Union

import numpy
import numpy as np
import torch
import torchaudio

from torch.utils.data import Dataset

from ww_trainer.utils import timed
from ww_trainer.augment import (
    _collect_audio_files,
    _load_audio_mono,
    mix_background as _mix_background,
    apply_reverb as _apply_reverb,
)

logger = logging.getLogger(__name__)



class AudioDataset(Dataset):
    """PyTorch Dataset for loading audio files with optional on-the-fly augmentation.

    Supports two augmentation modes:

    1. **Pipeline mode** (recommended): pass an ``AugmentationPipeline`` via
       the ``pipeline`` parameter.  Composable, testable, and decoupled.
    2. **Legacy mode**: pass individual folder paths (``bg_noise_folder``,
       ``rir_folder``, etc.).  Maintained for backward compatibility.

    When ``pipeline`` is provided, it takes priority over legacy kwargs.

    Args:
        samples: List of ``(audio_path, label_str)`` tuples.
        sample_rate: Target sample rate.
        aug_prob: Probability of applying augmentation per sample (0 disables).
        pipeline: Optional ``AugmentationPipeline`` for composable augmentation.
        vc_prob: Voice conversion probability (legacy mode only).
        bg_noise_folder: Background noise folder (legacy mode).
        music_folder: Music folder (legacy mode).
        bg_speech_folder: Background speech folder (legacy mode).
        mic_noise_folder: Microphone noise folder (legacy mode).
        rir_folder: Room impulse response folder (legacy mode).
        vc_folder: Voice conversion reference folder (legacy mode).
        snr_min: Minimum SNR for noise mixing (legacy mode).
        snr_max: Maximum SNR for noise mixing (legacy mode).
        pitch_min: Minimum pitch shift in semitones (legacy mode).
        pitch_max: Maximum pitch shift in semitones (legacy mode).
        speed_min: Minimum speed factor (legacy mode).
        speed_max: Maximum speed factor (legacy mode).
        device: Device for voice conversion (legacy mode).
    """

    def __init__(self, samples,
                 sample_rate: int = 16000,
                 aug_prob: float = 0.0,
                 pipeline=None,
                 vc_prob: float = 0.3,
                 bg_noise_folder: str = None,
                 music_folder: str = None,
                 bg_speech_folder: str = None,
                 mic_noise_folder: str = None,
                 rir_folder: str = None,
                 vc_folder: str = None,
                 snr_min: float = 0.0,
                 snr_max: float = 20.0,
                 pitch_min: float = -1.0,
                 pitch_max: float = 1.0,
                 speed_min: float = 0.95,
                 speed_max: float = 1.05,
                 device="auto"
                 ):
        self.pipeline = pipeline
        self.samples = samples
        self.sample_rate = sample_rate
        self.aug_prob = aug_prob
        self.vc_prob = vc_prob

        # Augmentation parameters
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.speed_min = speed_min
        self.speed_max = speed_max

        # Collect file paths for external augmentations
        self.bg_noise_files = _collect_audio_files(bg_noise_folder) if bg_noise_folder else []
        self.music_files = _collect_audio_files(music_folder) if music_folder else []
        self.bg_speech_files = _collect_audio_files(bg_speech_folder)  if bg_speech_folder else []
        self.mic_noise_files = _collect_audio_files(mic_noise_folder) if mic_noise_folder else []
        self.rir_files = _collect_audio_files(rir_folder) if rir_folder else []
        self.vc_files = []

        self.vc = None
        if vc_folder and vc_prob > 0:
            from chatterbox_onnx import ChatterboxOnnx  # optional dependency
            device: str = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            self.vc = ChatterboxOnnx(device=device)
            self.vc_files = _collect_audio_files(vc_folder)

        # Store data alias for validation (samples may be list of (path, label) tuples)
        self.data = self.samples

        # Validate files and log label distribution
        import os as _os
        missing = [path for path, _ in self.data if not _os.path.isfile(path)]
        if missing:
            logger.warning("AudioDataset: %d missing files (of %d total)", len(missing), len(self.data))
            for p in missing[:5]:
                logger.warning("  Missing: %s", p)
            if len(missing) > 5:
                logger.warning("  ... and %d more", len(missing) - 5)

        if self.data:
            labels = [label for _, label in self.data]
            unique = set(labels)
            dist = {lbl: labels.count(lbl) for lbl in sorted(unique)}
            logger.info("AudioDataset: %d samples, label distribution: %s", len(self.data), dist)

    def __len__(self):
        return len(self.samples)

    def get_augmented(self, wav: Union[numpy.ndarray, torch.Tensor]) -> torch.Tensor:
        """Apply augmentation pipeline to a single waveform.

        Args:
            wav: 1-D waveform as numpy array or torch tensor.

        Returns:
            Augmented waveform as torch.Tensor.
        """
        input_device = None
        if isinstance(wav, torch.Tensor):
            input_device = wav.device
            wav_np = wav.detach().cpu().numpy().astype(np.float32)
        elif isinstance(wav, np.ndarray):
            wav_np = wav.astype(np.float32)
        else:
            raise TypeError(f"Unsupported input type: {type(wav)}")

        if self.pipeline is not None:
            wav_np = self.pipeline(wav_np, sr=self.sample_rate)
        else:
            # Legacy augmentation path
            if self.bg_noise_files and random.random() < 0.6:
                bg_np = _load_audio_mono(random.choice(self.bg_noise_files), self.sample_rate)
                wav_np = _mix_background(wav_np, bg_np, random.uniform(self.snr_min, self.snr_max))
            if self.mic_noise_files and random.random() < 0.8:
                mic_np = _load_audio_mono(random.choice(self.mic_noise_files), self.sample_rate)
                wav_np = _mix_background(wav_np, mic_np, random.uniform(self.snr_min, self.snr_max))
            if self.music_files and random.random() < 0.3:
                music_np = _load_audio_mono(random.choice(self.music_files), self.sample_rate)
                wav_np = _mix_background(wav_np, music_np, random.uniform(0.0, 10.0))
            if self.bg_speech_files and random.random() < 0.5:
                speech_np = _load_audio_mono(random.choice(self.bg_speech_files), self.sample_rate)
                wav_np = _mix_background(wav_np, speech_np, random.uniform(10.0, 25.0))
            if self.rir_files and random.random() < 0.3:
                rir_np = _load_audio_mono(random.choice(self.rir_files), self.sample_rate)
                wav_np = _apply_reverb(wav_np, rir_np)
            if random.random() < 0.3:
                import librosa
                n_steps = random.uniform(self.pitch_min, self.pitch_max)
                wav_np = librosa.effects.pitch_shift(wav_np, sr=self.sample_rate, n_steps=n_steps).astype(np.float32)
            if random.random() < 0.3:
                import librosa
                factor = random.uniform(self.speed_min, self.speed_max)
                wav_np = librosa.effects.time_stretch(wav_np, rate=factor).astype(np.float32)
            peak = np.max(np.abs(wav_np))
            if peak > 1e-9:
                wav_np = wav_np / peak

        wav_t = torch.from_numpy(wav_np).float()
        if input_device is not None:
            wav_t = wav_t.to(input_device)
        return wav_t

    @timed
    def revoice(self, idx):
        path, label = self.samples[idx]
        target_voice = random.choice(self.vc_files)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            vc_path = f.name
        self.vc.voice_convert(
            source_audio_path=path,
            target_voice_path=target_voice,
            output_file_name=vc_path,
        )
        wav, sr = torchaudio.load(vc_path)
        return wav, sr

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # 0. Apply Voice Conversion -> simulate a new speaker
        if label == "1" and self.vc is not None and random.random() < self.vc_prob:
            try:
                wav, sr = self.revoice(idx)
            except Exception as exc:
                logger.warning("Voice conversion failed for %s: %s", path, exc)
                wav, sr = torchaudio.load(path)
        else:
            # 1. Load and Resample Source WAV
            wav, sr = torchaudio.load(path)

        # 1. Resample Source WAV if needed
        if sr != self.sample_rate:
            logger.warning(
                "Sample rate mismatch: expected %d Hz, got %d Hz for %s. "
                "Resampling automatically.", self.sample_rate, sr, path
            )
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)

        # Squeeze to mono [Length]
        wav = wav.squeeze(0)

        # Apply on-the-fly augmentation if enabled
        if self.aug_prob > 0 and random.random() < self.aug_prob:
            wav = self.get_augmented(wav)

        return wav, int(label), path


def collate_fn(batch, device="auto"):
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wavs, labels, paths = zip(*batch)
    max_len = max(w.shape[-1] for w in wavs)
    padded = torch.zeros(len(wavs), max_len)
    for i, w in enumerate(wavs):
        padded[i, :w.shape[-1]] = w

    return (
        padded.to(device),
        torch.tensor(labels, dtype=torch.float32).to(device),
        paths,
    )
