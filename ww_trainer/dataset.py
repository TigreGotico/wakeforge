import logging
import random
import re
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy
import numpy as np
import torch
import torchaudio

from torch.utils.data import Dataset


def _load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load an audio file, handling torchaudio 2.9+ torchcodec requirement.

    torchaudio 2.9 changed ``torchaudio.load()`` to use ``torchcodec`` by
    default. When ``torchcodec`` is not installed this raises an
    ``ImportError``.  This helper falls back to ``soundfile`` (always a
    project dependency) in that case.

    Args:
        path: Filesystem path to the audio file.

    Returns:
        A ``(waveform, sample_rate)`` tuple where ``waveform`` has shape
        ``[channels, time]`` and dtype ``float32``.
    """
    try:
        wav, sr = torchaudio.load(path)
        return wav, sr
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "torchaudio.load failed (%s); falling back to soundfile", exc
        )
    # Fallback: soundfile (always available as a project dependency)
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)  # [channels, time]
    return wav, int(sr)


def _save_audio(path: str, wav: torch.Tensor, sample_rate: int) -> None:
    """Save audio, handling torchaudio 2.9+ torchcodec requirement.

    Mirrors :func:`_load_audio`: prefers ``torchaudio.save``, falls back to
    ``soundfile`` when torchcodec is missing.

    Args:
        path: Output file path.
        wav: Waveform tensor with shape ``[channels, time]``.
        sample_rate: Sample rate in Hz.
    """
    try:
        torchaudio.save(str(path), wav, sample_rate)
        return
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "torchaudio.save failed (%s); falling back to soundfile", exc
        )
    import soundfile as sf
    arr = wav.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr.T  # soundfile expects [time, channels]
    sf.write(str(path), arr, int(sample_rate))


from ww_trainer.utils import timed
from ww_trainer.augment import (
    _collect_audio_files,
    _load_audio_mono,
    mix_background as _mix_background,
    apply_reverb as _apply_reverb,
    DeviceResponse,
)

logger = logging.getLogger(__name__)



_SILENCE_WINDOW_NAME = re.compile(r"(zeros|noise_m\d+dbfs)_\d+\.wav")


def is_silence_window(path: str) -> bool:
    """Whether *path* is a silence window as datagen writes it: ``negatives/silence/<class>_<n>.wav``."""
    p = Path(path)
    return (p.parent.name == "silence" and p.parent.parent.name == "negatives"
            and _SILENCE_WINDOW_NAME.fullmatch(p.name) is not None)


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
        wake_word_over_speech_folder: Folder of speech clips to mix *under* wake-word
            positives, simulating "wake word spoken over background conversation".
            Only applied to positive (label==1) samples. SNR range is controlled by
            ``wow_snr_min`` / ``wow_snr_max`` (default 8-18 dB — wake word stays
            clearly audible above the speech).
        wow_prob: Per-sample probability of applying wake-word-over-speech mixing.
        wow_snr_min: Minimum SNR (dB) for wake-word-over-speech mixing.
        wow_snr_max: Maximum SNR (dB) for wake-word-over-speech mixing.
        snr_min: Minimum SNR for noise mixing (legacy mode).
        snr_max: Maximum SNR for noise mixing (legacy mode).
        pitch_min: Minimum pitch shift in semitones (legacy mode).
        pitch_max: Maximum pitch shift in semitones (legacy mode).
        speed_min: Minimum speed factor (legacy mode).
        speed_max: Maximum speed factor (legacy mode).
        device: Device for voice conversion (legacy mode).
        feature_cache: Optional ``FeatureCache`` instance for caching un-augmented
            waveforms.  Bypassed when augmentation is applied to a sample.
        feature_store: Optional :class:`~ww_trainer.feature_store.FeatureStore`
            of a frozen featurizer. Items are then features ``[T, D]`` instead of
            waveforms, computed once per clip and reused.
        feature_variants: With a ``feature_store``, the number of augmented
            variants kept per clip: an augmented draw picks one of them and
            computes it only the first time. ``0`` augments on the fly.
        device_aug: Probability that an augmented draw also goes through
            :class:`~ww_trainer.augment.DeviceResponse`, applied last, with
            ``bg_speech_folder`` as its speech after the word. ``0`` disables
            it. The legacy reverb probability rises with it, from 0.3 at ``0``
            to 0.6 at ``1``.
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
                 wake_word_over_speech_folder: str = None,
                 wow_prob: float = 0.4,
                 wow_snr_min: float = 8.0,
                 wow_snr_max: float = 18.0,
                 snr_min: float = 0.0,
                 snr_max: float = 20.0,
                 pitch_min: float = -1.0,
                 pitch_max: float = 1.0,
                 speed_min: float = 0.95,
                 speed_max: float = 1.05,
                 device="auto",
                 validate: bool = False,
                 feature_cache=None,
                 feature_store=None,
                 feature_variants: int = 0,
                 device_aug: float = 0.0,
                 ):
        self.pipeline = pipeline
        self.feature_cache = feature_cache
        self.feature_store = feature_store
        self.feature_variants = feature_variants
        self.samples = samples
        self.sample_rate = sample_rate
        self.aug_prob = aug_prob
        self.vc_prob = vc_prob
        self.device_aug = device_aug
        self.device_response = DeviceResponse(bg_speech_folder) if device_aug > 0 else None

        # Augmentation parameters
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.speed_min = speed_min
        self.speed_max = speed_max

        # Wake-word-over-speech parameters
        self.wow_prob    = wow_prob
        self.wow_snr_min = wow_snr_min
        self.wow_snr_max = wow_snr_max

        # Collect file paths for external augmentations
        self.bg_noise_files = _collect_audio_files(bg_noise_folder) if bg_noise_folder else []
        self.music_files = _collect_audio_files(music_folder) if music_folder else []
        self.bg_speech_files = _collect_audio_files(bg_speech_folder) if bg_speech_folder else []
        self.mic_noise_files = _collect_audio_files(mic_noise_folder) if mic_noise_folder else []
        self.rir_files = _collect_audio_files(rir_folder) if rir_folder else []
        self.wow_files = _collect_audio_files(wake_word_over_speech_folder) \
            if wake_word_over_speech_folder else []
        if self.wow_files:
            logger.info("Wake-word-over-speech: %d donor clips (p=%.2f, SNR %.0f-%.0f dB)",
                        len(self.wow_files), wow_prob, wow_snr_min, wow_snr_max)
        self.vc_files = []

        self.vc = None
        if vc_folder and vc_prob > 0:
            # Voice conversion is delegated to the pure-ONNX voiceclonnx library.
            from ww_trainer.vc_helpers import load_vc_backend
            self.vc = load_vc_backend()
            self.vc_files = _collect_audio_files(vc_folder)

        # Store data alias for validation (samples may be list of (path, label) tuples)
        self.data = self.samples

        # Validate files and log label distribution
        import os as _os
        missing = [s[0] for s in self.data if not _os.path.isfile(s[0])]
        if missing:
            logger.warning("AudioDataset: %d missing files (of %d total)", len(missing), len(self.data))
            for p in missing[:5]:
                logger.warning("  Missing: %s", p)
            if len(missing) > 5:
                logger.warning("  ... and %d more", len(missing) - 5)

        if self.data:
            labels = [s[1] for s in self.data]
            unique = set(labels)
            dist = {lbl: labels.count(lbl) for lbl in sorted(unique)}
            logger.info("AudioDataset: %d samples, label distribution: %s", len(self.data), dist)

            # Warn on severe class imbalance (>10:1 ratio)
            counts = list(dist.values())
            if len(counts) >= 2:
                ratio = max(counts) / max(1, min(counts))
                if ratio > 20:
                    logger.warning(
                        "AudioDataset: class imbalance (%.1f:1). "
                        "Consider focal loss or neg_weight_schedule.", ratio
                    )
                elif ratio > 5:
                    logger.info("AudioDataset: class imbalance (%.1f:1).", ratio)

        # Deep validation: check files are readable audio
        if validate and self.data:
            import soundfile as sf
            bad_files = []
            for path, *_ in self.data:
                if not _os.path.isfile(path):
                    continue  # already reported above
                try:
                    info = sf.info(path)
                    if info.frames == 0:
                        bad_files.append((path, "empty audio"))
                except Exception as exc:
                    bad_files.append((path, str(exc)))
            if bad_files:
                logger.warning("AudioDataset validation: %d unreadable files", len(bad_files))
                for p, reason in bad_files[:5]:
                    logger.warning("  Bad file: %s — %s", p, reason)
                if len(bad_files) > 5:
                    logger.warning("  ... and %d more", len(bad_files) - 5)

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
            # Legacy augmentation path — always runs when called directly (e.g. by RPPL).
            # File-based augmentations only fire when the relevant file lists are populated.
            if self.bg_noise_files and random.random() < 0.6:
                bg_np = _load_audio_mono(random.choice(self.bg_noise_files), self.sample_rate)
                wav_np = _mix_background(wav_np, bg_np, random.uniform(self.snr_min, self.snr_max))
            if self.mic_noise_files and random.random() < 0.8:
                mic_np = _load_audio_mono(random.choice(self.mic_noise_files), self.sample_rate)
                wav_np = _mix_background(wav_np, mic_np, random.uniform(self.snr_min, self.snr_max))
            if self.music_files and random.random() < 0.3:
                music_np = _load_audio_mono(random.choice(self.music_files), self.sample_rate)
                wav_np = _mix_background(wav_np, music_np, random.uniform(10.0, 25.0))
            if self.bg_speech_files and random.random() < 0.5:
                speech_np = _load_audio_mono(random.choice(self.bg_speech_files), self.sample_rate)
                wav_np = _mix_background(wav_np, speech_np, random.uniform(10.0, 25.0))
            if self.rir_files and random.random() < 0.3 + 0.3 * self.device_aug:
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
            # Always apply lightweight perturbations so RPPL always gets a different view,
            # even when no file-based augmentation is configured.
            snr_db = random.uniform(15.0, 40.0)
            rms = np.sqrt(np.mean(wav_np ** 2) + 1e-9)
            noise = np.random.randn(len(wav_np)).astype(np.float32) * rms / (10 ** (snr_db / 20.0))
            wav_np = wav_np + noise
            gain = 10 ** (random.uniform(-3.0, 3.0) / 20.0)
            wav_np = wav_np * gain
            peak = np.max(np.abs(wav_np))
            if peak > 1e-9:
                wav_np = wav_np / peak

        if self.device_response is not None and random.random() < self.device_aug:
            wav_np = self.device_response(wav_np, sr=self.sample_rate)

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
        self.vc.vc(path, target_voice, vc_path)
        wav, sr = _load_audio(vc_path)
        return wav, sr

    def draw_augmentation(self, label: str):
        """One draw's augmentation decisions: ``(will_augment, use_vc, use_wow)``.

        The single place these are drawn, so a feature prefill fills each stored
        variant with the same mix of augmentations a training draw would.
        """
        will_augment = self.aug_prob > 0 and random.random() < self.aug_prob
        use_vc = label == "1" and self.vc is not None and random.random() < self.vc_prob
        use_wow = bool(self.wow_files) and label == "1" and random.random() < self.wow_prob
        return will_augment, use_vc, use_wow

    def __getitem__(self, idx):
        entry = self.samples[idx]
        path, label = entry[0], entry[1]
        keyword_ids = entry[2] if len(entry) > 2 else None

        will_augment, use_vc, use_wow = self.draw_augmentation(label)

        if self.feature_store is not None:
            return self._features(idx, will_augment, use_vc, use_wow), int(label), path, keyword_ids

        # Check feature cache (only for un-augmented samples)
        if not will_augment and self.feature_cache is not None:
            cached = self.feature_cache.get(path)
            if cached is not None:
                return torch.from_numpy(cached).float(), int(label), path, keyword_ids

        return self._waveform(idx, will_augment, use_vc, use_wow), int(label), path, keyword_ids

    def _features(self, idx: int, will_augment: bool, use_vc: bool, use_wow: bool) -> torch.Tensor:
        """Features ``[T, D]`` of sample *idx* from the feature store.

        Un-augmented draws use variant 0. An augmented draw picks one of the
        ``feature_variants`` variants and, the first time that variant is
        drawn, computes it with this draw's augmentations; with
        ``feature_variants`` 0 it is computed on the fly and not kept.
        """
        path = self.samples[idx][0]
        variant = 0
        if will_augment or use_vc or use_wow:
            variant = random.randint(1, self.feature_variants) if self.feature_variants else None
        if variant is not None:
            feats = self.feature_store.get(path, variant)
            if feats is not None:
                return feats
        feats = self.feature_store.featurize(self._waveform(idx, will_augment, use_vc, use_wow))
        if variant is not None:
            self.feature_store.put(path, variant, feats)
        return feats

    def _waveform(self, idx: int, will_augment: bool, use_vc: bool, use_wow: bool) -> torch.Tensor:
        """Load sample *idx* as a mono waveform, applying the chosen augmentations."""
        path, label = self.samples[idx][0], self.samples[idx][1]
        # Augmentation adds noise relative to the clip's own level and peak-normalises, which turns a
        # silence window into loud noise; the class exists to be heard at its level.
        will_augment = will_augment and not is_silence_window(path)

        # 0. Apply Voice Conversion -> simulate a new speaker
        if use_vc:
            try:
                wav, sr = self.revoice(idx)
            except Exception as exc:
                logger.warning("Voice conversion failed for %s: %s", path, exc)
                wav, sr = _load_audio(path)
        else:
            # 1. Load and Resample Source WAV
            wav, sr = _load_audio(path)

        # Resample to target rate if needed (handles any source SR transparently)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)

        # Mix down to mono [Length]
        if wav.dim() > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze(0)

        # Apply on-the-fly augmentation if enabled
        if will_augment:
            wav = self.get_augmented(wav)
        elif self.feature_cache is not None and self.feature_store is None:
            # Cache un-augmented waveform for future hits
            self.feature_cache.put(path, wav.numpy() if isinstance(wav, torch.Tensor) else wav)

        # Wake-word-over-speech: mix positives under a speech background at moderate SNR.
        if use_wow:
            donor_path = random.choice(self.wow_files)
            try:
                speech_np = _load_audio_mono(str(donor_path), self.sample_rate)
                wav_np = wav.numpy() if isinstance(wav, torch.Tensor) else wav
                snr = random.uniform(self.wow_snr_min, self.wow_snr_max)
                wav_np = _mix_background(wav_np, speech_np, snr)
                wav = torch.from_numpy(wav_np).float()
            except Exception as exc:
                logger.debug("WoW mixing failed for %s: %s", donor_path, exc)

        return wav


class Batch(tuple):
    """``(wavs, labels, paths, keyword_ids)`` from :func:`collate_fn`.

    Unpacks like the plain 4-tuple; ``lengths`` holds the real length of each
    padded row, in samples for waveform batches and in frames for batches of
    stored features (``[B, T, D]``). Pass it to the model
    (``model(wavs, lengths=batch.lengths)``) so padded frames are left out of
    pooling.
    """

    lengths: torch.Tensor

    def __new__(cls, items, lengths: torch.Tensor) -> "Batch":
        batch = super().__new__(cls, items)
        batch.lengths = lengths
        return batch


def collate_fn(batch, device="auto"):
    """Collate a batch of ``(wav, label, path[, keyword_ids])`` tuples.

    ``keyword_ids`` is an optional list of int token IDs per sample (e.g. IPA
    phoneme IDs for the keyword spoken in that sample).  When present, all
    per-sample lists are stacked into a padded ``[B, max_seq]`` int64 tensor;
    when absent (or all ``None``), ``None`` is returned as the fourth element.

    Items may hold 1-D waveforms (padded to ``[B, samples]``) or ``[T, D]``
    features from a :class:`~ww_trainer.feature_store.FeatureStore` (padded to
    ``[B, T, D]``); padding is zeros either way.

    Returns:
        :class:`Batch` ``(wavs, labels, paths, keyword_ids_tensor_or_None)``
        with ``lengths`` set.
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    has_kw = len(batch[0]) > 3
    if has_kw:
        wavs, labels, paths, kw_ids_list = zip(*batch)
    else:
        wavs, labels, paths = zip(*batch)
        kw_ids_list = None

    lengths = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    padded = torch.zeros(len(wavs), int(lengths.max()), *wavs[0].shape[1:])
    for i, w in enumerate(wavs):
        padded[i, :w.shape[0]] = w

    # Build padded keyword IDs tensor [B, max_seq] or None
    kw_tensor = None
    if kw_ids_list is not None and any(k is not None for k in kw_ids_list):
        max_seq = max((len(k) for k in kw_ids_list if k is not None), default=0)
        kw_arr = torch.zeros(len(kw_ids_list), max_seq, dtype=torch.long)
        for i, k in enumerate(kw_ids_list):
            if k is not None:
                kw_arr[i, :len(k)] = torch.tensor(k, dtype=torch.long)
        kw_tensor = kw_arr

    return Batch((
        padded.to(device),
        torch.tensor(labels, dtype=torch.float32).to(device),
        paths,
        kw_tensor,
    ), lengths)
