import random
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy
import numpy as np
import soundfile as sf
import torch
import torchaudio
from chatterbox_onnx import ChatterboxOnnx
from torch.utils.data import Dataset


def _load_audio_mono(path, sr=16000):
    """Load audio and ensure it's mono at the target sample rate."""
    wav, orig_sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if orig_sr != sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=orig_sr, target_sr=sr)
    return wav.astype(np.float32)


def _mix_background(clean, bg, snr_db):
    """
    Mix two sound waves at a specified Signal-to-Noise Ratio (SNR).
    The background audio is cropped/tiled to match the length of the clean audio.
    """
    clean_len = len(clean)

    if len(bg) < clean_len:
        # Tile the background audio if it's too short
        nrep = int(np.ceil(clean_len / len(bg)))
        bg = np.tile(bg, nrep)

    # Randomly select a segment of the background audio that is the length of the clean audio
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


def _apply_reverb(wav, rir):
    """Apply room impulse response (RIR) convolution."""
    REVERB_ATTENUATION_FACTOR = 0.5
    out = np.convolve(wav, rir)[:len(wav)]
    rms_wav = np.sqrt(np.mean(wav ** 2) + 1e-9)
    rms_out = np.sqrt(np.mean(out ** 2) + 1e-9)
    out = out * (rms_wav / rms_out) * REVERB_ATTENUATION_FACTOR
    return out.astype(np.float32)


def _pitch_shift(wav, sr, n_steps):
    """Apply pitch shift."""
    return librosa.effects.pitch_shift(wav, sr=sr, n_steps=n_steps).astype(np.float32)


def _speed_perturb(wav, factor):
    """Apply speed perturbation."""
    return librosa.effects.time_stretch(wav, rate=factor).astype(np.float32)


def _collect_audio_files(base_folder):
    """Collect all valid audio files from a folder."""
    exts = [".wav", ".flac", ".mp3", ".m4a", ".ogg"]
    files = []
    if not base_folder:
        return files
    p = Path(base_folder)
    if not p.exists():
        # Using print instead of click.echo here since we are inside a core PyTorch module
        print(f"Warning: Augmentation folder not found: {base_folder}")
        return files
    for ext in exts:
        files.extend(p.rglob(f"*{ext}"))
    return sorted(files)


# -----------------------------------------------------------------



class AudioDataset(Dataset):
    """
    A PyTorch Dataset for loading audio files with optional on-the-fly augmentation.
    """

    def __init__(self, samples,
                 sample_rate: int = 16000,
                 aug_prob: float = 0.7, # set to 0 to disable
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
                 speed_max: float = 1.05
                 ):

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
        self.bg_noise_files = _collect_audio_files(bg_noise_folder)
        self.music_files = _collect_audio_files(music_folder)
        self.bg_speech_files = _collect_audio_files(bg_speech_folder)
        self.mic_noise_files = _collect_audio_files(mic_noise_folder)
        self.rir_files = _collect_audio_files(rir_folder)
        self.vc_files = _collect_audio_files(vc_folder)

        self.vc: Optional[ChatterboxOnnx] = None
        if vc_folder and vc_prob > 0:
            self.vc = ChatterboxOnnx()

    def __len__(self):
        return len(self.samples)

    def get_augmented(self, wav: Union[numpy.ndarray, torch.Tensor]) -> torch.Tensor:
        # --- Handle input types ---
        input_device = None
        requires_grad = False

        if isinstance(wav, torch.Tensor):
            input_device = wav.device
            requires_grad = wav.requires_grad
            wav_np = wav.detach().cpu().numpy().astype(np.float32)
        elif isinstance(wav, np.ndarray):
            wav_np = wav.astype(np.float32)
        else:
            raise TypeError(f"Unsupported input type for get_augmented(): {type(wav)}")

        # --- General Noise Mixing ---
        # Apply general background noises (objects, animals, activities)
        if self.bg_noise_files and random.random() < 0.6:
            bg_path = random.choice(self.bg_noise_files)
            bg_np = _load_audio_mono(bg_path, self.sample_rate)
            snr = random.uniform(self.snr_min, self.snr_max)
            wav_np = _mix_background(wav_np, bg_np, snr)

        # Apply microphone silence/noise (80% probability if available)
        if self.mic_noise_files and random.random() < 0.8:
            mic_path = random.choice(self.mic_noise_files)
            mic_np = _load_audio_mono(mic_path, self.sample_rate)
            snr = random.uniform(self.snr_min, self.snr_max)
            wav_np = _mix_background(wav_np, mic_np, snr)

        # --- Music Mixing (30% probability, louder) ---
        if self.music_files and random.random() < 0.3:
            music_path = random.choice(self.music_files)
            music_np = _load_audio_mono(music_path, self.sample_rate)
            # Music is louder: lower SNR (0.0 to 10.0 dB)
            snr = random.uniform(0.0, 10.0)
            wav_np = _mix_background(wav_np, music_np, snr)

        # --- Background Speech Mixing (50% probability, quieter) ---
        if self.bg_speech_files and random.random() < 0.5:
            speech_path = random.choice(self.bg_speech_files)
            speech_np = _load_audio_mono(speech_path, self.sample_rate)
            # Background speech is quieter: higher SNR (10.0 to 25.0 dB)
            snr = random.uniform(10.0, 25.0)
            wav_np = _mix_background(wav_np, speech_np, snr)

        # --- Reverb ---
        # Apply RIR (30% probability if available)
        if self.rir_files and random.random() < 0.3:
            rir_path = random.choice(self.rir_files)
            rir_np = _load_audio_mono(rir_path, self.sample_rate)
            wav_np = _apply_reverb(wav_np, rir_np)

        # --- Pitch Shift ---
        # Apply pitch shift (30% probability)
        if random.random() < 0.3:
            n_steps = random.uniform(self.pitch_min, self.pitch_max)
            wav_np = _pitch_shift(wav_np, self.sample_rate, n_steps)

        # --- Speed Perturbation ---
        # Apply speed perturb (30% probability)
        if random.random() < 0.3:
            f = random.uniform(self.speed_min, self.speed_max)
            wav_np = _speed_perturb(wav_np, f)

        # 3. Post-Augmentation Normalization and Conversion
        # Re-normalize to avoid clipping
        peak = np.max(np.abs(wav_np))
        if peak > 1e-9:
            wav_np = wav_np / peak

        # --- Convert back to torch ---
        wav_t = torch.from_numpy(wav_np).float()

        # Send back to original device
        if input_device is not None:
            wav_t = wav_t.to(input_device)

        if requires_grad:
            wav_t.requires_grad_()

        return wav_t


    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # 0. Apply Voice Conversion -> simulate a new speaker
        if label == "1" and self.vc is not None and random.random() < self.vc_prob:
            target_voice = random.choice(self.vc_files)
            from uuid import uuid4
            vc_path = f"/tmp/vc_{uuid4()}.wav" # TODO
            try:
                self.vc.voice_convert(
                    source_audio_path=path,
                    target_voice_path=target_voice,
                    output_file_name=vc_path,
                )
                wav, sr = torchaudio.load(vc_path)
            except:
                wav, sr = torchaudio.load(path)
        else:
            # 1. Load and Resample Source WAV
            wav, sr = torchaudio.load(path)

        # 1. Resample Source WAV if needed
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

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
