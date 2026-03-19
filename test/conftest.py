"""Shared fixtures for ww-trainer tests.

All fixtures generate synthetic audio data (sine waves) so tests run
on CPU without real audio files or model weights.
"""
import numpy as np
import pytest
import soundfile as sf
import torch


SAMPLE_RATE = 16000
DURATION_S = 0.5  # seconds — keep short for fast CI


@pytest.fixture
def sample_rate() -> int:
    return SAMPLE_RATE


@pytest.fixture
def wav_tensor() -> torch.Tensor:
    """Single 1-D float32 waveform tensor at 16 kHz."""
    t = torch.linspace(0, DURATION_S, int(SAMPLE_RATE * DURATION_S))
    return torch.sin(2 * torch.pi * 440 * t)


@pytest.fixture
def wav_np() -> np.ndarray:
    """Single 1-D float32 numpy waveform at 16 kHz."""
    t = np.linspace(0, DURATION_S, int(SAMPLE_RATE * DURATION_S))
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def wav_file(tmp_path, wav_np) -> str:
    """Write a synthetic WAV to a temp file and return the path string."""
    path = tmp_path / "test_audio.wav"
    sf.write(str(path), wav_np, SAMPLE_RATE)
    return str(path)


@pytest.fixture
def two_wav_files(tmp_path) -> list:
    """Two synthetic WAV files at different frequencies."""
    paths = []
    for freq in [440, 880]:
        t = np.linspace(0, DURATION_S, int(SAMPLE_RATE * DURATION_S))
        wav = np.sin(2 * np.pi * freq * t).astype(np.float32)
        p = tmp_path / f"audio_{freq}.wav"
        sf.write(str(p), wav, SAMPLE_RATE)
        paths.append(str(p))
    return paths


@pytest.fixture
def random_feats() -> torch.Tensor:
    """Random feature tensor [2, 50, 768] simulating HuBERT output."""
    torch.manual_seed(42)
    return torch.randn(2, 50, 768)


@pytest.fixture
def binary_labels() -> torch.Tensor:
    """Simple binary label tensor [1, 0] for batch size 2."""
    return torch.tensor([1, 0], dtype=torch.float32)
