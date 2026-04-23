"""Extended tests for ww_trainer/dataset.py — targeting uncovered paths."""
import numpy as np
import pytest
import soundfile as sf
import torch
from unittest.mock import patch, MagicMock


SAMPLE_RATE = 16000
DURATION_S = 0.5


def _sine_wav(freq: int = 440, sr: int = SAMPLE_RATE, duration: float = DURATION_S) -> np.ndarray:
    """Generate a mono sine wave."""
    t = np.linspace(0, duration, int(sr * duration))
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _write_wav(tmp_path, name: str, wav: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """Write wav array to file and return path string."""
    p = tmp_path / name
    sf.write(str(p), wav, sr)
    return str(p)


def _make_samples(tmp_path, n: int = 4, sr: int = SAMPLE_RATE) -> list:
    """Return list of (path, label) tuples."""
    samples = []
    for i in range(n):
        path = _write_wav(tmp_path, f"s_{i}.wav", _sine_wav(440 + i * 100, sr))
        samples.append((path, str(i % 2)))
    return samples


# ---- _load_audio_mono ----

class TestLoadAudioMono:
    """Tests for _load_audio_mono covering mono, stereo, and resampling."""

    def test_mono_same_sr(self, tmp_path):
        from ww_trainer.dataset import _load_audio_mono
        wav = _sine_wav()
        path = _write_wav(tmp_path, "mono.wav", wav)
        result = _load_audio_mono(path, sr=SAMPLE_RATE)
        assert result.dtype == np.float32
        assert result.ndim == 1
        assert len(result) == len(wav)

    def test_stereo_downmix(self, tmp_path):
        """Stereo input is averaged to mono."""
        from ww_trainer.dataset import _load_audio_mono
        mono = _sine_wav()
        stereo = np.stack([mono, mono * 0.5], axis=1)  # [samples, 2]
        path = _write_wav(tmp_path, "stereo.wav", stereo)
        result = _load_audio_mono(path, sr=SAMPLE_RATE)
        assert result.ndim == 1
        assert result.dtype == np.float32

    def test_resample_different_sr(self, tmp_path):
        """Audio at 8kHz is resampled to 16kHz."""
        from ww_trainer.dataset import _load_audio_mono
        wav_8k = _sine_wav(sr=8000)
        path = _write_wav(tmp_path, "8k.wav", wav_8k, sr=8000)
        result = _load_audio_mono(path, sr=16000)
        assert result.dtype == np.float32
        # Resampled length should be approximately double
        assert len(result) > len(wav_8k)


# ---- _mix_background ----

class TestMixBackground:
    """Tests for _mix_background covering tiling and normalization."""

    def test_bg_shorter_than_clean_tiles(self):
        """Background shorter than clean is tiled."""
        from ww_trainer.dataset import _mix_background
        clean = np.ones(1000, dtype=np.float32) * 0.5
        bg = np.ones(300, dtype=np.float32) * 0.3
        result = _mix_background(clean, bg, snr_db=10)
        assert len(result) == len(clean)
        assert result.dtype == np.float32

    def test_bg_longer_than_clean(self):
        """Background longer than clean is cropped."""
        from ww_trainer.dataset import _mix_background
        clean = np.ones(500, dtype=np.float32) * 0.5
        bg = np.ones(2000, dtype=np.float32) * 0.3
        result = _mix_background(clean, bg, snr_db=5)
        assert len(result) == len(clean)

    def test_peak_clipping(self):
        """Output is clipped to [-1, 1] when peak > 1."""
        from ww_trainer.dataset import _mix_background
        clean = np.ones(1000, dtype=np.float32) * 0.9
        bg = np.ones(1000, dtype=np.float32) * 0.9
        result = _mix_background(clean, bg, snr_db=0)
        assert np.max(np.abs(result)) <= 1.0 + 1e-6


# ---- _apply_reverb ----

class TestApplyReverb:
    def test_output_dtype(self):
        from ww_trainer.dataset import _apply_reverb
        wav = np.random.randn(4000).astype(np.float32)
        rir = np.random.randn(100).astype(np.float32)
        out = _apply_reverb(wav, rir)
        assert out.dtype == np.float32
        assert len(out) == len(wav)


# ---- PitchShift / SpeedPerturb (now in augment.py) ----

class TestPitchShift:
    def test_returns_float32(self):
        from ww_trainer.augment import PitchShift
        wav = _sine_wav()
        result = PitchShift(min_steps=1.0, max_steps=1.0)(wav, sr=SAMPLE_RATE)
        assert result.dtype == np.float32
        assert len(result) > 0


class TestSpeedPerturb:
    def test_returns_float32(self):
        from ww_trainer.augment import SpeedPerturb
        wav = _sine_wav()
        result = SpeedPerturb(min_rate=1.1, max_rate=1.1)(wav, sr=SAMPLE_RATE)
        assert result.dtype == np.float32
        assert len(result) > 0

    def test_speed_up_shorter(self):
        from ww_trainer.augment import SpeedPerturb
        wav = _sine_wav()
        result = SpeedPerturb(min_rate=1.5, max_rate=1.5)(wav, sr=SAMPLE_RATE)
        assert len(result) < len(wav)


# ---- _collect_audio_files (now in augment.py) ----

class TestCollectAudioFiles:
    def test_empty_base_folder(self):
        from ww_trainer.augment import _collect_audio_files
        assert _collect_audio_files("") == []

    def test_nonexistent_folder(self, tmp_path):
        from ww_trainer.dataset import _collect_audio_files
        result = _collect_audio_files(str(tmp_path / "nope"))
        assert result == []

    def test_finds_wav_files(self, tmp_path):
        from ww_trainer.dataset import _collect_audio_files
        for name in ["a.wav", "b.flac", "c.txt", "d.mp3"]:
            (tmp_path / name).write_text("fake")
        result = _collect_audio_files(str(tmp_path))
        names = [p.name for p in result]
        assert "a.wav" in names
        assert "b.flac" in names
        assert "d.mp3" in names
        assert "c.txt" not in names

    def test_recursive(self, tmp_path):
        from ww_trainer.dataset import _collect_audio_files
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.wav").write_text("fake")
        result = _collect_audio_files(str(tmp_path))
        assert any("nested.wav" in str(p) for p in result)


# ---- AudioDataset.__getitem__ with augmentation ----

class TestAudioDatasetGetitem:
    def test_getitem_with_augmentation(self, tmp_path):
        """aug_prob=1.0 triggers augmentation path."""
        from ww_trainer.dataset import AudioDataset
        samples = _make_samples(tmp_path, n=2)
        ds = AudioDataset(samples, aug_prob=1.0)
        wav, label, path, kw_ids = ds[0]
        assert isinstance(wav, torch.Tensor)
        assert wav.ndim == 1

    def test_getitem_sample_rate_mismatch(self, tmp_path):
        """Audio at wrong sample rate triggers resampling."""
        from ww_trainer.dataset import AudioDataset
        wav_8k = _sine_wav(sr=8000)
        path = _write_wav(tmp_path, "8k.wav", wav_8k, sr=8000)
        samples = [(path, "0")]
        ds = AudioDataset(samples, sample_rate=16000, aug_prob=0.0)
        wav, label, p, kw_ids = ds[0]
        assert isinstance(wav, torch.Tensor)
        # Resampled from 8k to 16k should be longer
        assert wav.shape[0] > len(wav_8k)

    def test_getitem_revoice_fallback_on_error(self, tmp_path):
        """When vc is set but revoice fails, falls back to torchaudio.load."""
        from ww_trainer.dataset import AudioDataset
        samples = _make_samples(tmp_path, n=2)
        # Label must be "1" for revoice path
        samples = [(samples[0][0], "1"), (samples[1][0], "1")]
        ds = AudioDataset(samples, aug_prob=0.0)
        # Manually set vc to trigger the revoice branch
        ds.vc = MagicMock()
        ds.vc_prob = 1.0
        ds.vc_files = ["fake_voice.wav"]
        # Make revoice raise an exception
        ds.vc.voice_convert.side_effect = RuntimeError("VC failed")
        with patch("random.random", return_value=0.0):  # ensure random < vc_prob
            wav, label, path, kw_ids = ds[0]
        assert isinstance(wav, torch.Tensor)
        assert label == 1

    def test_missing_files_warning(self, tmp_path):
        """Missing files are warned about during init."""
        from ww_trainer.dataset import AudioDataset
        samples = [("/nonexistent/a.wav", "0"), ("/nonexistent/b.wav", "1")]
        # Should not raise, just warn
        ds = AudioDataset(samples, aug_prob=0.0)
        assert len(ds) == 2

    def test_missing_files_more_than_five(self, tmp_path):
        """When >5 files are missing, the 'and N more' path is hit."""
        from ww_trainer.dataset import AudioDataset
        samples = [(f"/nonexistent/{i}.wav", "0") for i in range(10)]
        ds = AudioDataset(samples, aug_prob=0.0)
        assert len(ds) == 10


# ---- get_augmented ----

class TestGetAugmented:
    def test_numpy_input(self, tmp_path):
        from ww_trainer.dataset import AudioDataset
        samples = _make_samples(tmp_path, n=2)
        ds = AudioDataset(samples, aug_prob=1.0)
        wav_np = _sine_wav()
        result = ds.get_augmented(wav_np)
        assert isinstance(result, torch.Tensor)

    def test_invalid_input_type(self, tmp_path):
        from ww_trainer.dataset import AudioDataset
        samples = _make_samples(tmp_path, n=2)
        ds = AudioDataset(samples, aug_prob=1.0)
        with pytest.raises(TypeError):
            ds.get_augmented("not an array")

    def test_with_bg_noise(self, tmp_path):
        """Augmentation with background noise files."""
        from ww_trainer.dataset import AudioDataset
        bg_dir = tmp_path / "bg"
        bg_dir.mkdir()
        _write_wav(bg_dir, "noise.wav", _sine_wav(200))
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        samples = _make_samples(data_dir, n=2)
        ds = AudioDataset(samples, aug_prob=1.0, bg_noise_folder=str(bg_dir))
        wav_t = torch.from_numpy(_sine_wav())
        with patch("random.random", return_value=0.0):  # ensure all augmentations fire
            result = ds.get_augmented(wav_t)
        assert isinstance(result, torch.Tensor)

    def test_with_rir(self, tmp_path):
        """Augmentation with RIR convolution."""
        from ww_trainer.dataset import AudioDataset
        rir_dir = tmp_path / "rir"
        rir_dir.mkdir()
        rir = np.random.randn(200).astype(np.float32) * 0.01
        _write_wav(rir_dir, "rir.wav", rir)
        data_dir = tmp_path / "data2"
        data_dir.mkdir(exist_ok=True)
        samples = _make_samples(data_dir, n=2)
        ds = AudioDataset(samples, aug_prob=1.0, rir_folder=str(rir_dir))
        wav_t = torch.from_numpy(_sine_wav())
        with patch("random.random", return_value=0.0):
            result = ds.get_augmented(wav_t)
        assert isinstance(result, torch.Tensor)

    def test_with_music_and_speech(self, tmp_path):
        """Augmentation with music and background speech."""
        from ww_trainer.dataset import AudioDataset
        music_dir = tmp_path / "music"
        music_dir.mkdir()
        _write_wav(music_dir, "track.wav", _sine_wav(330))
        speech_dir = tmp_path / "speech"
        speech_dir.mkdir()
        _write_wav(speech_dir, "talk.wav", _sine_wav(250))
        data_dir = tmp_path / "data3"
        data_dir.mkdir()
        samples = _make_samples(data_dir, n=2)
        ds = AudioDataset(samples, aug_prob=1.0,
                          music_folder=str(music_dir),
                          bg_speech_folder=str(speech_dir))
        wav_t = torch.from_numpy(_sine_wav())
        with patch("random.random", return_value=0.0):
            result = ds.get_augmented(wav_t)
        assert isinstance(result, torch.Tensor)

    def test_with_mic_noise(self, tmp_path):
        """Augmentation with microphone noise."""
        from ww_trainer.dataset import AudioDataset
        mic_dir = tmp_path / "mic"
        mic_dir.mkdir()
        _write_wav(mic_dir, "mic.wav", _sine_wav(60))
        data_dir = tmp_path / "data4"
        data_dir.mkdir()
        samples = _make_samples(data_dir, n=2)
        ds = AudioDataset(samples, aug_prob=1.0, mic_noise_folder=str(mic_dir))
        wav_t = torch.from_numpy(_sine_wav())
        with patch("random.random", return_value=0.0):
            result = ds.get_augmented(wav_t)
        assert isinstance(result, torch.Tensor)

    def test_normalization_near_zero(self, tmp_path):
        """Near-silent input does not cause division by zero."""
        from ww_trainer.dataset import AudioDataset
        samples = _make_samples(tmp_path, n=2)
        ds = AudioDataset(samples, aug_prob=1.0)
        silent = torch.zeros(4000)
        result = ds.get_augmented(silent)
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result).any()

    def test_preserves_device(self, tmp_path):
        """Output tensor is on the same device as input."""
        from ww_trainer.dataset import AudioDataset
        samples = _make_samples(tmp_path, n=2)
        ds = AudioDataset(samples, aug_prob=1.0)
        wav_t = torch.from_numpy(_sine_wav()).to("cpu")
        result = ds.get_augmented(wav_t)
        assert result.device == wav_t.device


# ---- collate_fn ----

class TestCollateFn:
    def test_auto_device(self):
        """device='auto' selects cpu when no GPU available."""
        from ww_trainer.dataset import collate_fn
        batch = [(torch.zeros(100), 0, "a.wav"), (torch.zeros(200), 1, "b.wav")]
        padded, labels, paths, kw_ids = collate_fn(batch, device="auto")
        assert padded.shape == (2, 200)
        assert labels.shape == (2,)
        assert kw_ids is None

    def test_explicit_cpu_device(self):
        from ww_trainer.dataset import collate_fn
        batch = [(torch.ones(50), 1, "x.wav")]
        padded, labels, paths, kw_ids = collate_fn(batch, device="cpu")
        assert padded.shape == (1, 50)
        assert labels.item() == 1.0
        assert kw_ids is None

    def test_with_keyword_ids(self):
        """4-tuple batch produces stacked [B, seq] keyword ID tensor."""
        from ww_trainer.dataset import collate_fn
        batch = [
            (torch.zeros(100), 0, "a.wav", [1, 2, 3]),
            (torch.zeros(200), 1, "b.wav", [4, 5, 6, 7]),
        ]
        padded, labels, paths, kw_ids = collate_fn(batch, device="cpu")
        assert kw_ids is not None
        assert kw_ids.shape == (2, 4)  # padded to longest
        assert kw_ids[0, 3].item() == 0  # padding
        assert kw_ids[1, 3].item() == 7
