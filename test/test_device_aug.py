"""DeviceResponse augmentation and its ``--device-aug`` wiring."""
import csv
import random
import wave

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from ww_trainer.augment import DeviceResponse
from ww_trainer.dataset import AudioDataset

SR = 16000


def _write_wav(path, freq, sr=SR, dur=0.5, amp=0.3):
    t = np.arange(int(sr * dur)) / sr
    wav = (amp * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(wav.tobytes())


def _bin_power(y, freq, sr=SR):
    spectrum = np.abs(np.fft.rfft(y)) ** 2
    return spectrum[int(round(freq * len(y) / sr))]


@pytest.fixture
def speech_dir(tmp_path):
    d = tmp_path / "speech"
    d.mkdir()
    for i in range(3):
        _write_wav(d / f"s{i}.wav", 300 + 50 * i, dur=0.3)
    return str(d)


@pytest.fixture
def call_spy(monkeypatch):
    calls = []
    original = DeviceResponse.__call__

    def spy(self, wav, sr=16000):
        out = original(self, wav, sr)
        calls.append(out)
        return out

    monkeypatch.setattr(DeviceResponse, "__call__", spy)
    return calls


def test_loud_input_comes_back_bounded_float32(speech_dir):
    random.seed(0)
    np.random.seed(0)
    t = np.arange(SR) / SR
    loud = (5.0 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    aug = DeviceResponse(speech_dir, speech_prob=1.0, band_prob=1.0, colour_prob=1.0,
                         compress_prob=1.0, hot_prob=1.0, noise_prob=1.0)
    for _ in range(20):
        out = aug(loud, SR)
        assert out.shape == loud.shape
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))
        assert np.abs(out).max() <= 1.0
        assert not np.allclose(out, loud)


def test_level_step_lands_between_minus_30_and_0_dbfs():
    random.seed(1)
    t = np.arange(SR) / SR
    wav = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    aug = DeviceResponse(band_prob=0, colour_prob=0, compress_prob=0, hot_prob=0, noise_prob=0)
    peaks = [20 * np.log10(np.abs(aug(wav, SR)).max()) for _ in range(200)]
    assert min(peaks) >= -30.01 and max(peaks) <= 0.01
    assert max(peaks) - min(peaks) > 20


def test_band_limit_removes_low_frequency_energy():
    random.seed(2)
    t = np.arange(SR) / SR
    wav = (0.4 * np.sin(2 * np.pi * 30 * t) + 0.4 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    before = _bin_power(wav, 30) / _bin_power(wav, 1000)
    banded = DeviceResponse(band_prob=1.0, colour_prob=0, compress_prob=0, hot_prob=0, noise_prob=0)
    flat = DeviceResponse(band_prob=0.0, colour_prob=0, compress_prob=0, hot_prob=0, noise_prob=0)
    for _ in range(50):
        out = banded(wav, SR)
        assert _bin_power(out, 30) / _bin_power(out, 1000) < 0.01 * before
        out = flat(wav, SR)
        assert _bin_power(out, 30) / _bin_power(out, 1000) == pytest.approx(before, rel=1e-3)


def test_speech_after_the_word_only_touches_the_second_half(speech_dir):
    random.seed(3)
    wav = np.zeros(SR, dtype=np.float32)
    wav[: SR // 2] = 0.1
    aug = DeviceResponse(speech_dir, speech_prob=1.0, band_prob=0, colour_prob=0,
                         compress_prob=0, hot_prob=0, noise_prob=0)
    for _ in range(10):
        out = aug(wav, SR)
        first = out[: SR // 2]
        assert np.allclose(first, first[0])
        assert np.abs(np.diff(out[SR // 2:])).max() > 1e-5


def _clips(tmp_path, n=4):
    samples = []
    for i in range(n):
        p = tmp_path / f"clip{i}.wav"
        _write_wav(p, 440 + i)
        samples.append((str(p), "1" if i % 2 else "0"))
    return samples


def test_dataset_applies_device_response_only_when_enabled(tmp_path, speech_dir, call_spy):
    samples = _clips(tmp_path)
    on = AudioDataset(samples, aug_prob=1.0, device_aug=1.0, bg_speech_folder=speech_dir)
    for i in range(len(on)):
        on[i]
    assert len(call_spy) == len(samples)

    call_spy.clear()
    off = AudioDataset(samples, aug_prob=1.0, device_aug=0.0, bg_speech_folder=speech_dir)
    for i in range(len(off)):
        off[i]
    assert call_spy == []


class _FakeStore:
    def __init__(self):
        self.kept = {}

    def get(self, path, variant=0):
        return self.kept.get((path, variant))

    def put(self, path, variant, feats):
        self.kept[(path, variant)] = feats

    def featurize(self, wav):
        return wav.reshape(-1, 1).float()


def test_feature_variants_are_computed_with_device_response(tmp_path, call_spy):
    random.seed(4)
    samples = _clips(tmp_path)
    store = _FakeStore()
    ds = AudioDataset(samples, aug_prob=1.0, device_aug=1.0,
                      feature_store=store, feature_variants=2)
    for _ in range(3):
        for i in range(len(ds)):
            ds[i]
    assert store.kept and all(v > 0 for _, v in store.kept)
    assert len(call_spy) == len(store.kept)
    outputs = [torch.from_numpy(out) for out in call_spy]
    for feats in store.kept.values():
        assert any(o.shape == feats[:, 0].shape and torch.equal(o, feats[:, 0]) for o in outputs)


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return str(path)


@pytest.mark.parametrize("prob, expect_calls", [("1.0", True), ("0.0", False)])
def test_cli_flag_reaches_training(tmp_path, call_spy, prob, expect_calls):
    rows = []
    for i in range(12):
        pos, neg = tmp_path / f"p{i}.wav", tmp_path / f"n{i}.wav"
        _write_wav(pos, 440 + i)
        _write_wav(neg, 110 + i)
        rows += [(str(pos), "1"), (str(neg), "0")]
    from ww_trainer.cli import train
    result = CliRunner().invoke(train, [
        "--wake-word", "hey_test",
        "--metadata", _write_csv(tmp_path / "train.csv", rows[:20]),
        "--test-metadata", _write_csv(tmp_path / "test.csv", rows[20:]),
        "--tier", "micro",
        "--epochs", "1",
        "--batch-size", "4",
        "--device", "cpu",
        "--seed", "42",
        "--output-dir", str(tmp_path / "model"),
        "--no-feature-cache",
        "--device-aug", prob,
    ])
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert bool(call_spy) is expect_calls


@pytest.mark.parametrize("n", [1, 10, 63])
def test_inputs_shorter_than_the_noise_kernel(n):
    random.seed(5)
    np.random.seed(5)
    wav = (0.5 * np.sin(np.arange(n))).astype(np.float32)
    aug = DeviceResponse(band_prob=1.0, colour_prob=1.0, compress_prob=1.0,
                         hot_prob=1.0, noise_prob=1.0)
    for _ in range(40):
        out = aug(wav, SR)
        assert out.shape == (n,) and np.all(np.isfinite(out))


def test_empty_speech_file_is_skipped(tmp_path):
    folder = tmp_path / "speech"
    folder.mkdir()
    with wave.open(str(folder / "empty.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"")
    _write_wav(folder / "short.wav", 300, dur=10 / SR)
    random.seed(6)
    wav = (0.3 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)
    aug = DeviceResponse(str(folder), speech_prob=1.0)
    assert len(aug.files) == 2
    for _ in range(40):
        out = aug(wav, SR)
        assert out.shape == wav.shape and np.all(np.isfinite(out))


@pytest.mark.parametrize("device_aug, expected", [(0.0, 0.3), (0.5, 0.45), (1.0, 0.6)])
def test_reverb_probability_scales_with_device_aug(tmp_path, monkeypatch, device_aug, expected):
    import librosa
    import ww_trainer.dataset as dataset_module
    rirs = tmp_path / "rirs"
    rirs.mkdir()
    _write_wav(rirs / "r.wav", 100, dur=0.01)
    reverbs = []

    def count_reverb(wav, rir, attenuation=0.5):
        reverbs.append(1)
        return wav

    monkeypatch.setattr(dataset_module, "_apply_reverb", count_reverb)
    monkeypatch.setattr(dataset_module, "_load_audio_mono", lambda path, sr: np.zeros(8, np.float32))
    monkeypatch.setattr(librosa.effects, "pitch_shift", lambda y, **kw: y)
    monkeypatch.setattr(librosa.effects, "time_stretch", lambda y, **kw: y)
    monkeypatch.setattr(DeviceResponse, "__call__", lambda self, wav, sr=16000: wav)
    random.seed(7)
    ds = AudioDataset([], aug_prob=1.0, rir_folder=str(rirs), device_aug=device_aug)
    wav = np.full(160, 0.1, dtype=np.float32)
    draws = 5000
    for _ in range(draws):
        ds.get_augmented(wav)
    assert len(reverbs) / draws == pytest.approx(expected, abs=0.025)


@pytest.mark.parametrize("device_aug", [1.0, 0.0])
def test_prefill_applies_device_response_to_augmented_variants(tmp_path, call_spy, device_aug):
    from conftest import export_stand_in_featurizer
    from ww_trainer.feats import OnnxFeatureExtractor
    from ww_trainer.feature_store import FeatureStore, prefill
    onnx = tmp_path / "feat.onnx"
    export_stand_in_featurizer(onnx, seed=0)
    store = FeatureStore(OnnxFeatureExtractor(str(onnx), sample_rate=SR, device="cpu", hop_samples=320),
                         str(tmp_path / "cache"))
    clips = _clips(tmp_path, 3)
    n = prefill(store, clips, variants=2, workers=1,
                dataset_kwargs={"aug_prob": 1.0, "device_aug": device_aug})
    assert n == 3 * 3
    if not device_aug:
        assert call_spy == []
        return
    assert len(call_spy) == 3 * 2
    first = torch.from_numpy(call_spy[0])
    torch.testing.assert_close(store.get(clips[0][0], 1), store.featurize(first), rtol=1e-2, atol=1e-2)
