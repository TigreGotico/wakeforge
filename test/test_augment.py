"""Tests for ww_trainer/augment.py — composable audio transforms."""
import numpy as np
import pytest

from ww_trainer.augment import (
    AudioTransform,
    AugmentationPipeline,
    GaussianNoise,
    VolumePerturb,
    TimeShift,
    SpecAugment,
    Normalize,
    mix_background,
    apply_reverb,
)


@pytest.fixture
def sine_wav() -> np.ndarray:
    t = np.linspace(0, 1, 16000, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5


class TestMixBackground:
    def test_output_shape(self, sine_wav):
        bg = np.random.randn(16000).astype(np.float32) * 0.01
        out = mix_background(sine_wav, bg, snr_db=10.0)
        assert out.shape == sine_wav.shape
        assert out.dtype == np.float32

    def test_tiling_short_bg(self, sine_wav):
        """Background shorter than clean should be tiled."""
        short_bg = np.random.randn(1000).astype(np.float32) * 0.01
        out = mix_background(sine_wav, short_bg, snr_db=10.0)
        assert out.shape == sine_wav.shape

    def test_peak_normalized(self, sine_wav):
        bg = np.random.randn(16000).astype(np.float32) * 10.0
        out = mix_background(sine_wav, bg, snr_db=0.0)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6


class TestApplyReverb:
    def test_output_shape(self, sine_wav):
        rir = np.array([0.0, 0.0, 1.0, 0.3, 0.1], dtype=np.float32)
        out = apply_reverb(sine_wav, rir)
        assert out.shape == sine_wav.shape


class TestGaussianNoise:
    def test_adds_noise(self, sine_wav):
        transform = GaussianNoise(snr_min=10, snr_max=10)
        out = transform(sine_wav)
        assert out.shape == sine_wav.shape
        assert not np.allclose(out, sine_wav)

    def test_high_snr_preserves_signal(self, sine_wav):
        transform = GaussianNoise(snr_min=60, snr_max=60)
        out = transform(sine_wav)
        np.testing.assert_allclose(out, sine_wav, atol=0.05)


class TestVolumePerturb:
    def test_changes_volume(self, sine_wav):
        transform = VolumePerturb(min_gain_db=6, max_gain_db=6)
        out = transform(sine_wav)
        assert np.max(np.abs(out)) > np.max(np.abs(sine_wav))

    def test_negative_gain(self, sine_wav):
        transform = VolumePerturb(min_gain_db=-6, max_gain_db=-6)
        out = transform(sine_wav)
        assert np.max(np.abs(out)) < np.max(np.abs(sine_wav))


class TestTimeShift:
    def test_output_shape(self, sine_wav):
        transform = TimeShift(max_shift_frac=0.1)
        out = transform(sine_wav)
        assert out.shape == sine_wav.shape

    def test_zero_shift(self):
        wav = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        transform = TimeShift(max_shift_frac=0.0)
        out = transform(wav)
        np.testing.assert_array_equal(out, wav)


class TestSpecAugment:
    def test_masks_some_samples(self, sine_wav):
        transform = SpecAugment(max_mask_frac=0.2, n_masks=2)
        out = transform(sine_wav)
        # Some samples should be zeroed
        assert np.sum(out == 0) > 0
        assert out.shape == sine_wav.shape


class TestNormalize:
    def test_peak_one(self, sine_wav):
        transform = Normalize(target_peak=1.0)
        out = transform(sine_wav)
        np.testing.assert_allclose(np.max(np.abs(out)), 1.0, atol=1e-6)

    def test_silence(self):
        silence = np.zeros(1000, dtype=np.float32)
        out = Normalize()(silence)
        np.testing.assert_array_equal(out, silence)


class TestAugmentationPipeline:
    def test_chains_transforms(self, sine_wav):
        pipeline = AugmentationPipeline([
            (GaussianNoise(snr_min=20, snr_max=20), 1.0),
            (Normalize(), 1.0),
        ])
        out = pipeline(sine_wav)
        assert out.shape == sine_wav.shape
        np.testing.assert_allclose(np.max(np.abs(out)), 1.0, atol=1e-6)

    def test_zero_probability_skips(self, sine_wav):
        pipeline = AugmentationPipeline([
            (GaussianNoise(), 0.0),  # never applied
        ])
        out = pipeline(sine_wav)
        np.testing.assert_array_equal(out, sine_wav)

    def test_empty_pipeline(self, sine_wav):
        pipeline = AugmentationPipeline([])
        out = pipeline(sine_wav)
        np.testing.assert_array_equal(out, sine_wav)

    def test_full_pipeline(self, sine_wav):
        pipeline = AugmentationPipeline([
            (GaussianNoise(snr_min=20, snr_max=30), 0.5),
            (VolumePerturb(min_gain_db=-3, max_gain_db=3), 0.5),
            (TimeShift(max_shift_frac=0.05), 0.5),
            (SpecAugment(max_mask_frac=0.05), 0.5),
            (Normalize(), 1.0),
        ])
        out = pipeline(sine_wav)
        assert out.shape == sine_wav.shape
        assert out.dtype == np.float32
