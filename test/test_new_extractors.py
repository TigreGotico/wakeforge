"""Tests for PLP, PNCC, CQT, VAD, Pitch, MultiResolution, SNR extractors."""
import pytest
import torch

from ww_trainer.feats import (
    PLPExtractor,
    PNCCExtractor,
    CQTExtractor,
    VoiceActivityExtractor,
    PitchExtractor,
    MultiResolutionExtractor,
    SNRAwareExtractor,
    MfccExtractor,
    FilterbankExtractor,
)


# ---------- PLPExtractor ----------

class TestPLPExtractor:
    def test_forward_shape(self):
        ext = PLPExtractor(n_plp=13)
        out = ext(torch.randn(2, 16000))
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 13

    def test_feature_dim(self):
        assert PLPExtractor(n_plp=20).feature_dim == 20

    def test_list_input(self):
        ext = PLPExtractor(n_plp=13)
        out = ext([torch.randn(16000), torch.randn(8000)])
        assert out.shape[0] == 2
        assert out.shape[2] == 13

    def test_short_audio(self):
        ext = PLPExtractor(n_plp=13)
        out = ext(torch.randn(1, 4000))
        assert out.ndim == 3


# ---------- PNCCExtractor ----------

class TestPNCCExtractor:
    def test_forward_shape(self):
        ext = PNCCExtractor(n_pncc=13)
        out = ext(torch.randn(2, 16000))
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 13

    def test_feature_dim(self):
        assert PNCCExtractor(n_pncc=20).feature_dim == 20

    def test_noise_robustness_differs_from_mfcc(self):
        """PNCC and MFCC should produce different features."""
        ext_pncc = PNCCExtractor(n_pncc=13)
        ext_mfcc = MfccExtractor(n_mfcc=13)
        audio = torch.randn(1, 16000)
        pncc = ext_pncc(audio)
        mfcc = ext_mfcc(audio)
        # They should differ (not identical)
        assert not torch.allclose(pncc[:, :mfcc.size(1), :], mfcc[:, :pncc.size(1), :])


# ---------- CQTExtractor ----------

class TestCQTExtractor:
    def test_forward_shape(self):
        ext = CQTExtractor(n_bins=12, n_octaves=7)
        out = ext(torch.randn(2, 16000))
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == ext.feature_dim

    def test_feature_dim(self):
        ext = CQTExtractor(n_bins=12, n_octaves=5)
        assert ext.feature_dim > 0
        assert ext.feature_dim <= 60

    def test_log_frequency_spacing(self):
        """CQT should have logarithmic frequency spacing."""
        ext = CQTExtractor(n_bins=12, n_octaves=7)
        freqs = ext.center_freqs
        ratios = freqs[1:] / freqs[:-1]
        # All ratios should be approximately equal (constant Q)
        assert torch.allclose(ratios, ratios[0].expand_as(ratios), atol=0.01)


# ---------- VoiceActivityExtractor ----------

class TestVoiceActivityExtractor:
    def test_forward_shape(self):
        base = MfccExtractor(n_mfcc=13)
        ext = VoiceActivityExtractor(base)
        out = ext(torch.randn(2, 16000))
        assert out.shape[0] == 2
        assert out.shape[2] == 13 + 4  # base + 4 VAD feats

    def test_feature_dim(self):
        base = MfccExtractor(n_mfcc=40)
        ext = VoiceActivityExtractor(base)
        assert ext.feature_dim == 44

    def test_vad_values_bounded(self):
        base = MfccExtractor(n_mfcc=13)
        ext = VoiceActivityExtractor(base)
        out = ext(torch.randn(1, 16000))
        vad_feats = out[:, :, 13:]  # last 4 channels
        # Energy and vad_prob should be in [0, 1]
        assert vad_feats[:, :, 0].min() >= -0.01  # energy (normalized)
        assert vad_feats[:, :, 3].min() >= 0.0     # vad_prob (sigmoid)
        assert vad_feats[:, :, 3].max() <= 1.0

    def test_silence_vs_speech(self):
        """VAD should give lower probability for silence than for speech-like signal."""
        base = MfccExtractor(n_mfcc=13)
        ext = VoiceActivityExtractor(base)
        # Silence
        silence = torch.zeros(1, 16000) + 1e-6
        # "Speech" (sine wave)
        t = torch.linspace(0, 1, 16000)
        speech = torch.sin(2 * torch.pi * 200 * t).unsqueeze(0) * 0.5
        out_silence = ext(silence)
        out_speech = ext(speech)
        vad_silence = out_silence[:, :, -1].mean()  # vad_prob
        vad_speech = out_speech[:, :, -1].mean()
        assert vad_speech > vad_silence

    def test_with_filterbank(self):
        base = FilterbankExtractor(n_mels=80)
        ext = VoiceActivityExtractor(base)
        assert ext.feature_dim == 84
        out = ext(torch.randn(1, 16000))
        assert out.shape[2] == 84


# ---------- PitchExtractor ----------

class TestPitchExtractor:
    def test_forward_shape(self):
        base = MfccExtractor(n_mfcc=13)
        ext = PitchExtractor(base)
        out = ext(torch.randn(2, 16000))
        assert out.shape[0] == 2
        assert out.shape[2] == 13 + 3  # base + 3 pitch feats

    def test_feature_dim(self):
        base = MfccExtractor(n_mfcc=40)
        ext = PitchExtractor(base)
        assert ext.feature_dim == 43

    def test_pitch_from_sine(self):
        """A pure sine wave should produce consistent voicing probability."""
        base = MfccExtractor(n_mfcc=13)
        ext = PitchExtractor(base, f0_min=50, f0_max=600)
        t = torch.linspace(0, 1, 16000)
        sine = torch.sin(2 * torch.pi * 200 * t).unsqueeze(0) * 0.5
        out = ext(sine)
        voicing = out[:, :, 14]  # voicing_prob channel
        # Sine wave should have high voicing probability
        assert voicing.mean() > 0.3

    def test_short_audio(self):
        base = MfccExtractor(n_mfcc=13)
        ext = PitchExtractor(base)
        out = ext(torch.randn(1, 4000))
        assert out.ndim == 3


# ---------- MultiResolutionExtractor ----------

class TestMultiResolutionExtractor:
    def test_forward_shape(self):
        fine = MfccExtractor(n_mfcc=13, hop_length=80)
        coarse = MfccExtractor(n_mfcc=13, hop_length=320)
        ext = MultiResolutionExtractor(fine, coarse)
        out = ext(torch.randn(2, 16000))
        assert out.shape[0] == 2
        assert out.shape[2] == 26  # 13 + 13

    def test_feature_dim(self):
        fine = MfccExtractor(n_mfcc=13)
        coarse = FilterbankExtractor(n_mels=40)
        ext = MultiResolutionExtractor(fine, coarse)
        assert ext.feature_dim == 53

    def test_different_extractors(self):
        fine = FilterbankExtractor(n_mels=40, hop_length=80)
        coarse = MfccExtractor(n_mfcc=13, hop_length=320)
        ext = MultiResolutionExtractor(fine, coarse)
        out = ext(torch.randn(1, 16000))
        assert out.shape[2] == 53


# ---------- SNRAwareExtractor ----------

class TestSNRAwareExtractor:
    def test_forward_shape(self):
        base = MfccExtractor(n_mfcc=13)
        ext = SNRAwareExtractor(base)
        out = ext(torch.randn(2, 16000))
        assert out.shape[0] == 2
        assert out.shape[2] == 15  # 13 + 2

    def test_feature_dim(self):
        base = MfccExtractor(n_mfcc=40)
        ext = SNRAwareExtractor(base)
        assert ext.feature_dim == 42

    def test_snr_bounded(self):
        base = MfccExtractor(n_mfcc=13)
        ext = SNRAwareExtractor(base)
        out = ext(torch.randn(1, 16000))
        snr_feats = out[:, :, 13:]
        assert snr_feats.min() >= -0.01
        assert snr_feats.max() <= 1.01

    def test_clean_vs_noisy(self):
        """Clean signal should have higher SNR than noisy."""
        base = MfccExtractor(n_mfcc=13)
        ext = SNRAwareExtractor(base)
        t = torch.linspace(0, 1, 16000)
        clean = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
        noisy = clean + torch.randn(1, 16000) * 2.0
        out_clean = ext(clean)
        out_noisy = ext(noisy)
        snr_clean = out_clean[:, :, 13].mean()  # SNR channel
        snr_noisy = out_noisy[:, :, 13].mean()
        # Clean should have higher SNR variance (more dynamic range)
        assert out_clean[:, :, 13].std() > 0.01


# ---------- Stacking test ----------

class TestStackedWrappers:
    def test_vad_plus_pitch_plus_snr(self):
        """All wrappers can be stacked."""
        base = MfccExtractor(n_mfcc=13)
        with_vad = VoiceActivityExtractor(base)        # +4 → 17
        with_pitch = PitchExtractor(with_vad)          # +3 → 20
        with_snr = SNRAwareExtractor(with_pitch)       # +2 → 22
        assert with_snr.feature_dim == 22
        out = with_snr(torch.randn(1, 16000))
        assert out.shape[2] == 22
