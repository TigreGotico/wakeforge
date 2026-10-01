"""Smoke tests: feature enrichment wrappers on top of MfccExtractor."""
import torch

DUMMY_WAV = torch.randn(2, 16000)


class TestVoiceActivityWrapper:
    def test_forward(self) -> None:
        from ww_trainer.feats import MfccExtractor, VoiceActivityExtractor
        base = MfccExtractor(n_mfcc=13)
        ext = VoiceActivityExtractor(base)
        out = ext(DUMMY_WAV)
        assert out.ndim == 3
        assert out.shape[2] == 13 + 4  # base + 4 VAD features


class TestPitchWrapper:
    def test_forward(self) -> None:
        from ww_trainer.feats import MfccExtractor, PitchExtractor
        base = MfccExtractor(n_mfcc=13)
        ext = PitchExtractor(base)
        out = ext(DUMMY_WAV)
        assert out.ndim == 3
        assert out.shape[2] == 13 + 3  # base + 3 pitch features


class TestSNRAwareWrapper:
    def test_forward(self) -> None:
        from ww_trainer.feats import MfccExtractor, SNRAwareExtractor
        base = MfccExtractor(n_mfcc=13)
        ext = SNRAwareExtractor(base)
        out = ext(DUMMY_WAV)
        assert out.ndim == 3
        assert out.shape[2] == 13 + 2  # base + 2 SNR features


class TestMultiResolutionWrapper:
    def test_forward(self) -> None:
        from ww_trainer.feats import MfccExtractor, MultiResolutionExtractor
        fine = MfccExtractor(n_mfcc=13, n_fft=400, hop_length=160)
        coarse = MfccExtractor(n_mfcc=13, n_fft=800, hop_length=320)
        ext = MultiResolutionExtractor(fine, coarse)
        out = ext(DUMMY_WAV)
        assert out.ndim == 3
        assert out.shape[2] == 26  # 13 + 13


class TestMarkovTransitionWrapper:
    def test_forward_unfitted(self) -> None:
        from ww_trainer.feats import MfccExtractor, MarkovTransitionExtractor
        base = MfccExtractor(n_mfcc=13)
        ext = MarkovTransitionExtractor(base, n_codes=8, order=1)
        out = ext(DUMMY_WAV)
        assert out.ndim == 3
        assert out.shape[2] == 13 + 8

    def test_forward_fitted(self) -> None:
        from ww_trainer.feats import MfccExtractor, MarkovTransitionExtractor
        base = MfccExtractor(n_mfcc=13)
        ext = MarkovTransitionExtractor(base, n_codes=8, order=1)
        ext.fit([torch.randn(16000) for _ in range(3)])
        out = ext(DUMMY_WAV)
        assert out.shape[2] == 13 + 8


class TestHMMStateWrapper:
    def test_forward_unfitted(self) -> None:
        from ww_trainer.feats import MfccExtractor, HMMStateExtractor
        base = MfccExtractor(n_mfcc=13)
        ext = HMMStateExtractor(base, n_states=4, n_codes=8)
        out = ext(DUMMY_WAV)
        assert out.shape[2] == 13 + 4

    def test_forward_fitted(self) -> None:
        from ww_trainer.feats import MfccExtractor, HMMStateExtractor
        base = MfccExtractor(n_mfcc=13)
        ext = HMMStateExtractor(base, n_states=4, n_codes=8)
        ext.fit([torch.randn(16000) for _ in range(3)], n_iter=1)
        out = ext(DUMMY_WAV)
        assert out.shape[2] == 13 + 4


class TestStackedWrappers:
    """Test stacking multiple wrappers."""

    def test_vad_plus_pitch(self) -> None:
        from ww_trainer.feats import MfccExtractor, VoiceActivityExtractor, PitchExtractor
        base = MfccExtractor(n_mfcc=13)
        ext = PitchExtractor(VoiceActivityExtractor(base))
        out = ext(DUMMY_WAV)
        assert out.shape[2] == 13 + 4 + 3  # mfcc + vad + pitch

    def test_vad_plus_snr(self) -> None:
        from ww_trainer.feats import MfccExtractor, VoiceActivityExtractor, SNRAwareExtractor
        base = MfccExtractor(n_mfcc=13)
        ext = SNRAwareExtractor(VoiceActivityExtractor(base))
        out = ext(DUMMY_WAV)
        assert out.shape[2] == 13 + 4 + 2
