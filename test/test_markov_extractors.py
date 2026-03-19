"""Tests for Markov-based feature extractors."""
import pytest
import torch

from ww_trainer.feats import (
    MfccExtractor,
    MarkovTransitionExtractor,
    HMMStateExtractor,
)


class TestMarkovTransitionExtractor:
    def test_forward_shape_untrained(self):
        base = MfccExtractor(n_mfcc=13)
        ext = MarkovTransitionExtractor(base, n_codes=16, order=2)
        wav = torch.randn(2, 16000)
        out = ext(wav)
        assert out.shape[0] == 2
        assert out.shape[2] == 13 + 16  # base + n_codes

    def test_feature_dim(self):
        base = MfccExtractor(n_mfcc=13)
        ext = MarkovTransitionExtractor(base, n_codes=32, order=2)
        assert ext.feature_dim == 13 + 32

    def test_fit_and_forward(self):
        base = MfccExtractor(n_mfcc=13)
        ext = MarkovTransitionExtractor(base, n_codes=8, order=1)
        # Train on synthetic audio
        audio_list = [torch.randn(8000) for _ in range(5)]
        ext.fit(audio_list)
        assert ext._fitted
        assert ext._codebook is not None
        assert ext._codebook.shape == (8, 13)

        # Forward should work after fitting
        out = ext(torch.randn(1, 16000))
        assert out.shape[2] == 13 + 8

    def test_transition_matrix_is_probability(self):
        base = MfccExtractor(n_mfcc=13)
        ext = MarkovTransitionExtractor(base, n_codes=8, order=1)
        ext.fit([torch.randn(8000) for _ in range(3)])
        # Each row should sum to ~1
        row_sums = ext._transition_matrix.sum(dim=1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=0.01, rtol=0.01)


class TestHMMStateExtractor:
    def test_forward_shape_untrained(self):
        base = MfccExtractor(n_mfcc=13)
        ext = HMMStateExtractor(base, n_states=4, n_codes=16)
        wav = torch.randn(2, 16000)
        out = ext(wav)
        assert out.shape[0] == 2
        assert out.shape[2] == 13 + 4  # base + n_states

    def test_feature_dim(self):
        base = MfccExtractor(n_mfcc=13)
        ext = HMMStateExtractor(base, n_states=8, n_codes=16)
        assert ext.feature_dim == 13 + 8

    def test_forward_posteriors_sum_to_one(self):
        base = MfccExtractor(n_mfcc=13)
        ext = HMMStateExtractor(base, n_states=4, n_codes=8)
        out = ext(torch.randn(1, 16000))
        # HMM posteriors (last 4 dims) should sum to ~1 per frame
        posteriors = out[:, :, 13:]
        sums = posteriors.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=0.01, rtol=0.01)

    def test_fit_with_markovonnx(self):
        pytest.importorskip("markovonnx")
        base = MfccExtractor(n_mfcc=13)
        ext = HMMStateExtractor(base, n_states=4, n_codes=8)
        audio_list = [torch.randn(8000) for _ in range(3)]
        ext.fit(audio_list, n_iter=2)
        assert ext._fitted
        assert ext._codebook is not None

        out = ext(torch.randn(1, 16000))
        assert out.shape[2] == 13 + 4


class TestStackWithMarkov:
    def test_vad_plus_markov(self):
        from ww_trainer.feats import VoiceActivityExtractor
        base = MfccExtractor(n_mfcc=13)
        vad = VoiceActivityExtractor(base)          # 13 + 4 = 17
        markov = MarkovTransitionExtractor(vad, n_codes=8, order=1)  # 17 + 8 = 25
        assert markov.feature_dim == 25
        out = markov(torch.randn(1, 16000))
        assert out.shape[2] == 25
