"""Smoke tests: every standalone extractor produces correct output shape with dummy audio."""
import pytest
import torch

DUMMY_WAV = torch.randn(2, 16000)  # batch of 2, 1-second audio


class TestMfccExtractor:
    def test_forward(self) -> None:
        from ww_trainer.feats import MfccExtractor
        ext = MfccExtractor(n_mfcc=13)
        out = ext(DUMMY_WAV)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 13

    def test_feature_dim(self) -> None:
        from ww_trainer.feats import MfccExtractor
        ext = MfccExtractor(n_mfcc=40)
        assert ext.feature_dim == 40


class TestFilterbankExtractor:
    def test_forward(self) -> None:
        from ww_trainer.feats import FilterbankExtractor
        ext = FilterbankExtractor(n_mels=40)
        out = ext(DUMMY_WAV)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 40


class TestSincNetExtractor:
    def test_forward(self) -> None:
        from ww_trainer.feats import SincNetExtractor
        ext = SincNetExtractor(n_filters=32)
        out = ext(DUMMY_WAV)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 32


class TestGammatoneExtractor:
    def test_forward(self) -> None:
        from ww_trainer.feats import GammatoneExtractor
        ext = GammatoneExtractor(n_filters=32)
        out = ext(DUMMY_WAV)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 32


class TestDeltaMfccExtractor:
    def test_forward(self) -> None:
        from ww_trainer.feats import MfccExtractor, DeltaExtractor
        base = MfccExtractor(n_mfcc=13)
        ext = DeltaExtractor(base)
        out = ext(DUMMY_WAV)
        assert out.ndim == 3
        assert out.shape[2] == 39  # 13 * 3


class TestDeltaFilterbankExtractor:
    def test_forward(self) -> None:
        from ww_trainer.feats import FilterbankExtractor, DeltaExtractor
        base = FilterbankExtractor(n_mels=20)
        ext = DeltaExtractor(base)
        out = ext(DUMMY_WAV)
        assert out.shape[2] == 60  # 20 * 3


class TestLEAFExtractor:
    def test_forward(self) -> None:
        from ww_trainer.feats import LEAFExtractor
        ext = LEAFExtractor(n_filters=16)
        out = ext(DUMMY_WAV)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 16


class TestPLPExtractor:
    def test_forward(self) -> None:
        from ww_trainer.feats import PLPExtractor
        ext = PLPExtractor(n_plp=13)
        out = ext(DUMMY_WAV)
        assert out.ndim == 3
        assert out.shape[2] == 13


class TestPNCCExtractor:
    def test_forward(self) -> None:
        from ww_trainer.feats import PNCCExtractor
        ext = PNCCExtractor(n_pncc=13)
        out = ext(DUMMY_WAV)
        assert out.ndim == 3
        assert out.shape[2] == 13


class TestCQTExtractor:
    def test_forward(self) -> None:
        from ww_trainer.feats import CQTExtractor
        ext = CQTExtractor(n_bins=12, n_octaves=4)
        out = ext(DUMMY_WAV)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 48  # 12 * 4
