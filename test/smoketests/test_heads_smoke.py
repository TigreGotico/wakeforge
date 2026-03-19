"""Smoke tests: every classifier head produces correct output with dummy features."""
import pytest
import torch

# Dummy features: [batch=2, time=50, features=13]
DUMMY_FEATS = torch.randn(2, 50, 13)
INPUT_SIZE = 13


class TestFfnHead:
    def test_forward(self) -> None:
        from ww_trainer.model import FfnClassifierHead
        head = FfnClassifierHead(input_size=INPUT_SIZE, hidden_dim=32, device="cpu")
        out = head(DUMMY_FEATS)
        assert out.shape == (2,)

    def test_embed(self) -> None:
        from ww_trainer.model import FfnClassifierHead
        head = FfnClassifierHead(input_size=INPUT_SIZE, hidden_dim=32, device="cpu")
        emb = head.embed(DUMMY_FEATS)
        assert emb.shape == (2, 32)


class TestGruHead:
    def test_forward(self) -> None:
        from ww_trainer.model import GruClassifierHead
        head = GruClassifierHead(input_size=INPUT_SIZE, hidden_dim=32, device="cpu")
        out = head(DUMMY_FEATS)
        assert out.shape == (2,)

    def test_embed(self) -> None:
        from ww_trainer.model import GruClassifierHead
        head = GruClassifierHead(input_size=INPUT_SIZE, hidden_dim=32, device="cpu")
        emb = head.embed(DUMMY_FEATS)
        assert emb.ndim == 2 and emb.shape[0] == 2


class TestCnnHead:
    def test_forward(self) -> None:
        from ww_trainer.model import CnnClassifierHead
        head = CnnClassifierHead(input_size=INPUT_SIZE, conv_dim=16, linear_dim=16, device="cpu")
        out = head(DUMMY_FEATS)
        assert out.shape == (2,)


class TestBCResNetHead:
    def test_forward(self) -> None:
        from ww_trainer.model import BCResNetHead
        head = BCResNetHead(input_size=INPUT_SIZE, tau=1, device="cpu")
        out = head(DUMMY_FEATS)
        assert out.shape == (2,)


class TestTCResNetHead:
    def test_forward(self) -> None:
        from ww_trainer.model import TCResNetHead
        head = TCResNetHead(input_size=INPUT_SIZE, variant=8, channels=16, device="cpu")
        out = head(DUMMY_FEATS)
        assert out.shape == (2,)


class TestDSCNNHead:
    def test_forward(self) -> None:
        from ww_trainer.model import DSCNNHead
        head = DSCNNHead(input_size=INPUT_SIZE, size="S", device="cpu")
        out = head(DUMMY_FEATS)
        assert out.shape == (2,)


class TestMatchboxNetHead:
    def test_forward(self) -> None:
        from ww_trainer.model import MatchboxNetHead
        head = MatchboxNetHead(input_size=INPUT_SIZE, B=1, R=1, C=16, device="cpu")
        out = head(DUMMY_FEATS)
        assert out.shape == (2,)


class TestRes15Head:
    def test_forward(self) -> None:
        from ww_trainer.model import Res15Head
        head = Res15Head(input_size=INPUT_SIZE, channels=16, device="cpu")
        out = head(DUMMY_FEATS)
        assert out.shape == (2,)


class TestKWTHead:
    def test_forward(self) -> None:
        from ww_trainer.model import KWTHead
        head = KWTHead(input_size=INPUT_SIZE, d_model=16, n_heads=2, n_layers=1, device="cpu")
        out = head(DUMMY_FEATS)
        assert out.shape == (2,)


class TestConformerHead:
    def test_forward(self) -> None:
        from ww_trainer.model import ConformerHead
        head = ConformerHead(input_size=INPUT_SIZE, d_model=16, n_heads=2, n_layers=1, device="cpu")
        out = head(DUMMY_FEATS)
        assert out.shape == (2,)


class TestCRNNHead:
    def test_forward(self) -> None:
        from ww_trainer.model import CRNNHead
        head = CRNNHead(input_size=INPUT_SIZE, conv_channels=8, gru_hidden=16, device="cpu")
        out = head(DUMMY_FEATS)
        assert out.shape == (2,)
