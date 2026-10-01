"""Smoke tests for MixConvHead classifier."""
import torch

from ww_trainer.model import MixConvHead


class TestMixConvHead:
    """Tests for MixConvHead forward/embed with dummy features."""

    def test_forward_default(self) -> None:
        head = MixConvHead(input_size=40, device="cpu")
        feats = torch.randn(2, 50, 40)
        logits = head(feats)
        assert logits.shape == (2,)

    def test_embed_default(self) -> None:
        head = MixConvHead(input_size=40, device="cpu")
        feats = torch.randn(2, 50, 40)
        emb = head.embed(feats)
        assert emb.shape == (2, 64)  # default filters=64

    def test_custom_config(self) -> None:
        head = MixConvHead(
            input_size=20, n_blocks=2, filters=48,
            kernel_groups=[[3], [7]], device="cpu",
        )
        feats = torch.randn(4, 30, 20)
        logits = head(feats)
        assert logits.shape == (4,)

    def test_single_sample(self) -> None:
        head = MixConvHead(input_size=40, device="cpu")
        feats = torch.randn(1, 10, 40)
        logits = head(feats)
        assert logits.shape == (1,)

    def test_registered_in_factory(self) -> None:
        from ww_trainer.factory import HEAD_REGISTRY
        assert "mixconv" in HEAD_REGISTRY
        cls, valid_kwargs = HEAD_REGISTRY["mixconv"]
        assert cls is MixConvHead
        assert "filters" in valid_kwargs
