"""Tests for new loss functions: Focal, LabelSmoothing, ArcFace, Center,
NTXent, SupCon, ProxyNCA, MultiSimilarity."""
import pytest
import torch

from ww_trainer.loss import (
    FocalLoss,
    LabelSmoothingBCE,
    ArcFaceLoss,
    CenterLoss,
    NTXentLoss,
    SupConLoss,
    ProxyNCALoss,
    MultiSimilarityLoss,
    LossManager,
)


class TestFocalLoss:
    def test_output_scalar(self):
        loss = FocalLoss()
        logits = torch.randn(8)
        labels = torch.randint(0, 2, (8,)).float()
        out = loss(logits, labels)
        assert out.ndim == 0
        assert out.item() >= 0

    def test_easy_examples_downweighted(self):
        """Focal loss should be lower than BCE for easy examples."""
        focal = FocalLoss(gamma=2.0)
        # Easy example: high logit for positive
        logits = torch.tensor([5.0])
        labels = torch.tensor([1.0])
        focal_val = focal(logits, labels)
        bce_val = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        assert focal_val < bce_val

    def test_gamma_zero_uniform_alpha_scales_bce(self):
        """With gamma=0 and alpha=0.5, focal = 0.5 * BCE."""
        focal = FocalLoss(alpha=0.5, gamma=0.0)
        logits = torch.randn(16)
        labels = torch.randint(0, 2, (16,)).float()
        focal_val = focal(logits, labels)
        bce_val = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        torch.testing.assert_close(focal_val, bce_val * 0.5, atol=1e-5, rtol=1e-5)


class TestLabelSmoothingBCE:
    def test_output_scalar(self):
        loss = LabelSmoothingBCE(smoothing=0.1)
        out = loss(torch.randn(8), torch.randint(0, 2, (8,)).float())
        assert out.ndim == 0

    def test_smoothing_zero_equals_bce(self):
        ls = LabelSmoothingBCE(smoothing=0.0)
        logits = torch.randn(16)
        labels = torch.randint(0, 2, (16,)).float()
        ls_val = ls(logits, labels)
        bce_val = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        torch.testing.assert_close(ls_val, bce_val, atol=1e-5, rtol=1e-5)


class TestArcFaceLoss:
    def test_output_scalar(self):
        loss = ArcFaceLoss(embed_dim=32)
        embeds = torch.randn(8, 32)
        labels = torch.randint(0, 2, (8,))
        out = loss(embeds, labels)
        assert out.ndim == 0
        assert out.item() >= 0

    def test_gradient_flow(self):
        loss = ArcFaceLoss(embed_dim=16)
        embeds = torch.randn(4, 16, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1])
        out = loss(embeds, labels)
        out.backward()
        assert embeds.grad is not None


class TestCenterLoss:
    def test_output_scalar(self):
        loss = CenterLoss(embed_dim=32)
        out = loss(torch.randn(8, 32), torch.randint(0, 2, (8,)))
        assert out.ndim == 0
        assert out.item() >= 0

    def test_zero_at_centers(self):
        """Loss should be zero when embeddings equal their centers."""
        loss = CenterLoss(embed_dim=4, num_classes=2)
        labels = torch.tensor([0, 0, 1, 1])
        embeds = loss.centers[labels].detach().clone()
        out = loss(embeds, labels)
        assert out.item() < 1e-5


class TestNTXentLoss:
    def test_output_scalar(self):
        loss = NTXentLoss(temperature=0.1)
        embeds = torch.randn(8, 32)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        out = loss(embeds, labels)
        assert out.ndim == 0

    def test_gradient_flow(self):
        loss = NTXentLoss()
        embeds = torch.randn(6, 16, requires_grad=True)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        out = loss(embeds, labels)
        out.backward()
        assert embeds.grad is not None


class TestSupConLoss:
    def test_output_scalar(self):
        loss = SupConLoss(temperature=0.1)
        embeds = torch.randn(8, 32)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        out = loss(embeds, labels)
        assert out.ndim == 0

    def test_no_positives_returns_zero(self):
        loss = SupConLoss()
        embeds = torch.randn(4, 16)
        labels = torch.tensor([0, 1, 2, 3])  # all unique
        out = loss(embeds, labels)
        assert out.item() == 0.0


class TestProxyNCALoss:
    def test_output_scalar(self):
        loss = ProxyNCALoss(embed_dim=32)
        out = loss(torch.randn(8, 32), torch.randint(0, 2, (8,)))
        assert out.ndim == 0
        assert out.item() >= 0

    def test_gradient_to_proxies(self):
        loss = ProxyNCALoss(embed_dim=16)
        embeds = torch.randn(4, 16)
        labels = torch.tensor([0, 1, 0, 1])
        out = loss(embeds, labels)
        out.backward()
        assert loss.proxies.grad is not None


class TestMultiSimilarityLoss:
    def test_output_scalar(self):
        loss = MultiSimilarityLoss()
        embeds = torch.randn(8, 32)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        out = loss(embeds, labels)
        assert out.ndim == 0

    def test_gradient_flow(self):
        loss = MultiSimilarityLoss()
        embeds = torch.randn(8, 16, requires_grad=True)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        out = loss(embeds, labels)
        out.backward()
        assert embeds.grad is not None


class TestLossManagerNewLosses:
    """Test that new losses are properly registered in LossManager."""

    @pytest.mark.parametrize("name", [
        "focal", "label_smoothing_bce", "arcface", "center",
        "ntxent", "supcon", "proxy_nca", "multi_similarity",
    ])
    def test_loss_manager_registers(self, name):
        cfg = {"name": name, "weight": 1.0, "embed_dim": 32}
        manager = LossManager([cfg], device="cpu")
        assert len(manager.losses) == 1
        assert manager.losses[0]["name"] == name
