"""Tests for BC-ResNet classifier head and SubSpectralNorm."""
import pytest
import torch

from ww_trainer.subspectralnorm import SubSpectralNorm
from ww_trainer.model import BCResNetHead


class TestSubSpectralNorm:
    def test_forward_shape_preserved(self):
        ssn = SubSpectralNorm(16, spec_groups=4)
        x = torch.randn(2, 16, 8, 50)
        out = ssn(x)
        assert out.shape == (2, 16, 8, 50)

    def test_affine_all(self):
        ssn = SubSpectralNorm(8, spec_groups=2, affine="All")
        assert hasattr(ssn, "weight")
        assert hasattr(ssn, "bias")
        x = torch.randn(1, 8, 4, 20)
        out = ssn(x)
        assert out.shape == x.shape

    def test_affine_sub(self):
        ssn = SubSpectralNorm(8, spec_groups=2, affine="Sub")
        assert ssn.ssnorm.affine is True
        x = torch.randn(1, 8, 4, 20)
        out = ssn(x)
        assert out.shape == x.shape

    def test_dim_width(self):
        """SubSpectralNorm along width dimension (dim=-1)."""
        ssn = SubSpectralNorm(8, spec_groups=5, dim=-1)
        x = torch.randn(1, 8, 20, 10)  # W=10, groups=5 → 10/5=2
        out = ssn(x)
        assert out.shape == x.shape

    def test_instance_norm_mode(self):
        ssn = SubSpectralNorm(4, spec_groups=2, batch=False)
        x = torch.randn(2, 4, 4, 10)
        out = ssn(x)
        assert out.shape == x.shape


class TestBCResNetHead:
    @pytest.mark.parametrize("tau", [1, 1.5, 2, 3, 6, 8])
    def test_forward_shape(self, tau):
        head = BCResNetHead(input_size=40, tau=tau, device="cpu")
        feats = torch.randn(2, 100, 40)  # [B, T, F]
        out = head(feats)
        assert out.shape == (2,), f"tau={tau}: expected (2,), got {out.shape}"

    def test_embed_shape(self):
        head = BCResNetHead(input_size=40, tau=1, device="cpu")
        feats = torch.randn(2, 100, 40)
        emb = head.embed(feats)
        assert emb.ndim == 2
        assert emb.shape[0] == 2

    def test_gradients_flow(self):
        head = BCResNetHead(input_size=40, tau=1, device="cpu")
        feats = torch.randn(1, 100, 40, requires_grad=True)
        out = head(feats)
        out.sum().backward()
        assert feats.grad is not None

    def test_export_to_onnx(self, tmp_path):
        head = BCResNetHead(input_size=40, tau=1, device="cpu")
        head.eval()
        out_path = str(tmp_path / "bcresnet.onnx")
        head.export_to_onnx(out_path)
        assert (tmp_path / "bcresnet.onnx").exists()

    def test_onnx_inference(self, tmp_path):
        """Exported ONNX model produces same output as PyTorch."""
        import numpy as np
        import onnxruntime as ort

        head = BCResNetHead(input_size=40, tau=1, device="cpu")
        head.eval()
        out_path = str(tmp_path / "bcresnet.onnx")
        head.export_to_onnx(out_path)

        feats = torch.randn(1, 100, 40)
        with torch.no_grad():
            pt_out = head(feats).numpy()

        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input_features": feats.numpy()})[0]

        np.testing.assert_allclose(pt_out, ort_out, atol=1e-4)

    def test_small_input(self):
        """Handles very short audio (few frames)."""
        head = BCResNetHead(input_size=40, tau=1, device="cpu")
        feats = torch.randn(1, 10, 40)
        out = head(feats)
        assert out.shape == (1,)

    def test_different_n_mels(self):
        """Works with different feature dimensions."""
        for n_mels in [20, 40, 80]:
            head = BCResNetHead(input_size=n_mels, tau=1, device="cpu")
            feats = torch.randn(1, 100, n_mels)
            out = head(feats)
            assert out.shape == (1,)
