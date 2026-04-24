"""Tests for all new classifier heads and LEAF extractor."""
import pytest
import torch
import numpy as np

from ww_trainer.model import (
    AttentionPooling,
    TCResNetHead,
    DSCNNHead,
    MatchboxNetHead,
    Res15Head,
    KWTHead,
    ConformerHead,
    CRNNHead,
    OCSVMHead,
)
from ww_trainer.feats import LEAFExtractor


# ---------- AttentionPooling ----------

class TestAttentionPooling:
    def test_shape(self):
        pool = AttentionPooling(dim=64, n_heads=4)
        x = torch.randn(2, 50, 64)
        out = pool(x)
        assert out.shape == (2, 64)

    def test_gradient_flow(self):
        pool = AttentionPooling(dim=32)
        x = torch.randn(1, 10, 32, requires_grad=True)
        out = pool(x)
        out.sum().backward()
        assert x.grad is not None


# ---------- TCResNetHead ----------

class TestTCResNetHead:
    @pytest.mark.parametrize("variant", [8, 14])
    def test_forward_shape(self, variant):
        head = TCResNetHead(input_size=40, variant=variant, channels=32, device="cpu")
        feats = torch.randn(2, 100, 40)
        out = head(feats)
        assert out.shape == (2,)

    def test_embed(self):
        head = TCResNetHead(input_size=40, variant=8, channels=32, device="cpu")
        feats = torch.randn(2, 100, 40)
        emb = head.embed(feats)
        assert emb.shape == (2, 32)

    def test_export_onnx(self, tmp_path):
        head = TCResNetHead(input_size=40, variant=8, channels=16, device="cpu")
        head.eval()
        head.export_to_onnx(str(tmp_path / "tc.onnx"))
        assert (tmp_path / "tc.onnx").exists()


# ---------- DSCNNHead ----------

class TestDSCNNHead:
    @pytest.mark.parametrize("size", ["S", "M", "L"])
    def test_forward_shape(self, size):
        head = DSCNNHead(input_size=40, size=size, device="cpu")
        feats = torch.randn(2, 100, 40)
        out = head(feats)
        assert out.shape == (2,)

    def test_embed(self):
        head = DSCNNHead(input_size=40, size="S", device="cpu")
        feats = torch.randn(1, 100, 40)
        emb = head.embed(feats)
        assert emb.ndim == 2

    def test_export_onnx(self, tmp_path):
        head = DSCNNHead(input_size=40, size="S", device="cpu")
        head.eval()
        head.export_to_onnx(str(tmp_path / "dscnn.onnx"))
        assert (tmp_path / "dscnn.onnx").exists()


# ---------- MatchboxNetHead ----------

class TestMatchboxNetHead:
    def test_forward_shape(self):
        head = MatchboxNetHead(input_size=40, B=3, R=1, C=32, device="cpu")
        feats = torch.randn(2, 100, 40)
        out = head(feats)
        assert out.shape == (2,)

    def test_embed(self):
        head = MatchboxNetHead(input_size=40, B=3, R=1, C=32, device="cpu")
        feats = torch.randn(1, 100, 40)
        emb = head.embed(feats)
        assert emb.shape == (1, 64)  # C * 2

    def test_different_configs(self):
        for B, R, C in [(3, 1, 32), (3, 2, 32), (6, 2, 64)]:
            head = MatchboxNetHead(input_size=40, B=B, R=R, C=C, device="cpu")
            out = head(torch.randn(1, 100, 40))
            assert out.shape == (1,)

    def test_export_onnx(self, tmp_path):
        head = MatchboxNetHead(input_size=40, B=3, R=1, C=16, device="cpu")
        head.eval()
        head.export_to_onnx(str(tmp_path / "matchbox.onnx"))
        assert (tmp_path / "matchbox.onnx").exists()


# ---------- Res15Head ----------

class TestRes15Head:
    def test_forward_shape(self):
        head = Res15Head(input_size=40, channels=32, device="cpu")
        feats = torch.randn(2, 100, 40)
        out = head(feats)
        assert out.shape == (2,)

    def test_embed(self):
        head = Res15Head(input_size=40, channels=32, device="cpu")
        emb = head.embed(torch.randn(1, 100, 40))
        assert emb.shape == (1, 32)

    def test_export_onnx(self, tmp_path):
        head = Res15Head(input_size=40, channels=16, device="cpu")
        head.eval()
        head.export_to_onnx(str(tmp_path / "res15.onnx"))
        assert (tmp_path / "res15.onnx").exists()


# ---------- KWTHead ----------

class TestKWTHead:
    def test_forward_shape(self):
        head = KWTHead(input_size=40, patch_len=5, d_model=32,
                       n_heads=4, n_layers=2, device="cpu")
        feats = torch.randn(2, 100, 40)
        out = head(feats)
        assert out.shape == (2,)

    def test_embed(self):
        head = KWTHead(input_size=40, patch_len=5, d_model=32,
                       n_heads=4, n_layers=2, device="cpu")
        emb = head.embed(torch.randn(1, 100, 40))
        assert emb.shape == (1, 32)

    def test_short_input(self):
        head = KWTHead(input_size=40, patch_len=5, d_model=32,
                       n_heads=4, n_layers=2, device="cpu")
        feats = torch.randn(1, 10, 40)  # only 2 patches
        out = head(feats)
        assert out.shape == (1,)

    def test_gradient_flow(self):
        head = KWTHead(input_size=40, patch_len=5, d_model=32,
                       n_heads=4, n_layers=2, device="cpu")
        feats = torch.randn(1, 50, 40, requires_grad=True)
        out = head(feats)
        out.sum().backward()
        assert feats.grad is not None

    def test_export_onnx(self, tmp_path):
        head = KWTHead(input_size=40, patch_len=5, d_model=32,
                       n_heads=4, n_layers=2, device="cpu")
        head.eval()
        head.export_to_onnx(str(tmp_path / "kwt.onnx"))
        assert (tmp_path / "kwt.onnx").exists()

    def test_onnx_inference(self, tmp_path):
        """ONNX inference must use same sequence length as export (attention is shape-fixed)."""
        import numpy as np
        import onnxruntime as ort
        head = KWTHead(input_size=40, patch_len=5, d_model=32,
                       n_heads=4, n_layers=2, device="cpu")
        head.eval()
        path = str(tmp_path / "kwt.onnx")
        # Export with T=200 (ClassifierHead.export_to_onnx uses [1, 200, F])
        head.export_to_onnx(path)
        # Inference must use same T=200 since attention shapes are fixed at trace time
        feats = torch.randn(1, 200, 40)
        with torch.no_grad():
            pt_out = head(feats).numpy()
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input_features": feats.numpy()})[0]
        np.testing.assert_allclose(pt_out, ort_out, atol=1e-4)


# ---------- ConformerHead ----------

class TestConformerHead:
    def test_forward_shape(self):
        head = ConformerHead(input_size=40, d_model=32, n_heads=4,
                             n_layers=2, conv_kernel=7, device="cpu")
        feats = torch.randn(2, 100, 40)
        out = head(feats)
        assert out.shape == (2,)

    def test_embed(self):
        head = ConformerHead(input_size=40, d_model=32, n_heads=4,
                             n_layers=2, device="cpu")
        emb = head.embed(torch.randn(1, 50, 40))
        assert emb.shape == (1, 32)

    def test_gradient_flow(self):
        head = ConformerHead(input_size=40, d_model=32, n_heads=4,
                             n_layers=2, device="cpu")
        feats = torch.randn(1, 50, 40, requires_grad=True)
        out = head(feats)
        out.sum().backward()
        assert feats.grad is not None

    def test_export_onnx(self, tmp_path):
        head = ConformerHead(input_size=40, d_model=32, n_heads=4,
                             n_layers=2, conv_kernel=7, device="cpu")
        head.eval()
        head.export_to_onnx(str(tmp_path / "conformer.onnx"))
        assert (tmp_path / "conformer.onnx").exists()

    def test_onnx_inference(self, tmp_path):
        """ONNX inference must use same sequence length as export (attention is shape-fixed)."""
        import numpy as np
        import onnxruntime as ort
        head = ConformerHead(input_size=40, d_model=32, n_heads=4,
                             n_layers=2, conv_kernel=7, device="cpu")
        head.eval()
        path = str(tmp_path / "conformer.onnx")
        head.export_to_onnx(path)
        # Use T=200 matching the export dummy input
        feats = torch.randn(1, 200, 40)
        with torch.no_grad():
            pt_out = head(feats).numpy()
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input_features": feats.numpy()})[0]
        np.testing.assert_allclose(pt_out, ort_out, atol=1e-4)


# ---------- CRNNHead ----------

class TestCRNNHead:
    def test_forward_shape(self):
        head = CRNNHead(input_size=40, conv_channels=16, gru_hidden=32, device="cpu")
        feats = torch.randn(2, 100, 40)
        out = head(feats)
        assert out.shape == (2,)

    def test_embed(self):
        head = CRNNHead(input_size=40, conv_channels=16, gru_hidden=32, device="cpu")
        emb = head.embed(torch.randn(1, 100, 40))
        assert emb.shape == (1, 32)

    def test_different_input_sizes(self):
        for n_mels in [20, 40, 80]:
            head = CRNNHead(input_size=n_mels, conv_channels=16, gru_hidden=32, device="cpu")
            out = head(torch.randn(1, 100, n_mels))
            assert out.shape == (1,)

    def test_export_onnx(self, tmp_path):
        head = CRNNHead(input_size=40, conv_channels=16, gru_hidden=16, device="cpu")
        head.eval()
        head.export_to_onnx(str(tmp_path / "crnn.onnx"))
        assert (tmp_path / "crnn.onnx").exists()


# ---------- LEAFExtractor ----------

class TestLEAFExtractor:
    def test_forward_shape(self):
        ext = LEAFExtractor(sr=16000, n_filters=40)
        wav = torch.randn(2, 16000)
        out = ext(wav)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 40

    def test_feature_dim(self):
        ext = LEAFExtractor(n_filters=64)
        assert ext.feature_dim == 64

    def test_list_input(self):
        ext = LEAFExtractor(n_filters=40)
        wavs = [torch.randn(16000), torch.randn(8000)]
        out = ext(wavs)
        assert out.shape[0] == 2
        assert out.shape[2] == 40

    def test_learnable_params(self):
        ext = LEAFExtractor(n_filters=40)
        params = list(ext.parameters())
        assert len(params) > 0
        # Check key learnable params exist
        param_names = [n for n, _ in ext.named_parameters()]
        assert "center_hz" in param_names
        assert "bandwidth_hz" in param_names
        assert "lowpass_sigma" in param_names
        assert "pcen_alpha" in param_names

    def test_gradient_flow(self):
        ext = LEAFExtractor(n_filters=20)
        wav = torch.randn(1, 16000)
        out = ext(wav)
        out.mean().backward()
        assert ext.center_hz.grad is not None
        assert ext.bandwidth_hz.grad is not None

    def test_short_audio(self):
        ext = LEAFExtractor(n_filters=40)
        wav = torch.randn(1, 4000)  # 0.25 seconds
        out = ext(wav)
        assert out.ndim == 3
        assert out.shape[2] == 40


# ---------- OCSVMHead ----------

class TestOCSVMHead:
    def _make_head(self) -> OCSVMHead:
        return OCSVMHead(input_size=40, hidden_dim=64, embed_dim=32, device="cpu")

    def test_forward_shape_unfitted(self):
        head = self._make_head()
        feats = torch.randn(3, 50, 40)
        out = head(feats)
        assert out.shape == (3,), f"Expected (3,), got {out.shape}"

    def test_embed_shape(self):
        head = self._make_head()
        feats = torch.randn(3, 50, 40)
        emb = head.embed(feats)
        assert emb.shape == (3, 32), f"Expected (3, 32), got {emb.shape}"

    def test_fit_ocsvm(self):
        pytest.importorskip("sklearn")
        head = self._make_head()
        # Simulate a dataloader yielding (feats, labels) batches
        batches = [
            (torch.randn(4, 50, 40), torch.tensor([1, 1, 0, 1])),
            (torch.randn(4, 50, 40), torch.tensor([0, 1, 1, 0])),
        ]
        head.fit_ocsvm(batches)
        assert head._sv_vectors.shape[0] > 1  # at least one real support vector
        assert head._sv_vectors.shape[1] == 32

    def test_forward_shape_fitted(self):
        pytest.importorskip("sklearn")
        head = self._make_head()
        batches = [(torch.randn(4, 50, 40), torch.tensor([1, 1, 1, 1]))]
        head.fit_ocsvm(batches)
        feats = torch.randn(5, 50, 40)
        out = head(feats)
        assert out.shape == (5,)

    def test_onnx_export_unfitted(self, tmp_path):
        head = self._make_head()
        out_path = str(tmp_path / "ocsvm_head.onnx")
        head.export_to_onnx(out_path)
        import onnx
        onnx.checker.check_model(out_path)

    def test_onnx_export_fitted(self, tmp_path):
        pytest.importorskip("sklearn")
        head = self._make_head()
        batches = [(torch.randn(4, 50, 40), torch.tensor([1, 1, 1, 1]))]
        head.fit_ocsvm(batches)
        out_path = str(tmp_path / "ocsvm_head_fitted.onnx")
        head.export_to_onnx(out_path)
        import onnx
        onnx.checker.check_model(out_path)

    def test_onnx_inference_parity(self, tmp_path):
        pytest.importorskip("sklearn")
        import onnxruntime as ort
        head = self._make_head()
        head.eval()
        batches = [(torch.randn(8, 50, 40), torch.tensor([1] * 8))]
        head.fit_ocsvm(batches)
        out_path = str(tmp_path / "ocsvm_parity.onnx")
        head.export_to_onnx(out_path)
        feats = torch.randn(2, 50, 40)
        with torch.no_grad():
            torch_out = head(feats).numpy()
        sess = ort.InferenceSession(out_path)
        ort_out = sess.run(None, {"input_features": feats.numpy()})[0]
        np.testing.assert_allclose(torch_out, ort_out, rtol=1e-4, atol=1e-5)

    def test_registered_in_factory(self):
        from ww_trainer.factory import HEAD_REGISTRY
        assert "ocsvm" in HEAD_REGISTRY
        cls, valid_kwargs = HEAD_REGISTRY["ocsvm"]
        assert cls is OCSVMHead
        assert "embed_dim" in valid_kwargs

    def test_ocsvm_small_tier(self):
        from ww_trainer.tiers import get_tier
        tc = get_tier("ocsvm_small")
        assert tc.head_arch == "ocsvm"
        assert tc.embed_dim == 64
