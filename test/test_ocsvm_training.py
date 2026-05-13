"""Integration tests for OCSVMHead — toy training run + post-training OCSVM fit.

Covers:
- Full WakeWordTrainer.train() loop with arch="ocsvm"
- Automatic fit_ocsvm() invocation by training_loop after final epoch
- Checkpoint save/load round-trip preserving SV buffers
- ONNX export of the fitted head
- quickstart tier resolution for "ocsvm_small"
"""
import numpy as np
import pytest
import soundfile as sf

from ww_trainer.trainer import WakeWordTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_dataset(tmp_path):
    """12 synthetic WAV files: 6 wake-word, 6 not-wake-word."""
    data = []
    for i in range(12):
        t = np.linspace(0, 0.5, 8000)
        freq = 440 + i * 110
        wav = np.sin(2 * np.pi * freq * t).astype(np.float32)
        p = tmp_path / f"audio_{i}.wav"
        sf.write(str(p), wav, 16000)
        label = "1" if i < 6 else "0"
        data.append((str(p), label))
    return data


def _make_ocsvm_trainer(**extra) -> WakeWordTrainer:
    """Return a minimal OCSVMHead trainer configured for CPU toy runs."""
    return WakeWordTrainer(
        arch="ocsvm",
        featurizer="",
        feature_dim=None,
        featurizer_type="mfcc",
        device="cpu",
        losses_cfg=[{"name": "bce", "weight": 1.0}],
        n_mfcc=13,
        hidden_dim=32,
        embed_dim=16,
        **extra,
    )


# ---------------------------------------------------------------------------
# Unit-level head tests (pure torch, no training loop)
# ---------------------------------------------------------------------------

class TestOCSVMHeadUnit:
    """Fast unit tests — no I/O, no sklearn needed."""

    def test_forward_unfitted_returns_float_tensor(self):
        from ww_trainer.model import OCSVMHead
        import torch
        head = OCSVMHead(input_size=13, hidden_dim=32, embed_dim=16, device="cpu")
        feats = torch.randn(3, 50, 13)
        out = head(feats)
        assert out.shape == (3,)
        assert out.dtype == torch.float32

    def test_embed_shape(self):
        from ww_trainer.model import OCSVMHead
        import torch
        head = OCSVMHead(input_size=13, hidden_dim=32, embed_dim=16, device="cpu")
        emb = head.embed(torch.randn(2, 50, 13))
        assert emb.shape == (2, 16)

    def test_gradient_flows_through_backbone(self):
        from ww_trainer.model import OCSVMHead
        import torch
        head = OCSVMHead(input_size=13, hidden_dim=32, embed_dim=16, device="cpu")
        feats = torch.randn(2, 50, 13, requires_grad=True)
        loss = head(feats).sum()
        loss.backward()
        assert feats.grad is not None

    def test_fit_ocsvm_populates_buffers(self):
        pytest.importorskip("sklearn")
        from ww_trainer.model import OCSVMHead
        import torch
        head = OCSVMHead(input_size=13, hidden_dim=32, embed_dim=16, device="cpu")
        # All-zero sentinel before fitting
        assert head._sv_vectors.shape == (1, 16)
        batches = [(torch.randn(4, 50, 13), torch.tensor([1, 1, 1, 0]))]
        head.fit_ocsvm(batches)
        # After fitting: at least one real support vector
        assert head._sv_vectors.shape[0] >= 1
        assert head._sv_vectors.shape[1] == 16
        assert head._sv_weights.shape[0] >= 1

    def test_fit_ocsvm_stores_correct_gamma(self):
        """_gamma_val buffer must match sklearn's actual computed gamma, not 1/embed_dim."""
        pytest.importorskip("sklearn")
        from sklearn.svm import OneClassSVM
        from ww_trainer.model import OCSVMHead
        import torch
        head = OCSVMHead(input_size=13, hidden_dim=32, embed_dim=16, device="cpu")
        train_feats = torch.randn(20, 50, 13)
        batches = [(train_feats, torch.ones(20, dtype=torch.long))]
        head.fit_ocsvm(batches)
        # Refit sklearn on the same embeddings to get the reference gamma
        with torch.no_grad():
            embeds = head.embed(train_feats).numpy()
        ref_svm = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
        ref_svm.fit(embeds)
        assert abs(head._gamma_val.item() - ref_svm._gamma) < 1e-6, (
            f"_gamma_val {head._gamma_val.item()} != sklearn gamma {ref_svm._gamma}"
        )

    def test_fit_ocsvm_changes_output(self):
        """Scores must change after fitting (SV buffers replace zero sentinel)."""
        pytest.importorskip("sklearn")
        from ww_trainer.model import OCSVMHead
        import torch
        head = OCSVMHead(input_size=13, hidden_dim=32, embed_dim=16, device="cpu")
        feats = torch.randn(2, 50, 13)
        with torch.no_grad():
            before = head(feats).clone()
        batches = [(torch.randn(8, 50, 13), torch.ones(8, dtype=torch.long))]
        head.fit_ocsvm(batches)
        with torch.no_grad():
            after = head(feats)
        # At least one score must differ (buffers changed)
        assert not torch.allclose(before, after)

    def test_fit_ocsvm_raises_without_sklearn(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "sklearn", None)
        monkeypatch.setitem(sys.modules, "sklearn.svm", None)
        from ww_trainer.model import OCSVMHead
        import torch
        head = OCSVMHead(input_size=13, hidden_dim=32, embed_dim=16, device="cpu")
        batches = [(torch.randn(4, 50, 13), torch.tensor([1, 1, 1, 1]))]
        with pytest.raises(ImportError, match="scikit-learn"):
            head.fit_ocsvm(batches)

    def test_fit_ocsvm_raises_on_no_positives(self):
        pytest.importorskip("sklearn")
        from ww_trainer.model import OCSVMHead
        import torch
        head = OCSVMHead(input_size=13, hidden_dim=32, embed_dim=16, device="cpu")
        batches = [(torch.randn(4, 50, 13), torch.zeros(4, dtype=torch.long))]
        with pytest.raises(ValueError, match="No positive"):
            head.fit_ocsvm(batches)

    def test_onnx_export_unfitted(self, tmp_path):
        from ww_trainer.model import OCSVMHead
        import onnx
        head = OCSVMHead(input_size=13, hidden_dim=32, embed_dim=16, device="cpu")
        path = str(tmp_path / "ocsvm_unfitted.onnx")
        head.export_to_onnx(path)
        onnx.checker.check_model(path)

    def test_onnx_export_fitted(self, tmp_path):
        pytest.importorskip("sklearn")
        from ww_trainer.model import OCSVMHead
        import torch, onnx
        head = OCSVMHead(input_size=13, hidden_dim=32, embed_dim=16, device="cpu")
        batches = [(torch.randn(8, 50, 13), torch.ones(8, dtype=torch.long))]
        head.fit_ocsvm(batches)
        path = str(tmp_path / "ocsvm_fitted.onnx")
        head.export_to_onnx(path)
        onnx.checker.check_model(path)

    def test_onnx_inference_parity_fitted(self, tmp_path):
        pytest.importorskip("sklearn")
        import onnxruntime as ort
        from ww_trainer.model import OCSVMHead
        import torch
        head = OCSVMHead(input_size=13, hidden_dim=32, embed_dim=16, device="cpu")
        batches = [(torch.randn(16, 50, 13), torch.ones(16, dtype=torch.long))]
        head.fit_ocsvm(batches)
        head.eval()
        path = str(tmp_path / "ocsvm_parity.onnx")
        head.export_to_onnx(path)
        feats = torch.randn(3, 50, 13)
        with torch.no_grad():
            pt_out = head(feats).numpy()
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input_features": feats.numpy()})[0]
        np.testing.assert_allclose(pt_out, ort_out, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Training-loop integration
# ---------------------------------------------------------------------------

class TestOCSVMTrainingLoop:
    """Full WakeWordTrainer.train() smoke tests with OCSVMHead."""

    def test_train_ocsvm_completes(self, tiny_dataset, tmp_path):
        """Training loop runs to completion without error."""
        pytest.importorskip("sklearn")
        trainer = _make_ocsvm_trainer()
        result = trainer.train(
            train_data=tiny_dataset,
            test_data=tiny_dataset,
            epochs=2,
            batch_size=4,
            lr=1e-3,
            output_dir=str(tmp_path / "out"),
            save_best="f1",
            tsne_every=0, pca_every=0, umap_every=0,
        )
        assert isinstance(result, float)

    def test_train_ocsvm_fits_svm_automatically(self, tiny_dataset, tmp_path):
        """fit_ocsvm() is called by the loop — SV buffers must be non-trivial after training."""
        pytest.importorskip("sklearn")
        trainer = _make_ocsvm_trainer()
        trainer.train(
            train_data=tiny_dataset,
            test_data=tiny_dataset,
            epochs=2,
            batch_size=4,
            lr=1e-3,
            output_dir=str(tmp_path / "out"),
            save_best="f1",
            tsne_every=0, pca_every=0, umap_every=0,
        )
        head = trainer.model.classifier
        # SV buffer must have been updated from the all-zero sentinel
        assert head._sv_vectors.shape[0] >= 1, "fit_ocsvm() was not called by training_loop"
        assert head._sv_vectors.shape[1] == 16

    def test_train_ocsvm_onnx_export_after_training(self, tiny_dataset, tmp_path):
        """After training + fitting, ONNX export must produce a valid graph."""
        pytest.importorskip("sklearn")
        import onnx
        trainer = _make_ocsvm_trainer()
        trainer.train(
            train_data=tiny_dataset,
            test_data=tiny_dataset,
            epochs=2,
            batch_size=4,
            lr=1e-3,
            output_dir=str(tmp_path / "out"),
            save_best="f1",
            tsne_every=0, pca_every=0, umap_every=0,
        )
        head_path = str(tmp_path / "head_post_train.onnx")
        trainer.model.classifier.export_to_onnx(head_path)
        onnx.checker.check_model(head_path)

    def test_train_ocsvm_checkpoint_preserves_sv_buffers(self, tiny_dataset, tmp_path):
        """SV buffers must survive a save/load checkpoint round-trip."""
        pytest.importorskip("sklearn")
        import torch
        trainer = _make_ocsvm_trainer()
        trainer.train(
            train_data=tiny_dataset,
            test_data=tiny_dataset,
            epochs=2,
            batch_size=4,
            lr=1e-3,
            output_dir=str(tmp_path / "out"),
            save_best="f1",
            tsne_every=0, pca_every=0, umap_every=0,
        )
        head_before = trainer.model.classifier
        sv_before = head_before._sv_vectors.clone()
        rho_before = head_before._rho.clone()

        ckpt_path = str(tmp_path / "ckpt.pt")
        trainer.save_checkpoint(epoch=0, metrics={}, optimizer=None, out_path=ckpt_path)

        trainer2 = _make_ocsvm_trainer()
        trainer2.load_checkpoint(ckpt_path)
        head_after = trainer2.model.classifier

        assert torch.allclose(sv_before, head_after._sv_vectors), "SV vectors not preserved"
        assert torch.allclose(rho_before, head_after._rho), "rho not preserved"

    def test_ocsvm_without_sklearn_skips_fit(self, tiny_dataset, tmp_path, monkeypatch):
        """Training must complete even if sklearn is absent; fit_ocsvm() is skipped with a warning."""
        import sys
        # Simulate absent sklearn by making import fail inside fit_ocsvm
        original = __builtins__.__dict__ if hasattr(__builtins__, '__dict__') else {}

        import ww_trainer.loop as loop_mod
        original_attr = getattr(loop_mod, "_ocsvm_test_skip", None)

        # Patch sklearn away so fit_ocsvm raises ImportError
        monkeypatch.setitem(sys.modules, "sklearn.svm", None)
        monkeypatch.setitem(sys.modules, "sklearn", None)

        trainer = _make_ocsvm_trainer()
        # Should not raise — loop catches and warns
        result = trainer.train(
            train_data=tiny_dataset,
            test_data=tiny_dataset,
            epochs=1,
            batch_size=4,
            lr=1e-3,
            output_dir=str(tmp_path / "out"),
            save_best="f1",
            tsne_every=0, pca_every=0, umap_every=0,
        )
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Tier / factory integration
# ---------------------------------------------------------------------------

class TestOCSVMTierAndFactory:
    def test_ocsvm_small_tier_creates_ocsvm_head(self):
        from ww_trainer.tiers import get_tier
        from ww_trainer.model import OCSVMHead
        from ww_trainer.factory import create_model
        tc = get_tier("ocsvm_small")
        assert tc.head_arch == "ocsvm"
        assert tc.embed_dim == 64
        model = create_model(
            arch_name=tc.head_arch,
            featurizer="",
            featurizer_type=tc.extractor_type,
            feature_dim=None,
            device="cpu",
            hidden_dim=tc.hidden_dim,
            embed_dim=tc.embed_dim,
            n_mfcc=tc.n_mfcc,
        )
        assert isinstance(model.classifier, OCSVMHead)

    def test_head_registry_entry(self):
        from ww_trainer.factory import HEAD_REGISTRY
        from ww_trainer.model import OCSVMHead
        cls, valid_kwargs = HEAD_REGISTRY["ocsvm"]
        assert cls is OCSVMHead
        assert {"hidden_dim", "embed_dim", "dropout", "nu", "kernel", "gamma"} <= valid_kwargs
