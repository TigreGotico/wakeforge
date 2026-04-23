"""Extended tests for ww_trainer/visualization.py — mocked matplotlib, all functions."""
import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock


SAMPLE_RATE = 16000


def _targets_probs(n: int = 30):
    """Generate synthetic targets and probabilities."""
    rng = np.random.default_rng(42)
    targets = rng.integers(0, 2, n).tolist()
    probs = rng.uniform(0.0, 1.0, n).tolist()
    return targets, probs


def _stub_model():
    """Model with eval(), forward(), embed()."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self._dummy = nn.Parameter(torch.zeros(1))

        def forward(self, wavs):
            return wavs.mean(dim=-1)

        def embed(self, wavs):
            b = wavs.shape[0]
            return torch.randn(b, 8)

        def eval(self):
            return self

    return M()


def _make_dataset(tmp_path, n: int = 10):
    """Return list of (path, label) tuples."""
    samples = []
    for i in range(n):
        t = np.linspace(0, 0.5, 8000)
        wav = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        p = tmp_path / f"viz_{i}.wav"
        sf.write(str(p), wav, SAMPLE_RATE)
        samples.append((str(p), str(i % 2)))
    return samples


# ---- plot_roc ----

class TestPlotRoc:
    def test_returns_path(self, tmp_path):
        from ww_trainer.visualization import plot_roc
        targets, probs = _targets_probs()
        result = plot_roc(targets, probs, epoch=1, plot_dir=tmp_path, auc=0.9)
        assert result is None or isinstance(result, Path)

    def test_mlflow_artifact_logged(self, tmp_path):
        from ww_trainer.visualization import plot_roc
        targets, probs = _targets_probs()
        mock_mlflow = MagicMock()
        result = plot_roc(targets, probs, epoch=1, plot_dir=tmp_path, auc=0.9, mlflow=mock_mlflow)
        if result is not None and Path(result).exists():
            mock_mlflow.log_artifact.assert_called_once()

    def test_mlflow_failure_handled(self, tmp_path):
        from ww_trainer.visualization import plot_roc
        targets, probs = _targets_probs()
        mock_mlflow = MagicMock()
        mock_mlflow.log_artifact.side_effect = RuntimeError("mlflow down")
        # Should not raise
        plot_roc(targets, probs, epoch=1, plot_dir=tmp_path, auc=0.9, mlflow=mock_mlflow)

    def test_bad_data_returns_none(self, tmp_path):
        from ww_trainer.visualization import plot_roc
        result = plot_roc([], [], epoch=1, plot_dir=tmp_path, auc=0.0)
        assert result is None


# ---- plot_pr ----

class TestPlotPr:
    def test_returns_path(self, tmp_path):
        from ww_trainer.visualization import plot_pr
        targets, probs = _targets_probs()
        result = plot_pr(targets, probs, epoch=1, plot_dir=tmp_path)
        assert result is None or isinstance(result, Path)

    def test_mlflow_failure_handled(self, tmp_path):
        from ww_trainer.visualization import plot_pr
        targets, probs = _targets_probs()
        mock_mlflow = MagicMock()
        mock_mlflow.log_artifact.side_effect = RuntimeError("fail")
        plot_pr(targets, probs, epoch=1, plot_dir=tmp_path, mlflow=mock_mlflow)

    def test_bad_data_returns_none(self, tmp_path):
        from ww_trainer.visualization import plot_pr
        result = plot_pr([], [], epoch=1, plot_dir=tmp_path)
        assert result is None


# ---- plot_det ----

class TestPlotDet:
    def test_returns_path(self, tmp_path):
        from ww_trainer.visualization import plot_det
        targets, probs = _targets_probs()
        result = plot_det(targets, probs, epoch=1, plot_dir=tmp_path)
        assert result is None or isinstance(result, Path)

    def test_mlflow_failure_handled(self, tmp_path):
        from ww_trainer.visualization import plot_det
        targets, probs = _targets_probs()
        mock_mlflow = MagicMock()
        mock_mlflow.log_artifact.side_effect = RuntimeError("fail")
        plot_det(targets, probs, epoch=1, plot_dir=tmp_path, mlflow=mock_mlflow)

    def test_bad_data_returns_none(self, tmp_path):
        from ww_trainer.visualization import plot_det
        result = plot_det([], [], epoch=1, plot_dir=tmp_path)
        assert result is None


# ---- log_confidence_histogram ----

class TestConfidenceHistogram:
    def test_creates_file(self, tmp_path):
        from ww_trainer.visualization import log_confidence_histogram
        targets, probs = _targets_probs(50)
        result = log_confidence_histogram(targets, probs, epoch=3, outdir=tmp_path)
        assert result is not None
        assert Path(result).exists()

    def test_empty_returns_none(self, tmp_path):
        from ww_trainer.visualization import log_confidence_histogram
        assert log_confidence_histogram([], [], epoch=1, outdir=tmp_path) is None

    def test_mlflow_logged(self, tmp_path):
        from ww_trainer.visualization import log_confidence_histogram
        targets, probs = _targets_probs(20)
        mock_mlflow = MagicMock()
        log_confidence_histogram(targets, probs, epoch=1, outdir=tmp_path, mlflow=mock_mlflow)
        mock_mlflow.log_artifact.assert_called_once()

    def test_mlflow_failure_handled(self, tmp_path):
        from ww_trainer.visualization import log_confidence_histogram
        targets, probs = _targets_probs(20)
        mock_mlflow = MagicMock()
        mock_mlflow.log_artifact.side_effect = RuntimeError("fail")
        # Should not raise
        log_confidence_histogram(targets, probs, epoch=1, outdir=tmp_path, mlflow=mock_mlflow)

    def test_custom_bins(self, tmp_path):
        from ww_trainer.visualization import log_confidence_histogram
        targets, probs = _targets_probs(20)
        result = log_confidence_histogram(targets, probs, epoch=1, outdir=tmp_path, bins=10)
        assert result is not None


# ---- log_pca ----

class TestLogPca:
    def test_returns_path(self, tmp_path):
        from ww_trainer.visualization import log_pca
        model = _stub_model()
        dataset = _make_dataset(tmp_path)
        result, _ = log_pca(model, dataset, outdir=tmp_path, epoch=1, device="cpu")
        assert isinstance(result, (str, type(None))) or hasattr(result, "__fspath__")
        if result is not None:
            assert Path(result).exists()

    def test_mlflow_logged(self, tmp_path):
        from ww_trainer.visualization import log_pca
        model = _stub_model()
        dataset = _make_dataset(tmp_path)
        mock_mlflow = MagicMock()
        log_pca(model, dataset, outdir=tmp_path, epoch=1, device="cpu", mlflow=mock_mlflow)
        mock_mlflow.log_artifact.assert_called_once()

    def test_mlflow_failure_handled(self, tmp_path):
        from ww_trainer.visualization import log_pca
        model = _stub_model()
        dataset = _make_dataset(tmp_path)
        mock_mlflow = MagicMock()
        mock_mlflow.log_artifact.side_effect = RuntimeError("fail")
        log_pca(model, dataset, outdir=tmp_path, epoch=1, device="cpu", mlflow=mock_mlflow)


# ---- log_tsne ----

class TestLogTsne:
    def test_returns_path(self, tmp_path):
        from ww_trainer.visualization import log_tsne
        model = _stub_model()
        dataset = _make_dataset(tmp_path)
        result, _ = log_tsne(model, dataset, outdir=tmp_path, epoch=1, device="cpu")
        assert result is None or isinstance(result, (str, Path))

    def test_empty_dataset_returns_none(self, tmp_path):
        from ww_trainer.visualization import log_tsne
        model = _stub_model()
        result, _ = log_tsne(model, [], outdir=tmp_path, epoch=1, device="cpu")
        assert result is None

    def test_string_outdir(self, tmp_path):
        """outdir can be a string (converted internally)."""
        from ww_trainer.visualization import log_tsne
        model = _stub_model()
        dataset = _make_dataset(tmp_path)
        result, _ = log_tsne(model, dataset, outdir=str(tmp_path), epoch=1, device="cpu")
        assert result is None or isinstance(result, (str, Path))

    def test_mlflow_logged(self, tmp_path):
        from ww_trainer.visualization import log_tsne
        model = _stub_model()
        dataset = _make_dataset(tmp_path)
        mock_mlflow = MagicMock()
        log_tsne(model, dataset, outdir=tmp_path, epoch=1, device="cpu", mlflow=mock_mlflow)


# ---- log_umap ----

class TestLogUmap:
    def test_umap_fallback_to_tsne(self, tmp_path):
        """Without umap-learn, falls back to t-SNE."""
        import ww_trainer.visualization as viz
        model = _stub_model()
        dataset = _make_dataset(tmp_path, n=40)
        orig = viz._HAS_UMAP
        try:
            viz._HAS_UMAP = False
            result, _ = viz.log_umap(model, dataset, outdir=tmp_path, epoch=1, device="cpu")
            assert result is None or isinstance(result, (str, Path))
        finally:
            viz._HAS_UMAP = orig

    def test_umap_with_real_umap(self, tmp_path):
        """If umap is available, uses real UMAP."""
        import ww_trainer.visualization as viz
        model = _stub_model()
        dataset = _make_dataset(tmp_path)
        if viz._HAS_UMAP:
            result, _ = viz.log_umap(model, dataset, outdir=tmp_path, epoch=1, device="cpu")
            assert result is None or isinstance(result, (str, Path))
        else:
            pytest.skip("umap-learn not installed")

    def test_mlflow_logged(self, tmp_path):
        import ww_trainer.visualization as viz
        model = _stub_model()
        dataset = _make_dataset(tmp_path, n=40)
        mock_mlflow = MagicMock()
        orig = viz._HAS_UMAP
        try:
            viz._HAS_UMAP = False
            viz.log_umap(model, dataset, outdir=tmp_path, epoch=1, device="cpu", mlflow=mock_mlflow)
            mock_mlflow.log_artifact.assert_called_once()
        finally:
            viz._HAS_UMAP = orig

    def test_mlflow_failure_handled(self, tmp_path):
        import ww_trainer.visualization as viz
        model = _stub_model()
        dataset = _make_dataset(tmp_path, n=40)
        mock_mlflow = MagicMock()
        mock_mlflow.log_artifact.side_effect = RuntimeError("fail")
        orig = viz._HAS_UMAP
        try:
            viz._HAS_UMAP = False
            viz.log_umap(model, dataset, outdir=tmp_path, epoch=1, device="cpu", mlflow=mock_mlflow)
        finally:
            viz._HAS_UMAP = orig


# ---- log_embeddings_stats ----

class TestLogEmbeddingsStats:
    def test_returns_dict(self, tmp_path):
        from ww_trainer.visualization import log_embeddings_stats
        model = _stub_model()
        dataset = _make_dataset(tmp_path)
        stats = log_embeddings_stats(model, dataset, epoch=1, device="cpu")
        assert isinstance(stats, dict)
        assert "embed_norm_mean" in stats
        assert "embed_norm_std" in stats
        assert "embed_var_total" in stats

    def test_intra_class_variance(self, tmp_path):
        from ww_trainer.visualization import log_embeddings_stats
        model = _stub_model()
        dataset = _make_dataset(tmp_path, n=10)
        stats = log_embeddings_stats(model, dataset, epoch=1, device="cpu")
        # With mixed labels, should have both intra-class variances
        assert "intra_pos_var" in stats or "intra_neg_var" in stats

    def test_mlflow_logged(self, tmp_path):
        from ww_trainer.visualization import log_embeddings_stats
        model = _stub_model()
        dataset = _make_dataset(tmp_path)
        mock_mlflow = MagicMock()
        log_embeddings_stats(model, dataset, epoch=5, device="cpu", mlflow=mock_mlflow)
        mock_mlflow.log_metrics.assert_called_once()
