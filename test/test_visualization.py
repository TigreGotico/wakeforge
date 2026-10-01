"""Tests for ww_trainer/visualization.py — plot functions with mocked matplotlib."""
import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---- synthetic data helpers ----

def _targets_probs(n: int = 20):
    rng = np.random.default_rng(42)
    targets = (rng.integers(0, 2, n)).tolist()
    probs = rng.uniform(0.0, 1.0, n).tolist()
    return targets, probs


def _make_stub_model_and_dataset(tmp_path):
    """Return (model, dataset_list) for log_pca / log_tsne tests."""
    import soundfile as sf

    class StubModel(nn.Module):
        def forward(self, wavs: torch.Tensor) -> torch.Tensor:
            return wavs.mean(dim=-1)

        def embed(self, wavs: torch.Tensor) -> torch.Tensor:
            # return fixed-dim embedding
            b = wavs.shape[0]
            return torch.randn(b, 8)

        def eval(self):
            return self

    samples = []
    for i in range(6):
        t = np.linspace(0, 0.5, 8000)
        wav = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        p = tmp_path / f"vis_{i}.wav"
        sf.write(str(p), wav, 16000)
        samples.append((str(p), str(i % 2)))
    return StubModel(), samples


# ---- plot_roc ----

def test_plot_roc_no_exception(tmp_path):
    from ww_trainer.visualization import plot_roc
    targets, probs = _targets_probs()
    with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
        result = plot_roc(targets, probs, epoch=1, plot_dir=tmp_path, auc=0.85)
    # Returns a Path or None (no exception is the key requirement)


def test_plot_roc_with_mlflow(tmp_path):
    from ww_trainer.visualization import plot_roc
    targets, probs = _targets_probs()
    mock_mlflow = MagicMock()
    with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
        plot_roc(targets, probs, epoch=2, plot_dir=tmp_path, auc=0.9, mlflow=mock_mlflow)


def test_plot_roc_returns_path_or_none(tmp_path):
    from ww_trainer.visualization import plot_roc
    targets, probs = _targets_probs()
    with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
        result = plot_roc(targets, probs, epoch=1, plot_dir=tmp_path, auc=0.5)
    assert result is None or isinstance(result, Path)


# ---- plot_pr ----

def test_plot_pr_no_exception(tmp_path):
    from ww_trainer.visualization import plot_pr
    targets, probs = _targets_probs()
    with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
        plot_pr(targets, probs, epoch=1, plot_dir=tmp_path)


def test_plot_pr_with_mlflow(tmp_path):
    from ww_trainer.visualization import plot_pr
    targets, probs = _targets_probs()
    mock_mlflow = MagicMock()
    with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
        plot_pr(targets, probs, epoch=1, plot_dir=tmp_path, mlflow=mock_mlflow)


# ---- plot_det ----

def test_plot_det_no_exception(tmp_path):
    from ww_trainer.visualization import plot_det
    targets, probs = _targets_probs()
    with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
        plot_det(targets, probs, epoch=1, plot_dir=tmp_path)


def test_plot_det_with_mlflow(tmp_path):
    from ww_trainer.visualization import plot_det
    targets, probs = _targets_probs()
    mock_mlflow = MagicMock()
    with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
        plot_det(targets, probs, epoch=1, plot_dir=tmp_path, mlflow=mock_mlflow)


# ---- log_confidence_histogram ----

def test_log_confidence_histogram_no_exception(tmp_path):
    from ww_trainer.visualization import log_confidence_histogram
    targets, probs = _targets_probs(40)
    with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
        result = log_confidence_histogram(targets, probs, epoch=1, outdir=tmp_path)
    assert result is None or isinstance(result, str)


def test_log_confidence_histogram_empty_returns_none(tmp_path):
    from ww_trainer.visualization import log_confidence_histogram
    result = log_confidence_histogram([], [], epoch=1, outdir=tmp_path)
    assert result is None


# ---- UMAP fallback to t-SNE ----

def test_log_umap_falls_back_to_tsne_when_umap_unavailable(tmp_path):
    """When umap is not installed, log_umap falls back to t-SNE."""
    import ww_trainer.visualization as viz_mod

    model, dataset = _make_stub_model_and_dataset(tmp_path)

    original_has_umap = viz_mod._HAS_UMAP
    try:
        viz_mod._HAS_UMAP = False
        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            with patch("sklearn.manifold.TSNE.fit_transform", return_value=np.zeros((6, 2))):
                result = viz_mod.log_umap(model, dataset, outdir=tmp_path, epoch=1, device="cpu")
    finally:
        viz_mod._HAS_UMAP = original_has_umap
