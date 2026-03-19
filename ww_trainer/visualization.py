"""Visualization helpers extracted from WakeWordTrainer."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, precision_recall_curve, det_curve
from torch.utils.data import DataLoader

from ww_trainer.dataset import AudioDataset, collate_fn

try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

_logger = logging.getLogger(__name__)


def plot_roc(targets, probs, epoch: int, plot_dir: Path, auc: float, mlflow=None) -> Optional[Path]:
    try:
        fpr, tpr, _ = roc_curve(targets, probs)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (Epoch {epoch}) AUC={auc:.3f}')
        roc_path = plot_dir / f'roc_epoch_{epoch}.png'
        plt.tight_layout()
        plt.savefig(roc_path, dpi=200)
        plt.close()
        if mlflow and roc_path.exists():
            try:
                mlflow.log_artifact(str(roc_path), artifact_path="roc_pr_det/roc")
            except Exception as e:
                logger.error("Failed to log ROC plot to MLflow: %s", e)
        return roc_path
    except Exception as e:
        logger.error("Failed to create ROC plot: %s", e)
        return None


def plot_pr(targets, probs, epoch: int, plot_dir: Path, mlflow=None) -> Optional[Path]:
    try:
        prec_curve, rec_curve, _ = precision_recall_curve(targets, probs)
        plt.figure()
        plt.plot(rec_curve, prec_curve)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (Epoch {epoch})')
        pr_path = plot_dir / f'pr_epoch_{epoch}.png'
        plt.tight_layout()
        plt.savefig(pr_path, dpi=200)
        plt.close()
        if mlflow and pr_path.exists():
            try:
                mlflow.log_artifact(str(pr_path), artifact_path="roc_pr_det/pr")
            except Exception as e:
                logger.error("Failed to log PR plot to MLflow: %s", e)
        return pr_path
    except Exception as e:
        logger.error("Failed to create PR plot: %s", e)
        return None


def plot_det(targets, probs, epoch: int, plot_dir: Path, mlflow=None) -> Optional[Path]:
    try:
        fpr_det, fnr_det, _ = det_curve(targets, probs)
        plt.figure()
        plt.plot(fpr_det, fnr_det)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('False Positive Rate')
        plt.ylabel('False Negative Rate')
        plt.title(f'DET Curve (Epoch {epoch})')
        det_path = plot_dir / f'det_epoch_{epoch}.png'
        plt.tight_layout()
        plt.savefig(det_path, dpi=200)
        plt.close()
        if mlflow and det_path.exists():
            try:
                mlflow.log_artifact(str(det_path), artifact_path="roc_pr_det/det")
            except Exception as e:
                logger.error("Failed to log DET plot to MLflow: %s", e)
        return det_path
    except Exception as e:
        logger.error("Failed to create DET plot: %s", e)
        return None


def log_confidence_histogram(
    targets: list,
    probs: list,
    epoch: int,
    outdir: Path,
    bins: int = 50,
    mlflow=None,
) -> Optional[str]:
    """Plot confidence distribution for positive and negative samples."""
    if len(targets) == 0 or len(probs) == 0:
        return None

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"confidence_hist_epoch_{epoch}.png"

    targets_arr = np.array(targets)
    probs_arr = np.array(probs)

    plt.figure(figsize=(6, 4))
    plt.hist(probs_arr[targets_arr == 1], bins=bins, range=(0, 1), alpha=0.6,
             label="Wake (label=1)", density=True, color="tab:blue")
    plt.hist(probs_arr[targets_arr == 0], bins=bins, range=(0, 1), alpha=0.6,
             label="Nonwake (label=0)", density=True, color="tab:orange")
    plt.title(f"Confidence Distribution (Epoch {epoch})")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.legend(loc="upper center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    if mlflow:
        try:
            mlflow.log_artifact(str(outpath), artifact_path="viz/confidence")
        except Exception as e:
            logger.error("Failed to log confidence histogram to MLflow: %s", e)

    return str(outpath)


def log_pca(model, dataset, outdir: Path, epoch: int, device, sample_size: int = 200, mlflow=None) -> str:
    subset = random.sample(dataset, min(len(dataset), sample_size))
    loader = DataLoader(AudioDataset(subset, aug_prob=0), batch_size=128, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, device))
    import torch
    embeddings, labels = [], []
    model.eval()
    with torch.no_grad():
        for wavs, lbls, _ in loader:
            emb = model.embed(wavs)
            embeddings.append(emb.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
    embeddings = np.concatenate(embeddings)
    labels = np.array(labels)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(embeddings)

    plt.figure(figsize=(6, 5))
    plt.scatter(proj[labels == 0, 0], proj[labels == 0, 1], s=15, alpha=0.5, label="Nonwake")
    plt.scatter(proj[labels == 1, 0], proj[labels == 1, 1], s=20, alpha=0.7, label="Wake")
    plt.legend()
    plt.title(f"PCA Embedding (Epoch {epoch})")
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"pca_epoch_{epoch}.png"
    plt.savefig(outpath, dpi=200)
    plt.close()

    if mlflow:
        try:
            mlflow.log_artifact(str(outpath), artifact_path="viz/pca")
        except Exception as e:
            logger.error("MLflow PCA log failed: %s", e)
    return str(outpath)


def log_tsne(model, dataset: List[Tuple[str, str]], outdir: Path, epoch: int, device,
             sample_size: int = 200, mlflow=None) -> Optional[str]:
    if len(dataset) == 0:
        return None
    subset = random.sample(dataset, min(len(dataset), sample_size))
    loader = DataLoader(AudioDataset(subset, aug_prob=0), batch_size=128, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, device))
    import torch
    embeddings, labels = [], []
    model.eval()
    with torch.no_grad():
        for wavs, lbls, _ in loader:
            emb = model.embed(wavs)
            embeddings.append(emb.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels)
    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
        proj = tsne.fit_transform(embeddings)
    except Exception as e:
        logger.error("t-SNE failed: %s", e)
        return None
    plt.figure(figsize=(6, 5))
    plt.scatter(proj[labels == 0, 0], proj[labels == 0, 1], alpha=0.5, s=15, label="Nonwake")
    plt.scatter(proj[labels == 1, 0], proj[labels == 1, 1], alpha=0.7, s=20, label="Wake")
    plt.legend()
    plt.title(f"t-SNE Embedding (Epoch {epoch})")
    plt.tight_layout()
    if isinstance(outdir, str):
        outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"tsne_epoch_{epoch}.png"
    plt.savefig(outpath, dpi=200)
    plt.close()
    if mlflow:
        try:
            mlflow.log_artifact(str(outpath), artifact_path="viz/tsne")
        except Exception as e:
            logger.error("Failed to log t-SNE to MLflow: %s", e)
    return str(outpath)


def log_umap(model, dataset, outdir: Path, epoch: int, device, sample_size: int = 200, mlflow=None) -> Optional[str]:
    subset = random.sample(dataset, min(len(dataset), sample_size))
    loader = DataLoader(AudioDataset(subset, aug_prob=0), batch_size=128, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, device))
    import torch
    embeddings, labels = [], []
    model.eval()
    with torch.no_grad():
        for wavs, lbls, _ in loader:
            emb = model.embed(wavs)
            embeddings.append(emb.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
    embeddings = np.concatenate(embeddings)
    labels = np.array(labels)

    title = f"UMAP Embedding (Epoch {epoch})"
    if not _HAS_UMAP:
        _logger.warning(
            "umap-learn not installed; falling back to t-SNE. "
            "Install with: pip install umap-learn"
        )
        reducer = TSNE(n_components=2, random_state=42)
        proj = reducer.fit_transform(embeddings)
        title = title.replace("UMAP", "t-SNE (UMAP fallback)")
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)
        proj = reducer.fit_transform(embeddings)

    plt.figure(figsize=(6, 5))
    plt.scatter(proj[labels == 0, 0], proj[labels == 0, 1], s=15, alpha=0.5, label="Nonwake")
    plt.scatter(proj[labels == 1, 0], proj[labels == 1, 1], s=20, alpha=0.7, label="Wake")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"umap_epoch_{epoch}.png"
    plt.savefig(outpath, dpi=200)
    plt.close()

    if mlflow:
        try:
            mlflow.log_artifact(str(outpath), artifact_path="viz/umap")
        except Exception as e:
            logger.error("MLflow UMAP log failed: %s", e)
    return str(outpath)


def log_embeddings_stats(model, dataset, epoch: int, device, batch_size: int = 128, mlflow=None) -> dict:
    import torch
    loader = DataLoader(AudioDataset(dataset, aug_prob=0), batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, device))
    embeds_all, labels_all = [], []
    with torch.no_grad():
        for wavs, labels, _ in loader:
            embeds = model.embed(wavs)
            embeds_all.append(embeds.cpu())
            labels_all.append(labels.cpu())
    embeds_all = torch.cat(embeds_all)
    labels_all = torch.cat(labels_all)
    norms = embeds_all.norm(p=2, dim=1)
    stats = {
        "embed_norm_mean": norms.mean().item(),
        "embed_norm_std": norms.std().item(),
        "embed_var_total": embeds_all.var(dim=0).mean().item()
    }
    pos, neg = embeds_all[labels_all == 1], embeds_all[labels_all == 0]
    if len(pos) > 1:
        stats["intra_pos_var"] = pos.var(dim=0).mean().item()
    if len(neg) > 1:
        stats["intra_neg_var"] = neg.var(dim=0).mean().item()
    if mlflow:
        mlflow.log_metrics(stats, step=epoch)
    return stats
