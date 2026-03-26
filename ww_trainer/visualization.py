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


def _mlflow_log_artifact(mlflow, path: Path, artifact_path: str) -> None:
    """Upload artifact to MLflow, silently skipping on failure."""
    if mlflow and path and Path(path).exists():
        try:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
        except Exception as e:
            logger.warning("MLflow artifact upload failed (%s): %s", artifact_path, e)


def plot_roc(targets, probs, epoch: int, plot_dir: Path, auc: float, mlflow=None) -> Optional[Path]:
    try:
        fpr, tpr, _ = roc_curve(targets, probs)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, lw=2, color="#2980b9")
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.fill_between(fpr, tpr, alpha=0.08, color="#2980b9")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC  AUC={auc:.4f}  (epoch {epoch})")
        ax.grid(alpha=0.3); fig.tight_layout()
        roc_path = plot_dir / "roc_latest.png"
        fig.savefig(roc_path, dpi=150); plt.close(fig)
        _mlflow_log_artifact(mlflow, roc_path, "curves")
        return roc_path
    except Exception as e:
        logger.error("Failed to create ROC plot: %s", e)
        return None


def plot_pr(targets, probs, epoch: int, plot_dir: Path, mlflow=None) -> Optional[Path]:
    try:
        prec_curve, rec_curve, _ = precision_recall_curve(targets, probs)
        baseline = sum(targets) / max(1, len(targets))
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rec_curve, prec_curve, lw=2, color="#27ae60")
        ax.axhline(baseline, linestyle="--", color="grey", lw=1, label=f"Baseline={baseline:.3f}")
        ax.fill_between(rec_curve, prec_curve, alpha=0.08, color="#27ae60")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall  (epoch {epoch})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
        pr_path = plot_dir / "pr_latest.png"
        fig.savefig(pr_path, dpi=150); plt.close(fig)
        _mlflow_log_artifact(mlflow, pr_path, "curves")
        return pr_path
    except Exception as e:
        logger.error("Failed to create PR plot: %s", e)
        return None


def plot_det(targets, probs, epoch: int, plot_dir: Path, mlflow=None) -> Optional[Path]:
    try:
        fpr_det, fnr_det, _ = det_curve(targets, probs)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr_det, fnr_det, lw=2, color="#e74c3c")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("False Negative Rate")
        ax.set_title(f"DET Curve  (epoch {epoch})")
        ax.grid(which="both", alpha=0.3); fig.tight_layout()
        det_path = plot_dir / "det_latest.png"
        fig.savefig(det_path, dpi=150); plt.close(fig)
        _mlflow_log_artifact(mlflow, det_path, "curves")
        return det_path
    except Exception as e:
        logger.error("Failed to create DET plot: %s", e)
        return None


def plot_confusion_matrix(targets, preds, epoch: int, plot_dir: Path, mlflow=None) -> Optional[Path]:
    """Plot and upload a styled confusion matrix."""
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(targets, preds)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        labels = ["Non-wake", "Wake"]
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix  (epoch {epoch})")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        fig.tight_layout()
        cm_path = plot_dir / "confusion_matrix_latest.png"
        fig.savefig(cm_path, dpi=150); plt.close(fig)
        _mlflow_log_artifact(mlflow, cm_path, "curves")
        return cm_path
    except Exception as e:
        logger.error("Failed to create confusion matrix: %s", e)
        return None


def plot_threshold_sensitivity(targets, probs, plot_dir: Path, mlflow=None) -> Optional[Path]:
    """Plot F1, precision, recall and FPR as a function of threshold."""
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score
        thresholds = np.linspace(0.05, 0.95, 50)
        f1s, precs, recs, fprs = [], [], [], []
        tgt = np.array(targets)
        prb = np.array(probs)
        n_neg = max(1, (tgt == 0).sum())
        for t in thresholds:
            p = (prb >= t).astype(int)
            f1s.append(f1_score(tgt, p, zero_division=0))
            precs.append(precision_score(tgt, p, zero_division=0))
            recs.append(recall_score(tgt, p, zero_division=0))
            fprs.append(((p == 1) & (tgt == 0)).sum() / n_neg)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(thresholds, f1s,   label="F1",        lw=2, color="#2980b9")
        ax.plot(thresholds, recs,  label="Recall",    lw=2, color="#27ae60", linestyle="--")
        ax.plot(thresholds, precs, label="Precision", lw=2, color="#e67e22", linestyle="--")
        ax.plot(thresholds, fprs,  label="FPR",       lw=2, color="#e74c3c", linestyle=":")
        best_t = thresholds[int(np.argmax(f1s))]
        ax.axvline(best_t, color="grey", linestyle="--", lw=1, label=f"Best F1 @ {best_t:.2f}")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
        ax.set_title("Threshold Sensitivity")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
        ts_path = plot_dir / "threshold_sensitivity.png"
        fig.savefig(ts_path, dpi=150); plt.close(fig)
        _mlflow_log_artifact(mlflow, ts_path, "curves")
        return ts_path
    except Exception as e:
        logger.error("Failed to create threshold sensitivity plot: %s", e)
        return None


def plot_training_curves(history: list, plot_dir: Path, mlflow=None) -> Optional[Path]:
    """Summary training curves: loss, F1, precision, recall, AUC on one figure."""
    if not history:
        return None
    try:
        epochs = [h["epoch"] for h in history]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Loss
        axes[0].plot(epochs, [h["loss"] for h in history], lw=2, color="#e74c3c", label="Train loss")
        axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch")
        axes[0].grid(alpha=0.3); axes[0].legend()

        # F1 + AUC
        axes[1].plot(epochs, [h["f1"]  for h in history], lw=2, color="#2980b9",  label="F1")
        axes[1].plot(epochs, [h["auc"] for h in history], lw=2, color="#8e44ad", label="AUC", linestyle="--")
        axes[1].set_title("F1 & AUC"); axes[1].set_xlabel("Epoch")
        axes[1].set_ylim(0, 1); axes[1].grid(alpha=0.3); axes[1].legend()

        # Precision / Recall
        axes[2].plot(epochs, [h["precision"] for h in history], lw=2, color="#e67e22", label="Precision")
        axes[2].plot(epochs, [h["recall"]    for h in history], lw=2, color="#27ae60", label="Recall", linestyle="--")
        axes[2].set_title("Precision & Recall"); axes[2].set_xlabel("Epoch")
        axes[2].set_ylim(0, 1); axes[2].grid(alpha=0.3); axes[2].legend()

        fig.suptitle("Training Summary", fontsize=12)
        fig.tight_layout()
        tc_path = plot_dir / "training_curves.png"
        fig.savefig(tc_path, dpi=150); plt.close(fig)
        _mlflow_log_artifact(mlflow, tc_path, "summary")
        return tc_path
    except Exception as e:
        logger.error("Failed to create training curves: %s", e)
        return None


def plot_rppl_dashboard(
    history: list,
    plot_dir: "Path | str",
    mlflow=None,
    warmup_epochs: int = 5,
) -> Optional[Path]:
    """RPPL-specific training dashboard — 6-panel figure logged to MLflow.

    Panels
    ------
    1. Sub-loss trajectories  (bce, proto, div, center, cons) over epochs
    2. Geo-scale ramp         (warmup visualisation) + total loss overlay
    3. Proto EMA norm         (should stabilise quickly; large swings = unstable)
    4. Embedding Fisher ratio (centroid separation / intra-class spread)
    5. Embedding silhouette   (–1 → 1; > 0.5 is good)
    6. Hard-negative count    (n_hard_negatives per epoch)

    Only panels for which data is present in *history* are populated.
    Missing keys are silently skipped.

    Args:
        history:       List of per-epoch dicts from ``_epoch_history`` in loop.py.
        plot_dir:      Directory to write the PNG.
        mlflow:        Active MLflow client — artifact logged to ``rppl/`` folder.
        warmup_epochs: Used to overlay the expected geo_scale ramp.
    """
    if not history:
        return None

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]

    # ── Palette ────────────────────────────────────────────────────────────
    COLORS = {
        "bce":    "#e74c3c",
        "proto":  "#2980b9",
        "div":    "#8e44ad",
        "center": "#e67e22",
        "cons":   "#27ae60",
        "total":  "#2c3e50",
        "geo":    "#95a5a6",
        "fisher": "#f39c12",
        "sil":    "#16a085",
        "hard":   "#c0392b",
        "ema":    "#7f8c8d",
    }

    try:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        ax = axes.flatten()

        # ── Panel 1: Sub-loss trajectories ─────────────────────────────────
        sub_keys = [("rppl_bce", "bce", "BCE"), ("rppl_proto", "proto", "Proto"),
                    ("rppl_div", "div", "Div"), ("rppl_center", "center", "Center"),
                    ("rppl_cons", "cons", "Cons")]
        any_sub = False
        for key, col, label in sub_keys:
            vals = [h.get(key) for h in history]
            if any(v is not None for v in vals):
                clean = [v if v is not None else float("nan") for v in vals]
                ax[0].plot(epochs, clean, lw=2, color=COLORS[col], label=label)
                any_sub = True
        ax[0].set_title("RPPL Sub-loss Trajectories")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss value")
        ax[0].grid(alpha=0.3)
        if any_sub:
            ax[0].legend(fontsize=8)

        # ── Panel 2: Geo-scale ramp + total loss ───────────────────────────
        expected_geo = [min(1.0, e / max(1, warmup_epochs)) for e in epochs]
        ax2r = ax[1].twinx()
        ax[1].plot(epochs, expected_geo, lw=2, color=COLORS["geo"],
                   linestyle="--", label="geo_scale (expected)")
        ax[1].axvline(warmup_epochs, color=COLORS["geo"], alpha=0.4, linestyle=":")
        ax[1].set_ylim(0, 1.05)
        ax[1].set_ylabel("geo_scale", color=COLORS["geo"])
        ax[1].tick_params(axis="y", labelcolor=COLORS["geo"])

        total_vals = [h.get("loss") for h in history]
        if any(v is not None for v in total_vals):
            clean_total = [v if v is not None else float("nan") for v in total_vals]
            ax2r.plot(epochs, clean_total, lw=2, color=COLORS["total"], label="Total loss")
            ax2r.set_ylabel("Total loss", color=COLORS["total"])
            ax2r.tick_params(axis="y", labelcolor=COLORS["total"])
        ax[1].set_title("Warmup Schedule & Total Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].grid(alpha=0.3)
        lines1, labels1 = ax[1].get_legend_handles_labels()
        lines2, labels2 = ax2r.get_legend_handles_labels()
        ax[1].legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        # ── Panel 3: Proto EMA norm ─────────────────────────────────────────
        ema_vals = [h.get("rppl_proto_ema_norm") for h in history]
        if any(v is not None for v in ema_vals):
            clean = [v if v is not None else float("nan") for v in ema_vals]
            ax[2].plot(epochs, clean, lw=2, color=COLORS["ema"])
            ax[2].set_title("Wake Prototype EMA Norm")
            ax[2].set_xlabel("Epoch")
            ax[2].set_ylabel("||p_w_ema||")
            ax[2].grid(alpha=0.3)
        else:
            ax[2].set_title("Wake Prototype EMA Norm")
            ax[2].text(0.5, 0.5, "No EMA norm data\n(set log_proto_ema_norm=True)",
                       ha="center", va="center", transform=ax[2].transAxes,
                       color="grey", fontsize=9)
            ax[2].set_axis_off()

        # ── Panel 4: Fisher ratio ───────────────────────────────────────────
        fisher_vals = [h.get("embed_fisher_ratio") for h in history]
        if any(v is not None for v in fisher_vals):
            clean = [v if v is not None else float("nan") for v in fisher_vals]
            ax[3].plot(epochs, clean, lw=2, color=COLORS["fisher"])
            ax[3].axhline(1.0, color=COLORS["fisher"], alpha=0.3, linestyle="--", label="target=1")
            ax[3].set_title("Embedding Fisher Ratio")
            ax[3].set_xlabel("Epoch")
            ax[3].set_ylabel("centroid_dist / intra_spread")
            ax[3].grid(alpha=0.3)
            ax[3].legend(fontsize=8)
        else:
            ax[3].set_title("Embedding Fisher Ratio")
            ax[3].text(0.5, 0.5, "Enable PCA/t-SNE viz\n(pca_every > 0)",
                       ha="center", va="center", transform=ax[3].transAxes,
                       color="grey", fontsize=9)
            ax[3].set_axis_off()

        # ── Panel 5: Silhouette score ───────────────────────────────────────
        sil_vals = [h.get("embed_silhouette") for h in history]
        if any(v is not None for v in sil_vals):
            clean = [v if v is not None else float("nan") for v in sil_vals]
            ax[4].plot(epochs, clean, lw=2, color=COLORS["sil"])
            ax[4].axhline(0.5, color=COLORS["sil"], alpha=0.3, linestyle="--", label="good≥0.5")
            ax[4].axhline(0.0, color="grey", alpha=0.2, linestyle=":")
            ax[4].set_ylim(-1, 1)
            ax[4].set_title("Cosine Silhouette Score")
            ax[4].set_xlabel("Epoch")
            ax[4].set_ylabel("Silhouette [-1, 1]")
            ax[4].grid(alpha=0.3)
            ax[4].legend(fontsize=8)
        else:
            ax[4].set_title("Cosine Silhouette Score")
            ax[4].text(0.5, 0.5, "Enable PCA/t-SNE viz\n(pca_every > 0)",
                       ha="center", va="center", transform=ax[4].transAxes,
                       color="grey", fontsize=9)
            ax[4].set_axis_off()

        # ── Panel 6: Hard-negative count ────────────────────────────────────
        hard_vals = [h.get("n_hard_negatives") for h in history]
        if any(v is not None for v in hard_vals):
            clean = [v if v is not None else float("nan") for v in hard_vals]
            ax[5].fill_between(epochs, clean, alpha=0.25, color=COLORS["hard"])
            ax[5].plot(epochs, clean, lw=2, color=COLORS["hard"])
            ax[5].set_title("Hard Negatives per Epoch")
            ax[5].set_xlabel("Epoch")
            ax[5].set_ylabel("Count")
            ax[5].grid(alpha=0.3)
        else:
            ax[5].set_title("Hard Negatives per Epoch")
            ax[5].text(0.5, 0.5, "No hard-neg mining data",
                       ha="center", va="center", transform=ax[5].transAxes,
                       color="grey", fontsize=9)
            ax[5].set_axis_off()

        fig.suptitle("RPPL Training Dashboard", fontsize=13, fontweight="bold")
        fig.tight_layout()
        out_path = plot_dir / "rppl_dashboard.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        _mlflow_log_artifact(mlflow, out_path, "rppl")
        return out_path

    except Exception as e:
        logger.error("Failed to create RPPL dashboard: %s", e)
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


# ---------------------------------------------------------------------------
# Shared embedding collection + metrics
# ---------------------------------------------------------------------------

def _collect_embeddings(
    model,
    dataset: List[Tuple[str, str]],
    device,
    sample_size: int = 300,
    aug_prob: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (embeddings, labels, confidences) arrays for a balanced sample.

    Balances wake/nonwake so neither class dominates the projection.
    Returns:
        embeddings:   (N, D) float32
        labels:       (N,)   int   0=nonwake 1=wake
        confidences:  (N,)   float sigmoid(logit)
    """
    import torch

    wakes    = [(p, l) for p, l in dataset if l == "1"]
    nonwakes = [(p, l) for p, l in dataset if l == "0"]
    n = min(sample_size // 2, len(wakes), len(nonwakes))
    if n == 0:
        return np.zeros((0, 1)), np.zeros(0), np.zeros(0)

    subset = random.sample(wakes, n) + random.sample(nonwakes, n)
    random.shuffle(subset)

    loader = DataLoader(
        AudioDataset(subset, aug_prob=aug_prob),
        batch_size=min(128, len(subset)), shuffle=False,
        collate_fn=lambda b: collate_fn(b, device),
    )

    embeddings, labels, confs = [], [], []
    model.eval()
    with torch.no_grad():
        for wavs, lbls, _ in loader:
            emb = model.embed(wavs)
            logits = model(wavs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            embeddings.append(emb.cpu().numpy())
            labels.extend(lbls.cpu().numpy().tolist())
            confs.extend(probs.tolist())

    return (
        np.concatenate(embeddings, axis=0),
        np.array(labels, dtype=int),
        np.array(confs, dtype=float),
    )


def _embedding_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Compute scalar embedding quality metrics."""
    pos = embeddings[labels == 1]
    neg = embeddings[labels == 0]
    metrics: dict = {}

    if len(pos) > 1 and len(neg) > 1:
        c_pos = pos.mean(axis=0)
        c_neg = neg.mean(axis=0)
        centroid_dist = float(np.linalg.norm(c_pos - c_neg))
        intra_pos = float(np.mean(np.linalg.norm(pos - c_pos, axis=1)))
        intra_neg = float(np.mean(np.linalg.norm(neg - c_neg, axis=1)))
        # Fisher-like ratio: between-class / within-class
        fisher = centroid_dist / max(1e-8, (intra_pos + intra_neg) / 2)

        # Cosine similarity between class centroids (lower = better separation)
        norm_pos = c_pos / (np.linalg.norm(c_pos) + 1e-8)
        norm_neg = c_neg / (np.linalg.norm(c_neg) + 1e-8)
        centroid_cos = float(np.dot(norm_pos, norm_neg))

        metrics = {
            "embed_centroid_dist":    centroid_dist,
            "embed_intra_pos":        intra_pos,
            "embed_intra_neg":        intra_neg,
            "embed_fisher_ratio":     fisher,
            "embed_centroid_cos_sim": centroid_cos,
        }

    try:
        from sklearn.metrics import silhouette_score
        if len(set(labels)) > 1 and len(labels) >= 4:
            metrics["embed_silhouette"] = float(
                silhouette_score(embeddings, labels, metric="cosine")
            )
    except Exception:
        pass

    return metrics


def _make_embed_figure(
    proj: np.ndarray,
    labels: np.ndarray,
    confs: np.ndarray,
    method: str,
    epoch: int,
    explained_var: Optional[float] = None,
) -> plt.Figure:
    """Three-panel embedding figure: labels | confidence | hard-neg highlight."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{method} — Epoch {epoch}"
        + (f"  (explained var {explained_var:.1%})" if explained_var else ""),
        fontsize=12, y=1.01,
    )

    wake_mask  = labels == 1
    neg_mask   = labels == 0
    hard_mask  = (labels == 0) & (confs >= 0.5)   # false positives the model is confused about
    easy_mask  = (labels == 0) & (confs < 0.5)

    # ── Panel 1: class labels ────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(proj[easy_mask, 0],  proj[easy_mask, 1],
               c="#4a90d9", s=12, alpha=0.45, label=f"Nonwake easy ({easy_mask.sum()})")
    ax.scatter(proj[hard_mask, 0],  proj[hard_mask, 1],
               c="#e74c3c", s=20, alpha=0.8, marker="x", linewidths=1.2,
               label=f"Nonwake HARD ({hard_mask.sum()})")
    ax.scatter(proj[wake_mask, 0],  proj[wake_mask, 1],
               c="#27ae60", s=20, alpha=0.75, marker="^",
               label=f"Wake ({wake_mask.sum()})")
    # draw class centroids
    for mask, color, marker in [(wake_mask, "#1a7a40", "^"), (neg_mask, "#1a5fa0", "o")]:
        if mask.sum() > 0:
            cx, cy = proj[mask, 0].mean(), proj[mask, 1].mean()
            ax.scatter([cx], [cy], c=color, s=120, marker=marker,
                       edgecolors="white", linewidths=1.5, zorder=5)
    ax.set_title("Class labels  (▲=wake, ×=hard neg)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xticks([]); ax.set_yticks([])

    # ── Panel 2: confidence heatmap ──────────────────────────────────────────
    ax = axes[1]
    sc = ax.scatter(proj[:, 0], proj[:, 1],
                    c=confs, cmap="RdYlGn_r", vmin=0, vmax=1,
                    s=14, alpha=0.75)
    plt.colorbar(sc, ax=ax, label="P(wake)", shrink=0.8)
    ax.set_title("Confidence  (green=low, red=high P(wake))")
    ax.set_xticks([]); ax.set_yticks([])

    # ── Panel 3: confidence histogram by class ───────────────────────────────
    ax = axes[2]
    bins = np.linspace(0, 1, 25)
    ax.hist(confs[wake_mask],  bins=bins, alpha=0.6, color="#27ae60", label="Wake",    density=True)
    ax.hist(confs[neg_mask],   bins=bins, alpha=0.6, color="#e74c3c", label="Nonwake", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="threshold=0.5")
    ax.set_xlabel("P(wake)"); ax.set_ylabel("Density")
    ax.set_title("Confidence distribution")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def log_pca(
    model,
    dataset: List[Tuple[str, str]],
    outdir: "Path | str",
    epoch: int,
    device,
    sample_size: int = 300,
    mlflow=None,
) -> Tuple[Optional[str], dict]:
    """PCA embedding plot — 3 panels: labels, confidence heatmap, conf histogram.

    Also logs scalar embedding metrics (centroid distance, Fisher ratio,
    silhouette) to MLflow.

    Returns:
        (path_or_None, embed_metrics_dict)
    """
    import torch
    if not hasattr(model, "embed"):
        return None, {}

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    embeddings, labels, confs = _collect_embeddings(model, dataset, device, sample_size)
    if len(embeddings) == 0:
        return None, {}

    pca = PCA(n_components=2)
    proj = pca.fit_transform(embeddings)
    explained = float(pca.explained_variance_ratio_.sum())

    fig = _make_embed_figure(proj, labels, confs, "PCA", epoch, explained_var=explained)
    outpath = outdir / f"pca_epoch_{epoch:04d}.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    embed_metrics = _embedding_metrics(embeddings, labels)
    embed_metrics["embed_pca_var_explained"] = explained
    if mlflow:
        try:
            mlflow.log_metrics(embed_metrics, step=epoch)
            mlflow.log_artifact(str(outpath), artifact_path="viz/pca")
        except Exception as e:
            logger.warning("MLflow PCA log failed: %s", e)

    logger.info("[PCA] epoch=%d  centroid_dist=%.3f  fisher=%.3f  silhouette=%.3f",
                epoch,
                embed_metrics.get("embed_centroid_dist", 0),
                embed_metrics.get("embed_fisher_ratio", 0),
                embed_metrics.get("embed_silhouette", float("nan")))
    return str(outpath), embed_metrics


def log_tsne(
    model,
    dataset: List[Tuple[str, str]],
    outdir: "Path | str",
    epoch: int,
    device,
    sample_size: int = 300,
    mlflow=None,
) -> Tuple[Optional[str], dict]:
    """t-SNE embedding plot — same 3-panel layout as log_pca, also logs embedding metrics.

    Returns:
        (path_or_None, embed_metrics_dict)
    """
    import torch
    if not hasattr(model, "embed") or not dataset:
        return None, {}

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    embeddings, labels, confs = _collect_embeddings(model, dataset, device, sample_size)
    if len(embeddings) == 0:
        return None, {}

    perplexity = min(30, max(5, len(embeddings) // 4))
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                    init="pca", learning_rate="auto", n_iter=1000)
        proj = tsne.fit_transform(embeddings)
    except Exception as e:
        logger.error("t-SNE failed: %s", e)
        return None, {}

    fig = _make_embed_figure(proj, labels, confs, "t-SNE", epoch)
    outpath = outdir / f"tsne_epoch_{epoch:04d}.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    embed_metrics = _embedding_metrics(embeddings, labels)
    if mlflow:
        try:
            mlflow.log_metrics(embed_metrics, step=epoch)
            mlflow.log_artifact(str(outpath), artifact_path="viz/tsne")
        except Exception as e:
            logger.warning("MLflow t-SNE log failed: %s", e)

    return str(outpath), embed_metrics


def log_umap(
    model,
    dataset: List[Tuple[str, str]],
    outdir: "Path | str",
    epoch: int,
    device,
    sample_size: int = 300,
    mlflow=None,
) -> Tuple[Optional[str], dict]:
    """UMAP embedding plot (falls back to t-SNE if umap-learn not installed).

    Returns:
        (path_or_None, embed_metrics_dict)
    """
    import torch
    if not hasattr(model, "embed"):
        return None, {}

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    embeddings, labels, confs = _collect_embeddings(model, dataset, device, sample_size)
    if len(embeddings) == 0:
        return None, {}

    if _HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        method = "UMAP"
    else:
        perplexity = min(30, max(5, len(embeddings) // 4))
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                       init="pca", learning_rate="auto")
        method = "t-SNE (UMAP fallback)"
        _logger.warning("umap-learn not installed — using t-SNE fallback. pip install umap-learn")

    try:
        proj = reducer.fit_transform(embeddings)
    except Exception as e:
        logger.error("%s failed: %s", method, e)
        return None, {}

    fig = _make_embed_figure(proj, labels, confs, method, epoch)
    outpath = outdir / f"umap_epoch_{epoch:04d}.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    embed_metrics = _embedding_metrics(embeddings, labels)
    if mlflow:
        try:
            mlflow.log_metrics(embed_metrics, step=epoch)
            mlflow.log_artifact(str(outpath), artifact_path="viz/umap")
        except Exception as e:
            logger.warning("MLflow UMAP log failed: %s", e)

    return str(outpath), embed_metrics


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


# ── Genetic-search plots ────────────────────────────────────────────────────

def plot_pareto_front(
    all_results: list[dict],
    param_budget: int,
    title: str = "Pareto front",
    out_path: Optional[Path] = None,
) -> Optional[Path]:
    """Scatter of n_params vs fitness, colored by F1, with the Pareto frontier.

    Non-dominated solutions (maximize fitness, minimize n_params) are connected
    by a step line.  A vertical dashed line marks *param_budget*.

    Args:
        all_results:  List of trial result dicts with keys ``fitness``,
                      ``f1``, ``n_params``, and optionally ``stage``.
        param_budget: Parameter count budget — drawn as a vertical line.
        title:        Plot title.
        out_path:     Where to save the PNG.  Returns ``None`` when omitted.
    """
    if not all_results:
        return None
    try:
        fitnesses  = np.array([r["fitness"]  for r in all_results], dtype=float)
        n_params   = np.array([r.get("n_params", 0) for r in all_results], dtype=float)
        f1s        = np.array([r["f1"]        for r in all_results], dtype=float)
        stages     = np.array([r.get("stage", 1)    for r in all_results], dtype=int)

        markers = {1: "o", 2: "s", 3: "^"}

        fig, ax = plt.subplots(figsize=(8, 5))

        for s, marker in markers.items():
            mask = stages == s
            if not mask.any():
                continue
            sc = ax.scatter(
                n_params[mask], fitnesses[mask],
                c=f1s[mask], cmap="YlGn", vmin=0, vmax=1,
                marker=marker, s=40, alpha=0.75,
                label=f"Stage {s}",
            )

        plt.colorbar(sc, ax=ax, label="F1")

        # Pareto frontier: non-dominated (max fitness, min n_params)
        order = np.argsort(n_params)
        pf_x, pf_y = [], []
        best_fit = -np.inf
        for idx in order:
            if fitnesses[idx] > best_fit:
                best_fit = fitnesses[idx]
                pf_x.append(n_params[idx])
                pf_y.append(fitnesses[idx])
        if len(pf_x) > 1:
            ax.step(pf_x, pf_y, where="post", color="#e74c3c", lw=1.5,
                    label="Pareto frontier", zorder=5)

        ax.axvline(param_budget, color="#3498db", ls="--", lw=1.2,
                   label=f"Budget ({param_budget:,})")

        ax.set_xlabel("Parameters"); ax.set_ylabel("Fitness")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path
        plt.close(fig)
    except Exception as exc:
        logger.warning("plot_pareto_front failed: %s", exc)
    return None


def plot_config_analysis(
    all_results: list[dict],
    search_space: dict[str, list],
    top_k: int = 10,
    out_path: Optional[Path] = None,
) -> Optional[Path]:
    """Bar chart of search-space value frequencies in top-K vs bottom-K trials.

    For each key in *search_space*, shows how often each candidate value
    appears among the top-*top_k* results (by fitness) vs the bottom-*top_k*.

    Args:
        all_results:  Trial result dicts; each must have a ``config`` key.
        search_space: Dict mapping param name → list of candidate values.
        top_k:        Number of best / worst trials to compare.
        out_path:     Where to save the PNG.
    """
    if not all_results or not search_space:
        return None
    try:
        sorted_results = sorted(all_results, key=lambda r: r["fitness"], reverse=True)
        top    = sorted_results[:top_k]
        bottom = sorted_results[-top_k:]

        keys   = list(search_space.keys())
        n_keys = len(keys)
        ncols  = min(4, n_keys)
        nrows  = (n_keys + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes_flat = np.array(axes).flatten() if n_keys > 1 else [axes]

        for i, key in enumerate(keys):
            ax     = axes_flat[i]
            values = [str(v) for v in search_space[key]]

            top_counts    = [sum(1 for r in top    if str(r.get("config", {}).get(key)) == v) for v in values]
            bottom_counts = [sum(1 for r in bottom if str(r.get("config", {}).get(key)) == v) for v in values]

            x   = np.arange(len(values))
            w   = 0.35
            ax.bar(x - w / 2, top_counts,    w, label=f"Top {top_k}",    color="#2ecc71", alpha=0.8)
            ax.bar(x + w / 2, bottom_counts, w, label=f"Bottom {top_k}", color="#e74c3c", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(values, rotation=30, ha="right", fontsize=8)
            ax.set_title(key, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)

        for j in range(n_keys, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(f"Config value frequency — top vs bottom {top_k}", fontsize=11)
        fig.tight_layout()

        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path
        plt.close(fig)
    except Exception as exc:
        logger.warning("plot_config_analysis failed: %s", exc)
    return None
