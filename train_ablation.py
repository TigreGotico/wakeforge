"""
Ablation study: losses × augmentation methods.

Trains every (loss, augmentation_combo) pair for a fixed epoch budget,
logs everything to MLflow, then generates summary heatmaps + DET overlays.

Designed to run unattended for ~24 h on a CPU-only machine.

Usage
-----
    .venv/bin/python train_ablation.py
    .venv/bin/python train_ablation.py --epochs 60 --budget-hours 20
    .venv/bin/python train_ablation.py --losses bce focal rppl --augs none all
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ww_trainer.env import load_env, env_default
load_env()

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")

import torch
torch.set_num_threads(12)
torch.set_num_interop_threads(4)

import psutil
if psutil.virtual_memory().available / 1e9 < 8:
    sys.exit("ERROR: less than 8 GB RAM available — aborting")
if psutil.disk_usage("/").free / 1e9 < 20:
    sys.exit("ERROR: less than 20 GB disk free — aborting")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_ablation")

# Raise the open-file-descriptor limit so a 100+ cell ablation doesn't exhaust it.
import resource as _resource
_soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
_target = min(_hard, 65536)
if _soft < _target:
    _resource.setrlimit(_resource.RLIMIT_NOFILE, (_target, _hard))
    logger.info("Raised RLIMIT_NOFILE: %d → %d", _soft, _target)

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--epochs",        type=int,   default=env_default("WW_EPOCHS",       50,    int))
parser.add_argument("--budget-hours",  type=float, default=env_default("WW_BUDGET_HOURS", 24.0,  float))
parser.add_argument("--arch",          type=str,   default=env_default("WW_ARCH",         "gru"))
parser.add_argument("--hidden-dim",    type=int,   default=env_default("WW_HIDDEN_DIM",   128,   int))
parser.add_argument("--lr",            type=float, default=env_default("WW_LR",           5e-4,  float))
parser.add_argument("--batch-size",    type=int,   default=env_default("WW_BATCH_SIZE",   16,    int))
parser.add_argument("--patience",      type=int,   default=env_default("WW_PATIENCE",     10,    int))
_losses_env = os.environ.get("WW_LOSSES")
_augs_env   = os.environ.get("WW_AUGS")
parser.add_argument(
    "--losses", nargs="+",
    default=_losses_env.split() if _losses_env else [
        "bce", "focal", "label_smoothing_bce", "rppl",
        "size_aware", "supcon", "arcface", "ntxent",
        "triplet", "contrastive", "angular", "multi_similarity",
        "center", "proxy_nca",
    ],
)
parser.add_argument(
    "--augs", nargs="+",
    default=_augs_env.split() if _augs_env else [
        "none", "bg_noise", "music", "rir", "spec_augment", "mixup",
        "bg_noise+rir", "all",
    ],
)
parser.add_argument("--resume",  action="store_true",
                    help="Skip cells that already have a best_f1.pt")
args = parser.parse_args()

DEADLINE = time.monotonic() + args.budget_hours * 3600

# ── Paths ──────────────────────────────────────────────────────────────────────

EXPERIMENT_NAME = env_default("WW_EXPERIMENT_NAME", "hey_mycroft")
DATASET_NAME    = env_default("WW_DATASET_NAME",    "hey_mycroft")

BASE        = Path(f"experiments/{EXPERIMENT_NAME}")
DATASET_DIR = Path(f"experiments/{DATASET_NAME}") / "dataset"
TRAIN_CSV   = DATASET_DIR / "train" / "metadata.csv"
TEST_CSV    = DATASET_DIR / "test"  / "metadata.csv"
NWW_DIR     = DATASET_DIR / "negatives" / "not_wake_word_subset"
AUG_DIR     = DATASET_DIR / "augmentation"
OUT_BASE    = BASE / "ablation"
PLOTS_DIR   = OUT_BASE / "plots"
RESULTS_JSON = OUT_BASE / "results.json"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI")

# ── Augmentation combos ────────────────────────────────────────────────────────

def _aug_opts(combo: str) -> dict:
    """Return WakeWordTrainer kwargs for a given augmentation combo name."""
    bg  = str(AUG_DIR / "bg_noise") if (AUG_DIR / "bg_noise").exists() else None
    mus = str(AUG_DIR / "music")    if (AUG_DIR / "music").exists()    else None
    rir = str(AUG_DIR / "rir")      if (AUG_DIR / "rir").exists()      else None
    wow = str(NWW_DIR)              if NWW_DIR.exists()                 else None

    opts: dict = {}
    parts = set(combo.split("+"))

    if "none" in parts:
        return {}

    if "bg_noise" in parts and bg:
        opts["bg_noise_folder"] = bg
    if "music" in parts and mus:
        opts["music_folder"] = mus
    if "rir" in parts and rir:
        opts["rir_folder"] = rir
    if "all" in parts:
        if bg:  opts["bg_noise_folder"] = bg
        if mus: opts["music_folder"]    = mus
        if rir: opts["rir_folder"]      = rir
        if wow: opts["wake_word_over_speech_folder"] = wow

    return opts


def _train_opts(combo: str) -> dict:
    """Return trainer.train() kwargs for a given augmentation combo name."""
    parts = set(combo.split("+"))
    use_spec = "spec_augment" in parts or "all" in parts
    use_mix  = "mixup"        in parts or "all" in parts
    return {
        "use_mixup":           use_mix,
        "mixup_alpha":         1.0,
        "spec_augment":        use_spec,
        "neg_weight_schedule": "linear",
    }

# ── Dataset ────────────────────────────────────────────────────────────────────

def _read_csv(p: Path) -> list:
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(",", 1)
                if len(parts) == 2:
                    rows.append((parts[0], parts[1]))
    return rows


def build_dataset() -> tuple[list, list]:
    if not TRAIN_CSV.exists():
        sys.exit(f"Dataset not found at {TRAIN_CSV}. Run train_hey_mycroft.py first.")

    train_data = _read_csv(TRAIN_CSV)
    test_data  = _read_csv(TEST_CSV)

    if NWW_DIR.exists():
        nww_wavs = sorted(NWW_DIR.glob("not_wake_word_*.wav"))
        rng = random.Random(42)
        rng.shuffle(nww_wavs)
        split = int(len(nww_wavs) * 0.8)
        train_data += [(str(p), "0") for p in nww_wavs[:split]]
        test_data  += [(str(p), "0") for p in nww_wavs[split:]]
        logger.info("NWW: %d files from %s", len(nww_wavs), NWW_DIR)
    else:
        logger.warning("NWW subset not found at %s — training without it", NWW_DIR)

    pos = sum(1 for _, l in train_data if l == "1")
    neg = sum(1 for _, l in train_data if l == "0")
    logger.info("Dataset: %d train (%d pos / %d neg, %.1f:1) | %d test",
                len(train_data), pos, neg, neg / max(1, pos), len(test_data))
    return train_data, test_data

# ── Single cell ────────────────────────────────────────────────────────────────

def run_cell(
    loss: str,
    aug_combo: str,
    train_data: list,
    test_data: list,
    feature_cache,
    cell_dir: Path,
) -> dict | None:
    """Train one (loss, aug) cell. Returns result dict or None on failure."""
    from ww_trainer.trainer import WakeWordTrainer
    from ww_trainer.evaluation import evaluate_model

    cell_dir.mkdir(parents=True, exist_ok=True)

    # Losses that need embed_dim
    losses_cfg = [{"name": loss, "weight": 1.0}]
    if loss in ("arcface", "center", "proxy_nca"):
        losses_cfg[0]["embed_dim"] = args.hidden_dim

    aug_kwargs   = _aug_opts(aug_combo)
    train_kwargs = _train_opts(aug_combo)

    t0 = time.monotonic()
    try:
        trainer = WakeWordTrainer(
            arch=args.arch,
            featurizer="",
            feature_dim=None,
            hidden_dim=args.hidden_dim,
            dropout=0.1,
            device="cpu",
            losses_cfg=losses_cfg,
            featurizer_type="mfcc",
            n_mfcc=40,
            export_onnx=True,
            seed=42,
            wake_word=EXPERIMENT_NAME,
            mlflow_uri=MLFLOW_URI,
            **aug_kwargs,
        )

        # Tag this run in MLflow
        if trainer.mlflow is not None:
            try:
                trainer.mlflow.set_tag("ablation.loss",      loss)
                trainer.mlflow.set_tag("ablation.aug_combo", aug_combo)
                trainer.mlflow.set_tag("ablation.study",     "loss_x_aug")
                trainer.mlflow.log_param("aug_bg_noise",    "bg_noise_folder"  in aug_kwargs)
                trainer.mlflow.log_param("aug_music",       "music_folder"     in aug_kwargs)
                trainer.mlflow.log_param("aug_rir",         "rir_folder"       in aug_kwargs)
                trainer.mlflow.log_param("aug_spec_augment", train_kwargs.get("spec_augment", False))
                trainer.mlflow.log_param("aug_mixup",        train_kwargs.get("use_mixup",    False))
            except Exception:
                pass

        trainer.train(
            output_dir=cell_dir,
            train_data=train_data,
            test_data=test_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            mine_fraction=0.33,
            patience=args.patience,
            feature_cache=feature_cache,
            **train_kwargs,
        )

        elapsed = time.monotonic() - t0

        # ── Deep evaluation ────────────────────────────────────────────────────
        result = evaluate_model(
            trainer.model, test_data, trainer.device,
            batch_size=128, threshold=0.5,
            feature_cache=feature_cache,
        )
        acc, prec, rec, f1, auc, fp_paths, fn_paths, paths_all, targets, preds, probs, report = result

        n_params = sum(p.numel() for p in trainer.model.parameters())

        cell_result = {
            "loss":        loss,
            "aug":         aug_combo,
            "f1":          float(f1),
            "precision":   float(prec),
            "recall":      float(rec),
            "accuracy":    float(acc),
            "auc":         float(auc),
            "eer":         float(report.eer),
            "eer_threshold": float(report.eer_threshold),
            "far":         float(report.far),
            "frr":         float(report.frr),
            "threshold":   float(report.threshold),
            "far_at_frr1": float(report.far_at_frr.get(0.01, 1.0)),
            "far_at_frr5": float(report.far_at_frr.get(0.05, 1.0)),
            "far_at_frr10":float(report.far_at_frr.get(0.10, 1.0)),
            "n_fp":        len(fp_paths),
            "n_fn":        len(fn_paths),
            "n_pos":       int((targets == 1).sum()),
            "n_neg":       int((targets == 0).sum()),
            "n_params":    n_params,
            "elapsed_s":   float(elapsed),
        }

        # Log summary metrics back to MLflow
        if trainer.mlflow is not None:
            try:
                mlf = trainer.mlflow
                mlf.log_metric("final/f1",          cell_result["f1"])
                mlf.log_metric("final/precision",   cell_result["precision"])
                mlf.log_metric("final/recall",      cell_result["recall"])
                mlf.log_metric("final/accuracy",    cell_result["accuracy"])
                mlf.log_metric("final/auc",         cell_result["auc"])
                mlf.log_metric("final/eer",         cell_result["eer"])
                mlf.log_metric("final/far",         cell_result["far"])
                mlf.log_metric("final/frr",         cell_result["frr"])
                mlf.log_metric("final/far_at_frr1", cell_result["far_at_frr1"])
                mlf.log_metric("final/far_at_frr5", cell_result["far_at_frr5"])
                mlf.log_metric("final/far_at_frr10",cell_result["far_at_frr10"])
                mlf.log_metric("final/n_fp",        cell_result["n_fp"])
                mlf.log_metric("final/n_fn",        cell_result["n_fn"])
                mlf.log_metric("final/n_params",    cell_result["n_params"])
                mlf.log_metric("final/elapsed_s",   cell_result["elapsed_s"])
                mlf.log_metric("final/threshold",   cell_result["threshold"])
            except Exception:
                pass

        logger.info(
            "[%s × %s] f1=%.4f eer=%.4f far=%.4f frr=%.4f auc=%.4f  (%.0fs)",
            loss, aug_combo,
            cell_result["f1"], cell_result["eer"],
            cell_result["far"], cell_result["frr"], cell_result["auc"],
            elapsed,
        )
        return cell_result

    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.error("[%s × %s] FAILED after %.0fs: %s", loss, aug_combo, elapsed, exc,
                     exc_info=True)
        return {
            "loss": loss, "aug": aug_combo,
            "f1": 0.0, "eer": 1.0, "far": 1.0, "frr": 1.0,
            "auc": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0,
            "far_at_frr1": 1.0, "far_at_frr5": 1.0, "far_at_frr10": 1.0,
            "n_fp": 0, "n_fn": 0, "n_pos": 0, "n_neg": 0,
            "threshold": 0.5, "eer_threshold": 0.5, "n_params": 0,
            "elapsed_s": float(elapsed), "error": str(exc),
        }

# ── Plots ──────────────────────────────────────────────────────────────────────

def _pivot(results: list[dict], metric: str) -> tuple[np.ndarray, list, list]:
    losses = args.losses
    augs   = args.augs
    mat = np.full((len(losses), len(augs)), np.nan)
    for r in results:
        if r["loss"] in losses and r["aug"] in augs:
            i = losses.index(r["loss"])
            j = augs.index(r["aug"])
            mat[i, j] = r.get(metric, np.nan)
    return mat, losses, augs


def plot_heatmap(results: list[dict], metric: str, title: str, cmap: str = "YlGn",
                 vmin: float = 0.0, vmax: float = 1.0, invert: bool = False) -> None:
    mat, losses, augs = _pivot(results, metric)
    if invert:
        display = 1.0 - mat
        cmap = cmap + "_r" if not cmap.endswith("_r") else cmap[:-2]
    else:
        display = mat

    fig, ax = plt.subplots(figsize=(max(6, len(augs) * 1.3), max(4, len(losses) * 0.7)))
    im = ax.imshow(display, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ax.set_xticks(range(len(augs)));   ax.set_xticklabels(augs,   rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(losses))); ax.set_yticklabels(losses, fontsize=8)
    ax.set_xlabel("Augmentation combo"); ax.set_ylabel("Loss")
    ax.set_title(title, fontsize=10, pad=10)

    for i in range(len(losses)):
        for j in range(len(augs)):
            v = mat[i, j]
            if not np.isnan(v):
                txt = f"{v:.3f}"
                bg  = display[i, j] if not np.isnan(display[i, j]) else 0.5
                col = "white" if bg < 0.4 or bg > 0.85 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=col)

    fig.tight_layout()
    path = PLOTS_DIR / f"heatmap_{metric}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)

    # Upload to MLflow if available
    try:
        import mlflow as _mlflow
        _mlflow.log_artifact(str(path), artifact_path="ablation_plots")
    except Exception:
        pass


def plot_best_per_loss(results: list[dict]) -> None:
    """Bar chart: best F1 per loss (across all augs)."""
    best: dict[str, float] = {}
    best_aug: dict[str, str] = {}
    for r in results:
        l = r["loss"]
        if r["f1"] > best.get(l, -1.0):
            best[l] = r["f1"]
            best_aug[l] = r["aug"]

    sorted_items = sorted(best.items(), key=lambda x: x[1])
    labels = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = ["#2ecc71" if v == max(values) else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.6 + 1)))
    bars = ax.barh(labels, values, color=colors, height=0.55)
    for bar, (lbl, val) in zip(bars, sorted_items):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f} ({best_aug[lbl]})", va="center", fontsize=8)
    ax.set_xlabel("Best F1 (across all aug combos)")
    ax.set_title(f"Best F1 per loss — ablation  (arch={args.arch} h={args.hidden_dim})")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "best_f1_per_loss.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_best_per_aug(results: list[dict]) -> None:
    """Bar chart: best F1 per augmentation combo (across all losses)."""
    best: dict[str, float] = {}
    best_loss: dict[str, str] = {}
    for r in results:
        a = r["aug"]
        if r["f1"] > best.get(a, -1.0):
            best[a] = r["f1"]
            best_loss[a] = r["loss"]

    sorted_items = sorted(best.items(), key=lambda x: x[1])
    labels = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = ["#e74c3c" if v == max(values) else "#e67e22" for v in values]

    fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.6 + 1)))
    bars = ax.barh(labels, values, color=colors, height=0.55)
    for bar, (lbl, val) in zip(bars, sorted_items):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f} ({best_loss[lbl]})", va="center", fontsize=8)
    ax.set_xlabel("Best F1 (across all losses)")
    ax.set_title("Best F1 per augmentation combo — ablation")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "best_f1_per_aug.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_eer_vs_f1(results: list[dict]) -> None:
    """Scatter: EER vs F1, coloured by loss."""
    fig, ax = plt.subplots(figsize=(8, 6))
    loss_names = sorted(set(r["loss"] for r in results))
    cmap = plt.get_cmap("tab20")
    for i, loss in enumerate(loss_names):
        pts = [r for r in results if r["loss"] == loss and r["f1"] > 0]
        if not pts:
            continue
        xs = [r["f1"]  for r in pts]
        ys = [r["eer"] for r in pts]
        ax.scatter(xs, ys, label=loss, color=cmap(i / len(loss_names)),
                   s=60, alpha=0.8, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("F1"); ax.set_ylabel("EER")
    ax.set_title("EER vs F1 across all cells (lower-right = better)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    path = PLOTS_DIR / "eer_vs_f1.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_summary_table(results: list[dict]) -> None:
    """Text-style ranked summary table saved as PNG."""
    sorted_r = sorted(results, key=lambda r: r["f1"], reverse=True)
    top = sorted_r[:min(20, len(sorted_r))]

    cols = ["rank", "loss", "aug", "F1", "EER", "FAR", "FRR", "AUC", "prec", "rec"]
    rows = []
    for rank, r in enumerate(top, 1):
        rows.append([
            str(rank), r["loss"], r["aug"],
            f"{r['f1']:.4f}", f"{r['eer']:.4f}",
            f"{r['far']:.4f}", f"{r['frr']:.4f}",
            f"{r['auc']:.4f}", f"{r['precision']:.4f}", f"{r['recall']:.4f}",
        ])

    fig, ax = plt.subplots(figsize=(16, max(3, len(rows) * 0.4 + 1.5)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(list(range(len(cols))))

    # Color top row green
    for j in range(len(cols)):
        tbl[(1, j)].set_facecolor("#d5f5e3")

    ax.set_title(f"Top-{len(rows)} cells by F1 — ablation study", fontsize=10, pad=10)
    fig.tight_layout()
    path = PLOTS_DIR / "top_cells_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def generate_all_plots(results: list[dict]) -> None:
    if not results:
        return

    plot_heatmap(results, "f1",    "F1 score — loss × augmentation",         "YlGn",  0.0, 1.0)
    plot_heatmap(results, "eer",   "EER — loss × augmentation",              "RdYlGn", 0.0, 0.5, invert=True)
    plot_heatmap(results, "auc",   "AUC — loss × augmentation",              "Blues",  0.5, 1.0)
    plot_heatmap(results, "far",   "FAR — loss × augmentation",              "Reds",   0.0, 0.5, invert=True)
    plot_heatmap(results, "frr",   "FRR — loss × augmentation",              "Reds",   0.0, 0.5, invert=True)
    plot_heatmap(results, "recall","Recall — loss × augmentation",           "Purples",0.0, 1.0)
    plot_heatmap(results, "far_at_frr5", "FAR@FRR=5% — loss × augmentation","Oranges",0.0, 0.5, invert=True)

    plot_best_per_loss(results)
    plot_best_per_aug(results)
    plot_eer_vs_f1(results)
    plot_summary_table(results)

    # Upload all plots to MLflow summary run
    if MLFLOW_URI:
        try:
            import mlflow as _mlflow
            for p in PLOTS_DIR.glob("*.png"):
                _mlflow.log_artifact(str(p), artifact_path="ablation_plots")
        except Exception:
            pass


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    train_data, test_data = build_dataset()

    # Pre-load all audio into shared memory (done once)
    all_paths = list({p for p, _ in train_data + test_data})
    from ww_trainer.cache import SharedWaveformCache
    cache = SharedWaveformCache(sr=16000, max_duration=2.0)
    cache.preload(all_paths, n_workers=1, show_progress=True)
    logger.info("Cache: %s", cache.stats())

    # Build full grid, shuffle so partial runs sample evenly
    grid = list(product(args.losses, args.augs))
    random.seed(0)
    random.shuffle(grid)
    total = len(grid)
    logger.info("Grid: %d losses × %d augs = %d cells  (budget %.1fh)",
                len(args.losses), len(args.augs), total, args.budget_hours)

    # Load previous results if resuming
    results: list[dict] = []
    if RESULTS_JSON.exists():
        try:
            results = json.loads(RESULTS_JSON.read_text())
            logger.info("Loaded %d prior results from %s", len(results), RESULTS_JSON)
        except Exception:
            pass

    done_keys = {(r["loss"], r["aug"]) for r in results}

    # Open a parent MLflow run for the whole study
    try:
        if not MLFLOW_URI:
            raise RuntimeError("MLFLOW_TRACKING_URI not set — skipping MLflow")
        import mlflow as _mlflow
        _mlflow.set_tracking_uri(MLFLOW_URI)
        _mlflow.set_experiment(f"ww: {EXPERIMENT_NAME.replace('_', ' ')}")
        parent_run = _mlflow.start_run(
            run_name=f"ablation · {args.arch} · {args.epochs}ep",
            tags={
                "study":      "ablation",
                "arch":       args.arch,
                "hidden_dim": str(args.hidden_dim),
                "n_losses":   str(len(args.losses)),
                "n_augs":     str(len(args.augs)),
                "n_cells":    str(total),
            },
        )
        parent_run_id = parent_run.info.run_id
        logger.info("MLflow parent run: %s", parent_run_id)
    except Exception as exc:
        logger.warning("MLflow parent run failed: %s", exc)
        parent_run_id = None

    completed = 0
    skipped   = 0
    failed    = 0

    for loss, aug in grid:
        if time.monotonic() > DEADLINE:
            logger.warning("Budget exhausted — stopping after %d/%d cells", completed + skipped, total)
            break

        key = (loss, aug)
        cell_dir = OUT_BASE / f"{loss}__{aug.replace('+', '_')}"

        if args.resume and key in done_keys:
            logger.info("Skipping %s × %s (already done)", loss, aug)
            skipped += 1
            continue
        if args.resume and (cell_dir / "best_f1.pt").exists() and key not in done_keys:
            logger.info("Skipping %s × %s (model exists, not in results — will re-eval)", loss, aug)

        elapsed_so_far = (completed + skipped + failed + 1)
        remaining      = total - elapsed_so_far
        logger.info(
            "━━ Cell %d/%d  [%s × %s]  (remaining: %d) ━━",
            elapsed_so_far, total, loss, aug, remaining,
        )

        result = run_cell(loss, aug, train_data, test_data, cache, cell_dir)
        if result is not None:
            # Remove any previous result for this key (re-run)
            results = [r for r in results if not (r["loss"] == loss and r["aug"] == aug)]
            results.append(result)
            done_keys.add(key)

            if "error" in result:
                failed += 1
            else:
                completed += 1

            # Persist incrementally
            RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
            RESULTS_JSON.write_text(json.dumps(results, indent=2))

            # Regenerate plots every 5 successful cells
            if completed % 5 == 0:
                generate_all_plots(results)

    # ── Final plots & summary ──────────────────────────────────────────────────
    generate_all_plots(results)

    # Close parent MLflow run and log final summary metrics
    if parent_run_id is not None:
        try:
            import mlflow as _mlflow
            with _mlflow.start_run(run_id=parent_run_id):
                good = [r for r in results if "error" not in r and r["f1"] > 0]
                if good:
                    best = max(good, key=lambda r: r["f1"])
                    _mlflow.log_metric("summary/best_f1",   best["f1"])
                    _mlflow.log_metric("summary/best_eer",  best["eer"])
                    _mlflow.log_metric("summary/best_far",  best["far"])
                    _mlflow.log_metric("summary/best_auc",  best["auc"])
                    _mlflow.log_param("summary/best_loss",  best["loss"])
                    _mlflow.log_param("summary/best_aug",   best["aug"])
                    _mlflow.log_metric("summary/n_completed", completed)
                    _mlflow.log_metric("summary/n_failed",    failed)
                    for p in PLOTS_DIR.glob("*.png"):
                        _mlflow.log_artifact(str(p), artifact_path="ablation_plots")
                    _mlflow.log_artifact(str(RESULTS_JSON), artifact_path="ablation_plots")
        except Exception as exc:
            logger.warning("Could not close parent MLflow run: %s", exc)

    # ── Print ranked table ─────────────────────────────────────────────────────
    good = [r for r in results if "error" not in r]
    ranked = sorted(good, key=lambda r: r["f1"], reverse=True)
    print("\n" + "═" * 90)
    print(f"  Ablation Results — hey mycroft  arch={args.arch} h={args.hidden_dim}  "
          f"({completed} ok / {failed} failed / {skipped} skipped)")
    print("═" * 90)
    print(f"  {'Loss':<22} {'Aug':<16} {'F1':>7} {'EER':>7} {'FAR':>7} "
          f"{'FRR':>7} {'AUC':>7} {'Prec':>7} {'Rec':>7}")
    print("  " + "─" * 85)
    for r in ranked[:30]:
        print(f"  {r['loss']:<22} {r['aug']:<16} {r['f1']:>7.4f} {r['eer']:>7.4f} "
              f"{r['far']:>7.4f} {r['frr']:>7.4f} {r['auc']:>7.4f} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f}")
    print("═" * 90)
    if ranked:
        b = ranked[0]
        print(f"\n  Best cell: {b['loss']} × {b['aug']}")
        print(f"    F1={b['f1']:.4f}  EER={b['eer']:.4f}  FAR={b['far']:.4f}  "
              f"FRR={b['frr']:.4f}  AUC={b['auc']:.4f}")
    print(f"\n  Results JSON : {RESULTS_JSON}")
    print(f"  Plots dir    : {PLOTS_DIR}")


if __name__ == "__main__":
    main()
