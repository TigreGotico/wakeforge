"""
Parallel loss-comparison training with a single shared waveform cache.

All worker processes read audio from the same shared-memory pages — the dataset
is loaded once in the main process, marked share_memory_(), then inherited by
each worker via fork (no serialization, no duplication).

Architecture
------------
1. Main process: build augmented train/test data, pre-load ALL WAVs into
   SharedWaveformCache, then fork N worker processes.
2. Each worker: trains one loss function for 50 epochs.
   Workers use the shared cache instead of reading disk, so I/O is eliminated
   after the initial preload.
3. Main process: collect results, render plots.

Workers per batch of parallel runs is capped at MAX_PARALLEL (default 2) to
keep peak RAM under control — swap is already 76% full.

Usage
-----
    python train_parallel.py               # runs all 6 losses, 2 at a time
    python train_parallel.py --max-parallel 3
"""
import argparse
import csv as csv_mod
import logging
import multiprocessing as mp
import os
import random
import resource
import sys
import time
from pathlib import Path

# Raise open-file limit before anything else touches file descriptors.
# share_memory_() uses memfd_create — each shared tensor consumes one fd.
# Default shell soft limit is often 1024; we need one fd per cached waveform.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
_target = min(max(_soft, 65536), _hard)   # raise to ≥65536, never exceed hard limit
if _soft < _target:
    resource.setrlimit(resource.RLIMIT_NOFILE, (_target, _hard))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── MLflow credentials (loaded from .env) ────────────────────────────────────
from ww_trainer.env import load_env, env_default
load_env()

# ── Thread cap — set before any torch ops ────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "6"   # per-worker budget (2 workers × 6 = 12 total)
os.environ["MKL_NUM_THREADS"] = "6"
torch.set_num_threads(6)
torch.set_num_interop_threads(2)

import psutil
vm = psutil.virtual_memory()
if vm.available / 1e9 < 10:
    sys.exit("ERROR: less than 10 GB available — aborting per AGENTS.md")
if psutil.disk_usage("/").free / 1e9 < 50:
    sys.exit("ERROR: less than 50 GB disk free — aborting per AGENTS.md")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_parallel")

# ── Experiment ─────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = env_default("WW_EXPERIMENT_NAME", "hey_mycroft")
DATASET_NAME    = env_default("WW_DATASET_NAME",    "hey_mycroft")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(f"experiments/{EXPERIMENT_NAME}")
DATASET_DIR = Path(f"experiments/{DATASET_NAME}") / "dataset"
TRAIN_CSV   = DATASET_DIR / "train" / "metadata.csv"
TEST_CSV    = DATASET_DIR / "test"  / "metadata.csv"
NWW_DIR = DATASET_DIR / "negatives" / "not_wake_word_subset"
OUT_BASE    = BASE / "models_parallel"
PLOTS_DIR   = BASE / "plots_parallel"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LOSSES = ["bce", "focal", "label_smoothing_bce", "rppl", "supcon", "arcface", "ntxent"]


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _read_csv(p):
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(",", 1)
                if len(parts) == 2:
                    rows.append((parts[0], parts[1]))
    return rows


def _build_dataset():
    if not TRAIN_CSV.exists():
        sys.exit(f"Dataset not found at {TRAIN_CSV}. Run train_hey_mycroft.py first.")
    if not NWW_DIR.exists():
        sys.exit(f"NWW subset not found at {NWW_DIR}. Run the rsync copy first.")

    train_data = _read_csv(TRAIN_CSV)
    test_data  = _read_csv(TEST_CSV)

    nww_wavs = sorted(NWW_DIR.glob("not_wake_word_*.wav"))
    rng = random.Random(42)
    rng.shuffle(nww_wavs)
    split = int(len(nww_wavs) * 0.8)
    train_data += [(str(p), "0") for p in nww_wavs[:split]]
    test_data  += [(str(p), "0") for p in nww_wavs[split:]]

    logger.info("Dataset: %d train / %d test", len(train_data), len(test_data))
    return train_data, test_data


# ── Worker function (runs in child process) ───────────────────────────────────

def _train_worker(
    loss_name: str,
    train_data: list,
    test_data: list,
    shared_cache,           # SharedWaveformCache — shared pages, no copy
    best_cfg: dict,
    augment_opts: dict,
    result_queue: mp.Queue,
):
    """Train one loss for 50 epochs. Puts (loss_name, best_f1, elapsed) into queue."""
    # Thread limits inherited from parent via fork — do not re-set interop threads.

    model_dir = OUT_BASE / f"loss_{loss_name}"
    model_dir.mkdir(parents=True, exist_ok=True)

    if (model_dir / "best_f1.pt").exists():
        logger.info("Skipping %s — already trained", loss_name)
        metrics_csv = model_dir / "metrics_log.csv"
        f1 = 0.0
        if metrics_csv.exists():
            rows = list(csv_mod.DictReader(open(metrics_csv)))
            if rows:
                f1 = float(rows[-1]["f1"])
        result_queue.put((loss_name, f1, 0.0, True))
        return

    from ww_trainer.trainer import WakeWordTrainer

    feat_type  = best_cfg.get("featurizer_type", "mfcc")
    arch       = best_cfg.get("arch", "gru")
    hidden_dim = best_cfg.get("hidden_dim", 128)
    dropout    = best_cfg.get("dropout", 0.1)
    n_features = best_cfg.get("n_features", 40)
    lr         = best_cfg.get("lr", 5e-4)
    batch_size = min(best_cfg.get("batch_size", 16), 16)

    _FEAT_KWARGS = {
        "mfcc":       lambda n: {"featurizer_type": "mfcc",      "n_mfcc":    n},
        "filterbank": lambda n: {"featurizer_type": "filterbank", "n_mels":    n},
        "sincnet":    lambda n: {"featurizer_type": "sincnet",    "n_filters": n},
        "gammatone":  lambda n: {"featurizer_type": "gammatone",  "n_filters": n},
    }
    feat_kwargs = _FEAT_KWARGS.get(feat_type, lambda n: {"featurizer_type": feat_type})(n_features)

    losses_cfg = [{"name": loss_name, "weight": 1.0}]
    if loss_name in ("arcface", "center", "proxy_nca"):
        losses_cfg[0]["embed_dim"] = hidden_dim

    t0 = time.time()
    try:
        trainer = WakeWordTrainer(
            arch=arch,
            featurizer="",
            feature_dim=None,
            hidden_dim=hidden_dim,
            dropout=dropout,
            device="cpu",
            losses_cfg=losses_cfg,
            export_onnx=True,
            seed=42,
            wake_word=EXPERIMENT_NAME,
            mlflow_uri=os.environ.get("MLFLOW_TRACKING_URI"),
            **feat_kwargs,
            **augment_opts,
        )

        best_f1 = trainer.train(
            output_dir=model_dir,
            train_data=train_data,
            test_data=test_data,
            epochs=50,
            batch_size=batch_size,
            lr=lr,
            mine_fraction=0.33,
            patience=7,
            use_mixup=True,
            mixup_alpha=1.0,
            spec_augment=True,
            neg_weight_schedule="linear",
            feature_cache=shared_cache,
        )
    except Exception as exc:
        logger.error("Worker %s failed: %s", loss_name, exc, exc_info=True)
        best_f1 = 0.0

    elapsed = time.time() - t0
    result_queue.put((loss_name, best_f1, elapsed, False))
    logger.info("[%s] Done — F1=%.4f in %.0fs", loss_name, best_f1, elapsed)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_loss_bars(results, best_cfg):
    sorted_r = sorted(results, key=lambda x: x["f1"])
    labels = [r["loss"] for r in sorted_r]
    values = [r["f1"]   for r in sorted_r]
    colors = ["#2ecc71" if v == max(values) else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.65 + 1)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.55)
    for bar, val in zip(bars, values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    arch = best_cfg.get("arch", "?")
    feat = best_cfg.get("featurizer_type", "?")
    ax.set_xlabel("Best F1 (50 epochs)")
    ax.set_title(f"Loss comparison — hey mycroft (+nww)\narch={arch}  feat={feat}")
    ax.set_xlim(0, min(max(values) * 1.15 if values else 1.0, 1.0))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "loss_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


def _plot_training_curves():
    loss_dirs = sorted(OUT_BASE.glob("loss_*"))
    if not loss_dirs:
        return
    ncols = 3
    nrows = (len(loss_dirs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
    for idx, d in enumerate(loss_dirs):
        ax = axes[idx // ncols][idx % ncols]
        loss_name = d.name.replace("loss_", "")
        metrics_csv = d / "metrics_log.csv"
        if not metrics_csv.exists():
            ax.set_title(f"{loss_name}\n(no data)")
            continue
        rows = list(csv_mod.DictReader(open(metrics_csv)))
        if not rows:
            continue
        epochs = [int(r["epoch"]) for r in rows]
        f1s    = [float(r["f1"]) for r in rows]
        ax2 = ax.twinx()
        ax.plot(epochs, f1s, "b-o", ms=3, lw=1.5, label="F1")
        if "loss" in rows[0]:
            losses = [float(r["loss"]) for r in rows]
            ax2.plot(epochs, losses, "r--", ms=2, lw=1, alpha=0.6)
            ax2.set_ylabel("Train loss", color="red", fontsize=7)
            ax2.tick_params(axis="y", labelcolor="red", labelsize=7)
        ax.set_title(f"{loss_name}  (best={max(f1s):.4f})", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("F1", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    for idx in range(len(loss_dirs), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle("Training curves per loss — parallel run", fontsize=11)
    fig.tight_layout()
    path = PLOTS_DIR / "training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-parallel", type=int,
                        default=env_default("WW_MAX_PARALLEL", 2, int),
                        help="Max simultaneous training workers (default 2)")
    parser.add_argument("--best-cfg", type=str,
                        default=env_default("WW_BEST_CFG", None),
                        help="JSON string of best config from genetic search")
    args = parser.parse_args()

    # Default best config — will be overridden if genetic search results exist
    best_cfg = {"featurizer_type": "mfcc", "arch": "gru", "hidden_dim": 128,
                "dropout": 0.1, "n_features": 40, "lr": 5e-4, "batch_size": 16}
    if args.best_cfg:
        import json
        best_cfg.update(json.loads(args.best_cfg))

    # Try to load best config from genetic search results
    sweep_json = BASE / "genetic_nww" / "stage2" / "best_result.json"
    if not sweep_json.exists():
        sweep_json = BASE / "genetic_nww" / "stage1" / "best_result.json"
    if sweep_json.exists():
        import json
        saved = json.loads(sweep_json.read_text())
        best_cfg.update(saved.get("best_config", {}))
        logger.info("Loaded best config from genetic search: %s", best_cfg)
    else:
        logger.info("No genetic search results found — using default config: %s", best_cfg)

    train_data, test_data = _build_dataset()

    # Check augmentation cache
    AUG_DIR   = DATASET_DIR / "augmentation"
    augment_opts = {}
    if any((AUG_DIR / "bg_noise").rglob("*.wav")):
        augment_opts["bg_noise_folder"] = str(AUG_DIR / "bg_noise")
    if any((AUG_DIR / "music").rglob("*.wav")):
        augment_opts["music_folder"] = str(AUG_DIR / "music")
    if any((AUG_DIR / "rir").rglob("*.wav")):
        augment_opts["rir_folder"] = str(AUG_DIR / "rir")
    wow_dir = DATASET_DIR / "negatives" / "not_wake_word_subset"
    if wow_dir.exists():
        augment_opts["wake_word_over_speech_folder"] = str(wow_dir)
    logger.info("Augmentation opts: %s", list(augment_opts.keys()) or "none")

    # ── Pre-load ALL audio into shared memory (done once in main process) ─────
    all_paths = [p for p, _ in train_data + test_data]
    from ww_trainer.cache import SharedWaveformCache
    shared_cache = SharedWaveformCache(sr=16000, max_duration=2.0)
    shared_cache.preload(all_paths, n_workers=4, show_progress=True)
    logger.info("Cache stats: %s", shared_cache.stats())

    # ── Spawn workers in batches of max_parallel ──────────────────────────────
    mp.set_start_method("fork", force=True)   # fork shares shared_memory pages
    result_queue: mp.Queue = mp.Queue()
    results = []
    pending = list(LOSSES)

    while pending or results is not None:
        batch = pending[:args.max_parallel]
        pending = pending[args.max_parallel:]
        if not batch:
            break

        logger.info("Starting batch: %s", batch)
        procs = []
        for loss_name in batch:
            p = mp.Process(
                target=_train_worker,
                args=(loss_name, train_data, test_data, shared_cache,
                      best_cfg, augment_opts, result_queue),
                name=f"worker-{loss_name}",
            )
            p.start()
            procs.append(p)

        # Collect results for this batch
        for _ in batch:
            loss_name, f1, elapsed, skipped = result_queue.get(timeout=7200)
            results.append({"loss": loss_name, "f1": f1,
                            "elapsed_s": elapsed, "skipped": skipped})
            logger.info("Collected %s → F1=%.4f", loss_name, f1)

        for p in procs:
            p.join()

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Loss Comparison — hey mycroft (+nww, parallel run)")
    print(f"  arch={best_cfg.get('arch')}  feat={best_cfg.get('featurizer_type')}  "
          f"hidden={best_cfg.get('hidden_dim')}")
    print("═" * 60)
    print(f"  {'Loss':<22} {'Best F1':>8}  {'Note'}")
    print("  " + "-" * 55)
    for r in sorted(results, key=lambda x: x["f1"], reverse=True):
        note = "reused" if r.get("skipped") else f"{r.get('elapsed_s', 0):.0f}s"
        print(f"  {r['loss']:<22} {r['f1']:>8.4f}  {note}")
    print("═" * 60)

    _plot_loss_bars(results, best_cfg)
    _plot_training_curves()
    logger.info("All done. Models: %s  Plots: %s", OUT_BASE.resolve(), PLOTS_DIR.resolve())


if __name__ == "__main__":
    main()
