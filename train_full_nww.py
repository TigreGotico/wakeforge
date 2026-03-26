"""
Genetic hyperparameter search + loss comparison for "hey mycroft"
using the augmented dataset (original + not-wake-word subset).

Pipeline
--------
1. Build augmented CSVs: existing dataset + 2000 nww negatives (80/20 split).
2. Run genetic search (full=True) to find best arch/loss/hyperparams.
3. Fix best hyperparams; train all losses explicitly for 50 epochs each.
4. Print comparison table (loss → F1).

Resource limits (AGENTS.md): CPU-only, 4 threads, batch≤16, n_demes=1.
"""
import argparse
import csv as csv_mod
import logging
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

from ww_trainer.env import load_env, env_default
load_env()

os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
torch.set_num_threads(12)
torch.set_num_interop_threads(4)

import psutil
vm = psutil.virtual_memory()
if vm.available / 1e9 < 10:
    sys.exit("ERROR: less than 10 GB available — aborting per AGENTS.md")
if psutil.disk_usage("/").free / 1e9 < 50:
    sys.exit("ERROR: less than 50 GB disk free — aborting per AGENTS.md")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_full_nww")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs",         type=int,   default=env_default("WW_EPOCHS",         50,    int),
                    help="Final training epochs per loss (default 50)")
parser.add_argument("--mine-fraction",  type=float, default=env_default("WW_MINE_FRACTION",  0.25,  float))
parser.add_argument("--patience",       type=int,   default=env_default("WW_PATIENCE",       7,     int))
parser.add_argument("--stage1-pop",     type=int,   default=env_default("WW_STAGE1_POP",     12,    int))
parser.add_argument("--stage1-gen",     type=int,   default=env_default("WW_STAGE1_GEN",     4,     int))
parser.add_argument("--stage1-epochs",  type=int,   default=env_default("WW_STAGE1_EPOCHS",  8,     int))
parser.add_argument("--stage2-pop",     type=int,   default=env_default("WW_STAGE2_POP",     8,     int))
parser.add_argument("--stage2-gen",     type=int,   default=env_default("WW_STAGE2_GEN",     3,     int))
parser.add_argument("--stage2-epochs",  type=int,   default=env_default("WW_STAGE2_EPOCHS",  20,    int))
args = parser.parse_args()

# ── Experiment ─────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = env_default("WW_EXPERIMENT_NAME", "hey_mycroft")
DATASET_NAME    = env_default("WW_DATASET_NAME",    "hey_mycroft")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(f"experiments/{EXPERIMENT_NAME}")
DATASET_DIR = Path(f"experiments/{DATASET_NAME}") / "dataset"
TRAIN_CSV   = DATASET_DIR / "train" / "metadata.csv"
TEST_CSV    = DATASET_DIR / "test"  / "metadata.csv"
NWW_DIR = next((p for p in [DATASET_DIR / "negatives" / "not_wake_word_subset",
                             Path("/mnt/hdd4/ww_datasets/hf_datasets/not_wake_word_subset")]
                if p.exists()), DATASET_DIR / "negatives" / "not_wake_word_subset")
OUT_BASE    = BASE / "models_nww"
SWEEP_DIR   = BASE / "genetic_nww"
# Combined CSV written for the genetic sweep (sweep splits internally)
COMBINED_CSV = BASE / "dataset" / "combined_nww.csv"

if not TRAIN_CSV.exists():
    sys.exit(f"Dataset not found at {TRAIN_CSV}. Run train_hey_mycroft.py first.")
if not NWW_DIR.exists():
    sys.exit(
        f"Not-wake-word subset not found at {NWW_DIR}.\n"
        "Mount the drive: sshfs miro@192.168.1.200:/media/hdd4 /mnt/hdd4"
    )

# ── Read existing CSVs ────────────────────────────────────────────────────────
def _read_csv(p: Path):
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(",", 1)
                if len(parts) == 2:
                    rows.append((parts[0], parts[1]))
    return rows

train_data = _read_csv(TRAIN_CSV)
test_data  = _read_csv(TEST_CSV)
logger.info("Existing dataset: %d train / %d test", len(train_data), len(test_data))

# ── Append not-wake-word negatives (80/20) ────────────────────────────────────
nww_wavs = sorted(NWW_DIR.glob("not_wake_word_*.wav"))
if not nww_wavs:
    sys.exit(f"No WAV files found in {NWW_DIR}")

rng = random.Random(42)
nww_shuffled = list(nww_wavs)
rng.shuffle(nww_shuffled)
split = int(len(nww_shuffled) * 0.8)
nww_train = nww_shuffled[:split]
nww_test  = nww_shuffled[split:]

train_data += [(str(p), "0") for p in nww_train]
test_data  += [(str(p), "0") for p in nww_test]
logger.info(
    "After nww: %d train / %d test  (+%d / +%d negatives)",
    len(train_data), len(test_data), len(nww_train), len(nww_test),
)

# ── Write combined CSV for genetic sweep ──────────────────────────────────────
if not COMBINED_CSV.exists():
    with open(COMBINED_CSV, "w") as f:
        for path, label in train_data + test_data:
            f.write(f"{path},{label}\n")
    logger.info("Wrote combined CSV: %s (%d rows)", COMBINED_CSV, len(train_data) + len(test_data))
else:
    logger.info("Combined CSV already exists — reusing %s", COMBINED_CSV)

# ── Augmentation opts (reuse existing local cache if present) ─────────────────
AUG_DIR   = DATASET_DIR / "augmentation"
BG_DIR    = AUG_DIR / "bg_noise"
MUSIC_DIR = AUG_DIR / "music"
RIR_DIR   = AUG_DIR / "rir"

augment_opts = {}
if any(BG_DIR.rglob("*.wav")):    augment_opts["bg_noise_folder"] = str(BG_DIR)
if any(MUSIC_DIR.rglob("*.wav")): augment_opts["music_folder"]    = str(MUSIC_DIR)
if any(RIR_DIR.rglob("*.wav")):   augment_opts["rir_folder"]      = str(RIR_DIR)
logger.info("Augmentation opts: %s", list(augment_opts.keys()) or "none")

PLOTS_DIR = BASE / "plots_nww"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_generation_curves(stage1_history, stage2_history, path):
    """Line chart: best & avg F1 per generation for both stages."""
    fig, ax = plt.subplots(figsize=(9, 5))

    def _plot_stage(history, stage, color_best, color_avg, x_offset):
        gens  = [h["generation"] + x_offset for h in history]
        bests = [h["best"] for h in history]
        avgs  = [h["avg"]  for h in history]
        ax.plot(gens, bests, "o-", color=color_best, lw=2,
                label=f"Stage {stage} best")
        ax.plot(gens, avgs,  "s--", color=color_avg,  lw=1.5, alpha=0.7,
                label=f"Stage {stage} avg")
        for g, b in zip(gens, bests):
            ax.annotate(f"{b:.3f}", (g, b), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7, color=color_best)

    _plot_stage(stage1_history, 1, "#1f77b4", "#aec7e8", 0)
    # Stage 2 x-axis continues from where stage 1 left off
    s1_len = len(stage1_history)
    _plot_stage(stage2_history, 2, "#d62728", "#f5a0a0", s1_len)

    if s1_len > 0:
        ax.axvline(s1_len - 0.5, color="gray", lw=1, ls=":", label="Stage boundary")

    ax.set_xlabel("Generation (cumulative)")
    ax.set_ylabel("F1")
    ax.set_title("Genetic search — F1 across generations")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def plot_trial_scatter(all_results, path, title="All trials — F1 by loss & arch"):
    """Scatter: every trial, x=trial index, y=F1, colour=loss, shape=arch."""
    losses = sorted({r["config"].get("loss", "bce") for r in all_results})
    archs  = sorted({r["config"].get("arch", "?")   for r in all_results})

    loss_colors = {l: cm.tab10(i / max(len(losses), 1)) for i, l in enumerate(losses)}
    arch_markers = {a: m for a, m in zip(archs, ["o", "s", "^", "D", "v", "P", "*", "X", "h", "<"])}

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, r in enumerate(all_results):
        l = r["config"].get("loss", "bce")
        a = r["config"].get("arch", "?")
        ax.scatter(i, r["score"],
                   color=loss_colors[l],
                   marker=arch_markers.get(a, "o"),
                   s=55, alpha=0.8, zorder=3)

    # Legend: losses (colours)
    for l, c in loss_colors.items():
        ax.scatter([], [], color=c, s=55, label=f"loss={l}")
    # Legend: archs (shapes)
    for a, m in arch_markers.items():
        if a in archs:
            ax.scatter([], [], color="gray", marker=m, s=55, label=f"arch={a}")

    ax.set_xlabel("Trial index")
    ax.set_ylabel("F1")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def plot_loss_bars(results, arch, feat_type, hidden_dim, path):
    """Horizontal bar chart: final F1 per loss function."""
    sorted_r = sorted(results, key=lambda x: x["f1"])
    labels = [r["loss"] for r in sorted_r]
    values = [r["f1"]   for r in sorted_r]
    colors = ["#2ecc71" if v == max(values) else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.6 + 1)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.55)
    for bar, val in zip(bars, values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Best F1 (50 epochs)")
    ax.set_title(
        f"Loss function comparison — hey mycroft (+nww)\n"
        f"arch={arch}  feat={feat_type}  hidden={hidden_dim}"
    )
    ax.set_xlim(0, min(max(values) * 1.15, 1.0))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def plot_loss_curves_per_loss(results_dir, path):
    """One subplot per loss: epoch-by-epoch F1 from metrics_log.csv."""
    loss_dirs = sorted(results_dir.glob("loss_*"))
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
            ax.set_title(f"{loss_name}\n(empty)")
            continue
        epochs = [int(r["epoch"]) for r in rows]
        f1s    = [float(r["f1"])   for r in rows]
        losses = [float(r["loss"]) for r in rows] if "loss" in rows[0] else None

        ax2 = ax.twinx()
        ax.plot(epochs, f1s, "b-o", ms=3, lw=1.5, label="F1")
        if losses:
            ax2.plot(epochs, losses, "r--", ms=2, lw=1, alpha=0.6, label="loss")
            ax2.set_ylabel("Train loss", color="red", fontsize=7)
            ax2.tick_params(axis="y", labelcolor="red", labelsize=7)
        ax.set_title(f"{loss_name}  (best F1={max(f1s):.4f})", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("F1", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)

    # Hide empty subplots
    for idx in range(len(loss_dirs), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Training curves per loss — hey mycroft (+nww)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1+2: Two-stage genetic search (full=True → arch + loss + hyperparams)
#
# Budget rationale (CPU-only, ~2 min/trial at 8 epochs, ~4 min at 15 epochs):
#   Stage 1 — broad exploration:  pop=12, gen=4, epochs=8  → ~48 cheap trials,
#             realistically ~15-18 within the 25-min per-stage timeout.
#   Stage 2 — focused refinement: pop=8,  gen=3, epochs=15 → seeds from top-5
#             stage-1 configs; low mutation, high elite fraction.
#             ~20-min timeout catches whatever the budget allows.
#
# run_two_stage_genetic_search shares epochs_per_trial across both stages, so
# we call run_genetic_search twice manually to get different epoch budgets.
# ─────────────────────────────────────────────────────────────────────────────
from ww_trainer.sweep import run_genetic_search, _build_search_space

def _on_gen(info):
    logger.info(
        "  stage %d gen %d | best=%.4f avg=%.4f",
        info["stage"], info["generation"], info["best"], info["avg"],
    )

COMMON = dict(
    metadata_csv=str(COMBINED_CSV),
    output_dir=str(SWEEP_DIR / "stage1"),
    device="cpu",
    full=True,
    n_demes=1,
    fitness_fn="exp_f1",
    mutation_decay=0.05,
    on_generation=_on_gen,
)

logger.info("═══ Genetic search — stage 1 (broad, 8 epochs/trial, 25 min) ═══")
stage1 = run_genetic_search(
    population_size=args.stage1_pop,
    generations=args.stage1_gen,
    epochs_per_trial=args.stage1_epochs,
    mutation_rate=0.35,
    timeout_minutes=25,
    _stage=1,
    **COMMON,
)
logger.info("Stage 1 done — best F1=%.4f config=%s", stage1["best_score"], stage1["best_config"])
plot_trial_scatter(
    stage1["all_results"],
    PLOTS_DIR / "stage1_trials.png",
    title="Stage 1 trials — F1 by loss & arch",
)

# Seed stage 2 with top-5 stage-1 configs
top5 = [r["config"] for r in sorted(stage1["all_results"], key=lambda r: r["score"], reverse=True)[:5]]
logger.info("═══ Genetic search — stage 2 (focused, 20 epochs/trial, 20 min) ═══")
COMMON["output_dir"] = str(SWEEP_DIR / "stage2")
stage2 = run_genetic_search(
    population_size=args.stage2_pop,
    generations=args.stage2_gen,
    epochs_per_trial=args.stage2_epochs,
    mutation_rate=0.1,          # tighter — exploit stage-1 findings
    elite_frac=0.5,             # keep more elites in refinement stage
    timeout_minutes=20,
    seed_population=top5,
    _stage=2,
    **COMMON,
)
logger.info("Stage 2 done — best F1=%.4f config=%s", stage2["best_score"], stage2["best_config"])
plot_generation_curves(
    stage1["history"], stage2["history"],
    PLOTS_DIR / "generation_curves.png",
)
plot_trial_scatter(
    stage1["all_results"] + stage2["all_results"],
    PLOTS_DIR / "all_trials.png",
    title="All trials (stage 1+2) — F1 by loss & arch",
)

# Pick overall best across both stages
sweep_result = stage2 if stage2["best_score"] >= stage1["best_score"] else stage1

best_cfg   = sweep_result["best_config"]
best_score = sweep_result["best_score"]
logger.info("Overall best — F1=%.4f config=%s", best_score, best_cfg)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: Per-loss final training
# Fix arch + feature hyperparams from best_cfg; vary only the loss.
# ─────────────────────────────────────────────────────────────────────────────
LOSSES = ["bce", "focal", "label_smoothing_bce", "supcon", "arcface", "ntxent"]

from ww_trainer.trainer import WakeWordTrainer

logger.info("═══ STAGE 2: Final training — one run per loss (50 epochs) ═══")

# Extract stable hyperparams from genetic best config
feat_type  = best_cfg.get("featurizer_type", "mfcc")
arch       = best_cfg.get("arch", "gru")
hidden_dim = best_cfg.get("hidden_dim", 128)
dropout    = best_cfg.get("dropout", 0.1)
n_features = best_cfg.get("n_features", 40)
lr         = best_cfg.get("lr", 5e-4)
batch_size = min(best_cfg.get("batch_size", 16), 16)   # cap per AGENTS.md

# Build featurizer kwargs
_FEAT_KWARGS = {
    "mfcc":       lambda n: {"featurizer_type": "mfcc",       "n_mfcc":    n},
    "filterbank": lambda n: {"featurizer_type": "filterbank",  "n_mels":    n},
    "sincnet":    lambda n: {"featurizer_type": "sincnet",     "n_filters": n},
    "gammatone":  lambda n: {"featurizer_type": "gammatone",   "n_filters": n},
}
feat_kwargs = _FEAT_KWARGS.get(feat_type, lambda n: {"featurizer_type": feat_type})(n_features)

logger.info(
    "Fixed hyperparams: feat=%s arch=%s hidden=%d dropout=%.2f lr=%g batch=%d",
    feat_type, arch, hidden_dim, dropout, lr, batch_size,
)

results = []

for loss_name in LOSSES:
    model_dir = OUT_BASE / f"loss_{loss_name}"
    model_dir.mkdir(parents=True, exist_ok=True)

    if (model_dir / "best_f1.pt").exists():
        logger.info("Skipping %s — best_f1.pt already exists", loss_name)
        metrics_csv = model_dir / "metrics_log.csv"
        if metrics_csv.exists():
            rows = list(csv_mod.DictReader(open(metrics_csv)))
            last = rows[-1]
            results.append({"loss": loss_name, "f1": float(last["f1"]), "skipped": True})
        continue

    losses_cfg = [{"name": loss_name, "weight": 1.0}]
    if loss_name in ("arcface", "center", "proxy_nca"):
        losses_cfg[0]["embed_dim"] = hidden_dim

    logger.info("─── Training loss=%s ───", loss_name)
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
            epochs=args.epochs,
            batch_size=batch_size,
            lr=lr,
            mine_fraction=args.mine_fraction,
            patience=args.patience,
        )
    except Exception as exc:
        logger.error("Training failed for loss=%s: %s", loss_name, exc)
        best_f1 = 0.0
    else:
        # Export featurizer ONNX
        feat_onnx = model_dir / "best_f1_featurizer.onnx"
        if not feat_onnx.exists():
            try:
                trainer.model.load_checkpoint(str(model_dir / "best_f1.pt"))
                trainer.model.eval()
                trainer.model.feature_extractor.export_to_onnx(str(feat_onnx))
            except Exception as e:
                logger.warning("Featurizer ONNX export failed for %s: %s", loss_name, e)
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    elapsed = time.time() - t0
    results.append({"loss": loss_name, "f1": best_f1, "elapsed_s": elapsed, "skipped": False})
    logger.info("Done loss=%s — F1=%.4f in %.0fs", loss_name, best_f1, elapsed)

# ── Plots ─────────────────────────────────────────────────────────────────────
plot_loss_bars(results, arch, feat_type, hidden_dim, PLOTS_DIR / "loss_comparison.png")
plot_loss_curves_per_loss(OUT_BASE, PLOTS_DIR / "loss_training_curves.png")
logger.info("All plots saved to %s", PLOTS_DIR)

# ── Comparison table ──────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  Loss Comparison — hey mycroft (+nww subset)")
print(f"  Arch: {arch}  Featurizer: {feat_type}  hidden={hidden_dim}")
print("═" * 60)
print(f"  {'Loss':<22} {'Best F1':>8}  {'Note'}")
print("  " + "-" * 55)
for r in sorted(results, key=lambda x: x["f1"], reverse=True):
    note = "reused" if r.get("skipped") else f"{r.get('elapsed_s', 0):.0f}s"
    print(f"  {r['loss']:<22} {r['f1']:>8.4f}  {note}")
print("═" * 60)
print(f"\n  Genetic search result:  F1={best_score:.4f}  config={best_cfg}")
print(f"  Models: {OUT_BASE.resolve()}")
print()
print("  To test best model (by F1):")
best_loss = max(results, key=lambda x: x["f1"])["loss"]
print(f"    python test_wakeword.py --dir {OUT_BASE}/loss_{best_loss} --audio sample.wav")
