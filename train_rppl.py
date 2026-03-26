"""RPPL-loss wake-word experiment with rich embedding visualisation.

RPPL (Representation-Pair Purity Loss) trains the model to produce
*consistent* embeddings for the same word under different acoustic
conditions — positive pairs should be close, negative pairs far.
This script trains a single run and logs PCA + t-SNE projections to
MLflow every ``--viz-every`` epochs so you can watch the embedding
space organise itself over time.

What you'll see in MLflow
-------------------------
Artifacts  viz/pca/pca_epoch_NNNN.png
           viz/tsne/tsne_epoch_NNNN.png
  Each image is a 3-panel figure:
    1. Class labels  — wake (green ▲), easy neg (blue •), hard neg (red ×)
       Hard negatives are NWW clips the model currently confuses for wake.
    2. Confidence heatmap — continuous P(wake) colour per point.
       A well-trained RPPL model → wake cluster glows red, NWW cluster green.
    3. Confidence histogram — how separable the two classes are.
       A good model → two sharp peaks near 0 and 1.

Scalar metrics logged per epoch
---------------------------------
  embed_centroid_dist   Euclidean distance between wake / nonwake centroids
  embed_fisher_ratio    centroid_dist / mean intra-class spread  (higher = better)
  embed_silhouette      Cosine silhouette score  [-1, 1]  (higher = better)
  embed_pca_var_explained  % variance captured by first 2 PCs
  readiness             Linear separability of embeddings  (0–1)
  train_loss, f1, eer, auc, far, frr …

Usage
-----
    .venv/bin/python train_rppl.py
    .venv/bin/python train_rppl.py --epochs 60 --arch bcresnet --viz-every 3
    .venv/bin/python train_rppl.py --loss bce          # compare baseline
    .venv/bin/python train_rppl.py --loss rppl --hidden-dim 256 --lr 2e-4
"""
import argparse
import logging
import os
import sys
from pathlib import Path

from ww_trainer.env import load_env
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
logger = logging.getLogger("train_rppl")


def _e(key, default):
    val = os.environ.get(key)
    if val is None:
        return default
    if default is None:
        return val
    try:
        return type(default)(val)
    except (ValueError, TypeError):
        return default


# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="RPPL wake-word experiment with embedding viz",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--epochs",      type=int,   default=_e("WW_EPOCHS", 50))
parser.add_argument("--arch",        default=_e("WW_ARCH", "gru"),
                    choices=["ffn", "gru", "cnn", "bcresnet", "tcresnet", "dscnn"])
parser.add_argument("--hidden-dim",  type=int,   default=_e("WW_HIDDEN_DIM", 128))
parser.add_argument("--lr",          type=float, default=_e("WW_LR", 5e-4))
parser.add_argument("--batch-size",  type=int,   default=_e("WW_BATCH_SIZE", 16))
parser.add_argument("--loss",        default=_e("WW_LOSS", "rppl"),
                    choices=["bce", "focal", "label_smoothing_bce", "rppl"],
                    help="Loss function. Default rppl — pass --loss bce to compare baseline.")
parser.add_argument("--wake-word",   default=_e("WW_WAKE_WORD", "hey_mycroft"))

# Embedding viz schedule
parser.add_argument("--viz-every",   type=int, default=_e("WW_VIZ_EVERY", 5),
                    help="Log PCA + t-SNE to MLflow every N epochs (0 = end only)")
parser.add_argument("--viz-samples", type=int, default=_e("WW_VIZ_SAMPLES", 300),
                    help="Points per projection (balanced wake/nonwake)")

# Training knobs
parser.add_argument("--patience",    type=int,   default=_e("WW_PATIENCE", 10))
parser.add_argument("--mine-frac",   type=float, default=_e("WW_MINE_FRAC", 0.33))
parser.add_argument("--aug-prob",    type=float, default=_e("WW_AUG_PROB", 0.7))
parser.add_argument("--no-mixup",    action="store_true")
parser.add_argument("--no-spec-aug", action="store_true")

args = parser.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE        = Path(f"experiments/{args.wake_word}")
DATASET_DIR = BASE / "dataset"
TRAIN_CSV   = DATASET_DIR / "train" / "metadata.csv"
TEST_CSV    = DATASET_DIR / "test"  / "metadata.csv"
AUG_DIR     = DATASET_DIR / "augmentation"
NWW_DIR     = DATASET_DIR / "negatives" / "not_wake_word_subset"
OUT_DIR     = BASE / "models" / f"{args.arch}_{args.loss}_viz"

if not TRAIN_CSV.exists():
    sys.exit(f"Dataset not found at {TRAIN_CSV}. Run train_full_nww.py first.")

# ── Dataset ────────────────────────────────────────────────────────────────
train_data = read_dataset_csv(TRAIN_CSV)
test_data  = read_dataset_csv(TEST_CSV)

if NWW_DIR.exists():
    import random
    nww_wavs = sorted(NWW_DIR.glob("not_wake_word_*.wav"))
    rng = random.Random(42)
    rng.shuffle(nww_wavs)
    split = int(len(nww_wavs) * 0.8)
    train_data += [(str(p), "0") for p in nww_wavs[:split]]
    test_data  += [(str(p), "0") for p in nww_wavs[split:]]
    logger.info("NWW: %d files", len(nww_wavs))

pos = sum(1 for _, l in train_data if l == "1")
neg = sum(1 for _, l in train_data if l == "0")
logger.info("Train: %d pos / %d neg (%.1f:1) | Test: %d",
            pos, neg, neg / max(1, pos), len(test_data))

# ── Augmentation ───────────────────────────────────────────────────────────
augment_opts = {}
if any((AUG_DIR / "bg_noise").rglob("*.wav")):
    augment_opts["bg_noise_folder"] = str(AUG_DIR / "bg_noise")
if any((AUG_DIR / "music").rglob("*.wav")):
    augment_opts["music_folder"] = str(AUG_DIR / "music")
if any((AUG_DIR / "rir").rglob("*.wav")):
    augment_opts["rir_folder"] = str(AUG_DIR / "rir")
if NWW_DIR.exists():
    augment_opts["wake_word_over_speech_folder"] = str(NWW_DIR)
logger.info("Augmentation: %s", list(augment_opts.keys()) or "none")

# ── Trainer ────────────────────────────────────────────────────────────────
from ww_trainer.trainer import WakeWordTrainer

OUT_DIR.mkdir(parents=True, exist_ok=True)

trainer = WakeWordTrainer(
    arch=args.arch,
    featurizer="",
    feature_dim=None,
    hidden_dim=args.hidden_dim,
    dropout=0.1,
    device="cpu",
    losses_cfg=[{"name": args.loss, "weight": 1.0}],
    featurizer_type="mfcc",
    n_mfcc=40,
    export_onnx=True,
    seed=42,
    wake_word=args.wake_word,
    mlflow_uri=os.environ.get("MLFLOW_TRACKING_URI"),
    **augment_opts,
)

# Log experiment intent as MLflow tags so it's easy to filter runs
if trainer.mlflow:
    try:
        trainer.mlflow.set_tags({
            "experiment": "rppl_embedding_viz",
            "loss":        args.loss,
            "arch":        args.arch,
            "viz_every":   args.viz_every,
        })
    except Exception:
        pass

logger.info("Loss: %s | Arch: %s | Hidden: %d | Epochs: %d | Viz every: %d",
            args.loss, args.arch, args.hidden_dim, args.epochs, args.viz_every)

# ── Train ──────────────────────────────────────────────────────────────────
viz_every = args.viz_every if args.viz_every > 0 else args.epochs  # 0 → end only

best_f1 = trainer.train(
    output_dir=OUT_DIR,
    train_data=train_data,
    test_data=test_data,
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    mine_fraction=args.mine_frac,
    patience=args.patience,
    use_mixup=not args.no_mixup,
    mixup_alpha=1.0,
    spec_augment=not args.no_spec_aug,
    neg_weight_schedule="linear",
    aug_prob=args.aug_prob,
    # Embedding viz — every N epochs
    pca_every=viz_every,
    tsne_every=viz_every,
    umap_every=0,           # UMAP is slow on CPU; enable with --viz-every if desired
)

# ── Final viz pass ─────────────────────────────────────────────────────────
# Generate projections for the final trained model regardless of schedule
from ww_trainer.visualization import log_pca, log_tsne
from ww_trainer.utils import read_dataset_csv

logger.info("Generating final embedding projections …")
log_pca(
    trainer.model, test_data,
    OUT_DIR / "viz" / "pca",
    epoch=args.epochs,
    device=trainer.device,
    sample_size=args.viz_samples,
    mlflow=trainer.mlflow,
)
log_tsne(
    trainer.model, test_data,
    OUT_DIR / "viz" / "tsne",
    epoch=args.epochs,
    device=trainer.device,
    sample_size=args.viz_samples,
    mlflow=trainer.mlflow,
)

print(f"\nBest F1 : {best_f1:.4f}")
print(f"Model   : {OUT_DIR}/best_f1.pt")
print(f"ONNX    : {OUT_DIR}/best_f1.onnx")
print(f"Viz     : {OUT_DIR}/viz/")
if os.environ.get("MLFLOW_TRACKING_URI"):
    print(f"MLflow  : {os.environ['MLFLOW_TRACKING_URI']}")
