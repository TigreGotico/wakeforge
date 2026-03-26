"""Infinite training mode for wake-word detection.

Each epoch:
  1. Random subset of the NWW pool is inferred → only hard negatives survive.
  2. (Optional) VC positives are synthesised on-the-fly via chatterbox-onnx.
  3. Model trains on positives + hard negatives.
  4. Repeat until stopping goals are met or a hard plateau is reached.

This gives effectively unlimited training data: the model always trains on
the examples it currently finds difficult.

Usage
-----
    # Basic — keep going until F1 ≥ 0.92 and EER ≤ 0.08
    .venv/bin/python train_infinite.py

    # Custom goals
    .venv/bin/python train_infinite.py --target-f1 0.95 --target-eer 0.05

    # With VC synthesis (5 new positives per epoch)
    .venv/bin/python train_infinite.py --vc-per-epoch 5

    # Larger NWW scan per epoch
    .venv/bin/python train_infinite.py --scan-size 20000

    # Different architecture / loss
    .venv/bin/python train_infinite.py --arch bcresnet --loss focal
"""
import argparse
import logging
import os
import random
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
logger = logging.getLogger("train_infinite")


def _e(key: str, default):
    """Return env var *key* cast to the type of *default*, or *default* if unset."""
    val = os.environ.get(key)
    if val is None:
        return default
    if default is None:
        return val  # can't infer type; return raw string
    try:
        return type(default)(val)
    except (ValueError, TypeError):
        return default


# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Infinite hard-negative training",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Model
parser.add_argument("--arch",        default=_e("WW_ARCH", "gru"),
                    choices=["ffn", "gru", "cnn", "bcresnet", "tcresnet", "dscnn"])
parser.add_argument("--hidden-dim",  type=int,   default=_e("WW_HIDDEN_DIM", 128))
parser.add_argument("--loss",        default=_e("WW_LOSS", "bce"),
                    choices=["bce", "focal", "label_smoothing_bce", "rppl"])
parser.add_argument("--lr",          type=float, default=_e("WW_LR", 5e-4))
parser.add_argument("--batch-size",  type=int,   default=_e("WW_BATCH_SIZE", 16))

# Stopping goals
parser.add_argument("--target-f1",       type=float, default=_e("WW_TARGET_F1", 0.92),
                    help="Stop when F1 ≥ this")
parser.add_argument("--target-eer",      type=float, default=_e("WW_TARGET_EER", 0.08),
                    help="Stop when EER ≤ this")
parser.add_argument("--target-far-frr1", type=float, default=_e("WW_TARGET_FAR_FRR1", None))
parser.add_argument("--patience-after",  type=int,   default=_e("WW_PATIENCE_AFTER", 10),
                    help="Epochs of no improvement after goals are met before stopping")
parser.add_argument("--hard-plateau",    type=int,   default=_e("WW_HARD_PLATEAU", 20),
                    help="Stop if no new hard negatives for this many epochs")
parser.add_argument("--min-epochs",      type=int,   default=_e("WW_MIN_EPOCHS", 20))
parser.add_argument("--max-epochs",      type=int,   default=_e("WW_MAX_EPOCHS", None),
                    help="Hard ceiling on epochs (default: unlimited)")

# Mining
parser.add_argument("--scan-size",      type=int,   default=_e("WW_SCAN_SIZE", 5000),
                    help="NWW files to infer each epoch for hard-neg mining")
parser.add_argument("--neg-threshold",  type=float, default=_e("WW_NEG_THRESHOLD", 0.5),
                    help="Confidence threshold for 'hard' classification")
parser.add_argument("--neg-multiplier", type=float, default=_e("WW_NEG_MULTIPLIER", 3.0),
                    help="Target neg count = pos count × this")

# Voice conversion
parser.add_argument("--vc-per-epoch",    type=int,   default=_e("WW_VC_PER_EPOCH", 0),
                    help="VC positives to generate per epoch via voice conversion (0=disabled)")
parser.add_argument("--vc-backend",      default=_e("WW_VC_BACKEND", "auto"),
                    choices=["auto", "chatterbox-onnx", "chatterbox", "linacodec"],
                    help="VC backend (audio-to-audio voice cloning)")
parser.add_argument("--vc-device",       default=_e("WW_VC_DEVICE", "auto"),
                    help="PyTorch device for torch VC backend")

# Augmentation
parser.add_argument("--aug-prob",        type=float, default=_e("WW_AUG_PROB", 0.7))
parser.add_argument("--no-mixup",        action="store_true")
parser.add_argument("--no-spec-augment", action="store_true")

# Paths / misc
parser.add_argument("--wake-word",  default=_e("WW_WAKE_WORD", "hey_mycroft"))
parser.add_argument("--out-dir",    default=_e("WW_OUT_DIR", None),
                    help="Output directory (default: experiments/<ww>/models/<arch>_<loss>_infinite)")
parser.add_argument("--resume-cache", action="store_true", default=True,
                    help="Resume mining cache from previous run")
parser.add_argument("--no-resume-cache", dest="resume_cache", action="store_false")

args = parser.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE        = Path(f"experiments/{args.wake_word}")
DATASET_DIR = BASE / "dataset"
TRAIN_CSV   = DATASET_DIR / "train" / "metadata.csv"
TEST_CSV    = DATASET_DIR / "test"  / "metadata.csv"
AUG_DIR     = DATASET_DIR / "augmentation"
NWW_DIR     = DATASET_DIR / "negatives" / "not_wake_word_subset"
WOW_DIR     = NWW_DIR  # same clips used as WoW background

OUT_DIR = Path(args.out_dir) if args.out_dir else (
    BASE / "models" / f"{args.arch}_{args.loss}_infinite"
)

if not TRAIN_CSV.exists():
    sys.exit(f"Dataset not found at {TRAIN_CSV}. Run train_full_nww.py first.")

# ── Read dataset ────────────────────────────────────────────────────────────
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

train_data = _read_csv(TRAIN_CSV)
test_data  = _read_csv(TEST_CSV)

# Split positives from negatives — negatives go into the NWW pool
csv_wakes    = [(p, l) for p, l in train_data if l == "1"]
csv_nonwakes = [(p, l) for p, l in train_data if l == "0"]

# Build NWW pool: CSV negatives + any local not_wake_word_*.wav files
nww_pool = list(csv_nonwakes)
if NWW_DIR.exists():
    local_nww = sorted(NWW_DIR.glob("not_wake_word_*.wav"))
    existing_paths = {p for p, _ in nww_pool}
    new_nww = [(str(p), "0") for p in local_nww if str(p) not in existing_paths]
    nww_pool += new_nww
    logger.info("NWW pool: %d from CSV + %d local = %d total",
                len(csv_nonwakes), len(new_nww), len(nww_pool))
else:
    logger.info("NWW pool: %d from CSV only", len(nww_pool))

if not nww_pool:
    sys.exit("ERROR: NWW pool is empty — need at least some negatives")

logger.info("Positives (train): %d | Test: %d pos / %d neg",
            len(csv_wakes),
            sum(1 for _, l in test_data if l == "1"),
            sum(1 for _, l in test_data if l == "0"))

# ── Augmentation ────────────────────────────────────────────────────────────
augment_opts = {}
if any((AUG_DIR / "bg_noise").rglob("*.wav")):
    augment_opts["bg_noise_folder"] = str(AUG_DIR / "bg_noise")
if any((AUG_DIR / "music").rglob("*.wav")):
    augment_opts["music_folder"] = str(AUG_DIR / "music")
if any((AUG_DIR / "rir").rglob("*.wav")):
    augment_opts["rir_folder"] = str(AUG_DIR / "rir")
if WOW_DIR.exists():
    augment_opts["wake_word_over_speech_folder"] = str(WOW_DIR)
logger.info("Augmentation: %s", list(augment_opts.keys()) or "none")

# ── Trainer ─────────────────────────────────────────────────────────────────
from ww_trainer.trainer import WakeWordTrainer
from ww_trainer.infinite_loop import StoppingGoal, infinite_training_loop

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

goal = StoppingGoal(
    target_f1       = args.target_f1,
    target_eer      = args.target_eer,
    target_far_frr1 = args.target_far_frr1,
    patience_after  = args.patience_after,
    hard_plateau    = args.hard_plateau,
    min_epochs      = args.min_epochs,
    max_epochs      = args.max_epochs,
)

# ── Run ─────────────────────────────────────────────────────────────────────
best_f1 = infinite_training_loop(
    trainer=trainer,
    output_dir=OUT_DIR,
    wakes=csv_wakes,
    nww_pool=nww_pool,
    test_data=test_data,
    goal=goal,
    scan_size=args.scan_size,
    neg_threshold=args.neg_threshold,
    neg_multiplier=args.neg_multiplier,
    lr=args.lr,
    batch_size=args.batch_size,
    aug_prob=args.aug_prob,
    use_mixup=not args.no_mixup,
    mixup_alpha=1.0,
    spec_augment=not args.no_spec_augment,
    neg_weight_schedule="linear",
    vc_per_epoch=args.vc_per_epoch,
    vc_backend=args.vc_backend,
    vc_device=args.vc_device,
    resume_cache=args.resume_cache,
)

print(f"\nBest F1: {best_f1:.4f}")
print(f"Model:   {OUT_DIR}/best_f1.pt")
print(f"ONNX:    {OUT_DIR}/best_f1.onnx")
