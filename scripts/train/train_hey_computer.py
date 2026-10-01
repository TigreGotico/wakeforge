"""
Infinite hard-negative training for "hey computer".

Positives  — synth_output/en/hey_computer/  (OVOS TTS)
           — vc_output/hey_computer/         (voice-cloned)
NWW pool   — hf_datasets/not_wake_word_subset/    (10 000 clips, mined each epoch)

Runtime augmentations (no VC):
  - spec_augment, mixup
  - wake_word_over_speech (WoW) using NWW clips as background

Usage
-----
    .venv/bin/python train_hey_computer.py
    .venv/bin/python train_hey_computer.py --arch gru --loss rppl
"""
import argparse
import logging
import os
import random
import sys
from pathlib import Path

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
logger = logging.getLogger("train_hey_computer")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--arch",            default=env_default("WW_ARCH",       "bcresnet"),
                    choices=["ffn", "gru", "cnn", "bcresnet", "tcresnet", "dscnn"])
parser.add_argument("--hidden-dim",      type=int,   default=env_default("WW_HIDDEN_DIM", 128,   int))
parser.add_argument("--tau",             type=float, default=env_default("WW_TAU",        3.0,   float),
                    help="BCResNet width multiplier (1=6K, 1.5=12K, 2=20K, 3=43K, 6=160K, 8=280K)")
parser.add_argument("--loss",            default=env_default("WW_LOSS",       "bce"),
                    choices=["bce", "focal", "label_smoothing_bce", "rppl"])
parser.add_argument("--lr",              type=float, default=env_default("WW_LR",         5e-4,  float))
parser.add_argument("--batch-size",      type=int,   default=env_default("WW_BATCH_SIZE", 16,    int))
parser.add_argument("--target-f1",       type=float, default=0.92)
parser.add_argument("--target-eer",      type=float, default=0.08)
parser.add_argument("--min-epochs",      type=int,   default=20)
parser.add_argument("--max-epochs",      type=int,   default=None)
parser.add_argument("--patience-after",  type=int,   default=10)
parser.add_argument("--hard-plateau",    type=int,   default=20)
parser.add_argument("--scan-size",       type=int,   default=5000)
parser.add_argument("--neg-threshold",   type=float, default=0.5)
parser.add_argument("--neg-multiplier",  type=float, default=3.0)
parser.add_argument("--aug-prob",        type=float, default=0.7)
args = parser.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE      = Path("experiments/hey_computer")
POS_DIR   = BASE / "dataset" / "positives"
NWW_DIR   = BASE / "dataset" / "negatives" / "not_wake_word_subset"
OUT_DIR   = BASE / "models" / f"{args.arch}_tau{args.tau}_{args.loss}_infinite"

if not POS_DIR.exists():
    sys.exit(f"ERROR: positives not found at {POS_DIR} — run rsync from hdd4 first")

# ── Build dataset ──────────────────────────────────────────────────────────────

positives = sorted(p for p in POS_DIR.glob("*.wav") if not p.name.endswith("_tts.wav"))
if not positives:
    sys.exit(f"ERROR: no positives found in {POS_DIR}")

negatives = sorted(NWW_DIR.glob("not_wake_word_*.wav"))
if not negatives:
    sys.exit(f"ERROR: no NWW clips found in {NWW_DIR}")

rng = random.Random(42)
rng.shuffle(positives)

pos_split = int(len(positives) * 0.8)
neg_split = int(len(negatives) * 0.8)

wakes     = [(str(p), "1") for p in positives[:pos_split]]
nww_pool  = [(str(p), "0") for p in negatives[:neg_split]]
test_data = (
    [(str(p), "1") for p in positives[pos_split:]]
    + [(str(p), "0") for p in negatives[neg_split:]]
)

logger.info("Positives : %d", len(positives))
logger.info("NWW pool  : %d  |  Test: %d", len(nww_pool), len(test_data))

# ── Augmentation ───────────────────────────────────────────────────────────────

augment_opts = {
    # WoW: overlay the wake word audio over NWW speech clips as background
    "wake_word_over_speech_folder": str(NWW_DIR),
}
logger.info("Augmentation: spec_augment, mixup, wake_word_over_speech (WoW)")

# ── Trainer ────────────────────────────────────────────────────────────────────

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
    featurizer_type="filterbank",   # log-mel spectrogram — matches BC-ResNet paper
    n_mels=40,
    tau=args.tau,                   # BC-ResNet width multiplier
    export_onnx=True,
    seed=42,
    wake_word="hey_computer",
    mlflow_uri=os.environ.get("MLFLOW_TRACKING_URI"),
    **augment_opts,
)

goal = StoppingGoal(
    target_f1      = args.target_f1,
    target_eer     = args.target_eer,
    patience_after = args.patience_after,
    hard_plateau   = args.hard_plateau,
    min_epochs     = args.min_epochs,
    max_epochs     = args.max_epochs,
)

best_f1 = infinite_training_loop(
    trainer        = trainer,
    output_dir     = OUT_DIR,
    wakes          = wakes,
    nww_pool       = nww_pool,
    test_data      = test_data,
    goal           = goal,
    scan_size      = args.scan_size,
    neg_threshold  = args.neg_threshold,
    neg_multiplier = args.neg_multiplier,
    lr             = args.lr,
    batch_size     = args.batch_size,
    aug_prob       = args.aug_prob,
    use_mixup      = True,
    mixup_alpha    = 1.0,
    spec_augment   = True,
    neg_weight_schedule = "linear",
    vc_per_epoch   = 0,
    resume_cache   = True,
)

print(f"\nBest F1 : {best_f1:.4f}")
print(f"Model   : {OUT_DIR}/best_f1.pt")
print(f"ONNX    : {OUT_DIR}/best_f1.onnx")
