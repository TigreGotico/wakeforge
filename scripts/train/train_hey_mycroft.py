"""
Quick single-run training for "hey mycroft" — uses the fully local dataset.

Reads from experiments/hey_mycroft/dataset/ (all data on NVMe, no HDD access).
Falls back to the mounted HDD for NWW if the local copy isn't ready yet.
Logs to MLflow automatically.

Usage
-----
    .venv/bin/python train_hey_mycroft.py
    .venv/bin/python train_hey_mycroft.py --epochs 50 --arch gru
"""
import argparse
import logging
import os
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
logger = logging.getLogger("train_hey_mycroft")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs",     type=int,   default=env_default("WW_EPOCHS",     30,    int))
parser.add_argument("--arch",       type=str,   default=env_default("WW_ARCH",       "gru"),
                    choices=["ffn", "gru", "cnn", "bcresnet", "tcresnet", "dscnn"])
parser.add_argument("--hidden-dim", type=int,   default=env_default("WW_HIDDEN_DIM", 128,   int))
parser.add_argument("--lr",         type=float, default=env_default("WW_LR",         5e-4,  float))
parser.add_argument("--batch-size", type=int,   default=env_default("WW_BATCH_SIZE", 16,    int))
parser.add_argument("--loss",       type=str,   default=env_default("WW_LOSS",       "bce"),
                    choices=["bce", "focal", "label_smoothing_bce", "rppl"])
args = parser.parse_args()

# ── Experiment ─────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = env_default("WW_EXPERIMENT_NAME", "hey_mycroft")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = Path(f"experiments/{EXPERIMENT_NAME}")
DATASET_DIR = BASE / "dataset"
TRAIN_CSV   = DATASET_DIR / "train" / "metadata.csv"
TEST_CSV    = DATASET_DIR / "test"  / "metadata.csv"
OUT_DIR     = BASE / "models" / f"{args.arch}_{args.loss}"
AUG_DIR     = DATASET_DIR / "augmentation"
WOW_DIR     = DATASET_DIR / "negatives" / "not_wake_word_subset"

# NWW: prefer local copy, fall back to HDD
NWW_DIR = next(
    (p for p in [
        DATASET_DIR / "negatives" / "not_wake_word_subset",
        Path("/mnt/hdd4/ww_datasets/hf_datasets/not_wake_word_subset"),
    ] if p.exists()),
    None,
)

if not TRAIN_CSV.exists():
    sys.exit(f"Dataset not found at {TRAIN_CSV}. Run train_full_nww.py first.")

# ── Load dataset ───────────────────────────────────────────────────────────────
import random

train_data = read_dataset_csv(TRAIN_CSV)
test_data  = read_dataset_csv(TEST_CSV)

if NWW_DIR:
    nww_wavs = sorted(NWW_DIR.glob("not_wake_word_*.wav"))
    rng = random.Random(42)
    rng.shuffle(nww_wavs)
    split = int(len(nww_wavs) * 0.8)
    train_data += [(str(p), "0") for p in nww_wavs[:split]]
    test_data  += [(str(p), "0") for p in nww_wavs[split:]]
    logger.info("NWW from: %s (%d files)", NWW_DIR, len(nww_wavs))
else:
    logger.warning("NWW subset not found — training without it")

pos = sum(1 for _, l in train_data if l == "1")
neg = sum(1 for _, l in train_data if l == "0")
logger.info("Dataset: %d train (%d pos / %d neg, ratio %.1f:1) | %d test",
            len(train_data), pos, neg, neg/max(1,pos), len(test_data))

# ── Augmentation ───────────────────────────────────────────────────────────────
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

# ── Train ──────────────────────────────────────────────────────────────────────
from ww_trainer.trainer import WakeWordTrainer
from ww_trainer.utils import read_dataset_csv

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
    wake_word=EXPERIMENT_NAME,
    mlflow_uri=os.environ.get("MLFLOW_TRACKING_URI"),
    **augment_opts,
)

best_f1 = trainer.train(
    output_dir=OUT_DIR,
    train_data=train_data,
    test_data=test_data,
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    mine_fraction=0.33,
    patience=7,
    use_mixup=True,
    mixup_alpha=1.0,
    spec_augment=True,
    neg_weight_schedule="linear",
)

print(f"\nBest F1: {best_f1:.4f}")
print(f"Model:   {OUT_DIR}/best_f1.pt")
print(f"ONNX:    {OUT_DIR}/best_f1.onnx")
