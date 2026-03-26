"""
Full multi-architecture training run for "hey mycroft".

Trains 6 safe CPU architectures sequentially with full augmentation:
  micro          MFCC-40  + FFN       ~50K params
  delta_micro    ΔMFCC-13 + FFN       ~55K params
  small          MFCC-40  + GRU       ~200K params
  filterbank     LogMel   + GRU       ~200K params
  gammatone      Gammatone+ GRU       ~200K params
  sincnet        SincNet  + GRU       ~300K params  (learnable front-end)

Steps:
  1. Reuse the existing dataset (already downloaded)
  2. Download augmentation data once (bg_noise, music, RIR) if not cached
  3. Train each arch for 50 epochs with augmentation
  4. Export featurizer + head ONNX for each
  5. Print comparison table

Resource budget (AGENTS.md):
  torch threads = 4, batch_size = 8, num_workers = 0
"""
import argparse, csv, logging, os, sys, time
from pathlib import Path
import torch, numpy as np

from ww_trainer.env import load_env, env_default
load_env()

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

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
logger = logging.getLogger("train_full")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs",        type=int,   default=env_default("WW_EPOCHS",        50,    int))
parser.add_argument("--batch-size",    type=int,   default=env_default("WW_BATCH_SIZE",    8,     int))
parser.add_argument("--lr",            type=float, default=env_default("WW_LR",            5e-4,  float))
parser.add_argument("--mine-fraction", type=float, default=env_default("WW_MINE_FRACTION", 0.25,  float))
parser.add_argument("--patience",      type=int,   default=env_default("WW_PATIENCE",      5,     int))
args = parser.parse_args()

# ── Experiment ───────────────────────────────────────────────────────────────
EXPERIMENT_NAME = env_default("WW_EXPERIMENT_NAME", "hey_mycroft")
DATASET_NAME    = env_default("WW_DATASET_NAME",    "hey_mycroft")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(f"experiments/{EXPERIMENT_NAME}")
DATASET_DIR  = Path(f"experiments/{DATASET_NAME}") / "dataset"
AUG_DIR      = DATASET_DIR / "augmentation"
BG_DIR       = AUG_DIR / "bg_noise"
MUSIC_DIR    = AUG_DIR / "music"
RIR_DIR      = AUG_DIR / "rir"
TRAIN_CSV    = DATASET_DIR / "train" / "metadata.csv"
TEST_CSV     = DATASET_DIR / "test"  / "metadata.csv"

if not TRAIN_CSV.exists():
    sys.exit(f"Dataset not found at {TRAIN_CSV}. Run train_hey_mycroft.py first.")

# ── Step 1: download augmentation data (once) ─────────────────────────────
from ww_trainer.datagen import download_hf_audio_dataset, NEGATIVE_DATASETS

def _download_aug_split(ds_ids, root):
    for ds_id in ds_ids:
        name = ds_id.split("/")[-1]
        download_hf_audio_dataset(ds_id, root / name, sr=16000)

already_have_bg    = any(BG_DIR.rglob("*.wav"))
already_have_music = any(MUSIC_DIR.rglob("*.wav"))
already_have_rir   = any(RIR_DIR.rglob("*.wav"))

if not (already_have_bg and already_have_music and already_have_rir):
    logger.info("Downloading augmentation datasets …")
    if not already_have_bg:
        _download_aug_split(NEGATIVE_DATASETS["bg_noise"], BG_DIR)
    if not already_have_music:
        _download_aug_split(NEGATIVE_DATASETS["music"], MUSIC_DIR)
    if not already_have_rir:
        _download_aug_split(NEGATIVE_DATASETS["rir"], RIR_DIR)
else:
    logger.info("Augmentation data already present — skipping download")

bg_wav_count    = len(list(BG_DIR.rglob("*.wav")))
music_wav_count = len(list(MUSIC_DIR.rglob("*.wav")))
rir_wav_count   = len(list(RIR_DIR.rglob("*.wav")))
logger.info("Augmentation: bg=%d  music=%d  rir=%d", bg_wav_count, music_wav_count, rir_wav_count)

augment_opts = {}
if bg_wav_count:    augment_opts["bg_noise_folder"] = str(BG_DIR)
if music_wav_count: augment_opts["music_folder"]    = str(MUSIC_DIR)
if rir_wav_count:   augment_opts["rir_folder"]      = str(RIR_DIR)

# ── Step 2: read dataset CSVs ─────────────────────────────────────────────
train_data = read_dataset_csv(TRAIN_CSV)
test_data  = read_dataset_csv(TEST_CSV)
logger.info("Dataset: %d train / %d test", len(train_data), len(test_data))

# ── Step 3: architecture definitions ─────────────────────────────────────
ARCHS = [
    # (run_name, tier_name, extra_kwargs)
    ("micro",       "micro",            {}),
    ("delta_micro", "delta_micro",       {}),
    ("small",       "small",            {}),
    ("filterbank",  "filterbank_small", {}),
    ("gammatone",   "gammatone_small",  {}),
    ("sincnet",     "sincnet_small",    {}),
]

# ── Step 4: train each arch ───────────────────────────────────────────────
from ww_trainer.trainer import WakeWordTrainer
from ww_trainer.tiers import get_tier
from ww_trainer.utils import read_dataset_csv

results = []   # list of dicts for comparison table

for run_name, tier_name, extra in ARCHS:
    model_dir = BASE / "models" / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already trained
    if (model_dir / "best_f1.pt").exists():
        logger.info("Skipping %s — best_f1.pt already exists", run_name)
        metrics_csv = model_dir / "metrics_log.csv"
        if metrics_csv.exists():
            rows = list(csv.DictReader(open(metrics_csv)))
            last = rows[-1]
            results.append({
                "arch": run_name, "tier": tier_name,
                "f1": float(last["f1"]), "auc": float(last["auc"]),
                "loss": float(last["loss"]), "skipped": True,
            })
        continue

    tc = get_tier(tier_name)
    model_kwargs = {"hidden_dim": tc.hidden_dim}
    if tc.extractor_type in ("mfcc", "delta_mfcc"):
        model_kwargs["n_mfcc"] = tc.n_mfcc
    if tc.head_arch == "gru":
        model_kwargs["bidirectional"] = tc.bidirectional
        model_kwargs["gru_n_layers"]  = tc.gru_n_layers
    model_kwargs.update(extra)

    logger.info("═══ Training %s (%s) ═══", run_name, tier_name)
    t0 = time.time()

    trainer = WakeWordTrainer(
        arch=tc.head_arch,
        featurizer="",
        feature_dim=None,
        featurizer_type=tc.extractor_type,
        wake_word=EXPERIMENT_NAME,
        device="cpu",
        losses_cfg=[{"name": "bce", "weight": 1.0}],
        export_onnx=True,
        seed=42,
        **model_kwargs,
        **augment_opts,
    )

    best_f1 = trainer.train(
        output_dir=model_dir,
        train_data=train_data,
        test_data=test_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mine_fraction=args.mine_fraction,
        patience=args.patience,
    )

    elapsed = time.time() - t0
    logger.info("Done %s — F1=%.4f in %.1fs", run_name, best_f1, elapsed)

    # Export featurizer ONNX (head already exported by trainer)
    feat_onnx = model_dir / "best_f1_featurizer.onnx"
    if not feat_onnx.exists():
        try:
            trainer.model.load_checkpoint(str(model_dir / "best_f1.pt"))
            trainer.model.eval()
            trainer.model.feature_extractor.export_to_onnx(str(feat_onnx))
            logger.info("Exported featurizer ONNX: %s", feat_onnx)
        except Exception as e:
            logger.warning("Featurizer ONNX export failed for %s: %s", run_name, e)

    results.append({
        "arch": run_name, "tier": tier_name,
        "f1": best_f1, "auc": 0.0, "loss": 0.0,
        "elapsed_s": elapsed, "skipped": False,
    })

    # Free memory before next run
    del trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ── Step 5: comparison table ──────────────────────────────────────────────
print("\n" + "═" * 65)
print("  Architecture Comparison — hey mycroft")
print("═" * 65)
print(f"  {'Arch':<14} {'Tier':<18} {'Best F1':>8}  {'Note'}")
print("  " + "-" * 61)
for r in results:
    note = "reused" if r.get("skipped") else f"{r.get('elapsed_s', 0):.0f}s"
    print(f"  {r['arch']:<14} {r['tier']:<18} {r['f1']:>8.4f}  {note}")
print("═" * 65)
print(f"\n  Models saved under: {(BASE / 'models').resolve()}")
print("  Each model dir contains:")
print("    best_f1.pt              PyTorch checkpoint")
print("    best_f1.onnx            Classifier head ONNX")
print("    best_f1_featurizer.onnx Feature extractor ONNX")
print("    metrics_log.csv         Per-epoch metrics")
print()
print("  To test a model:")
print("    python test_wakeword.py --dir experiments/hey_mycroft/models/small --audio sample.wav")
print("    python test_wakeword.py --dir experiments/hey_mycroft/models/small --mic")
