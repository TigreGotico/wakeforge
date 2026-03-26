"""
Rebuild the hey mycroft dataset with speech negatives added.

The original dataset used only ESC-50 (environmental sounds) + NAR (ambient
noise) as negatives. Models trained on this learn "speech vs silence" rather
than the specific wake word. This script downloads speech negatives and
rebuilds the dataset from scratch, then retrains all architectures.

Run:
  .venv/bin/python rebuild_dataset.py
"""
import logging
import os
import sys
from pathlib import Path

import torch

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

import psutil
vm = psutil.virtual_memory()
if vm.available / 1e9 < 10:
    sys.exit("ERROR: less than 10 GB available — aborting")
if psutil.disk_usage("/").free / 1e9 < 50:
    sys.exit("ERROR: less than 50 GB disk free — aborting")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rebuild_dataset")

OUTPUT_DIR = Path("experiments/hey_mycroft_v2")

logger.info("Rebuilding dataset with speech negatives → %s", OUTPUT_DIR)
logger.info("This will download LibriSpeech demo + SpeechCommands as negatives.")

from ww_trainer.quickstart import QuickstartConfig, train_from_wakeword

cfg = QuickstartConfig(
    wake_word="hey mycroft",
    output_dir=OUTPUT_DIR,
    n_positive=500,
    lang="en",
    vad_trim=False,
    adversarial=False,
    download_augmentation=False,
    vc_refs_dir=None,
    tier="micro",
    epochs=1,            # just build dataset; retrain separately
    batch_size=8,
    lr=5e-4,
    device="cpu",
    export_onnx=False,
    reuse_dataset=False,  # force fresh download
    seed=42,
)

logger.info("Building dataset (this downloads positives + all negative categories)...")
result = train_from_wakeword(
    wake_word=cfg.wake_word,
    output_dir=cfg.output_dir,
    n_positive=cfg.n_positive,
    lang=cfg.lang,
    vad_trim=cfg.vad_trim,
    adversarial=cfg.adversarial,
    download_augmentation=cfg.download_augmentation,
    tier=cfg.tier,
    epochs=cfg.epochs,
    batch_size=cfg.batch_size,
    lr=cfg.lr,
    device=cfg.device,
    export_onnx=cfg.export_onnx,
    reuse_dataset=cfg.reuse_dataset,
    seed=cfg.seed,
)

logger.info("Dataset ready: %s", result.dataset_dir)

# Count negatives by type
neg_dir = Path(result.dataset_dir) / "negatives"
for subdir in sorted(neg_dir.iterdir()):
    if subdir.is_dir():
        n = len(list(subdir.glob("*.wav")))
        logger.info("  negatives/%-35s  %d files", subdir.name + "/", n)

print("\n  Dataset rebuilt. Now run train_full_v2.py to retrain all archs.")
print(f"  Dataset dir: {result.dataset_dir}")
print(f"\n  Negative breakdown:")
for subdir in sorted(neg_dir.iterdir()):
    if subdir.is_dir():
        n = len(list(subdir.glob("*.wav")))
        print(f"    {subdir.name:<35} {n} files")
