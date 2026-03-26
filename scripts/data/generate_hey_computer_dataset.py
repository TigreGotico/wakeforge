"""
Generate training data for the "hey computer" wake word.

Saves dataset to /mnt/hdd4/ww_datasets/hey_computer/.
Uses existing vc_output/ on hdd4 as voice-conversion reference voices.
Adversarial hard-negatives are generated to help distinguish from
phonetically similar phrases.

Usage
-----
    .venv/bin/python generate_hey_computer_dataset.py
    .venv/bin/python generate_hey_computer_dataset.py 2>&1 | tee /mnt/hdd4/ww_datasets/hey_computer_datagen.log
"""
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")

import torch
torch.set_num_threads(12)
torch.set_num_interop_threads(4)

import psutil
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resource guards
# ---------------------------------------------------------------------------

HDD4_ROOT = Path("/mnt/hdd4/ww_datasets")
OUTPUT_DIR = HDD4_ROOT / "hey_computer"
VC_REFS_DIR = HDD4_ROOT / "vc_output"

if not HDD4_ROOT.exists():
    logger.error("/mnt/hdd4/ww_datasets is not accessible — is hdd4 mounted?")
    sys.exit(1)

ram_gb = psutil.virtual_memory().available / 1e9
if ram_gb < 10:
    logger.error("Less than 10 GB RAM available (%.1f GB) — aborting.", ram_gb)
    sys.exit(1)

disk = shutil.disk_usage(HDD4_ROOT)
disk_free_gb = disk.free / 1e9
if disk_free_gb < 50:
    logger.error("Less than 50 GB free on hdd4 (%.1f GB) — aborting.", disk_free_gb)
    sys.exit(1)

logger.info("hdd4 accessible — %.1f GB RAM available, %.1f GB disk free", ram_gb, disk_free_gb)

# ---------------------------------------------------------------------------
# Plugin availability check
# ---------------------------------------------------------------------------

_REQUIRED_PLUGINS = {
    "ovos_vad_plugin_silero": "ovos-vad-plugin-silero",
    "ovos_tts_plugin_edge_tts": "ovos-tts-plugin-edge-tts",
}
_OPTIONAL_PLUGINS = {
    "ovos_tts_plugin_google_tx": "ovos-tts-plugin-google-tx",
    "ovos_tts_plugin_piper": "ovos-tts-plugin-piper",
}

for module, pkg in _REQUIRED_PLUGINS.items():
    try:
        __import__(module)
    except ImportError:
        logger.error("Required OVOS plugin missing: %s  →  pip install %s", module, pkg)
        sys.exit(1)

for module, pkg in _OPTIONAL_PLUGINS.items():
    try:
        __import__(module)
        logger.info("Optional plugin available: %s", pkg)
    except ImportError:
        logger.warning("Optional plugin not installed: %s  →  pip install %s", module, pkg)

# ---------------------------------------------------------------------------
# Run datagen pipeline
# ---------------------------------------------------------------------------

from ww_trainer.datagen import DatagenConfig, run_datagen_pipeline

cfg = DatagenConfig(
    wake_word="hey computer",
    output_dir=OUTPUT_DIR,
    n_positive=1500,
    max_negative=10000,
    lang="en",
    sample_rate=16000,
    vad_trim=True,
    test_split=0.2,
    vc_refs_dir=str(VC_REFS_DIR) if VC_REFS_DIR.exists() else None,
    vc_device="cpu",
    adversarial=True,
    adversarial_n=30,
    llm_url=None,
    llm_model="gemma3:4b",
    download_augmentation=True,
    pre_augment=False,
    seed=42,
)

logger.info("Starting datagen pipeline for '%s'", cfg.wake_word)
logger.info("Output: %s", OUTPUT_DIR)
if cfg.vc_refs_dir:
    logger.info("VC references: %s", cfg.vc_refs_dir)
else:
    logger.warning("vc_output/ not found on hdd4 — voice conversion will be skipped")

result = run_datagen_pipeline(cfg)

# ---------------------------------------------------------------------------
# Next steps
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Next step — train a model:")
print("  " + result.suggested_train_command(cfg.wake_word, tier="small"))
print("=" * 60)
