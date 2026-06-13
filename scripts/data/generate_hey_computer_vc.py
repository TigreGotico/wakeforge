"""
Voice-clone positives for the "hey computer" wake word.

Reads source positives from synth_output/en/hey_computer/ (OVOS TTS files)
and clones each one into NWW donor voices via pure audio-to-audio VC — no text,
no Chatterbox TTS synthesis.

Output is flat UUID-named WAVs under vc_output/hey_computer/, matching
the structure of all other wake words on hdd4.

Run 02_tts_synth.py first if synth_output is empty.

Usage
-----
    .venv/bin/python generate_hey_computer_vc.py --dry-run
    .venv/bin/python generate_hey_computer_vc.py
    .venv/bin/python generate_hey_computer_vc.py 2>&1 | tee /mnt/hdd4/ww_datasets/hey_computer_vc.log
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")

import torch
torch.set_num_threads(12)
torch.set_num_interop_threads(4)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_hey_computer_vc")

# ── Paths (hdd4 convention) ────────────────────────────────────────────────────

HDD4      = Path("/mnt/hdd4/ww_datasets")
NWW_DIR   = HDD4 / "hf_datasets" / "not_wake_word_subset"
SYNTH_DIR = HDD4 / "synth_output" / "en" / "hey_computer"
OUT_DIR   = HDD4 / "vc_output" / "hey_computer"

# ── Sanity checks ──────────────────────────────────────────────────────────────

if not HDD4.exists():
    sys.exit("ERROR: /mnt/hdd4/ww_datasets not accessible — is hdd4 mounted?")

if not NWW_DIR.exists():
    sys.exit(f"ERROR: NWW donor dir not found: {NWW_DIR}")

synth_files = sorted(SYNTH_DIR.glob("*.wav")) if SYNTH_DIR.exists() else []
if not synth_files:
    sys.exit(
        f"ERROR: No source positives in {SYNTH_DIR}\n"
        f"  Run scripts/dataset_generation/02_tts_synth.py first."
    )

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Voice-clone hey_computer positives using NWW donor voices",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--n-donors", type=int, default=500,
                    help="NWW donor clips to use as voice references")
parser.add_argument("--n-sources", type=int, default=50,
                    help="Source positives from synth_output to clone from")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--budget-hours", type=float, default=12.0)
parser.add_argument("--vc-backend", default="auto",
                    choices=["auto", "chatterbox-onnx", "chatterbox"])
parser.add_argument("--vc-device", default="cpu")
parser.add_argument("--skip-existing", action="store_true", default=True)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

random.seed(args.seed)
DEADLINE = time.monotonic() + args.budget_hours * 3600

# ── Build source × donor pairs ─────────────────────────────────────────────────

all_donors = sorted(NWW_DIR.glob("not_wake_word_*.wav"))
donors = random.sample(all_donors, min(args.n_donors, len(all_donors)))
sources = random.sample(synth_files, min(args.n_sources, len(synth_files)))

donors_per_source = max(1, len(donors) // max(1, len(sources)))
pairs: list[tuple[Path, Path]] = [
    (src, don)
    for src in sources
    for don in random.sample(donors, min(donors_per_source, len(donors)))
]

logger.info("Sources : %d files from %s", len(sources), SYNTH_DIR)
logger.info("Donors  : %d clips from %d available", len(donors), len(all_donors))
logger.info("Pairs   : %d (source × donor)", len(pairs))

if args.dry_run:
    print(f"\nOutput: {OUT_DIR}")
    print(f"Would generate {len(pairs)} files via audio VC (no TTS)")
    for src, don in pairs[:5]:
        print(f"  {src.name} → voice of {don.name}")
    if len(pairs) > 5:
        print(f"  ... and {len(pairs)-5} more")
    sys.exit(0)

# ── Load backend ───────────────────────────────────────────────────────────────

from ww_trainer.vc_helpers import load_vc_backend

try:
    vc = load_vc_backend(backend=args.vc_backend, device=args.vc_device)
except RuntimeError as exc:
    sys.exit(str(exc))

logger.info("Backend : %s  sample_rate=%d", vc.name, vc.sample_rate)

# ── Voice cloning ──────────────────────────────────────────────────────────────

generated, failed = 0, 0

for i, (src, donor) in enumerate(pairs):
    if time.monotonic() > DEADLINE:
        logger.warning("Budget exhausted — stopping at %d/%d", i, len(pairs))
        break

    out_path = OUT_DIR / f"{str(uuid4())[4:]}.wav"

    try:
        vc.vc(src, donor, out_path)
        generated += 1
        if (i + 1) % 20 == 0:
            logger.info("[VC] %d/%d (%.0f%%)", i + 1, len(pairs), 100 * (i + 1) / len(pairs))
    except Exception as exc:
        failed += 1
        logger.warning("[VC] Failed %s → %s: %s", src.name, donor.name, exc)

logger.info("Done: %d generated, %d failed", generated, failed)
total = len(list(OUT_DIR.glob("*.wav")))
print(f"\n{generated} new files  ({total} total in {OUT_DIR})")
print(f"Note: output is {vc.sample_rate} Hz — trainer resamples to 16 kHz on load.")
