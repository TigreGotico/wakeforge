"""
Generate wake-word positives via voice conversion.

Uses NWW (not-wake-word) clips as voice donor references — each clip captures
a real speaker's vocal identity. We synthesise "hey mycroft" in that speaker's
voice, giving us effectively unlimited diverse positives.

Two generation modes (both can be run together):
  --mode tts   : synthesise text cloned to each donor voice
  --mode vc    : voice-convert an existing positive to each donor voice

Output is 24kHz WAV. The training pipeline resamples to 16kHz on load.

Backend selection (in priority order):
  1. --vc-backend CLI arg
  2. WW_VC_BACKEND env var  (set in .env)
  3. "auto" — GPU chatterbox if available, else chatterbox-onnx (CPU)

All CLI defaults can be overridden via .env:
  WW_VC_MODE, WW_N_DONORS, WW_N_SOURCES, WW_VC_TEXT, WW_VC_EXAGGERATION,
  WW_SEED, WW_BUDGET_HOURS, WW_VC_BACKEND, WW_WAKE_WORD

Usage
-----
    .venv/bin/python generate_vc_positives.py --dry-run
    .venv/bin/python generate_vc_positives.py --mode tts --n-donors 500
    .venv/bin/python generate_vc_positives.py --mode both --vc-backend chatterbox
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

from ww_trainer.env import load_env
load_env()  # populate os.environ from .env before reading defaults below

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_vc_positives")


def _e(key: str, default):
    """Return env var *key* cast to the same type as *default*, or *default*."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return type(default)(val)
    except (ValueError, TypeError):
        return default


# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Generate wake-word positives via TTS/VC",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--mode",         choices=["tts", "vc", "both"],
                    default=_e("WW_VC_MODE", "both"))
parser.add_argument("--n-donors",     type=int,
                    default=_e("WW_N_DONORS", 500),
                    help="NWW donor clips to use (randomly sampled)")
parser.add_argument("--n-sources",    type=int,
                    default=_e("WW_N_SOURCES", 50),
                    help="Existing positives to use as VC sources (--mode vc)")
parser.add_argument("--text",         default=_e("WW_VC_TEXT", "hey mycroft"),
                    help="Text to synthesise in TTS mode")
parser.add_argument("--exaggeration", type=float,
                    default=_e("WW_VC_EXAGGERATION", 0.4),
                    help="Voice expressiveness (0=flat, 1=max). 0.3-0.5 suits wake words.")
parser.add_argument("--seed",         type=int,
                    default=_e("WW_SEED", 42))
parser.add_argument("--budget-hours", type=float,
                    default=_e("WW_BUDGET_HOURS", 12.0))
parser.add_argument("--wake-word",    default=_e("WW_WAKE_WORD", "hey_mycroft"),
                    help="Wake word slug (used to locate dataset paths)")
parser.add_argument("--vc-backend",   default=_e("WW_VC_BACKEND", "auto"),
                    choices=["auto", "chatterbox-onnx", "chatterbox"],
                    help="TTS/VC backend. 'auto' uses GPU chatterbox if available, "
                         "else chatterbox-onnx (CPU).")
parser.add_argument("--vc-device",    default=_e("WW_VC_DEVICE", "auto"),
                    help="PyTorch device for the torch backend (ignored for onnx)")
parser.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip files that already exist in the output directory")
parser.add_argument("--dry-run",      action="store_true",
                    help="Print what would be generated without running inference")
args = parser.parse_args()

random.seed(args.seed)
DEADLINE = time.monotonic() + args.budget_hours * 3600

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE          = Path(f"experiments/{args.wake_word}/dataset")
NWW_DIR       = BASE / "negatives" / "not_wake_word_subset"
POSITIVES_DIR = BASE / "positives"
OUT_TTS       = BASE / "positives" / "vc_nww_tts"
OUT_VC        = BASE / "positives" / "vc_nww_vc"
TRAIN_CSV     = BASE / "train" / "metadata.csv"
TEST_CSV      = BASE / "test"  / "metadata.csv"

OUT_TTS.mkdir(parents=True, exist_ok=True)
OUT_VC.mkdir(parents=True, exist_ok=True)

# ── Donor pool ─────────────────────────────────────────────────────────────────

all_donors = sorted(NWW_DIR.glob("not_wake_word_*.wav"))
if not all_donors:
    sys.exit(f"No NWW clips found at {NWW_DIR}. Run rsync first.")

donors = random.sample(all_donors, min(args.n_donors, len(all_donors)))
logger.info("Donor pool: %d clips from %d available", len(donors), len(all_donors))

# ── Source pool for VC mode ────────────────────────────────────────────────────

existing_positives = (list(POSITIVES_DIR.glob("*.wav"))
                      + list((POSITIVES_DIR / "processed").glob("*.wav")))
if args.mode in ("vc", "both") and not existing_positives:
    sys.exit(f"No existing positives found under {POSITIVES_DIR} for VC mode.")
sources = (random.sample(existing_positives, min(args.n_sources, len(existing_positives)))
           if existing_positives else [])
logger.info("VC sources: %d positives", len(sources))

# ── Dry run ────────────────────────────────────────────────────────────────────

if args.dry_run:
    logger.info("Backend: %s (dry-run, no inference)", args.vc_backend)
    if args.mode in ("tts", "both"):
        print(f"\nTTS mode: would generate {len(donors)} files → {OUT_TTS}")
        for d in donors[:5]:
            print(f"  '{args.text}' voiced as {d.name}")
        if len(donors) > 5:
            print(f"  ... and {len(donors)-5} more")
    if args.mode in ("vc", "both"):
        pairs = [(s, d) for s in sources for d in random.sample(donors, min(3, len(donors)))]
        print(f"\nVC mode: would generate {len(pairs)} files → {OUT_VC}")
        for s, d in pairs[:5]:
            print(f"  {Path(s).name} → voice of {d.name}")
        if len(pairs) > 5:
            print(f"  ... and {len(pairs)-5} more")
    sys.exit(0)

# ── Load backend ───────────────────────────────────────────────────────────────

from ww_trainer.vc_helpers import load_vc_backend

try:
    vc = load_vc_backend(backend=args.vc_backend, device=args.vc_device)
except RuntimeError as exc:
    sys.exit(str(exc))

logger.info("Backend: %s  sample_rate=%d", vc.name, vc.sample_rate)

# ── CSV helpers ────────────────────────────────────────────────────────────────

def _append_to_csvs(new_paths: list[Path], split: float = 0.8) -> None:
    if not new_paths:
        return
    rng = random.Random(args.seed)
    shuffled = list(new_paths)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * split)
    train_paths = shuffled[:split_idx]
    test_paths  = shuffled[split_idx:]

    existing_train = set(TRAIN_CSV.read_text().splitlines()) if TRAIN_CSV.exists() else set()
    existing_test  = set(TEST_CSV.read_text().splitlines())  if TEST_CSV.exists()  else set()

    with open(TRAIN_CSV, "a") as f:
        for p in train_paths:
            line = f"{p},1"
            if line not in existing_train:
                f.write(line + "\n")
    with open(TEST_CSV, "a") as f:
        for p in test_paths:
            line = f"{p},1"
            if line not in existing_test:
                f.write(line + "\n")

    logger.info("CSV update: +%d train / +%d test positives", len(train_paths), len(test_paths))

# ── TTS generation ─────────────────────────────────────────────────────────────

def run_tts(donors: list[Path]) -> list[Path]:
    generated, failed = [], 0
    for i, donor in enumerate(donors):
        if time.monotonic() > DEADLINE:
            logger.warning("Budget exhausted — stopping TTS at %d/%d", i, len(donors))
            break
        out_path = OUT_TTS / f"tts_{donor.stem}.wav"
        if args.skip_existing and out_path.exists():
            generated.append(out_path)
            continue
        try:
            vc.tts(args.text, donor, out_path, exaggeration=args.exaggeration)
            generated.append(out_path)
            if (i + 1) % 20 == 0:
                logger.info("[TTS] %d/%d (%.0f%%) — %s",
                            i + 1, len(donors), 100 * (i + 1) / len(donors), out_path.name)
        except Exception as exc:
            failed += 1
            logger.warning("[TTS] Failed donor %s: %s", donor.name, exc)
    logger.info("[TTS] Done: %d generated, %d failed", len(generated), failed)
    return generated

# ── VC generation ──────────────────────────────────────────────────────────────

def run_vc(sources: list[Path], donors: list[Path]) -> list[Path]:
    generated, failed = [], 0
    donors_per_source = max(1, len(donors) // max(1, len(sources)))
    pairs: list[tuple[Path, Path]] = []
    for src in sources:
        for don in random.sample(donors, min(donors_per_source, len(donors))):
            pairs.append((src, don))
    logger.info("[VC] %d source × donor pairs", len(pairs))

    for i, (src, donor) in enumerate(pairs):
        if time.monotonic() > DEADLINE:
            logger.warning("Budget exhausted — stopping VC at %d/%d", i, len(pairs))
            break
        out_path = OUT_VC / f"vc_{Path(src).stem}__{donor.stem}.wav"
        if args.skip_existing and out_path.exists():
            generated.append(out_path)
            continue
        try:
            vc.vc(src, donor, out_path)
            generated.append(out_path)
            if (i + 1) % 20 == 0:
                logger.info("[VC] %d/%d (%.0f%%)", i + 1, len(pairs), 100 * (i + 1) / len(pairs))
        except Exception as exc:
            failed += 1
            logger.warning("[VC] Failed %s→%s: %s", Path(src).name, donor.name, exc)
    logger.info("[VC] Done: %d generated, %d failed", len(generated), failed)
    return generated

# ── Main ───────────────────────────────────────────────────────────────────────

all_generated: list[Path] = []

if args.mode in ("tts", "both"):
    logger.info("=== TTS: %d donors, text=%r ===", len(donors), args.text)
    all_generated += run_tts(donors)

if args.mode in ("vc", "both"):
    logger.info("=== VC: %d sources ===", len(sources))
    all_generated += run_vc(sources, donors)

logger.info("Total generated: %d files", len(all_generated))

if all_generated:
    _append_to_csvs(all_generated)
    print(f"\nDone. {len(all_generated)} new positives.")
    print(f"  TTS output : {OUT_TTS}")
    print(f"  VC output  : {OUT_VC}")
    print(f"  CSVs updated: {TRAIN_CSV}, {TEST_CSV}")
    print(f"\nNote: output is {vc.sample_rate}Hz — training pipeline resamples to 16kHz on load.")
else:
    print("\nNo new files generated.")
