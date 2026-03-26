"""
Copy a training-ready subset from hdd4 into a local fast-access directory.

Reads from the git-lfs repos cloned by download_hdd4_datasets.py and copies
WAV files into a flat local cache.  The copy is size-capped per dataset so
local disk is never exhausted.

Default local target: ./data/hf_datasets/   (fast SSD next to the project)

Layout after running::

    data/hf_datasets/
        not-wake-words-speech-en/   ← speech negatives
        ESC-50/                     ← general
        NAR/                        ← general
        AudioSet/                   ← general (subset)
        ambient_noises/             ← bg_noise
        building_106_kitchen_3secs/ ← bg_noise
        public_domain_sounds_3secs/ ← bg_noise
        FMA_3secs/                  ← music
        MIT_environmental_impulse_responses/  ← rir

Usage::

    # Copy everything with default per-dataset caps
    .venv/bin/python select_training_subset.py

    # Tighter local budget
    .venv/bin/python select_training_subset.py --max-total-mb 20000

    # Only refresh specific datasets
    .venv/bin/python select_training_subset.py --dataset ESC-50 ambient_noises

    # Show current local status and exit
    .venv/bin/python select_training_subset.py --summary
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("select_subset")

HDD4_BASE  = Path("/mnt/hdd4/ww_datasets/hf_datasets")
LOCAL_BASE = Path("data/hf_datasets")

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

# ── Per-dataset local copy caps (MB) ──────────────────────────────────────────
# These caps control local SSD usage.  hdd4 holds the full datasets.
# Raise any cap freely — the script will copy more files on next run.
DATASET_CAPS_MB: dict[str, int] = {
    "not-wake-words-speech-en":            2000,
    "ESC-50":                               500,
    "NAR":                                  500,
    "AudioSet":                            3000,
    "ambient_noises":                      2000,
    "building_106_kitchen_3secs":           500,
    "public_domain_sounds_3secs":           700,
    "FMA_3secs":                           3000,
    "MIT_environmental_impulse_responses":  300,
}

# Total default local budget (MB) — 0 means no global cap
DEFAULT_MAX_TOTAL_MB = 0


def _check_hdd4() -> None:
    if not HDD4_BASE.parent.parent.exists():
        sys.exit("/mnt/hdd4 is not mounted — run: sshfs miro@192.168.1.200:/media/hdd4 /mnt/hdd4")


def _check_local_disk(required_mb: int) -> None:
    local_root = LOCAL_BASE.parent
    local_root.mkdir(parents=True, exist_ok=True)
    free_mb = shutil.disk_usage(local_root).free / 1e6
    if required_mb and free_mb < required_mb:
        sys.exit(f"Only {free_mb:.0f} MB free locally, need {required_mb} MB")


def _mb(path: Path) -> float:
    """Disk usage of path in MB."""
    total = 0
    if path.is_file():
        return path.stat().st_size / 1e6
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / 1e6


def _collect_audio_files(src: Path) -> list[Path]:
    """Return all audio files under src, shuffled."""
    files = [p for p in src.rglob("*") if p.suffix.lower() in AUDIO_EXTS and p.is_file()]
    random.shuffle(files)
    return files


def copy_dataset(
    name: str,
    cap_mb: int,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """Copy up to *cap_mb* worth of audio files from hdd4 to local.

    Already-present files are not re-copied (checked by name).
    Returns number of files copied.
    """
    src_dir   = HDD4_BASE  / name
    dest_dir  = LOCAL_BASE / name

    if not src_dir.exists():
        logger.warning("[SKIP] %s not found on hdd4 — run download_hdd4_datasets.py first", name)
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Files already present locally
    existing = {f.name for f in dest_dir.rglob("*") if f.is_file()}
    local_mb = _mb(dest_dir)

    if local_mb >= cap_mb and not force:
        logger.info("[OK] %s already has %.0f / %d MB locally", name, local_mb, cap_mb)
        return 0

    remaining_mb = cap_mb - local_mb
    src_files = _collect_audio_files(src_dir)
    copied = 0
    copied_mb = 0.0

    for src_file in src_files:
        if src_file.name in existing:
            continue
        file_mb = src_file.stat().st_size / 1e6
        if copied_mb + file_mb > remaining_mb:
            break
        dest_file = dest_dir / src_file.name
        if dry_run:
            copied_mb += file_mb
            copied += 1
        else:
            try:
                shutil.copy2(src_file, dest_file)
                copied_mb += file_mb
                copied += 1
            except Exception as e:
                logger.warning("  copy failed %s: %s", src_file.name, e)

    action = "would copy" if dry_run else "copied"
    logger.info("[%s] %s: %s %d files (%.0f MB) → total %.0f / %d MB",
                "DRY" if dry_run else "DONE", name, action, copied,
                copied_mb, local_mb + copied_mb, cap_mb)
    return copied


def _summary() -> None:
    print("\n" + "═" * 72)
    print(f"  {'Dataset':<42} {'hdd4':>10}  {'local':>10}  {'cap':>8}")
    print("  " + "-" * 68)
    for name, cap_mb in DATASET_CAPS_MB.items():
        hdd4_mb  = _mb(HDD4_BASE  / name) if (HDD4_BASE  / name).exists() else 0
        local_mb = _mb(LOCAL_BASE / name) if (LOCAL_BASE / name).exists() else 0
        hdd4_str  = f"{hdd4_mb:>8.0f} MB" if hdd4_mb  else "   missing"
        local_str = f"{local_mb:>8.0f} MB" if local_mb else "   empty  "
        print(f"  {name:<42} {hdd4_str}  {local_str}  {cap_mb:>6} MB")
    total_local = _mb(LOCAL_BASE)
    total_hdd4  = sum(_mb(HDD4_BASE / n) for n in DATASET_CAPS_MB if (HDD4_BASE / n).exists())
    print("  " + "-" * 68)
    print(f"  {'TOTAL':<42} {total_hdd4:>8.0f} MB  {total_local:>8.0f} MB")
    print("═" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local-base", type=Path, default=LOCAL_BASE,
                        help=f"Local output directory (default: {LOCAL_BASE})")
    parser.add_argument("--max-total-mb", type=int, default=DEFAULT_MAX_TOTAL_MB,
                        help="Hard cap on total local usage in MB (0 = no cap)")
    parser.add_argument("--dataset", nargs="+", metavar="NAME",
                        help="Only copy these datasets (by local name)")
    parser.add_argument("--cap-mb", type=int, default=None,
                        help="Override per-dataset cap (MB) for all selected datasets")
    parser.add_argument("--force", action="store_true",
                        help="Copy more files even if cap is already met")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be copied without touching disk")
    parser.add_argument("--summary", action="store_true",
                        help="Show hdd4 / local usage summary and exit")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for file selection order (default: 42)")
    args = parser.parse_args()

    global LOCAL_BASE
    LOCAL_BASE = args.local_base

    random.seed(args.seed)

    _check_hdd4()

    if args.summary:
        _summary()
        return

    targets = list(DATASET_CAPS_MB.items())
    if args.dataset:
        names = set(args.dataset)
        targets = [(n, c) for n, c in targets if n in names]
        if not targets:
            sys.exit(f"No matching datasets for: {args.dataset}")

    if args.cap_mb is not None:
        targets = [(n, args.cap_mb) for n, _ in targets]

    # Check free local disk once before starting
    total_cap = sum(c for _, c in targets)
    _check_local_disk(min(total_cap, 5000))  # only sanity-check 5 GB minimum

    total_copied = 0
    total_local_mb = 0.0

    for name, cap_mb in targets:
        if args.max_total_mb:
            current_total = _mb(LOCAL_BASE)
            if current_total >= args.max_total_mb:
                logger.info("Global cap reached (%.0f / %d MB) — stopping",
                            current_total, args.max_total_mb)
                break
            # Shrink per-dataset cap if global cap would be exceeded
            remaining = args.max_total_mb - current_total
            cap_mb = min(cap_mb, int(remaining))
            if cap_mb <= 0:
                continue

        n = copy_dataset(name, cap_mb, dry_run=args.dry_run, force=args.force)
        total_copied += n

    _summary()
    logger.info("Done. %d files %s.", total_copied,
                "would be copied" if args.dry_run else "copied")

    if not args.dry_run:
        print(f"\nPass this to your trainer augmentation args:")
        print(f"  bg_noise_folder  = {LOCAL_BASE / 'ambient_noises'}")
        print(f"  music_folder     = {LOCAL_BASE / 'FMA_3secs'}")
        print(f"  rir_folder       = {LOCAL_BASE / 'MIT_environmental_impulse_responses'}")
        print(f"  (speech negatives and general negatives are in train/test CSV via datagen)")


if __name__ == "__main__":
    main()
