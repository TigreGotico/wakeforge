"""
Download all NotWakeWord collection datasets to /mnt/hdd4 via git-lfs.

Uses `git clone` (with git-lfs) rather than the HuggingFace datasets API to
avoid rate limits and Arrow cache bloat.  Already-present repos are updated
with `git pull` instead of re-cloned.

Target layout on hdd4::

    /mnt/hdd4/ww_datasets/hf_datasets/
        not-wake-words-speech-en/      ← speech negatives (primary)
        ESC-50/                        ← general env sounds
        NAR/                           ← general env sounds
        AudioSet/                      ← general (large, capped)
        ambient_noises/                ← bg_noise augmentation
        building_106_kitchen_3secs/    ← bg_noise augmentation
        public_domain_sounds_3secs/    ← bg_noise augmentation
        FMA_3secs/                     ← music augmentation
        MIT_environmental_impulse_responses/  ← RIR augmentation

Usage::

    .venv/bin/python download_hdd4_datasets.py
    .venv/bin/python download_hdd4_datasets.py --max-size-mb 50000
    .venv/bin/python download_hdd4_datasets.py --dataset ESC-50 NAR
    .venv/bin/python download_hdd4_datasets.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("download_hdd4")

HDD4_BASE = Path("/mnt/hdd4/ww_datasets/hf_datasets")
HF_BASE   = "https://huggingface.co/datasets"

# ── Dataset registry ───────────────────────────────────────────────────────────
# Each entry: (repo_id, local_name, category, size_hint_mb)
# size_hint_mb is approximate — used to skip when --max-size-mb is set.
# AudioSet is listed last and flagged as "large" (partial clone via LFS).
DATASETS = [
    # Speech negatives — most critical for wake-word quality
    ("TigreGotico/not-wake-words-speech-en",          "not-wake-words-speech-en",          "speech",   800),
    # General environmental sound negatives
    ("TigreGotico/ESC-50",                             "ESC-50",                            "general",  600),
    ("TigreGotico/NAR",                                "NAR",                               "general",  400),
    # Background noise augmentation
    ("TigreGotico/ambient_noises",                     "ambient_noises",                    "bg_noise", 1500),
    ("TigreGotico/building_106_kitchen_3secs",         "building_106_kitchen_3secs",        "bg_noise", 500),
    ("TigreGotico/public_domain_sounds_3secs",         "public_domain_sounds_3secs",        "bg_noise", 700),
    # Music augmentation
    ("TigreGotico/FMA_3secs",                          "FMA_3secs",                         "music",    3000),
    # Room impulse responses
    ("davidscripka/MIT_environmental_impulse_responses", "MIT_environmental_impulse_responses", "rir",  200),
    # Large general audio (partial clone, see MAX_AUDIOSET_MB)
    ("agkphysics/AudioSet",                            "AudioSet",                          "general",  50000),
]

# AudioSet is ~2 TB total; we hard-limit it so the drive doesn't fill up.
MAX_AUDIOSET_MB = 20_000   # 20 GB — adjust freely


def _check_git_lfs() -> None:
    if not shutil.which("git"):
        sys.exit("git not found — install git")
    try:
        subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit("git-lfs not found — install with: sudo pacman -S git-lfs  OR  sudo apt install git-lfs")


def _check_hdd4() -> None:
    if not HDD4_BASE.parent.parent.exists():
        sys.exit(f"/mnt/hdd4 is not mounted — run: sshfs miro@192.168.1.200:/media/hdd4 /mnt/hdd4")
    HDD4_BASE.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(HDD4_BASE).free / 1e9
    logger.info("hdd4 free: %.0f GB", free_gb)
    if free_gb < 10:
        sys.exit(f"Only {free_gb:.1f} GB free on hdd4 — aborting")


def _disk_used_mb(path: Path) -> float:
    """Return disk usage of *path* in MB (via du -sm)."""
    try:
        out = subprocess.check_output(["du", "-sm", str(path)], text=True)
        return float(out.split()[0])
    except Exception:
        return 0.0


def _clone_or_update(
    repo_id: str,
    dest: Path,
    max_size_mb: int | None,
    dry_run: bool,
) -> bool:
    """Clone (or pull) a HF dataset repo into *dest*.

    For AudioSet, sets GIT_LFS_SKIP_SMUDGE and fetches LFS objects in batches
    until *max_size_mb* is reached.

    Returns True on success.
    """
    url = f"{HF_BASE}/{repo_id}"
    is_audioset = repo_id == "agkphysics/AudioSet"

    # ── Already present ────────────────────────────────────────────────────────
    if (dest / ".git").exists():
        used = _disk_used_mb(dest)
        if max_size_mb and used >= max_size_mb:
            logger.info("[SKIP] %s already at %.0f MB (limit %d MB)", dest.name, used, max_size_mb)
            return True
        logger.info("[UPDATE] %s (%.0f MB on disk)", dest.name, used)
        if dry_run:
            logger.info("  dry-run: would run git pull")
            return True
        try:
            subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"],
                           check=True, timeout=600)
            if not is_audioset:
                subprocess.run(["git", "-C", str(dest), "lfs", "pull"],
                               check=True, timeout=3600)
            return True
        except subprocess.CalledProcessError as e:
            logger.warning("[FAIL] update %s: %s", dest.name, e)
            return False

    # ── Fresh clone ────────────────────────────────────────────────────────────
    logger.info("[CLONE] %s → %s", repo_id, dest)
    if dry_run:
        logger.info("  dry-run: would clone %s", url)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    env_skip_lfs = {"GIT_LFS_SKIP_SMUDGE": "1"}
    import os
    clone_env = {**os.environ, **env_skip_lfs}

    try:
        # Always clone with LFS smudge skipped first (fast metadata clone)
        subprocess.run(
            ["git", "clone", "--depth=1", url, str(dest)],
            check=True, timeout=600, env=clone_env,
        )
    except subprocess.CalledProcessError as e:
        logger.warning("[FAIL] clone %s: %s", repo_id, e)
        return False

    if is_audioset:
        # Fetch LFS objects in chunks until size limit
        _fetch_lfs_partial(dest, max_size_mb or MAX_AUDIOSET_MB)
    else:
        # Pull all LFS objects for normal-sized datasets
        logger.info("  Fetching LFS objects for %s …", dest.name)
        try:
            subprocess.run(["git", "-C", str(dest), "lfs", "pull"],
                           check=True, timeout=7200)
        except subprocess.CalledProcessError as e:
            logger.warning("[WARN] lfs pull failed for %s: %s", dest.name, e)

    used = _disk_used_mb(dest)
    logger.info("  Done: %s (%.0f MB)", dest.name, used)
    return True


def _fetch_lfs_partial(dest: Path, max_mb: int) -> None:
    """Fetch LFS objects file-by-file until *max_mb* is reached."""
    logger.info("  Partial LFS fetch for %s (cap: %d MB) …", dest.name, max_mb)
    try:
        # List all LFS-tracked pointers
        out = subprocess.check_output(
            ["git", "-C", str(dest), "lfs", "ls-files", "--name-only"],
            text=True, timeout=120,
        )
        lfs_files = [l.strip() for l in out.splitlines() if l.strip()]
    except subprocess.CalledProcessError as e:
        logger.warning("  lfs ls-files failed: %s", e)
        return

    logger.info("  %d LFS-tracked files in %s", len(lfs_files), dest.name)
    fetched = 0
    for fname in lfs_files:
        used = _disk_used_mb(dest)
        if used >= max_mb:
            logger.info("  Cap reached (%.0f / %d MB) — stopping LFS fetch", used, max_mb)
            break
        try:
            subprocess.run(
                ["git", "-C", str(dest), "lfs", "fetch", "--include", fname],
                check=True, capture_output=True, timeout=300,
            )
            subprocess.run(
                ["git", "-C", str(dest), "lfs", "checkout", fname],
                check=True, capture_output=True, timeout=60,
            )
            fetched += 1
        except subprocess.CalledProcessError:
            pass  # individual file failures are non-fatal
    logger.info("  Fetched %d files for %s", fetched, dest.name)


def _summary(datasets: list[tuple]) -> None:
    print("\n" + "═" * 70)
    print(f"  {'Dataset':<42} {'Category':<10} {'On disk':>10}")
    print("  " + "-" * 65)
    for repo_id, name, category, _ in datasets:
        dest = HDD4_BASE / name
        if dest.exists():
            used = _disk_used_mb(dest)
            status = f"{used:>8.0f} MB"
        else:
            status = "   missing"
        print(f"  {name:<42} {category:<10} {status}")
    total = _disk_used_mb(HDD4_BASE)
    print("  " + "-" * 65)
    print(f"  {'TOTAL':52} {total:>8.0f} MB")
    print("═" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-size-mb", type=int, default=None,
                        help="Per-dataset size cap in MB (overrides defaults for large datasets)")
    parser.add_argument("--dataset", nargs="+", metavar="NAME",
                        help="Download only these datasets by local name (e.g. ESC-50 FMA_3secs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without downloading")
    parser.add_argument("--summary", action="store_true",
                        help="Show disk usage summary and exit")
    args = parser.parse_args()

    _check_git_lfs()
    _check_hdd4()

    target_datasets = DATASETS
    if args.dataset:
        names = set(args.dataset)
        target_datasets = [d for d in DATASETS if d[1] in names]
        if not target_datasets:
            sys.exit(f"No matching datasets for: {args.dataset}")

    if args.summary:
        _summary(DATASETS)
        return

    failed = []
    for repo_id, name, category, size_hint_mb in target_datasets:
        dest = HDD4_BASE / name

        cap = args.max_size_mb
        if repo_id == "agkphysics/AudioSet" and cap is None:
            cap = MAX_AUDIOSET_MB

        if cap and size_hint_mb > cap * 2:
            logger.info("[SKIP] %s: estimated size %d MB exceeds cap %d MB",
                        name, size_hint_mb, cap)
            continue

        ok = _clone_or_update(repo_id, dest, max_size_mb=cap, dry_run=args.dry_run)
        if not ok:
            failed.append(name)

    _summary(DATASETS)

    if failed:
        logger.warning("Failed: %s", failed)
        sys.exit(1)
    else:
        logger.info("All datasets present on hdd4.")


if __name__ == "__main__":
    main()
