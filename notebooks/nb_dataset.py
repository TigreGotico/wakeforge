"""Dataset helpers for genetic_search.ipynb.

Provides three dataset loading modes:
1. BYO CSV  — CUSTOM_TRAIN_CSV / CUSTOM_TEST_CSV
2. HF override — HF_DATASET patches positives source
3. Auto  — library default (HF for known wake words, TTS otherwise)

Augmentation / negative overrides are applied post-datagen via
:func:`apply_overrides` without modifying library code.
"""
from __future__ import annotations

import copy
import random
import unittest.mock as mock
from pathlib import Path
from typing import Optional

import ww_trainer.datagen as _datagen_mod
from ww_trainer.datagen import DatagenResult, download_hf_audio_dataset
from ww_trainer.quickstart import QuickstartConfig, _run_or_load_datagen


def parse_hf_list(env_val: str) -> list[str]:
    """Split a comma-separated HF repo string into a clean list."""
    return [s.strip() for s in env_val.split(",") if s.strip()]


def _patched_negative_datasets(
    extra_general: str = "",
    extra_bg_noise: str = "",
    extra_music: str = "",
    extra_rir: str = "",
) -> dict:
    """Return NEGATIVE_DATASETS with any extra HF repos appended."""
    patched = copy.deepcopy(_datagen_mod.NEGATIVE_DATASETS)
    for key, env_val in [
        ("general",  extra_general),
        ("bg_noise", extra_bg_noise),
        ("music",    extra_music),
        ("rir",      extra_rir),
    ]:
        extras = parse_hf_list(env_val)
        if extras:
            patched[key] = patched.get(key, []) + extras
            print(f"  [{key}] extra HF repos: {extras}")
    return patched


def load_byo(
    train_csv: str,
    test_csv: str,
    output_dir: str,
    seed: int,
    negatives_dir: str = "",
    bg_noise_dir: str = "",
    music_dir: str = "",
    rir_dir: str = "",
) -> DatagenResult:
    """Build a DatagenResult from user-supplied CSV paths.

    Args:
        train_csv: Absolute path to ``path,label`` train CSV.
        test_csv: Absolute path to test CSV, or empty to auto-split 80/20.
        output_dir: Root output dir (split written here if needed).
        seed: RNG seed for the 80/20 split.
        negatives_dir: Optional local audio dir for negatives.
        bg_noise_dir: Optional local dir for bg-noise augmentation.
        music_dir: Optional local dir for music augmentation.
        rir_dir: Optional local dir for RIR augmentation.

    Returns:
        :class:`~ww_trainer.datagen.DatagenResult` ready for training.
    """
    train_path = Path(train_csv)
    if not train_path.exists():
        raise FileNotFoundError(f"CUSTOM_TRAIN_CSV not found: {train_path}")

    if test_csv:
        test_path = Path(test_csv)
        if not test_path.exists():
            raise FileNotFoundError(f"CUSTOM_TEST_CSV not found: {test_path}")
    else:
        test_path = _auto_split(train_path, output_dir, seed)
        train_path = test_path.parent / "train_metadata.csv"

    return DatagenResult(
        train_csv=train_path,
        test_csv=test_path,
        positives_dir=train_path.parent,
        negatives_dir=Path(negatives_dir) if negatives_dir else train_path.parent,
        bg_noise_dir=Path(bg_noise_dir) if bg_noise_dir else None,
        music_dir=Path(music_dir)      if music_dir      else None,
        rir_dir=Path(rir_dir)          if rir_dir        else None,
    )


def _auto_split(train_path: Path, output_dir: str, seed: int) -> Path:
    """Write an 80/20 train/test split of *train_path*; idempotent."""
    split_dir = Path(output_dir) / "dataset_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_out = split_dir / "train_metadata.csv"
    test_out  = split_dir / "test_metadata.csv"
    if train_out.exists() and test_out.exists():
        print("Reusing existing 80/20 split (resume)")
        return test_out
    rows = train_path.read_text().splitlines()
    random.seed(seed)
    random.shuffle(rows)
    cut = int(len(rows) * 0.8)
    train_out.write_text("\n".join(rows[:cut]))
    test_out.write_text("\n".join(rows[cut:]))
    print(f"Split {len(rows)} rows → train={cut}, test={len(rows) - cut}")
    return test_out


def load_generated(
    wake_word: str,
    output_dir: str,
    n_positive: int,
    lang: str,
    adversarial: bool,
    download_augmentation: bool,
    seed: int,
    hf_dataset: str = "",
    extra_negatives_hf: str = "",
    extra_bg_noise_hf: str = "",
    extra_music_hf: str = "",
    extra_rir_hf: str = "",
) -> DatagenResult:
    """Run (or resume) datagen with optional HF and negative overrides.

    Args:
        wake_word: Wake word phrase.
        output_dir: Root output directory.
        n_positive: Number of positive samples to synthesise.
        lang: BCP-47 language code for TTS.
        adversarial: Generate adversarial hard-negatives.
        download_augmentation: Download HF bg-noise/music/RIR data.
        seed: Random seed.
        hf_dataset: If set, force this HF repo ID for positives.
        extra_negatives_hf: Comma-sep HF repos appended to general negatives.
        extra_bg_noise_hf: Comma-sep HF repos appended to bg-noise.
        extra_music_hf: Comma-sep HF repos appended to music.
        extra_rir_hf: Comma-sep HF repos appended to RIR.

    Returns:
        :class:`~ww_trainer.datagen.DatagenResult`.
    """
    patched_neg = _patched_negative_datasets(
        extra_negatives_hf, extra_bg_noise_hf, extra_music_hf, extra_rir_hf
    )
    pos_side_effect = (
        mock.patch.object(_datagen_mod, "find_positive_dataset", return_value=hf_dataset)
        if hf_dataset
        else mock.patch.object(
            _datagen_mod, "find_positive_dataset",
            side_effect=_datagen_mod.find_positive_dataset,
        )
    )
    cfg = QuickstartConfig(
        wake_word=wake_word,
        output_dir=Path(output_dir),
        n_positive=n_positive,
        lang=lang,
        adversarial=adversarial,
        download_augmentation=download_augmentation,
        reuse_dataset=True,
        seed=seed,
    )
    with mock.patch.object(_datagen_mod, "NEGATIVE_DATASETS", patched_neg), pos_side_effect:
        return _run_or_load_datagen(cfg)


def apply_local_overrides(
    result: DatagenResult,
    negatives_dir: str = "",
    bg_noise_dir: str = "",
    music_dir: str = "",
    rir_dir: str = "",
    output_dir: str = "",
    extra_bg_noise_hf: str = "",
    extra_music_hf: str = "",
    extra_rir_hf: str = "",
    download_augmentation: bool = False,
) -> DatagenResult:
    """Override DatagenResult aug/neg dirs with local paths or extra HF repos.

    Local-path overrides replace the corresponding field on *result* in-place.
    Extra HF repos are downloaded only when ``download_augmentation=False``
    (i.e. datagen skipped them); otherwise they were already handled by
    :func:`load_generated` via the patched ``NEGATIVE_DATASETS``.

    Args:
        result: DatagenResult to mutate.
        negatives_dir: Local dir to use as general negatives.
        bg_noise_dir: Local dir for bg-noise augmentation.
        music_dir: Local dir for music augmentation.
        rir_dir: Local dir for RIR augmentation.
        output_dir: Root output dir (used to place HF extra downloads).
        extra_bg_noise_hf: Comma-sep HF repos to download for bg-noise.
        extra_music_hf: Comma-sep HF repos to download for music.
        extra_rir_hf: Comma-sep HF repos to download for RIR.
        download_augmentation: Whether datagen already downloaded aug data.

    Returns:
        The mutated *result*.
    """
    if negatives_dir:
        result.negatives_dir = Path(negatives_dir)
        print(f"  negatives_dir → {negatives_dir}")
    for attr, local_val, hf_extra, sub in [
        ("bg_noise_dir", bg_noise_dir, extra_bg_noise_hf, "bg_noise"),
        ("music_dir",    music_dir,    extra_music_hf,    "music"),
        ("rir_dir",      rir_dir,      extra_rir_hf,      "rir"),
    ]:
        if local_val:
            setattr(result, attr, Path(local_val))
            print(f"  {attr} → {local_val}")
        elif parse_hf_list(hf_extra) and not download_augmentation:
            dest = Path(output_dir) / "augmentation" / sub
            dest.mkdir(parents=True, exist_ok=True)
            for ds_id in parse_hf_list(hf_extra):
                download_hf_audio_dataset(ds_id, dest / ds_id.split("/")[-1])
            if not getattr(result, attr):
                setattr(result, attr, dest)
    return result
