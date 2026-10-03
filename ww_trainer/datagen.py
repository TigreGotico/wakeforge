"""Synthetic data pipeline for wake-word training.

Single-command UX: provide a wake word string, get a complete dataset
ready for ``ww_trainer-train`` — including properly organised augmentation
data for train-time use.

Entry point: ``ww_trainer-datagen`` (see :func:`cli_main`).
"""

from __future__ import annotations

import csv
import glob
import inspect
import io
import itertools
import json
import logging
import os
import random
import shutil
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import numpy as np
import soundfile as sf
import torch
import torchaudio

from ww_trainer.dataset import _save_audio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_POSITIVE_DATASETS: Dict[str, str] = {
    "alexa": "TigreGotico/synthetic-wakeword-alexa",
    "hey_mycroft": "TigreGotico/synthetic-wakeword-hey_mycroft",
    "hey_siri": "TigreGotico/synthetic-wakeword-hey_siri",
    "wake_up": "TigreGotico/synthetic-wakeword-wake_up",
    "hey_computer": "TigreGotico/synthetic-wakeword-hey_computer",
    "voice_assistant": "TigreGotico/synthetic-wakeword-voice_assistant",
    "home_assistant": "TigreGotico/synthetic-wakeword-home_assistant",
}

#: A source is ``repo`` or ``repo:config``. A negative repo that declares several Parquet configs must name
#: one, because configs can overlap: AudioSet's ``full`` holds ``balanced`` and ``unbalanced``, so reading
#: every config writes a clip twice. ``balanced`` (about 22,000 clips) covers the 5,000-clip cap.
NEGATIVE_DATASETS: Dict[str, List[str]] = {
    # Non-speech environmental sounds (label=0)
    "general": [
        "TigreGotico/ESC-50",
        "TigreGotico/NAR",
        "agkphysics/AudioSet:balanced",  # large general audio; capped by max_negative
    ],
    # Human speech that is NOT the wake word — critical for preventing
    # models from learning "speech vs silence" instead of the specific phrase.
    # These are downloaded alongside "general" negatives and mixed in.
    "speech": [
        "TigreGotico/not-wake-words-speech-en",        # primary: curated NWW speech
    ],
    "bg_noise": [
        "TigreGotico/ambient_noises",
        "TigreGotico/building_106_kitchen_3secs",
        "TigreGotico/public_domain_sounds_3secs",
    ],
    "music": [
        "TigreGotico/FMA_3secs",
    ],
    "rir": [
        "davidscripka/MIT_environmental_impulse_responses",
    ],
}

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

# Datasets that are too large to materialise locally — always stream them.
# A non-streaming load_dataset() call on these would download hundreds of GB
# (or terabytes) into ~/.cache/huggingface/ before iteration even starts;
# the per-call `max_samples` cap only limits iteration, not the cache fill.
STREAMING_ONLY_DATASETS: set = {
    "agkphysics/AudioSet",  # ~2.4 TB upstream
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def split_source(source: str) -> tuple[str, Optional[str]]:
    """``"org/repo:config"`` → ``("org/repo", "config")``; a bare repo has no config."""
    repo, _, config = source.partition(":")
    return repo, config or None


def normalize_wake_word(s: str) -> str:
    """Normalize a wake-word string to a canonical key.

    ``"hey jarvis"`` → ``"hey_jarvis"``
    """
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def find_positive_dataset(wake_word: str) -> Optional[str]:
    """Return the HuggingFace dataset ID for a known wake word, or *None*."""
    key = normalize_wake_word(wake_word)
    return KNOWN_POSITIVE_DATASETS.get(key)


# ---------------------------------------------------------------------------
# Audio preprocessing (adapted from preprocess.py)
# ---------------------------------------------------------------------------


_VAD_ENGINE = None  # module-level singleton — loading the model is not free


def _load_vad_engine() -> Any:
    """Load the pure-ONNX `vadonnx` VAD (bundled Silero by default).

    Cached after the first call. The bundled Silero model works offline.
    """
    global _VAD_ENGINE
    if _VAD_ENGINE is None:
        from vadonnx import load_vad
        _VAD_ENGINE = load_vad("silero")
    return _VAD_ENGINE


def trim_silence_vad(wav: np.ndarray, sr: int, frame_ms: int = 30) -> np.ndarray:
    """Trim leading/trailing silence using vadonnx (pure-ONNX Silero VAD).

    ``frame_ms`` is accepted for backwards compatibility but unused — vadonnx
    handles framing internally. Returns the span from the first speech segment's
    start to the last segment's end; returns *wav* unchanged if no speech found.
    """
    vad = _load_vad_engine()
    try:
        segments = vad.get_speech_segments(wav.astype(np.float32), sample_rate=sr)
    except Exception:
        return wav
    if not segments:
        return wav
    start = max(0, int(segments[0].start * sr))
    end = min(len(wav), int(segments[-1].end * sr))
    return wav[start:end] if end > start else wav


def preprocess_audio(
    src: Path,
    dst: Path,
    sr: int = 16000,
    vad_trim: bool = True,
) -> bool:
    """Convert *src* to 16 kHz mono WAV at *dst*, with optional VAD trim.

    Returns ``True`` on success.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        arr, orig_sr = sf.read(str(src), dtype="float32", always_2d=True)
        wav = arr.mean(axis=1)
        if orig_sr != sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
        if vad_trim:
            wav = trim_silence_vad(wav, sr)
        max_abs = np.max(np.abs(wav))
        if max_abs > 0:
            wav = wav / max_abs
        _save_audio(str(dst), torch.tensor(wav).unsqueeze(0).float(), sr)
        return True
    except Exception as e:
        logger.warning("Error preprocessing %s: %s", src, e)
        return False


def fit_to_window(wav: np.ndarray, sr: int, window_s: float, rng: random.Random) -> np.ndarray:
    """Crop or zero-pad *wav* to exactly ``window_s`` seconds, at a random offset."""
    n = int(sr * window_s)
    if len(wav) >= n:
        start = rng.randint(0, len(wav) - n)
        return wav[start:start + n]
    out = np.zeros(n, dtype=np.float32)
    start = rng.randint(0, n - len(wav))
    out[start:start + len(wav)] = wav
    return out


def negative_windows(wav: np.ndarray, sr: int, window_s: float, per_clip: int,
                     rng: random.Random) -> List[np.ndarray]:
    """Cut a raw negative clip into up to *per_clip* windows of ``window_s``.

    Long clips give windows at random offsets. Every clip also yields one short
    fragment (0.4 to 1.0 s) zero-padded into a window, so that short padded
    audio exists in both classes and neither length nor padding marks a class.
    """
    n = int(sr * window_s)
    out: List[np.ndarray] = []
    if len(wav) >= n:
        for _ in range(per_clip):
            out.append(fit_to_window(wav, sr, window_s, rng))
    else:
        out.append(fit_to_window(wav, sr, window_s, rng))
    frag = int(sr * rng.uniform(0.4, 1.0))
    if len(wav) > frag:
        start = rng.randint(0, len(wav) - frag)
        out.append(fit_to_window(wav[start:start + frag], sr, window_s, rng))
    return out


def trim_silence_energy(wav: np.ndarray, sr: int, frame_ms: int = 20, floor_db: float = -40.0,
                        margin_db: float = 8.0) -> np.ndarray:
    """Trim leading and trailing frames that sit near the clip's noise floor.

    A frame is kept when it exceeds both *floor_db* under the loudest frame and
    *margin_db* over the clip's noise floor, taken as the 10th percentile of
    frame energy, so recorded clips with background noise trim as well as
    digitally padded ones.
    """
    frame = max(1, int(sr * frame_ms / 1000))
    n_frames = len(wav) // frame
    if n_frames == 0:
        return wav
    rms = np.sqrt((wav[:n_frames * frame].reshape(n_frames, frame) ** 2).mean(axis=1))
    peak = rms.max()
    if peak <= 0:
        return wav
    threshold = max(peak * 10 ** (floor_db / 20), np.percentile(rms, 10) * 10 ** (margin_db / 20))
    voiced = np.flatnonzero(rms > threshold)
    if len(voiced) == 0:
        return wav
    return wav[voiced[0] * frame:(voiced[-1] + 1) * frame]


def split_groups(groups: List[List[tuple[str, int]]], test_split: float,
                 ) -> tuple[List[tuple[str, int]], List[tuple[str, int]]]:
    """Split entries into train and test, keeping each group on one side.

    A group is every window cut from one source clip: windows of one clip
    overlap, so a test window with a sibling in train measures memorised audio.
    Independent examples (a silence window, a single-window positive) are
    groups of one and are split at random. At least one group stays in train.
    """
    groups = [g for g in groups if g]
    random.shuffle(groups)
    if len(groups) < 2:
        return [e for g in groups for e in g], []
    total = sum(len(g) for g in groups)
    n_test = max(1, int(total * test_split))
    test: List[tuple[str, int]] = []
    train: List[tuple[str, int]] = []
    for i, g in enumerate(groups):
        if len(test) < n_test and i < len(groups) - 1:
            test.extend(g)
        else:
            train.extend(g)
    return train, test


SILENCE_CLASSES: tuple[tuple[str, Optional[float]], ...] = (
    ("zeros", None),
    ("noise_m60dbfs", -60.0),
    ("noise_m40dbfs", -40.0),
)


def silence_windows(sr: int, window_s: float, per_class: int, seed: int,
                    classes: tuple[tuple[str, Optional[float]], ...] = SILENCE_CLASSES,
                    ) -> List[tuple[str, np.ndarray]]:
    """Make *per_class* windows of digital silence and near-silence per class.

    A fresh capture buffer on a muted or quiet channel is all zeros or noise
    far under speech level. The shipped precise model scored 0.67 on its first
    chunk of zeros because no training window looked like that. Each class is
    ``(name, level_dbfs)``: ``None`` is all zeros, a level is white noise with
    that RMS in dBFS. The windows are written as they are, never peak
    normalised, or the level would be lost.
    """
    n = int(sr * window_s)
    rs = np.random.RandomState(seed)
    out: List[tuple[str, np.ndarray]] = []
    for name, level in classes:
        for _ in range(per_class):
            if level is None:
                win = np.zeros(n, dtype=np.float32)
            else:
                rms = 10.0 ** (level / 20.0)
                win = np.clip(rs.randn(n) * rms, -1.0, 1.0).astype(np.float32)
            out.append((name, win))
    return out


def _load_mono(src: Path, sr: int) -> np.ndarray:
    arr, orig_sr = sf.read(str(src), dtype="float32", always_2d=True)
    wav = arr.mean(axis=1)
    if orig_sr != sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
    return wav


def _write_window(wav: np.ndarray, dst: Path, sr: int, normalize: bool = True) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    max_abs = np.max(np.abs(wav)) if normalize else 0.0
    if max_abs > 0:
        wav = wav / max_abs
    _save_audio(str(dst), torch.tensor(wav).unsqueeze(0).float(), sr)


# ---------------------------------------------------------------------------
# HuggingFace dataset download
# ---------------------------------------------------------------------------



def _list_repo_audio_files(dataset_id: str) -> List[str]:
    """Sorted audio file paths in a Hugging Face dataset repo, or [] when it is
    not a plain folder of audio files. A failed listing raises."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(dataset_id, repo_type="dataset")
    return sorted(f for f in files if Path(f).suffix.lower() in AUDIO_EXTS)


def _download_audio_files(dataset_id: str, files: List[str], output_dir: Path, sr: int) -> List[Path]:
    """Fetch *files* in one batched snapshot, decode each with soundfile and save 16-bit mono WAV at *sr*.

    One ``snapshot_download`` call fetches only the named files, in parallel,
    where a ``hf_hub_download`` per file costs a sequential round trip each.
    A failed fetch raises, so a source is never silently left with no clips.
    """
    from huggingface_hub import snapshot_download

    try:
        root = Path(snapshot_download(
            dataset_id,
            repo_type="dataset",
            allow_patterns=[glob.escape(name) for name in files],
            max_workers=8,
        ))
    except Exception as exc:
        raise RuntimeError(f"Download of {len(files)} files from {dataset_id} failed: {exc}") from exc

    written: List[Path] = []
    for i, name in enumerate(files):
        try:
            arr, orig_sr = sf.read(root / name, dtype="float32")
        except Exception as exc:
            logger.warning("Skipping %s/%s: %s", dataset_id, name, exc)
            continue
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if orig_sr != sr:
            arr = torchaudio.functional.resample(torch.tensor(arr), orig_sr, sr).numpy()
        fname = output_dir / f"{i:06d}.wav"
        _save_audio(str(fname), torch.tensor(arr).unsqueeze(0).float(), sr)
        written.append(fname)
    logger.info("  Downloaded %d files from %s", len(written), dataset_id)
    return written


def download_hf_audio_dataset(
    dataset_id: str,
    output_dir: Path,
    max_samples: Optional[int] = None,
    sr: int = 16000,
    seed: int = 0,
    hf_config: Optional[List[str]] = None,
    keep_labels: Optional[Set[str]] = None,
    all_configs: bool = True,
) -> List[Path]:
    """Download a HuggingFace audio dataset and save WAVs to *output_dir*.

    A repo stored as Parquet shards is streamed, bounded by *max_samples*, from
    the configs it declares or the ones named in *hf_config*; a named config the
    repo lacks, or *hf_config* on a repo that is not Parquet, raises. With
    *keep_labels* set, rows of a repo with a ``label`` column are kept only when
    their lower-cased label is in it; a ClassLabel feature is read by name, not by index.
    With *all_configs* false, a Parquet repo that declares several configs needs one
    named in *hf_config*, or the call raises. A repo that is a folder of audio files is
    fetched file by file.

    A capped download from a folder repo draws its files with a generator
    seeded by *seed* and *dataset_id*, so the draw is the same on every run and
    never moves the global random state that the train/test split uses.

    Already-downloaded WAV files are reused without any network access.
    With *max_samples* set the corpus is streamed and at most that many rows are
    read, so the download is bounded by the cap. Without it the HuggingFace Arrow
    cache (``~/.cache/huggingface/datasets``) is filled on the first download so
    that subsequent calls with the same dataset skip the HTTP transfer; streaming
    is then only a fallback when the cached download fails.

    Returns list of written file paths.
    """
    from datasets import Audio, load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Skip if already downloaded ──────────────────────────────────────────
    existing = sorted(output_dir.glob("*.wav"))
    need = max_samples if max_samples is not None else 1
    if len(existing) >= need:
        logger.info("Reusing %d cached WAVs from %s (skipping download)", len(existing), output_dir)
        return existing[:max_samples] if max_samples is not None else existing

    logger.info("Downloading %s → %s (max=%s)", dataset_id, output_dir, max_samples)

    # ── Repos stored as Parquet shards with an embedded audio column ───────
    parquet_configs = _list_parquet_configs(dataset_id)
    if parquet_configs:
        missing = [c for c in hf_config or [] if c not in parquet_configs]
        if missing:
            raise ValueError(f"{dataset_id} has no config {missing}; its configs are {parquet_configs}")
        if not all_configs and not hf_config and len(parquet_configs) > 1:
            raise ValueError(f"{dataset_id} declares configs {parquet_configs}; name one as 'repo:config'")
        chosen = hf_config or parquet_configs
        return _download_parquet_dataset(dataset_id, chosen, output_dir, max_samples, sr, seed, keep_labels)
    if hf_config:
        raise ValueError(f"{dataset_id} is not stored as Parquet, so it has no configs to select {hf_config} from")

    # ── Repos that are folders of audio files: fetch the files themselves ──
    # This needs no `datasets` builder at all, so it works where torchcodec
    # (which the Audio feature requires to encode or decode) cannot load.
    listing_error: Optional[Exception] = None
    try:
        audio_files = _list_repo_audio_files(dataset_id)
    except Exception as exc:
        logger.warning("Could not list files of %s: %s", dataset_id, exc)
        listing_error = exc
        audio_files = []
    if audio_files:
        if max_samples is not None and max_samples < len(audio_files):
            # Sample, do not slice. _list_repo_audio_files returns a sorted
            # path list, and a repo that groups its clips in per-class
            # directories puts one class first: the first 20 of
            # TigreGotico/NAR are 20 clips of one refrigerator alarm, out of
            # 42 classes. A capped run is the normal path now that the smoke
            # job sets MAX_NEGATIVE, so the cap has to draw from the whole
            # repo. The generator is local: a draw from the global state would
            # happen only when the files are not cached yet, and would shift
            # the split of a re-run with the same seed.
            rng = random.Random(f"{seed}:{dataset_id}")
            audio_files = sorted(rng.sample(audio_files, max_samples))
        return _download_audio_files(dataset_id, audio_files, output_dir, sr)

    # ── Choose how the corpus is read ───────────────────────────────────────
    # A capped call streams: non-streaming load_dataset() fills the whole Arrow
    # cache before iteration starts, so max_samples would only trim what was
    # already downloaded. An uncapped call prefers the cached download, so
    # re-runs with the same dataset_id are instant, and falls back to streaming
    # if that fails. STREAMING_ONLY_DATASETS never take the non-streaming path.
    # Note: trust_remote_code was removed in datasets ≥ 3.x; omit it.
    ds = None
    if dataset_id in STREAMING_ONLY_DATASETS:
        modes = (True,)
    elif max_samples is not None:
        modes = (True, False)
    else:
        modes = (False, True)
    for streaming in modes:
        try:
            ds = load_dataset(dataset_id, split="train", streaming=streaming)
            # Read the encoded bytes and decode them with soundfile: the default
            # decoder needs torchcodec, whose wheel links CUDA libraries and
            # cannot load next to a ROCm or CPU-only torch.
            for col in ("audio", "Audio", "sound", "file"):
                features = ds.features or {}
                if col in features and isinstance(features[col], Audio):
                    ds = ds.cast_column(col, Audio(decode=False))
                    break
            break
        except Exception as exc:
            if streaming != modes[-1]:
                logger.warning(
                    "Loading %s (streaming=%s) failed (%s); retrying with streaming=%s",
                    dataset_id, streaming, exc, not streaming,
                )
            else:
                raise RuntimeError(
                    f"Download of {dataset_id} failed: listing: {listing_error or 'ok, not a folder of audio files'}; "
                    f"load (streaming={streaming}): {exc}"
                ) from exc

    if ds is None:
        return []

    written, _ = _write_rows(
        ds if max_samples is None else itertools.islice(ds, max_samples),
        dataset_id, output_dir, sr,
    )
    logger.info("  Downloaded %d files from %s", len(written), dataset_id)
    return written


def _write_rows(
    rows: Any, dataset_id: str, output_dir: Path, sr: int, start: int = 0,
) -> tuple[List[Path], List[tuple[str, str]]]:
    """Decode the audio column of *rows* and save each as a 16-bit mono WAV at *sr*.

    Files are named by row position from *start*. Returns the written paths and,
    where the rows carry a ``label`` column, a ``(file name, label)`` pair for each.
    """
    written: List[Path] = []
    labels: List[tuple[str, str]] = []
    audio_col = None
    for i, example in enumerate(rows, start):
        if audio_col is None:
            for col in ("audio", "Audio", "sound", "file"):
                if col in example:
                    audio_col = col
                    break
            if audio_col is None:
                # fallback: first column whose value is a dict with "array"
                for col, val in example.items():
                    if isinstance(val, dict) and "array" in val:
                        audio_col = col
                        break
            if audio_col is None:
                logger.warning("No audio column found in %s", dataset_id)
                return written, labels

        audio = example[audio_col]
        try:
            if isinstance(audio, dict) and audio.get("bytes") is not None:
                # encoded file bytes (Audio(decode=False))
                arr, orig_sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            elif isinstance(audio, dict) and audio.get("path"):
                arr, orig_sr = sf.read(audio["path"], dtype="float32")
            elif isinstance(audio, dict) and "array" in audio:
                arr = np.array(audio["array"], dtype=np.float32)
                orig_sr = audio.get("sampling_rate", sr)
            else:
                continue
        except Exception as exc:
            # soundfile has no decoder for some formats (m4a among them).
            logger.warning("Skipping %s row %d: %s", dataset_id, i, exc)
            continue
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        if orig_sr != sr:
            arr = torchaudio.functional.resample(
                torch.tensor(arr), orig_sr, sr
            ).numpy()

        fname = output_dir / f"{i:06d}.wav"
        _save_audio(str(fname), torch.tensor(arr).unsqueeze(0).float(), sr)
        written.append(fname)
        if "label" in example:
            labels.append((fname.name, str(example["label"])))
    return written, labels


def _list_parquet_configs(dataset_id: str) -> List[str]:
    """Config names of a Hugging Face dataset repo stored as Parquet shards, or []
    when the repo holds no ``*.parquet`` file or cannot be listed."""
    from huggingface_hub import HfApi

    try:
        files = HfApi().list_repo_files(dataset_id, repo_type="dataset")
    except Exception as exc:
        logger.warning("Could not list files of %s: %s", dataset_id, exc)
        return []
    if not any(f.endswith(".parquet") for f in files):
        return []
    from datasets import get_dataset_config_names

    return list(get_dataset_config_names(dataset_id))


def _label_text(value: Any, names: Any) -> str:
    """Lower-cased label of a row; *names* is a ClassLabel feature that turns an index into its name."""
    if names is not None and isinstance(value, int):
        value = names.int2str(value)
    return str(value).strip().lower()


def _download_parquet_dataset(
    dataset_id: str,
    configs: List[str],
    output_dir: Path,
    max_samples: Optional[int],
    sr: int,
    seed: int,
    keep_labels: Optional[Set[str]] = None,
) -> List[Path]:
    """Stream the ``train`` split of each of *configs* and save its clips as WAVs.

    A cap is shared out between the configs, rounded up, and the run stops at the
    cap. A ``labels.csv`` beside the WAVs records the ``label`` column where the
    repo has one. A failed load raises.
    """
    import zlib
    from datasets import Audio, ClassLabel, load_dataset

    written: List[Path] = []
    labels: List[tuple[str, str]] = []
    quota = None if max_samples is None else -(-max_samples // len(configs))
    for config in configs:
        remaining = None if max_samples is None else min(quota, max_samples - len(written))
        if remaining is not None and remaining <= 0:
            break
        try:
            ds = load_dataset(dataset_id, name=config, split="train", streaming=True)
            features = ds.features or {}
            if "audio" in features:
                ds = ds.cast_column("audio", Audio(decode=False))
            names = features["label"] if isinstance(features.get("label"), ClassLabel) else None
            if remaining is not None:
                ds = ds.shuffle(seed=zlib.crc32(f"{seed}:{dataset_id}:{config}".encode()),
                                buffer_size=min(1000, 4 * remaining))
        except Exception as exc:
            raise RuntimeError(f"Load of {dataset_id} config {config} failed: {exc}") from exc
        rows = ds
        if keep_labels is not None:
            rows = (r for r in rows if "label" not in r or _label_text(r["label"], names) in keep_labels)
        if remaining is not None:
            rows = itertools.islice(rows, remaining)
        got, got_labels = _write_rows(rows, dataset_id, output_dir, sr, start=len(written))
        written.extend(got)
        labels.extend(got_labels)
    if labels:
        with open(output_dir / "labels.csv", "w", newline="") as f:
            csv.writer(f).writerows([("file", "label"), *labels])
    logger.info("  Downloaded %d files from %s (configs %s)", len(written), dataset_id, configs)
    return written


# ---------------------------------------------------------------------------
# TTS synthesis (adapted from 02_tts_synth.py)
# ---------------------------------------------------------------------------


#: Plugins that synthesize through a remote server: without a configured host they fall back to public
#: servers, so they are used only when ``--tts-config`` names the host.
_HOST_PLUGINS = {"ovos-tts-plugin-server"}

#: Speaking rates drawn per edge-tts clip.
_EDGE_RATES = ("-20%", "-10%", "+0%", "+10%", "+20%", "+30%")

#: Seconds between two requests to one plugin, so a long run is not rate limited by the online services;
#: ``tts_config[name]["min_interval_s"]`` overrides it.
_MIN_INTERVAL_S = {"ovos-tts-plugin-edge-tts": 0.25, "ovos-tts-plugin-google-tx": 1.0, "ovos-tts-plugin-server": 0.0}

#: Attempts per clip, with the wait doubling after each failure.
_ATTEMPTS = 3


def load_tts_config(path: Optional[str]) -> Dict[str, dict]:
    """Read ``{plugin name: plugin config}`` from a JSON file; ``None`` gives an empty mapping."""
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _collect_tts_plugins(lang: str, tts_config: Optional[Dict[str, dict]] = None) -> List[tuple[str, Any]]:
    """Discover all installed OVOS TTS plugins via OPM and return ``(name, instance)`` pairs.

    Uses ``ovos_plugin_manager.tts.find_tts_plugins()`` for entry-point
    discovery, so any installed OVOS TTS plugin (edge-tts, google-tx,
    phoonnx, piper, …) is automatically picked up. ``tts_config`` gives a
    plugin its config; a remote-server plugin is used only when that config
    names a ``host``.
    """
    from ovos_plugin_manager.tts import find_tts_plugins

    tts_config = tts_config or {}
    plugins = find_tts_plugins()  # Dict[str, Type[TTS]]
    instances: List[tuple[str, Any]] = []
    for name, plugin_cls in plugins.items():
        cfg = {"lang": lang, **tts_config.get(name, {})}
        if name in _HOST_PLUGINS and not cfg.get("host"):
            logger.info("Skipping %s: no host in the TTS config", name)
            continue
        try:
            instance = plugin_cls(config=cfg)
            instances.append((name, instance))
        except Exception:
            continue
    if not instances:
        raise RuntimeError(
            "No TTS plugins installed. Install at least one OVOS TTS plugin:\n"
            "  uv pip install ovos-tts-plugin-edge-tts\n"
            "  uv pip install ovos-tts-plugin-google-tx\n"
            "  uv pip install ovos-tts-plugin-phoonnx"
        )
    if len(instances) == 1:
        logger.warning(
            "Only one TTS plugin installed (%s) — synthesised positives will "
            "have very limited voice diversity. Install additional plugins "
            "(ovos-tts-plugin-google-tx, ovos-tts-plugin-phoonnx, …) for "
            "better generalisation. Also ensure --lang carries a region tag "
            "(e.g. 'nl-nl', not 'nl') so the plugin picks a native voice "
            "rather than falling back to its default (English).",
            instances[0][0],
        )
    return instances


def voice_options(name: str, plugin: Any, lang: str,
                  tts_config: Optional[Dict[str, dict]] = None) -> List[Dict[str, Any]]:
    """Every per-call option set one plugin can synthesize with: its voices, accents and speaking rates.

    A plugin left at its default voice says the wake word in one voice however many clips are asked for,
    so each clip draws one of these instead. ``tts_config[name]["voices"]`` lists voices explicitly (a
    remote server's catalogue, say); edge-tts offers every voice of the language at several rates;
    google-tx offers its regional accents; any other plugin exposing ``available_voices`` offers those.
    """
    prefix = lang.split("-")[0].lower()
    listed = (tts_config or {}).get(name, {}).get("voices")
    if listed:
        return [{"voice": v} for v in listed]
    if name == "ovos-tts-plugin-edge-tts":
        try:
            from ovos_tts_plugin_edge_tts import VOICES
        except ImportError:
            return [{}]
        voices = [v for loc, vs in VOICES.items() if loc.lower().split("-")[0] == prefix for v in vs]
        return [{"voice": v, "rate": r} for v in voices for r in _EDGE_RATES] or [{}]
    if name == "ovos-tts-plugin-google-tx":
        try:
            from ovos_tts_plugin_google_tx import REGIONAL_CONFIGS
        except ImportError:
            return [{}]
        return [{"lang": loc} for loc in REGIONAL_CONFIGS if loc.lower().split("-")[0] == prefix] or [{}]
    voices = getattr(plugin, "available_voices", None)
    if isinstance(voices, (list, tuple, set, dict)) and voices:
        return [{"voice": v} for v in voices]
    return [{}]


def _accepted_options(plugin: Any, opts: Dict[str, Any]) -> Dict[str, Any]:
    """The options ``plugin.get_tts`` accepts; a keyword it would reject is dropped so the call can succeed."""
    try:
        params = inspect.signature(plugin.get_tts).parameters
    except (TypeError, ValueError):
        return opts
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return opts
    return {k: v for k, v in opts.items() if k in params}


def synthesize_positives(
    wake_word: str,
    output_dir: Path,
    n: int = 1000,
    lang: str = "en",
    tts_config: Optional[Dict[str, dict]] = None,
) -> List[Path]:
    """Synthesize *n* positive samples using available TTS plugins.

    Each clip draws a plugin and then one of that plugin's voices, accents or
    speaking rates (:func:`voice_options`). Every clip written gets a row in
    ``synthesis.csv`` beside it naming the plugin and the options used.

    Returns list of written WAV paths.

    Raises:
        RuntimeError: If no TTS plugins are installed.
    """
    tts_instances = _collect_tts_plugins(lang, tts_config)
    options = {name: voice_options(name, plugin, lang, tts_config) for name, plugin in tts_instances}
    interval = {name: float((tts_config or {}).get(name, {}).get("min_interval_s", _MIN_INTERVAL_S.get(name, 0.0)))
                for name, _ in tts_instances}
    last_call: Dict[str, float] = {}
    failures: Dict[str, int] = {}
    last_error: Dict[str, str] = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    text = wake_word.replace("_", " ").replace("-", " ")
    written: List[Path] = []
    manifest = output_dir / "synthesis.csv"
    new_manifest = not manifest.exists()
    with open(manifest, "a", newline="", encoding="utf-8") as fh:
        rows = csv.writer(fh)
        if new_manifest:
            rows.writerow(["file", "text", "plugin", "options"])
        for i in range(n):
            name, plugin = random.choice(tts_instances)
            opts = random.choice(options[name])
            call_opts = _accepted_options(plugin, {"lang": lang, **opts})
            out_path = output_dir / f"{uuid4().hex[:12]}.wav"
            # written under a temporary name and moved into place only when the synthesis returned audio, so a
            # failed or refused request leaves no file the manifest does not list
            tmp_path = out_path.with_suffix(".part.wav")
            wait = max(interval[name], 0.5)
            for attempt in range(_ATTEMPTS):
                pause = interval[name] - (time.monotonic() - last_call.get(name, -1e9))
                if pause > 0:
                    time.sleep(pause)
                last_call[name] = time.monotonic()
                try:
                    plugin.get_tts(text, str(tmp_path), **call_opts)
                    if tmp_path.exists() and tmp_path.stat().st_size > 44:
                        tmp_path.replace(out_path)
                        written.append(out_path)
                        rows.writerow([out_path.name, text, name, json.dumps(call_opts, sort_keys=True)])
                        break
                    last_error[name] = "no audio written"
                except Exception as e:
                    last_error[name] = f"{type(e).__name__}: {e}"
                    logger.debug("%s failed on attempt %d: %s", name, attempt + 1, e)
                tmp_path.unlink(missing_ok=True)
                if attempt + 1 < _ATTEMPTS:
                    time.sleep(wait)
                    wait *= 2
            else:
                failures[name] = failures.get(name, 0) + 1

    for name, count in sorted(failures.items()):
        logger.warning("%s produced no audio for %d clips after %d attempts each; last error: %s",
                       name, count, _ATTEMPTS, last_error.get(name))
    logger.info("Synthesized %d / %d positive samples", len(written), n)
    return written


# ---------------------------------------------------------------------------
# Voice conversion
# ---------------------------------------------------------------------------


def voice_convert_batch(
    input_dir: Path,
    output_dir: Path,
    vc_refs_dir: Path,
    device: str = "auto",
    n_target: Optional[int] = None,
    engine: Optional[str] = None,
) -> List[Path]:
    """Voice-convert all WAVs in *input_dir* using random reference voices.

    Delegates to the pure-ONNX `voiceclonnx` library via
    :func:`ww_trainer.vc_helpers.load_vc_backend`. Each source WAV is converted
    using a randomly chosen reference voice. If *n_target* > number of source
    files, sources are reused with different references.

    Args:
        engine: voiceclonnx engine alias (default: ``WW_VC_ENGINE`` env, then
            ``knnvc``). ``device`` is accepted for backwards compatibility but
            ignored (voiceclonnx is CPU/ONNX).

    Returns list of output WAV paths.
    """
    from ww_trainer.vc_helpers import load_vc_backend

    ref_voices = sorted(vc_refs_dir.rglob("*.wav"))
    if not ref_voices:
        logger.warning("No reference voices found in %s", vc_refs_dir)
        return []

    sources = sorted(input_dir.rglob("*.wav"))
    if not sources:
        logger.warning("No source WAVs in %s", input_dir)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    vc_model = load_vc_backend(engine=engine)
    logger.info("Voice conversion via %s (sr=%d)", vc_model.name, vc_model.sample_rate)

    if n_target is None:
        n_target = len(sources)

    written: List[Path] = []
    for i in range(n_target):
        src = sources[i % len(sources)]
        ref = random.choice(ref_voices)
        out_path = output_dir / f"vc_{uuid4().hex[:12]}.wav"
        try:
            vc_model.vc(str(src), str(ref), str(out_path))
            if out_path.exists():
                written.append(out_path)
        except Exception as e:
            logger.warning("VC failed for %s: %s", src.name, e)

    logger.info("Voice-converted %d / %d samples", len(written), n_target)
    return written


# ---------------------------------------------------------------------------
# Adversarial generation (adapted from 01_adversarial_gen.py)
# ---------------------------------------------------------------------------

# Import GraphemeAugmenter and LLM generation from the scripts module
# We copy the classes here to keep datagen self-contained.

class GraphemeAugmenter:
    """Generate hard-negative confusable strings via single-grapheme edits."""

    STANDARD_VOWELS: Set[str] = set("aeiou")
    STANDARD_CONSONANTS: Set[str] = set("bcdfghjklmnpqrstvwxyz")

    def __init__(
        self,
        max_edits: int = 1,
        min_edits: int = 1,
        language_vowels: Optional[Set[str]] = None,
        language_consonants: Optional[Set[str]] = None,
    ) -> None:
        if max_edits < 1:
            raise ValueError("max_edits must be 1 or greater.")
        if min_edits < 0:
            raise ValueError("min_edits cannot be negative.")
        if min_edits > max_edits:
            raise ValueError("min_edits cannot be greater than max_edits.")
        self.max_edits = max_edits
        self.min_edits = min_edits
        self.VOWELS = self.STANDARD_VOWELS.union(
            {c.lower() for c in (language_vowels or set())}
        )
        self.CONSONANTS = self.STANDARD_CONSONANTS.union(
            {c.lower() for c in (language_consonants or set())}
        )
        self.ALL_GRAPHEMES = self.VOWELS.union(self.CONSONANTS)

    def _get_char_class(self, char: str) -> Optional[str]:
        c = char.lower()
        if c in self.VOWELS:
            return "vowel"
        if c in self.CONSONANTS:
            return "consonant"
        return None

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return GraphemeAugmenter._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                cost = 0 if c1 == c2 else 1
                current_row.append(
                    min(
                        previous_row[j + 1] + 1,
                        current_row[j] + 1,
                        previous_row[j] + cost,
                    )
                )
            previous_row = current_row
        return previous_row[-1]

    def _get_random_one_edit(self, text: str) -> str:
        if not text:
            return random.choice(list(self.ALL_GRAPHEMES))
        edit_type = random.randint(0, 2)
        if edit_type == 0:  # Insertion
            pos = random.randint(0, len(text))
            g = random.choice(list(self.ALL_GRAPHEMES))
            return text[:pos] + g + text[pos:]
        elif edit_type == 1:  # Deletion
            pos = random.randint(0, len(text) - 1)
            return text[:pos] + text[pos + 1:]
        else:  # Substitution
            for _ in range(10):
                pos = random.randint(0, len(text) - 1)
                char_class = self._get_char_class(text[pos])
                if char_class == "vowel":
                    target_set = self.VOWELS
                elif char_class == "consonant":
                    target_set = self.CONSONANTS
                else:
                    continue
                possible = list(target_set - {text[pos].lower()})
                if possible:
                    return text[:pos] + random.choice(possible) + text[pos + 1:]
            return self._get_random_one_edit(text)

    def generate_confusables(self, keyword: str, n_samples: int) -> List[str]:
        """Generate *n_samples* confusable strings for *keyword*."""
        original = keyword.lower()
        generated: Set[str] = set()
        if n_samples <= 0:
            return []
        seed_pool: Set[str] = {original}
        attempts = 0
        max_attempts = n_samples * 10 + 100
        while len(generated) < n_samples and attempts < max_attempts:
            word = random.choice(list(seed_pool))
            candidate = self._get_random_one_edit(word)
            distance = self._levenshtein_distance(original, candidate)
            if (
                self.min_edits <= distance <= self.max_edits
                and candidate != original
                and candidate not in generated
            ):
                generated.add(candidate)
                if distance < self.max_edits:
                    seed_pool.add(candidate)
            elif distance < self.max_edits and candidate != original:
                seed_pool.add(candidate)
            attempts += 1
        return sorted(generated)


_ADV_SYSTEM_PROMPT = (
    "You are an expert in computational linguistics, phonetics, and adversarial "
    "machine learning. Your core function is to generate lists of strings that are "
    "acoustically and linguistically similar to a given wake word or keyword. "
    "These generated strings serve as adversarial samples intended to trick an ASR "
    "system or keyword spotter.\n\n"
    "Generation Goal: The adversarial samples must primarily rhyme with the components "
    "of the input keyword or mimic its overall rhythm and length.\n\n"
    "Output Constraints (CRITICAL):\n"
    "1. Format: Output only the generated adversarial strings.\n"
    "2. Delimiter: Use a newline to separate each sample (one per line).\n"
    "3. Exclusion: DO NOT include any introductory text, numbering, bullet points, "
    "explanations, or concluding remarks.\n"
    "4. Diversity: All samples MUST be unique and PHONETICALLY similar."
)

_ADV_USER_TEMPLATE = (
    "Generate {n_samples} adversarial, rhyming samples for the "
    "following keyword: {word}"
)


def generate_adversarial_texts(
    wake_word: str,
    n: int = 20,
    llm_url: Optional[str] = None,
    llm_model: str = "gemma3:4b",
) -> List[str]:
    """Generate adversarial hard-negative phrases for *wake_word*.

    Uses :class:`GraphemeAugmenter` and optionally an Ollama LLM endpoint.
    """
    results: Set[str] = set()

    # Grapheme augmentation (always available)
    aug = GraphemeAugmenter(max_edits=3, min_edits=2)
    graph_n = n if not llm_url else max(1, n // 5)
    for s in aug.generate_confusables(wake_word, graph_n):
        results.add(s.strip().lower())

    # LLM generation (optional)
    if llm_url:
        import requests

        llm_n = n - len(results)
        if llm_n > 0:
            try:
                resp = requests.post(
                    f"{llm_url}/api/generate",
                    json={
                        "model": llm_model,
                        "prompt": _ADV_USER_TEMPLATE.format(
                            word=wake_word, n_samples=llm_n * 2
                        ),
                        "system": _ADV_SYSTEM_PROMPT,
                        "stream": False,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                raw = resp.json()["response"].strip().split("\n")
                word_count = len(wake_word.split())
                for line in raw:
                    line = line.strip().lower()
                    if line and len(line.split()) == word_count:
                        results.add(line)
            except Exception as e:
                logger.warning("LLM adversarial generation failed: %s", e)

    return sorted(results)[:n]


# ---------------------------------------------------------------------------
# Metadata CSV I/O
# ---------------------------------------------------------------------------


def write_metadata_csv(path: Path, entries: List[tuple[str, int]]) -> None:
    """Write a ``path,label`` metadata CSV (no header)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for file_path, label in entries:
            writer.writerow([file_path, label])


def read_metadata_csv(path: Path) -> List[tuple[str, int]]:
    """Read a ``path,label`` metadata CSV (no header)."""
    entries: List[tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                entries.append((row[0], int(row[1])))
    return entries


# ---------------------------------------------------------------------------
# Config & Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DatagenConfig:
    """Configuration for the synthetic data pipeline."""

    wake_word: str
    output_dir: Path
    n_positive: int = 1000
    max_negative: Optional[int] = None
    hf_config: Optional[List[str]] = None
    lang: str = "en"
    sample_rate: int = 16000
    vad_trim: bool = True
    window_s: float = 1.5
    neg_windows_per_clip: int = 3
    silence_windows: int = 300
    adversarial_file: Optional[str] = None
    adversarial_per_text: int = 3
    allow_positive_skips: bool = False
    test_split: float = 0.2
    # Voice cloning
    vc_refs_dir: Optional[str] = None
    vc_device: str = "auto"
    # Adversarial
    adversarial: bool = False
    adversarial_n: int = 20
    llm_url: Optional[str] = None
    llm_model: str = "gemma3:4b"
    # TTS plugin configs, {plugin name: config} as JSON (a remote server's host and voice list)
    tts_config: Optional[str] = None
    # Augmentation resources
    download_augmentation: bool = True
    pre_augment: bool = False
    seed: int = 42


@dataclass
class DatagenResult:
    """Result of a completed datagen pipeline run."""

    train_csv: Path
    test_csv: Path
    positives_dir: Path
    negatives_dir: Path
    bg_noise_dir: Optional[Path] = None
    music_dir: Optional[Path] = None
    rir_dir: Optional[Path] = None
    n_positive: int = 0
    n_negative: int = 0
    config_path: Optional[Path] = None

    def suggested_train_command(
        self, wake_word: str, tier: str = "small"
    ) -> str:
        """Return a ready-to-copy ``ww_trainer-train`` command string."""
        parts = [
            "ww_trainer-train",
            f"--wake-word '{normalize_wake_word(wake_word)}'",
            f"--metadata {self.train_csv}",
            f"--test-metadata {self.test_csv}",
        ]
        if self.bg_noise_dir and self.bg_noise_dir.exists():
            parts.append(f"--bg-noise-folder {self.bg_noise_dir}")
        if self.music_dir and self.music_dir.exists():
            parts.append(f"--music-folder {self.music_dir}")
        if self.rir_dir and self.rir_dir.exists():
            parts.append(f"--rir-folder {self.rir_dir}")
        parts.extend([f"--tier {tier}", "--save-best"])
        return " \\\n  ".join(parts)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def run_datagen_pipeline(config: DatagenConfig) -> DatagenResult:
    """Run the full synthetic data generation pipeline.

    Stages:
        1. Acquire positives (HF download or TTS synthesis, optional VC)
        2. Acquire negatives (purpose-mapped HF downloads)
        3. Optional adversarial hard-negatives
        4. Preprocess all audio (positives + general negatives)
        5. Train/test split + write metadata CSVs
        6. Write config JSON for reproducibility

    Returns:
        :class:`DatagenResult` with paths and counts.
    """
    random.seed(config.seed)
    np.random.seed(config.seed)

    out = Path(config.output_dir)
    positives_dir = out / "positives"
    negatives_dir = out / "negatives"
    aug_dir = out / "augmentation"
    bg_noise_dir = aug_dir / "bg_noise"
    music_dir = aug_dir / "music"
    rir_dir = aug_dir / "rir"
    train_dir = out / "train"
    test_dir = out / "test"

    ww_key = normalize_wake_word(config.wake_word)

    # ── Stage 1: Acquire positives ──────────────────────────────────────
    logger.info("Stage 1: Acquiring positive samples for '%s'", ww_key)

    hf_dataset = find_positive_dataset(ww_key)
    if hf_dataset:
        logger.info("  Known wake word — downloading from %s", hf_dataset)
        pos_files = download_hf_audio_dataset(
            hf_dataset, positives_dir, max_samples=config.n_positive,
            sr=config.sample_rate, seed=config.seed, hf_config=config.hf_config,
            keep_labels={"1", "true", ww_key, ww_key.replace("_", " "), ww_key.replace("_", "-")},
        )
    else:
        logger.info("  Unknown wake word — synthesizing via TTS")
        pos_files = synthesize_positives(
            ww_key, positives_dir, n=config.n_positive, lang=config.lang,
            tts_config=load_tts_config(config.tts_config),
        )

    # Optional voice conversion
    if config.vc_refs_dir:
        logger.info("  Applying voice conversion with refs from %s", config.vc_refs_dir)
        vc_out = positives_dir / "vc"
        vc_files = voice_convert_batch(
            positives_dir,
            vc_out,
            Path(config.vc_refs_dir),
            device=config.vc_device,
            n_target=config.n_positive,
        )
        # Move VC files up into positives_dir
        for f in vc_files:
            dst = positives_dir / f.name
            shutil.move(str(f), str(dst))
        if vc_out.exists():
            shutil.rmtree(vc_out, ignore_errors=True)
        # Re-collect all positives
        pos_files = sorted(
            p for p in positives_dir.iterdir()
            if p.suffix.lower() in AUDIO_EXTS
        )

    # ── Stage 2: Acquire negatives ──────────────────────────────────────
    logger.info("Stage 2: Acquiring negative samples (purpose-mapped)")

    # General negatives (environmental sounds) + speech negatives → label=0
    # Speech negatives are essential: without them models learn "speech vs
    # silence" rather than "this specific wake word vs everything else".
    # Datasets that are extremely large and must always be capped.
    _LARGE_DATASETS: Dict[str, int] = {
        "agkphysics/AudioSet": 5000,
    }

    neg_files: List[Path] = []
    phrase_of: dict[Path, str] = {}
    for category in ("general", "speech"):
        for source in NEGATIVE_DATASETS.get(category, []):
            ds_id, ds_config = split_source(source)
            ds_name = ds_id.split("/")[-1]
            ds_out = negatives_dir / ds_name
            # Respect user's max_negative, but enforce a hard cap for huge datasets.
            hard_cap = _LARGE_DATASETS.get(ds_id)
            if hard_cap is not None:
                cap = min(config.max_negative, hard_cap) if config.max_negative else hard_cap
            else:
                cap = config.max_negative
            files = download_hf_audio_dataset(
                ds_id, ds_out, max_samples=cap, sr=config.sample_rate, seed=config.seed,
                hf_config=[ds_config] if ds_config else None, all_configs=False,
            )
            neg_files.extend(files)

    # Augmentation resources (NOT in CSV)
    if config.download_augmentation:
        for source in NEGATIVE_DATASETS["bg_noise"]:
            ds_id, ds_config = split_source(source)
            ds_name = ds_id.split("/")[-1]
            download_hf_audio_dataset(
                ds_id, bg_noise_dir / ds_name, sr=config.sample_rate,
                hf_config=[ds_config] if ds_config else None, all_configs=False,
            )

        for source in NEGATIVE_DATASETS["music"]:
            ds_id, ds_config = split_source(source)
            ds_name = ds_id.split("/")[-1]
            download_hf_audio_dataset(
                ds_id, music_dir / ds_name, sr=config.sample_rate,
                hf_config=[ds_config] if ds_config else None, all_configs=False,
            )

        for source in NEGATIVE_DATASETS["rir"]:
            ds_id, ds_config = split_source(source)
            ds_name = ds_id.split("/")[-1]
            download_hf_audio_dataset(
                ds_id, rir_dir / ds_name, sr=config.sample_rate,
                hf_config=[ds_config] if ds_config else None, all_configs=False,
            )

    # ── Stage 3: Optional adversarial hard-negatives ────────────────────
    if config.adversarial:
        logger.info("Stage 3: Generating adversarial hard-negatives")
        adv_texts = generate_adversarial_texts(
            ww_key,
            n=config.adversarial_n,
            llm_url=config.llm_url,
            llm_model=config.llm_model,
        )
        if config.adversarial_file:
            with open(config.adversarial_file, encoding="utf-8") as fh:
                adv_texts += [line.strip() for line in fh if line.strip() and not line.startswith("#")]
        if adv_texts:
            adv_dir = negatives_dir / "adversarial"
            try:
                adv_files = synthesize_positives(
                    adv_texts[0], adv_dir, n=0, lang=config.lang,
                    tts_config=load_tts_config(config.tts_config),
                )
                # Synthesize each adversarial text
                for text in adv_texts:
                    files = synthesize_positives(text, adv_dir, n=config.adversarial_per_text, lang=config.lang,
                                                 tts_config=load_tts_config(config.tts_config))
                    neg_files.extend(files)
                    phrase_of.update({f: text for f in files})
            except RuntimeError:
                logger.warning("No TTS available for adversarial synthesis")

    # ── Stage 4: Preprocess all audio ───────────────────────────────────
    # Every training example is exactly window_s long: positives are VAD
    # trimmed then placed at a random offset in a zero-padded window; negatives
    # are windows cut from the full clips plus one short zero-padded fragment
    # per clip. Length and padding therefore mark neither class. window_s=0
    # keeps the clips as they are.
    logger.info("Stage 4: Preprocessing audio files (window %.2f s)", config.window_s)
    rng = random.Random(config.seed)

    processed_positives: List[Path] = []
    skipped_too_long = 0
    skipped_other = 0
    proc_pos_dir = positives_dir / "processed"
    for f in pos_files:
        dst = proc_pos_dir / f.name
        if config.window_s <= 0:
            if preprocess_audio(f, dst, sr=config.sample_rate, vad_trim=config.vad_trim):
                processed_positives.append(dst)
            else:
                skipped_other += 1
            continue
        try:
            wav = _load_mono(f, config.sample_rate)
            if config.vad_trim:
                wav = trim_silence_vad(wav, config.sample_rate)
            window = int(config.sample_rate * config.window_s)
            if len(wav) > window:
                # VAD is off, raised or found nothing: cut the padding a TTS
                # engine leaves around the word before judging the length.
                wav = trim_silence_energy(wav, config.sample_rate)
            # fit_to_window crops a clip longer than the window at a uniform
            # random offset with no notion of where the wake word sits in it,
            # so a positive still longer than the window is skipped rather
            # than cropped: a blind crop can carry none of the word under the
            # positive label.
            if len(wav) > window:
                logger.warning(
                    "Skipping %s: %.2fs after trimming exceeds the %.2fs window; "
                    "a blind crop could drop the wake word", f,
                    len(wav) / config.sample_rate, config.window_s,
                )
                skipped_too_long += 1
                continue
            _write_window(fit_to_window(wav, config.sample_rate, config.window_s, rng), dst, config.sample_rate)
            processed_positives.append(dst)
        except Exception as e:
            logger.warning("Error preprocessing %s: %s", f, e)
            skipped_other += 1

    skipped = skipped_too_long + skipped_other
    logger.info("Positives: %d kept, %d skipped too long, %d skipped other (of %d)",
                len(processed_positives), skipped_too_long, skipped_other, len(pos_files))
    if skipped > 0.2 * len(pos_files) and not config.allow_positive_skips:
        raise RuntimeError(
            f"{skipped} of {len(pos_files)} positives were skipped "
            f"({skipped_too_long} longer than the {config.window_s:.2f} s window after "
            f"trimming, {skipped_other} unreadable); the positive class would be a "
            "biased remainder. Raise --window-s, fix the source clips, or pass "
            "--allow-positive-skips to continue."
        )

    processed_negatives: List[Path] = []
    negative_groups: List[List[tuple[str, int]]] = []
    phrase_groups: dict[str, List[tuple[str, int]]] = {}
    proc_neg_dir = negatives_dir / "processed"

    def _group_for(f: Path) -> List[tuple[str, int]]:
        # Renderings of one near-miss phrase share a group: the split keeps
        # them on one side, so a phrase is never in both train and test.
        if f not in phrase_of:
            group: List[tuple[str, int]] = []
            negative_groups.append(group)
            return group
        if phrase_of[f] not in phrase_groups:
            phrase_groups[phrase_of[f]] = []
            negative_groups.append(phrase_groups[phrase_of[f]])
        return phrase_groups[phrase_of[f]]

    for f in neg_files:
        # Each repo numbers its clips from 000000.wav; the source directory
        # keeps two repos from writing the same processed file.
        if config.window_s <= 0:
            dst = proc_neg_dir / f"{f.parent.name}_{f.name}"
            if preprocess_audio(f, dst, sr=config.sample_rate, vad_trim=config.vad_trim):
                processed_negatives.append(dst)
                _group_for(f).append((str(dst), 0))
            continue
        try:
            wav = _load_mono(f, config.sample_rate)
        except Exception as e:
            logger.warning("Error preprocessing %s: %s", f, e)
            continue
        group = _group_for(f)
        for i, win in enumerate(negative_windows(wav, config.sample_rate, config.window_s,
                                                 config.neg_windows_per_clip, rng)):
            dst = proc_neg_dir / f"{f.parent.name}_{f.stem}_w{i}.wav"
            _write_window(win, dst, config.sample_rate)
            processed_negatives.append(dst)
            group.append((str(dst), 0))

    # Silence and near-silence negatives: zeros, -60 dBFS and -40 dBFS white
    # noise, silence_windows of each, so that a fresh buffer on a muted or
    # quiet channel is in-distribution. Written at their level, not normalised.
    n_silence: dict[str, int] = {}
    if config.window_s > 0 and config.silence_windows > 0:
        silence_dir = negatives_dir / "silence"
        for name, win in silence_windows(config.sample_rate, config.window_s,
                                         config.silence_windows, config.seed):
            i = n_silence.get(name, 0)
            dst = silence_dir / f"{name}_{i:04d}.wav"
            _write_window(win, dst, config.sample_rate, normalize=False)
            processed_negatives.append(dst)
            negative_groups.append([(str(dst), 0)])
            n_silence[name] = i + 1
        logger.info("Silence negatives: %s", n_silence)

    # ── Stage 5: Train/test split + metadata CSVs ───────────────────────
    if not processed_positives or not processed_negatives:
        raise RuntimeError(
            f"datagen produced {len(processed_positives)} positives and "
            f"{len(processed_negatives)} negatives; a training set needs both. "
            "Check the download errors above (network, disk space, HF_HOME)."
        )
    logger.info("Stage 5: Splitting into train/test sets")

    # Split each label separately. A single shuffled cut puts one class in the
    # test set whenever the set is small: with one positive and one negative it
    # does so every time, and the guard above then reports the dataset as
    # complete. Partitioning first means both files hold both labels whenever
    # the input does. Windows cut from one negative clip stay on one side.
    positive_entries: List[tuple[str, int]] = [(str(p), 1) for p in processed_positives]
    negative_entries: List[tuple[str, int]] = [e for g in negative_groups for e in g]

    train_positive, test_positive = split_groups([[e] for e in positive_entries], config.test_split)
    train_negative, test_negative = split_groups(negative_groups, config.test_split)

    train_entries = train_positive + train_negative
    test_entries = test_positive + test_negative
    random.shuffle(train_entries)
    random.shuffle(test_entries)

    if not train_positive or not train_negative:
        raise RuntimeError(
            f"the train split holds {len(train_positive)} positives and "
            f"{len(train_negative)} negatives; a model cannot be trained on one "
            "class. Raise n_positive or max_negative."
        )
    if not test_positive or not test_negative:
        raise RuntimeError(
            f"the test split holds {len(test_positive)} positives and "
            f"{len(test_negative)} negatives; a single-class test set makes "
            "every reported score meaningless, so this is refused rather than "
            "written. Each label needs at least 2 clips: this run had "
            f"{len(positive_entries)} positives and {len(negative_entries)} "
            "negatives. Raise n_positive or max_negative."
        )

    train_csv = train_dir / "metadata.csv"
    test_csv = test_dir / "metadata.csv"
    write_metadata_csv(train_csv, train_entries)
    write_metadata_csv(test_csv, test_entries)

    # ── Stage 6: Write config for reproducibility ───────────────────────
    config_path = out / "datagen_config.json"
    config_dict = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in asdict(config).items()
    }
    config_dict["wake_word_normalized"] = ww_key
    config_dict["n_positive_actual"] = len(processed_positives)
    config_dict["n_negative_actual"] = len(processed_negatives)
    config_dict["n_silence_windows"] = n_silence
    config_dict["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    result = DatagenResult(
        train_csv=train_csv,
        test_csv=test_csv,
        positives_dir=positives_dir,
        negatives_dir=negatives_dir,
        bg_noise_dir=bg_noise_dir if config.download_augmentation else None,
        music_dir=music_dir if config.download_augmentation else None,
        rir_dir=rir_dir if config.download_augmentation else None,
        n_positive=len(processed_positives),
        n_negative=len(processed_negatives),
        config_path=config_path,
    )

    # Log summary
    logger.info("Dataset ready. Positives: %d  Negatives: %d",
                result.n_positive, result.n_negative)
    if config.download_augmentation:
        _count = lambda d: sum(1 for _ in d.rglob("*.wav")) if d.exists() else 0
        logger.info("Augmentation data: bg_noise/ (%d files), music/ (%d files), rir/ (%d files)",
                    _count(bg_noise_dir), _count(music_dir), _count(rir_dir))
    logger.info("Train: %s (%d samples)", train_csv, len(train_entries))
    logger.info("Test:  %s (%d samples)", test_csv, len(test_entries))
    logger.info("To train:\n  %s", result.suggested_train_command(config.wake_word))

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def cli_main() -> None:
    """CLI entry point for ``ww_trainer-datagen``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="ww_trainer-datagen",
        description="Generate a complete wake-word dataset from a single command.",
    )
    parser.add_argument(
        "--wake-word", required=True, help="Wake word string (e.g. 'hey jarvis')"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Output directory"
    )
    parser.add_argument(
        "--n-positive", type=int, default=1000, help="Target positive samples (default: 1000)"
    )
    parser.add_argument(
        "--max-negative", type=int, default=None, help="Max general negative samples"
    )
    parser.add_argument(
        "--hf-config", nargs="+", default=None, metavar="CONFIG",
        help="Configs to read from the positive Parquet dataset repo, e.g. default omnivoice "
             "(default: every config the repo declares); negatives always read all of theirs",
    )
    parser.add_argument(
        "--lang", default="en", help="Language for TTS (default: en)"
    )
    parser.add_argument(
        "--test-split", type=float, default=0.2, help="Test set fraction (default: 0.2)"
    )
    parser.add_argument("--window-s", type=float, default=1.5,
                        help="Length of every training example in seconds (0 keeps clips as they are)")
    parser.add_argument("--neg-windows-per-clip", type=int, default=3,
                        help="Windows cut from each long negative clip")
    parser.add_argument("--silence-windows", type=int, default=300,
                        help="Windows per silence class (zeros, -60 dBFS, -40 dBFS white noise); 0 disables")
    parser.add_argument("--adversarial-file", default=None,
                        help="Text file of near-miss phrases to synthesise as hard negatives, one per line")
    parser.add_argument("--allow-positive-skips", action="store_true",
                        help="Continue when more than 20%% of positives are skipped for exceeding the window")
    parser.add_argument("--adversarial-per-text", type=int, default=3,
                        help="Renderings per near-miss phrase")
    parser.add_argument(
        "--no-vad", action="store_true", help="Disable VAD trimming"
    )
    parser.add_argument(
        "--vc-refs", type=str, default=None,
        help="Reference voices dir for voice conversion (voiceclonnx)"
    )
    parser.add_argument(
        "--vc-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for VC model (default: auto)",
    )
    parser.add_argument(
        "--adversarial", action="store_true", help="Generate adversarial hard-negatives"
    )
    parser.add_argument(
        "--adversarial-n", type=int, default=20, help="Adversarial phrases count (default: 20)"
    )
    parser.add_argument(
        "--llm-url", type=str, default=None, help="Ollama URL for LLM adversarial gen"
    )
    parser.add_argument(
        "--llm-model", default="gemma3:4b", help="LLM model name (default: gemma3:4b)"
    )
    parser.add_argument(
        "--tts-config", default=None,
        help="JSON file of {plugin name: config} for the TTS plugins: a remote server's host and voice list, "
             "or a plugin's own settings. Without a host, ovos-tts-plugin-server is not used.",
    )
    parser.add_argument(
        "--no-augmentation-data",
        action="store_true",
        help="Skip downloading bg_noise/music/rir",
    )
    parser.add_argument(
        "--pre-augment",
        action="store_true",
        help="Run offline augmentation pass (noise/reverb/pitch)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()
    if not args.wake_word.strip():
        parser.error("--wake-word must not be empty or whitespace")

    cfg = DatagenConfig(
        wake_word=args.wake_word,
        output_dir=args.output_dir,
        n_positive=args.n_positive,
        max_negative=args.max_negative,
        hf_config=args.hf_config,
        lang=args.lang,
        sample_rate=16000,
        vad_trim=not args.no_vad,
        window_s=args.window_s,
        neg_windows_per_clip=args.neg_windows_per_clip,
        silence_windows=args.silence_windows,
        adversarial_file=args.adversarial_file,
        adversarial_per_text=args.adversarial_per_text,
        allow_positive_skips=args.allow_positive_skips,
        test_split=args.test_split,
        vc_refs_dir=args.vc_refs,
        vc_device=args.vc_device,
        adversarial=args.adversarial,
        adversarial_n=args.adversarial_n,
        llm_url=args.llm_url,
        llm_model=args.llm_model,
        tts_config=args.tts_config,
        download_augmentation=not args.no_augmentation_data,
        pre_augment=args.pre_augment,
        seed=args.seed,
    )

    run_datagen_pipeline(cfg)


if __name__ == "__main__":
    cli_main()
