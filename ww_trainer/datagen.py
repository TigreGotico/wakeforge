"""Synthetic data pipeline for wake-word training.

Single-command UX: provide a wake word string, get a complete dataset
ready for ``ww_trainer-train`` — including properly organised augmentation
data for train-time use.

Entry point: ``ww_trainer-datagen`` (see :func:`cli_main`).
"""

from __future__ import annotations

import csv
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

NEGATIVE_DATASETS: Dict[str, List[str]] = {
    # Non-speech environmental sounds (label=0)
    "general": [
        "TigreGotico/ESC-50",
        "TigreGotico/NAR",
        "agkphysics/AudioSet",          # large general audio; capped by max_negative
    ],
    # Human speech that is NOT the wake word — critical for preventing
    # models from learning "speech vs silence" instead of the specific phrase.
    # These are downloaded alongside "general" negatives and mixed in.
    "speech": [
        "TigreGotico/not-wake-words-speech-en",        # primary: curated NWW speech
        "hf-internal-testing/librispeech_asr_demo",    # ~70 clips, LibriSpeech clean
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


def _load_vad_engine() -> Any:
    """Load OVOS VAD plugin (silero by default).

    Returns an object with ``is_silence(chunk_bytes) -> bool``.
    """
    from ovos_plugin_manager.vad import OVOSVADFactory

    return OVOSVADFactory.create({"module": "ovos-vad-plugin-silero"})


def trim_silence_vad(wav: np.ndarray, sr: int, frame_ms: int = 30) -> np.ndarray:
    """Trim leading/trailing silence using OVOS VAD (silero)."""
    vad = _load_vad_engine()
    frame_len = int(sr * frame_ms / 1000)
    pcm = (wav * 32767).astype(np.int16).tobytes()
    voiced: List[bool] = []
    for i in range(0, len(pcm), frame_len * 2):
        frame = pcm[i: i + frame_len * 2]
        if len(frame) < frame_len * 2:
            break
        try:
            # is_silence returns True for silence — invert for voiced detection
            voiced.append(not vad.is_silence(frame))
        except Exception:
            voiced.append(False)
    if not any(voiced):
        return wav
    idx = np.where(voiced)[0]
    start = max(0, idx[0] * frame_len)
    end = min(len(wav), (idx[-1] + 1) * frame_len)
    return wav[start:end]


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


# ---------------------------------------------------------------------------
# HuggingFace dataset download
# ---------------------------------------------------------------------------


def download_hf_audio_dataset(
    dataset_id: str,
    output_dir: Path,
    max_samples: Optional[int] = None,
    sr: int = 16000,
) -> List[Path]:
    """Download a HuggingFace audio dataset and save WAVs to *output_dir*.

    Already-downloaded WAV files are reused without any network access.
    The HuggingFace Arrow cache (``~/.cache/huggingface/datasets``) is used on
    the first download so that subsequent calls with the same dataset skip the
    HTTP transfer.  Streaming mode (which bypasses the cache) is only used as a
    last resort when the cached download fails.

    Returns list of written file paths.
    """
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Skip if already downloaded ──────────────────────────────────────────
    existing = sorted(output_dir.glob("*.wav"))
    need = max_samples if max_samples is not None else 1
    if len(existing) >= need:
        logger.info("Reusing %d cached WAVs from %s (skipping download)", len(existing), output_dir)
        return existing[:max_samples] if max_samples is not None else existing

    logger.info("Downloading %s → %s (max=%s)", dataset_id, output_dir, max_samples)

    # ── Prefer cached (non-streaming) download ──────────────────────────────
    # Non-streaming stores Arrow files in ~/.cache/huggingface/datasets so
    # re-runs with the same dataset_id are instant.  Fall back to streaming
    # only if the non-streaming load fails (e.g. dataset too large for RAM).
    # Known huge datasets (STREAMING_ONLY_DATASETS) skip the non-streaming
    # attempt entirely — load_dataset(streaming=False) would download the
    # full upstream (hundreds of GB / TB) before iteration starts.
    # Note: trust_remote_code was removed in datasets ≥ 3.x; omit it.
    ds = None
    modes = (True,) if dataset_id in STREAMING_ONLY_DATASETS else (False, True)
    for streaming in modes:
        try:
            ds = load_dataset(dataset_id, split="train", streaming=streaming)
            break
        except Exception as exc:
            if not streaming:
                logger.warning(
                    "Cached download of %s failed (%s); retrying with streaming", dataset_id, exc
                )
            else:
                logger.error("Streaming download of %s also failed: %s", dataset_id, exc)
                return []

    if ds is None:
        return []

    written: List[Path] = []
    audio_col = None
    for i, example in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break

        # Auto-detect audio column on first row
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
                return written

        audio = example[audio_col]
        if hasattr(audio, "get_all_samples"):
            # datasets ≥ 3.x with torchcodec backend: AudioDecoder object
            samples = audio.get_all_samples()
            # samples.data shape: (channels, num_samples) — mix down to mono 1D
            arr = samples.data.float().mean(dim=0).numpy().astype(np.float32)
            orig_sr = int(samples.sample_rate)
        elif isinstance(audio, dict):
            # datasets < 3.x: {"array": np.ndarray, "sampling_rate": int}
            arr = np.array(audio["array"], dtype=np.float32)
            orig_sr = audio.get("sampling_rate", sr)
        else:
            continue

        if orig_sr != sr:
            arr = torchaudio.functional.resample(
                torch.tensor(arr), orig_sr, sr
            ).numpy()

        fname = output_dir / f"{i:06d}.wav"
        _save_audio(str(fname), torch.tensor(arr).unsqueeze(0).float(), sr)
        written.append(fname)

    logger.info("  Downloaded %d files from %s", len(written), dataset_id)
    return written


# ---------------------------------------------------------------------------
# TTS synthesis (adapted from 02_tts_synth.py)
# ---------------------------------------------------------------------------


def _collect_tts_plugins(lang: str) -> List[tuple[str, Any]]:
    """Discover all installed OVOS TTS plugins via OPM and return ``(name, instance)`` pairs.

    Uses ``ovos_plugin_manager.tts.find_tts_plugins()`` for entry-point
    discovery, so any installed OVOS TTS plugin (edge-tts, google-tx,
    phoonnx, piper, …) is automatically picked up.
    """
    from ovos_plugin_manager.tts import find_tts_plugins

    plugins = find_tts_plugins()  # Dict[str, Type[TTS]]
    instances: List[tuple[str, Any]] = []
    for name, plugin_cls in plugins.items():
        try:
            instance = plugin_cls(config={"lang": lang})
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


def synthesize_positives(
    wake_word: str,
    output_dir: Path,
    n: int = 1000,
    lang: str = "en",
) -> List[Path]:
    """Synthesize *n* positive samples using available TTS plugins.

    Returns list of written WAV paths.

    Raises:
        RuntimeError: If no TTS plugins are installed.
    """
    tts_instances = _collect_tts_plugins(lang)

    output_dir.mkdir(parents=True, exist_ok=True)
    text = wake_word.replace("_", " ").replace("-", " ")
    written: List[Path] = []

    for i in range(n):
        name, plugin = random.choice(tts_instances)
        out_path = output_dir / f"{uuid4().hex[:12]}.wav"
        try:
            plugin.get_tts(text, str(out_path), lang=lang)
            if out_path.exists():
                written.append(out_path)
        except Exception:
            continue

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
) -> List[Path]:
    """Apply Chatterbox VC to all WAVs in *input_dir* using random reference voices.

    Each source WAV is converted using a randomly chosen reference voice.
    If *n_target* > number of source files, sources are reused with different
    references.

    Returns list of output WAV paths.
    """
    try:
        from chatterbox_onnx import ChatterboxOnnx
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Voice conversion dependency missing: {e.name!r}. "
            "Install a VC backend extra:\n"
            "    uv pip install -e \".[vc-onnx]\"   # CPU ONNX, recommended\n"
            "    uv pip install -e \".[vc-torch]\"  # GPU PyTorch backend\n"
            "Or run datagen without --vc-refs to skip voice conversion."
        ) from e

    ref_voices = sorted(vc_refs_dir.rglob("*.wav"))
    if not ref_voices:
        logger.warning("No reference voices found in %s", vc_refs_dir)
        return []

    sources = sorted(input_dir.rglob("*.wav"))
    if not sources:
        logger.warning("No source WAVs in %s", input_dir)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    vc_model = ChatterboxOnnx(device=device)

    if n_target is None:
        n_target = len(sources)

    written: List[Path] = []
    for i in range(n_target):
        src = sources[i % len(sources)]
        ref = random.choice(ref_voices)
        out_path = output_dir / f"vc_{uuid4().hex[:12]}.wav"
        try:
            vc_model.voice_convert(
                source_audio_path=str(src),
                target_voice_path=str(ref),
                output_file_name=str(out_path),
            )
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
    lang: str = "en"
    sample_rate: int = 16000
    vad_trim: bool = True
    test_split: float = 0.2
    # Voice cloning
    vc_refs_dir: Optional[str] = None
    vc_device: str = "auto"
    # Adversarial
    adversarial: bool = False
    adversarial_n: int = 20
    llm_url: Optional[str] = None
    llm_model: str = "gemma3:4b"
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
            hf_dataset, positives_dir, max_samples=config.n_positive, sr=config.sample_rate
        )
    else:
        logger.info("  Unknown wake word — synthesizing via TTS")
        pos_files = synthesize_positives(
            ww_key, positives_dir, n=config.n_positive, lang=config.lang
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
    for category in ("general", "speech"):
        for ds_id in NEGATIVE_DATASETS.get(category, []):
            ds_name = ds_id.split("/")[-1]
            ds_out = negatives_dir / ds_name
            # Respect user's max_negative, but enforce a hard cap for huge datasets.
            hard_cap = _LARGE_DATASETS.get(ds_id)
            if hard_cap is not None:
                cap = min(config.max_negative, hard_cap) if config.max_negative else hard_cap
            else:
                cap = config.max_negative
            files = download_hf_audio_dataset(
                ds_id, ds_out, max_samples=cap, sr=config.sample_rate
            )
            neg_files.extend(files)

    # Augmentation resources (NOT in CSV)
    if config.download_augmentation:
        for ds_id in NEGATIVE_DATASETS["bg_noise"]:
            ds_name = ds_id.split("/")[-1]
            download_hf_audio_dataset(ds_id, bg_noise_dir / ds_name, sr=config.sample_rate)

        for ds_id in NEGATIVE_DATASETS["music"]:
            ds_name = ds_id.split("/")[-1]
            download_hf_audio_dataset(ds_id, music_dir / ds_name, sr=config.sample_rate)

        for ds_id in NEGATIVE_DATASETS["rir"]:
            ds_name = ds_id.split("/")[-1]
            download_hf_audio_dataset(ds_id, rir_dir / ds_name, sr=config.sample_rate)

    # ── Stage 3: Optional adversarial hard-negatives ────────────────────
    if config.adversarial:
        logger.info("Stage 3: Generating adversarial hard-negatives")
        adv_texts = generate_adversarial_texts(
            ww_key,
            n=config.adversarial_n,
            llm_url=config.llm_url,
            llm_model=config.llm_model,
        )
        if adv_texts:
            adv_dir = negatives_dir / "adversarial"
            try:
                adv_files = synthesize_positives(
                    adv_texts[0], adv_dir, n=0, lang=config.lang
                )
                # Synthesize each adversarial text
                for text in adv_texts:
                    files = synthesize_positives(text, adv_dir, n=1, lang=config.lang)
                    neg_files.extend(files)
            except RuntimeError:
                logger.warning("No TTS available for adversarial synthesis")

    # ── Stage 4: Preprocess all audio ───────────────────────────────────
    logger.info("Stage 4: Preprocessing audio files")

    processed_positives: List[Path] = []
    proc_pos_dir = positives_dir / "processed"
    for f in pos_files:
        dst = proc_pos_dir / f.name
        if preprocess_audio(f, dst, sr=config.sample_rate, vad_trim=config.vad_trim):
            processed_positives.append(dst)

    processed_negatives: List[Path] = []
    proc_neg_dir = negatives_dir / "processed"
    for f in neg_files:
        dst = proc_neg_dir / f.name
        if preprocess_audio(f, dst, sr=config.sample_rate, vad_trim=config.vad_trim):
            processed_negatives.append(dst)

    # ── Stage 5: Train/test split + metadata CSVs ───────────────────────
    logger.info("Stage 5: Splitting into train/test sets")

    all_entries: List[tuple[str, int]] = []
    for p in processed_positives:
        all_entries.append((str(p), 1))
    for p in processed_negatives:
        all_entries.append((str(p), 0))

    random.shuffle(all_entries)

    n_test = max(1, int(len(all_entries) * config.test_split))
    test_entries = all_entries[:n_test]
    train_entries = all_entries[n_test:]

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
        "--lang", default="en", help="Language for TTS (default: en)"
    )
    parser.add_argument(
        "--test-split", type=float, default=0.2, help="Test set fraction (default: 0.2)"
    )
    parser.add_argument(
        "--no-vad", action="store_true", help="Disable VAD trimming"
    )
    parser.add_argument(
        "--vc-refs", type=str, default=None, help="Reference voices dir for Chatterbox VC"
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

    cfg = DatagenConfig(
        wake_word=args.wake_word,
        output_dir=args.output_dir,
        n_positive=args.n_positive,
        max_negative=args.max_negative,
        lang=args.lang,
        sample_rate=16000,
        vad_trim=not args.no_vad,
        test_split=args.test_split,
        vc_refs_dir=args.vc_refs,
        vc_device=args.vc_device,
        adversarial=args.adversarial,
        adversarial_n=args.adversarial_n,
        llm_url=args.llm_url,
        llm_model=args.llm_model,
        download_augmentation=not args.no_augmentation_data,
        pre_augment=args.pre_augment,
        seed=args.seed,
    )

    run_datagen_pipeline(cfg)


if __name__ == "__main__":
    cli_main()
