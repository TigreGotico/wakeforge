"""Voice-grid synthetic data: every donor voice at every speaking rate, pitch and text form.

For one phrase and one language the grid is the product of the donor voices, the speaking
rates, the pitches and the text forms ("x", "x!", "x?"). Clips are produced by a bounded
worker pool, written as 16 kHz mono PCM under names derived from the grid cell, and listed in
``manifest.csv`` with every parameter. A rerun skips what exists, so an interrupted run resumes
and a finished one is a no-op.

Optional stages follow the clean synthesis: voice cloning of the clean clips onto reference
speakers (``voiceclonnx`` Chatterbox) and speaker-embedding deduplication (``speakeronnx``).

Entry point: ``ww_trainer-voicegrid`` (see :func:`cli_main`).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import fnmatch
import difflib
import functools
import hashlib
import importlib.util
import io
import itertools
import json
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MIN_SAMPLES = 1600
RATES = (-25, -12, 0, 12, 25)
PITCHES = (-10, 0, 10)
FORMS = {"plain": "", "exclaim": "!", "question": "?", "ellipsis": "..."}
DEFAULT_FORMS = ("plain", "exclaim", "question")
DEFAULT_DEDUP_THRESHOLD = 0.9

#: Benchmark held-out Kokoro speakers, matched across every Kokoro version (``kokoro/``, ``kokoro-v019/``, ``kokoro-v11/``).
HELD_OUT_KOKORO = ("af_heart", "af_kore", "am_echo", "am_puck", "bf_lily", "bm_fable")
#: Edge voices the OVOS edge-tts plugin's tests hold out, with their Multilingual twins.
HELD_OUT_EDGE = ("en-GB-Maisie", "en-IE-Connor", "en-IN-Prabhat", "en-KE-Asilia", "en-NZ-Molly", "en-SG-Wayne")
DEFAULT_EXCLUDE_VOICES = (
    tuple(f"kokoro*/{speaker}" for speaker in HELD_OUT_KOKORO)
    + tuple(f"{voice}{kind}Neural" for voice in HELD_OUT_EDGE for kind in ("", "Multilingual"))
)

#: Edge voices that are one speaker under two names; the first present name of each group is kept.
SAME_SPEAKER_TWINS = (
    ("en-US-AvaNeural", "en-US-AvaMultilingualNeural"),
    ("en-US-BrianNeural", "en-US-BrianMultilingualNeural"),
    ("en-US-EmmaNeural", "en-US-EmmaMultilingualNeural"),
)

MANIFEST_COLUMNS = ("file", "stage", "engine", "voice", "lang", "text", "form", "rate", "pitch", "copy", "label", "style", "score",
                    "threshold", "threshold_source", "source", "reference")
REJECTED_COLUMNS = ("file", "engine", "voice", "text", "form", "rate", "pitch", "style", "reason", "score", "threshold", "threshold_source", "transcript")
MIN_SECONDS = 0.3
MAX_SECONDS = 3.0
TRIM_MARGIN_S = 0.1
MIN_VOICED_RATIO = 0.3
MIN_RMS = 0.005
MIN_REFERENCES = 10
VERIFY_PERCENTILE = 5.0
FALLBACK_THRESHOLD = -0.64
REFERENCE_DONOR = "edge"
OMNIVOICE_STYLES = ("elderly male", "female, british accent", "male, american accent", "young adult female", "auto")
STYLE_FLOOR = 0.10
OMNIVOICE_RATES = (0, 15)
OMNIVOICE_CLIPS = 300
ASR_MODEL = "nemo-parakeet-tdt-0.6b-v3"
ASR_THRESHOLD = 0.8
_ATTEMPTS = 4
_RETRY_WAIT_S = 1.0


# ---------------------------------------------------------------------------
# Donors
# ---------------------------------------------------------------------------


class GridEngine:
    """A donor: a source of voices that can say a text, with whatever rate and pitch control it has.

    ``has_rate`` and ``has_pitch`` say per voice which grid axes the voice honours; an axis it
    ignores collapses to one value so no clip duplicates another. ``max_concurrency`` and
    ``pause_s`` bound how hard the worker pool leans on the donor: a remote service needs pacing,
    a local model needs a limit on simultaneous inferences.
    """

    name = ""
    max_concurrency = 1
    pause_s = 0.0

    def voices(self, lang: str) -> List[str]:
        raise NotImplementedError

    def has_rate(self, voice: str) -> bool:
        return True

    def has_pitch(self, voice: str) -> bool:
        return True

    def synth(self, text: str, voice: str, rate: Optional[int], pitch: Optional[int]) -> Tuple[np.ndarray, int]:
        """Mono float audio and its sample rate; ``rate`` is a percentage and ``pitch`` a shift in Hz, ``None`` when unsupported."""
        raise NotImplementedError


def lang_matches(voice_lang: str, wanted: str) -> bool:
    """Whether a voice tagged ``voice_lang`` serves ``wanted``; a bare language matches every region of it."""
    v = voice_lang.lower().replace("_", "-")
    w = wanted.lower().replace("_", "-")
    if "-" in w:
        return v == w or v == w.split("-")[0]
    return v.split("-")[0] == w


@dataclass(frozen=True)
class CatalogVoice:
    voice_id: str
    lang: str
    engine: str
    index: str


@functools.lru_cache(maxsize=1)
def _phoonnx_catalog() -> Tuple[CatalogVoice, ...]:
    """Every voice phoonnx lists, with the engine that runs it and the voice-index file it comes from."""
    from phoonnx.model_manager import TTSModelManager

    with tempfile.TemporaryDirectory() as tmp:
        manager = TTSModelManager(cache_path=os.path.join(tmp, "voices.json"))
        manager.merge_default_voices()
        known = dict(manager.voices)
        by_source = manager.get_available_voice_ids_by_source()
    out = []
    for index, ids in by_source.items():
        for vid in ids:
            info = known.get(vid)
            if info is not None:
                out.append(CatalogVoice(vid, info.lang, str(getattr(info.engine, "value", info.engine)), index))
    return tuple(out)


def catalog_voices(index: str, patterns: Sequence[str], lang: str) -> List[str]:
    """Ids of the voices of one voice-index file serving ``lang``, narrowed by case-insensitive id globs."""
    pats = [p.lower() for p in patterns]
    return sorted(
        v.voice_id for v in _phoonnx_catalog()
        if v.index == index.lower() and lang_matches(v.lang, lang)
        and (not pats or any(fnmatch.fnmatchcase(v.voice_id.lower(), p) for p in pats))
    )


#: phoonnx engines whose speed parameter multiplies the speaking rate.
_SPEED_ENGINES = frozenset({"styletts2", "supertonic"})
#: phoonnx engines whose length scale divides it.
_LENGTH_ENGINES = frozenset({"piper", "matcha", "coqui", "phoonnx", "omnivoice"})


def realize_links(model_path: Path) -> None:
    """Give a graph and its external-data sidecar real directory entries in place of Hub-cache symlinks.

    The Hub cache stores each file as a symlink into ``blobs/``, and onnxruntime rejects external data
    whose resolved path leaves the graph's directory (``External data path escapes model directory``).
    A hard link is a real entry of the snapshot directory that shares the blob's bytes, so nothing is
    downloaded or duplicated; across filesystems the file is copied instead.
    """
    for entry in model_path.parent.iterdir():
        if ".onnx" in entry.name and entry.is_symlink() and entry.exists():
            real = entry.resolve()
            tmp = entry.with_name(entry.name + ".real")
            try:
                os.link(real, tmp)
            except OSError:
                shutil.copy2(real, tmp)
            os.replace(tmp, entry)


class PhoonnxDonor(GridEngine):
    """The voices of one phoonnx voice-index file, optionally narrowed by id globs, for the requested language.

    Rate is applied through the engine's own control (a speed multiplier or a length scale); an
    engine with neither, and every phoonnx engine for pitch, collapses that axis.
    """

    def __init__(self, name: str, index: str, patterns: Sequence[str] = (), workers: int = 1):
        from phoonnx.model_manager import TTSModelManager
        from phoonnx.voice_cache import VoiceCache

        self.name = name
        self._index = index.lower()
        self._patterns = list(patterns)
        self.max_concurrency = max(1, workers)
        self._tmp = tempfile.TemporaryDirectory()
        self._manager = TTSModelManager(cache_path=os.path.join(self._tmp.name, "voices.json"))
        self._manager.merge_default_voices()

        def resolve(voice_id):
            info = self._manager.get_voice(voice_id)
            if info is None:
                raise KeyError(f"unknown phoonnx voice {voice_id}")
            return info

        def load(info):
            realize_links(info.download_model())
            return info.load()

        self._cache = VoiceCache(resolve=resolve, load=load, max_loaded_voices=self.max_concurrency)

    def _engine_of(self, voice: str) -> str:
        return next((v.engine for v in _phoonnx_catalog() if v.voice_id == voice), "")

    def voices(self, lang: str) -> List[str]:
        return catalog_voices(self._index, self._patterns, lang)

    def has_rate(self, voice: str) -> bool:
        return self._engine_of(voice) in _SPEED_ENGINES | _LENGTH_ENGINES

    def has_pitch(self, voice: str) -> bool:
        return False

    def _config(self, model, engine: str, rate: Optional[int], extra: Optional[dict] = None):
        from phoonnx.config import SynthesisConfig

        extra = dict(extra or {})
        if rate is None:
            return SynthesisConfig(extra_params=extra)
        factor = 1 + rate / 100
        if engine in _SPEED_ENGINES:
            extra["speed"] = float(model.adapter.default_params().get("speed", 1.0)) * factor
            return SynthesisConfig(extra_params=extra)
        return SynthesisConfig(length_scale=float(model.config.length_scale or 1.0) / factor, extra_params=extra)

    def _run(self, voice_id: str, text: str, rate: Optional[int], extra: Optional[dict] = None):
        with self._cache.lease(voice_id) as model:
            chunks = list(model.synthesize(text, self._config(model, self._engine_of(voice_id), rate, extra)))
        if not chunks:
            raise ValueError("no audio")
        return np.concatenate([c.audio_float_array for c in chunks]), chunks[0].sample_rate

    def synth(self, text, voice, rate, pitch):
        return self._run(voice, text, rate)


class OmniVoiceDonor(GridEngine):
    """OmniVoice: one model for hundreds of languages and no fixed speakers.

    Without a reference clip the model invents the voice, so this donor does not enumerate voices.
    :func:`generate_omnivoice` draws one clip per index with its own seed, and so its own random
    speaker, rotating the text forms and speaking rates and a voice-design style that
    :func:`generate_omnivoice` weights by how often each style has been accepted.
    """

    def __init__(self, name: str, macrolanguages: Dict[str, List[str]]):
        self.name = name
        self.macrolanguages = macrolanguages

    def voices(self, lang: str) -> List[str]:
        return []


@dataclass(frozen=True)
class OmniItem:
    """One OmniVoice request: the text, the language code, the speed percentage, the seed and an optional voice design."""

    text: str
    code: str
    rate: int
    seed: int
    instruct: Optional[str] = None


def omnivoice_codes(lang: str, macrolanguages: Dict[str, List[str]]) -> List[str]:
    """OmniVoice language codes serving ``lang``: the code itself and, for a macrolanguage, the varieties the roster lists."""
    primary = lang.lower().replace("_", "-").split("-")[0]
    wanted = {primary, *macrolanguages.get(primary, ())}
    try:
        listed = {v.voice_id.split("/", 1)[1] for v in _phoonnx_catalog() if v.index == "omnivoice"}
    except ImportError:
        return [primary]
    return sorted(wanted & listed)


def clip_seed(seed: int, lang: str, phrase: str, index: int) -> int:
    """The seed of clip ``index``: a function of the run seed, language, phrase and index alone, so a rerun draws the same speakers."""
    return int(hashlib.sha256(f"{seed}:{lang}:{phrase}:{index}".encode()).hexdigest()[:8], 16)


def omnivoice_plan(index: int, phrase: str, label: int, lang: str, codes: Sequence[str], forms: Sequence[str],
                   rates: Sequence[int], seed: int) -> Tuple["Cell", OmniItem]:
    """The cell and request for clip ``index``: forms rotate fastest, then rates, and every clip has its own seed and speaker.

    The voice-design style is not part of the clip's identity: :func:`generate_omnivoice` draws it.
    """
    form = forms[index % len(forms)]
    rate = rates[(index // len(forms)) % len(rates)]
    code = codes[index % len(codes)]
    clip = clip_seed(seed, lang, phrase, index)
    text = phrase + FORMS[form]
    item = OmniItem(text, code, rate, clip)
    return Cell("omnivoice", f"omnivoice/{code}#{clip}", rate, None, form, text, phrase, label), item


class OmniBackend:
    """Runs OmniVoice requests in batches and returns mono float audio with its sample rate."""

    def generate(self, items: Sequence[OmniItem]) -> List[Tuple[np.ndarray, int]]:
        raise NotImplementedError


class TorchOmniBackend(OmniBackend):
    """The official ``omnivoice`` PyTorch package; ``rocm`` is torch's ``cuda`` device on AMD."""

    def __init__(self, device: str):
        import torch
        from omnivoice import OmniVoice

        want = "cuda" if device in ("cuda", "rocm") or (device == "auto" and torch.cuda.is_available()) else "cpu"
        self._torch = torch
        self.model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0" if want == "cuda" else "cpu",
                                               dtype=torch.float16 if want == "cuda" else torch.float32)
        got = next(self.model.parameters()).device.type
        if got != want:
            raise SystemExit(f"OmniVoice requested {want} but the model is on {got}")
        self.device = got

    def generate(self, items):
        out: List[Any] = [None] * len(items)
        for style in dict.fromkeys(i.instruct for i in items):
            group = [k for k, i in enumerate(items) if i.instruct == style]
            self._torch.manual_seed(items[group[0]].seed)
            kwargs = {"instruct": [style] * len(group)} if style else {}
            audio = self.model.generate(text=[items[k].text for k in group], language=[items[k].code for k in group],
                                        speed=[1 + items[k].rate / 100 for k in group], **kwargs)
            for k, a in zip(group, audio):
                out[k] = (a, 24000)
        return out


class PhoonnxOmniBackend(OmniBackend):
    """OmniVoice through phoonnx's ONNX export, one request at a time; the fallback when the official package is absent."""

    def __init__(self, workers: int = 1):
        self._donor = PhoonnxDonor("omnivoice", "omnivoice", (), workers)

    def generate(self, items):
        out = []
        for i in items:
            extra = {"seed": i.seed}
            if i.instruct:
                extra["instruct"] = i.instruct
            out.append(self._donor._run(f"omnivoice/{i.code}", i.text, i.rate, extra))
        return out


def _omnivoice_package() -> bool:
    return importlib.util.find_spec("omnivoice") is not None and importlib.util.find_spec("torch") is not None


def omnivoice_backend(device: str, workers: int = 1) -> OmniBackend:
    """The official PyTorch package when it is importable, else phoonnx's ONNX OmniVoice."""
    return TorchOmniBackend(device) if _omnivoice_package() else PhoonnxOmniBackend(workers)


class EdgeEngine(GridEngine):
    """Microsoft Edge neural voices through ``edge-tts``: the only donor here with a pitch control."""

    name = "edge"
    max_concurrency = 3
    pause_s = 0.3

    def voices(self, lang: str) -> List[str]:
        from ovos_tts_plugin_edge_tts import VOICES

        return sorted(v for loc, vs in VOICES.items() if lang_matches(loc, lang) for v in vs)

    def synth(self, text, voice, rate, pitch):
        import edge_tts

        async def collect():
            buf = io.BytesIO()
            async for chunk in edge_tts.Communicate(text, voice, rate=f"{rate:+d}%", pitch=f"{pitch:+d}Hz").stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            return buf

        buf = asyncio.run(collect())
        buf.seek(0)
        audio, sr = sf.read(buf, dtype="float32", always_2d=True)
        return audio.mean(1), sr


class PluginEngine(GridEngine):
    """Any installed OVOS TTS plugin, through the voices and options :mod:`ww_trainer.datagen` already knows.

    Rate and pitch are passed as ``"+12%"`` and ``"+10Hz"`` to a plugin whose ``get_tts`` accepts a
    ``rate`` or ``pitch`` keyword; the others collapse.
    """

    def __init__(self, name: str, plugin_name: str, lang: str, tts_config: Optional[dict] = None):
        from ww_trainer import datagen

        self.name = name
        self._lang = lang
        self._tts_config = tts_config
        self._plugin_name = plugin_name
        self._plugin = dict(datagen._collect_tts_plugins(lang, tts_config))[plugin_name]
        params = datagen._accepted_options(self._plugin, {"rate": 0, "pitch": 0})
        self._rate = "rate" in params
        self._pitch = "pitch" in params
        self._options: Dict[str, dict] = {}

    def voices(self, lang: str) -> List[str]:
        from ww_trainer import datagen

        options = datagen.voice_options(self._plugin_name, self._plugin, lang, self._tts_config)
        self._options = {(o.get("voice") or o.get("lang") or "default"): {k: v for k, v in o.items() if k != "rate"} for o in options}
        return sorted(self._options)

    def has_rate(self, voice):
        return self._rate

    def has_pitch(self, voice):
        return self._pitch

    def synth(self, text, voice, rate, pitch):
        from ww_trainer import datagen

        opts = {"lang": self._lang, **self._options.get(voice, {})}
        if rate is not None:
            opts["rate"] = f"{rate:+d}%"
        if pitch is not None:
            opts["pitch"] = f"{pitch:+d}Hz"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.audio")
            self._plugin.get_tts(text, path, **datagen._accepted_options(self._plugin, opts))
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
        return audio.mean(1), sr


# ---------------------------------------------------------------------------
# Donor roster
# ---------------------------------------------------------------------------

ROSTER_PATH = Path(__file__).parent / "data" / "donor_roster.json"


def load_roster(path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """The donor roster: ``donors`` (how each named donor is built), ``languages`` (which donors serve a language) and ``fallback``."""
    with open(path or ROSTER_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def donors_for(roster: Dict[str, Any], lang: str) -> List[str]:
    """Donor names the roster gives ``lang``, matched on its primary subtag, else the roster's fallback."""
    primary = lang.lower().replace("_", "-").split("-")[0]
    return list(roster["languages"].get(lang.lower(), roster["languages"].get(primary, roster["fallback"])))


def make_donor(name: str, roster: Dict[str, Any], lang: str, workers: int, tts_config: Optional[dict] = None) -> GridEngine:
    """Build the donor ``name`` from its roster definition."""
    if name not in roster["donors"]:
        raise ValueError(f"unknown donor {name!r}; the roster defines {', '.join(sorted(roster['donors']))}")
    spec = roster["donors"][name]
    kind = spec["kind"]
    if kind == "phoonnx":
        return PhoonnxDonor(name, spec["index"], spec.get("voices", ()), workers)
    if kind == "omnivoice":
        return OmniVoiceDonor(name, roster.get("macrolanguages", {}))
    if kind == "edge":
        return EdgeEngine()
    if kind == "plugin":
        return PluginEngine(name, spec["plugin"], lang, tts_config)
    raise ValueError(f"donor {name!r} has unknown kind {kind!r}")


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One clip of the grid. ``rate`` and ``pitch`` are ``None`` where the engine has no such control."""

    engine: str
    voice: str
    rate: Optional[int]
    pitch: Optional[int]
    form: str
    text: str
    phrase: str = ""
    label: int = 1

    @property
    def filename(self) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", self.voice).strip("-")
        rate = "na" if self.rate is None else f"{self.rate:+d}"
        pitch = "na" if self.pitch is None else f"{self.pitch:+d}"
        engine = re.sub(r"[^A-Za-z0-9._-]+", "-", self.engine)
        slug = re.sub(r"[^A-Za-z0-9]+", "-", self.phrase.lower()).strip("-")[:30]
        tag = hashlib.sha256(f"{self.phrase}:{self.label}".encode()).hexdigest()[:6]
        return f"{slug}-{tag}__{engine}__{safe}__r{rate}__p{pitch}__{self.form}__c0.wav"


def exclusions(patterns: Iterable[str]) -> Callable[[str], bool]:
    """A predicate true for a voice id matching any of the case-insensitive glob ``patterns``."""
    pats = [p.lower() for p in patterns]
    return lambda voice: any(fnmatch.fnmatchcase(voice.lower(), p) for p in pats)


def drop_twins(voices: Sequence[str]) -> List[str]:
    """``voices`` with one name of each same-speaker pair removed: the first name of a pair that is present stays."""
    present = set(voices)
    dropped = set()
    for group in SAME_SPEAKER_TWINS:
        here = [v for v in group if v in present]
        dropped.update(here[1:])
    return [v for v in voices if v not in dropped]


def select_voices(engine: GridEngine, lang: str, exclude: Iterable[str], max_voices: Optional[int], seed: int) -> List[str]:
    """The donor voices an engine contributes: its language voices minus exclusions and twins, optionally a seeded sample."""
    skip = exclusions(exclude)
    voices = drop_twins([v for v in engine.voices(lang) if not skip(v)])
    if max_voices is not None and len(voices) > max_voices:
        voices = sorted(random.Random(f"{seed}:{engine.name}").sample(voices, max_voices))
    return voices


def enumerate_grid(
    engines: Sequence[GridEngine],
    voices: Dict[str, List[str]],
    phrase: str,
    rates: Sequence[int] = RATES,
    pitches: Sequence[int] = PITCHES,
    forms: Sequence[str] = DEFAULT_FORMS,
    max_clean_per_voice: Optional[int] = None,
    seed: int = 0,
    label: int = 1,
) -> List[Cell]:
    """Every cell of the grid in a stable order; ``max_clean_per_voice`` keeps a seeded subset of each voice's cells."""
    cells: List[Cell] = []
    for engine in engines:
        for voice in voices[engine.name]:
            axes = itertools.product(
                rates if engine.has_rate(voice) else (None,),
                pitches if engine.has_pitch(voice) else (None,),
                forms,
            )
            own = [Cell(engine.name, voice, r, p, f, phrase + FORMS[f], phrase, label) for r, p, f in axes]
            if max_clean_per_voice is not None and len(own) > max_clean_per_voice:
                keep = set(random.Random(f"{seed}:{engine.name}:{voice}").sample(range(len(own)), max_clean_per_voice))
                own = [c for i, c in enumerate(own) if i in keep]
            cells.extend(own)
    return cells


# ---------------------------------------------------------------------------
# Audio and manifest
# ---------------------------------------------------------------------------


def to_16k_mono(audio: np.ndarray, sr: int) -> np.ndarray:
    """Float32 mono at 16 kHz; raises on audio too short or non-finite to be a clip."""
    import librosa

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio, got shape {audio.shape}")
    if not np.isfinite(audio).all():
        raise ValueError("audio is not finite")
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE, res_type="soxr_hq")
    if len(audio) < MIN_SAMPLES:
        raise ValueError("audio too short")
    return np.clip(audio, -1.0, 1.0)


def write_wav(path: Path, audio: np.ndarray) -> None:
    """16 kHz mono PCM, written under a temporary name and moved into place so a partial file never has a final name."""
    tmp = path.with_name(path.name + ".part")
    sf.write(tmp, audio, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    os.replace(tmp, path)


def _write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in MANIFEST_COLUMNS})
    os.replace(tmp, path)


def _clean_row(cell: Cell, lang: str) -> Dict[str, Any]:
    return {"file": cell.filename, "stage": "tts", "engine": cell.engine, "voice": cell.voice, "lang": lang,
            "text": cell.text, "form": cell.form, "rate": cell.rate, "pitch": cell.pitch, "copy": 0, "label": cell.label}


def describe_error(error: BaseException) -> str:
    """The error and every error it was raised from, naming the missing module of an ImportError.

    A library that catches an ImportError and raises its own message (``misaki is required``) hides
    which module was actually missing; the chain shows it.
    """
    parts, seen, current = [], set(), error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__}: {current}"
        if isinstance(current, ImportError) and current.name:
            text += f" (module {current.name})"
        parts.append(text)
        current = current.__cause__ or current.__context__
    return " <- ".join(parts)


def _run_pool(tasks: Sequence[Any], work: Callable[[Any], bool], workers: int) -> Tuple[int, int]:
    """Run ``work`` over ``tasks`` on ``workers`` threads, in order; returns (succeeded, failed)."""
    if not tasks:
        return 0, 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(work, tasks))
    return sum(results), len(results) - sum(results)


def _summary(stage: str, **counts: Any) -> None:
    print("VOICEGRID", f"stage={stage}", *(f"{k}={v}" for k, v in counts.items()), flush=True)


# ---------------------------------------------------------------------------
# Stage 1: synthesis
# ---------------------------------------------------------------------------


def normalize_transcript(text: str) -> str:
    """Lower case, accents folded, punctuation dropped, whitespace collapsed."""
    folded = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w\s]", " ", folded).split())


def transcript_match(transcript: str, phrase: str, spellings: Sequence[str] = ()) -> float:
    """Best similarity (0 to 1) of the normalised transcript to the phrase or to one of its alternative spellings."""
    heard = normalize_transcript(transcript)
    return max((difflib.SequenceMatcher(None, heard, normalize_transcript(t)).ratio() for t in (phrase, *spellings)), default=0.0)


def _load_asr(model: str) -> Callable[[str], str]:
    import onnx_asr

    recognize = onnx_asr.load_model(model, providers=["CPUExecutionProvider"]).recognize
    return lambda path: str(recognize(path))


@dataclass(frozen=True)
class Alignment:
    """A forced alignment of a phrase: the mean per-frame log-probability of its path and where it starts and ends, in seconds."""

    score: float
    start: float
    end: float


def ctc_force_align(logp: np.ndarray, tokens: Sequence[int], blank: int = 0) -> Optional[Tuple[float, int, int]]:
    """Viterbi forced alignment of ``tokens`` to CTC log-probabilities ``logp`` of shape (frames, classes).

    Returns the mean log-probability along the best path between its first and last token frame, and those
    two frame indexes, or ``None`` when there are too few frames for the tokens.
    """
    frames, n = logp.shape[0], len(tokens)
    if n == 0 or frames < n:
        return None
    labels = np.full(2 * n + 1, blank)
    labels[1::2] = tokens
    states = len(labels)
    skip = np.zeros(states, dtype=bool)
    skip[3:] = labels[3:] != labels[1:-2]
    skip &= labels != blank
    score = np.full((frames, states), -np.inf)
    back = np.zeros((frames, states), dtype=np.int8)
    score[0, 0] = logp[0, blank]
    score[0, 1] = logp[0, labels[1]]
    for t in range(1, frames):
        stay = score[t - 1]
        prev1 = np.concatenate(([-np.inf], score[t - 1, :-1]))
        prev2 = np.concatenate(([-np.inf, -np.inf], score[t - 1, :-2]))
        prev2 = np.where(skip, prev2, -np.inf)
        stacked = np.stack([stay, prev1, prev2])
        pick = stacked.argmax(0)
        score[t] = stacked[pick, np.arange(states)] + logp[t, labels]
        back[t] = pick
    state = states - 1 if score[-1, -1] >= score[-1, -2] else states - 2
    if not np.isfinite(score[-1, state]):
        return None
    path = np.zeros(frames, dtype=int)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state -= back[t, state]
    inside = np.flatnonzero(path % 2 == 1)
    first, last = int(inside[0]), int(inside[-1])
    picked = logp[np.arange(first, last + 1), labels[path[first:last + 1]]]
    return float(picked.mean()), first, last


class MmsAligner:
    """CTC forced alignment with torchaudio's MMS_FA bundle (about 1,100 languages), text romanised with uroman."""

    def __init__(self, device: str = "cpu"):
        import torch
        from torchaudio.pipelines import MMS_FA

        self._torch = torch
        self.device = "cuda" if device in ("cuda", "rocm") else "cpu"
        self.model = MMS_FA.get_model().to(self.device).eval()
        self.dictionary = MMS_FA.get_dict()
        try:
            from uroman import Uroman

            self._roman = Uroman().romanize_string
        except ImportError:
            logger.warning("uroman is not installed; non-Latin phrases cannot be aligned")
            self._roman = lambda text: "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

    def tokens(self, phrase: str) -> List[int]:
        text = re.sub(r"[^a-z' ]", "", self._roman(phrase).lower())
        return [self.dictionary[c] for c in text if c in self.dictionary and c not in ("-", "*")]

    def align(self, wav: np.ndarray, phrase: str) -> Optional[Alignment]:
        tokens = self.tokens(phrase)
        with self._torch.inference_mode():
            emission, _ = self.model(self._torch.from_numpy(wav)[None].to(self.device))
            logp = self._torch.log_softmax(emission[0], -1).cpu().numpy()
        found = ctc_force_align(logp, tokens)
        if found is None:
            return None
        score, first, last = found
        stride = len(wav) / SAMPLE_RATE / logp.shape[0]
        return Alignment(score, first * stride, (last + 1) * stride)


def voiced_ratio(wav: np.ndarray) -> float:
    """Share of 20 ms frames whose RMS is at least a twentieth of the loudest frame's."""
    size = SAMPLE_RATE // 50
    frames = wav[: len(wav) // size * size].reshape(-1, size)
    rms = np.sqrt((frames ** 2).mean(1)) if len(frames) else np.zeros(1)
    return float((rms >= 0.05 * rms.max()).mean()) if rms.max() > 0 else 0.0


@dataclass
class Verdict:
    accepted: bool
    reason: str = ""
    score: Optional[float] = None
    transcript: str = ""
    audio: Optional[np.ndarray] = None
    threshold: Optional[float] = None
    source: str = ""


class Verifier:
    """Decides whether a synthesized clip says its phrase.

    ``align`` scores the clip by forced alignment and trims it to the phrase plus a margin; ``asr`` compares
    an ASR transcript to the phrase or its alternative spellings; ``both`` requires both. Every mode also
    rejects a clip outside ``MIN_SECONDS`` to ``MAX_SECONDS`` or with too little voiced energy. The alignment
    threshold is kept per phrase, set by :func:`reference_threshold` from edge-tts clips of that phrase or a fixed
    fallback; ``require_asr`` makes ``align`` also demand the transcript match.
    """

    def __init__(self, mode: str = "align", aligner: Optional[Any] = None, transcribe: Optional[Callable[[str], str]] = None,
                 spellings: Sequence[str] = (), asr_threshold: float = ASR_THRESHOLD):
        self.mode = mode
        self.aligner = aligner
        self.transcribe = transcribe
        self.spellings = list(spellings)
        self.asr_threshold = asr_threshold
        self.thresholds: Dict[str, Tuple[float, str]] = {}
        self.default: Optional[Tuple[float, str]] = None
        self._lock = threading.Lock()

    def score(self, wav: np.ndarray, phrase: str) -> Optional[Alignment]:
        with self._lock:
            return self.aligner.align(wav, phrase)

    def set_threshold(self, phrase: str, scores: Sequence[float], percentile: float, source: str) -> float:
        value = float(np.percentile(scores, percentile))
        self.thresholds[phrase] = (value, source)
        return value

    def threshold_for(self, phrase: str) -> Optional[Tuple[float, str]]:
        return self.thresholds.get(phrase, self.default)

    def check(self, wav: np.ndarray, phrase: str) -> Verdict:
        alignment, audio, score, limit = None, wav, None, None
        if self.mode in ("align", "both"):
            alignment = self.score(wav, phrase)
            if alignment is None:
                return Verdict(False, "align-failed")
            limit = self.threshold_for(phrase)
            score = alignment.score
            lo = max(0, int((alignment.start - TRIM_MARGIN_S) * SAMPLE_RATE))
            audio = wav[lo:int((alignment.end + TRIM_MARGIN_S) * SAMPLE_RATE)]
        reason = ""
        if not MIN_SECONDS <= len(audio) / SAMPLE_RATE <= MAX_SECONDS:
            reason = "duration"
        elif voiced_ratio(audio) < MIN_VOICED_RATIO or float(np.sqrt((audio ** 2).mean())) < MIN_RMS:
            reason = "energy"
        elif alignment is not None and (limit is None or score < limit[0]):
            reason = "score"
        heard = ""
        if not reason and self.mode in ("asr", "both"):
            with self._lock, tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "clip.wav")
                sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
                heard = self.transcribe(path)
            if transcript_match(heard, phrase, self.spellings) < self.asr_threshold:
                reason = "asr"
        return Verdict(not reason, reason, score, heard, audio, limit[0] if limit else None, limit[1] if limit else "")


class ClipSink:
    """Where a synthesized clip lands: ``clips/`` when accepted, ``rejected/`` when the verifier refuses it.

    A rejected clip is kept with its reason, score and transcript in ``manifest_rejected.csv`` and counts as
    attempted, so a rerun does not generate it again. Verification applies to the donors named in ``verified``.
    Scores are kept in ``verify_scores.csv`` so a rerun does not score a clip twice.
    """

    def __init__(self, out: Path, verifier: Optional[Verifier] = None, verified: Iterable[str] = ()):
        self.clips = out / "clips"
        self.rejected = out / "rejected"
        self.manifest = out / "manifest_rejected.csv"
        self.score_file = out / "verify_scores.csv"
        self.style_file = out / "omnivoice_styles.csv"
        self.styles: Dict[str, Tuple[str, bool]] = {}
        self.verifier = verifier
        self.verified = set(verified)
        self.rejected_count = 0
        self.scores: Dict[str, Tuple[float, Optional[float], str]] = {}
        self._lock = threading.Lock()
        self.clips.mkdir(parents=True, exist_ok=True)
        if self.style_file.exists():
            with open(self.style_file, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    self.styles[row["file"]] = (row["style"], row["accepted"] == "1")
        if self.score_file.exists():
            with open(self.score_file, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    self.scores[row["file"]] = (float(row["score"]), float(row["threshold"]) if row["threshold"] else None,
                                                row["threshold_source"])

    def record_style(self, name: str, style: str, accepted: bool) -> None:
        with self._lock:
            fresh = not self.style_file.exists()
            with open(self.style_file, "a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if fresh:
                    writer.writerow(("file", "style", "accepted"))
                writer.writerow((name, style, int(accepted)))
            self.styles[name] = (style, accepted)

    def style_counts(self) -> Dict[str, Tuple[int, int]]:
        """Accepted and attempted clips per voice-design style, over this run and earlier ones."""
        counts: Dict[str, List[int]] = {}
        for style, accepted in self.styles.values():
            entry = counts.setdefault(style, [0, 0])
            entry[0] += accepted
            entry[1] += 1
        return {k: (v[0], v[1]) for k, v in counts.items()}

    def known(self, cell: Cell) -> bool:
        return (self.clips / cell.filename).exists() or (self.rejected / cell.filename).exists()

    def remember(self, name: str, score: float, threshold: Optional[float], source: str = "") -> None:
        with self._lock:
            fresh = not self.score_file.exists()
            with open(self.score_file, "a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if fresh:
                    writer.writerow(("file", "score", "threshold", "threshold_source"))
                writer.writerow((name, f"{score:.4f}", "" if threshold is None else f"{threshold:.4f}", source))
            self.scores[name] = (score, threshold, source)

    def put(self, cell: Cell, audio: np.ndarray, sr: int, style: str = "") -> bool:
        """Store the clip; returns whether it was accepted."""
        wav = to_16k_mono(audio, sr)
        final = self.clips / cell.filename
        if self.verifier is None or cell.engine not in self.verified:
            write_wav(final, wav)
            return True
        verdict = self.verifier.check(wav, cell.phrase)
        if verdict.score is not None:
            self.remember(cell.filename, verdict.score, verdict.threshold, verdict.source)
        if verdict.accepted:
            write_wav(final, verdict.audio)
            return True
        self.rejected.mkdir(exist_ok=True)
        write_wav(self.rejected / cell.filename, wav)
        with self._lock:
            fresh = not self.manifest.exists()
            with open(self.manifest, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=REJECTED_COLUMNS)
                if fresh:
                    writer.writeheader()
                writer.writerow({"file": cell.filename, "engine": cell.engine, "voice": cell.voice, "text": cell.text,
                                 "form": cell.form, "rate": "" if cell.rate is None else cell.rate,
                                 "pitch": "" if cell.pitch is None else cell.pitch, "reason": verdict.reason,
                                 "score": "" if verdict.score is None else f"{verdict.score:.4f}",
                                 "threshold": "" if verdict.threshold is None else f"{verdict.threshold:.4f}",
                                 "threshold_source": verdict.source, "style": style,
                                 "transcript": verdict.transcript})
            self.rejected_count += 1
        return False


def reference_threshold(verifier: Verifier, sink: ClipSink, phrase: str, references: Sequence[Cell],
                        percentile: float) -> Tuple[float, str, int]:
    """Set the threshold of ``phrase`` to a percentile of the alignment scores of its edge-tts clips.

    Only edge-tts clips of the phrase count: their scores look like clean speech, while other synthesizers
    score lower than OmniVoice's bad clips and make the threshold lenient. With fewer than ``MIN_REFERENCES``
    such clips (edge-tts lacks the language) the fixed ``FALLBACK_THRESHOLD`` is used and the source is
    ``fallback``. Returns the threshold, its source and the number of reference clips scored.
    """
    scores = []
    for cell in references:
        path = sink.clips / cell.filename
        if cell.engine != REFERENCE_DONOR or cell.phrase != phrase or not path.exists():
            continue
        if cell.filename in sink.scores:
            scores.append(sink.scores[cell.filename][0])
            continue
        wav, sr = sf.read(path, dtype="float32")
        found = verifier.score(to_16k_mono(wav, sr), phrase)
        if found is not None:
            sink.remember(cell.filename, found.score, None, "")
            scores.append(found.score)
    if len(scores) < MIN_REFERENCES:
        verifier.thresholds[phrase] = (FALLBACK_THRESHOLD, "fallback")
        return FALLBACK_THRESHOLD, "fallback", len(scores)
    return verifier.set_threshold(phrase, scores, percentile, REFERENCE_DONOR), REFERENCE_DONOR, len(scores)


def synthesize_grid(
    cells: Sequence[Cell],
    engines: Sequence[GridEngine],
    sink: ClipSink,
    workers: int,
) -> Tuple[int, int, int]:
    """Synthesize the cells not yet attempted; returns (planned, skipped, failed).

    Clips the ASR check rejects count as written here and are listed in ``manifest_rejected.csv``.
    """
    by_name = {e.name: e for e in engines}
    gates = {e.name: threading.BoundedSemaphore(max(1, e.max_concurrency)) for e in engines}
    todo = [c for c in cells if not sink.known(c)]
    errors: Dict[str, str] = {}

    def work(cell: Cell) -> bool:
        engine = by_name[cell.engine]
        wait = _RETRY_WAIT_S
        for attempt in range(_ATTEMPTS):
            try:
                with gates[cell.engine]:
                    audio, sr = engine.synth(cell.text, cell.voice, cell.rate, cell.pitch)
                    if engine.pause_s:
                        time.sleep(engine.pause_s)
                sink.put(cell, audio, sr)
                return True
            except Exception as e:
                errors[cell.engine] = describe_error(e)
                if attempt + 1 < _ATTEMPTS:
                    time.sleep(wait)
                    wait *= 2
        return False

    ok, failed = _run_pool(todo, work, workers)
    for name, err in sorted(errors.items()):
        if failed:
            logger.warning("%s: last error: %s", name, err)
    return len(cells), len(cells) - len(todo), failed


def style_weights(styles: Sequence[str], counts: Dict[str, Tuple[int, int]], floor: float = STYLE_FLOOR) -> List[float]:
    """Probability of each voice-design style: proportional to its smoothed accept rate, never below ``floor``."""
    if floor * len(styles) > 1:
        raise ValueError(f"{len(styles)} styles cannot each have {floor:.0%}")
    rates = [(counts.get(st, (0, 0))[0] + 1) / (counts.get(st, (0, 0))[1] + 2) for st in styles]
    total = sum(rates)
    return [floor + (1 - floor * len(styles)) * r / total for r in rates]


def pick_style(styles: Sequence[str], weights: Sequence[float], seed: int) -> str:
    """The style for the clip with this seed: a function of the seed and the weights, so equal weights give equal draws."""
    u, edge = random.Random(seed).random(), 0.0
    for style, weight in zip(styles, weights):
        edge += weight
        if u < edge:
            return style
    return styles[-1]


def generate_omnivoice(
    plan: Callable[[int], Tuple[Cell, OmniItem]],
    target: int,
    backend: OmniBackend,
    sink: ClipSink,
    batch: int,
    styles: Sequence[str] = ("auto",),
) -> Dict[str, int]:
    """Generate OmniVoice clips until ``target`` are accepted or ``4 * target`` clip indexes have been tried.

    Clip ``i`` is always the same cell and seed, so clips that exist, accepted or rejected, are skipped
    and a rerun continues where the last one stopped. Each clip also gets a voice-design style (``auto``
    meaning none), drawn per batch with weights from the accept rate of every style so far, floored at
    ``STYLE_FLOOR`` so no style disappears; the style of every clip is recorded in ``omnivoice_styles.csv``.
    """
    cap = 4 * target
    accepted = sum((sink.clips / plan(i)[0].filename).exists() for i in range(cap))
    attempted = failed = 0
    nxt = 0
    rejected_before = sink.rejected_count
    while accepted < target:
        todo = []
        while nxt < cap and len(todo) < min(batch, target - accepted):
            cell, item = plan(nxt)
            nxt += 1
            if not sink.known(cell):
                todo.append((cell, item))
        if not todo:
            break
        weights = style_weights(styles, sink.style_counts())
        chosen = [pick_style(styles, weights, item.seed) for _, item in todo]
        attempted += len(todo)
        try:
            audio = backend.generate([replace(item, instruct=None if st == "auto" else st) for (_, item), st in zip(todo, chosen)])
        except Exception as e:
            logger.warning("omnivoice batch of %d failed: %s", len(todo), describe_error(e))
            failed += len(todo)
            continue
        for (cell, _), (wav, sr), st in zip(todo, audio, chosen):
            try:
                ok = sink.put(cell, wav, sr, st)
            except ValueError as e:
                logger.warning("omnivoice clip %s dropped: %s", cell.filename, e)
                failed += 1
                continue
            sink.record_style(cell.filename, st, ok)
            accepted += ok
    return {"target": target, "accepted": accepted, "attempted": attempted,
            "rejected": sink.rejected_count - rejected_before, "failed": failed}


# ---------------------------------------------------------------------------
# Stage 2: voice cloning
# ---------------------------------------------------------------------------


_VC_PROVIDERS = {"cuda": "CUDAExecutionProvider", "rocm": "ROCMExecutionProvider", "cpu": "CPUExecutionProvider"}


class Cloner:
    """voiceclonnx Chatterbox on a requested device, which it verifies it is really using."""

    def __init__(self, engine: str, device: str):
        import onnxruntime as ort
        from voiceclonnx import VoiceCloner

        available = ort.get_available_providers()
        if device == "auto":
            device = next((d for d in ("cuda", "rocm") if _VC_PROVIDERS[d] in available), "cpu")
        self.provider = _VC_PROVIDERS[device]
        if self.provider not in available:
            raise RuntimeError(f"{self.provider} is not available to onnxruntime (has {available})")
        cfg = {"providers": list(dict.fromkeys([self.provider, "CPUExecutionProvider"]))} if engine == "chatterbox" else {}
        self._cloner = VoiceCloner(engine=engine, **cfg)
        self.sample_rate = self._cloner.sample_rate
        self.engine = engine

    def clone(self, source: str, reference: str, out_path: str) -> None:
        self._cloner.clone_voice(source, reference, out_path)

    def providers_used(self) -> List[str]:
        """The first provider of every onnxruntime session the cloner holds, read after a clip has run."""
        adapter = self._cloner._adapter
        sessions = [s for s in vars(adapter).values() if hasattr(s, "get_providers")]
        return [s.get_providers()[0] for s in sessions]


def _load_cloner(engine: str, device: str) -> Cloner:
    return Cloner(engine, device)


def clone_name(source: str, copy: int) -> str:
    return re.sub(r"__c\d+\.wav$", f"__c{copy}.wav", source)


def pick_reference(refs: Sequence[Path], seed: int, source: str, copy: int) -> Path:
    """The reference speaker for one (source clip, copy): a function of the seed and the clip's name alone, so reruns agree."""
    return refs[random.Random(f"{seed}:{source}:{copy}").randrange(len(refs))]


def clone_clips(
    sources: Sequence[Dict[str, Any]],
    clips_dir: Path,
    refs_dir: Path,
    copies: int,
    engine: str,
    device: str,
    workers: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """Convert every source clip ``copies`` times onto seeded reference speakers.

    Returns (manifest rows of all cloned clips present, planned, skipped, failed). The first clip
    is converted alone and the cloner's sessions are checked against the requested provider before
    the pool starts.
    """
    refs = sorted(refs_dir.rglob("*.wav"))
    if not refs:
        raise SystemExit(f"no .wav reference voices under {refs_dir}")
    jobs = [(row, c) for row in sources for c in range(1, copies + 1)]
    todo = [(row, c) for row, c in jobs if not (clips_dir / clone_name(row["file"], c)).exists()]
    cloner = _load_cloner(engine, device) if todo else None
    tmp_dir = clips_dir / ".tmp"
    tmp_dir.mkdir(exist_ok=True)

    def work(job) -> bool:
        row, c = job
        name = clone_name(row["file"], c)
        ref = pick_reference(refs, seed, row["file"], c)
        out = tmp_dir / f"{threading.get_ident()}-{name}"
        try:
            cloner.clone(str(clips_dir / row["file"]), str(ref), str(out))
            audio, sr = sf.read(out, dtype="float32", always_2d=True)
            write_wav(clips_dir / name, to_16k_mono(audio.mean(1), sr))
            return True
        except Exception as e:
            logger.warning("clone failed for %s copy %d: %s: %s", row["file"], c, type(e).__name__, e)
            return False
        finally:
            out.unlink(missing_ok=True)

    failed = 0
    if todo:
        first_ok = work(todo[0])
        failed += not first_ok
        used = cloner.providers_used()
        if not first_ok or not used or any(p != cloner.provider for p in used):
            raise SystemExit(f"voice cloning requested {cloner.provider} but sessions run on {used or 'nothing'} "
                             f"after the first clip ({'ok' if first_ok else 'failed'})")
        _, f = _run_pool(todo[1:], work, workers)
        failed += f
    rows = []
    for row, c in jobs:
        name = clone_name(row["file"], c)
        if (clips_dir / name).exists():
            ref = pick_reference(refs, seed, row["file"], c)
            rows.append({**row, "file": name, "stage": "vc", "copy": c, "source": row["file"],
                         "reference": os.path.relpath(ref, refs_dir)})
    return rows, len(jobs), len(jobs) - len(todo), failed


# ---------------------------------------------------------------------------
# Stage 3: speaker deduplication
# ---------------------------------------------------------------------------


def _load_embedder(model: str) -> Callable[[str], np.ndarray]:
    from speakeronnx import SpeakerEmbedder

    return SpeakerEmbedder(model=model).embed


def dedup_speakers(
    rows: Sequence[Dict[str, Any]],
    clips_dir: Path,
    threshold: float,
    model: str,
    workers: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Greedy speaker deduplication: a clip is kept unless its cosine similarity to an already kept clip reaches ``threshold``.

    Cloned clips are considered before clean ones, each group in a seeded order, so a duplicate is
    resolved in favour of the cloned clip. Returns one row per clip with ``kept``, ``duplicate_of``
    and ``similarity``.
    """
    embed = _load_embedder(model)
    files = [r["file"] for r in rows]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        vectors = list(pool.map(lambda f: np.asarray(embed(str(clips_dir / f)), dtype=np.float32).ravel(), files))
    unit = np.stack(vectors)
    unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-9)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    order.sort(key=lambda i: rows[i]["stage"] != "vc")
    kept: List[int] = []
    result: Dict[int, Dict[str, Any]] = {}
    for i in order:
        if kept:
            sims = unit[kept] @ unit[i]
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                result[i] = {"file": files[i], "kept": 0, "duplicate_of": files[kept[j]], "similarity": f"{sims[j]:.4f}"}
                continue
        kept.append(i)
        result[i] = {"file": files[i], "kept": 1, "duplicate_of": "", "similarity": ""}
    return [result[i] for i in range(len(rows))]


# ---------------------------------------------------------------------------
# Pipeline and CLI
# ---------------------------------------------------------------------------


def parse_ints(value: str) -> Tuple[int, ...]:
    return tuple(int(v) for v in value.split(",") if v.strip())


def load_phrases(args: argparse.Namespace) -> List[Tuple[str, int]]:
    """``(phrase, label)`` pairs: ``--wake-word`` and the lines of ``--phrases-file`` (``phrase`` or ``phrase<TAB>label``)."""
    def clean(text: str) -> str:
        return text.replace("_", " ").replace("-", " ").strip()

    phrases = [(clean(args.wake_word), args.label)] if args.wake_word else []
    if args.phrases_file:
        for line in Path(args.phrases_file).read_text(encoding="utf-8").splitlines():
            text, _, label = line.partition("\t")
            if text.strip():
                phrases.append((clean(text), int(label) if label.strip() else args.label))
    return list(dict.fromkeys(phrases))


def run_voicegrid(args: argparse.Namespace, engines: Optional[Sequence[GridEngine]] = None,
                  backend: Optional[OmniBackend] = None) -> int:
    """Run the stages selected by ``args``; returns the number of clips that failed."""
    phrases = load_phrases(args)
    out = Path(args.output_dir)
    exclude = [] if args.no_default_excludes else list(DEFAULT_EXCLUDE_VOICES)
    for item in args.exclude_voices:
        exclude += [p.strip() for p in item.split(",") if p.strip()]
    if engines is None:
        engines = build_donors(args)
    omni = [e for e in engines if isinstance(e, OmniVoiceDonor)]
    grid = [e for e in engines if not isinstance(e, OmniVoiceDonor)]
    voices = {e.name: select_voices(e, args.lang, exclude, args.max_voices_per_engine, args.seed) for e in grid}
    cells: List[Cell] = []
    for phrase, label in phrases:
        cells += enumerate_grid(grid, voices, phrase, args.rates, args.pitches, args.forms,
                                args.max_clean_per_voice, args.seed, label)
    print("VOICEGRID", "voices", {k: len(v) for k, v in voices.items()}, "omnivoice", bool(omni),
          "excluded patterns", len(exclude), flush=True)
    omni_clips = args.omnivoice_clips * len(phrases) if omni else 0
    if args.dry_run:
        _summary("plan", cells=len(cells) + omni_clips, clones=(len(cells) + omni_clips) * args.vc_copies if args.vc_refs else 0)
        return 0

    verified = {e.name for e in engines} if args.verify_all_donors else {o.name for o in omni}
    verifier = None
    mode = args.verify
    if mode == "align" and args.lang.lower().split("-")[0] in {x.strip().lower() for x in args.require_asr_langs.split(",")}:
        mode = "both"
    if mode != "none" and verified:
        aligns = mode in ("align", "both")
        spellings = [t for item in args.accept_spellings for t in item.split(",") if t.strip()]
        verifier = Verifier(mode, MmsAligner(args.device) if aligns else None,
                            _load_asr(args.asr_model) if mode in ("asr", "both") else None, spellings, args.asr_threshold)
        if aligns and args.verify_threshold is not None:
            verifier.default = (args.verify_threshold, "manual")
    sink = ClipSink(out, verifier, verified)
    clips_dir = sink.clips
    started = time.monotonic()
    planned, skipped, failed = synthesize_grid(cells, grid, sink, args.workers)
    _summary("tts", planned=planned, existing=skipped, written=planned - skipped - failed, failed=failed,
             rejected=sink.rejected_count, seconds=round(time.monotonic() - started, 1))
    plans: List[Cell] = []
    if omni:
        started = time.monotonic()
        if verifier is not None and verifier.aligner is not None and verifier.default is None:
            for phrase, _ in phrases:
                threshold, source, count = reference_threshold(verifier, sink, phrase, cells, args.verify_percentile)
                _summary("threshold", lang=args.lang, phrase=repr(phrase), source=source, references=count,
                         percentile=args.verify_percentile, value=round(threshold, 4))
        backend = backend or omnivoice_backend(args.device, args.workers)
        codes = omnivoice_codes(args.lang, omni[0].macrolanguages)
        if not codes:
            raise SystemExit(f"OmniVoice has no voice for language {args.lang!r}")
        for phrase, label in phrases:
            def plan(i, phrase=phrase, label=label):
                return omnivoice_plan(i, phrase, label, args.lang, codes, args.forms, args.omnivoice_rates, args.seed)
            result = generate_omnivoice(plan, args.omnivoice_clips, backend, sink, args.batch_size,
                                        args.omnivoice_style or OMNIVOICE_STYLES)
            failed += result["failed"]
            plans += [c for c in (plan(i)[0] for i in range(4 * args.omnivoice_clips)) if (clips_dir / c.filename).exists()]
            _summary("omnivoice", phrase=repr(phrase), backend=type(backend).__name__, **result,
                     seconds=round(time.monotonic() - started, 1))
    rows = [_clean_row(c, args.lang) for c in cells if (clips_dir / c.filename).exists()] + [_clean_row(c, args.lang) for c in plans]
    for row in rows:
        if row["file"] in sink.scores:
            score, threshold, source = sink.scores[row["file"]]
            row["score"] = f"{score:.4f}"
            row["threshold"] = "" if threshold is None else f"{threshold:.4f}"
            row["threshold_source"] = source
        if row["file"] in sink.styles:
            row["style"] = sink.styles[row["file"]][0]

    if args.vc_refs:
        started = time.monotonic()
        cloned, planned, skipped, vc_failed = clone_clips(rows, clips_dir, Path(args.vc_refs), args.vc_copies,
                                                          args.vc_engine, args.vc_device, args.workers, args.seed)
        _summary("clone", planned=planned, existing=skipped, written=planned - skipped - vc_failed, failed=vc_failed,
                 seconds=round(time.monotonic() - started, 1))
        rows += cloned
        failed += vc_failed
    _write_manifest(out / "manifest.csv", rows)
    kept = {r["file"] for r in rows}

    if args.dedup_speaker_threshold is not None:
        verdict = dedup_speakers(rows, clips_dir, args.dedup_speaker_threshold, args.dedup_model, args.workers, args.seed)
        with open(out / "dedup.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=("file", "kept", "duplicate_of", "similarity"))
            writer.writeheader()
            writer.writerows(verdict)
        _summary("dedup", clips=len(verdict), kept=sum(v["kept"] for v in verdict),
                 dropped=sum(1 - v["kept"] for v in verdict),
                 kept_cloned=sum(v["kept"] for v, r in zip(verdict, rows) if r["stage"] == "vc"))
        kept = {v["file"] for v in verdict if v["kept"]}
    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows((f"clips/{r['file']}", r["label"]) for r in rows if r["file"] in kept)
    _summary("done", manifest=len(rows), failed=failed)
    return failed


def build_donors(args: argparse.Namespace) -> List[GridEngine]:
    """The donors for ``args.lang``: ``--donors`` as a roster file or a comma list of donor names, else the packaged roster."""
    spec = args.donors
    if spec and Path(spec).is_file():
        roster, names = load_roster(spec), None
    else:
        roster, names = load_roster(), ([n.strip() for n in spec.split(",") if n.strip()] if spec else None)
    explicit = bool(names)
    names = names or donors_for(roster, args.lang)
    tts_config = None
    if args.tts_config:
        from ww_trainer.datagen import load_tts_config

        tts_config = load_tts_config(args.tts_config)
    donors = []
    for n in names:
        try:
            donors.append(make_donor(n, roster, args.lang, args.workers, tts_config))
        except (ImportError, KeyError) as e:
            if explicit:
                raise
            logger.warning("roster donor %s skipped, its package is not installed: %s", n, e)
    return donors


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ww_trainer-voicegrid",
        description="Synthesize a wake phrase with every donor voice at several rates, pitches and text forms.",
    )
    p.add_argument("--wake-word", default=None, help="the phrase, e.g. 'hey jarvis'")
    p.add_argument("--phrases-file", default=None,
                   help="text file with one phrase per line, or 'phrase<TAB>label'; each phrase gets the whole grid")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--lang", default="en", help="language of the donor voices (default: en)")
    p.add_argument("--donors", default=None,
                   help="a roster JSON file, or a comma list of donor names (e.g. omnivoice,styletts2,edge); "
                        "default: the packaged roster's donors for --lang")
    p.add_argument("--omnivoice-clips", type=int, default=OMNIVOICE_CLIPS,
                   help=f"accepted OmniVoice clips per phrase, one random speaker each (default: {OMNIVOICE_CLIPS})")
    p.add_argument("--omnivoice-rates", type=parse_ints, default=OMNIVOICE_RATES,
                   help="speed percentages the OmniVoice clips rotate through (default: 0,15)")
    p.add_argument("--omnivoice-style", action="append", default=[], metavar="TEXT",
                   help="a voice-design style for OmniVoice clips ('auto' for none); repeatable, replaces the default list "
                        f"({'; '.join(OMNIVOICE_STYLES)}). Styles are drawn with weights from their accept rates, "
                        f"at least {STYLE_FLOOR * 100:.0f}%% each")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "rocm"],
                   help="device of the official omnivoice package; rocm is torch's cuda device on AMD (default: auto)")
    p.add_argument("--batch-size", type=int, default=10, help="OmniVoice clips per batch (default: 10)")
    p.add_argument("--verify", default="align", choices=["align", "asr", "both", "none"],
                   help="intelligibility check of OmniVoice clips: CTC forced alignment of the phrase scored against "
                        "the other donors' clips (default), an ASR transcript match, both, or none")
    p.add_argument("--verify-percentile", type=float, default=VERIFY_PERCENTILE,
                   help="reject OmniVoice clips whose alignment score is below this percentile of the edge-tts "
                        f"clips of the same phrase (default: {VERIFY_PERCENTILE})")
    p.add_argument("--require-asr-langs", default="", metavar="LANG[,LANG...]",
                   help="languages for which an OmniVoice clip must pass both the alignment and the ASR check, for "
                        "languages where the alignment was too lenient (for example da,it,fr; default: none)")
    p.add_argument("--verify-threshold", type=float, default=None,
                   help="alignment score threshold to use instead of deriving it from reference clips")
    p.add_argument("--verify-all-donors", action="store_true",
                   help="verify every donor's clips, not only OmniVoice's; alignment then needs --verify-threshold")
    p.add_argument("--asr-model", default=ASR_MODEL, help=f"onnx_asr model for --verify asr, run on the CPU (default: {ASR_MODEL})")
    p.add_argument("--asr-threshold", type=float, default=ASR_THRESHOLD,
                   help=f"similarity of the normalised transcript to the phrase that accepts a clip (default: {ASR_THRESHOLD})")
    p.add_argument("--accept-spellings", action="append", default=[], metavar="TEXT[,TEXT...]",
                   help="alternative spellings of the phrase an ASR transcript may match (--verify asr or both); repeatable")
    p.add_argument("--tts-config", default=None, help="JSON {plugin name: config} for plugin donors")
    p.add_argument("--workers", type=int, default=4, help="parallel synthesis, cloning and embedding workers (default: 4)")
    p.add_argument("--rates", type=parse_ints, default=RATES, help="speaking-rate percentages (default: -25,-12,0,12,25)")
    p.add_argument("--pitches", type=parse_ints, default=PITCHES, help="pitch shifts in Hz (default: -10,0,10)")
    p.add_argument("--forms", type=lambda s: tuple(x.strip() for x in s.split(",")), default=DEFAULT_FORMS,
                   help=f"text forms from {','.join(FORMS)} (default: {','.join(DEFAULT_FORMS)})")
    p.add_argument("--exclude-voices", action="append", default=[], metavar="GLOB[,GLOB...]",
                   help="voice ids to leave out, case-insensitive globs, e.g. 'kokoro*/am_adam'; repeatable. "
                        "Added to the built-in held-out Kokoro speakers")
    p.add_argument("--no-default-excludes", action="store_true", help="do not exclude the held-out Kokoro speakers")
    p.add_argument("--max-voices-per-engine", type=int, default=None, help="seeded sample of this many voices per engine")
    p.add_argument("--max-clean-per-voice", type=int, default=None,
                   help="synthesize only this many seeded grid cells per voice")
    p.add_argument("--vc-refs", default=None, help="directory of reference speaker wavs; enables the cloning stage")
    p.add_argument("--vc-copies", type=int, default=1, help="clones per clean clip, each on a different seeded reference")
    p.add_argument("--vc-engine", default="chatterbox", help="voiceclonnx engine (default: chatterbox)")
    p.add_argument("--vc-device", default="auto", choices=["auto", "cpu", "cuda", "rocm"],
                   help="device for cloning, verified against the sessions after the first clip (default: auto)")
    p.add_argument("--dedup-speaker-threshold", type=float, default=DEFAULT_DEDUP_THRESHOLD,
                   help="cosine similarity at or above which a clip is a speaker duplicate of a kept clip; writes "
                        f"dedup.csv (default: {DEFAULT_DEDUP_THRESHOLD})")
    p.add_argument("--no-dedup", dest="dedup_speaker_threshold", action="store_const", const=None,
                   help="skip speaker deduplication")
    p.add_argument("--dedup-model", default="wespeaker-resnet34", help="speakeronnx model (default: wespeaker-resnet34)")
    p.add_argument("--label", type=int, default=1, choices=[0, 1],
                   help="metadata.csv label of a phrase without its own: 1 for the wake word, 0 for a negative phrase")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="print the plan and write nothing")
    return p


def cli_main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for ``ww_trainer-voicegrid``."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.wake_word and not args.phrases_file:
        parser.error("give --wake-word or --phrases-file")
    if args.verify_all_donors and args.verify in ("align", "both") and args.verify_threshold is None:
        parser.error("--verify-all-donors with alignment needs --verify-threshold")
    bad = [f for f in args.forms if f not in FORMS]
    if bad:
        parser.error(f"unknown form {bad}; choose from {','.join(FORMS)}")
    if args.vc_refs and args.vc_copies < 1:
        parser.error("--vc-copies must be at least 1")
    sys.exit(1 if run_voicegrid(args) else 0)


if __name__ == "__main__":
    cli_main()
