"""Voice conversion via `voiceclonnx <https://github.com/TigreGotico/voiceclonnx>`_.

This module is a thin delegation to the **voiceclonnx** unified API — a pure-ONNX
voice-conversion library (zero PyTorch at runtime, 14 engines).  There is no
bespoke backend hierarchy any more: pick an engine and convert.

Audio-to-audio only.  Text→speech (TTS) is a separate concern handled by the
OVOS TTS plugins in :mod:`ww_trainer.datagen`.

Selecting an engine
-------------------
* ``WW_VC_ENGINE`` env var (e.g. ``knnvc``, ``facodec``, ``freevc``,
  ``openvoice``, ``rvc``, ``linacodec``, ``chatterbox`` …), or
* the ``engine=`` argument to :func:`load_vc_backend`.

``knnvc`` is the default: zero-shot any-to-any, pure-numpy kNN matching, 16 kHz
output that matches the wake-word training sample rate.

Usage
-----
    from ww_trainer.vc_helpers import load_vc_backend, list_engines

    vc = load_vc_backend()                 # WW_VC_ENGINE or knnvc
    vc = load_vc_backend(engine="facodec") # explicit engine
    vc.vc(source_path, donor_path, out_path)
    vc.sample_rate                         # int, engine-dependent
    list_engines()                         # -> ['bicodec', 'chatterbox', ...]
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# Default engine: env override, else knnvc (16 kHz zero-shot any-to-any).
DEFAULT_ENGINE = os.environ.get("WW_VC_ENGINE", "knnvc").strip()

# Legacy backend names (from the old multi-backend wrapper) → voiceclonnx engines.
# Kept so existing scripts / .env files keep working without edits.
_LEGACY_ALIASES = {
    "auto": DEFAULT_ENGINE,
    "voiceclonnx": DEFAULT_ENGINE,
    "chatterbox-onnx": "chatterbox",
    "chatterbox": "chatterbox",
    "linacodec": "linacodec",
}


class VoiceConverter:
    """Thin wrapper over :class:`voiceclonnx.VoiceCloner`.

    Exposes the ``.vc()`` / ``.sample_rate`` / ``.name`` interface the rest of
    the codebase expects, delegating all real work to voiceclonnx.
    """

    def __init__(self, engine: str = DEFAULT_ENGINE) -> None:
        from voiceclonnx import VoiceCloner  # pip: voiceclonnx
        logger.info("[VC] Loading voiceclonnx engine=%s (HF download on first use)", engine)
        self._cloner = VoiceCloner(engine=engine)
        self.engine = engine
        self.name = f"voiceclonnx:{engine}"
        self.sample_rate = self._cloner.sample_rate
        logger.info("[VC] ready  engine=%s  sr=%d", engine, self.sample_rate)

    def vc(self, source_path, donor_path, out_path) -> None:
        """Voice-convert *source_path* to *donor_path*'s voice → *out_path*."""
        self._cloner.clone_voice(
            audio=str(source_path),
            reference_voice=str(donor_path),
            out_path=str(out_path),
        )

    # Alias matching voiceclonnx's own naming.
    convert = vc


def list_engines() -> List[str]:
    """Return the sorted list of available voiceclonnx engine aliases."""
    from voiceclonnx import ENGINE_REGISTRY
    return sorted(ENGINE_REGISTRY.keys())


def load_vc_backend(
    backend: Optional[str] = None,
    engine: Optional[str] = None,
    **_ignored,
) -> VoiceConverter:
    """Return a :class:`VoiceConverter` for the requested voiceclonnx engine.

    Args:
        backend: Legacy alias for ``engine`` (kept for older callers / ``.env``
            files). Legacy values ``auto``/``voiceclonnx`` map to the default
            engine; ``chatterbox-onnx``/``chatterbox`` → the ``chatterbox``
            engine; ``linacodec`` → the ``linacodec`` engine.
        engine: voiceclonnx engine alias (see :func:`list_engines`). Takes
            precedence over ``backend``. Defaults to ``WW_VC_ENGINE`` then
            ``knnvc``.
        **_ignored: Accepts and ignores obsolete kwargs (``device``,
            ``quantized``) so existing call sites need no changes — voiceclonnx
            is CPU/ONNX and engine-agnostic about device.

    Raises:
        RuntimeError: if voiceclonnx is not installed.
    """
    choice = (engine or backend or DEFAULT_ENGINE).strip()
    if choice.startswith("voiceclonnx:"):
        choice = choice.split(":", 1)[1]
    choice = _LEGACY_ALIASES.get(choice, choice)

    try:
        return VoiceConverter(engine=choice)
    except ImportError as exc:
        raise RuntimeError(
            f"voiceclonnx not installed ({exc}).\n"
            "Run: uv pip install voiceclonnx --python .venv/bin/python"
        ) from exc
    except KeyError as exc:
        raise ValueError(
            f"Unknown voiceclonnx engine {choice!r}. "
            f"Available engines: {', '.join(list_engines())}"
        ) from exc
