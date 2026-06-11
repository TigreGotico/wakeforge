"""Voice-conversion / TTS backend abstraction.

Supported backends
------------------
  ``chatterbox-onnx``  CPU-only ONNX runtime.  Supports TTS + VC.
                       pip install chatterbox-onnx
                       env: WW_VC_BACKEND=chatterbox-onnx

  ``chatterbox``       Original PyTorch model (GPU recommended).  TTS + VC.
                       pip install chatterbox-tts
                       env: WW_VC_BACKEND=chatterbox


  ``auto``             Priority: GPU chatterbox → chatterbox-onnx (CPU).
                       Default when WW_VC_BACKEND is not set.

Usage
-----
    from ww_trainer.vc_helpers import load_vc_backend

    backend = load_vc_backend()            # auto-detect
    backend.tts("hey mycroft", donor_path, out_path, exaggeration=0.4)
    backend.vc(source_path, donor_path, out_path)
    backend.sample_rate   # int, e.g. 24000 or 48000
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default backend from env; callers can override with load_vc_backend(backend=...)
_DEFAULT_BACKEND = os.environ.get("WW_VC_BACKEND", "auto")


# ---------------------------------------------------------------------------
# Unified backend interface
# ---------------------------------------------------------------------------

class _VCBackend:
    """Abstract base — subclasses implement tts() and vc()."""

    sample_rate: int = 24000
    name: str = "unknown"

    def tts(
        self,
        text: str,
        donor_path: "str | Path",
        out_path: "str | Path",
        exaggeration: float = 0.4,
    ) -> None:
        """Synthesise *text* in *donor_path*'s voice, write to *out_path*."""
        raise NotImplementedError

    def vc(
        self,
        source_path: "str | Path",
        donor_path: "str | Path",
        out_path: "str | Path",
    ) -> None:
        """Voice-convert *source_path* to *donor_path*'s voice, write to *out_path*."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# chatterbox-onnx backend  (CPU, quantized ONNX)
# ---------------------------------------------------------------------------

class _OnnxBackend(_VCBackend):
    name = "chatterbox-onnx"
    sample_rate = 24000

    def __init__(self, quantized: bool = True) -> None:
        from chatterbox_onnx import ChatterboxOnnx  # type: ignore — pip: chatterbox-onnx
        logger.info("[VC] Loading ChatterboxOnnx (quantized=%s)", quantized)
        self._model = ChatterboxOnnx(quantized=quantized)
        logger.info("[VC] chatterbox-onnx ready")

    def tts(self, text, donor_path, out_path, exaggeration=0.4):
        self._model.synthesize(
            text=text,
            target_voice_path=str(donor_path),
            exaggeration=exaggeration,
            output_file_name=str(out_path),
        )

    def vc(self, source_path, donor_path, out_path):
        self._model.voice_convert(
            source_audio_path=str(source_path),
            target_voice_path=str(donor_path),
            output_file_name=str(out_path),
        )


# ---------------------------------------------------------------------------
# chatterbox (PyTorch) backend  (GPU recommended)
# ---------------------------------------------------------------------------

class _TorchBackend(_VCBackend):
    name = "chatterbox"

    def __init__(self, device: str = "auto") -> None:
        import torch
        from chatterbox.tts import ChatterboxTTS  # type: ignore
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[VC] Loading ChatterboxTTS on %s", device)
        self._model = ChatterboxTTS.from_pretrained(device=device)
        self.sample_rate = self._model.sr
        self._device = device
        logger.info("[VC] chatterbox (torch) ready on %s  sr=%d", device, self.sample_rate)

    def tts(self, text, donor_path, out_path, exaggeration=0.4):
        import torchaudio
        wav = self._model.generate(
            text,
            audio_prompt_path=str(donor_path),
            exaggeration=exaggeration,
        )
        from ww_trainer.dataset import _save_audio
        _save_audio(str(out_path), wav.cpu(), self.sample_rate)

    def vc(self, source_path, donor_path, out_path):
        import torchaudio
        # ChatterboxTTS uses generate() with cfm_steps or a dedicated vc method.
        # Try the dedicated vc method first, fall back to generate().
        if hasattr(self._model, "voice_convert"):
            wav = self._model.voice_convert(
                source_audio_path=str(source_path),
                target_voice_path=str(donor_path),
            )
        else:
            # Fallback: use generate() — the donor voice guides prosody/timbre
            wav = self._model.generate(
                "",  # empty text = voice conversion mode on some versions
                audio_prompt_path=str(donor_path),
                # Some versions accept source_audio for VC:
                **_maybe_kwarg("source_audio_path", str(source_path), self._model.generate),
            )
        from ww_trainer.dataset import _save_audio
        _save_audio(str(out_path), wav.cpu(), self.sample_rate)


# ---------------------------------------------------------------------------


def _maybe_kwarg(key: str, value, fn) -> dict:
    """Return {key: value} only if fn accepts *key* as a keyword argument."""
    import inspect
    try:
        sig = inspect.signature(fn)
        return {key: value} if key in sig.parameters else {}
    except (ValueError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_vc_backend(
    backend: Optional[str] = None,
    device: str = "auto",
    quantized: bool = True,
) -> _VCBackend:
    """Instantiate and return the requested VC backend.

    Args:
        backend: ``"chatterbox-onnx"``, ``"chatterbox"``, or ``"auto"``.
                 Defaults to the ``WW_VC_BACKEND`` env var, then ``"auto"``.
        device:  PyTorch device for the torch backend (``"auto"`` = GPU if available).
        quantized: Whether to use quantized ONNX weights (onnx backend only).

    Returns:
        A ``_VCBackend`` instance with ``.tts()`` and ``.vc()`` methods.

    Raises:
        RuntimeError: if the requested backend is not installed.
    """
    choice = (backend or _DEFAULT_BACKEND).lower().strip()

    if choice == "auto":
        # Prefer GPU chatterbox if importable and GPU available
        try:
            import torch
            if torch.cuda.is_available():
                import chatterbox.tts  # noqa: F401
                return _TorchBackend(device=device)
        except ImportError:
            pass
        # Fall through to CPU onnx
        choice = "chatterbox-onnx"

    if choice == "chatterbox-onnx":
        try:
            return _OnnxBackend(quantized=quantized)
        except ImportError:
            raise RuntimeError(
                "chatterbox-onnx not installed.\n"
                "Run: uv pip install chatterbox-onnx --python .venv/bin/python"
            )

    if choice == "chatterbox":
        try:
            return _TorchBackend(device=device)
        except ImportError:
            raise RuntimeError(
                "chatterbox not installed.\n"
                "Run: uv pip install chatterbox-tts --python .venv/bin/python"
            )

    raise ValueError(
        f"Unknown VC backend {choice!r}. "
        "Choose 'chatterbox-onnx', 'chatterbox', or 'auto'."
    )
