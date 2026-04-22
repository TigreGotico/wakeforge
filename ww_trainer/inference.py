"""ONNX-only wake word inference — no PyTorch dependency at runtime."""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
import onnxruntime as ort


class PredictionSmoother:
    """Smooths raw per-frame predictions for robust wake word detection.

    Inspired by openWakeWord's prediction buffer + patience mechanism
    and EfficientWord-Net's relaxation time.

    Supports three smoothing methods:
    - ``"ema"``: Exponential moving average (fast response, smooth decay).
    - ``"mean"``: Rolling window mean (stable, uniform weighting).
    - ``"max"``: Rolling window max (aggressive, catches peaks).

    Args:
        method: Smoothing method — ``"ema"``, ``"mean"``, or ``"max"``.
        window_size: Number of frames in the rolling buffer (for mean/max).
        threshold: Detection threshold for ``is_triggered()``.
        patience: Number of consecutive above-threshold frames required.
        debounce_sec: Minimum seconds between consecutive triggers.
        frame_rate_hz: Approximate frame rate for debounce timing.
        ema_alpha: Smoothing factor for EMA (default 0.3). Higher = more responsive.
    """

    def __init__(
        self,
        method: str = "ema",
        window_size: int = 5,
        threshold: float = 0.5,
        patience: int = 3,
        debounce_sec: float = 1.0,
        frame_rate_hz: float = 10.0,
        ema_alpha: float = 0.3,
    ) -> None:
        if method not in ("ema", "mean", "max"):
            raise ValueError(f"Unknown smoothing method: {method!r}. Use 'ema', 'mean', or 'max'.")
        self.method = method
        self.window_size = window_size
        self.threshold = threshold
        self.patience = patience
        self.debounce_frames = int(debounce_sec * frame_rate_hz)
        self.ema_alpha = ema_alpha

        self._buffer: deque = deque(maxlen=window_size)
        self._ema_value: float = 0.0
        self._consecutive_above: int = 0
        self._frames_since_trigger: int = self.debounce_frames  # allow first trigger
        self._smoothed: float = 0.0

    def update(self, raw_prob: float) -> float:
        """Feed a new raw prediction and return the smoothed value.

        Args:
            raw_prob: Raw sigmoid probability from model.

        Returns:
            Smoothed probability.
        """
        self._frames_since_trigger += 1

        if self.method == "ema":
            self._ema_value = self.ema_alpha * raw_prob + (1.0 - self.ema_alpha) * self._ema_value
            self._smoothed = self._ema_value
        elif self.method == "mean":
            self._buffer.append(raw_prob)
            self._smoothed = sum(self._buffer) / len(self._buffer)
        elif self.method == "max":
            self._buffer.append(raw_prob)
            self._smoothed = max(self._buffer)

        if self._smoothed >= self.threshold:
            self._consecutive_above += 1
        else:
            self._consecutive_above = 0

        return self._smoothed

    def is_triggered(self) -> bool:
        """Check if a detection should fire.

        Returns True when smoothed probability exceeds threshold for
        ``patience`` consecutive frames AND debounce window has elapsed.

        Returns:
            True if wake word detected.
        """
        if (self._consecutive_above >= self.patience
                and self._frames_since_trigger >= self.debounce_frames):
            self._frames_since_trigger = 0
            self._consecutive_above = 0
            return True
        return False

    def reset(self) -> None:
        """Reset smoother state."""
        self._buffer.clear()
        self._ema_value = 0.0
        self._consecutive_above = 0
        self._frames_since_trigger = self.debounce_frames
        self._smoothed = 0.0


class OnnxWakeWordInferencer:
    """Run wake word detection using multiple ONNX sessions.

    This class has no dependency on PyTorch. Only numpy and onnxruntime are required.

    A text featurizer ONNX can optionally be supplied alongside the audio featurizer.
    When present, :meth:`infer` accepts ``text_token_ids`` and appends the resulting
    text embedding as extra feature channels before running the classifier head —
    exactly the same pattern used for the optional VAD channel.

    Args:
        extractor_path: Path to the audio feature extractor ONNX file.
        head_path: Path to the classifier head ONNX file.
        vad_path: Optional path to a VAD ONNX file (e.g. Silero VAD).
        text_extractor_path: Optional path to a text-encoder ONNX file.
            The ONNX must accept ``[1, seq_len]`` int64 token IDs and return
            ``[1, 1, D]`` or ``[1, D]`` float32 embeddings.
        text_emb_dim: Output embedding dimension of the text featurizer (``D``).
        sample_rate: Expected audio sample rate in Hz (default 16000).
        device: ``"cpu"``, ``"cuda"``, or ``"auto"`` (selects CUDA if available).
    """

    def __init__(self, extractor_path: str, head_path: str,
                 vad_path: Optional[str] = None,
                 text_extractor_path: Optional[str] = None,
                 text_emb_dim: int = 128,
                 sample_rate: int = 16000, device: str = "auto") -> None:
        if device == "auto":
            available = ort.get_available_providers()
            device = "cuda" if "CUDAExecutionProvider" in available else "cpu"
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        self.extractor = ort.InferenceSession(extractor_path, providers=providers)
        self.head = ort.InferenceSession(head_path, providers=providers)
        self.vad = ort.InferenceSession(vad_path, providers=providers) if vad_path else None
        self.text_ext = (
            ort.InferenceSession(text_extractor_path, providers=providers)
            if text_extractor_path else None
        )
        self._text_emb_dim = text_emb_dim
        self.sample_rate = sample_rate

        self._ext_input = self.extractor.get_inputs()[0].name
        self._ext_output = self.extractor.get_outputs()[0].name
        self._head_input = self.head.get_inputs()[0].name
        self._head_output = self.head.get_outputs()[0].name

        if self.vad:
            self._vad_input = self.vad.get_inputs()[0].name
            self._vad_output = self.vad.get_outputs()[0].name
        if self.text_ext:
            self._text_input = self.text_ext.get_inputs()[0].name
            self._text_output = self.text_ext.get_outputs()[0].name

    def _run_text(self, token_ids: "list[int]", target_t: int) -> np.ndarray:
        """Encode token IDs and expand to ``[1, target_t, D]``."""
        ids = np.array([token_ids], dtype=np.int64)  # [1, seq_len]
        out = self.text_ext.run([self._text_output], {self._text_input: ids})[0]
        # out may be [1, 1, D] or [1, D]
        if out.ndim == 2:
            out = out[:, np.newaxis, :]   # [1, 1, D]
        return np.repeat(out, target_t, axis=1)  # [1, target_t, D]

    def _run_vad(self, wav: np.ndarray, target_t: int) -> np.ndarray:
        """Run VAD and interpolate to match target time dimension."""
        # Silero VAD (v4+) specific: expects 512 sample chunks
        chunk_size = 512
        B, L = wav.shape
        # Pad to multiple of chunk_size
        pad_len = (chunk_size - (L % chunk_size)) % chunk_size
        if pad_len > 0:
            wav = np.pad(wav, ((0, 0), (0, pad_len)))
        
        L_padded = wav.shape[1]
        T_chunks = L_padded // chunk_size
        
        # Reshape to batch of chunks
        chunks = wav.reshape(B * T_chunks, chunk_size)
        
        # Run VAD
        # Note: some Silero versions expect [B, L] or [B, T, 512]. 
        # We assume [B_total, 512] for consistency with the feats.py wrapper.
        vad_out = self.vad.run([self._vad_output], {self._vad_input: chunks})[0]
        # [B * T_chunks, 1]
        vad_probs = vad_out.reshape(B, T_chunks, 1)
        
        if T_chunks == target_t:
            return vad_probs
            
        # Linear interpolation to match base feature length
        # [B, T_chunks, 1] -> [B, target_t, 1]
        x = np.linspace(0, 1, T_chunks)
        x_new = np.linspace(0, 1, target_t)
        
        aligned = np.zeros((B, target_t, 1), dtype=np.float32)
        for b in range(B):
            # numpy.interp expects 1D arrays
            aligned[b, :, 0] = np.interp(x_new, x, vad_probs[b, :, 0])
            
        return aligned

    def infer(self, audio: np.ndarray,
              text_token_ids: "Optional[list[int]]" = None) -> float:
        """Infer wake word probability for a single audio waveform.

        Args:
            audio: 1-D float32 numpy array at ``self.sample_rate``.
            text_token_ids: Optional list of integer token IDs for the keyword.
                Required when ``text_extractor_path`` was supplied at init.
                The token scheme must match whatever the text-encoder ONNX expects.

        Returns:
            Sigmoid probability in [0, 1].
        """
        if audio.ndim != 1:
            raise ValueError("audio must be a 1-D numpy array")
        wav = audio[np.newaxis, :].astype(np.float32)  # [1, T]
        feats = self.extractor.run(
            [self._ext_output], {self._ext_input: wav}
        )[0]  # [1, T, F]

        if self.vad:
            vad_probs = self._run_vad(wav, feats.shape[1])
            feats = np.concatenate([feats, vad_probs], axis=-1)

        if self.text_ext is not None and text_token_ids is not None:
            text_feats = self._run_text(text_token_ids, feats.shape[1])
            feats = np.concatenate([feats, text_feats], axis=-1)  # [1, T, F+D]

        logit = self.head.run(
            [self._head_output], {self._head_input: feats}
        )[0]  # [1] or scalar
        logit_val = float(np.asarray(logit).ravel()[0])
        return float(1.0 / (1.0 + np.exp(-logit_val)))

    def infer_batch(self, audio_batch: np.ndarray,
                    text_token_ids: "Optional[list[int]]" = None) -> np.ndarray:
        """Infer for a batch of equal-length waveforms.

        Args:
            audio_batch: float32 array of shape ``[B, T]``.
            text_token_ids: Optional token IDs shared across the batch.

        Returns:
            Sigmoid probabilities array of shape ``[B]``.
        """
        if audio_batch.ndim != 2:
            raise ValueError("audio_batch must be a 2-D numpy array [B, T]")
        wav = audio_batch.astype(np.float32)
        feats = self.extractor.run([self._ext_output], {self._ext_input: wav})[0]

        if self.vad:
            vad_probs = self._run_vad(wav, feats.shape[1])
            feats = np.concatenate([feats, vad_probs], axis=-1)

        if self.text_ext is not None and text_token_ids is not None:
            B = feats.shape[0]
            text_feats = self._run_text(text_token_ids, feats.shape[1])  # [1, T, D]
            text_feats = np.repeat(text_feats, B, axis=0)               # [B, T, D]
            feats = np.concatenate([feats, text_feats], axis=-1)

        logits = self.head.run([self._head_output], {self._head_input: feats})[0]
        logits = np.asarray(logits).ravel()
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def infer_streaming(self, audio_chunk: np.ndarray,
                        cache: np.ndarray | None,
                        smoother: Optional[PredictionSmoother] = None,
                        ) -> tuple[float, np.ndarray]:
        """Process one audio chunk with a rolling feature cache.

        Args:
            audio_chunk: 1-D float32 array (one chunk of audio).
            cache: Previous feature cache [T_cached, F] or None for first call.
            smoother: Optional PredictionSmoother for temporal smoothing.

        Returns:
            (probability, updated_cache) — probability is smoothed if
            smoother is provided, raw otherwise.
        """
        wav = audio_chunk[np.newaxis, :].astype(np.float32)  # [1, T_chunk]
        new_feats = self.extractor.run(
            [self._ext_output], {self._ext_input: wav}
        )[0]  # [1, T_new, F]
        
        if self.vad:
            vad_probs = self._run_vad(wav, new_feats.shape[1])
            new_feats = np.concatenate([new_feats, vad_probs], axis=-1)

        new_feats = new_feats.squeeze(0) # [T_new, F_total]

        if cache is None:
            cache = new_feats
        else:
            cache = np.concatenate([cache, new_feats], axis=0)

        # Keep last 50 frames (matching SlidingFeatureCacheTensor default)
        window = 50
        if cache.shape[0] > window:
            cache = cache[-window:]

        feats_input = cache[np.newaxis, :, :]  # [1, T_cached, F]
        logit = self.head.run(
            [self._head_output], {self._head_input: feats_input}
        )[0]
        logit_val = float(np.asarray(logit).ravel()[0])
        prob = float(1.0 / (1.0 + np.exp(-logit_val)))
        if smoother is not None:
            prob = smoother.update(prob)
        return prob, cache


def cli_main() -> None:
    """CLI entry point for ``ww_trainer-infer``.

    Run a trained ONNX wake-word model on an audio file and print the
    confidence score.
    """
    import argparse
    import logging
    import sys

    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="ww_trainer-infer",
        description="Score an audio file with a trained ONNX wake-word model.",
    )
    parser.add_argument("--featurizer", required=True, help="Path to featurizer ONNX file.")
    parser.add_argument("--model", required=True, help="Path to head ONNX file.")
    parser.add_argument("--audio", required=True, help="Path to WAV file (16 kHz mono float32).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (default 0.5).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    try:
        import soundfile as sf
    except ImportError:
        try:
            import torchaudio
            wav, sr = torchaudio.load(args.audio)
            if sr != 16000:
                import torchaudio.functional as F_ta
                wav = F_ta.resample(wav, sr, 16000)
            audio = wav.squeeze().numpy().astype(np.float32)
        except ImportError:
            print("Install soundfile or torchaudio to load audio files.", file=sys.stderr)
            sys.exit(1)
    else:
        audio, sr = sf.read(args.audio, dtype="float32", always_2d=False)
        if sr != 16000:
            print(f"Warning: sample rate is {sr} Hz, expected 16000.", file=sys.stderr)

    model = OnnxWakeWordInferencer(args.featurizer, args.model, device=args.device)
    score = model.infer(audio)
    detected = score >= args.threshold
    print(f"score={score:.4f}  threshold={args.threshold}  detected={detected}")
    sys.exit(0 if detected else 1)


if __name__ == "__main__":
    cli_main()
