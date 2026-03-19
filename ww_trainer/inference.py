"""ONNX-only wake word inference — no PyTorch dependency at runtime."""
from __future__ import annotations

import numpy as np
import onnxruntime as ort


class OnnxWakeWordInferencer:
    """Run wake word detection using two ONNX sessions (extractor + head).

    This class has no dependency on PyTorch. Only numpy and onnxruntime are required.

    Args:
        extractor_path: Path to the feature extractor ONNX file.
        head_path: Path to the classifier head ONNX file.
        sample_rate: Expected audio sample rate in Hz (default 16000).
        device: "cpu", "cuda", or "auto" (selects CUDA if available).
    """

    def __init__(self, extractor_path: str, head_path: str,
                 sample_rate: int = 16000, device: str = "auto") -> None:
        if device == "auto":
            available = [p.device_type() for p in ort.get_all_providers()]
            device = "cuda" if "CUDAExecutionProvider" in available else "cpu"
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        self.extractor = ort.InferenceSession(extractor_path, providers=providers)
        self.head = ort.InferenceSession(head_path, providers=providers)
        self.sample_rate = sample_rate

        self._ext_input = self.extractor.get_inputs()[0].name
        self._ext_output = self.extractor.get_outputs()[0].name
        self._head_input = self.head.get_inputs()[0].name
        self._head_output = self.head.get_outputs()[0].name

    def infer(self, audio: np.ndarray) -> float:
        """Infer wake word probability for a single audio waveform.

        Args:
            audio: 1-D float32 numpy array at self.sample_rate.

        Returns:
            Sigmoid probability in [0, 1].
        """
        if audio.ndim != 1:
            raise ValueError("audio must be a 1-D numpy array")
        wav = audio[np.newaxis, :].astype(np.float32)  # [1, T]
        feats = self.extractor.run(
            [self._ext_output], {self._ext_input: wav}
        )[0]  # [1, T, F]
        logit = self.head.run(
            [self._head_output], {self._head_input: feats}
        )[0]  # [1] or scalar
        logit_val = float(np.asarray(logit).ravel()[0])
        return float(1.0 / (1.0 + np.exp(-logit_val)))

    def infer_batch(self, audio_batch: np.ndarray) -> np.ndarray:
        """Infer for a batch of equal-length waveforms.

        Args:
            audio_batch: float32 array of shape [B, T].

        Returns:
            Sigmoid probabilities array of shape [B].
        """
        if audio_batch.ndim != 2:
            raise ValueError("audio_batch must be a 2-D numpy array [B, T]")
        wav = audio_batch.astype(np.float32)
        feats = self.extractor.run([self._ext_output], {self._ext_input: wav})[0]
        logits = self.head.run([self._head_output], {self._head_input: feats})[0]
        logits = np.asarray(logits).ravel()
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def infer_streaming(self, audio_chunk: np.ndarray,
                        cache: np.ndarray | None) -> tuple[float, np.ndarray]:
        """Process one audio chunk with a rolling feature cache.

        Args:
            audio_chunk: 1-D float32 array (one chunk of audio).
            cache: Previous feature cache [T_cached, F] or None for first call.

        Returns:
            (probability, updated_cache)
        """
        wav = audio_chunk[np.newaxis, :].astype(np.float32)  # [1, T_chunk]
        new_feats = self.extractor.run(
            [self._ext_output], {self._ext_input: wav}
        )[0].squeeze(0)  # [T_new, F]

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
        return prob, cache
