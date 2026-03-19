"""ONNX-only wake word inference — no PyTorch dependency at runtime."""
from __future__ import annotations

import numpy as np
import onnxruntime as ort


class OnnxWakeWordInferencer:
    """Run wake word detection using multiple ONNX sessions.

    This class has no dependency on PyTorch. Only numpy and onnxruntime are required.

    Args:
        extractor_path: Path to the feature extractor ONNX file.
        head_path: Path to the classifier head ONNX file.
        vad_path: Optional path to a VAD ONNX file (e.g. Silero VAD).
        sample_rate: Expected audio sample rate in Hz (default 16000).
        device: "cpu", "cuda", or "auto" (selects CUDA if available).
    """

    def __init__(self, extractor_path: str, head_path: str,
                 vad_path: str | None = None,
                 sample_rate: int = 16000, device: str = "auto") -> None:
        if device == "auto":
            available = ort.get_available_providers()
            device = "cuda" if "CUDAExecutionProvider" in available else "cpu"
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        self.extractor = ort.InferenceSession(extractor_path, providers=providers)
        self.head = ort.InferenceSession(head_path, providers=providers)
        self.vad = ort.InferenceSession(vad_path, providers=providers) if vad_path else None
        self.sample_rate = sample_rate

        self._ext_input = self.extractor.get_inputs()[0].name
        self._ext_output = self.extractor.get_outputs()[0].name
        self._head_input = self.head.get_inputs()[0].name
        self._head_output = self.head.get_outputs()[0].name
        
        if self.vad:
            self._vad_input = self.vad.get_inputs()[0].name
            self._vad_output = self.vad.get_outputs()[0].name

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
        
        if self.vad:
            vad_probs = self._run_vad(wav, feats.shape[1])
            feats = np.concatenate([feats, vad_probs], axis=-1)

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
        
        if self.vad:
            vad_probs = self._run_vad(wav, feats.shape[1])
            feats = np.concatenate([feats, vad_probs], axis=-1)

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
        return prob, cache
