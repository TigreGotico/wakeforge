"""Few-shot wake word detection via reference embeddings.

Inspired by EfficientWord-Net (max cosine similarity to N reference
embeddings) and openWakeWord (LogisticRegression verifier on top of
embeddings).

Usage::

    from ww_trainer.few_shot import generate_reference, FewShotDetector

    # Generate reference embeddings from a few samples
    generate_reference(model, ["wake1.wav", "wake2.wav"], "ref.json")

    # Detect using cosine similarity
    detector = FewShotDetector("ref.json", model, threshold=0.7)
    result = detector.detect(audio_np)
    print(result)  # {"match": True, "confidence": 0.85}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def generate_reference(
    model: Any,
    audio_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    """Extract embeddings from audio files and save as a reference JSON.

    Args:
        model: A model with an ``embed()`` method (BaseWakeModel or similar).
            Must accept a list of 1-D tensors.
        audio_paths: Paths to wake word audio samples.
        output_path: Path to save the reference JSON file.
        sample_rate: Expected sample rate of audio files.

    Returns:
        The reference dict (also saved to ``output_path``).
    """
    import soundfile as sf
    import torch

    embeddings: List[List[float]] = []
    for path in audio_paths:
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            except ImportError:
                logger.warning("librosa not available for resampling %s", path)
                continue

        wav_tensor = torch.tensor(audio, dtype=torch.float32)
        with torch.no_grad():
            emb = model.embed([wav_tensor])  # [1, D]
        embeddings.append(emb.squeeze(0).cpu().numpy().tolist())

    ref = {
        "embeddings": embeddings,
        "n_samples": len(embeddings),
        "model_type": type(model).__name__,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ref, f, indent=2)

    logger.info("Saved %d reference embeddings to %s", len(embeddings), output_path)
    return ref


def load_reference(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a reference embedding file.

    Args:
        path: Path to reference JSON.

    Returns:
        Dict with keys ``"embeddings"`` (list of lists), ``"n_samples"``,
        ``"model_type"``.
    """
    with open(str(path), "r") as f:
        return json.load(f)


class FewShotDetector:
    """Detect wake words by comparing embeddings to stored references.

    Inspired by EfficientWord-Net's cosine similarity detection and
    openWakeWord's LogisticRegression verifier.

    Args:
        reference_path: Path to reference JSON from ``generate_reference()``.
        model: Model with ``embed()`` method (PyTorch) or None if using
            raw embeddings externally.
        threshold: Cosine similarity threshold for detection.
        method: ``"cosine_max"`` (max similarity to any reference) or
            ``"cosine_mean"`` (mean similarity to all references).
    """

    def __init__(
        self,
        reference_path: Union[str, Path],
        model: Any = None,
        threshold: float = 0.7,
        method: str = "cosine_max",
    ) -> None:
        if method not in ("cosine_max", "cosine_mean"):
            raise ValueError(f"Unknown method: {method!r}. Use 'cosine_max' or 'cosine_mean'.")
        ref = load_reference(reference_path)
        self.ref_embeddings = np.array(ref["embeddings"], dtype=np.float32)
        # L2-normalize reference embeddings
        norms = np.linalg.norm(self.ref_embeddings, axis=1, keepdims=True)
        self.ref_embeddings = self.ref_embeddings / np.maximum(norms, 1e-8)
        self.model = model
        self.threshold = threshold
        self.method = method

    def detect_from_embedding(self, embedding: np.ndarray) -> Dict[str, Any]:
        """Detect from a pre-computed embedding.

        Args:
            embedding: 1-D float32 numpy array (embedding vector).

        Returns:
            ``{"match": bool, "confidence": float}``
        """
        emb = embedding / max(np.linalg.norm(embedding), 1e-8)
        sims = self.ref_embeddings @ emb  # [N]

        if self.method == "cosine_max":
            confidence = float(np.max(sims))
        else:
            confidence = float(np.mean(sims))

        return {"match": confidence >= self.threshold, "confidence": confidence}

    def detect(self, audio: np.ndarray) -> Dict[str, Any]:
        """Detect wake word from raw audio.

        Args:
            audio: 1-D float32 numpy array.

        Returns:
            ``{"match": bool, "confidence": float}``

        Raises:
            RuntimeError: If no model was provided.
        """
        if self.model is None:
            raise RuntimeError("No model provided — use detect_from_embedding() instead.")

        import torch
        wav_tensor = torch.tensor(audio, dtype=torch.float32)
        with torch.no_grad():
            emb = self.model.embed([wav_tensor])  # [1, D]
        emb_np = emb.squeeze(0).cpu().numpy()
        return self.detect_from_embedding(emb_np)


def train_verifier(
    model: Any,
    reference_embeddings: np.ndarray,
    negative_audio_paths: List[Union[str, Path]],
    output_path: Optional[Union[str, Path]] = None,
    sample_rate: int = 16000,
) -> Any:
    """Train a LogisticRegression verifier on reference + negative embeddings.

    Second-stage classifier inspired by openWakeWord's verification pattern.

    Args:
        model: Model with ``embed()`` method.
        reference_embeddings: Positive reference embeddings ``[N_pos, D]``.
        negative_audio_paths: Paths to negative audio samples.
        output_path: Optional path to save the verifier via joblib.
        sample_rate: Audio sample rate.

    Returns:
        Fitted sklearn LogisticRegression classifier.
    """
    import soundfile as sf
    import torch
    from sklearn.linear_model import LogisticRegression

    # Collect negative embeddings
    neg_embs: List[np.ndarray] = []
    for path in negative_audio_paths:
        try:
            audio, sr = sf.read(str(path), dtype="float32")
        except Exception:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            except ImportError:
                continue

        wav_tensor = torch.tensor(audio, dtype=torch.float32)
        with torch.no_grad():
            emb = model.embed([wav_tensor])
        neg_embs.append(emb.squeeze(0).cpu().numpy())

    if not neg_embs:
        raise ValueError("No valid negative samples found.")

    X_pos = reference_embeddings
    X_neg = np.stack(neg_embs)
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X, y)

    if output_path is not None:
        import joblib
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, str(output_path))
        logger.info("Saved verifier to %s", output_path)

    return clf
