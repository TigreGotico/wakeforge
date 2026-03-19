"""Smoke tests for few-shot embedding detection."""
import json

import numpy as np

from ww_trainer.few_shot import load_reference, FewShotDetector


def _make_reference(tmp_path, n_refs: int = 3, dim: int = 64) -> str:
    """Create a synthetic reference JSON file."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(n_refs, dim).tolist()
    ref = {"embeddings": embeddings, "n_samples": n_refs, "model_type": "test"}
    path = tmp_path / "ref.json"
    with open(path, "w") as f:
        json.dump(ref, f)
    return str(path)


class TestLoadReference:
    def test_load_valid(self, tmp_path) -> None:
        path = _make_reference(tmp_path)
        ref = load_reference(path)
        assert len(ref["embeddings"]) == 3
        assert ref["n_samples"] == 3


class TestFewShotDetector:
    def test_cosine_max_match(self, tmp_path) -> None:
        path = _make_reference(tmp_path, n_refs=2, dim=8)
        ref = load_reference(path)
        detector = FewShotDetector(path, threshold=0.5, method="cosine_max")
        # Use first ref embedding as query — should match perfectly
        emb = np.array(ref["embeddings"][0], dtype=np.float32)
        result = detector.detect_from_embedding(emb)
        assert result["match"] is True
        assert result["confidence"] > 0.99

    def test_cosine_mean_self_high(self, tmp_path) -> None:
        # Create refs where all embeddings are the same -> mean cosine = 1.0
        ref_data = {"embeddings": [[1.0, 0.0, 0.0]] * 3, "n_samples": 3, "model_type": "test"}
        path = tmp_path / "ref_same.json"
        with open(path, "w") as f:
            json.dump(ref_data, f)
        detector = FewShotDetector(str(path), threshold=0.9, method="cosine_mean")
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = detector.detect_from_embedding(emb)
        assert result["match"] is True
        assert result["confidence"] > 0.99

    def test_no_match_random(self, tmp_path) -> None:
        path = _make_reference(tmp_path, n_refs=3, dim=64)
        detector = FewShotDetector(path, threshold=0.99, method="cosine_max")
        rng = np.random.RandomState(999)
        emb = rng.randn(64).astype(np.float32)
        result = detector.detect_from_embedding(emb)
        # Random vector unlikely to match at 0.99 threshold
        assert result["confidence"] < 0.99

    def test_detect_without_model_raises(self, tmp_path) -> None:
        path = _make_reference(tmp_path)
        detector = FewShotDetector(path, model=None)
        try:
            detector.detect(np.zeros(16000, dtype=np.float32))
            assert False, "Expected RuntimeError"
        except RuntimeError:
            pass

    def test_invalid_method(self, tmp_path) -> None:
        path = _make_reference(tmp_path)
        try:
            FewShotDetector(path, method="invalid")
            assert False, "Expected ValueError"
        except ValueError:
            pass
