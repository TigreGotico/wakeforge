"""Tests for the optional text modality (OnnxTextExtractor + BaseWakeModel text conditioning)."""
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from ww_trainer.feats import OnnxTextExtractor
from ww_trainer.model import BaseWakeModel
from ww_trainer.factory import create_model


# ---------------------------------------------------------------------------
# Helpers — build a trivial text-encoder ONNX in a temp dir
# ---------------------------------------------------------------------------

def _build_dummy_text_onnx(out_path: str, emb_dim: int = 32, seq_len: int = 5) -> None:
    """Export a minimal ONNX that maps [1, seq_len] int64 → [1, 1, emb_dim] float32."""
    import onnx
    from onnx import helper, TensorProto

    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, seq_len])
    output_emb = helper.make_tensor_value_info("text_emb", TensorProto.FLOAT, [1, 1, emb_dim])

    # Constant node: ignore input, return fixed embedding
    const_val = np.zeros((1, 1, emb_dim), dtype=np.float32)
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["text_emb"],
        value=helper.make_tensor("val", TensorProto.FLOAT, const_val.shape, const_val.flatten().tolist()),
    )
    graph = helper.make_graph([const_node], "text_encoder", [input_ids], [output_emb])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, out_path)


# ---------------------------------------------------------------------------
# OnnxTextExtractor tests
# ---------------------------------------------------------------------------

class TestOnnxTextExtractor:

    def test_precompute_and_forward(self, tmp_path):
        onnx_path = str(tmp_path / "text_enc.onnx")
        _build_dummy_text_onnx(onnx_path, emb_dim=32)
        ext = OnnxTextExtractor(onnx_path, emb_dim=32)
        ext.precompute([1, 2, 3, 4, 0])
        dummy_wavs = [torch.zeros(16000)] * 4
        out = ext(dummy_wavs)
        assert out.shape == (4, 1, 32), f"Expected [4,1,32], got {out.shape}"

    def test_feature_dim(self, tmp_path):
        onnx_path = str(tmp_path / "text_enc.onnx")
        _build_dummy_text_onnx(onnx_path, emb_dim=64)
        ext = OnnxTextExtractor(onnx_path, emb_dim=64)
        assert ext.feature_dim == 64

    def test_forward_raises_without_precompute_or_ids(self, tmp_path):
        onnx_path = str(tmp_path / "text_enc.onnx")
        _build_dummy_text_onnx(onnx_path, emb_dim=32)
        ext = OnnxTextExtractor(onnx_path, emb_dim=32)
        with pytest.raises(RuntimeError):
            ext([torch.zeros(16000)])

    def test_forward_with_per_batch_token_ids(self, tmp_path):
        """Pass token_ids=[B, seq] to bypass precompute — training mode."""
        from ww_trainer.phonmatch import PhonMatchTextEncoder, ipa_to_ids

        enc = PhonMatchTextEncoder(emb_dim=32)
        onnx_path = str(tmp_path / "text_enc.onnx")
        enc.export_to_onnx(onnx_path, seq_len=5)

        ext = OnnxTextExtractor(onnx_path, emb_dim=32)
        ids = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.long)
        out = ext([torch.zeros(16000)] * 3, token_ids=ids)
        assert out.shape == (3, 1, 32)

    def test_export_to_onnx_raises(self, tmp_path):
        onnx_path = str(tmp_path / "text_enc.onnx")
        _build_dummy_text_onnx(onnx_path, emb_dim=32)
        ext = OnnxTextExtractor(onnx_path, emb_dim=32)
        with pytest.raises(NotImplementedError):
            ext.export_to_onnx("out.onnx")

    def test_batch_expansion(self, tmp_path):
        onnx_path = str(tmp_path / "text_enc.onnx")
        _build_dummy_text_onnx(onnx_path, emb_dim=16)
        ext = OnnxTextExtractor(onnx_path, emb_dim=16)
        ext.precompute([1, 0, 0, 0, 0])
        for batch_size in (1, 3, 8):
            out = ext([torch.zeros(16000)] * batch_size)
            assert out.shape == (batch_size, 1, 16)


# ---------------------------------------------------------------------------
# BaseWakeModel text conditioning tests
# ---------------------------------------------------------------------------

class TestBaseWakeModelTextConditioning:

    def _make_model(self, tmp_path, emb_dim=32, audio_feat_dim=40, hidden=32):
        from ww_trainer.feats import MfccExtractor
        from ww_trainer.model import FfnClassifierHead

        onnx_path = str(tmp_path / "text_enc.onnx")
        _build_dummy_text_onnx(onnx_path, emb_dim=emb_dim)
        text_ext = OnnxTextExtractor(onnx_path, emb_dim=emb_dim)
        text_ext.precompute([1, 2, 3, 4, 0])

        extractor = MfccExtractor(sr=16000, n_mfcc=audio_feat_dim)
        head = FfnClassifierHead(
            input_size=audio_feat_dim + emb_dim,
            hidden_dim=hidden,
            device="cpu",
        )
        model = BaseWakeModel(
            extractor, head, device="cpu",
            text_extractor=text_ext, keyword="hey test",
        )
        return model

    def test_forward_with_text_produces_scalar_per_sample(self, tmp_path):
        model = self._make_model(tmp_path)
        wavs = [torch.randn(16000) for _ in range(4)]
        logits = model(wavs)
        assert logits.shape == (4,)

    def test_embed_with_text_produces_embeddings(self, tmp_path):
        model = self._make_model(tmp_path, emb_dim=32, hidden=32)
        wavs = [torch.randn(16000) for _ in range(3)]
        embs = model.embed(wavs)
        assert embs.ndim == 2
        assert embs.shape[0] == 3

    def test_keyword_stored_on_model(self, tmp_path):
        model = self._make_model(tmp_path)
        assert model.keyword == "hey test"

    def test_forward_without_text_extractor_unchanged(self):
        """Model without text_extractor behaves exactly as before."""
        from ww_trainer.feats import MfccExtractor
        from ww_trainer.model import FfnClassifierHead

        extractor = MfccExtractor(sr=16000, n_mfcc=40)
        head = FfnClassifierHead(input_size=40, hidden_dim=32, device="cpu")
        model = BaseWakeModel(extractor, head, device="cpu")
        wavs = [torch.randn(16000) for _ in range(2)]
        logits = model(wavs)
        assert logits.shape == (2,)
        assert model.text_extractor is None
        assert model.keyword is None


# ---------------------------------------------------------------------------
# factory.py / create_model integration tests
# ---------------------------------------------------------------------------

class TestCreateModelTextModality:

    def test_create_model_without_text_featurizer(self):
        model = create_model("ffn", "mfcc", featurizer_type="mfcc", device="cpu")
        assert model.text_extractor is None
        assert model.keyword is None

    def test_keyword_stored_via_factory(self):
        model = create_model("ffn", "mfcc", featurizer_type="mfcc", device="cpu",
                             keyword="hey mycroft")
        assert model.keyword == "hey mycroft"

    def test_create_model_with_text_featurizer(self, tmp_path):
        onnx_path = str(tmp_path / "text_enc.onnx")
        _build_dummy_text_onnx(onnx_path, emb_dim=32)
        model = create_model(
            "ffn", "mfcc", featurizer_type="mfcc", device="cpu",
            text_featurizer=onnx_path, text_emb_dim=32,
            keyword="hey test",
        )
        assert model.text_extractor is not None
        # head input_size should be 40 (mfcc) + 32 (text) = 72
        assert model.classifier.input_size == 72

    def test_forward_via_factory_with_text(self, tmp_path):
        onnx_path = str(tmp_path / "text_enc.onnx")
        _build_dummy_text_onnx(onnx_path, emb_dim=32)
        model = create_model(
            "ffn", "mfcc", featurizer_type="mfcc", device="cpu",
            text_featurizer=onnx_path, text_emb_dim=32,
            keyword="hey test",
        )
        model.text_extractor.precompute([1, 2, 3, 4, 0])
        wavs = [torch.randn(16000) for _ in range(2)]
        logits = model(wavs)
        assert logits.shape == (2,)
        assert torch.isfinite(logits).all()
