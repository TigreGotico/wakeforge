"""Tests for PhonMatchNet concrete implementation (phonmatch.py)."""
import tempfile
from pathlib import Path

import pytest
import torch

from ww_trainer.phonmatch import (
    PHONEME_VOCAB,
    VOCAB_SIZE,
    phonemes_to_ids,
    PhonMatchTextEncoder,
    PhonMatchHead,
)
from ww_trainer.factory import create_model, HEAD_REGISTRY


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

class TestPhonemeVocab:
    def test_vocab_size_constant(self):
        assert VOCAB_SIZE == len(PHONEME_VOCAB) + 1

    def test_no_zero_index_in_vocab(self):
        assert 0 not in PHONEME_VOCAB.values()

    def test_phonemes_to_ids_known(self):
        ids = phonemes_to_ids(["HH", "EY", "M", "AY"])
        assert all(i > 0 for i in ids)
        assert len(ids) == 4

    def test_phonemes_to_ids_unknown_maps_to_zero(self):
        ids = phonemes_to_ids(["UNKNOWN_PHONEME"])
        assert ids == [0]

    def test_phonemes_to_ids_case_insensitive(self):
        assert phonemes_to_ids(["hh"]) == phonemes_to_ids(["HH"])


# ---------------------------------------------------------------------------
# PhonMatchTextEncoder
# ---------------------------------------------------------------------------

class TestPhonMatchTextEncoder:
    def test_forward_shape(self):
        enc = PhonMatchTextEncoder(emb_dim=64)
        ids = torch.tensor([[43, 37, 55, 21, 53, 67, 9, 41, 70]], dtype=torch.long)
        out = enc(ids)
        assert out.shape == (1, 1, 64)

    def test_batch_forward(self):
        enc = PhonMatchTextEncoder(emb_dim=32)
        ids = torch.randint(1, VOCAB_SIZE, (4, 8))
        out = enc(ids)
        assert out.shape == (4, 1, 32)

    def test_padding_excluded_from_pool(self):
        """Padding tokens (id=0) should not affect the pooled embedding."""
        enc = PhonMatchTextEncoder(emb_dim=32)
        enc.eval()
        ids_no_pad = torch.tensor([[1, 2, 3]], dtype=torch.long)
        ids_padded = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long)
        with torch.no_grad():
            out_no_pad = enc(ids_no_pad)
            out_padded = enc(ids_padded)
        assert torch.allclose(out_no_pad, out_padded, atol=1e-5)

    def test_gradients_flow(self):
        enc = PhonMatchTextEncoder(emb_dim=32)
        ids = torch.randint(1, VOCAB_SIZE, (2, 6))
        out = enc(ids)
        out.sum().backward()
        for p in enc.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break

    def test_export_to_onnx(self, tmp_path):
        enc = PhonMatchTextEncoder(emb_dim=32)
        out_path = str(tmp_path / "text_enc.onnx")
        enc.export_to_onnx(out_path, seq_len=8)
        assert Path(out_path).exists()
        assert Path(out_path).stat().st_size > 0

    def test_onnx_output_matches_pytorch(self, tmp_path):
        import numpy as np
        import onnxruntime as ort

        enc = PhonMatchTextEncoder(emb_dim=32)
        enc.eval()
        out_path = str(tmp_path / "text_enc.onnx")
        enc.export_to_onnx(out_path, seq_len=8)

        ids = torch.randint(1, VOCAB_SIZE, (1, 8))
        with torch.no_grad():
            pt_out = enc(ids).numpy()

        session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        ort_out = session.run(None, {"token_ids": ids.numpy()})[0]
        assert np.allclose(pt_out, ort_out, atol=1e-4)

    def test_onnx_integrates_with_onnx_text_extractor(self, tmp_path):
        from ww_trainer.feats import OnnxTextExtractor

        enc = PhonMatchTextEncoder(emb_dim=32)
        out_path = str(tmp_path / "text_enc.onnx")
        enc.export_to_onnx(out_path, seq_len=8)

        ext = OnnxTextExtractor(out_path, emb_dim=32)
        ids = phonemes_to_ids(["HH", "EY", "M", "AY", "K", "R", "AH", "F"])
        ext.precompute(ids)

        dummy_wavs = [torch.zeros(16000)] * 3
        out = ext(dummy_wavs)
        assert out.shape == (3, 1, 32)


# ---------------------------------------------------------------------------
# PhonMatchHead (cross-attention classifier)
# ---------------------------------------------------------------------------

HEY_MYCROFT_IDS = phonemes_to_ids(["HH", "EY", "M", "AY", "K", "R", "AH", "F", "T"])


class TestPhonMatchHead:
    def test_forward_shape(self):
        head = PhonMatchHead(
            input_size=40,
            keyword_token_ids=HEY_MYCROFT_IDS,
            hidden_dim=32,
            device="cpu",
        )
        feats = torch.randn(4, 100, 40)
        out = head(feats)
        assert out.shape == (4,)

    def test_embed_shape(self):
        head = PhonMatchHead(input_size=40, keyword_token_ids=HEY_MYCROFT_IDS, hidden_dim=32, device="cpu")
        feats = torch.randn(2, 50, 40)
        emb = head.embed(feats)
        assert emb.shape == (2, 32)

    def test_gradients_flow(self):
        head = PhonMatchHead(input_size=40, keyword_token_ids=HEY_MYCROFT_IDS, hidden_dim=32, device="cpu")
        feats = torch.randn(2, 100, 40, requires_grad=True)
        out = head(feats)
        out.sum().backward()
        assert feats.grad is not None

    def test_phoneme_ids_stored_as_buffer(self):
        head = PhonMatchHead(input_size=40, keyword_token_ids=HEY_MYCROFT_IDS, hidden_dim=32, device="cpu")
        assert hasattr(head, "phoneme_ids")
        assert isinstance(head.phoneme_ids, torch.Tensor)
        assert head.phoneme_ids.tolist() == HEY_MYCROFT_IDS

    def test_different_keywords_different_outputs(self):
        """Two heads with different keywords should produce different logits on the same audio."""
        ids_a = phonemes_to_ids(["HH", "EY"])
        ids_b = phonemes_to_ids(["AH", "L", "EH", "K", "S", "AH"])
        head_a = PhonMatchHead(input_size=40, keyword_token_ids=ids_a, hidden_dim=32, device="cpu")
        head_b = PhonMatchHead(input_size=40, keyword_token_ids=ids_b, hidden_dim=32, device="cpu")
        feats = torch.randn(2, 100, 40)
        with torch.no_grad():
            out_a = head_a(feats)
            out_b = head_b(feats)
        assert not torch.allclose(out_a, out_b)

    def test_export_to_onnx(self, tmp_path):
        head = PhonMatchHead(input_size=40, keyword_token_ids=HEY_MYCROFT_IDS, hidden_dim=32, device="cpu")
        out_path = str(tmp_path / "phonmatch_head.onnx")
        head.export_to_onnx(out_path)
        assert Path(out_path).exists()
        assert Path(out_path).stat().st_size > 0

    def test_export_phoneme_ids_baked_in(self, tmp_path):
        """Exported ONNX takes only audio features — no token_ids input."""
        import onnxruntime as ort
        import numpy as np

        head = PhonMatchHead(input_size=40, keyword_token_ids=HEY_MYCROFT_IDS, hidden_dim=32, device="cpu")
        out_path = str(tmp_path / "phonmatch_head.onnx")
        head.export_to_onnx(out_path)

        session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        input_names = [i.name for i in session.get_inputs()]
        assert input_names == ["input_features"], f"Expected only audio input, got: {input_names}"

        feats = np.random.randn(2, 100, 40).astype(np.float32)
        logits = session.run(None, {"input_features": feats})[0]
        assert logits.shape == (2,)


# ---------------------------------------------------------------------------
# Factory registration
# ---------------------------------------------------------------------------

class TestPhonMatchFactory:
    def test_registered_in_head_registry(self):
        assert "phonmatch" in HEAD_REGISTRY

    def test_create_model_phonmatch(self):
        model = create_model(
            "phonmatch", "mfcc", featurizer_type="mfcc", device="cpu",
            keyword_token_ids=HEY_MYCROFT_IDS,
            hidden_dim=32,
        )
        assert model is not None
        wavs = [torch.randn(16000) for _ in range(2)]
        logits = model(wavs)
        assert logits.shape == (2,)
        assert torch.isfinite(logits).all()

    def test_embed_via_factory(self):
        model = create_model(
            "phonmatch", "mfcc", featurizer_type="mfcc", device="cpu",
            keyword_token_ids=HEY_MYCROFT_IDS,
            hidden_dim=32,
        )
        wavs = [torch.randn(16000) for _ in range(3)]
        embs = model.embed(wavs)
        assert embs.shape == (3, 32)
