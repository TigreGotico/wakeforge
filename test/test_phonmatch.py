"""Tests for PhonMatchNet concrete implementation (phonmatch.py)."""
from pathlib import Path

import pytest
import torch

from ww_trainer.phonmatch import (
    IPA_VOCAB,
    ARPABET_TO_IPA,
    PHONEME_VOCAB,
    VOCAB_SIZE,
    ipa_to_ids,
    arpabet_to_ids,
    phonemes_to_ids,
    PhonMatchTextEncoder,
    PhonMatchHead,
)
from ww_trainer.factory import create_model, HEAD_REGISTRY


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

class TestPhonemeVocab:
    def test_ipa_vocab_size_constant(self):
        assert VOCAB_SIZE == len(IPA_VOCAB) + 1

    def test_no_zero_index_in_ipa_vocab(self):
        assert 0 not in IPA_VOCAB.values()

    def test_phoneme_vocab_is_ipa_vocab(self):
        assert PHONEME_VOCAB is IPA_VOCAB

    def test_ipa_to_ids_known(self):
        ids = ipa_to_ids(["h", "eɪ", "m", "aɪ"])
        assert all(i > 0 for i in ids)
        assert len(ids) == 4

    def test_ipa_to_ids_unknown_maps_to_zero(self):
        assert ipa_to_ids(["NOT_A_PHONEME"]) == [0]

    def test_arpabet_to_ids_strips_stress(self):
        # HH EY1 M AY1 → h eɪ m aɪ — all should resolve
        ids = arpabet_to_ids(["HH", "EY1", "M", "AY1"])
        assert all(i > 0 for i in ids)
        assert len(ids) == 4

    def test_arpabet_to_ids_nostress_same_as_stressed(self):
        assert arpabet_to_ids(["EY"]) == arpabet_to_ids(["EY1"]) == arpabet_to_ids(["EY2"])

    def test_arpabet_to_ids_unknown_maps_to_zero(self):
        assert arpabet_to_ids(["NOTARPABET"]) == [0]

    def test_phonemes_to_ids_is_arpabet_compat_alias(self):
        assert phonemes_to_ids(["HH", "EY1"]) == arpabet_to_ids(["HH", "EY1"])

    def test_arpabet_to_ipa_table_complete(self):
        """Every ARPAbet symbol in ARPABET_TO_IPA should map to a valid IPA entry."""
        for arp, ipa in ARPABET_TO_IPA.items():
            assert ipa in IPA_VOCAB, f"IPA symbol {ipa!r} (from ARPAbet {arp!r}) missing in IPA_VOCAB"


# ---------------------------------------------------------------------------
# PhonMatchTextEncoder
# ---------------------------------------------------------------------------

class TestPhonMatchTextEncoder:
    def test_forward_shape(self):
        enc = PhonMatchTextEncoder(emb_dim=64)
        # "hey mycroft" in IPA
        ids = torch.tensor([ipa_to_ids(["h", "eɪ", "m", "aɪ", "k", "ɹ", "ʌ", "f", "t"])],
                           dtype=torch.long)
        out = enc(ids)
        assert out.shape == (1, 1, 64)

    def test_batch_forward(self):
        enc = PhonMatchTextEncoder(emb_dim=32)
        ids = torch.randint(1, VOCAB_SIZE, (4, 8))
        out = enc(ids)
        assert out.shape == (4, 1, 32)

    def test_padding_excluded_from_pool(self):
        """Padding tokens (id=0) must not change the pooled embedding."""
        enc = PhonMatchTextEncoder(emb_dim=32)
        enc.eval()
        ids_no_pad = torch.tensor([[1, 2, 3]], dtype=torch.long)
        ids_padded = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long)
        with torch.no_grad():
            assert torch.allclose(enc(ids_no_pad), enc(ids_padded), atol=1e-5)

    def test_gradients_flow(self):
        enc = PhonMatchTextEncoder(emb_dim=32)
        ids = torch.randint(1, VOCAB_SIZE, (2, 6))
        enc(ids).sum().backward()
        for p in enc.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break

    def test_export_to_onnx(self, tmp_path):
        enc = PhonMatchTextEncoder(emb_dim=32)
        out_path = str(tmp_path / "text_enc.onnx")
        enc.export_to_onnx(out_path, seq_len=10)
        assert Path(out_path).stat().st_size > 0

    def test_onnx_output_matches_pytorch(self, tmp_path):
        import numpy as np
        import onnxruntime as ort

        enc = PhonMatchTextEncoder(emb_dim=32)
        enc.eval()
        out_path = str(tmp_path / "text_enc.onnx")
        enc.export_to_onnx(out_path, seq_len=10)

        ids = torch.randint(1, VOCAB_SIZE, (1, 10))
        with torch.no_grad():
            pt_out = enc(ids).numpy()

        session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        ort_out = session.run(None, {"token_ids": ids.numpy()})[0]
        assert np.allclose(pt_out, ort_out, atol=1e-4)

    def test_onnx_integrates_with_onnx_text_extractor(self, tmp_path):
        from ww_trainer.feats import OnnxTextExtractor

        enc = PhonMatchTextEncoder(emb_dim=32)
        out_path = str(tmp_path / "text_enc.onnx")
        enc.export_to_onnx(out_path, seq_len=10)

        ext = OnnxTextExtractor(out_path, emb_dim=32)
        ids = ipa_to_ids(["h", "eɪ", "m", "aɪ", "k", "ɹ", "ʌ", "f", "t"])
        ext.precompute(ids)

        out = ext([torch.zeros(16000)] * 3)
        assert out.shape == (3, 1, 32)

    def test_arpabet_compat_path(self, tmp_path):
        """ARPAbet IDs (via arpabet_to_ids) work with the encoder."""
        from ww_trainer.feats import OnnxTextExtractor

        enc = PhonMatchTextEncoder(emb_dim=32)
        out_path = str(tmp_path / "text_enc.onnx")
        enc.export_to_onnx(out_path, seq_len=10)

        ext = OnnxTextExtractor(out_path, emb_dim=32)
        ids = arpabet_to_ids(["HH", "EY1", "M", "AY1", "K", "R", "AH0", "F", "T"])
        ext.precompute(ids)
        out = ext([torch.zeros(16000)])
        assert out.shape == (1, 1, 32)


# ---------------------------------------------------------------------------
# PhonMatchHead (cross-attention classifier)
# ---------------------------------------------------------------------------

HEY_MYCROFT_IPA = ipa_to_ids(["h", "eɪ", "m", "aɪ", "k", "ɹ", "ʌ", "f", "t"])
_IDS_T = torch.tensor([HEY_MYCROFT_IPA], dtype=torch.long)  # [1, P]
_IDS_BATCH = _IDS_T.expand(4, -1)                            # [4, P]


class TestPhonMatchHead:
    def test_forward_shape_batch_ids(self):
        """Per-batch [B, P] phoneme IDs."""
        head = PhonMatchHead(input_size=40, hidden_dim=32, device="cpu")
        out = head(torch.randn(4, 100, 40), phoneme_ids=_IDS_BATCH)
        assert out.shape == (4,)

    def test_forward_shape_shared_ids(self):
        """Shared [P] phoneme IDs broadcast across batch."""
        head = PhonMatchHead(input_size=40, hidden_dim=32, device="cpu")
        ids_1d = torch.tensor(HEY_MYCROFT_IPA, dtype=torch.long)  # [P]
        out = head(torch.randn(3, 100, 40), phoneme_ids=ids_1d)
        assert out.shape == (3,)

    def test_embed_shape(self):
        head = PhonMatchHead(input_size=40, hidden_dim=32, device="cpu")
        assert head.embed(torch.randn(2, 50, 40), phoneme_ids=_IDS_T.expand(2, -1)).shape == (2, 32)

    def test_gradients_flow(self):
        head = PhonMatchHead(input_size=40, hidden_dim=32, device="cpu")
        feats = torch.randn(2, 100, 40, requires_grad=True)
        head(feats, phoneme_ids=_IDS_T.expand(2, -1)).sum().backward()
        assert feats.grad is not None

    def test_no_fixed_buffer(self):
        """Head must not have a baked-in phoneme_ids buffer."""
        head = PhonMatchHead(input_size=40, hidden_dim=32, device="cpu")
        assert not hasattr(head, 'phoneme_ids') or 'phoneme_ids' not in dict(head.named_buffers())

    def test_forward_raises_without_ids(self):
        head = PhonMatchHead(input_size=40, hidden_dim=32, device="cpu")
        with pytest.raises(ValueError):
            head(torch.randn(2, 100, 40))

    def test_different_keywords_different_outputs(self):
        ids_a = torch.tensor([ipa_to_ids(["h", "eɪ"])], dtype=torch.long)
        ids_b = torch.tensor([ipa_to_ids(["ɑ", "l", "ɛ", "k", "s", "ʌ"])], dtype=torch.long)
        head = PhonMatchHead(input_size=40, hidden_dim=32, device="cpu")
        feats = torch.randn(1, 100, 40)
        with torch.no_grad():
            assert not torch.allclose(head(feats, phoneme_ids=ids_a), head(feats, phoneme_ids=ids_b))

    def test_export_to_onnx(self, tmp_path):
        head = PhonMatchHead(input_size=40, hidden_dim=32, device="cpu")
        out_path = str(tmp_path / "phonmatch_head.onnx")
        head.export_to_onnx(out_path)
        assert Path(out_path).stat().st_size > 0

    def test_export_has_two_inputs(self, tmp_path):
        """ONNX graph must have audio features AND phoneme_ids as inputs."""
        import numpy as np
        import onnxruntime as ort

        head = PhonMatchHead(input_size=40, hidden_dim=32, device="cpu")
        out_path = str(tmp_path / "phonmatch_head.onnx")
        head.export_to_onnx(out_path, seq_len=9)

        session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        input_names = [i.name for i in session.get_inputs()]
        assert "input_features" in input_names
        assert "phoneme_ids" in input_names

        ph_ids = np.array([HEY_MYCROFT_IPA], dtype=np.int64)
        logits = session.run(None, {
            "input_features": np.random.randn(1, 100, 40).astype(np.float32),
            "phoneme_ids": ph_ids,
        })[0]
        assert logits.shape == (1,)


# ---------------------------------------------------------------------------
# Factory registration
# ---------------------------------------------------------------------------

class TestPhonMatchFactory:
    def test_registered_in_head_registry(self):
        assert "phonmatch" in HEAD_REGISTRY

    def test_create_model_phonmatch_forward(self):
        model = create_model(
            "phonmatch", "mfcc", featurizer_type="mfcc", device="cpu",
            hidden_dim=32,
        )
        wavs = [torch.randn(16000) for _ in range(2)]
        ids = torch.tensor([HEY_MYCROFT_IPA, HEY_MYCROFT_IPA], dtype=torch.long)
        logits = model(wavs, text_token_ids=ids)
        assert logits.shape == (2,)
        assert torch.isfinite(logits).all()

    def test_create_model_phonmatch_arpabet_compat(self):
        ids_list = arpabet_to_ids(["HH", "EY1", "M", "AY1", "K", "R", "AH0", "F", "T"])
        ids = torch.tensor([ids_list] * 2, dtype=torch.long)
        model = create_model(
            "phonmatch", "mfcc", featurizer_type="mfcc", device="cpu",
            hidden_dim=32,
        )
        logits = model([torch.randn(16000)] * 2, text_token_ids=ids)
        assert logits.shape == (2,)

    def test_embed_via_factory(self):
        model = create_model(
            "phonmatch", "mfcc", featurizer_type="mfcc", device="cpu",
            hidden_dim=32,
        )
        ids = torch.tensor([HEY_MYCROFT_IPA] * 3, dtype=torch.long)
        embs = model.embed([torch.randn(16000)] * 3, text_token_ids=ids)
        assert embs.shape == (3, 32)
