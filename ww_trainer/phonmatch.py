"""PhonMatchNet text encoder — concrete implementation of the text featurizer.

Implements the text encoder from PhonMatchNet (INTERSPEECH 2023, ncsoft/PhonMatchNet),
extended to use the **International Phonetic Alphabet (IPA)** instead of ARPAbet.
IPA covers all world languages with a single consistent symbol set; ARPAbet is
English-only and a strictly inferior choice for multilingual wake-word detection.

Primary vocabulary: :data:`IPA_VOCAB` — 159 IPA symbols (index 1–159).
Index 0 is reserved as the unknown/padding token.

An :data:`ARPABET_TO_IPA` table is provided for users migrating from ARPAbet-based G2P
tools (e.g. g2p_en); use :func:`arpabet_to_ids` for that path.

Usage (train + export)::

    from ww_trainer.phonmatch import PhonMatchTextEncoder, ipa_to_ids

    # IPA phonemes produced externally (espeak-ng, phonemizer, gruut, etc.)
    ids = ipa_to_ids(["h", "eɪ", "m", "aɪ", "k", "ɹ", "ʌ", "f", "t"])

    encoder = PhonMatchTextEncoder(emb_dim=128)
    encoder.export_to_onnx("phoneme_encoder.onnx")

    # Then use with OnnxTextExtractor:
    from ww_trainer.feats import OnnxTextExtractor
    text_ext = OnnxTextExtractor("phoneme_encoder.onnx", emb_dim=128)
    text_ext.precompute(ids)

    # ARPAbet compat (e.g. output of g2p_en):
    from ww_trainer.phonmatch import arpabet_to_ids
    ids = arpabet_to_ids(["HH", "EY1", "M", "AY1", "K", "R", "AH0", "F", "T"])
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IPA vocabulary — 159 symbols covering major world languages
# Organised by IPA chart section for readability.
# Index 0 is reserved as unknown/padding (excluded from mean-pool).
# ---------------------------------------------------------------------------

IPA_VOCAB: Dict[str, int] = {
    # --- Pulmonic consonants: plosives ---
    "p": 1,   "b": 2,   "t": 3,   "d": 4,   "ʈ": 5,   "ɖ": 6,
    "c": 7,   "ɟ": 8,   "k": 9,   "ɡ": 10,  "q": 11,  "ɢ": 12,
    "ʔ": 13,
    # --- Nasals ---
    "m": 14,  "ɱ": 15,  "n": 16,  "ɳ": 17,  "ɲ": 18,  "ŋ": 19,  "ɴ": 20,
    # --- Trills ---
    "ʙ": 21,  "r": 22,  "ʀ": 23,
    # --- Taps/flaps ---
    "ⱱ": 24,  "ɾ": 25,  "ɽ": 26,
    # --- Fricatives ---
    "ɸ": 27,  "β": 28,  "f": 29,  "v": 30,  "θ": 31,  "ð": 32,
    "s": 33,  "z": 34,  "ʃ": 35,  "ʒ": 36,  "ʂ": 37,  "ʐ": 38,
    "ç": 39,  "ʝ": 40,  "x": 41,  "ɣ": 42,  "χ": 43,  "ʁ": 44,
    "ħ": 45,  "ʕ": 46,  "h": 47,  "ɦ": 48,
    # --- Lateral fricatives ---
    "ɬ": 49,  "ɮ": 50,
    # --- Approximants ---
    "ʋ": 51,  "ɹ": 52,  "ɻ": 53,  "j": 54,  "ɰ": 55,
    # --- Lateral approximants ---
    "l": 56,  "ɭ": 57,  "ʎ": 58,  "ʟ": 59,
    # --- Affricates (common) ---
    "tʃ": 60,  "dʒ": 61,  "ts": 62,  "dz": 63,  "tɕ": 64,  "dʑ": 65,
    "ʈʂ": 66,  "ɖʐ": 67,  "pf": 68,  "bv": 69,
    # --- Close vowels ---
    "i": 70,  "y": 71,  "ɨ": 72,  "ʉ": 73,  "ɯ": 74,  "u": 75,
    # --- Near-close vowels ---
    "ɪ": 76,  "ʏ": 77,  "ʊ": 78,
    # --- Close-mid vowels ---
    "e": 79,  "ø": 80,  "ɘ": 81,  "ɵ": 82,  "ɤ": 83,  "o": 84,
    # --- Mid vowels ---
    "e̞": 85,  "ə": 86,  "ɵ̞": 87,
    # --- Open-mid vowels ---
    "ɛ": 88,  "œ": 89,  "ɜ": 90,  "ɞ": 91,  "ʌ": 92,  "ɔ": 93,
    # --- Near-open vowels ---
    "æ": 94,  "ɐ": 95,
    # --- Open vowels ---
    "a": 96,  "ɶ": 97,  "ä": 98,  "ɑ": 99,  "ɒ": 100,
    # --- Common diphthongs (English, German, …) ---
    "eɪ": 101,  "aɪ": 102,  "ɔɪ": 103,  "aʊ": 104,  "əʊ": 105,
    "oʊ": 106,  "ɪə": 107,  "eə": 108,  "ʊə": 109,
    # --- Suprasegmentals ---
    "ˈ": 110,  "ˌ": 111,  "ː": 112,  "ˑ": 113,
    # --- Tone letters (common) ---
    "˥": 114,  "˦": 115,  "˧": 116,  "˨": 117,  "˩": 118,
    # --- Click consonants ---
    "ʘ": 119,  "ǀ": 120,  "ǃ": 121,  "ǂ": 122,  "ǁ": 123,
    # --- Voiced implosives ---
    "ɓ": 124,  "ɗ": 125,  "ʄ": 126,  "ɠ": 127,  "ʛ": 128,
    # --- Additional consonants in major languages ---
    "w": 129,  "ɥ": 130,  "ʍ": 131,   # labial-velars
    "ɕ": 132,  "ʑ": 133,               # alveolopalatal fricatives (Mandarin, Polish…)
    "ɺ": 134,                           # lateral flap (Japanese)
    "ʈ͡ʂ": 135, "ɖ͡ʐ": 136,             # retroflex affricates
    "n̪": 137,  "t̪": 138,  "d̪": 139,   # dental variants
    "ɫ": 140,                           # velarised l (Russian, Portuguese…)
    "ʲ": 141,                           # palatalisation diacritic
    # --- Rhotic vowels (English) ---
    "ɚ": 142,  "ɝ": 143,
    # --- Nasalised vowels (French, Portuguese, …) ---
    "ã": 144,  "ẽ": 145,  "ĩ": 146,  "õ": 147,  "ũ": 148,
    "ɑ̃": 149,  "ɛ̃": 150,  "œ̃": 151,  "ɔ̃": 152,
    # --- Extra approximants ---
    "ɴ̥": 153,
    # --- Lateral click ---
    "ʗ": 154,
    # --- Additional affricates ---
    "tθ": 155, "dð": 156,
    # --- Syllabic consonants ---
    "n̩": 157, "l̩": 158, "m̩": 159,
}

VOCAB_SIZE: int = len(IPA_VOCAB) + 1  # +1 for index-0 unknown/padding

# ---------------------------------------------------------------------------
# ARPAbet → IPA conversion table (English, for g2p_en / CMUdict users)
# Stress digits (0/1/2) are stripped before lookup.
# ---------------------------------------------------------------------------

ARPABET_TO_IPA: Dict[str, str] = {
    "AA": "ɑ",  "AE": "æ",  "AH": "ʌ",  "AO": "ɔ",
    "AW": "aʊ", "AY": "aɪ", "B":  "b",   "CH": "tʃ",
    "D":  "d",  "DH": "ð",  "EH": "ɛ",  "ER": "ɝ",
    "EY": "eɪ", "F":  "f",  "G":  "ɡ",  "HH": "h",
    "IH": "ɪ",  "IY": "i",  "JH": "dʒ", "K":  "k",
    "L":  "l",  "M":  "m",  "N":  "n",  "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P":  "p",  "R":  "ɹ",
    "S":  "s",  "SH": "ʃ",  "T":  "t",  "TH": "θ",
    "UH": "ʊ",  "UW": "u",  "V":  "v",  "W":  "w",
    "Y":  "j",  "Z":  "z",  "ZH": "ʒ",
}

# Backward-compat alias (the original PhonMatchNet ARPAbet vocab is superseded)
PHONEME_VOCAB = IPA_VOCAB


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def ipa_to_ids(phonemes: List[str]) -> List[int]:
    """Map a sequence of IPA symbols to integer token IDs.

    Unknown symbols (including stress marks if not in vocab) silently map to 0
    (the padding index, excluded from mean-pool).

    Args:
        phonemes: List of IPA symbol strings, e.g. ``["h", "eɪ", "m", "aɪ"]``.

    Returns:
        List of integer IDs, same length as ``phonemes``.
    """
    return [IPA_VOCAB.get(p, 0) for p in phonemes]


def arpabet_to_ids(phonemes: List[str]) -> List[int]:
    """Convert ARPAbet phoneme strings (e.g. from g2p_en) to IPA token IDs.

    Stress digits (0/1/2) appended to vowels are stripped before conversion.
    Unknown ARPAbet symbols map to 0.

    Args:
        phonemes: List of ARPAbet strings, e.g. ``["HH", "EY1", "M", "AY1"]``.

    Returns:
        List of IPA integer IDs.
    """
    ids = []
    for p in phonemes:
        base = p.rstrip("012").upper()
        ipa = ARPABET_TO_IPA.get(base)
        ids.append(IPA_VOCAB.get(ipa, 0) if ipa else 0)
    return ids


# Backward-compat alias
phonemes_to_ids = arpabet_to_ids


# ---------------------------------------------------------------------------
# PhonMatchTextEncoder
# ---------------------------------------------------------------------------

class PhonMatchTextEncoder(nn.Module):
    """Learnable phoneme-sequence encoder.

    Maps a sequence of ARPAbet token IDs ``[B, P]`` to a mean-pooled embedding
    ``[B, 1, emb_dim]`` suitable for use as the ``text_featurizer`` ONNX in
    :class:`~ww_trainer.feats.OnnxTextExtractor`.

    Architecture (following PhonMatchNet TextEncoder):
    - ``nn.Embedding(VOCAB_SIZE, emb_dim, padding_idx=0)``
    - ``nn.Linear(emb_dim, emb_dim)`` + ReLU
    - Mean-pool over non-padding positions → ``[B, 1, emb_dim]``

    Args:
        emb_dim: Embedding dimension ``D``.  Must match ``text_emb_dim`` used
            in :func:`~ww_trainer.factory.create_model`.
        dropout: Dropout probability applied after the linear projection.
    """

    def __init__(self, emb_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(VOCAB_SIZE, emb_dim, padding_idx=0)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode token IDs into a single pooled embedding vector.

        Args:
            token_ids: ``[B, P]`` int64 tensor of phoneme IDs.  Padding positions
                (ID == 0) are excluded from the mean pool.

        Returns:
            ``[B, 1, emb_dim]`` float32 tensor.
        """
        embs = self.embedding(token_ids)           # [B, P, D]
        embs = F.relu(self.proj(embs))             # [B, P, D]
        embs = self.drop(embs)

        # Mean-pool over non-padding positions only
        mask = (token_ids != 0).float().unsqueeze(-1)  # [B, P, 1]
        summed = (embs * mask).sum(dim=1)              # [B, D]
        count = mask.sum(dim=1).clamp(min=1.0)         # [B, 1]
        pooled = summed / count                         # [B, D]

        return pooled.unsqueeze(1)                      # [B, 1, D]

    def export_to_onnx(
        self,
        out: str,
        seq_len: int = 16,
        quantize: bool = False,
        metadata: Optional[dict] = None,
    ) -> None:
        """Export this encoder to a standalone ONNX file.

        The exported model accepts ``[1, seq_len]`` int64 token IDs and returns
        ``[1, 1, emb_dim]`` float32 embeddings — matching the contract expected
        by :class:`~ww_trainer.feats.OnnxTextExtractor`.

        Args:
            out: Output file path (should end in ``.onnx``).
            seq_len: Fixed sequence length used for the dummy export input.
                At runtime the ONNX accepts any sequence length via dynamic axes.
            quantize: Apply INT8 dynamic quantization after export.
            metadata: Optional ``dict`` of string key-value pairs embedded into
                the ONNX model metadata.
        """
        import onnx
        from ww_trainer.utils import embed_onnx_metadata

        self.eval()
        dummy = torch.zeros(1, seq_len, dtype=torch.long)
        torch.onnx.export(
            self,
            (dummy,),
            out,
            input_names=["token_ids"],
            output_names=["text_emb"],
            dynamic_axes={
                "token_ids": {0: "batch_size", 1: "seq_len"},
                "text_emb": {0: "batch_size"},
            },
            opset_version=17,
        )
        if metadata:
            embed_onnx_metadata(out, metadata)
        if quantize:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(out, out, weight_type=QuantType.QInt8)
        logger.info("Exported PhonMatchTextEncoder to %s (emb_dim=%d)", out, self.emb_dim)


# ---------------------------------------------------------------------------
# Cross-attention head — faithful PhonMatchNet classifier
# ---------------------------------------------------------------------------

class PhonMatchHead(nn.Module):
    """Full PhonMatchNet classifier head with phoneme-audio cross-attention.

    Phoneme IDs are passed **per batch** at forward time — not stored as a
    fixed buffer.  This enables true multi-keyword training: each batch may
    contain samples from different keywords, each carrying its own IPA phoneme
    sequence.  At inference, pass the target keyword's IPA IDs to classify.

    Architecture:
    - ``audio_proj``: Linear(F, D) — projects audio features to attention dim
    - ``phoneme_emb``: Embedding(VOCAB_SIZE, D) — learnable phoneme table
    - ``phoneme_proj``: Linear(D, D) — projects phoneme embeddings
    - ``cross_attn``: MultiheadAttention(D, n_heads) — phonemes query audio
    - ``gru``: GRU(D, D, gru_layers) — temporal discriminator over attended feats
    - ``fc``: Linear(D, 1) — final logit

    Args:
        input_size: Audio feature dimension ``F`` (from the extractor).
        hidden_dim: Attention and GRU dimension ``D``.
        n_heads: Number of attention heads.
        gru_layers: Number of GRU layers in the discriminator.
        dropout: Dropout probability.
        device: Device placement.
        sample_rate: Audio sample rate (stored for export metadata).
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 128,
        n_heads: int = 1,
        gru_layers: int = 2,
        dropout: float = 0.1,
        device: str = "auto",
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.input_size = input_size
        self.hidden_dim = hidden_dim

        # Layers
        self.audio_proj = nn.Linear(input_size, hidden_dim)
        self.phoneme_emb = nn.Embedding(VOCAB_SIZE, hidden_dim, padding_idx=0)
        self.phoneme_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, gru_layers,
            batch_first=True, dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.to(self.device)

    def _attend(self, feats: torch.Tensor, phoneme_ids: torch.Tensor) -> torch.Tensor:
        """Cross-attend phoneme queries against audio frame keys/values.

        Args:
            feats: Audio features ``[B, T, F]``.
            phoneme_ids: ``[B, P]`` or ``[P]`` int64 keyword phoneme IDs.
                1-D input is broadcast across the batch.

        Returns:
            Attended representation ``[B, hidden_dim]``.
        """
        B = feats.shape[0]

        # Broadcast [P] → [B, P] when a single sequence is shared
        if phoneme_ids.dim() == 1:
            phoneme_ids = phoneme_ids.unsqueeze(0).expand(B, -1)  # [B, P]

        # Project audio: [B, T, F] → [B, T, D]
        audio = F.relu(self.audio_proj(feats))

        # Build phoneme queries: [B, P] → [B, P, D]
        ph_emb = F.relu(self.phoneme_proj(
            self.phoneme_emb(phoneme_ids)  # [B, P, D]
        ))                                 # [B, P, D]

        attended, _ = self.cross_attn(ph_emb, audio, audio)  # [B, P, D]

        # Mean-pool over phoneme dimension (mask padding zeros)
        pad_mask = (phoneme_ids == 0)  # [B, P] — True where padding
        attended = attended.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        n_valid = (~pad_mask).sum(dim=1, keepdim=True).clamp(min=1).float()  # [B, 1]
        pooled = attended.sum(dim=1) / n_valid                               # [B, D]
        return self.drop(pooled)

    def forward(self, feats: torch.Tensor,
                phoneme_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Classify audio via phoneme-guided attention.

        Args:
            feats: ``[B, T, F]`` audio features from the extractor.
            phoneme_ids: ``[B, P]`` or ``[P]`` int64 IPA token IDs for the
                keyword(s) in this batch.

        Returns:
            ``[B]`` raw logits (pre-sigmoid).

        Raises:
            ValueError: If ``phoneme_ids`` is ``None``.
        """
        if phoneme_ids is None:
            raise ValueError(
                "PhonMatchHead.forward() requires phoneme_ids — "
                "pass keyword IPA token IDs as a [B, P] or [P] int64 tensor."
            )
        h = self._attend(feats, phoneme_ids).unsqueeze(1)  # [B, 1, D]
        out, _ = self.gru(h)                                # [B, 1, D]
        return self.fc(out.squeeze(1)).squeeze(-1)          # [B]

    def embed(self, feats: torch.Tensor,
              phoneme_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return phoneme-aligned embeddings for metric losses.

        Args:
            feats: ``[B, T, F]`` audio features.
            phoneme_ids: ``[B, P]`` or ``[P]`` int64 keyword phoneme IDs.

        Returns:
            ``[B, hidden_dim]`` embeddings.

        Raises:
            ValueError: If ``phoneme_ids`` is ``None``.
        """
        if phoneme_ids is None:
            raise ValueError(
                "PhonMatchHead.embed() requires phoneme_ids."
            )
        return self._attend(feats, phoneme_ids)

    def export_to_onnx(
        self,
        out: str,
        seq_len: int = 16,
        quantize: bool = False,
        dynamo: bool = False,
        metadata: Optional[dict] = None,
    ) -> None:
        """Export to ONNX with two dynamic inputs: audio features and phoneme IDs.

        The exported graph accepts:
        - ``input_features``: ``[B, T, F]`` float32 audio features
        - ``phoneme_ids``:    ``[B, P]`` int64 IPA token IDs

        Both ``B`` and ``T``/``P`` are dynamic axes.

        Args:
            out: Output ``.onnx`` file path.
            seq_len: Phoneme sequence length for the export dummy input.
            quantize: Apply INT8 dynamic quantization.
            dynamo: Use ``torch.export``-based exporter (experimental).
            metadata: Optional string metadata dict to embed in the ONNX file.
        """
        from ww_trainer.utils import embed_onnx_metadata

        self.eval()
        dummy_feats = torch.zeros(1, 100, self.input_size, device=self.device)
        dummy_ids = torch.ones(1, seq_len, dtype=torch.long, device=self.device)
        torch.onnx.export(
            self,
            (dummy_feats, dummy_ids),
            out,
            input_names=["input_features", "phoneme_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_features": {0: "batch_size", 1: "T_features"},
                "phoneme_ids":    {0: "batch_size", 1: "seq_len"},
                "logits":         {0: "batch_size"},
            },
            opset_version=17,
            dynamo=False,
        )
        if metadata:
            embed_onnx_metadata(out, metadata)
        if quantize:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(out, out, weight_type=QuantType.QInt8)
        logger.info("Exported PhonMatchHead to %s", out)
