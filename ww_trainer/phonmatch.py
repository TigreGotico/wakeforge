"""PhonMatchNet text encoder — concrete implementation of the text featurizer.

Implements the text encoder from PhonMatchNet (INTERSPEECH 2023, ncsoft/PhonMatchNet).
Converts a phoneme sequence (as ARPAbet token IDs) to a fixed-size embedding that can
be exported as ONNX and used as the ``text_featurizer`` for :class:`OnnxTextExtractor`.

Usage (train + export)::

    from ww_trainer.phonmatch import PhonMatchTextEncoder, phonemes_to_ids, PHONEME_VOCAB

    # Phoneme IDs produced externally (e.g. via g2p_en or any other G2P tool)
    ids = phonemes_to_ids(["HH", "EY", "M", "AY", "K", "R", "AH", "F", "T"])

    encoder = PhonMatchTextEncoder(emb_dim=128)
    encoder.export_to_onnx("phoneme_encoder.onnx")

    # Then use with OnnxTextExtractor:
    from ww_trainer.feats import OnnxTextExtractor
    text_ext = OnnxTextExtractor("phoneme_encoder.onnx", emb_dim=128)
    text_ext.precompute(ids)

Phoneme vocabulary is the 70-token ARPAbet set used by PhonMatchNet.  Unknown phonemes
map to index 0 (treated as padding/unknown — not used in mean-pool).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ARPAbet vocabulary — 70 phonemes as used by PhonMatchNet
# Index 0 is reserved as the unknown/padding token.
# ---------------------------------------------------------------------------

PHONEME_VOCAB: Dict[str, int] = {
    "AA": 1, "AA0": 2, "AA1": 3, "AA2": 4,
    "AE": 5, "AE0": 6, "AE1": 7, "AE2": 8,
    "AH": 9, "AH0": 10, "AH1": 11, "AH2": 12,
    "AO": 13, "AO0": 14, "AO1": 15, "AO2": 16,
    "AW": 17, "AW0": 18, "AW1": 19, "AW2": 20,
    "AY": 21, "AY0": 22, "AY1": 23, "AY2": 24,
    "B": 25, "CH": 26, "D": 27, "DH": 28,
    "EH": 29, "EH0": 30, "EH1": 31, "EH2": 32,
    "ER": 33, "ER0": 34, "ER1": 35, "ER2": 36,
    "EY": 37, "EY0": 38, "EY1": 39, "EY2": 40,
    "F": 41, "G": 42, "HH": 43, "IH": 44,
    "IH0": 45, "IH1": 46, "IH2": 47, "IY": 48,
    "IY0": 49, "IY1": 50, "IY2": 51, "JH": 52,
    "K": 53, "L": 54, "M": 55, "N": 56,
    "NG": 57, "OW": 58, "OW0": 59, "OW1": 60,
    "OW2": 61, "OY": 62, "OY0": 63, "OY1": 64,
    "OY2": 65, "P": 66, "R": 67, "S": 68,
    "SH": 69, "T": 70, "TH": 71, "UH": 72,
    "UH0": 73, "UH1": 74, "UH2": 75, "UW": 76,
    "UW0": 77, "UW1": 78, "UW2": 79, "V": 80,
    "W": 81, "Y": 82, "Z": 83, "ZH": 84,
}

VOCAB_SIZE: int = len(PHONEME_VOCAB) + 1  # +1 for index-0 unknown/padding


def phonemes_to_ids(phonemes: List[str]) -> List[int]:
    """Map a sequence of ARPAbet phoneme strings to integer token IDs.

    Unknown phonemes silently map to 0 (padding index).

    Args:
        phonemes: List of ARPAbet phoneme strings, e.g. ``["HH", "EY1", "M"]``.

    Returns:
        List of integer IDs, same length as ``phonemes``.
    """
    return [PHONEME_VOCAB.get(p.upper(), 0) for p in phonemes]


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

    Unlike the simple text-featurizer path (which appends a pooled text embedding
    to audio features), this head faithfully replicates PhonMatchNet's
    cross-modal attention: phoneme embeddings act as *queries* over audio frame
    *keys/values*, producing phoneme-aligned attended representations that are
    then passed through a GRU discriminator.

    This head is registered in :data:`~ww_trainer.factory.HEAD_REGISTRY` as
    ``"phonmatch"`` and is instantiated by :func:`~ww_trainer.factory.create_model`
    when ``arch="phonmatch"``.  The wake word phoneme IDs are passed at init
    time via ``keyword_token_ids``; from that point the head is audio-only (the
    phoneme buffer is baked in and treated as a fixed learned query).

    Architecture:
    - ``audio_proj``: Linear(F, D) — projects audio features to attention dim
    - ``phoneme_emb``: Embedding(VOCAB_SIZE, D) — learnable phoneme table
    - ``phoneme_proj``: Linear(D, D) — projects phoneme embeddings
    - ``cross_attn``: MultiheadAttention(D, n_heads) — phonemes query audio
    - ``gru``: GRU(D, D, gru_layers) — temporal discriminator over attended feats
    - ``fc``: Linear(D, 1) — final logit

    Args:
        input_size: Audio feature dimension ``F`` (from the extractor).
        keyword_token_ids: Pre-tokenised phoneme IDs for the target keyword.
            Produced by :func:`phonemes_to_ids`.  Stored as a buffer.
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
        keyword_token_ids: List[int],
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

        # Phoneme query buffer — fixed after init, baked into ONNX export
        ids = torch.tensor(keyword_token_ids, dtype=torch.long)
        self.register_buffer("phoneme_ids", ids)  # [P]

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

    def _attend(self, feats: torch.Tensor) -> torch.Tensor:
        """Cross-attend phoneme queries against audio frame keys/values.

        Args:
            feats: Audio features ``[B, T, F]``.

        Returns:
            Attended representation ``[B, hidden_dim]``.
        """
        B = feats.shape[0]

        # Project audio: [B, T, F] → [B, T, D]
        audio = F.relu(self.audio_proj(feats))

        # Build phoneme queries: [P] → [B, P, D]
        ph_emb = F.relu(self.phoneme_proj(
            self.phoneme_emb(self.phoneme_ids)  # [P, D]
        )).unsqueeze(0).expand(B, -1, -1)       # [B, P, D]

        # Key-padding mask: silence padding tokens in audio (all-zero frames)
        # No explicit padding mask here — rely on the model to learn it.
        attended, _ = self.cross_attn(ph_emb, audio, audio)  # [B, P, D]

        # Mean-pool over phoneme dimension → [B, D]
        return self.drop(attended.mean(dim=1))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Classify audio via phoneme-guided attention.

        Args:
            feats: ``[B, T, F]`` audio features from the extractor.

        Returns:
            ``[B]`` raw logits (pre-sigmoid).
        """
        h = self._attend(feats).unsqueeze(1)      # [B, 1, D]
        out, _ = self.gru(h)                       # [B, 1, D]
        return self.fc(out.squeeze(1)).squeeze(-1)  # [B]

    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        """Return the attended phoneme-aligned embedding for metric losses.

        Args:
            feats: ``[B, T, F]`` audio features.

        Returns:
            ``[B, hidden_dim]`` embeddings.
        """
        return self._attend(feats)

    def export_to_onnx(
        self,
        out: str,
        quantize: bool = False,
        dynamo: bool = False,
        metadata: Optional[dict] = None,
    ) -> None:
        """Export the head (with phoneme buffer baked in) to ONNX.

        The exported graph accepts only audio features ``[B, T, F]`` — the
        phoneme IDs are constants embedded in the graph at export time.

        Args:
            out: Output ``.onnx`` file path.
            quantize: Apply INT8 dynamic quantization.
            dynamo: Use ``torch.export``-based exporter (experimental).
            metadata: Optional string metadata dict to embed in the ONNX file.
        """
        import onnx
        from ww_trainer.utils import embed_onnx_metadata

        self.eval()
        dummy = torch.zeros(1, 100, self.input_size, device=self.device)
        torch.onnx.export(
            self,
            (dummy,),
            out,
            input_names=["input_features"],
            output_names=["logits"],
            dynamic_axes={
                "input_features": {0: "batch_size", 1: "T_features"},
                "logits": {0: "batch_size"},
            },
            opset_version=17,
        )
        if metadata:
            embed_onnx_metadata(out, metadata)
        if quantize:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(out, out, weight_type=QuantType.QInt8)
        logger.info("Exported PhonMatchHead to %s", out)
