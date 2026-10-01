"""Smoke tests: end-to-end extractor → head → forward pipelines via factory."""
import pytest
import torch

from ww_trainer.factory import create_model

DUMMY_WAV = torch.randn(2, 16000)

# All (extractor_type, arch) combinations that need no external deps.
# Each tuple: (featurizer_type, arch, extra_kwargs)
PIPELINE_COMBOS = [
    # MFCC with every head
    ("mfcc", "ffn", {"n_mfcc": 13, "hidden_dim": 16}),
    ("mfcc", "gru", {"n_mfcc": 13, "hidden_dim": 16}),
    ("mfcc", "cnn", {"n_mfcc": 13, "conv_dim": 16, "linear_dim": 16}),
    ("mfcc", "bcresnet", {"n_mfcc": 13, "tau": 1}),
    ("mfcc", "tcresnet", {"n_mfcc": 13, "variant": 8, "channels": 16}),
    ("mfcc", "dscnn", {"n_mfcc": 13, "size": "S"}),
    ("mfcc", "matchboxnet", {"n_mfcc": 13, "B": 1, "R": 1, "C": 16}),
    ("mfcc", "res15", {"n_mfcc": 13, "channels": 16}),
    ("mfcc", "kwt", {"n_mfcc": 13, "d_model": 16, "n_heads": 2, "n_layers": 1}),
    ("mfcc", "conformer", {"n_mfcc": 13, "d_model": 16, "n_heads": 2, "n_layers": 1}),
    ("mfcc", "crnn", {"n_mfcc": 13, "conv_channels": 8, "gru_hidden": 16}),
    # Other extractors with FFN head
    ("filterbank", "ffn", {"n_mels": 20, "hidden_dim": 16}),
    ("sincnet", "ffn", {"n_filters": 16, "hidden_dim": 16}),
    ("gammatone", "ffn", {"n_filters": 16, "hidden_dim": 16}),
    ("delta_mfcc", "ffn", {"n_mfcc": 13, "hidden_dim": 16}),
    ("delta_filterbank", "ffn", {"n_mels": 20, "hidden_dim": 16}),
    ("leaf", "ffn", {"n_filters": 16, "hidden_dim": 16}),
    ("plp", "ffn", {"n_plp": 13, "hidden_dim": 16}),
    ("pncc", "ffn", {"n_pncc": 13, "hidden_dim": 16}),
    ("cqt", "ffn", {"hidden_dim": 16}),
    # Other extractors with GRU head
    ("filterbank", "gru", {"n_mels": 20, "hidden_dim": 16}),
    ("gammatone", "gru", {"n_filters": 16, "hidden_dim": 16}),
    # Enriched extractors
    ("mfcc", "ffn", {"n_mfcc": 13, "hidden_dim": 16, "use_vad": True}),
    ("mfcc", "ffn", {"n_mfcc": 13, "hidden_dim": 16, "use_pitch": True}),
    ("mfcc", "ffn", {"n_mfcc": 13, "hidden_dim": 16, "use_snr": True}),
    ("mfcc", "ffn", {"n_mfcc": 13, "hidden_dim": 16, "use_vad": True, "use_pitch": True}),
]


@pytest.mark.parametrize(
    "feat_type,arch,kwargs",
    PIPELINE_COMBOS,
    ids=[f"{f}-{a}" + ("-enriched" if any(k.startswith("use_") for k in kw) else "")
         for f, a, kw in PIPELINE_COMBOS],
)
def test_pipeline_forward(feat_type: str, arch: str, kwargs: dict) -> None:
    """Create model via factory and verify forward pass produces output."""
    model = create_model(
        arch_name=arch,
        featurizer="",
        featurizer_type=feat_type,
        device="cpu",
        **kwargs,
    )
    with torch.no_grad():
        logits = model(DUMMY_WAV)
    assert logits.shape == (2,), f"Expected (2,), got {logits.shape}"
    assert not torch.isnan(logits).any()


@pytest.mark.parametrize(
    "feat_type,arch,kwargs",
    [
        ("mfcc", "ffn", {"n_mfcc": 13, "hidden_dim": 16}),
        ("mfcc", "gru", {"n_mfcc": 13, "hidden_dim": 16}),
        ("filterbank", "cnn", {"n_mels": 20, "conv_dim": 16, "linear_dim": 16}),
    ],
    ids=["mfcc-ffn", "mfcc-gru", "filterbank-cnn"],
)
def test_pipeline_embed(feat_type: str, arch: str, kwargs: dict) -> None:
    """Embedding extraction works for key combos."""
    model = create_model(
        arch_name=arch,
        featurizer="",
        featurizer_type=feat_type,
        device="cpu",
        **kwargs,
    )
    with torch.no_grad():
        emb = model.embed(DUMMY_WAV)
    assert emb.ndim == 2
    assert emb.shape[0] == 2
    assert not torch.isnan(emb).any()
