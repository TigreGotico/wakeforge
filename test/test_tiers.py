"""Tests for ww_trainer/tiers.py — hardware tier presets."""
import pytest

from ww_trainer.tiers import get_tier, list_tiers, HARDWARE_TIERS, TierConfig


# ---- get_tier ----

def test_get_tier_micro():
    cfg = get_tier("micro")
    assert isinstance(cfg, TierConfig)
    assert cfg.name == "micro"
    assert cfg.extractor_type == "mfcc"
    assert cfg.head_arch == "ffn"
    assert cfg.hidden_dim == 128
    assert cfg.n_mfcc == 40


def test_get_tier_small():
    cfg = get_tier("small")
    assert cfg.name == "small"
    assert cfg.extractor_type == "mfcc"
    assert cfg.head_arch == "gru"


def test_get_tier_medium():
    cfg = get_tier("medium")
    assert cfg.name == "medium"
    assert cfg.extractor_type == "onnx"
    assert cfg.head_arch == "ffn"


def test_get_tier_large():
    cfg = get_tier("large")
    assert cfg.name == "large"
    assert cfg.extractor_type == "hubert"
    assert cfg.head_arch == "gru"
    assert cfg.bidirectional is True
    assert cfg.gru_n_layers == 2


def test_get_tier_unknown_raises():
    with pytest.raises(ValueError, match="Unknown tier"):
        get_tier("nonexistent_tier")


def test_get_tier_unknown_message_contains_available():
    try:
        get_tier("bogus")
    except ValueError as exc:
        msg = str(exc)
        for name in ["micro", "small", "medium", "large"]:
            assert name in msg


# ---- list_tiers ----

def test_list_tiers_returns_string():
    result = list_tiers()
    assert isinstance(result, str)


def test_list_tiers_contains_all_names():
    result = list_tiers()
    for name in ["micro", "small", "medium", "large"]:
        assert name in result


def test_list_tiers_contains_extractor_types():
    result = list_tiers()
    assert "mfcc" in result
    assert "onnx" in result
    assert "hubert" in result


# ---- TierConfig fields ----

def test_tier_config_is_dataclass():
    cfg = TierConfig(
        name="test",
        extractor_type="mfcc",
        head_arch="ffn",
        hidden_dim=64,
    )
    assert cfg.name == "test"
    assert cfg.n_mfcc == 40  # default
    assert cfg.bidirectional is False  # default
    assert cfg.gru_n_layers == 1  # default


def test_all_tiers_in_dict():
    assert {"micro", "small", "medium", "large"}.issubset(set(HARDWARE_TIERS.keys()))


# ---- Instantiate models for micro and small (no external deps) ----

def test_micro_tier_model_instantiation():
    """micro tier (mfcc + ffn) can be instantiated without external deps."""
    from ww_trainer.feats import MfccExtractor
    from ww_trainer.model import FfnClassifierHead, BaseWakeModel

    cfg = get_tier("micro")
    extractor = MfccExtractor(sr=16000, n_mfcc=cfg.n_mfcc)
    head = FfnClassifierHead(input_size=cfg.n_mfcc, hidden_dim=cfg.hidden_dim, device="cpu")
    model = BaseWakeModel(feature_extractor=extractor, classifier=head, device="cpu")
    assert model is not None


def test_small_tier_model_instantiation():
    """small tier (mfcc + gru) can be instantiated without external deps."""
    from ww_trainer.feats import MfccExtractor
    from ww_trainer.model import GruClassifierHead, BaseWakeModel

    cfg = get_tier("small")
    extractor = MfccExtractor(sr=16000, n_mfcc=cfg.n_mfcc)
    head = GruClassifierHead(
        input_size=cfg.n_mfcc,
        hidden_dim=cfg.hidden_dim,
        device="cpu",
    )
    model = BaseWakeModel(feature_extractor=extractor, classifier=head, device="cpu")
    assert model is not None
