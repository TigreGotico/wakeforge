"""Integration tests for WakeWordTrainer.create_model with all architectures."""
import pytest
import torch

from ww_trainer.trainer import WakeWordTrainer


_ALL_HEADS = [
    "ffn", "gru", "cnn", "bcresnet", "tcresnet", "dscnn",
    "matchboxnet", "res15", "kwt", "conformer", "crnn",
]

_ALL_EXTRACTORS = [
    "mfcc", "filterbank", "sincnet", "gammatone", "leaf",
    "plp", "pncc", "cqt", "delta_mfcc", "delta_filterbank",
]


class TestCreateModelHeads:
    """Every classifier head can be instantiated via create_model."""

    @pytest.mark.parametrize("arch", _ALL_HEADS)
    def test_head_forward(self, arch):
        model = WakeWordTrainer.create_model(
            arch, featurizer="", feature_dim=None,
            featurizer_type="mfcc", device="cpu",
            n_mfcc=40, hidden_dim=32, channels=32,
            tau=1, variant=8, size="S", B=3, R=1, C=32,
            d_model=32, n_heads=4, n_layers=2, dim_ff=64,
            conv_channels=16, gru_hidden=32, patch_len=5,
            conv_kernel=7,
        )
        audio = torch.randn(1, 16000)
        with torch.no_grad():
            out = model(audio)
        assert out.shape == (1,)


class TestCreateModelExtractors:
    """Every extractor type can be instantiated via create_model."""

    @pytest.mark.parametrize("feat_type", _ALL_EXTRACTORS)
    def test_extractor_forward(self, feat_type):
        model = WakeWordTrainer.create_model(
            "ffn", featurizer="", feature_dim=None,
            featurizer_type=feat_type, device="cpu",
            n_mfcc=13, n_mels=40, n_filters=40,
            hidden_dim=32,
        )
        audio = torch.randn(1, 16000)
        with torch.no_grad():
            out = model(audio)
        assert out.shape == (1,)


class TestCreateModelEnrichment:
    """Enrichment wrappers (VAD, Pitch, SNR) work via kwargs."""

    def test_vad_enrichment(self):
        model = WakeWordTrainer.create_model(
            "ffn", featurizer="", feature_dim=None,
            featurizer_type="mfcc", device="cpu",
            n_mfcc=13, hidden_dim=32, use_vad=True,
        )
        # MFCC-13 + VAD(4) = 17
        assert model.feature_extractor.feature_dim == 17
        with torch.no_grad():
            out = model(torch.randn(1, 16000))
        assert out.shape == (1,)

    def test_pitch_enrichment(self):
        model = WakeWordTrainer.create_model(
            "ffn", featurizer="", feature_dim=None,
            featurizer_type="mfcc", device="cpu",
            n_mfcc=13, hidden_dim=32, use_pitch=True,
        )
        assert model.feature_extractor.feature_dim == 16  # 13 + 3

    def test_snr_enrichment(self):
        model = WakeWordTrainer.create_model(
            "ffn", featurizer="", feature_dim=None,
            featurizer_type="mfcc", device="cpu",
            n_mfcc=13, hidden_dim=32, use_snr=True,
        )
        assert model.feature_extractor.feature_dim == 15  # 13 + 2

    def test_stacked_enrichment(self):
        model = WakeWordTrainer.create_model(
            "gru", featurizer="", feature_dim=None,
            featurizer_type="mfcc", device="cpu",
            n_mfcc=13, hidden_dim=32,
            use_vad=True, use_pitch=True, use_snr=True,
        )
        assert model.feature_extractor.feature_dim == 22  # 13+4+3+2


class TestCreateModelErrors:
    def test_unknown_head(self):
        with pytest.raises(ValueError, match="Unknown classifier"):
            WakeWordTrainer.create_model(
                "nonexistent", featurizer="", featurizer_type="mfcc", device="cpu"
            )

    def test_unknown_extractor(self):
        with pytest.raises(ValueError, match="Unknown featurizer_type"):
            WakeWordTrainer.create_model(
                "ffn", featurizer="", featurizer_type="nonexistent", device="cpu"
            )
