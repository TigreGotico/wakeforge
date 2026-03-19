"""Smoke tests: layer freezing sets requires_grad=False, unfreeze restores."""
import pytest

from ww_trainer.trainer import WakeWordTrainer


@pytest.fixture()
def trainer() -> WakeWordTrainer:
    """Create a minimal trainer with no freezing."""
    return WakeWordTrainer(
        arch="ffn",
        featurizer="",
        featurizer_type="mfcc",
        device="cpu",
        losses_cfg=[{"name": "bce", "weight": 1.0}],
        n_mfcc=13,
        hidden_dim=16,
    )


class TestLayerFreezing:
    """Verify freeze/unfreeze parameter toggling."""

    def test_freeze_extractor(self, trainer: WakeWordTrainer) -> None:
        trainer.freeze_extractor = True
        trainer._freeze()
        if hasattr(trainer.model, "feature_extractor"):
            for p in trainer.model.feature_extractor.parameters():
                assert not p.requires_grad

    def test_freeze_layers(self, trainer: WakeWordTrainer) -> None:
        trainer.freeze_layers = 2
        trainer._freeze()
        if hasattr(trainer.model, "classifier"):
            frozen = [p for p in trainer.model.classifier.parameters() if not p.requires_grad]
            assert len(frozen) >= 1  # at least some params frozen

    def test_unfreeze_restores(self, trainer: WakeWordTrainer) -> None:
        trainer.freeze_extractor = True
        trainer.freeze_layers = 2
        trainer._freeze()
        trainer._unfreeze()
        for p in trainer.model.parameters():
            assert p.requires_grad

    def test_init_with_freeze(self) -> None:
        """Trainer constructed with freeze_extractor=True starts frozen."""
        t = WakeWordTrainer(
            arch="ffn",
            featurizer="",
            featurizer_type="mfcc",
            device="cpu",
            losses_cfg=[{"name": "bce", "weight": 1.0}],
            n_mfcc=13,
            hidden_dim=16,
            freeze_extractor=True,
        )
        if hasattr(t.model, "feature_extractor"):
            for p in t.model.feature_extractor.parameters():
                assert not p.requires_grad
