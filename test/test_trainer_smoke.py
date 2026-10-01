"""Integration smoke test for WakeWordTrainer.train() with synthetic data."""
import numpy as np
import pytest
import soundfile as sf

from ww_trainer.trainer import WakeWordTrainer


@pytest.fixture
def tiny_dataset(tmp_path):
    """Create 8 synthetic WAV files: 4 wake, 4 not-wake."""
    data = []
    for i in range(8):
        t = np.linspace(0, 0.5, 8000)
        wav = np.sin(2 * np.pi * (440 + i * 100) * t).astype(np.float32)
        p = tmp_path / f"audio_{i}.wav"
        sf.write(str(p), wav, 16000)
        label = "1" if i < 4 else "0"
        data.append((str(p), label))
    return data


class TestTrainerSmoke:
    """Minimal end-to-end training to verify the pipeline doesn't crash."""

    def test_train_mfcc_ffn(self, tiny_dataset, tmp_path):
        trainer = WakeWordTrainer(
            arch="ffn", featurizer="", feature_dim=None,
            featurizer_type="mfcc", device="cpu",
            losses_cfg=[{"name": "bce", "weight": 1.0}],
            n_mfcc=13, hidden_dim=32,
        )
        result = trainer.train(
            train_data=tiny_dataset,
            test_data=tiny_dataset,
            epochs=2, batch_size=4, lr=1e-3,
            output_dir=str(tmp_path / "out"),
            save_best="f1",
            tsne_every=0, pca_every=0, umap_every=0,
        )
        assert isinstance(result, float)

    def test_train_mfcc_gru(self, tiny_dataset, tmp_path):
        trainer = WakeWordTrainer(
            arch="gru", featurizer="", feature_dim=None,
            featurizer_type="mfcc", device="cpu",
            losses_cfg=[{"name": "bce", "weight": 1.0}],
            n_mfcc=13, hidden_dim=32,
        )
        result = trainer.train(
            train_data=tiny_dataset,
            test_data=tiny_dataset,
            epochs=1, batch_size=4, lr=1e-3,
            output_dir=str(tmp_path / "out"),
            save_best="f1",
            tsne_every=0, pca_every=0, umap_every=0,
        )
        assert isinstance(result, float)

    def test_train_filterbank_bcresnet(self, tiny_dataset, tmp_path):
        trainer = WakeWordTrainer(
            arch="bcresnet", featurizer="", feature_dim=None,
            featurizer_type="filterbank", device="cpu",
            losses_cfg=[{"name": "bce", "weight": 1.0}],
            n_mels=40, tau=1,
        )
        result = trainer.train(
            train_data=tiny_dataset,
            test_data=tiny_dataset,
            epochs=1, batch_size=4, lr=1e-3,
            output_dir=str(tmp_path / "out"),
            save_best="f1",
            tsne_every=0, pca_every=0, umap_every=0,
        )
        assert isinstance(result, float)

    def test_train_focal_loss(self, tiny_dataset, tmp_path):
        trainer = WakeWordTrainer(
            arch="ffn", featurizer="", feature_dim=None,
            featurizer_type="mfcc", device="cpu",
            losses_cfg=[{"name": "focal", "weight": 1.0}],
            n_mfcc=13, hidden_dim=32,
        )
        result = trainer.train(
            train_data=tiny_dataset,
            test_data=tiny_dataset,
            epochs=1, batch_size=4, lr=1e-3,
            output_dir=str(tmp_path / "out"),
            save_best="f1",
            tsne_every=0, pca_every=0, umap_every=0,
        )
        assert isinstance(result, float)

    def test_train_with_vad_enrichment(self, tiny_dataset, tmp_path):
        trainer = WakeWordTrainer(
            arch="ffn", featurizer="", feature_dim=None,
            featurizer_type="mfcc", device="cpu",
            losses_cfg=[{"name": "bce", "weight": 1.0}],
            n_mfcc=13, hidden_dim=32, use_vad=True,
        )
        assert trainer.model.feature_extractor.feature_dim == 17
        result = trainer.train(
            train_data=tiny_dataset,
            test_data=tiny_dataset,
            epochs=1, batch_size=4, lr=1e-3,
            output_dir=str(tmp_path / "out"),
            save_best="f1",
            tsne_every=0, pca_every=0, umap_every=0,
        )
        assert isinstance(result, float)
