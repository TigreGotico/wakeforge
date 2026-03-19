"""Smoke tests: WakeWordTrainer end-to-end training with dummy data."""
import os
import pytest
import numpy as np
import soundfile as sf
import torch


@pytest.fixture(scope="module")
def dummy_dataset(tmp_path_factory) -> tuple:
    """Create a minimal dataset with real wav files for training."""
    base = tmp_path_factory.mktemp("dataset")
    train_data = []
    test_data = []
    for i in range(16):
        wav = np.random.randn(16000).astype(np.float32) * 0.1
        label = "1" if i % 2 == 0 else "0"
        path = str(base / f"sample_{i}.wav")
        sf.write(path, wav, 16000)
        if i < 12:
            train_data.append((path, label))
        else:
            test_data.append((path, label))
    return train_data, test_data


class TestTrainerMfccFfn:
    """Minimal training: MFCC + FFN + BCE, 1 epoch."""

    def test_train_bce(self, dummy_dataset, tmp_path) -> None:
        from ww_trainer.trainer import WakeWordTrainer
        train_data, test_data = dummy_dataset
        trainer = WakeWordTrainer(
            arch="ffn",
            featurizer="",
            featurizer_type="mfcc",
            device="cpu",
            losses_cfg=[{"name": "bce", "weight": 1.0}],
            n_mfcc=13,
            hidden_dim=16,
        )
        f1 = trainer.train(
            train_data=train_data,
            test_data=test_data,
            epochs=1,
            batch_size=4,
            lr=1e-3,
            output_dir=str(tmp_path / "train_bce"),
            pca_every=0, tsne_every=0, umap_every=0,
        )
        assert isinstance(f1, float)


class TestTrainerMfccGru:
    """Minimal training: MFCC + GRU + focal, 1 epoch."""

    def test_train_focal(self, dummy_dataset, tmp_path) -> None:
        from ww_trainer.trainer import WakeWordTrainer
        train_data, test_data = dummy_dataset
        trainer = WakeWordTrainer(
            arch="gru",
            featurizer="",
            featurizer_type="mfcc",
            device="cpu",
            losses_cfg=[{"name": "focal", "weight": 1.0}],
            n_mfcc=13,
            hidden_dim=16,
        )
        f1 = trainer.train(
            train_data=train_data,
            test_data=test_data,
            epochs=1,
            batch_size=4,
            lr=1e-3,
            output_dir=str(tmp_path / "train_focal"),
            pca_every=0, tsne_every=0, umap_every=0,
        )
        assert isinstance(f1, float)


class TestTrainerFilterbankCnn:
    """Minimal training: filterbank + CNN + BCE, 1 epoch."""

    def test_train(self, dummy_dataset, tmp_path) -> None:
        from ww_trainer.trainer import WakeWordTrainer
        train_data, test_data = dummy_dataset
        trainer = WakeWordTrainer(
            arch="cnn",
            featurizer="",
            featurizer_type="filterbank",
            device="cpu",
            losses_cfg=[{"name": "bce", "weight": 1.0}],
            n_mels=20,
            conv_dim=16,
            linear_dim=16,
        )
        f1 = trainer.train(
            train_data=train_data,
            test_data=test_data,
            epochs=1,
            batch_size=4,
            lr=1e-3,
            output_dir=str(tmp_path / "train_cnn"),
            pca_every=0, tsne_every=0, umap_every=0,
        )
        assert isinstance(f1, float)


class TestTrainerMultiLoss:
    """Training with combined losses: BCE + contrastive."""

    def test_train(self, dummy_dataset, tmp_path) -> None:
        from ww_trainer.trainer import WakeWordTrainer
        train_data, test_data = dummy_dataset
        trainer = WakeWordTrainer(
            arch="ffn",
            featurizer="",
            featurizer_type="mfcc",
            device="cpu",
            losses_cfg=[
                {"name": "bce", "weight": 1.0},
                {"name": "contrastive", "weight": 0.3},
            ],
            n_mfcc=13,
            hidden_dim=16,
        )
        f1 = trainer.train(
            train_data=train_data,
            test_data=test_data,
            epochs=1,
            batch_size=8,
            lr=1e-3,
            output_dir=str(tmp_path / "train_multi"),
            pca_every=0, tsne_every=0, umap_every=0,
        )
        assert isinstance(f1, float)


class TestTrainerSizeAware:
    """Training with SizeAwareLoss for ESP32."""

    def test_train(self, dummy_dataset, tmp_path) -> None:
        from ww_trainer.trainer import WakeWordTrainer
        train_data, test_data = dummy_dataset
        trainer = WakeWordTrainer(
            arch="ffn",
            featurizer="",
            featurizer_type="mfcc",
            device="cpu",
            losses_cfg=[{"name": "size_aware", "weight": 1.0, "param_budget": 1024}],
            n_mfcc=13,
            hidden_dim=16,
        )
        f1 = trainer.train(
            train_data=train_data,
            test_data=test_data,
            epochs=1,
            batch_size=4,
            lr=1e-3,
            output_dir=str(tmp_path / "train_size"),
            pca_every=0, tsne_every=0, umap_every=0,
        )
        assert isinstance(f1, float)
