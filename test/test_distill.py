"""Tests for ww_trainer/distill.py — knowledge distillation."""
import numpy as np
import pytest
import soundfile as sf
import torch

from ww_trainer.distill import CnnLstmExtractor, KnowledgeDistillationTrainer
from ww_trainer.feats import MfccExtractor
from ww_trainer.model import FfnClassifierHead

SAMPLE_RATE = 16000


@pytest.fixture
def tiny_train_data(tmp_path):
    """4 synthetic audio files: 2 wake, 2 not-wake."""
    data = []
    for i in range(4):
        t = np.linspace(0, 0.5, int(SAMPLE_RATE * 0.5))
        wav = np.sin(2 * np.pi * (440 + i * 100) * t).astype(np.float32)
        p = tmp_path / f"audio_{i}.wav"
        sf.write(str(p), wav, SAMPLE_RATE)
        label = "1" if i < 2 else "0"
        data.append((str(p), label))
    return data


class TestCnnLstmExtractor:
    def test_forward_shape(self):
        model = CnnLstmExtractor(
            sr=16000, output_dim=64, conv_channels=32, lstm_hidden=32, lstm_layers=1
        )
        wav = torch.randn(2, 16000)
        out = model(wav)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 64

    def test_feature_dim(self):
        model = CnnLstmExtractor(output_dim=128)
        assert model.feature_dim == 128

    def test_list_input(self):
        model = CnnLstmExtractor(
            sr=16000, output_dim=32, conv_channels=16, lstm_hidden=16, lstm_layers=1
        )
        wavs = [torch.randn(16000), torch.randn(8000)]
        out = model(wavs)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 32

    def test_export_to_onnx(self, tmp_path):
        model = CnnLstmExtractor(
            sr=16000, output_dim=32, conv_channels=16, lstm_hidden=16, lstm_layers=1
        )
        model.eval()
        out_path = str(tmp_path / "student.onnx")
        model.export_to_onnx(out_path)
        assert (tmp_path / "student.onnx").exists()

    def test_gradients_flow(self):
        model = CnnLstmExtractor(
            sr=16000, output_dim=32, conv_channels=16, lstm_hidden=16, lstm_layers=1
        )
        wav = torch.randn(1, 16000)
        out = model(wav)
        loss = out.mean()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad


class TestKnowledgeDistillationTrainer:
    def test_init(self):
        teacher = MfccExtractor(sr=16000, n_mfcc=40)
        student = CnnLstmExtractor(
            sr=16000, output_dim=40, conv_channels=16, lstm_hidden=16, lstm_layers=1
        )
        head = FfnClassifierHead(input_size=40, hidden_dim=16, device="cpu")
        trainer = KnowledgeDistillationTrainer(teacher, student, head, device="cpu")
        assert trainer.alpha == 0.7
        assert trainer.student_proj is None  # same dim, no projection needed

    def test_init_with_projection(self):
        """When dims differ, a projection layer is created."""
        teacher = MfccExtractor(sr=16000, n_mfcc=40)
        student = CnnLstmExtractor(
            sr=16000, output_dim=32, conv_channels=16, lstm_hidden=16, lstm_layers=1
        )
        head = FfnClassifierHead(input_size=32, hidden_dim=16, device="cpu")
        trainer = KnowledgeDistillationTrainer(teacher, student, head, device="cpu")
        assert trainer.student_proj is not None

    def test_distill_loss(self):
        teacher = MfccExtractor(sr=16000, n_mfcc=40)
        student = CnnLstmExtractor(
            sr=16000, output_dim=40, conv_channels=16, lstm_hidden=16, lstm_layers=1
        )
        head = FfnClassifierHead(input_size=40, hidden_dim=16, device="cpu")
        trainer = KnowledgeDistillationTrainer(teacher, student, head, device="cpu")

        s_feats = torch.randn(2, 50, 40)
        t_feats = torch.randn(2, 50, 40)
        loss = trainer._distill_loss(s_feats, t_feats)
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0

    def test_distill_loss_length_mismatch(self):
        """Distillation loss handles different T dimensions."""
        teacher = MfccExtractor(sr=16000, n_mfcc=40)
        student = CnnLstmExtractor(
            sr=16000, output_dim=40, conv_channels=16, lstm_hidden=16, lstm_layers=1
        )
        head = FfnClassifierHead(input_size=40, hidden_dim=16, device="cpu")
        trainer = KnowledgeDistillationTrainer(teacher, student, head, device="cpu")

        s_feats = torch.randn(2, 30, 40)  # shorter student
        t_feats = torch.randn(2, 50, 40)  # longer teacher
        loss = trainer._distill_loss(s_feats, t_feats)
        assert isinstance(loss.item(), float)

    def test_train_smoke(self, tiny_train_data, tmp_path):
        """Full train loop runs without error on tiny synthetic data."""
        teacher = MfccExtractor(sr=16000, n_mfcc=40)
        student = CnnLstmExtractor(
            sr=16000, output_dim=40, conv_channels=16, lstm_hidden=16, lstm_layers=1
        )
        head = FfnClassifierHead(input_size=40, hidden_dim=16, device="cpu")
        trainer = KnowledgeDistillationTrainer(
            teacher, student, head, alpha=0.5, device="cpu"
        )
        history = trainer.train(
            train_data=tiny_train_data,
            val_data=tiny_train_data,
            epochs=2,
            batch_size=2,
            lr=1e-3,
            output_dir=str(tmp_path / "distilled"),
            patience=10,
        )
        assert "train_loss" in history
        assert len(history["train_loss"]) == 2
        assert (tmp_path / "distilled" / "best_student.pt").exists()
