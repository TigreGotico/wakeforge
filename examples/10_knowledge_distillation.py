#!/usr/bin/env python3
"""Knowledge distillation: compress a large teacher into a small student.

Uses CnnLstmExtractor as the student model, learning to mimic a larger
teacher (here MfccExtractor for demo; in practice use HuBERT/Wav2Vec2
via OnnxFeatureExtractor).

The student learns two objectives:
  - Feature distillation (MSE): match teacher's feature representations
  - Task loss (BCE): classify wake words correctly

After training, export the student to ONNX for edge deployment.
"""
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ww_trainer.distill import CnnLstmExtractor, KnowledgeDistillationTrainer
from ww_trainer.feats import MfccExtractor
from ww_trainer.model import FfnClassifierHead


def _make_synthetic_data(tmpdir: str, n_samples: int = 20) -> list:
    """Generate synthetic audio data for demo."""
    data = []
    for i in range(n_samples):
        t = np.linspace(0, 0.5, 8000)
        freq = 440 + np.random.randint(0, 500)
        wav = np.sin(2 * np.pi * freq * t).astype(np.float32)
        wav += np.random.randn(len(wav)).astype(np.float32) * 0.1
        p = Path(tmpdir) / f"sample_{i}.wav"
        sf.write(str(p), wav, 16000)
        label = "1" if i < n_samples // 2 else "0"
        data.append((str(p), label))
    return data


def main() -> None:
    tmpdir = tempfile.mkdtemp()

    # --- Synthetic data ---
    train_data = _make_synthetic_data(tmpdir, n_samples=20)
    val_data = _make_synthetic_data(tmpdir, n_samples=8)

    # --- Teacher: MFCC-40 (in practice, use OnnxFeatureExtractor with HuBERT) ---
    teacher = MfccExtractor(sr=16000, n_mfcc=40)
    print(f"Teacher feature_dim: {teacher.feature_dim}")

    # --- Student: compact CNN+LSTM ---
    student = CnnLstmExtractor(
        sr=16000,
        output_dim=teacher.feature_dim,  # match teacher's dim
        conv_channels=32,
        lstm_hidden=32,
        lstm_layers=1,
        device="cpu",
    )
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Student: {student_params:,} parameters")

    # --- Classifier head ---
    head = FfnClassifierHead(
        input_size=student.feature_dim,
        hidden_dim=16,
        device="cpu",
    )

    # --- Distillation ---
    trainer = KnowledgeDistillationTrainer(
        teacher=teacher,
        student=student,
        classifier_head=head,
        alpha=0.7,        # 70% distillation, 30% task loss
        temperature=1.0,
        device="cpu",
    )

    out_dir = str(Path(tmpdir) / "distilled")
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=3,
        batch_size=4,
        lr=1e-3,
        output_dir=out_dir,
        patience=10,
    )

    print(f"\nTraining loss: {[f'{l:.4f}' for l in history['train_loss']]}")
    print(f"Val loss:      {[f'{l:.4f}' for l in history['val_loss']]}")

    # --- Export student ---
    onnx_path = str(Path(out_dir) / "student.onnx")
    student.eval()
    student.export_to_onnx(onnx_path)
    print(f"\nStudent exported to {onnx_path}")
    print(f"Size: {Path(onnx_path).stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
