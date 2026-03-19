"""Knowledge distillation module for ww-trainer.

Trains a compact CNN+LSTM student extractor to mimic a large teacher extractor
(e.g. HuBERT) while also learning to classify wake words.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ww_trainer.feats import BaseExtractor, WavInput
from ww_trainer.model import ClassifierHead

logger = logging.getLogger(__name__)


def _num_groups(conv_channels: int) -> int:
    """Compute a valid num_groups for GroupNorm: largest divisor of conv_channels <= 32."""
    if conv_channels < 2:
        return 1
    for g in range(min(conv_channels, 32), 0, -1):
        if conv_channels % g == 0:
            return g
    return 1  # fallback: 1 group = LayerNorm-like


class CnnLstmExtractor(BaseExtractor):
    """Compact CNN+LSTM feature extractor for knowledge distillation.

    Designed as a student model to mimic large transformer extractors
    (HuBERT, Wav2Vec2) at a fraction of the parameter count.

    Architecture:
        - 4-layer strided Conv1D front-end (raw audio → frame features)
          Total stride 10 × 2 × 4 × 2 = 160 → 10 ms frames at 16 kHz.
        - 2-layer bidirectional LSTM
        - Linear projection to output_dim

    Output: [B, T_frames, output_dim]
    Typical params: 500K–2M depending on dims.

    Args:
        sr: Sample rate (default 16000).
        output_dim: Feature dimension to output (default 256).
                    Set to teacher's feature_dim for distillation.
        conv_channels: Channels in CNN front-end layers (default 256).
        lstm_hidden: LSTM hidden size per direction (default 256).
        lstm_layers: Number of LSTM layers (default 2).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        sr: int = 16000,
        output_dim: int = 256,
        conv_channels: int = 256,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        device: str = "auto",
    ) -> None:
        super().__init__(sample_rate=sr, device=device)
        self._feature_dim = output_dim

        num_groups = _num_groups(conv_channels)

        # CNN front-end: raw audio → frame features
        # Total stride: 10 × 2 × 4 × 2 = 160 → 10 ms frames at 16 kHz
        self.conv_frontend = nn.Sequential(
            # Layer 1: large kernel, stride 10 → [B, C, T/10]
            nn.Conv1d(1, conv_channels, kernel_size=10, stride=10, padding=0),
            nn.GELU(),
            nn.GroupNorm(num_groups, conv_channels),
            # Layer 2: stride 2 → [B, C, T/20]
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.GroupNorm(num_groups, conv_channels),
            # Layer 3: stride 4 → [B, C, T/80]
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, stride=4, padding=1),
            nn.GELU(),
            nn.GroupNorm(num_groups, conv_channels),
            # Layer 4: stride 2 → [B, C, T/160]
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        # LSTM context encoder
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Project to output_dim (*2 for bidirectional)
        self.proj = nn.Linear(lstm_hidden * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.to(self.device)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, wavs: WavInput) -> torch.Tensor:
        """Extract CNN+LSTM features from raw audio.

        Args:
            wavs: [B, T] tensor or list of 1-D tensors.

        Returns:
            Tensor of shape [B, T_frames, output_dim].
        """
        if isinstance(wavs, list):
            wavs = [w.to(torch.float32) for w in wavs]
            max_len = max(w.shape[-1] for w in wavs)
            batch = torch.stack(
                [F.pad(w, (0, max_len - w.shape[-1])) for w in wavs]
            )
        else:
            batch = wavs.unsqueeze(0) if wavs.ndim == 1 else wavs
        _dev = next(self.parameters()).device
        batch = batch.to(device=_dev, dtype=torch.float32)

        # CNN front-end: [B, T] → [B, C, T_frames]
        x = self.conv_frontend(batch.unsqueeze(1))  # [B, 1, T] → [B, C, T_frames]
        x = x.transpose(1, 2)  # [B, T_frames, C]

        # LSTM: [B, T_frames, C] → [B, T_frames, 2*lstm_hidden]
        x, _ = self.lstm(x)
        x = self.dropout(x)

        # Project: [B, T_frames, output_dim]
        return self.proj(x)


class KnowledgeDistillationTrainer:
    """Train a compact student extractor to mimic a large teacher extractor.

    Uses a combination of:
    - Feature distillation loss: MSE between student and teacher feature sequences
      (after optional linear projection if dims differ).
    - Task loss: BCE on wake word classification using student features.

    The teacher is kept frozen. The student learns both to reproduce teacher
    representations AND to classify wake words.

    After training, the student is exported to ONNX and can be used as a
    drop-in replacement for the teacher in OnnxFeatureExtractor.

    Args:
        teacher: A BaseExtractor (typically OnnxFeatureExtractor with HuBERT).
                 Kept frozen throughout training.
        student: A CnnLstmExtractor (or any BaseExtractor) to train.
        classifier_head: A ClassifierHead for wake word task loss.
        alpha: Weight for distillation loss (default 0.7).
               Task loss weight = 1 - alpha.
        temperature: Softening temperature for distillation (default 1.0).
        device: Training device ('cpu', 'cuda', or 'auto').

    Usage::

        teacher = OnnxFeatureExtractor("hubert.onnx")
        student = CnnLstmExtractor(output_dim=teacher.feature_dim)
        head = FfnClassifierHead(input_size=student.feature_dim)

        trainer = KnowledgeDistillationTrainer(teacher, student, head)
        trainer.train(train_data, val_data, epochs=50, output_dir="distilled/")

        # Export student for deployment
        student.export_to_onnx("student_extractor.onnx")
    """

    def __init__(
        self,
        teacher: BaseExtractor,
        student: BaseExtractor,
        classifier_head: ClassifierHead,
        alpha: float = 0.7,
        temperature: float = 1.0,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.teacher = teacher  # frozen
        self.student = student.to(self.device)
        self.classifier = classifier_head.to(self.device)
        self.alpha = alpha
        self.temperature = temperature

        # If teacher and student have different feature dims, add a projection
        # so we can compute MSE loss in the same space.
        t_dim = teacher.feature_dim
        s_dim = student.feature_dim
        if t_dim != s_dim:
            self.student_proj: nn.Linear | None = nn.Linear(s_dim, t_dim).to(self.device)
        else:
            self.student_proj = None

    def _distill_loss(
        self,
        student_feats: torch.Tensor,
        teacher_feats: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between student and teacher feature sequences.

        Handles length mismatch by truncating or interpolating to the shorter sequence.

        Args:
            student_feats: [B, T_s, D_s]
            teacher_feats: [B, T_t, D_t]

        Returns:
            Scalar MSE loss.
        """
        if self.student_proj is not None:
            student_feats = self.student_proj(student_feats)

        T_s = student_feats.shape[1]
        T_t = teacher_feats.shape[1]

        if T_s != T_t:
            T_target = min(T_s, T_t)
            if T_s > T_target:
                student_feats = F.interpolate(
                    student_feats.transpose(1, 2),
                    size=T_target,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            if T_t > T_target:
                teacher_feats = teacher_feats[:, :T_target, :]

        return F.mse_loss(
            student_feats / self.temperature,
            teacher_feats.detach() / self.temperature,
        )

    def train(
        self,
        train_data: list,
        val_data: list,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        sample_rate: int = 16000,
        output_dir: str = "distilled",
        save_best: str = "loss",
        patience: int = 10,
        log_every: int = 5,
    ) -> dict:
        """Run the distillation training loop.

        Args:
            train_data: List of (audio_path, label_str) tuples.
            val_data: List of (audio_path, label_str) tuples.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            lr: Learning rate.
            sample_rate: Audio sample rate.
            output_dir: Directory for checkpoints and exported ONNX.
            save_best: Metric to track for best model ('loss' or 'f1').
            patience: Early stopping patience.
            log_every: Log every N epochs.

        Returns:
            dict with keys 'train_loss', 'val_loss', 'val_f1' — each a list
            of per-epoch values.
        """
        from torch.utils.data import DataLoader

        from ww_trainer.dataset import AudioDataset, collate_fn

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build datasets
        train_ds = AudioDataset(train_data, sample_rate=sample_rate)
        val_ds = AudioDataset(val_data, sample_rate=sample_rate)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        # Optimise student + classifier + optional projection
        params = list(self.student.parameters()) + list(self.classifier.parameters())
        if self.student_proj is not None:
            params += list(self.student_proj.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        # Freeze teacher (only applies to nn.Module teachers)
        if hasattr(self.teacher, "parameters"):
            for p in self.teacher.parameters():
                p.requires_grad_(False)

        history: dict = {"train_loss": [], "val_loss": [], "val_f1": []}
        best_val = float("inf")
        patience_counter = 0

        bce = nn.BCEWithLogitsLoss()

        for epoch in range(1, epochs + 1):
            # --- Training ---
            self.student.train()
            self.classifier.train()
            if self.student_proj is not None:
                self.student_proj.train()

            train_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                # collate_fn returns (wavs, labels, paths) — unpack accordingly
                wavs, labels = batch[0], batch[1]
                wavs = wavs.to(self.device)
                labels = labels.to(self.device).float()

                optimizer.zero_grad()

                # Teacher features (no grad)
                with torch.no_grad():
                    teacher_feats = self.teacher(wavs)
                    if not isinstance(teacher_feats, torch.Tensor):
                        teacher_feats = torch.tensor(
                            teacher_feats, device=self.device, dtype=torch.float32
                        )
                    teacher_feats = teacher_feats.to(self.device)

                # Student features
                student_feats = self.student(wavs)

                # Distillation loss
                loss_distill = self._distill_loss(student_feats, teacher_feats)

                # Task loss
                logits = self.classifier(student_feats)
                loss_task = bce(logits, labels)

                # Combined loss
                loss = self.alpha * loss_distill + (1.0 - self.alpha) * loss_task

                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            scheduler.step()
            train_loss /= max(n_batches, 1)

            # --- Validation ---
            self.student.eval()
            self.classifier.eval()
            if self.student_proj is not None:
                self.student_proj.eval()

            val_loss = 0.0
            all_preds: list = []
            all_labels: list = []

            with torch.no_grad():
                for batch in val_loader:
                    wavs, labels = batch[0], batch[1]
                    wavs = wavs.to(self.device)
                    labels = labels.to(self.device).float()

                    teacher_feats = self.teacher(wavs)
                    if not isinstance(teacher_feats, torch.Tensor):
                        teacher_feats = torch.tensor(
                            teacher_feats, device=self.device, dtype=torch.float32
                        )
                    teacher_feats = teacher_feats.to(self.device)

                    student_feats = self.student(wavs)
                    loss_d = self._distill_loss(student_feats, teacher_feats)
                    logits = self.classifier(student_feats)
                    loss_t = bce(logits, labels)
                    val_loss += (
                        self.alpha * loss_d + (1.0 - self.alpha) * loss_t
                    ).item()

                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())

            val_loss /= max(len(val_loader), 1)

            try:
                from sklearn.metrics import f1_score

                val_f1 = float(f1_score(all_labels, all_preds, zero_division=0))
            except Exception:
                val_f1 = 0.0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)

            if epoch % log_every == 0 or epoch == 1:
                logger.info(
                    "Epoch %d/%d — train_loss=%.4f  val_loss=%.4f  val_f1=%.4f",
                    epoch,
                    epochs,
                    train_loss,
                    val_loss,
                    val_f1,
                )

            # Save best checkpoint
            metric = val_loss if save_best == "loss" else (1.0 - val_f1)
            if metric < best_val:
                best_val = metric
                patience_counter = 0
                torch.save(
                    self.student.state_dict(), out_dir / "best_student.pt"
                )
                torch.save(
                    self.classifier.state_dict(), out_dir / "best_classifier.pt"
                )
                logger.info(
                    "  New best (val_%s=%.4f), checkpoint saved",
                    save_best,
                    val_loss if save_best == "loss" else val_f1,
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # Load best and export to ONNX
        self.student.load_state_dict(
            torch.load(
                out_dir / "best_student.pt", map_location=self.device, weights_only=True
            )
        )
        self.student.eval()

        student_onnx = str(out_dir / "student_extractor.onnx")
        try:
            self.student.export_to_onnx(student_onnx)
            logger.info("Student exported to ONNX: %s", student_onnx)
        except Exception as exc:
            logger.warning("ONNX export failed: %s", exc)

        return history


def distill_hubert_to_cnn_lstm(
    teacher_onnx_path: str,
    train_data: list,
    val_data: list,
    output_dir: str = "distilled_student",
    student_dim: int = 256,
    conv_channels: int = 256,
    lstm_hidden: int = 256,
    lstm_layers: int = 2,
    alpha: float = 0.7,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "auto",
) -> KnowledgeDistillationTrainer:
    """Convenience function: distill a HuBERT ONNX teacher into a CnnLstmExtractor student.

    Args:
        teacher_onnx_path: Path to the teacher ONNX extractor (e.g. HuBERT).
        train_data: List of (audio_path, label_str) tuples.
        val_data: List of (audio_path, label_str) tuples.
        output_dir: Where to save student checkpoints and ONNX export.
        student_dim: Output feature dimension of the student.
        conv_channels: CNN front-end channels.
        lstm_hidden: LSTM hidden size per direction.
        lstm_layers: Number of LSTM layers.
        alpha: Distillation loss weight (0 = task only, 1 = distill only).
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        device: Training device.

    Returns:
        Trained KnowledgeDistillationTrainer (student is exported to output_dir/).

    Example::

        trainer = distill_hubert_to_cnn_lstm(
            teacher_onnx_path="hubert.onnx",
            train_data=[("audio/hey_jarvis_001.wav", "1"), ...],
            val_data=[...],
            output_dir="models/hey_jarvis_student/",
        )
        # Student ONNX at: models/hey_jarvis_student/student_extractor.onnx
    """
    from ww_trainer.feats import OnnxFeatureExtractor
    from ww_trainer.model import FfnClassifierHead

    teacher = OnnxFeatureExtractor(teacher_onnx_path, device=device)
    student = CnnLstmExtractor(
        sr=16000,
        output_dim=student_dim,
        conv_channels=conv_channels,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
    )
    head = FfnClassifierHead(input_size=student_dim, hidden_dim=128, device=device)

    trainer = KnowledgeDistillationTrainer(
        teacher=teacher,
        student=student,
        classifier_head=head,
        alpha=alpha,
        device=device,
    )
    trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        output_dir=output_dir,
    )
    return trainer
