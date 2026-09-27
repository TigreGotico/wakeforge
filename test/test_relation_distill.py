"""Tests for causal temporal relation distillation (Zhang et al., Interspeech 2026).

Expected values come from hand-built frame sequences whose cosine matrices
are known exactly, never from the loss itself.
"""
import pytest
import torch

from ww_trainer.distill import (
    CnnLstmExtractor,
    KnowledgeDistillationTrainer,
    temporal_relation_loss,
)
from ww_trainer.feats import MfccExtractor
from ww_trainer.model import FfnClassifierHead

# Two frames, so pooling to two steps is the identity.
# Teacher frames are orthogonal: G_t = [[1, 0], [0, 1]].
# Student frames are identical: G_s = [[1, 1], [1, 1]].
# G_s - G_t = [[0, 1], [1, 0]].
TEACHER = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
STUDENT = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])


def test_causal_value():
    # The lower triangle with diagonal holds 3 entries; only (1, 0) differs.
    loss = temporal_relation_loss(STUDENT, TEACHER, steps=2)
    assert loss.item() == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_bidirectional_adds_full_matrix_term():
    # Full matrix: two entries of squared error 1, divided by L^2 = 4.
    loss = temporal_relation_loss(STUDENT, TEACHER, steps=2, bidirectional=True)
    assert loss.item() == pytest.approx(1.0 / 3.0 + 2.0 / 4.0, abs=1e-6)


def test_same_topology_in_other_dim_and_scale_costs_nothing():
    """Relations ignore feature dimension and scale; MSE cannot."""
    torch.manual_seed(0)
    teacher = torch.randn(3, 24, 16)
    lift, _ = torch.linalg.qr(torch.randn(64, 64))
    student = 7.0 * teacher @ lift[:16, :]
    assert temporal_relation_loss(student, teacher, steps=24).item() == pytest.approx(0.0, abs=1e-5)
    assert temporal_relation_loss(
        student, teacher, steps=24, bidirectional=True
    ).item() == pytest.approx(0.0, abs=1e-5)


def test_teacher_gets_no_gradient_student_does():
    teacher = torch.randn(2, 30, 8, requires_grad=True)
    student = torch.randn(2, 50, 4, requires_grad=True)
    temporal_relation_loss(student, teacher, steps=6).backward()
    assert teacher.grad is None
    assert student.grad is not None and student.grad.abs().sum() > 0


def _trainer(feature_loss: str) -> KnowledgeDistillationTrainer:
    teacher = MfccExtractor(n_mfcc=40)
    student = CnnLstmExtractor(output_dim=16, conv_channels=8, lstm_hidden=8, lstm_layers=1)
    head = FfnClassifierHead(input_size=16, hidden_dim=8, device="cpu")
    return KnowledgeDistillationTrainer(teacher, student, head, device="cpu", feature_loss=feature_loss)


def test_trainer_relation_needs_no_projection():
    assert _trainer("mse").student_proj is not None
    assert _trainer("relation").student_proj is None
    assert _trainer("relation_bi").student_proj is None


def test_trainer_distill_loss_dispatches_to_relation():
    trainer = _trainer("relation_bi")
    student = torch.randn(2, 40, 16)
    teacher = torch.randn(2, 60, 40)
    expected = temporal_relation_loss(student, teacher, steps=24, bidirectional=True)
    assert trainer._distill_loss(student, teacher).item() == pytest.approx(expected.item(), abs=1e-6)


def test_trainer_rejects_unknown_feature_loss():
    with pytest.raises(ValueError):
        _trainer("cosine")
