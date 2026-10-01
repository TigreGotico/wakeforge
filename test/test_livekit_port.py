"""Tests for components ported from livekit/livekit-wakeword.

Covers:

- :class:`ww_trainer.model.ConvAttentionHead` — Conv1d + self-attention head.
- :func:`ww_trainer.metrics.area_under_det` — AUT metric.
- :func:`ww_trainer.checkpoint.average_checkpoints` — uniform weight averaging.
- :func:`ww_trainer.checkpoint.select_best_checkpoints` — percentile gating.
"""
from __future__ import annotations

import numpy as np
import torch

from ww_trainer.checkpoint import average_checkpoints, select_best_checkpoints
from ww_trainer.metrics import area_under_det
from ww_trainer.model import ConvAttentionHead


class TestConvAttentionHead:
    def test_forward_shape(self) -> None:
        head = ConvAttentionHead(input_size=96, layer_dim=32, n_blocks=1,
                                 n_heads=4, device="cpu")
        out = head(torch.randn(3, 16, 96))
        assert out.shape == (3,)

    def test_variable_time_dim(self) -> None:
        """Head must accept any ``T`` so dynamic ONNX axis works."""
        head = ConvAttentionHead(input_size=40, layer_dim=16, device="cpu")
        assert head(torch.randn(2, 30, 40)).shape == (2,)
        assert head(torch.randn(2, 100, 40)).shape == (2,)

    def test_embed_shape(self) -> None:
        head = ConvAttentionHead(input_size=96, layer_dim=24, device="cpu")
        emb = head.embed(torch.randn(4, 16, 96))
        assert emb.shape == (4, 24)

    def test_n_heads_auto_adjusted(self) -> None:
        """If ``n_heads`` does not divide ``layer_dim``, it is auto-clipped."""
        head = ConvAttentionHead(input_size=8, layer_dim=10, n_heads=4,
                                 device="cpu")
        # 10 not divisible by 4 → falls back to 2.
        assert head.n_heads == 2
        head(torch.randn(1, 5, 8))  # still runs

    def test_export_to_onnx(self, tmp_path) -> None:
        import onnx

        head = ConvAttentionHead(input_size=96, layer_dim=16, n_blocks=1,
                                 device="cpu")
        head.eval()
        out = tmp_path / "ca.onnx"
        head.export_to_onnx(str(out))
        assert out.exists()
        onnx.checker.check_model(onnx.load(str(out)))

    def test_gradient_flow(self) -> None:
        head = ConvAttentionHead(input_size=32, layer_dim=16, device="cpu")
        x = torch.randn(2, 10, 32, requires_grad=True)
        head(x).sum().backward()
        assert x.grad is not None


class TestAreaUnderDet:
    def test_perfect_separation(self) -> None:
        y_true = np.array([0] * 50 + [1] * 50)
        y_scores = np.concatenate([np.zeros(50), np.ones(50)])
        aut = area_under_det(y_true, y_scores)
        assert aut < 0.01

    def test_random_classifier(self) -> None:
        rng = np.random.default_rng(0)
        y_true = np.array([0] * 200 + [1] * 200)
        y_scores = rng.random(400)
        aut = area_under_det(y_true, y_scores)
        # Random classifier — AUT integrates the diagonal ≈ 0.5.
        assert 0.35 < aut < 0.65


class TestAverageCheckpoints:
    def test_uniform_mean(self, tmp_path) -> None:
        paths = []
        for i in range(4):
            p = tmp_path / f"c{i}.pt"
            torch.save({"w": torch.full((3, 3), float(i))}, str(p))
            paths.append(p)
        avg = average_checkpoints(paths)
        # mean(0,1,2,3) = 1.5
        assert torch.allclose(avg["w"], torch.full((3, 3), 1.5))

    def test_writes_output(self, tmp_path) -> None:
        p1 = tmp_path / "a.pt"
        torch.save({"w": torch.ones(2)}, str(p1))
        out = tmp_path / "avg.pt"
        average_checkpoints([p1], out_path=out)
        assert out.exists()

    def test_empty_list(self) -> None:
        assert average_checkpoints([]) == {}

    def test_preserves_non_float_buffers(self, tmp_path) -> None:
        p1 = tmp_path / "a.pt"
        torch.save({"step": torch.tensor(7, dtype=torch.long),
                    "w": torch.ones(2)}, str(p1))
        avg = average_checkpoints([p1])
        assert avg["step"].dtype == torch.long
        assert int(avg["step"]) == 7


class TestSelectBestCheckpoints:
    def test_selects_top_quantile(self) -> None:
        # The 10 ckpts: only #9 has low FPPH + high recall + high accuracy.
        history = [
            {"fpph": 100.0, "recall": 0.50, "accuracy": 0.60, "path": str(i)}
            for i in range(9)
        ]
        history.append({"fpph": 1.0, "recall": 0.99, "accuracy": 0.98,
                        "path": "best"})
        sel = select_best_checkpoints(history)
        assert [s["path"] for s in sel] == ["best"]

    def test_fallback_to_highest_recall(self) -> None:
        # No checkpoint passes all three gates simultaneously.
        history = [
            {"fpph": 1.0, "recall": 0.10, "accuracy": 0.10, "path": "a"},
            {"fpph": 100.0, "recall": 0.99, "accuracy": 0.10, "path": "b"},
            {"fpph": 100.0, "recall": 0.10, "accuracy": 0.99, "path": "c"},
        ]
        sel = select_best_checkpoints(history)
        assert len(sel) == 1
        assert sel[0]["path"] == "b"

    def test_empty(self) -> None:
        assert select_best_checkpoints([]) == []


class TestFpphAdaptiveDoubling:
    """``target_fp_per_hour`` should double ``max_neg_weight`` when exceeded.

    The doubling itself lives on :class:`ww_trainer.loss.LossManager` and is
    triggered inline inside :func:`ww_trainer.loop.train_loop`. Rather than
    spinning up a full training loop, we exercise the two contracts
    separately: (1) ``train_loop`` accepts the kwarg, (2)
    ``LossManager.adjust_max_neg_weight(2.0)`` performs the doubling.
    """

    def test_train_loop_signature_accepts_kwarg(self) -> None:
        import inspect

        from ww_trainer.loop import training_loop
        params = inspect.signature(training_loop).parameters
        assert "target_fp_per_hour" in params
        assert params["target_fp_per_hour"].default is None

    def test_loss_manager_doubles(self) -> None:
        from ww_trainer.loss import LossManager

        lm = LossManager(loss_configs=[{"name": "bce", "weight": 1.0}],
                         neg_weight_schedule="linear",
                         max_neg_weight=100.0,
                         device="cpu")
        lm.adjust_max_neg_weight(2.0)
        assert lm.max_neg_weight == 200.0
        lm.adjust_max_neg_weight(2.0)
        assert lm.max_neg_weight == 400.0

    def test_cli_exposes_flag(self) -> None:
        from click.testing import CliRunner

        from ww_trainer.cli import train as train_cmd
        result = CliRunner().invoke(train_cmd, ["--help"])
        assert "--target-fp-per-hour" in result.output
