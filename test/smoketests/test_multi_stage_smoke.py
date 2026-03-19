"""Smoke tests for multi-stage training and checkpoint averaging."""
from collections import OrderedDict

import torch

from ww_trainer.multi_stage import average_checkpoints, parse_stage_spec


class TestAverageCheckpoints:
    """Tests for checkpoint weight averaging."""

    def test_single_checkpoint(self, tmp_path) -> None:
        state = OrderedDict({"w": torch.tensor([1.0, 2.0, 3.0])})
        path = tmp_path / "ckpt.pt"
        torch.save(state, path)
        avg = average_checkpoints([path])
        assert torch.allclose(avg["w"], state["w"])

    def test_two_checkpoints_averaged(self, tmp_path) -> None:
        s1 = OrderedDict({"w": torch.tensor([2.0, 4.0])})
        s2 = OrderedDict({"w": torch.tensor([6.0, 8.0])})
        p1 = tmp_path / "ckpt1.pt"
        p2 = tmp_path / "ckpt2.pt"
        torch.save(s1, p1)
        torch.save(s2, p2)
        avg = average_checkpoints([p1, p2])
        assert torch.allclose(avg["w"], torch.tensor([4.0, 6.0]))

    def test_top_k(self, tmp_path) -> None:
        paths = []
        for i in range(5):
            state = OrderedDict({"w": torch.tensor([float(i)])})
            p = tmp_path / f"ckpt_{i}.pt"
            torch.save(state, p)
            paths.append(p)
        # top_k=2 should average first two: (0+1)/2 = 0.5
        avg = average_checkpoints(paths, top_k=2)
        assert torch.allclose(avg["w"], torch.tensor([0.5]))

    def test_empty_raises(self) -> None:
        try:
            average_checkpoints([])
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_wrapped_checkpoint(self, tmp_path) -> None:
        state = {"model": OrderedDict({"w": torch.tensor([1.0])}), "epoch": 5}
        p = tmp_path / "ckpt.pt"
        torch.save(state, p)
        avg = average_checkpoints([p])
        assert torch.allclose(avg["w"], torch.tensor([1.0]))


class TestParseStageSpec:
    """Tests for CLI stage spec parsing."""

    def test_single_stage(self) -> None:
        stages = parse_stage_spec("30:1e-4")
        assert len(stages) == 1
        assert stages[0]["epochs"] == 30
        assert stages[0]["lr"] == 1e-4

    def test_three_stages(self) -> None:
        stages = parse_stage_spec("30:1e-4,5:1e-5,5:1e-6")
        assert len(stages) == 3
        assert stages[2]["lr"] == 1e-6

    def test_invalid_spec(self) -> None:
        try:
            parse_stage_spec("invalid")
            assert False, "Expected ValueError"
        except ValueError:
            pass
