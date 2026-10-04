"""scripts/research/wakephonehubert/wph_model.py: the trunk wrapper and the tap heads build, give the documented shapes and stay causal.

The WakeHuBERT-tiny trunk is replaced by a stand-in with the same interface (log-mel at 100 Hz, a stride-2 stem, eight
residual blocks, a 128-wide output projection), so nothing is downloaded.
"""
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_P = Path(__file__).resolve().parent.parent / "scripts" / "research" / "wakephonehubert" / "wph_model.py"
_S = importlib.util.spec_from_file_location("wph_model", _P)
wm = importlib.util.module_from_spec(_S)
_S.loader.exec_module(wm)

B, T = 2, 12
SAMPLES = T * 320


class StandInTrunk(nn.Module):
    """The attributes wph_model.Trunk reads: mel, norm, stem, stem_bn, blocks, out."""

    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm1d(64)
        self.stem = nn.Conv1d(64, 256, 3, stride=2, padding=1)
        self.stem_bn = nn.BatchNorm1d(256)
        self.blocks = nn.ModuleList(nn.Conv1d(256, 256, 1) for _ in range(8))
        self.out = nn.Conv1d(256, 128, 1)

    def mel(self, wav):
        return F.avg_pool1d(wav[:, None], 160).repeat(1, 64, 1)


@pytest.fixture(scope="module")
def taps():
    torch.manual_seed(0)
    trunk = wm.Trunk(StandInTrunk().eval())
    with torch.no_grad():
        return trunk(torch.randn(B, SAMPLES))


def test_trunk_taps_have_the_documented_shapes(taps):
    stacked, out, mel2 = taps
    assert tuple(stacked.shape) == (B, 9, 256, T)
    assert tuple(out.shape) == (B, 128, T)
    assert tuple(mel2.shape) == (B, 128, T)


@pytest.mark.parametrize("n_out,hidden,side", [(1, 32, 0), (392, 192, 128)])
def test_head_output_is_batch_frames_classes(taps, n_out, hidden, side):
    head = wm.TapHead(n_out, hidden=hidden, side=side).eval()
    with torch.no_grad():
        z = head(*taps)
    assert tuple(z.shape) == (B, T, n_out)


def test_head_parameter_counts():
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(wm.TapHead(1, hidden=32)) == 85098
    assert count(wm.TapHead(392, hidden=192, side=128)) == 903953


def test_tap_mix_starts_uniform():
    head = wm.TapHead(1, hidden=32)
    assert head.mix.shape == (9,)
    assert torch.allclose(torch.softmax(head.mix, 0), torch.full((9,), 1 / 9))


@pytest.mark.parametrize("side", [0, 128])
def test_head_is_causal(side):
    torch.manual_seed(1)
    head = wm.TapHead(5, hidden=16, side=side).eval()
    a = (torch.randn(1, 9, 256, 40), torch.randn(1, 128, 40), torch.randn(1, 128, 40))
    cut = 25
    b = (a[0].clone(), a[1].clone(), a[2].clone())
    b[0][..., cut:] += torch.randn(1, 9, 256, 40 - cut)
    b[1][..., cut:] += torch.randn(1, 128, 40 - cut)
    b[2][..., cut:] += torch.randn(1, 128, 40 - cut)
    with torch.no_grad():
        za, zb = head(*a), head(*b)
    assert torch.equal(za[:, :cut], zb[:, :cut])
    assert not torch.equal(za[:, cut:], zb[:, cut:])

