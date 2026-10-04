"""Frozen WakeHuBERT-tiny trunk exposing its taps, and the head that reads them.

WakePhoneHuBERT is that trunk with two TapHeads on its taps: a VAD head (TapHead(1, 32)) and an IPA head
(TapHead(392, 192, side=128)).

Taps per 50 Hz frame: the stem output and the eight block outputs (9 x 256), the 128-dim output, and the
normalised log-mel with its two 100 Hz frames stacked (128).
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download

TRUNK_REPO = "TigreGotico/wakehubert-tiny"
TRUNK_REVISION = "3caa605725a932440442692f644b7217f54a29c3"


def load_trunk():
    d = snapshot_download(TRUNK_REPO, revision=TRUNK_REVISION, allow_patterns=["student.py", "model.safetensors", "config.json"])
    sys.path.insert(0, d)
    from student import load_wakehubert_tiny
    m = load_wakehubert_tiny(f"{d}/model.safetensors")
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


class Trunk(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, wav):
        m = self.m
        mel = m.norm(m.mel(wav))
        x = F.relu(m.stem_bn(m.stem(mel)))
        taps = [x]
        for b in m.blocks:
            x = b(x)
            taps.append(x)
        out = m.out(x)
        t = out.shape[-1]
        mel = F.pad(mel, (0, max(0, 2 * t - mel.shape[-1])))[..., :2 * t]
        mel2 = mel.reshape(mel.shape[0], mel.shape[1], t, 2).permute(0, 1, 3, 2).reshape(mel.shape[0], -1, t)
        return torch.stack(taps, 1), out, mel2


class SideBranch(nn.Module):
    """Trainable causal mel-TCN beside the frozen trunk: stacked log-mel (128) in, `ch` channels out at 50 Hz,
    dilations 1, 2, 4, 8, 16 with kernel 3 (about 1.2 s of look-back)."""

    def __init__(self, ch=128, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.inp = nn.Conv1d(128, ch, 1)
        self.dw = nn.ModuleList(nn.Conv1d(ch, ch, 3, dilation=d, groups=ch) for d in dilations)
        self.pw = nn.ModuleList(nn.Conv1d(ch, ch, 1) for _ in dilations)
        self.norm = nn.ModuleList(nn.BatchNorm1d(ch) for _ in dilations)
        self.dil = dilations

    def forward(self, mel2):
        x = self.inp(mel2)
        for dw, pw, bn, d in zip(self.dw, self.pw, self.norm, self.dil):
            x = F.relu(bn(pw(F.relu(dw(F.pad(x, (2 * d, 0)))))) + x)
        return x


class TapHead(nn.Module):
    """Softmax-weighted mix of the 9 taps, concatenated with the 128 output, stacked log-mel and (optionally) a
    trainable side branch over the log-mel, then a causal adapter. Output [batch, frames, n_out] (logits)."""

    def __init__(self, n_out, hidden=192, n_taps=9, tap_dim=256, side=0):
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(n_taps))
        self.side = SideBranch(side) if side else None
        self.c1 = nn.Conv1d(tap_dim + 128 + 128 + side, hidden, 5)
        self.c2 = nn.Conv1d(hidden, hidden, 3)
        self.o = nn.Conv1d(hidden, n_out, 1)

    def forward(self, taps, out, mel2):
        w = torch.softmax(self.mix, 0)
        parts = [(taps * w[None, :, None, None]).sum(1), out, mel2]
        if self.side is not None:
            parts.append(self.side(mel2))
        x = torch.cat(parts, 1)
        x = F.relu(self.c1(F.pad(x, (4, 0))))
        x = F.relu(self.c2(F.pad(x, (2, 0))))
        return self.o(x).transpose(1, 2)
