"""The aligner head (approach B) and the constrained Viterbi used to align with it."""
import numpy as np
import torch
from torch import nn


class AlignHead(nn.Module):
    """Bidirectional frame classifier over frozen WakePhoneHuBERT outputs.
    In: layers (B, T, 9, 256), features (B, T, 521). Out: log posteriors (B, T, n_cls)."""

    def __init__(self, n_cls=41, d=128, h=96, p=0.1):
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(9))
        self.norm = nn.LayerNorm(256 + 521)
        self.inp = nn.Linear(256 + 521, d)
        self.conv = nn.Conv1d(d, d, 5, padding=2)
        self.gru = nn.GRU(d, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(2 * h, n_cls)

    def forward(self, layers, features):
        w = torch.softmax(self.mix, 0)
        x = (layers * w[:, None]).sum(2)
        ipa = torch.log(features[..., 129:521].clamp_min(1e-6)) / 8.0
        f = torch.cat([features[..., :129], ipa], -1)
        x = nn.functional.gelu(self.inp(self.norm(torch.cat([x, f], -1))))
        x = x + nn.functional.gelu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        x, _ = self.gru(self.drop(x))
        return self.out(self.drop(x)).log_softmax(-1)


def viterbi_align(logp, words, sil=0, hop=0.02):
    """Monotonic alignment of a known phone sequence to per-frame log posteriors.
    logp: (T, C) array. words: list of lists of class ids, one list per word.
    Silence is optional before, between and after words; every phone takes at least one frame.
    Returns a list per word of (class_id, start_s, end_s) phone segments."""
    T = logp.shape[0]
    cls, wid, first = [sil], [-1], [False]
    for k, w in enumerate(words):
        for j, c in enumerate(w):
            cls.append(c); wid.append(k); first.append(j == 0)
        cls.append(sil); wid.append(-1); first.append(False)
    cls, first = np.array(cls), np.array(first)
    S = len(cls)
    can_skip = first & (np.arange(S) >= 2)
    NEG = -1e30
    em = logp[:, cls]
    d = np.full(S, NEG); d[0] = em[0, 0]; d[1] = em[0, 1]
    bp = np.zeros((T, S), np.int8)
    for t in range(1, T):
        cand = np.full((3, S), NEG)
        cand[0] = d
        cand[1, 1:] = d[:-1]
        cand[2, 2:] = np.where(can_skip[2:], d[:-2], NEG)
        bp[t] = cand.argmax(0)
        d = cand[bp[t], np.arange(S)] + em[t]
    s = S - 1 if d[S - 1] >= d[S - 2] else S - 2
    path = np.empty(T, np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    out = [[] for _ in words]
    for s in range(S):
        if wid[s] < 0:
            continue
        fr = np.nonzero(path == s)[0]
        out[wid[s]].append((int(cls[s]), fr[0] * hop, (fr[-1] + 1) * hop))
    return out
