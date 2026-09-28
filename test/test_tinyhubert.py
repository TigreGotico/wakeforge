"""TinyHuBERT recipe: frame rate, causality, frame pairing, target statistics, losses, and the
exported extractor loading through OnnxFeatureExtractor."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

_P = Path(__file__).resolve().parents[1] / "scripts" / "research" / "tinyhubert.py"
_S = importlib.util.spec_from_file_location("tinyhubert", _P)
th = importlib.util.module_from_spec(_S)
_S.loader.exec_module(th)


def _student(**kw):
    torch.manual_seed(0)
    return th.WakeHuBERTStudent(**{"cnn_dim": 8, "gru_hidden": 16, "n_targets": 2, "target_dim": 12, **kw}).eval()


@pytest.mark.parametrize("n", [16000, 32000, 21920, 320, 639, 641])
def test_the_student_emits_one_frame_per_320_samples(n):
    assert _student()(torch.randn(2, n)).shape == (2, n // 320, 16)


def test_hubert_and_student_frame_rates_agree():
    # HuBERT base: seven convolutions with total stride 320 and a 400-sample receptive field
    hubert_frames = (32000 - 400) // 320 + 1
    student_frames = _student()(torch.randn(1, 32000)).shape[1]
    assert abs(student_frames - hubert_frames) <= 1


def test_a_frame_does_not_depend_on_any_later_sample():
    s = _student()
    x = torch.randn(1, 16000)
    base = s(x)
    for t in (0, 7, 30):
        y = x.clone()
        y[:, 320 * (t + 1):] = torch.randn(1, 16000 - 320 * (t + 1)) * 5
        out = s(y)
        assert torch.allclose(out[:, :t + 1], base[:, :t + 1], atol=1e-6), f"frame <= {t} saw the future"
        assert not torch.allclose(out[:, t + 1:], base[:, t + 1:], atol=1e-6)


@pytest.mark.parametrize("st, tt, lag, want", [(100, 99, 0, 99), (100, 99, 4, 95), (50, 49, 49, 0), (10, 99, 2, 7)])
def test_frame_pairing(st, tt, lag, want):
    assert th.pair_frames(st, tt, lag) == want


def test_pairing_gives_the_student_the_teacher_frames_whole_window():
    # teacher frame t reads samples [320t, 320t + 400); its paired student frame t + 1 + lag has
    # heard every sample before 320 (t + 2 + lag)
    for t in range(50):
        assert 320 * (t + 2) >= 320 * t + 400


def test_target_statistics_standardise_each_layer_and_dimension():
    g = torch.Generator().manual_seed(0)
    scale = torch.tensor([1.0, 50.0, 0.01])
    batches = [torch.randn(2, 4, 30, 3, generator=g) * scale + torch.tensor([0.0, 3.0, -7.0]) for _ in range(20)]
    norm = th.TargetNorm(2, 3)
    norm.fit(iter(batches))
    z = norm(torch.cat(batches, dim=1)).flatten(0, 2)
    assert torch.allclose(z.mean(0), torch.zeros(3), atol=1e-4)
    assert torch.allclose(z.std(0), torch.ones(3), atol=1e-2)
    assert bool(norm.fitted)


def test_distil_loss_is_lowest_at_the_target_and_reports_per_layer():
    t = torch.randn(3, 2, 10, 8)
    same, l1, cos = th.distil_loss(t.clone(), t)
    assert torch.allclose(l1, torch.zeros(3)) and torch.allclose(cos, torch.ones(3), atol=1e-6)
    assert math.isclose(same.item(), -torch.nn.functional.logsigmoid(torch.tensor(1.0)).item(), rel_tol=1e-5)
    worse, _, _ = th.distil_loss(t + 0.5 * torch.randn_like(t), t)
    assert worse > same


def test_cross_clip_nce_ignores_frames_of_the_same_clip():
    # every frame of a clip is identical: counting same-clip frames as negatives would make the
    # positive indistinguishable from them and hold the loss at log(T) or above
    B, T, D = 4, 6, 16
    torch.manual_seed(0)
    clips = torch.nn.functional.normalize(torch.randn(B, 1, D), dim=-1).expand(B, T, D).contiguous()
    loss = th.cross_clip_nce(clips, clips, temperature=0.05)
    assert loss.item() < 0.1
    assert th.cross_clip_nce(clips[:1], clips[:1]).item() == 0.0


def test_mix_at_snr_hits_the_requested_snr():
    g = torch.Generator().manual_seed(0)
    clean, noise = torch.randn(16000, generator=g), torch.randn(16000, generator=g) * 3
    mixed = th.mix_at_snr(clean, noise, 10.0)
    snr = 10 * torch.log10(clean.pow(2).mean() / (mixed - clean).pow(2).mean())
    assert abs(snr.item() - 10.0) < 1e-3


def test_the_export_loads_as_an_onnx_feature_extractor(tmp_path):
    from ww_trainer.feats import OnnxFeatureExtractor

    s = _student()
    th.export(s, tmp_path, {"layers": [4, 8]}, "cpu")
    ext = OnnxFeatureExtractor(str(tmp_path / "tinyhubert.onnx"), device="cpu")
    assert ext.feature_dim == 16
    wavs = [torch.randn(16000), torch.randn(24000)]
    feats = ext(wavs)
    assert feats.shape == (2, 75, 16)
    with torch.no_grad():
        ref = s(wavs[1][None])[0]
    assert torch.allclose(feats[1], ref, atol=1e-4)
    meta = json.loads((tmp_path / "tinyhubert.json").read_text())
    assert meta["params_exported"] == sum(p.numel() for n, p in s.named_parameters() if not n.startswith("proj."))


def test_an_export_that_disagrees_with_torch_is_refused(tmp_path, monkeypatch):
    import onnxruntime

    real = onnxruntime.InferenceSession

    class Off(real):
        def run(self, *a, **k):
            return [o + 0.01 for o in super().run(*a, **k)]

    monkeypatch.setattr(onnxruntime, "InferenceSession", Off)
    with pytest.raises(SystemExit, match="disagrees with torch"):
        th.export(_student(), tmp_path, {}, "cpu")
    assert not (tmp_path / "tinyhubert.json").exists()


class _FakeTeacher:
    """Stands in for HuBERT: deterministic features of the clean audio, HuBERT's frame count."""

    def __init__(self, name, layers, device, dtype=torch.float32):
        self.layers, self.dim, self.device = layers, 12, device
        g = torch.Generator().manual_seed(1)
        self.w = torch.randn(len(layers), 400, 12, generator=g)

    def __call__(self, wav):
        frames = wav.unfold(1, 400, 320)  # [B, T, 400], T = (N - 400) // 320 + 1
        return torch.einsum("btk,lkd->lbtd", frames, self.w)


def test_a_short_run_trains_evaluates_resumes_and_exports(tmp_path, monkeypatch):
    for name, n in (("train", 6), ("val", 3), ("noise", 2)):
        d = tmp_path / name
        d.mkdir()
        for i in range(n):
            sf.write(str(d / f"{i}.wav"), np.random.RandomState(i).randn(24000).astype(np.float32) * 0.1, 16000)
    monkeypatch.setattr(th, "Teacher", _FakeTeacher)
    out = tmp_path / "run"
    common = ["--audio-dir", str(tmp_path / "train"), "--val-dir", str(tmp_path / "val"), "--noise-dir",
              str(tmp_path / "noise"), "--out-dir", str(out), "--cnn-dim", "4", "--gru-hidden", "8", "--layers", "1,2",
              "--batch-size", "2", "--crop-seconds", "1.0", "--norm-batches", "2", "--eval-every", "2", "--workers", "0",
              "--device", "cpu", "--nce-weight", "0.1"]
    th.main(common + ["--steps", "2"])
    th.main(common + ["--steps", "4", "--resume", str(out / "last.pt")])
    vals = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines() if "val_loss" in l]
    assert [v["step"] for v in vals] == [2, 4] and all(len(v["val_cos"]) == 2 for v in vals)
    assert torch.load(out / "last.pt")["step"] == 4
    assert (out / "tinyhubert.onnx").exists()
    meta = json.loads((out / "tinyhubert.json").read_text())
    assert meta["frame_rate_hz"] == 50.0 and meta["latency_ms"] == 100.0 and meta["feature_dim"] == 8
