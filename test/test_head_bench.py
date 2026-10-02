"""scripts/research/head_bench.py: the content-addressed feature store and the learning-rate schedule."""
import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import onnxruntime
import pytest
import soundfile as sf
import torch

from conftest import export_stand_in_featurizer

_P = Path(__file__).resolve().parent.parent / "scripts" / "research" / "head_bench.py"
_S = importlib.util.spec_from_file_location("head_bench", _P)
hb = importlib.util.module_from_spec(_S)
_S.loader.exec_module(hb)

SR = 16000
POS_AUG = 2
REAL_SESSION = onnxruntime.InferenceSession


class SpySession:
    """onnxruntime session that counts the windows it featurizes."""
    rows = 0

    def __init__(self, *args, **kwargs):
        self._s = REAL_SESSION(*args, **kwargs)

    def get_inputs(self):
        return self._s.get_inputs()

    def get_providers(self):
        return self._s.get_providers()

    def run(self, out, feed):
        SpySession.rows += next(iter(feed.values())).shape[0]
        return self._s.run(out, feed)


def _wav(path, seconds, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = np.random.default_rng(seed).standard_normal(int(seconds * SR)).astype(np.float32) * 0.1
    sf.write(str(path), w, SR)
    return str(path)


def _csv(path, rows):
    path.write_text("".join(f"{f},{l}\n" for f, l in rows))
    return str(path)


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A working directory with the data/ pools head_bench reads, a tiny word and two stand-in featurizers."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hb, "CALIB_SECONDS", 20)
    monkeypatch.setattr(onnxruntime, "InferenceSession", SpySession)
    for i in range(3):
        _wav(tmp_path / f"data/noise-audioset/n{i}.wav", 2, 100 + i)
        _wav(tmp_path / f"data/LibriSpeech/dev-other/s{i}.flac", 6, 200 + i)
        _wav(tmp_path / f"data/LibriSpeech/train-clean-100/t{i}.flac", 3, 300 + i)
    _wav(tmp_path / "data/rir/r0.wav", 0.3, 400)
    pos = [(_wav(tmp_path / f"kws/p{i}.wav", 1.0, 10 + i), 1) for i in range(4)]
    neg = [(_wav(tmp_path / f"kws/n{i}.wav", 1.5, 20 + i), 0) for i in range(8)]
    val = [(_wav(tmp_path / f"kws/vp{i}.wav", 1.0, 30 + i), 1) for i in range(2)] + \
          [(_wav(tmp_path / f"kws/vn{i}.wav", 1.5, 40 + i), 0) for i in range(2)]
    feat = tmp_path / "feat.onnx"; export_stand_in_featurizer(feat, 0, 16)
    other = tmp_path / "other.onnx"; export_stand_in_featurizer(other, 1, 16)
    return argparse.Namespace(root=tmp_path, pos=pos, neg=neg, feat=str(feat), other=str(other),
                              train=_csv(tmp_path / "train.csv", pos + neg), val=_csv(tmp_path / "val.csv", val))


def run_cache(ws, name, store="store", featurizer=None, metadata=None):
    SpySession.rows = 0
    a = argparse.Namespace(featurizer=featurizer or ws.feat, cache_dir=str(ws.root / name), metadata=metadata or ws.train,
                           val=ws.val, pos_aug=POS_AUG, feature_store=str(ws.root / store) if store else None)
    hb.cache(a)
    c = ws.root / name
    return SpySession.rows, {s: np.load(c / f"{s}.npy") for s in ("train", "val", "calib")}


def test_second_run_featurizes_nothing(ws):
    first, a = run_cache(ws, "c1")
    total = sum(len(v) for v in a.values())
    assert first == total
    again, b = run_cache(ws, "c2")
    assert again == 0
    for s in a:
        np.testing.assert_array_equal(a[s], b[s])


def test_new_clip_featurizes_only_its_windows(ws):
    run_cache(ws, "c1")
    new = (_wav(ws.root / "kws/p_new.wav", 1.0, 99), 1)
    grown = _csv(ws.root / "grown.csv", [new] + ws.pos + ws.neg)
    rows, feats = run_cache(ws, "c2", metadata=grown)
    assert rows == 1 + POS_AUG
    n_pos = len(ws.pos) + 1
    assert len(feats["train"]) == n_pos + len(ws.neg) + n_pos * POS_AUG + len(ws.neg[::4])


def test_store_features_equal_fresh(ws):
    _, fresh = run_cache(ws, "plain", store=None)
    _, first = run_cache(ws, "c1")
    _, cached = run_cache(ws, "c2")
    for s in fresh:
        np.testing.assert_array_equal(first[s], fresh[s])
        np.testing.assert_array_equal(cached[s], fresh[s])


def test_featurizer_change_invalidates(ws):
    _, a = run_cache(ws, "c1")
    total = sum(len(v) for v in a.values())
    rows, b = run_cache(ws, "c2", featurizer=ws.other)
    assert rows == total
    assert not np.array_equal(a["val"], b["val"])
    same_bytes = ws.root / "copy.onnx"; shutil.copy(ws.feat, same_bytes)
    rows, _ = run_cache(ws, "c3", featurizer=str(same_bytes))
    assert rows == 0


def test_store_key_depends_on_runtime(ws, monkeypatch):
    wav = np.random.default_rng(0).standard_normal(SR).astype(np.float32)
    key = lambda provider, device="cpu0": hb.FeatureStore(ws.root / "s", ws.feat, provider, device).key(wav)
    assert key("CPUExecutionProvider") == key("CPUExecutionProvider")
    assert key("CUDAExecutionProvider") != key("CPUExecutionProvider")
    assert key("CUDAExecutionProvider", "RTX 3090") != key("CUDAExecutionProvider", "RTX 4000 Ada")
    before = key("CPUExecutionProvider")
    monkeypatch.setattr(onnxruntime, "__version__", "0.0.0-other")
    assert key("CPUExecutionProvider") != before


def test_cache_keys_on_session_provider(ws, monkeypatch):
    class CudaSession(SpySession):
        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setattr(hb, "device_name", lambda provider: "card")
    monkeypatch.setattr(onnxruntime, "InferenceSession", CudaSession)
    first, _ = run_cache(ws, "c1")
    monkeypatch.setattr(onnxruntime, "InferenceSession", SpySession)
    rows, _ = run_cache(ws, "c2")
    assert rows == first > 0
    rows, _ = run_cache(ws, "c3")
    assert rows == 0


def test_augmented_copies_identical_across_runs(ws):
    _, a = run_cache(ws, "c1", store=None)
    fewer = _csv(ws.root / "fewer.csv", ws.pos[1:] + ws.neg)
    _, b = run_cache(ws, "c2", store=None, metadata=fewer)
    n_a, n_b = len(ws.pos) + len(ws.neg), len(ws.pos) - 1 + len(ws.neg)
    for i in range(1, len(ws.pos)):
        for t in range(POS_AUG):
            np.testing.assert_array_equal(a["train"][n_a + i * POS_AUG + t], b["train"][n_b + (i - 1) * POS_AUG + t])


@pytest.mark.parametrize("device", [False, True])
def test_augmentation_keyed_not_sequential(ws, monkeypatch, device):
    monkeypatch.setattr(hb, "DEVICE", device)
    x = hb.load_window(ws.pos[0][0])
    one, two = hb.augmenter(), hb.augmenter()
    first = [one(x.copy(), k) for k in ("a:0", "b:0", "a:1")]
    second = [two(x.copy(), k) for k in ("a:1", "b:0", "a:0")][::-1]
    for u, v in zip(first, second):
        np.testing.assert_array_equal(u, v)
    assert not np.array_equal(first[0], first[2])


def _expected_lr(lr, epochs, warmup, schedule):
    if schedule == "constant":
        return [lr] * epochs
    out, low, span = [], lr / 100, epochs - 1 - warmup
    for e in range(epochs):
        if e < warmup:
            out.append(lr * (e + 1) / (warmup + 1))
        elif not span:
            out.append(lr)
        else:
            out.append(low + (lr - low) * (1 + math.cos(math.pi * (e - warmup) / span)) / 2)
    return out


@pytest.mark.parametrize("schedule,warmup", [("constant", 0), ("cosine", 0), ("cosine", 1), ("cosine", 3)])
def test_lr_scheduler_sequence(schedule, warmup):
    lr, epochs = 0.1, 10
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    sch = hb.lr_scheduler(opt, schedule, epochs, warmup)
    seen = []
    for _ in range(epochs):
        seen.append(opt.param_groups[0]["lr"])
        if sch is not None:
            sch.step()
    assert seen == pytest.approx(_expected_lr(lr, epochs, warmup, schedule), rel=1e-9, abs=1e-12)


def _multipliers(epochs, warmup, lr=0.1):
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    sch = hb.lr_scheduler(opt, "cosine", epochs, warmup)
    seen = []
    for _ in range(epochs):
        seen.append(opt.param_groups[0]["lr"] / lr)
        sch.step()
    return seen


@pytest.mark.parametrize("epochs,warmup", [(10, 0), (10, 1), (10, 2), (5, 3), (5, 4), (1, 0)])
def test_cosine_multipliers_peak_once_and_end_at_floor(epochs, warmup):
    m = _multipliers(epochs, warmup)
    print(epochs, warmup, [round(x, 4) for x in m])
    assert m[warmup] == pytest.approx(1.0)
    assert sum(abs(x - 1.0) < 1e-12 for x in m) == 1
    assert m[:warmup] == pytest.approx([(e + 1) / (warmup + 1) for e in range(warmup)])
    assert all(a > b for a, b in zip(m[warmup:], m[warmup + 1:]))
    if epochs > warmup + 1:
        assert m[-1] == pytest.approx(0.01, rel=1e-9)
    else:
        assert m[-1] == pytest.approx(1.0)


def test_one_epoch_cosine_runs_at_full_rate():
    assert _multipliers(1, 0) == [1.0]


def test_device_name_is_nonempty_for_cpu():
    assert hb.device_name("CPUExecutionProvider")


def test_cosine_rejects_warmup_not_below_epochs():
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    for warmup in (5, 6):
        with pytest.raises(SystemExit, match="below --epochs"):
            hb.lr_scheduler(opt, "cosine", 5, warmup)


def test_constant_rejects_warmup():
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    with pytest.raises(SystemExit):
        hb.lr_scheduler(opt, "constant", 10, 2)


def _synthetic_cache(c):
    rng = np.random.default_rng(0)
    c.mkdir()
    for split, n, n_pos in (("train", 60, 10), ("val", 12, 4), ("calib", 12, 4)):
        labels = np.r_[np.ones(n_pos), np.zeros(n - n_pos)].astype(np.int8)
        feats = rng.standard_normal((n, 75, 8)).astype(np.float16) + labels[:, None, None]
        np.save(c / f"{split}.npy", feats.astype(np.float16)); np.save(c / f"{split}_labels.npy", labels)


@pytest.mark.parametrize("schedule,warmup", [("constant", 0), ("cosine", 2)])
def test_fit_logs_lr_per_epoch(tmp_path, schedule, warmup):
    _synthetic_cache(tmp_path / "cache")
    a = argparse.Namespace(cache_dir=str(tmp_path / "cache"), out_dir=str(tmp_path / "out"), heads="gru-h16", epochs=6,
                           batch_size=16, lr=5e-3, seed=0, max_pos=0, pos_aug=0, subset_seed=0,
                           lr_schedule=schedule, warmup_epochs=warmup)
    hb.fit(a)
    recs = [json.loads(l) for l in (tmp_path / "out" / "fit.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in recs] == list(range(1, 7))
    assert [r["lr"] for r in recs] == pytest.approx(_expected_lr(5e-3, 6, warmup, schedule), rel=1e-9)
