"""scripts/research/head_bench.py: the content-addressed feature store, the learning-rate schedule, the head and loss axes."""
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
                           val=ws.val, pos_aug=POS_AUG, feature_store=str(ws.root / store) if store else None,
                           featurizer_revision=None, calib_stream_hours=0)
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
                           lr_schedule=schedule, warmup_epochs=warmup, loss="bce")
    hb.fit(a)
    recs = [json.loads(l) for l in (tmp_path / "out" / "fit.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in recs] == list(range(1, 7))
    assert [r["lr"] for r in recs] == pytest.approx(_expected_lr(5e-3, 6, warmup, schedule), rel=1e-9)


@pytest.mark.parametrize("name", ["wakehubert-tiny", "wakehubert-tiny-int8", "wakewav-mel-tcn"])
def test_cache_resolves_pretrained_name(ws, fake_hub, name):
    _, calls = fake_hub
    run_cache(ws, "c1", featurizer=name)
    meta = json.loads((ws.root / "c1" / "meta.json").read_text())
    assert meta["featurizer_name"] == name
    assert meta["featurizer_sha1"] == hb.file_sha1(meta["featurizer"])
    assert Path(meta["featurizer"]).exists()
    assert any(c[1] == "config.json" for c in calls)


def test_cache_path_wins_over_name(ws, fake_hub):
    _, calls = fake_hub
    run_cache(ws, "c1")
    meta = json.loads((ws.root / "c1" / "meta.json").read_text())
    assert meta["featurizer"] == ws.feat and meta["featurizer_name"] is None
    assert meta["featurizer_sha1"] == hb.file_sha1(ws.feat)
    assert calls == []


def test_cache_unknown_featurizer_lists_names(ws):
    with pytest.raises(ValueError, match="wakehubert-tiny"):
        run_cache(ws, "c1", featurizer="no-such-featurizer")


DIM, FRAMES = 16, 75
HEADS = ["gru", "gru-h32-l2", "bigru-l2", "ffn", "gru:hidden_dim=32,linear_dim=16", "bigru", "cnn", "bcresnet", "tcresnet",
         "dscnn", "matchboxnet", "res15", "kwt:d_model=16,n_layers=1,dim_ff=32,n_heads=2", "conformer:d_model=16,n_layers=1,dim_ff=32,n_heads=2",
         "crnn", "mixconv", "efficientnet"]
LOSSES = ["bce", "focal", "bce+0.5*label_smoothing_bce"]


def _fit_ns(tmp_path, heads, loss, epochs=1):
    return argparse.Namespace(cache_dir=str(tmp_path / "cache"), out_dir=str(tmp_path / f"out-{hb.stem(loss)}"), heads=heads, epochs=epochs,
                              batch_size=16, lr=5e-3, seed=0, max_pos=0, pos_aug=0, subset_seed=0,
                              lr_schedule="constant", warmup_epochs=0, loss=loss)


def _wide_cache(c):
    rng = np.random.default_rng(1)
    c.mkdir()
    for split, n, n_pos in (("train", 61, 10), ("val", 12, 4), ("calib", 12, 4)):
        labels = np.r_[np.ones(n_pos), np.zeros(n - n_pos)].astype(np.int8)
        feats = rng.standard_normal((n, FRAMES, DIM)).astype(np.float16) + labels[:, None, None]
        np.save(c / f"{split}.npy", feats); np.save(c / f"{split}_labels.npy", labels)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    torch.set_num_threads(4)
    root = tmp_path_factory.mktemp("heads")
    _wide_cache(root / "cache")
    outs = {}
    for loss in LOSSES:
        a = _fit_ns(root, ",".join(HEADS), loss)
        hb.fit(a)
        outs[loss] = Path(a.out_dir)
    return outs


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("head", HEADS)
def test_head_fits_exports_and_matches_onnxruntime(fitted, head, loss):
    out = fitted[loss]
    ck = torch.load(out / f"{hb.stem(head)}.pt", map_location="cpu")
    assert ck["spec"] == head and ck["frames"] == FRAMES and ck["loss"] == loss
    hb.export(argparse.Namespace(out_dir=str(out), head=head, allow_fixed_frames=True, calibrate=False))
    sess = onnxruntime.InferenceSession(str(out / f"{hb.stem(head)}.onnx"), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["features"] and [o.name for o in sess.get_outputs()] == ["logit"]
    x = torch.randn(5, FRAMES, DIM, generator=torch.Generator().manual_seed(3))
    h = hb.build_head(head, DIM, FRAMES); h.load_state_dict(ck["state"]); h.eval()
    with torch.no_grad():
        want = h(x).numpy()
    got = sess.run(None, {"features": x.numpy()})[0]
    assert got.shape == (5,)
    assert np.abs(got - want).max() < 1e-4


def test_fit_log_has_every_head(fitted):
    rec = json.loads((fitted["bce"] / "fit.jsonl").read_text().splitlines()[0])
    for head in HEADS:
        assert head in rec and {"val_loss", "calib_recall"} <= set(rec[head])


@pytest.mark.parametrize("head", ["ocsvm", "phonmatch"])
def test_unsupported_heads_refused_with_list(head):
    with pytest.raises(SystemExit, match="Supported: .*kwt"):
        hb.build_head(head, DIM, FRAMES)


def test_unknown_head_and_override_refused():
    with pytest.raises(SystemExit, match="unknown head"):
        hb.build_head("nope", DIM, FRAMES)
    with pytest.raises(SystemExit, match="bad override"):
        hb.build_head("kwt:wrong=1", DIM, FRAMES)


@pytest.mark.parametrize("loss", ["triplet", "arcface", "bce+supcon"])
def test_metric_losses_refused(loss):
    with pytest.raises(SystemExit, match="unsupported loss"):
        hb.make_loss(loss)


def test_split_specs_keeps_overrides_with_their_head():
    assert hb.split_specs("gru,kwt:d_model=16,n_layers=1,gru-h256,matchboxnet:kernel_sizes=[3,5],B=2") == [
        "gru", "kwt:d_model=16,n_layers=1", "gru-h256", "matchboxnet:kernel_sizes=[3,5],B=2"]


def test_loss_values_match_manual_formulas():
    logits, y = torch.randn(32, generator=torch.Generator().manual_seed(5)), (torch.arange(32) % 3 == 0).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    assert hb.make_loss("bce") is bce
    p = torch.sigmoid(logits); pt = p * y + (1 - p) * (1 - y); at = 0.9 * y + 0.1 * (1 - y)
    focal = (at * (1 - pt) ** 2 * bce(logits, y, reduction="none")).mean()
    torch.testing.assert_close(hb.make_loss("focal")(logits, y), focal)
    smooth = bce(logits, y * 0.97 + (1 - y) * 0.03)
    torch.testing.assert_close(hb.make_loss("bce+0.5*label_smoothing_bce")(logits, y), bce(logits, y) + 0.5 * smooth)


def test_gru_specs_unchanged(tmp_path):
    _synthetic_cache(tmp_path / "cache")
    for spec, kw in (("gru", {}), ("gru-h16", {"hidden": 16}), ("bigru-l2", {"bidirectional": True, "layers": 2}), ("gru-h8-l2", {"hidden": 8, "layers": 2})):
        torch.manual_seed(0); want = hb.GRUHead(8, **({"bidirectional": spec.startswith("bigru")} | kw))
        torch.manual_seed(0); got = hb.build_head(spec, 8, 75)
        assert type(got) is hb.GRUHead
        for (k, u), (_, v) in zip(want.state_dict().items(), got.state_dict().items()):
            torch.testing.assert_close(u, v, rtol=0, atol=0)
    a = _fit_ns(tmp_path, "gru-h16", "bce", epochs=2)
    hb.fit(a)
    assert (Path(a.out_dir) / "gru-h16.pt").exists()
    assert list(json.loads((Path(a.out_dir) / "fit.jsonl").read_text().splitlines()[0])) == ["epoch", "seconds", "lr", "gru-h16"]


GRU_STATS = json.loads('''{"gru":{"gru.weight_ih_l0":{"shape":[384,8],"sum":20.26725,"norm":3.37355,"head":[0.009572542,-0.04344591,0.01307862,0.03318278,-0.0470199,-0.01020402,-0.04417735,0.02415994]},"gru.weight_hh_l0":{"shape":[384,128],"sum":-18.41922,"norm":13.21476,"head":[0.06359977,0.02968492,-0.01772043,0.04110917,0.03661107,-0.00641006,0.04050865,-0.02374963]},"gru.bias_ih_l0":{"shape":[384],"sum":-1.109461,"norm":1.30126,"head":[0.1050964,-0.0670229,0.04357982,0.02946633,0.03721077,-0.0164894,-0.02817171,-0.009849576]},"gru.bias_hh_l0":{"shape":[384],"sum":-1.526922,"norm":1.164877,"head":[0.008595122,-0.07926787,0.04152873,-0.04916365,0.03407814,0.04634423,0.07036353,0.07263907]},"fc1.weight":{"shape":[128,128],"sum":2.481877,"norm":7.676919,"head":[-0.0289989,-0.08591951,-0.01424739,-0.03440384,0.09355224,0.08658664,0.02158919,0.01792734]},"fc1.bias":{"shape":[128],"sum":0.7504538,"norm":0.714903,"head":[0.006167553,-0.0811911,0.07100405,-0.00470586,-0.04676404,-0.01172547,-0.07187676,-0.05688899]},"fc2.weight":{"shape":[1,128],"sum":-1.005166,"norm":0.8577093,"head":[0.05829805,0.02049479,-0.09076831,0.1238771,0.1195284,0.001469808,0.0418725,0.09401953]},"fc2.bias":{"shape":[1],"sum":-0.0940637,"norm":0.0940637,"head":[-0.0940637]}},"gru-h32":{"gru.weight_ih_l0":{"shape":[96,8],"sum":-0.870832,"norm":3.162624,"head":[0.07667761,0.1176465,-0.2186604,0.002159037,0.1434402,-0.1906859,-0.1165402,0.06154232]},"gru.weight_hh_l0":{"shape":[96,32],"sum":-3.724532,"norm":6.167195,"head":[-0.02532764,0.007438764,0.1935365,0.09775934,0.1785522,-0.1357577,-0.1829407,0.1695127]},"gru.bias_ih_l0":{"shape":[96],"sum":-0.7068716,"norm":1.076083,"head":[0.02580158,-0.02741534,0.2024384,0.156803,-0.04963875,0.06807456,-0.09930953,-0.02878264]},"gru.bias_hh_l0":{"shape":[96],"sum":0.2639527,"norm":1.198856,"head":[0.07658324,0.1453234,0.127755,0.1623382,-0.1340939,-0.004884576,0.02692464,0.09944936]},"fc1.weight":{"shape":[128,32],"sum":-21.1835,"norm":7.134829,"head":[0.02072056,-0.1306497,0.1036961,0.2230873,0.009144102,-0.05623695,0.07290937,-0.1285676]},"fc1.bias":{"shape":[128],"sum":1.011538,"norm":1.158447,"head":[0.04138747,-0.1522674,-0.1652324,-0.0006585387,0.07145704,-0.0006595038,0.119404,-0.09512439]},"fc2.weight":{"shape":[1,128],"sum":-2.611837,"norm":0.8331646,"head":[-0.143642,0.002042858,0.00405497,-0.08073004,0.05472939,-0.007306389,0.0119018,0.05095988]},"fc2.bias":{"shape":[1],"sum":0.008032484,"norm":0.008032484,"head":[0.008032484]}},"bigru-l2":{"gru.weight_ih_l0":{"shape":[384,8],"sum":4.503182,"norm":3.217521,"head":[0.01435531,-0.0003689538,-0.04315874,0.02623316,-0.03166587,-0.01747333,0.05637708,-0.009962951]},"gru.weight_hh_l0":{"shape":[384,128],"sum":7.44817,"norm":12.76154,"head":[-0.02056624,-0.0004645991,-0.04159992,-0.1223268,0.08794341,0.1277049,-0.1199016,-0.1067933]},"gru.bias_ih_l0":{"shape":[384],"sum":0.6856212,"norm":1.204723,"head":[0.08676196,0.1086999,-0.05619913,0.01236032,0.1153004,-0.02737043,-0.09646345,0.05671671]},"gru.bias_hh_l0":{"shape":[384],"sum":0.383342,"norm":1.089446,"head":[0.1083648,0.09681996,-0.03223198,-0.04870263,0.09873038,-0.07596004,-0.0100605,-0.02963528]},"gru.weight_ih_l0_reverse":{"shape":[384,8],"sum":7.383418,"norm":3.138508,"head":[-0.05937491,-0.04489981,-0.03599151,-0.01719253,0.0952749,0.06167167,0.01763724,0.07929633]},"gru.weight_hh_l0_reverse":{"shape":[384,128],"sum":5.59718,"norm":12.61592,"head":[0.004825538,0.04632576,0.009257033,-0.03382205,-0.05265555,-0.08046119,-0.05445744,-0.0380164]},"gru.bias_ih_l0_reverse":{"shape":[384],"sum":-0.00814016,"norm":1.194678,"head":[0.1177038,0.01678623,0.05907471,0.05428985,-0.02015381,-0.07818746,0.02328757,-0.08665131]},"gru.bias_hh_l0_reverse":{"shape":[384],"sum":0.09553804,"norm":1.175989,"head":[0.02622033,0.04329649,-0.013301,-0.01892591,-0.05498704,-0.01118626,-0.07085761,0.02488047]},"gru.weight_ih_l1":{"shape":[384,256],"sum":3.783804,"norm":17.97034,"head":[0.09953619,0.0626815,-0.04553741,0.01672893,-0.0311462,-0.1115331,-0.03736183,0.0470848]},"gru.weight_hh_l1":{"shape":[384,128],"sum":1.153591,"norm":12.72104,"head":[0.02705745,0.03663265,-0.02965341,0.05871195,0.03291792,0.08691599,0.02796552,0.07071445]},"gru.bias_ih_l1":{"shape":[384],"sum":1.280908,"norm":1.165493,"head":[0.05289906,-0.008743019,-0.00150025,0.05528134,-0.0333676,-0.04252719,-0.005398712,0.03404135]},"gru.bias_hh_l1":{"shape":[384],"sum":0.04718511,"norm":1.204926,"head":[-0.006365377,-0.01178271,-0.05532683,-0.0028132,0.09165192,-0.04757838,0.08290718,-0.07795914]},"gru.weight_ih_l1_reverse":{"shape":[384,256],"sum":36.75859,"norm":17.95124,"head":[-0.003612138,0.003534326,-0.004282323,-0.01518323,-0.08352892,0.08732971,0.09786547,0.03520533]},"gru.weight_hh_l1_reverse":{"shape":[384,128],"sum":4.575017,"norm":12.69029,"head":[0.02583395,0.0003129198,-0.03328868,-0.06351355,0.09542017,0.01477515,-0.02606452,-0.04155136]},"gru.bias_ih_l1_reverse":{"shape":[384],"sum":-3.181193,"norm":1.169044,"head":[-0.06960434,-0.008663265,-0.04980082,0.01403959,-0.04638567,0.05685255,0.02876914,-0.02175559]},"gru.bias_hh_l1_reverse":{"shape":[384],"sum":-1.800464,"norm":1.184855,"head":[0.06629532,-0.01790849,0.09095639,-0.04529354,0.04266104,0.1217548,-0.04409941,0.006745295]},"fc1.weight":{"shape":[128,256],"sum":-7.918663,"norm":8.132463,"head":[-0.02737196,0.01934603,-0.052309,0.00378366,-0.02006071,0.09488328,-0.05042426,-0.06275605]},"fc1.bias":{"shape":[128],"sum":0.2158613,"norm":0.4885212,"head":[0.07895437,0.008392983,0.06530862,-0.05919674,-0.02758342,-0.02810712,0.001321153,-0.0710808]},"fc2.weight":{"shape":[1,128],"sum":0.7071619,"norm":0.727278,"head":[-0.1113341,0.03590879,-0.06276818,0.06220233,-0.04883652,0.05514604,-0.08345251,-0.03616302]},"fc2.bias":{"shape":[1],"sum":-0.01190628,"norm":0.01190628,"head":[-0.01190628]}}}''')


def test_gru_fit_matches_statistics_computed_on_dev(tmp_path):
    """Per-tensor shape, sum, L2 norm and first eight values of the state_dicts that origin/dev's head_bench.py saves
    for this seeded fit. Float bits differ between CPUs, so the comparison is a tolerance, not a digest."""
    torch.set_num_threads(2)
    _synthetic_cache(tmp_path / "cache")
    a = _fit_ns(tmp_path, "gru,gru-h32,bigru-l2", "bce", epochs=3)
    a.seed = 7
    hb.fit(a)
    for name, want in GRU_STATS.items():
        state = torch.load(Path(a.out_dir) / f"{name}.pt")["state"]
        assert list(state) == list(want)
        for key, w in want.items():
            v = state[key]
            assert list(v.shape) == w["shape"], (name, key)
            got = [v.double().sum().item(), v.double().norm().item()] + v.flatten()[:8].tolist()
            np.testing.assert_allclose(got, [w["sum"], w["norm"]] + w["head"], rtol=1e-3, atol=1e-5, err_msg=f"{name} {key}")


@pytest.mark.parametrize("head", ["kwt:d_model=16,n_layers=1,dim_ff=32,n_heads=2", "conformer:d_model=16,n_layers=1,dim_ff=32,n_heads=2"])
def test_fixed_frame_heads_need_the_flag_and_record_their_length(fitted, head):
    out = fitted["bce"]
    stem = hb.stem(head)
    (out / f"{stem}.onnx").unlink(missing_ok=True)
    with pytest.raises(SystemExit, match="allow-fixed-frames"):
        hb.export(argparse.Namespace(out_dir=str(out), head=head, allow_fixed_frames=False, calibrate=False))
    assert not (out / f"{stem}.onnx").exists()
    hb.export(argparse.Namespace(out_dir=str(out), head=head, allow_fixed_frames=True, calibrate=False))
    import onnx
    meta = {m.key: m.value for m in onnx.load(str(out / f"{stem}.onnx")).metadata_props}
    assert meta["fixed_frames"] == str(FRAMES)
    sess = onnxruntime.InferenceSession(str(out / f"{stem}.onnx"), providers=["CPUExecutionProvider"])
    with pytest.raises(Exception):
        sess.run(None, {"features": np.zeros((1, FRAMES + 25, DIM), np.float32)})


def test_dynamic_heads_export_without_the_flag_and_carry_no_fixed_frames(fitted):
    import onnx
    out = fitted["bce"]
    hb.export(argparse.Namespace(out_dir=str(out), head="gru", allow_fixed_frames=False, calibrate=False))
    assert "fixed_frames" not in {m.key for m in onnx.load(str(out / "gru.onnx")).metadata_props}


def _calibration_cache(c, stream_hours=1.0):
    rng = np.random.default_rng(1)
    c.mkdir()
    for split, n, n_pos in (("train", 400, 100), ("val", 40, 10), ("calib", 700, 200)):
        labels = np.r_[np.ones(n_pos), np.zeros(n - n_pos)].astype(np.int8)
        shift = labels * (rng.uniform(0.3, 3.0, n) if split == "calib" else 3.0)
        feats = rng.standard_normal((n, 75, 8)) + shift[:, None, None]
        np.save(c / f"{split}.npy", feats.astype(np.float16)); np.save(c / f"{split}_labels.npy", labels)
    np.save(c / "calib_stream.npy", _negative_stream(2, stream_hours))


def _negative_stream(seed, hours):
    return np.random.default_rng(seed).standard_normal((int(hours * 3600 / hb.BLOCK_SECONDS), 75, 8)).astype(np.float16)


def _export(tmp_path, name, **kw):
    a = argparse.Namespace(out_dir=str(tmp_path / "out-bce"), head="gru-h16", calibrate=False, cache_dir=None,
                           target_fa_per_hour=1.0, **kw)
    path = Path(a.out_dir) / "gru-h16.onnx"
    hb.export(a)
    path.rename(path.with_name(f"{name}.onnx"))
    return str(path.with_name(f"{name}.onnx"))


def _logits(onnx_path, feats):
    s = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return np.concatenate([s.run(None, {"features": feats[i:i + 128]})[0] for i in range(0, len(feats), 128)]).astype(np.float64)


def test_calibrated_export_unsaturates_and_holds_the_detection_rate_on_held_out_streams(tmp_path):
    c = tmp_path / "cache"
    _calibration_cache(c)
    hb.fit(_fit_ns(tmp_path, "gru-h16", "bce", epochs=15))
    ck_path = tmp_path / "out-bce" / "gru-h16.pt"
    ck = torch.load(ck_path)
    for k in ("fc2.weight", "fc2.bias"):
        ck["state"][k] *= 10
    torch.save(ck, ck_path)
    target = 120.0
    plain = _export(tmp_path, "plain")
    hb.export(argparse.Namespace(out_dir=str(tmp_path / "out-bce"), head="gru-h16", calibrate=True, cache_dir=str(c),
                                 target_fa_per_hour=target, positives_per_hour=4.0, allow_fixed_frames=False,
                                 allow_extrapolation=False, calib_positives="both"))
    cal = str(tmp_path / "out-bce" / "gru-h16.onnx")
    x = np.load(c / "calib.npy").astype(np.float32)
    y = np.load(c / "calib_labels.npy")
    zp, zc = _logits(plain, x), _logits(cal, x)
    sig = lambda z: 1 / (1 + np.exp(-z))
    assert (sig(zp[y == 1]) >= 0.999999).mean() > 0.25
    assert (sig(zc[y == 1]) >= 0.999999).mean() < 0.05
    assert np.array_equal(np.argsort(zp, kind="stable"), np.argsort(zc, kind="stable"))
    assert abs(np.median(sig(zc[y == 1])) - 0.9) < 0.02
    held_out = _negative_stream(3, 1.0).astype(np.float32)
    detections = hb.debounced_events(_logits(cal, held_out), 0.0)
    assert target / 1.5 <= detections <= target * 1.5
    meta = {m.key: m.value for m in __import__("onnx").load(cal).metadata_props}
    assert meta["calibrated"] == "1"
    assert float(meta["target_fa_per_hour"]) == target and float(meta["calib_a"]) > 0
    assert float(meta["calib_hours"]) == pytest.approx(1.0, rel=0.01) and meta["calib_extrapolated"] == "False"
    assert target / 1.5 <= int(meta["calib_events"]) <= target * 1.5
    assert float(meta["positives_per_hour"]) == 4.0 and 0.5 < float(meta["roc_auc"]) <= 1.0
    assert float(meta["default_threshold"]) == float(meta["f2_threshold"]) and 0.05 <= float(meta["f2_threshold"]) <= 0.99
    assert 0 < float(meta["f2"]) <= 1 and 0 < float(meta["f2_recall"]) <= 1


def test_debounce_counts_one_event_per_quiet_period():
    z = np.full(200, -5.0)
    z[[10, 11, 12, 36, 37, 63]] = 5.0
    assert hb.debounced_events(z, 0.0) == 3
    assert hb.debounced_events(z, 6.0) == 0


def test_thin_tail_is_refused_without_the_flag():
    rng = np.random.default_rng(5)
    hours = 6.0
    fit = rng.laplace(size=int(hours * 3600 / hb.BLOCK_SECONDS))
    with pytest.raises(SystemExit, match="fewer than 20: build one of at least 20 h"):
        hb.fit_calibration(np.full(50, 15.0), fit, hours, 1.0)
    a, b, info = hb.fit_calibration(np.full(50, 15.0), fit, hours, 20 / hours)
    assert not info["calib_extrapolated"] and info["calib_events"] == 20


def test_thin_tail_is_extended_from_the_measured_one():
    rng = np.random.default_rng(5)
    hours, target = 6.0, 2.0
    fit = rng.laplace(size=int(hours * 3600 / hb.BLOCK_SECONDS))
    a, b, info = hb.fit_calibration(np.full(50, 15.0), fit, hours, target, allow_extrapolation=True)
    assert info["calib_extrapolated"] and a > 0
    held_out = rng.laplace(size=int(60 * 3600 / hb.BLOCK_SECONDS))
    rate = hb.debounced_events(a * held_out + b, 0.0) / 60
    assert target / 2 <= rate <= target * 2


def _spiked_stream(spikes=30, blocks=45000):
    z = np.full(blocks, -10.0)
    z[50::blocks // spikes] = 1.0
    return z


def test_f2_threshold_trades_recall_against_false_activations():
    pos = np.r_[np.full(100, 3.0), np.zeros(100)]
    stream = _spiked_stream()
    hours = len(stream) * hb.BLOCK_SECONDS / 3600
    plenty = hb.f2_operating_point(pos, stream, hours, 300 / hours)
    assert plenty["f2_threshold"] == 0.5 and plenty["f2_recall"] == 1.0
    assert plenty["f2"] == pytest.approx(5 * (300 / 330) / (4 * (300 / 330) + 1))
    scarce = hb.f2_operating_point(pos, stream, hours, 3 / hours)
    assert scarce["f2_threshold"] == 0.95 and scarce["f2_recall"] == 0.5
    assert scarce["f2"] == pytest.approx(5 * 0.5 / (4 + 0.5))


def test_roc_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(7)
    pos, neg = rng.normal(1.0, 1.0, 300), np.r_[rng.normal(0.0, 1.0, 5000), np.full(20, 1.0), np.full(20, 0.5)]
    pos = np.r_[pos, 1.0, 0.5]
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    assert hb.roc_auc(pos, neg) == pytest.approx(roc_auc_score(labels, np.r_[pos, neg]), abs=1e-12)


@pytest.mark.parametrize("threshold,good", [(0.5, True), (0.6, True), (0.7, True), (0.49, False), (0.71, False), (0.95, False)])
def test_verdict_accepts_f2_thresholds_between_half_and_seven_tenths(threshold, good):
    verdict = hb.calibration_verdict(threshold)
    assert verdict.startswith("calibration good") == good
    assert good or (verdict.startswith("warning") and f"{threshold:.2f}" in verdict)


def test_calibration_speech_must_not_overlap_training_audio(ws):
    train = hb.read_meta(ws.train)[0]
    with pytest.raises(SystemExit, match="holds the training file"):
        hb.refuse_overlap([str(ws.root / "kws")], ws.train, "data/LibriSpeech/train-clean-100")
    with pytest.raises(SystemExit, match="augmentation pool"):
        hb.refuse_overlap(["data/LibriSpeech"], None, "data/LibriSpeech/train-clean-100")
    with pytest.raises(SystemExit, match="augmentation pool"):
        hb.refuse_overlap(["data/LibriSpeech/train-clean-100/sub"], None, "data/LibriSpeech/train-clean-100")
    hb.refuse_overlap(["data/LibriSpeech/dev-other"], ws.train, "data/LibriSpeech/train-clean-100")
    assert train


def test_calibration_refuses_a_head_that_does_not_separate():
    with pytest.raises(SystemExit, match="nothing to calibrate"):
        hb.fit_calibration(np.zeros(20), np.random.default_rng(0).laplace(size=40000) + 1, 1.0, 100.0)


def test_plugin_windows_are_the_plugins_buffer_after_each_block():
    rng = np.random.default_rng(4)
    hop = round(hb.BLOCK_SECONDS * hb.SR)
    segments = [rng.standard_normal(30 * hop).astype(np.float32) for _ in range(3)]
    padded = np.r_[np.zeros(hb.N, np.float32), *segments]
    wins = [w.copy() for w in hb.plugin_windows(segments)]
    assert len(wins) == sum(len(x) for x in segments) // hop
    for k in (0, 1, 18, 19, len(wins) - 1):
        assert np.array_equal(wins[k], padded[(k + 1) * hop:(k + 1) * hop + hb.N])


def test_calib_stream_holds_each_window_featurized_on_its_own(ws):
    out = ws.root / "stream"
    hours = 0.02
    hb.calib_stream(argparse.Namespace(featurizer=ws.feat, featurizer_revision=None, cache_dir=str(out), calib_stream_hours=hours,
                                       calib_stream_speech=["data/LibriSpeech/dev-other"], metadata=ws.train,
                                       aug_talk="data/LibriSpeech/train-clean-100"))
    stream = np.load(out / "calib_stream.npy")
    segments = list(hb.calibration_stream_audio(hours, ["data/LibriSpeech/dev-other"]))
    audio = np.concatenate(segments)
    assert stream.dtype == np.float16 and stream.ndim == 3
    assert len(stream) == len(audio) // round(hb.BLOCK_SECONDS * hb.SR)
    sess = onnxruntime.InferenceSession(ws.feat, providers=["CPUExecutionProvider"])
    whole = sess.run(None, {"waveform": audio[None]})[0][0]
    hop = round(hb.BLOCK_SECONDS * hb.SR)
    padded = np.r_[np.zeros(hb.N, np.float32), audio]
    for k in (0, 300, len(stream) - 1):
        alone = sess.run(None, {"waveform": padded[None, (k + 1) * hop:(k + 1) * hop + hb.N]})[0][0]
        assert np.allclose(stream[k].astype(np.float32), alone, atol=2e-3)
    k = 300
    end = (k + 1) * hop // 320
    sliced = whole[end - stream.shape[1]:end]
    assert not np.allclose(stream[k].astype(np.float32), sliced, atol=2e-3)


def test_calibration_refuses_a_continuous_feature_stream(tmp_path):
    c = tmp_path / "cache"
    _calibration_cache(c, stream_hours=0.1)
    np.save(c / "calib_stream.npy", np.zeros((int(0.1 * 3600 * 50), 8), np.float16))
    hb.fit(_fit_ns(tmp_path, "gru-h16", "bce", epochs=1))
    with pytest.raises(SystemExit, match="featurizes every 1.5 s window on its own"):
        hb.export(argparse.Namespace(out_dir=str(tmp_path / "out-bce"), head="gru-h16", calibrate=True, cache_dir=str(c),
                                     target_fa_per_hour=1.0, positives_per_hour=4.0, allow_fixed_frames=False,
                                     allow_extrapolation=True, calib_positives="both"))



@pytest.mark.parametrize("mode,expected", [("clean", [1.0] * 3), ("noisy", [2.0] * 4), ("both", [1.0] * 3 + [2.0] * 4)])
def test_calibration_positives_come_from_the_chosen_split(tmp_path, monkeypatch, mode, expected):
    c = tmp_path / "cache"; c.mkdir()
    for split, value, n_pos in (("val", 1.0, 3), ("calib", 2.0, 4)):
        labels = np.r_[np.ones(n_pos), np.zeros(5)].astype(np.int8)
        np.save(c / f"{split}.npy", (labels * value)[:, None, None] * np.ones((1, 75, 8), np.float16))
        np.save(c / f"{split}_labels.npy", labels)
    np.save(c / "calib_stream.npy", np.zeros((50, 75, 8), np.float16))
    seen = {}

    def fit(pos, stream, hours, *args, **kw):
        seen["pos"] = list(np.asarray(pos, np.float64))
        return 1.0, 0.0, {}

    monkeypatch.setattr(hb, "fit_calibration", fit)
    hb.calibration(lambda x: x.mean((1, 2)), argparse.Namespace(cache_dir=str(c), calib_positives=mode, target_fa_per_hour=1.0,
                                                                  positives_per_hour=4.0, allow_extrapolation=False))
    assert seen["pos"] == expected
