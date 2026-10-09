"""Wake-word heads on streamed features (scripts/research/microwakehubert/stream_heads.py): frame labels, the window
index, debounced counting, thresholds at a rate and their extrapolation, the voice-disjoint positive split and the
noise split, the trunk record, carried context in ``negs``, the calibration positives of ``calib``, the training
streams of ``streams``, and a window head trained, calibrated and exported on toy shards. The trunk is a toy stateful
ONNX graph built here, so nothing is downloaded."""
import argparse
import importlib.util
import json
import math
import random
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import soundfile as sf
import torch

_P = Path(__file__).resolve().parents[1] / "scripts" / "research" / "microwakehubert" / "stream_heads.py"
_S = importlib.util.spec_from_file_location("stream_heads", _P)
sh = importlib.util.module_from_spec(_S)
_S.loader.exec_module(sh)

RECORD = {"trunk_sha256": "ab" * 32, "pretrained_featurizer": "wakehubert-int8", "featurizer_sha256": "cd" * 32,
          "featurizer_revision": "e30726f3a1c28bb5dffadf101fe26c97e0e3c3e1"}


def test_labels_mark_frames_after_the_word():
    # word over samples 3200..12800: frames 10..39 hold it, frame 40 starts at its end sample; positives are frames
    # 42..54 (2 to 14 after), unscored from the word's middle (frame 25) to frame 70 (30 after)
    y = sh.labels_for(200, [(10 * 320, 40 * 320)])
    assert (y[42:55] == 1).all() and y[41] == -1 and y[55] == -1
    assert (y[25:42] == -1).all() and (y[55:71] == -1).all()
    assert (y[:25] == 0).all() and (y[71:] == 0).all()


def test_window_index_reads_the_frame_before_each_block_end():
    lab = np.zeros((2, 200), np.int8)
    lab[1, 119] = 1
    lab[0, 99] = -1
    pos, neg = sh.window_index([(None, lab)])
    assert pos.tolist() == [[0, 1, 120]]
    ends = set(neg[neg[:, 1] == 0][:, 2].tolist())
    assert min(ends) == 76 and 100 not in ends and all(e % sh.FPB == 0 for e in ends)


def test_debounce_counts_one_detection_per_quiet_period():
    z = np.full(200, -5.0)
    z[[10, 12, 30, 36, 37, 100]] = 5.0
    assert sh.debounced_events(z, 0.0) == 3
    assert sh.debounced_events(z, 6.0) == 0


@pytest.mark.parametrize("gap, events", [(25, 1), (26, 2)])
def test_debounce_skips_exactly_the_plugins_quiet_blocks(gap, events):
    # the plugin's _quiet_samples is 32000 after a detection: 25 blocks of 1280 samples are not scored, the 26th is
    z = np.full(100, -5.0)
    z[[3, 3 + gap]] = 5.0
    assert sh.debounced_events(z, 0.0) == events


def test_threshold_for_rate_keeps_the_count_at_the_rate():
    rng = np.random.default_rng(0)
    streams = [rng.normal(size=45000) for _ in range(2)]
    hours = 2 * 45000 * sh.BLOCK / sh.SR / 3600
    thr = sh.threshold_for_rate(streams, hours, 5.0)
    assert sum(sh.debounced_events(z, thr) for z in streams) <= 5.0 * hours
    assert sum(sh.debounced_events(z, thr - 0.05) for z in streams) > 5.0 * hours


def test_threshold_for_rate_ignores_blocks_before_a_full_window():
    rng = np.random.default_rng(0)
    z = rng.normal(size=45000)
    hours = 45000 * sh.BLOCK / sh.SR / 3600
    lead = z.copy()
    lead[:18] = -np.inf
    want = sh.threshold_for_rate([z[18:]], hours, 5.0)
    assert sh.threshold_for_rate([lead], hours, 5.0) == pytest.approx(want, abs=1e-9)
    assert want < z.max() - 0.5


def test_threshold_for_rate_names_the_streams_that_are_too_short():
    short = np.full(30, -np.inf)
    with pytest.raises(SystemExit, match="no scored block"):
        sh.threshold_for_rate([short, short], 0.01, 1.0)


def test_stream_threshold_extends_an_exponential_tail():
    # spike k of 600, 30 blocks apart, at ln(600 / k): above t lie floor(600 exp(-t)) detections. 20 detections need
    # a threshold just above the 21st spike, 60 just above the 61st; 0.4 h at 1 FA/h is 0.4 detections, extended
    # from there along ln 3 per tripling
    n = 600
    z = np.full(n * 30, -50.0)
    z[np.arange(n) * 30] = np.log(n / np.arange(1, n + 1))
    hours = len(z) * sh.BLOCK / sh.SR / 3600
    assert hours == pytest.approx(0.4)
    thr, extrapolated = sh.stream_threshold(z, hours, 1.0)
    want = math.log(n / 21) + math.log(20 / 0.4) * math.log(61 / 21) / math.log(3)
    assert extrapolated and thr == pytest.approx(want, abs=1e-6)


class _Sum:
    """A stand-in window head session: the logit is the sum of the window."""

    def get_inputs(self):
        return [argparse.Namespace(name="features")]

    def run(self, _, feeds):
        return [feeds["features"].sum((1, 2))]


@pytest.mark.parametrize("tail", [sh.TAIL, 0])
def test_piecewise_negatives_equal_the_whole_stream(tmp_path, tail):
    rng = np.random.default_rng(1)
    feats = rng.normal(size=(3 * 400, 128)).astype(np.float16)
    for j in range(3):
        np.savez(tmp_path / f"d-{j:04d}.npz", seconds=400 / 50, feats=feats[j * 400:(j + 1) * 400])
    sess = _Sum()
    run = sh._window_piece(sess)
    if tail == 0:
        run = lambda d, t, f=sh._window_piece(sess): f(d, None)
    got, hours = sh._neg_logits(sh._by_dir(tmp_path), run)
    want = sh.window_logits(sess, feats)
    assert hours == pytest.approx(3 * 8 / 3600)
    if tail:
        assert np.allclose(got[0], want, equal_nan=True)
    else:
        assert not np.allclose(got[0], want, equal_nan=True)


# ------------------------------------------------------------------ positive pool and splits
def _voices(prefix, n):
    return [f"{prefix}{i}" for i in range(n)]


def test_voice_split_is_disjoint_and_stable():
    rng = random.Random(0)
    pool = [(Path(f"/d/{v}/{k}.wav"), v) for v in _voices("v", 60) for k in range(5)]
    train, held = sh.split_by_voice(pool)
    side = {p: p in set(held) for p, _ in pool}
    assert held and train and not set(train) & set(held) and len(train) + len(held) == len(pool)
    voice = dict(pool)
    assert not {voice[p] for p in train} & {voice[p] for p in held}
    # more files, more voices and another order move no file
    more = pool + [(Path(f"/e/{v}/x.wav"), v) for v in _voices("w", 30)] + [(Path("/e/v3/y.wav"), "v3")]
    rng.shuffle(more)
    train2, held2 = sh.split_by_voice(more)
    held2 = set(held2)
    assert all((p in held2) == side[p] for p in side)
    assert (Path("/e/v3/y.wav") in held2) == side[Path("/d/v3/0.wav")]


def test_noise_halves_are_disjoint_and_stable():
    paths = [Path(f"/n/{i:03d}.wav") for i in range(40)]
    a, b = sh.noise_halves(paths)
    assert a and b and not set(a) & set(b) and sorted(a + b) == sorted(paths)
    a2, b2 = sh.noise_halves(list(reversed(paths)) + [Path("/m/new.wav")])
    assert set(a) <= set(a2) and set(b) <= set(b2)


def _wav(path, n, seed=0, square=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w = np.where(np.arange(n) % 40 < 20, 0.5, -0.5) if square else rng.normal(0, 0.1, n)
    sf.write(str(path), w.astype(np.float32), sh.SR, subtype="FLOAT" if path.suffix == ".wav" else None)
    return path


def test_positive_pool_reads_voices_from_records_patterns_and_directories(tmp_path):
    d = tmp_path / "data"
    for f in ("a1.wav", "a2.wav", "b1.wav", "held_af_heart.wav"):
        _wav(d / "tts" / f, 160)
    (d / "meta").mkdir()
    (d / "meta" / "synthesis.csv").write_text(
        'file,text,plugin,options\n'
        'a1.wav,w,ovos-tts-plugin-edge-tts,"{""voice"": ""en-GB-RyanNeural""}"\n'
        'a2.wav,w,ovos-tts-plugin-edge-tts,"{""voice"": ""en-GB-RyanNeural""}"\n'
        'b1.wav,w,ovos-tts-plugin-google-tx,"{""lang"": ""en-IE""}"\n'
        'held_af_heart.wav,w,kokoro,"{""voice"": ""af_heart""}"\n')
    for f in ("piper/0.flac", "piper/1.flac", "vc/x.wav", "omni/s1.wav"):
        _wav(d / "hub" / f, 160)
    (d / "hub" / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"file_name": "piper/0.flac", "voice": "", "speaker_id": "513", "source_clip": ""},
        {"file_name": "piper/1.flac", "voice": "", "speaker_id": "", "source_clip": "en-US-AvaNeural_1.wav"},
        {"file_name": "vc/x.wav", "voice": "", "speaker_id": "", "source_clip": "orig.wav"},
        {"file_name": "omni/s1.wav", "voice": "auto; seed7", "speaker_id": ""}]) + "\n")
    for f in ("en-AU-NatashaNeural_+0%_0.wav", "en-US-AvaNeural_+0%_0.wav", "noname.wav"):
        _wav(d / "grid" / f, 160)
    _wav(d / "man" / "m1.wav", 160)
    _wav(d / "man" / "unlisted.wav", 160)
    (d / "man" / "manifest.csv").write_text("file,engine,voice_or_seed\nm1.wav,omnivoice,seed2\n")
    sources = [{"dir": "tts", "metadata": "meta/synthesis.csv"}, {"dir": "hub", "metadata": "hub/metadata.jsonl"},
               {"dir": "grid", "voice_pattern": r"^([a-z]{2}-[A-Z]{2}-[A-Za-z]+Neural)_"},
               {"dir": "man", "metadata": "man/manifest.csv", "listed_only": True}]
    got = {p.relative_to(d).as_posix(): v for p, v in sh.positive_pool(sources, d, "en-US-Ava|af_heart")}
    assert got == {"tts/a1.wav": "en-GB-RyanNeural", "tts/a2.wav": "en-GB-RyanNeural", "tts/b1.wav": "en-IE",
                   "hub/piper/0.flac": "513", "hub/vc/x.wav": "dir:vc", "hub/omni/s1.wav": "seed7",
                   "grid/en-AU-NatashaNeural_+0%_0.wav": "en-AU-NatashaNeural", "grid/noname.wav": "dir:grid",
                   "man/m1.wav": "seed2"}
    rev = {p.relative_to(d).as_posix(): v for p, v in sh.positive_pool(sources[::-1], d, "en-US-Ava|af_heart")}
    assert rev == got


def test_positive_pool_finds_clips_by_their_path_under_the_records_directory(tmp_path):
    # a Hub snapshot: train/metadata.jsonl names "piper/<n>.flac", the source directory is train/piper
    snap = tmp_path / "snap" / "train"
    speakers = {"0": "513", "1": "513", "2": "77", "3": "77", "4": "9001"}
    for n in speakers:
        _wav(snap / "piper" / f"{n}.flac", 160)
    _wav(snap / "piper" / "9.flac", 160)
    (snap / "metadata.jsonl").write_text("\n".join(json.dumps(
        {"file_name": f"piper/{n}.flac", "voice": "", "speaker_id": v}) for n, v in speakers.items()) + "\n")
    sources = [{"dir": "snap/train/piper", "metadata": "snap/train/metadata.jsonl"}]
    want = {f"snap/train/piper/{n}.flac": v for n, v in speakers.items()} | {"snap/train/piper/9.flac": "dir:piper"}
    for order in (sources, sources[::-1]):
        got = {p.relative_to(tmp_path).as_posix(): v for p, v in sh.positive_pool(order, tmp_path)}
        assert got == want


def test_committed_sources_hold_every_word_and_exclude_the_evaluation_voices():
    cfg = json.loads(sh.SOURCES.read_text())
    assert set(cfg["words"]) == {"alexa", "hey_mycroft", "hey_jarvis"}
    import re
    held = re.compile(cfg["exclude"])
    for v in ("af_heart", "af_kore", "am_echo", "am_puck", "bf_lily", "bm_fable", "en-US-AvaNeural",
              "en-US-AvaMultilingualNeural", "en-GB-MaisieNeural", "en-IE-ConnorNeural", "en-NZ-MollyNeural",
              "en-SG-WayneNeural", "en-KE-AsiliaNeural", "en-IN-PrabhatNeural"):
        assert held.search(v), v
    for w in cfg["words"].values():
        assert w["sources"] and all("dir" in s for s in w["sources"])


# ------------------------------------------------------------------ toy trunk, negs, calib and streams
class _ToyTrunk(torch.nn.Module):
    """A stateful trunk: frame k of a block is the block's 320-sample frame k plus half and a quarter of the frames
    one and two blocks before, so a frame depends on 2.4 blocks of past audio, carried in the state."""

    def forward(self, wav, state):
        full = torch.cat([state, wav], 1)
        fr = full.reshape(1, 12, 320).abs().mean(-1)
        f = fr[:, 8:12] + 0.5 * fr[:, 4:8] + 0.25 * fr[:, 0:4]
        return f[..., None] * torch.linspace(0.5, 1.5, 128)[None, None], full[:, sh.BLOCK:]


def _toy_trunk(path):
    with torch.no_grad():
        torch.onnx.export(_ToyTrunk(), (torch.zeros(1, sh.BLOCK), torch.zeros(1, 2 * sh.BLOCK)), str(path),
                          input_names=["waveform", "state"], output_names=["features", "state_out"], opset_version=17,
                          dynamo=False)
    m = onnx.load(str(path))
    for k, v in {"source_sha256": "cd" * 32, "source_revision": "r1"}.items():
        e = m.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(m, str(path))
    return path


class _LongTrunk(torch.nn.Module):
    """A stateful trunk whose frame depends on the 31 blocks of audio before its own, each with weight 0.9 ** age."""

    BLOCKS = 31

    def forward(self, wav, state):
        full = torch.cat([state, wav], 1)
        fr = full.reshape(1, 4 * (self.BLOCKS + 1), 320).abs().mean(-1)
        f = sum(0.9 ** age * fr[:, 4 * (self.BLOCKS - age):4 * (self.BLOCKS - age + 1)] for age in range(self.BLOCKS + 1))
        return f[..., None] * torch.linspace(0.5, 1.5, 128)[None, None], full[:, sh.BLOCK:]


def _long_trunk(path):
    with torch.no_grad():
        torch.onnx.export(_LongTrunk(), (torch.zeros(1, sh.BLOCK), torch.zeros(1, 31 * sh.BLOCK)), str(path),
                          input_names=["waveform", "state"], output_names=["features", "state_out"], opset_version=17,
                          dynamo=False)
    return path


def test_trunk_record_names_the_source_featurizer(tmp_path):
    import hashlib
    t = _toy_trunk(tmp_path / "t.onnx")
    rec = sh.trunk_record(t)
    assert rec == {"trunk_sha256": hashlib.sha256(t.read_bytes()).hexdigest(), "pretrained_featurizer": "wakehubert",
                   "featurizer_sha256": "cd" * 32, "featurizer_revision": "r1"}
    q = onnx.load(str(t))
    q.graph.node.append(onnx.helper.make_node("DequantizeLinear", ["q", "qs"], ["dq"]))
    onnx.save(q, str(tmp_path / "q.onnx"))
    assert sh.trunk_record(tmp_path / "q.onnx")["pretrained_featurizer"] == "wakehubert-int8"
    sh.write_trunk_record(tmp_path, rec)
    sh.write_trunk_record(tmp_path, rec)
    with pytest.raises(SystemExit):
        sh.write_trunk_record(tmp_path, {**rec, "trunk_sha256": "00"})


def test_negs_pieces_equal_the_whole_stream(tmp_path, monkeypatch):
    trunk = _toy_trunk(tmp_path / "t.onnx")
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 0.2, 70000), rng.normal(0, 0.4, 63000)
    d = tmp_path / "stream"
    d.mkdir()
    for name, w in (("0.wav", a), ("1.wav", b)):
        sf.write(str(d / name), w.astype(np.float32), sh.SR, subtype="FLOAT")
    monkeypatch.setattr(sh, "NEG_PIECE", 40 * sh.BLOCK)
    sh.main(["negs", "--dirs", str(d), "--trunk", str(trunk), "--out", str(tmp_path / "out")])
    got = np.concatenate([np.load(p)["feats"] for p in sorted((tmp_path / "out").glob("stream-*.npz"))])
    allw = np.concatenate([a, b]).astype(np.float32)
    s = sh.st.OnnxStream(trunk)
    want = s.run(allw[:len(allw) // sh.BLOCK * sh.BLOCK]).astype(np.float16)
    assert len(list((tmp_path / "out").glob("stream-*.npz"))) == 3
    assert got.shape == want.shape and np.array_equal(got, want)


def test_negs_context_covers_a_trunk_of_31_blocks(tmp_path, monkeypatch):
    # the published trunk needs 31 blocks of past audio for exact frames; a context shorter than the trunk's
    # receptive field changes the first frames of every piece after the first
    trunk = _long_trunk(tmp_path / "t.onnx")
    w = np.random.default_rng(4).normal(0, 0.3, 100 * sh.BLOCK).astype(np.float32)
    d = tmp_path / "stream"
    d.mkdir()
    sf.write(str(d / "0.wav"), w, sh.SR, subtype="FLOAT")
    monkeypatch.setattr(sh, "NEG_PIECE", 40 * sh.BLOCK)
    sh.main(["negs", "--dirs", str(d), "--trunk", str(trunk), "--out", str(tmp_path / "out")])
    got = np.concatenate([np.load(p)["feats"] for p in sorted((tmp_path / "out").glob("stream-*.npz"))])
    want = sh.st.OnnxStream(trunk).run(w).astype(np.float16)
    assert len(list((tmp_path / "out").glob("stream-*.npz"))) == 3
    assert got.shape == want.shape and np.array_equal(got, want)


def _calib_fixture(tmp_path):
    """A data root with clips of 12 voices (lengths 8000 + 400 k), the sources config, speech and noise."""
    d = tmp_path / "data"
    voices = [f"spk{i}" for i in range(40)]
    held = [v for v in voices if sh.held_out(v)][:2]
    train = [v for v in voices if not sh.held_out(v)][:10]
    assert len(held) == 2
    rows, lengths = [], {}
    for k, v in enumerate(held + train):
        n = 8000 + 400 * k
        p = _wav(d / "clips" / f"{v}.wav", n, square=True)
        lengths[p] = n
        rows.append({"file_name": f"{v}.wav", "voice": v})
    (d / "clips" / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for i in range(3):
        _wav(d / "speech" / f"s{i}.wav", 3 * sh.SR, seed=i)
    names = [f"n{i}.wav" for i in range(8)]
    assert any(sh.held_out(n, 2) for n in names) and not all(sh.held_out(n, 2) for n in names)
    for i, n in enumerate(names):
        _wav(d / "noise" / n, 2 * sh.SR, seed=10 + i)
    cfg = tmp_path / "sources.json"
    cfg.write_text(json.dumps({"words": {"toy": {"sources": [{"dir": "clips", "metadata": "clips/metadata.jsonl"}]}}}))
    return d, cfg, held, train, lengths


def test_calib_positives_are_the_held_out_voices_with_their_spans(tmp_path):
    d, cfg, held, _, lengths = _calib_fixture(tmp_path)
    trunk = _toy_trunk(tmp_path / "t.onnx")
    out = tmp_path / "calib"
    sh.main(["calib", "--word", "toy", "--sources", str(cfg), "--data-root", str(d), "--speech-dirs",
             str(d / "speech"), "--noise-dir", str(d / "noise"), "--trunk", str(trunk), "--out", str(out)])
    rows = list(np.load(out / "toy-00.npy", allow_pickle=True))
    clips = sorted(d / "clips" / f"{v}.wav" for v in held)
    assert len(rows) == 2
    for (f, s, e), clip in zip(rows, clips):
        n = lengths[clip]
        # the word starts at 3 s (block 37) and the span ends 1.5 s after it; the stream is 4.5 s plus the word
        assert s == 48000 // 1280 == 37
        assert e == (48000 + n + 24000) // 1280
        assert f.shape == ((72000 + n) // 1280 * 4, 128)
    assert json.loads((out / "trunk.json").read_text()) == sh.trunk_record(trunk)


def test_streams_draw_words_from_training_voices_only(tmp_path, monkeypatch):
    d, cfg, held, train, _ = _calib_fixture(tmp_path)
    trunk = _toy_trunk(tmp_path / "t.onnx")
    seen, read = [], sh.read
    monkeypatch.setattr(sh, "read", lambda p: seen.append(Path(p)) or read(p))
    out = tmp_path / "shards"
    sh.main(["streams", "--word", "toy", "--sources", str(cfg), "--data-root", str(d), "--speech-dirs",
             str(d / "speech"), "--window-dirs", str(d / "speech"), "--noise-dir", str(d / "noise"), "--n-streams", "3",
             "--seconds", "12", "--per-stream", "20", "--trunk", str(trunk), "--out", str(out)])
    words = {p.stem for p in seen if p.parent.name == "clips"}
    assert words and words <= set(train) and not words & set(held)
    noise = {p.name for p in seen if p.parent.name == "noise"}
    assert noise and not any(sh.held_out(n, 2) for n in noise)
    assert json.loads((out / "trunk.json").read_text()) == sh.trunk_record(trunk)


# ------------------------------------------------------------------ fit and export
def test_hardest_takes_the_highest_scores():
    rows = np.arange(10)[:, None] * np.ones((1, 3), int)
    scores = np.array([0.1, 5.0, -2.0, 3.0, 0.0, 9.0, 1.0, 2.0, -1.0, 4.0])
    assert sorted(sh.hardest(rows, scores, 3)[:, 0].tolist()) == [1, 5, 9]


def _toy_shards(tmp_path, rng, word, n=24, T=300):
    feats = rng.normal(0, 0.3, size=(n, T, 128)).astype(np.float16)
    labels = np.zeros((n, T), np.int8)
    if word != "none":
        for i in range(n):
            e = rng.integers(100, 250)
            feats[i, e - 30:e, :8] += 2.0
            labels[i] = sh.labels_for(T, [((e - 30) * 320, e * 320)])
    np.save(tmp_path / f"{word}-00-000.feats.npy", feats)
    np.save(tmp_path / f"{word}-00-000.labels.npy", labels)


def _toy_run(tmp_path):
    rng = np.random.default_rng(2)
    shards, calib = tmp_path / "shards", tmp_path / "calib"
    shards.mkdir()
    calib.mkdir()
    _toy_shards(shards, rng, "toy")
    _toy_shards(shards, rng, "none")
    np.savez(calib / "negs-00.npz", feats=rng.normal(0, 0.3, size=(60000, 128)).astype(np.float16))
    rows = []
    for _ in range(12):
        f = rng.normal(0, 0.3, size=(300, 128)).astype(np.float16)
        f[170:200, :8] += 2.0
        rows.append((f, 170 // sh.FPB, 240 // sh.FPB))
    np.save(calib / "toy-00.npy", np.array(rows, dtype=object), allow_pickle=True)
    for d in (shards, calib):
        sh.write_trunk_record(d, RECORD)
    return shards, calib, rows


def _debounced(z, thr):
    n, i = 0, 0
    while i < len(z):
        if z[i] >= thr:
            n, i = n + 1, i + 26
        else:
            i += 1
    return n


SHIPPED_KEYS = {"calibrated", "calib_a", "calib_b", "default_threshold", "target_fa_per_hour", "calib_hours",
                "calib_events", "calib_extrapolated", "roc_auc", "positives_per_hour", "f2", "f2_threshold", "f2_recall",
                "calib_positives", "pretrained_featurizer", "window_frames", "feature_dim", "wake_word", "arch",
                "trainer", "plugin_calibration", "license", "training_data"}


def test_fit_and_export_a_calibrated_window_head(tmp_path):
    from ww_trainer.version import __version__
    shards, calib, rows = _toy_run(tmp_path)
    head = tmp_path / "toy.pt"
    sh.main(["fit", "--shards", str(shards), "--word", "toy", "--calib", str(calib), "--out", str(head),
             "--epochs", "3", "--neg-pool", "2000", "--threads", "1", "--select-rate", "60"])
    log = [json.loads(line) for line in head.with_suffix(".jsonl").read_text().splitlines()]
    assert all({"calib_recall_at_rate", "calib_threshold_at_rate", "calib_recall_at_p999"} <= set(r) for r in log)
    ck = torch.load(head, map_location="cpu")
    best = max(log, key=lambda r: (r["calib_recall_at_rate"], r["calib_recall"]))
    assert ck["epoch"] == best["epoch"] and ck["trunk"] == RECORD

    out = tmp_path / "toy.onnx"
    sh.main(["export", "--head", str(head), "--calib", str(calib), "--rate", "60", "--training-data", "toy",
             "--out", str(out)])
    m = onnx.load(str(out))
    assert [o.name for o in m.graph.output] == ["logit_calibrated"] and [i.name for i in m.graph.input] == ["features"]
    meta = {p.key: p.value for p in m.metadata_props}
    assert SHIPPED_KEYS <= set(meta)
    assert meta["pretrained_featurizer"] == "wakehubert-int8" and meta["featurizer_sha256"] == "cd" * 32
    assert meta["featurizer_revision"] == RECORD["featurizer_revision"] and meta["stream_trunk_sha256"] == "ab" * 32
    assert meta["wakeforge_version"] == __version__ and meta["license"] == "Apache-2.0"
    assert meta["feature_dim"] == "128" and meta["arch"] == "gru" and meta["window_frames"] == "75"
    assert meta["featurizer_features"] == "streamed" and meta["target_fa_per_hour"] == "60.0"
    assert meta["calib_extrapolated"] == "False"

    # the calibration, recomputed here from the trained head: the median positive maps to logit ln 9 (0.9), and the
    # stream's 60 FA/h point to 0 (0.5)
    h = sh.WindowHead()
    h.load_state_dict(ck["state"])
    h.eval()
    a, b = float(meta["calib_a"]), float(meta["calib_b"])

    def block_logits(f):
        ends = [e for e in range(sh.FPB, len(f) + 1, sh.FPB) if e >= 75]
        with torch.no_grad():
            z = h(torch.from_numpy(np.stack([f[e - 75:e] for e in ends]).astype(np.float32))).numpy()
        return np.r_[np.full(len(f) // sh.FPB - len(ends), -np.inf), z].astype(np.float64)

    pos = np.array([block_logits(f)[s:e + 1].max() for f, s, e in rows])
    assert a * np.median(pos) + b == pytest.approx(math.log(9), abs=1e-4)
    z = a * block_logits(np.load(calib / "negs-00.npz")["feats"]) + b
    want = 60 * len(z) * sh.BLOCK / sh.SR / 3600
    assert _debounced(z, 0.0) <= want < _debounced(z, -1e-4)
    assert int(meta["calib_events"]) == _debounced(z, 0.0)

    # the F2-optimal probability, searched here: precision from the hits a stream of positives_per_hour would give
    zpos, hours_ = a * pos + b, len(z) * sh.BLOCK / sh.SR / 3600
    best_f2, best_p = -1.0, None
    for pct in range(5, 100):
        thr = math.log(pct / (100 - pct))
        recall = float(np.mean(zpos >= thr))
        hits = recall * float(meta["positives_per_hour"]) * hours_
        fa = _debounced(z, thr)
        f2 = 5 * hits / (hits + fa) * recall / (4 * hits / (hits + fa) + recall) if hits else 0.0
        if f2 >= best_f2:
            best_f2, best_p = f2, pct / 100
    assert best_p != 0.5
    assert float(meta["f2_threshold"]) == best_p and float(meta["f2"]) == pytest.approx(best_f2, abs=1e-6)
    assert float(meta["default_threshold"]) == best_p

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(5).normal(0, 0.3, size=(2, 75, 128)).astype(np.float32)
    x[1, 40:70, :8] += 2.0
    got = sess.run(None, {"features": x})[0]
    with torch.no_grad():
        ref = a * h(torch.from_numpy(x)).numpy() + b
    assert got.shape == (2,) and np.allclose(got, ref, atol=1e-4)


def test_fit_refuses_a_word_with_no_shards_or_no_positives(tmp_path):
    shards, calib, _ = _toy_run(tmp_path)
    fit = ["fit", "--shards", str(shards), "--calib", str(calib), "--out", str(tmp_path / "x.pt"), "--epochs", "1",
           "--neg-pool", "500", "--threads", "1"]
    with pytest.raises(SystemExit, match="no shard of 'other'"):
        sh.main(fit + ["--word", "other"])
    _toy_shards(shards, np.random.default_rng(3), "none")
    for f in shards.glob("none-00-000.*.npy"):
        f.rename(shards / f.name.replace("none", "quiet"))
    with pytest.raises(SystemExit, match="no positive window of 'quiet'"):
        sh.main(fit + ["--word", "quiet"])
    for f in shards.glob("*.npy"):
        f.unlink()
    with pytest.raises(SystemExit, match="no shard of 'toy'"):
        sh.main(fit + ["--word", "toy"])
    assert not (tmp_path / "x.pt").exists()


def test_fit_and_export_refuse_a_trunk_mismatch(tmp_path):
    shards, calib, _ = _toy_run(tmp_path)
    (calib / "trunk.json").write_text(json.dumps({**RECORD, "trunk_sha256": "ee" * 32}))
    with pytest.raises(SystemExit, match="different trunks"):
        sh.main(["fit", "--shards", str(shards), "--word", "toy", "--calib", str(calib), "--out",
                 str(tmp_path / "x.pt"), "--epochs", "1", "--neg-pool", "500", "--threads", "1"])
    (calib / "trunk.json").unlink()
    with pytest.raises(SystemExit, match="no trunk.json"):
        sh.main(["fit", "--shards", str(shards), "--word", "toy", "--calib", str(calib), "--out",
                 str(tmp_path / "x.pt"), "--epochs", "1", "--neg-pool", "500", "--threads", "1"])
    sh.write_trunk_record(calib, RECORD)
    head = tmp_path / "x.pt"
    sh.main(["fit", "--shards", str(shards), "--word", "toy", "--calib", str(calib), "--out", str(head),
             "--epochs", "1", "--neg-pool", "500", "--threads", "1"])
    (calib / "trunk.json").write_text(json.dumps({**RECORD, "trunk_sha256": "ee" * 32}))
    with pytest.raises(SystemExit, match="another trunk"):
        sh.main(["export", "--head", str(head), "--calib", str(calib), "--training-data", "toy",
                 "--out", str(tmp_path / "x.onnx")])
    ck = torch.load(head, map_location="cpu")
    ck["trunk"] = {**RECORD, "featurizer_revision": ""}
    torch.save(ck, head)
    (calib / "trunk.json").write_text(json.dumps(ck["trunk"]))
    with pytest.raises(SystemExit, match="no source featurizer"):
        sh.main(["export", "--head", str(head), "--calib", str(calib), "--training-data", "toy",
                 "--out", str(tmp_path / "x.onnx")])


@pytest.mark.parametrize("order, kept", [("AB", 2), ("BA", 1)])
def test_fit_keeps_the_epoch_with_the_best_recall_at_the_rate(tmp_path, monkeypatch, order, kept):
    # 0.4 h of stream at 1 FA/h allows no detection, so the threshold sits above the stream's maximum. Epoch A: every
    # positive beats the 99.9th percentile block and none beats the maximum. Epoch B: half beat the maximum.
    shards, calib, _ = _toy_run(tmp_path)
    cz = np.random.default_rng(0).normal(size=18000)
    hours, top, q = 0.4, cz.max(), np.quantile(cz, 0.999)
    epochs = {"A": np.full(4, (top + q) / 2), "B": np.r_[np.full(2, top + 1), np.full(2, q - 1)]}
    calls = iter(order)
    monkeypatch.setattr(sh, "calibration_scores", lambda head, c, w: (epochs[next(calls)], cz, hours))
    head = tmp_path / "x.pt"
    sh.main(["fit", "--shards", str(shards), "--word", "toy", "--calib", str(calib), "--out", str(head),
             "--epochs", "2", "--neg-pool", "500", "--threads", "1"])
    log = {r["epoch"]: r for r in map(json.loads, head.with_suffix(".jsonl").read_text().splitlines())}
    a = log[order.index("A") + 1]
    assert a["calib_recall_at_p999"] == 1.0 and a["calib_recall_at_rate"] == 0.0
    assert log[order.index("B") + 1]["calib_recall_at_rate"] == 0.5
    assert torch.load(head, map_location="cpu")["epoch"] == kept


@pytest.mark.parametrize("order", ["AB", "BA"])
def test_fit_ranks_epochs_by_recall_at_the_rate_before_recall_above_the_maximum(tmp_path, monkeypatch, order):
    # 0.4 h at 10 FA/h allows 4 detections, so the threshold sits below the stream's maximum. Epoch A: every
    # positive beats the threshold and none beats the maximum. Epoch B: half beat the maximum, half miss the threshold.
    shards, calib, _ = _toy_run(tmp_path)
    cz = np.random.default_rng(0).normal(size=18000)
    hours, top = 0.4, cz.max()
    want = 10 * hours
    thr = min(v for v in np.unique(cz) if _debounced(cz, v) <= want)
    assert thr < top - 0.1
    epochs = {"A": np.full(4, (thr + top) / 2), "B": np.r_[np.full(2, top + 1), np.full(2, thr - 1)]}
    calls = iter(order)
    monkeypatch.setattr(sh, "calibration_scores", lambda head, c, w: (epochs[next(calls)], cz, hours))
    head = tmp_path / "x.pt"
    sh.main(["fit", "--shards", str(shards), "--word", "toy", "--calib", str(calib), "--out", str(head),
             "--epochs", "2", "--neg-pool", "500", "--threads", "1", "--select-rate", "10"])
    log = {r["epoch"]: r for r in map(json.loads, head.with_suffix(".jsonl").read_text().splitlines())}
    a, b = log[order.index("A") + 1], log[order.index("B") + 1]
    assert (a["calib_recall_at_rate"], a["calib_recall"]) == (1.0, 0.0)
    assert (b["calib_recall_at_rate"], b["calib_recall"]) == (0.5, 0.5)
    assert torch.load(head, map_location="cpu")["epoch"] == order.index("A") + 1
