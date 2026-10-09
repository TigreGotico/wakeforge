"""Wake-word heads on streamed features (scripts/research/microwakehubert/stream_heads.py): frame labels, the window
index, debounced counting, piecewise scoring of continuous negatives, and a window head trained, calibrated and
exported on toy shards. Nothing is downloaded and no audio is read."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

_P = Path(__file__).resolve().parents[1] / "scripts" / "research" / "microwakehubert" / "stream_heads.py"
_S = importlib.util.spec_from_file_location("stream_heads", _P)
sh = importlib.util.module_from_spec(_S)
_S.loader.exec_module(sh)


def test_labels_mark_frames_after_the_word():
    y = sh.labels_for(200, [(10 * 320, 40 * 320)])
    assert (y[40 + sh.POS_FROM:40 + sh.POS_TO + 1] == 1).all()
    assert (y[25:40 + sh.POS_FROM] == -1).all() and (y[40 + sh.POS_TO + 1:40 + sh.IGNORE_TO + 1] == -1).all()
    assert (y[:25] == 0).all() and (y[40 + sh.IGNORE_TO + 1:] == 0).all()


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


def test_threshold_for_rate_keeps_the_count_at_the_rate():
    rng = np.random.default_rng(0)
    streams = [rng.normal(size=45000) for _ in range(2)]
    hours = 2 * 45000 * sh.BLOCK / sh.SR / 3600
    thr = sh.threshold_for_rate(streams, hours, 5.0)
    assert sum(sh.debounced_events(z, thr) for z in streams) <= 5.0 * hours
    assert sum(sh.debounced_events(z, thr - 0.05) for z in streams) > 5.0 * hours


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


def test_fit_and_export_a_calibrated_window_head(tmp_path):
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
    head = tmp_path / "toy.pt"
    sh.main(["fit", "--shards", str(shards), "--word", "toy", "--calib", str(calib), "--out", str(head),
             "--epochs", "3", "--neg-pool", "2000", "--threads", "1"])
    out = tmp_path / "toy.onnx"
    sh.main(["export", "--head", str(head), "--calib", str(calib), "--training-data", "toy", "--out", str(out)])
    meta = {p.key: p.value for p in onnx.load(str(out)).metadata_props}
    assert meta["featurizer_features"] == "streamed" and meta["window_frames"] == "75"
    assert 0.05 <= float(meta["default_threshold"]) <= 0.99
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = rng.normal(0, 0.3, size=(2, 75, 128)).astype(np.float32)
    x[1, 40:70, :8] += 2.0
    z = sess.run(None, {"features": x})[0]
    assert z.shape == (2,) and z[1] > z[0]
