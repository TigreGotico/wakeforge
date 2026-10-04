"""Pretrained Hub featurizers ("wakehubert", "wakehubert-int8").

The Hub download is replaced by the ``fake_hub`` fixture (``conftest.py``): a
local directory holding a stand-in ONNX with the published I/O contract
(``waveform`` [B, samples] -> ``features`` [B, samples // 320, 128]) and a
``config.json`` shaped like the published one, so these tests never touch the
network.
"""
import csv
import logging
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from ww_trainer.cli import train
from ww_trainer.factory import create_model
from ww_trainer.feats import OnnxFeatureExtractor
from ww_trainer.inference import OnnxStreamingWakeWord
from ww_trainer.model import GruClassifierHead
from ww_trainer.pretrained import PRETRAINED_FEATURIZERS, _REPOS as REPOS
from ww_trainer.tiers import get_tier

HOP = 320
DIM = 128


def test_featurizer_type_resolves_to_onnx_extractor(fake_hub):
    repo, calls = fake_hub
    model = create_model("gru", featurizer=None, featurizer_type="wakehubert", device="cpu")
    ext = model.feature_extractor
    assert isinstance(ext, OnnxFeatureExtractor)
    assert ext.feature_dim == DIM
    assert Path(ext.model_path) == repo / "wakehubert.onnx"
    assert ext.hop_samples == HOP and ext.frame_rate_hz == 50.0
    assert ext.context_samples == 40000
    assert {c[0] for c in calls} == {"TigreGotico/wakehubert-tiny"}
    assert model.classifier.input_size == DIM


def test_one_and_a_half_seconds_gives_75_frames(fake_hub):
    ext = OnnxFeatureExtractor.from_pretrained("wakehubert", device="cpu")
    feats = ext(torch.zeros(1, 24000))
    assert tuple(feats.shape) == (1, 75, DIM)


def test_int8_name_picks_int8_file_and_pins_revision(fake_hub):
    repo, calls = fake_hub
    model = create_model("gru", featurizer=None, featurizer_type="wakehubert-int8",
                         device="cpu", featurizer_revision="abc123")
    assert Path(model.feature_extractor.model_path) == repo / "wakehubert_int8.onnx"
    assert ("TigreGotico/wakehubert-tiny", "wakehubert_int8.onnx", "abc123") in calls
    assert ("TigreGotico/wakehubert-tiny", "config.json", "abc123") in calls


def test_featurizer_name_with_default_onnx_type(fake_hub):
    repo, _ = fake_hub
    model = create_model("gru", featurizer="wakehubert", device="cpu")
    assert Path(model.feature_extractor.model_path) == repo / "wakehubert.onnx"


def test_cache_key_distinguishes_onnx_files(fake_hub):
    fp32 = OnnxFeatureExtractor.from_pretrained("wakehubert", device="cpu")
    int8 = OnnxFeatureExtractor.from_pretrained("wakehubert-int8", device="cpu")
    assert fp32.feature_dim == int8.feature_dim
    assert fp32.cache_key != int8.cache_key


def test_wakehubert_tier():
    tc = get_tier("wakehubert")
    assert (tc.extractor_type, tc.head_arch, tc.hidden_dim) == ("wakehubert", "gru", 128)


def test_streaming_matches_offline_at_50_fps(fake_hub, tmp_path):
    ext = OnnxFeatureExtractor.from_pretrained("wakehubert", device="cpu")
    torch.manual_seed(0)
    window = 75
    head = GruClassifierHead(hidden_dim=32, input_size=DIM, bidirectional=False,
                             device="cpu").eval()
    head_path = str(tmp_path / "head_streaming.onnx")
    head.export_streaming_onnx(head_path, window=window)

    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(window * HOP)).astype(np.float32)
    with torch.no_grad():
        ref = float(torch.sigmoid(head(ext(torch.from_numpy(audio).unsqueeze(0)))))

    def stream(context_samples):
        sw = OnnxStreamingWakeWord(ext.model_path, head_path, window=window, hidden_dim=32,
                                   hop_samples=ext.hop_samples,
                                   context_samples=context_samples)
        prob = None
        for start in range(0, len(audio), 5 * HOP):  # 0.1 s chunks
            prob = sw.push(audio[start:start + 5 * HOP])
        return prob

    prob = stream(ext.context_samples)
    assert abs(prob - ref) < 1e-4, f"stream={prob:.6f} offline={ref:.6f}"
    # The MFCC-sized default context cuts the stand-in's 20-frame receptive field.
    assert abs(stream(640) - ref) > 1e-3


def _write_wav(path, freq, sr=16000, dur=1.0):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    wav = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(wav.tobytes())


@pytest.mark.parametrize("selector", [["--featurizer-type", "wakehubert", "--arch", "gru"],
                                      ["--tier", "wakehubert"]])
def test_cli_train_with_wakehubert(fake_hub, tmp_path, selector):
    rows_train, rows_test = [], []
    for i in range(6):
        pos, neg = tmp_path / f"pos_{i}.wav", tmp_path / f"neg_{i}.wav"
        _write_wav(pos, 440 + 10 * i)
        _write_wav(neg, 110 + 5 * i)
        rows = rows_train if i < 4 else rows_test
        rows += [(str(pos), "1"), (str(neg), "0")]
    csvs = {}
    for name, rows in (("train.csv", rows_train), ("test.csv", rows_test)):
        csvs[name] = tmp_path / name
        with open(csvs[name], "w", newline="") as f:
            csv.writer(f).writerows(rows)
    out_dir = tmp_path / "model"

    result = CliRunner().invoke(train, [
        "--wake-word", "hey_test",
        "--metadata", str(csvs["train.csv"]),
        "--test-metadata", str(csvs["test.csv"]),
        *selector,
        "--epochs", "1",
        "--batch-size", "4",
        "--device", "cpu",
        "--output-dir", str(out_dir),
        "--feature-cache-dir", str(tmp_path / "feature_cache"),
        "--mine-sample", "0",
        "--pca-every", "0",
    ])
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert list(out_dir.glob("*.pt")), sorted(p.name for p in out_dir.iterdir())
    assert list(out_dir.glob("*.onnx")), sorted(p.name for p in out_dir.iterdir())


@pytest.mark.parametrize("name", sorted(n for n, e in PRETRAINED_FEATURIZERS.items()
                                        if e.repo_id.split("/")[-1] in REPOS))
def test_every_registry_name_resolves(fake_hub, name):
    entry = PRETRAINED_FEATURIZERS[name]
    ext = OnnxFeatureExtractor.from_pretrained(name, device="cpu")
    repo = entry.repo_id.split("/")[-1]
    assert Path(ext.model_path).name == ("wakehubert_int8.onnx" if name.endswith("-int8")
                                         else "wakehubert.onnx")
    assert Path(ext.model_path).parent.name == repo
    assert ext.feature_dim == (256 if repo.endswith("-wide") else DIM)
    assert tuple(ext(torch.zeros(1, 24000)).shape) == (1, 75, ext.feature_dim)
    assert ext.license == REPOS[repo][0]


def test_alias_is_wakehubert_tiny():
    assert PRETRAINED_FEATURIZERS["wakehubert"].repo_id == "TigreGotico/wakehubert-tiny"
    assert PRETRAINED_FEATURIZERS["wakehubert-int8"].variant == "int8"


@pytest.mark.parametrize("name", ["wakehubert-mel-bigru", "wakewav-mel-bigru-int8",
                                  "wakexeus-mel-gru", "wakehubert-mel-attn-int8"])
def test_streaming_refuses_unstreamable(fake_hub, tmp_path, name):
    ext = OnnxFeatureExtractor.from_pretrained(name, device="cpu")
    assert not ext.streaming
    with pytest.raises(ValueError, match="not streamable"):
        OnnxStreamingWakeWord.from_extractor(ext, str(tmp_path / "head.onnx"))


def test_streaming_from_extractor_takes_geometry(fake_hub, tmp_path):
    ext = OnnxFeatureExtractor.from_pretrained("wakehubert-mel-tcn-wide", device="cpu")
    head = GruClassifierHead(hidden_dim=16, input_size=256, bidirectional=False,
                             device="cpu").eval()
    head_path = str(tmp_path / "head.onnx")
    head.export_streaming_onnx(head_path, window=20)
    sw = OnnxStreamingWakeWord.from_extractor(ext, head_path, window=20, hidden_dim=16)
    assert (sw.hop, sw.context) == (HOP, 200 * HOP)
    assert 0.0 <= sw.push(np.zeros(1600, np.float32)) <= 1.0


def test_non_commercial_licence_warns(fake_hub, caplog):
    with caplog.at_level(logging.WARNING, logger="ww_trainer.pretrained"):
        OnnxFeatureExtractor.from_pretrained("wakehubert", device="cpu")
    assert "non-commercial" not in caplog.text
    with caplog.at_level(logging.WARNING, logger="ww_trainer.pretrained"):
        OnnxFeatureExtractor.from_pretrained("wakexeus-mel-tcn", device="cpu")
    assert "non-commercial" in caplog.text and "cc-by-nc-sa-4.0" in caplog.text


class _CacheKeyCaptured(Exception):
    pass


def test_cli_feature_cache_key_distinguishes_onnx_files(fake_hub, tmp_path, monkeypatch):
    """Two ONNX featurizers with the same feature_dim must not share cached features."""
    import ww_trainer.feature_store

    identities = []

    def recording_store(extractor, cache_dir=None, max_bytes=None):
        identities.append(ww_trainer.feature_store.extractor_identity(extractor))
        raise _CacheKeyCaptured

    monkeypatch.setattr(ww_trainer.feature_store.FeatureStore, "for_extractor",
                        staticmethod(recording_store))
    wav = tmp_path / "a.wav"
    _write_wav(wav, 440)
    meta = tmp_path / "train.csv"
    meta.write_text(f"{wav},1\n{wav},0\n")
    for name in ("wakehubert", "wakehubert-int8"):
        result = CliRunner().invoke(train, [
            "--wake-word", "hey_test", "--metadata", str(meta), "--test-metadata", str(meta),
            "--featurizer-type", name, "--device", "cpu", "--output-dir", str(tmp_path / name),
            "--feature-cache-dir", str(tmp_path / "feature_cache"),
        ])
        assert isinstance(result.exception, _CacheKeyCaptured), result.output
    key_a, key_b = identities
    assert key_a and key_b and key_a != key_b
