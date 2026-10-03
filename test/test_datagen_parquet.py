"""Datagen reads Hugging Face repos stored as Parquet shards with an embedded audio column."""

import io
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ww_trainer import datagen


def _wav_bytes(sr: int = 22050, seconds: float = 1.0, freq: float = 440) -> bytes:
    tone = (0.2 * np.sin(2 * np.pi * freq * np.arange(int(sr * seconds)) / sr)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, tone, sr, format="WAV")
    return buf.getvalue()


class _Audio:
    def __init__(self, decode: bool = True) -> None:
        self.decode = decode


class _ClassLabel:
    def __init__(self, names) -> None:
        self.names = names

    def int2str(self, i):
        return self.names[i]


class _FakeStream:
    def __init__(self, rows, pulled, features=None) -> None:
        self.rows = rows
        self.pulled = pulled
        self.features = features or {"audio": _Audio()}

    def cast_column(self, col, feature):
        assert col == "audio" and feature.decode is False
        return self

    def shuffle(self, seed, buffer_size):
        import random
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return _FakeStream(rows, self.pulled, self.features)

    def __iter__(self):
        for row in self.rows:
            self.pulled.append(1)
            yield row


@pytest.fixture
def hub(monkeypatch):
    """A fake Hub: ``files`` is the repo listing, ``configs`` maps a config name to its rows."""
    import huggingface_hub

    state = types.SimpleNamespace(files=[], configs={}, loads=[], pulled=[], folder_calls=[], label_names=None)
    mod = types.ModuleType("datasets")
    mod.Audio = _Audio
    mod.ClassLabel = _ClassLabel
    mod.get_dataset_config_names = lambda repo: list(state.configs)

    def load_dataset(repo, name=None, split=None, streaming=None, **kw):
        state.loads.append((repo, name, split, streaming))
        features = {"audio": _Audio()}
        if state.label_names:
            features["label"] = _ClassLabel(state.label_names)
        return _FakeStream(state.configs[name], state.pulled, features)

    mod.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", mod)
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files",
                        lambda self, repo, repo_type=None: state.files)
    monkeypatch.setattr(datagen, "_download_audio_files",
                        lambda repo, files, out, sr: state.folder_calls.append(files) or [])
    return state


def _rows(n, label=None):
    return [{"audio": {"bytes": _wav_bytes(freq=200 + 10 * i), "path": f"{i}.wav"},
             **({"label": label} if label is not None else {})}
            for i in range(n)]


def test_parquet_repo_gives_16k_wavs_and_stops_at_the_cap(hub, tmp_path):
    hub.files = ["README.md", "data/train-00000-of-00001.parquet"]
    hub.configs = {"default": _rows(50)}

    files = datagen.download_hf_audio_dataset("org/pq", tmp_path / "out", max_samples=5, sr=16000)

    assert len(files) == 5 and len(hub.pulled) == 5
    assert hub.loads == [("org/pq", "default", "train", True)]
    audio, sr = sf.read(files[0], dtype="float32")
    assert sr == 16000 and abs(len(audio) - 16000) <= 2


def test_uncapped_parquet_repo_reads_every_row(hub, tmp_path):
    hub.files = ["data/train-00000-of-00001.parquet"]
    hub.configs = {"default": _rows(7)}

    assert len(datagen.download_hf_audio_dataset("org/pq", tmp_path / "out", sr=16000)) == 7


def test_every_declared_config_is_read_and_the_cap_is_shared(hub, tmp_path):
    hub.files = ["data/train-00000-of-00001.parquet", "data/omnivoice-00000-of-00001.parquet"]
    hub.configs = {"default": _rows(20), "omnivoice": _rows(20)}

    files = datagen.download_hf_audio_dataset("org/pq", tmp_path / "out", max_samples=5, sr=16000)

    assert [l[1] for l in hub.loads] == ["default", "omnivoice"]
    assert len(files) == 5
    assert len({f.name for f in files}) == 5


def test_hf_config_selects_one_config(hub, tmp_path):
    hub.files = ["data/train-00000-of-00001.parquet", "data/omnivoice-00000-of-00001.parquet"]
    hub.configs = {"default": _rows(20), "omnivoice": _rows(20)}

    files = datagen.download_hf_audio_dataset(
        "org/pq", tmp_path / "out", max_samples=4, sr=16000, hf_config=["omnivoice"])

    assert [l[1] for l in hub.loads] == ["omnivoice"]
    assert len(files) == 4


def test_label_column_is_recorded_beside_the_wavs(hub, tmp_path):
    hub.files = ["data/train-00000-of-00001.parquet"]
    hub.configs = {"default": _rows(3, label="alarm")}

    files = datagen.download_hf_audio_dataset("org/pq", tmp_path / "out", sr=16000)

    lines = (tmp_path / "out" / "labels.csv").read_text().splitlines()
    assert lines[0] == "file,label" and lines[1] == f"{files[0].name},alarm" and len(lines) == 4


def test_missing_config_is_refused_naming_the_repo_and_its_configs(hub, tmp_path):
    hub.files = ["default/train-00000-of-00001.parquet"]
    hub.configs = {"default": _rows(3)}

    with pytest.raises(ValueError, match=r"org/pq.*omnivoice.*\['default'\]"):
        datagen.download_hf_audio_dataset("org/pq", tmp_path / "out", hf_config=["omnivoice"], sr=16000)


def test_hf_config_on_a_repo_without_parquet_is_refused(hub, tmp_path):
    hub.files = ["a.wav"]

    with pytest.raises(ValueError, match="org/folder"):
        datagen.download_hf_audio_dataset("org/folder", tmp_path / "out", hf_config=["default"], sr=16000)


def test_named_config_directories_count_as_parquet(hub, tmp_path):
    hub.files = ["README.md", "omnivoice/train-00000-of-00001.parquet"]
    hub.configs = {"omnivoice": _rows(4)}

    files = datagen.download_hf_audio_dataset("org/pq", tmp_path / "out", sr=16000)

    assert len(files) == 4 and hub.folder_calls == []


def _capped_bytes(hub, tmp_path, name, seed):
    out = tmp_path / name
    files = datagen.download_hf_audio_dataset("org/pq", out, max_samples=5, sr=16000, seed=seed)
    return [f.read_bytes() for f in files]


def test_capped_draw_is_deterministic_per_seed(hub, tmp_path):
    hub.files = ["data/train-00000-of-00001.parquet"]
    hub.configs = {"default": _rows(50)}

    first = _capped_bytes(hub, tmp_path, "a", seed=1)
    again = _capped_bytes(hub, tmp_path, "b", seed=1)
    other = _capped_bytes(hub, tmp_path, "c", seed=2)

    assert first == again
    assert first != other


def test_positives_keep_only_the_positive_label(hub, tmp_path):
    hub.files = ["data/train-00000-of-00001.parquet"]
    rows = _rows(6)
    for i, row in enumerate(rows):
        row["label"] = 1 if i % 2 == 0 else 0
    hub.configs = {"default": rows}

    files = datagen.download_hf_audio_dataset(
        "org/pq", tmp_path / "out", sr=16000, keep_labels={"1", "hey_jarvis"})

    assert len(files) == 3


def test_positives_keep_the_wake_word_label_and_the_cap_counts_kept_rows(hub, tmp_path):
    hub.files = ["data/train-00000-of-00001.parquet"]
    rows = _rows(20)
    for i, row in enumerate(rows):
        row["label"] = "hey_jarvis" if i % 4 == 0 else "other"
    hub.configs = {"default": rows}

    files = datagen.download_hf_audio_dataset(
        "org/pq", tmp_path / "out", max_samples=4, sr=16000, keep_labels={"1", "hey_jarvis"})

    assert len(files) == 4


def test_class_label_index_is_read_by_name(hub, tmp_path):
    hub.files = ["data/train-00000-of-00001.parquet"]
    hub.label_names = ["hey_mycroft", "other"]
    rows = _rows(10)
    for i, row in enumerate(rows):
        row["label"] = 0 if i < 6 else 1
    hub.configs = {"default": rows}

    files = datagen.download_hf_audio_dataset(
        "org/pq", tmp_path / "out", sr=16000, keep_labels={"1", "hey_mycroft"})

    assert len(files) == 6


def test_one_class_label_feature_keeps_its_rows(hub, tmp_path):
    hub.files = ["data/train-00000-of-00001.parquet"]
    hub.label_names = ["hey_mycroft"]
    rows = _rows(4)
    for row in rows:
        row["label"] = 0
    hub.configs = {"default": rows}

    files = datagen.download_hf_audio_dataset(
        "org/pq", tmp_path / "out", sr=16000, keep_labels={"1", "hey_mycroft"})

    assert len(files) == 4


def test_negative_repo_with_several_configs_needs_one_named(hub, tmp_path):
    hub.files = ["data/bal_train/00.parquet"]
    hub.configs = {"balanced": _rows(5), "unbalanced": _rows(5), "full": _rows(5)}

    with pytest.raises(ValueError, match=r"org/audio.*balanced.*repo:config"):
        datagen.download_hf_audio_dataset("org/audio", tmp_path / "out", sr=16000, all_configs=False)

    files = datagen.download_hf_audio_dataset(
        "org/audio", tmp_path / "out", sr=16000, hf_config=["balanced"], all_configs=False)
    assert len(files) == 5
    assert [l[1] for l in hub.loads] == ["balanced"]


def test_audioset_is_listed_with_an_explicit_config():
    assert "agkphysics/AudioSet:balanced" in datagen.NEGATIVE_DATASETS["general"]
    assert datagen.split_source("agkphysics/AudioSet:balanced") == ("agkphysics/AudioSet", "balanced")
    assert datagen.split_source("org/repo") == ("org/repo", None)


def test_pipeline_applies_hf_config_to_the_positive_source_only(tmp_path, monkeypatch):
    calls = {}

    def fake_download(dataset_id, output_dir, **kwargs):
        calls[dataset_id] = kwargs
        return []

    monkeypatch.setattr(datagen, "download_hf_audio_dataset", fake_download)
    cfg = datagen.DatagenConfig(wake_word="hey mycroft", output_dir=tmp_path, hf_config=["omnivoice"],
                                download_augmentation=False)
    with pytest.raises(Exception):
        datagen.run_datagen_pipeline(cfg)

    positive = datagen.find_positive_dataset("hey mycroft")
    assert calls[positive]["hf_config"] == ["omnivoice"]
    assert "1" in calls[positive]["keep_labels"] and "hey_mycroft" in calls[positive]["keep_labels"]
    negatives = [k for k in calls if k != positive]
    assert negatives and all(calls[k]["all_configs"] is False for k in negatives)
    assert calls["agkphysics/AudioSet"]["hf_config"] == ["balanced"]
    assert all(calls[k]["hf_config"] is None for k in negatives if k != "agkphysics/AudioSet")


def test_folder_only_repo_keeps_the_file_path(hub, tmp_path):
    hub.files = ["README.md", "a.wav", "b.wav"]

    datagen.download_hf_audio_dataset("org/folder", tmp_path / "out", max_samples=5, sr=16000)

    assert hub.folder_calls == [["a.wav", "b.wav"]]
    assert hub.loads == []


def test_repo_with_both_layouts_prefers_parquet(hub, tmp_path):
    hub.files = ["a.wav", "b.wav", "data/train-00000-of-00001.parquet"]
    hub.configs = {"default": _rows(4)}

    files = datagen.download_hf_audio_dataset("org/both", tmp_path / "out", max_samples=3, sr=16000)

    assert len(files) == 3
    assert hub.folder_calls == []


def test_real_parquet_shard_round_trips_without_network(tmp_path, monkeypatch):
    import datasets
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard_dir = tmp_path / "repo" / "data"
    shard_dir.mkdir(parents=True)
    table = pa.table({
        "audio": pa.array([{"bytes": _wav_bytes(), "path": f"{i}.wav"} for i in range(5)]),
        "label": ["x"] * 5,
    })
    pq.write_table(table, str(shard_dir / "train-00000-of-00001.parquet"))
    real_load = datasets.load_dataset

    monkeypatch.setattr(datagen, "_list_parquet_configs", lambda repo: ["default"])
    monkeypatch.setattr(
        datasets, "load_dataset",
        lambda repo, name=None, split=None, streaming=None: real_load(
            "parquet", data_files=str(shard_dir / "train-*.parquet"), split=split, streaming=streaming),
    )

    files = datagen.download_hf_audio_dataset("org/local", tmp_path / "out", max_samples=3, sr=16000)

    assert len(files) == 3
    audio, sr = sf.read(files[0], dtype="float32")
    assert sr == 16000 and abs(len(audio) - 16000) <= 2
