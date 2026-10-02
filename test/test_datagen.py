"""Tests for ww_trainer.datagen — synthetic data pipeline."""

import csv
import random
import sys
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torchaudio

from ww_trainer import datagen
from ww_trainer.datagen import (
    DatagenConfig,
    DatagenResult,
    GraphemeAugmenter,
    KNOWN_POSITIVE_DATASETS,
    NEGATIVE_DATASETS,
    download_hf_audio_dataset,
    find_positive_dataset,
    normalize_wake_word,
    preprocess_audio,
    read_metadata_csv,
    run_datagen_pipeline,
    synthesize_positives,
    write_metadata_csv,
)


@pytest.fixture(autouse=True)
def _no_parquet_probe(monkeypatch):
    """Keep these tests off the Hub: no repo here is probed for Parquet shards."""
    monkeypatch.setattr(datagen, "_list_parquet_configs", lambda repo: [])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path: Path, sr: int = 16000, duration: float = 0.5) -> Path:
    """Create a short synthetic WAV file."""
    n_samples = int(sr * duration)
    wav = torch.randn(1, n_samples) * 0.5
    path.parent.mkdir(parents=True, exist_ok=True)
    from ww_trainer.dataset import _save_audio
    _save_audio(str(path), wav, sr)
    return path


# ---------------------------------------------------------------------------
# normalize / find
# ---------------------------------------------------------------------------

class TestNormalizeWakeWord:
    def test_spaces_to_underscores(self) -> None:
        assert normalize_wake_word("hey jarvis") == "hey_jarvis"

    def test_uppercase(self) -> None:
        assert normalize_wake_word("Hey Mycroft") == "hey_mycroft"

    def test_hyphens(self) -> None:
        assert normalize_wake_word("hey-computer") == "hey_computer"

    def test_strip(self) -> None:
        assert normalize_wake_word("  alexa  ") == "alexa"


class TestFindPositiveDataset:
    def test_known(self) -> None:
        result = find_positive_dataset("hey_mycroft")
        assert result == "TigreGotico/synthetic-wakeword-hey_mycroft"

    def test_known_with_spaces(self) -> None:
        result = find_positive_dataset("hey mycroft")
        assert result == "TigreGotico/synthetic-wakeword-hey_mycroft"

    def test_unknown(self) -> None:
        assert find_positive_dataset("hey jarvis") is None


# ---------------------------------------------------------------------------
# Metadata CSV
# ---------------------------------------------------------------------------

class TestMetadataCSV:
    def test_roundtrip(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "meta.csv"
        entries = [("/path/a.wav", 1), ("/path/b.wav", 0), ("/path/c.wav", 1)]
        write_metadata_csv(csv_path, entries)
        loaded = read_metadata_csv(csv_path)
        assert loaded == entries

    def test_no_header(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "meta.csv"
        write_metadata_csv(csv_path, [("/a.wav", 1)])
        text = csv_path.read_text()
        assert "path" not in text.lower()
        assert "label" not in text.lower()

    def test_correct_format(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "meta.csv"
        write_metadata_csv(csv_path, [("/a.wav", 1), ("/b.wav", 0)])
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "/a.wav,1"
        assert lines[1] == "/b.wav,0"


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessAudio:
    def test_creates_16khz_mono(self, tmp_path: Path) -> None:
        from ww_trainer.dataset import _load_audio
        src = _make_wav(tmp_path / "src.wav", sr=22050)
        dst = tmp_path / "out" / "dst.wav"
        ok = preprocess_audio(src, dst, sr=16000, vad_trim=False)
        assert ok
        assert dst.exists()
        wav, sr = _load_audio(str(dst))
        assert sr == 16000
        assert wav.shape[0] == 1  # mono

    def test_returns_false_on_bad_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.wav"
        bad.write_text("not audio")
        dst = tmp_path / "out.wav"
        ok = preprocess_audio(bad, dst, sr=16000, vad_trim=False)
        assert not ok


# ---------------------------------------------------------------------------
# HF download (mocked)
# ---------------------------------------------------------------------------

class TestDownloadHFMocked:
    def test_writes_wavs(self, tmp_path: Path) -> None:
        """Mock load_dataset to return synthetic audio examples."""
        n = 5
        fake_examples = []
        for i in range(n):
            fake_examples.append({
                "audio": {
                    "array": np.random.randn(8000).astype(np.float32),
                    "sampling_rate": 16000,
                }
            })

        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(fake_examples))

        # datasets may not be installed; patch the import inside the function
        mock_datasets_mod = MagicMock()
        mock_datasets_mod.load_dataset.return_value = mock_ds
        import sys
        with patch.dict(sys.modules, {"datasets": mock_datasets_mod}):
            out = tmp_path / "hf_out"
            files = download_hf_audio_dataset("fake/dataset", out, max_samples=n)
        assert len(files) == n
        for f in files:
            assert f.exists()
            assert f.suffix == ".wav"


# ---------------------------------------------------------------------------
# TTS synthesis
# ---------------------------------------------------------------------------

class TestCollectTTSPluginsMocked:
    def test_discovers_plugins(self) -> None:
        """Mock find_tts_plugins to verify OPM discovery."""
        import sys
        from ww_trainer.datagen import _collect_tts_plugins

        fake_cls = MagicMock()
        fake_instance = MagicMock()
        fake_cls.return_value = fake_instance

        mock_tts_mod = MagicMock()
        mock_tts_mod.find_tts_plugins.return_value = {"ovos-tts-plugin-fake": fake_cls}

        with patch.dict(sys.modules, {
            "ovos_plugin_manager": MagicMock(),
            "ovos_plugin_manager.tts": mock_tts_mod,
        }):
            result = _collect_tts_plugins("en")

        assert len(result) == 1
        assert result[0][0] == "ovos-tts-plugin-fake"
        assert result[0][1] is fake_instance
        fake_cls.assert_called_once_with(config={"lang": "en"})


class TestSynthesizeNoTTS:
    @patch("ww_trainer.datagen._collect_tts_plugins", side_effect=RuntimeError("No TTS plugins installed"))
    def test_raises_runtime_error(self, _mock: MagicMock, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="No TTS plugins"):
            synthesize_positives("hey_test", tmp_path / "pos", n=5)


# ---------------------------------------------------------------------------
# GraphemeAugmenter
# ---------------------------------------------------------------------------

class TestGraphemeAugmenter:
    def test_generates_n_confusables(self) -> None:
        aug = GraphemeAugmenter(max_edits=2, min_edits=1)
        results = aug.generate_confusables("hello", n_samples=10)
        assert len(results) <= 10
        assert len(results) > 0
        assert "hello" not in results

    def test_all_different_from_original(self) -> None:
        aug = GraphemeAugmenter(max_edits=3, min_edits=1)
        results = aug.generate_confusables("test", n_samples=20)
        for r in results:
            assert r != "test"

    def test_empty_n(self) -> None:
        aug = GraphemeAugmenter(max_edits=1, min_edits=1)
        assert aug.generate_confusables("word", 0) == []


# ---------------------------------------------------------------------------
# Negative dataset purpose mapping
# ---------------------------------------------------------------------------

class TestNegativeDatasetPurposeMapping:
    def test_general_has_esc50_and_nar(self) -> None:
        assert "TigreGotico/ESC-50" in NEGATIVE_DATASETS["general"]
        assert "TigreGotico/NAR" in NEGATIVE_DATASETS["general"]

    def test_bg_noise_datasets(self) -> None:
        bg = NEGATIVE_DATASETS["bg_noise"]
        assert "TigreGotico/ambient_noises" in bg
        assert "TigreGotico/building_106_kitchen_3secs" in bg
        assert "TigreGotico/public_domain_sounds_3secs" in bg

    def test_music_datasets(self) -> None:
        assert "TigreGotico/FMA_3secs" in NEGATIVE_DATASETS["music"]

    def test_rir_datasets(self) -> None:
        assert "davidscripka/MIT_environmental_impulse_responses" in NEGATIVE_DATASETS["rir"]

    def test_no_augmentation_in_general(self) -> None:
        """Augmentation datasets must NOT appear in the general category."""
        general = set(NEGATIVE_DATASETS["general"])
        aug = set(
            NEGATIVE_DATASETS["bg_noise"]
            + NEGATIVE_DATASETS["music"]
            + NEGATIVE_DATASETS["rir"]
        )
        assert general.isdisjoint(aug)


# ---------------------------------------------------------------------------
# DatagenResult.suggested_train_command
# ---------------------------------------------------------------------------

class TestDatagenResultSuggestedCommand:
    def test_includes_augmentation_flags(self, tmp_path: Path) -> None:
        bg = tmp_path / "aug" / "bg_noise"
        bg.mkdir(parents=True)
        music = tmp_path / "aug" / "music"
        music.mkdir(parents=True)
        rir = tmp_path / "aug" / "rir"
        rir.mkdir(parents=True)

        result = DatagenResult(
            train_csv=tmp_path / "train" / "metadata.csv",
            test_csv=tmp_path / "test" / "metadata.csv",
            positives_dir=tmp_path / "positives",
            negatives_dir=tmp_path / "negatives",
            bg_noise_dir=bg,
            music_dir=music,
            rir_dir=rir,
            n_positive=100,
            n_negative=200,
        )
        cmd = result.suggested_train_command("hey jarvis")
        assert "--bg-noise-folder" in cmd
        assert "--music-folder" in cmd
        assert "--rir-folder" in cmd
        assert "--wake-word 'hey_jarvis'" in cmd
        assert "--save-best" in cmd

    def test_omits_missing_dirs(self, tmp_path: Path) -> None:
        result = DatagenResult(
            train_csv=tmp_path / "train" / "metadata.csv",
            test_csv=tmp_path / "test" / "metadata.csv",
            positives_dir=tmp_path / "positives",
            negatives_dir=tmp_path / "negatives",
            bg_noise_dir=None,
            music_dir=None,
            rir_dir=None,
        )
        cmd = result.suggested_train_command("alexa")
        assert "--bg-noise-folder" not in cmd
        assert "--music-folder" not in cmd
        assert "--rir-folder" not in cmd


# ---------------------------------------------------------------------------
# Full pipeline (mocked end-to-end)
# ---------------------------------------------------------------------------

class TestPipelineMockedEndToEnd:
    @patch("ww_trainer.datagen.download_hf_audio_dataset")
    def test_known_wakeword_pipeline(
        self, mock_download: MagicMock, tmp_path: Path
    ) -> None:
        """Mock HF downloads, verify train/test CSVs and dir structure."""
        def fake_download(dataset_id: str, output_dir: Path, **kwargs) -> list:
            output_dir.mkdir(parents=True, exist_ok=True)
            files = []
            n = min(kwargs.get("max_samples", 5) or 5, 5)
            for i in range(n):
                f = _make_wav(output_dir / f"{i:04d}.wav")
                files.append(f)
            return files

        mock_download.side_effect = fake_download

        cfg = DatagenConfig(
            wake_word="hey_mycroft",
            output_dir=tmp_path / "dataset",
            n_positive=5,
            max_negative=5,
            vad_trim=False,
            download_augmentation=True,
            seed=42,
        )
        result = run_datagen_pipeline(cfg)

        assert result.train_csv.exists()
        assert result.test_csv.exists()
        assert result.n_positive > 0
        assert result.n_negative > 0
        assert (tmp_path / "dataset" / "datagen_config.json").exists()

        # Verify augmentation dirs were created
        assert result.bg_noise_dir is not None
        assert result.music_dir is not None
        assert result.rir_dir is not None

        # Verify CSV content
        train = read_metadata_csv(result.train_csv)
        test = read_metadata_csv(result.test_csv)
        assert len(train) + len(test) == result.n_positive + result.n_negative
        # All labels are 0 or 1
        for _, label in train + test:
            assert label in (0, 1)


# ---------------------------------------------------------------------------
# download_hf_audio_dataset decodes encoded bytes without torchcodec
# ---------------------------------------------------------------------------

def _fake_datasets_module(load_dataset=None):
    """A stand-in `datasets` module: the test extra does not install it."""
    import types

    class Audio:
        def __init__(self, decode: bool = True) -> None:
            self.decode = decode

    mod = types.ModuleType("datasets")
    mod.Audio = Audio
    mod.load_dataset = load_dataset
    return mod


class TestDownloadDecodesWithSoundfile:
    def test_encoded_bytes_become_16k_wavs(self, tmp_path, monkeypatch) -> None:
        import io
        import sys
        import numpy as np
        import soundfile as sf
        from ww_trainer import datagen

        hf_datasets = _fake_datasets_module()
        monkeypatch.setitem(sys.modules, "datasets", hf_datasets)

        sr_in = 22050
        tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(sr_in) / sr_in)).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, tone, sr_in, format="WAV")
        row = {"audio": {"bytes": buf.getvalue(), "path": "clip.wav"}}

        class FakeDataset:
            # Like datasets >= 4 without torchcodec: iterating an Audio column
            # that still decodes raises, and only a decode=False cast yields
            # the encoded bytes.
            features = {"audio": hf_datasets.Audio()}

            def __init__(self, decode: bool = True) -> None:
                self.decode = decode

            def cast_column(self, col, feature):
                assert col == "audio"
                return FakeDataset(decode=feature.decode)

            def __iter__(self):
                if self.decode:
                    raise ImportError("To support decoding audio data, please install 'torchcodec'.")
                return iter([row, row])

        monkeypatch.setitem(sys.modules, "torchcodec", None)
        # The folder-repo fast path probes the Hub before load_dataset runs.
        # Stub it, or the test makes a real HfApi call against "org/fake".
        monkeypatch.setattr(datagen, "_list_repo_audio_files", lambda repo: [])
        monkeypatch.setattr(hf_datasets, "load_dataset", lambda *a, **k: FakeDataset())

        files = datagen.download_hf_audio_dataset("org/fake", tmp_path, max_samples=2, sr=16000)

        assert [f.name for f in files] == ["000000.wav", "000001.wav"]
        audio, sr = sf.read(files[0], dtype="float32")
        assert sr == 16000
        assert abs(len(audio) - 16000) <= 2


class TestFolderReposBypassTheDatasetsBuilder:
    def test_audio_files_are_fetched_one_by_one(self, tmp_path, monkeypatch) -> None:
        import sys
        import numpy as np
        import soundfile as sf
        from ww_trainer import datagen

        src = tmp_path / "src"
        src.mkdir()
        for n in ("b.wav", "a.wav", "README.md"):
            if n.endswith(".wav"):
                sf.write(src / n, np.zeros(22050, dtype=np.float32), 22050)
            else:
                (src / n).write_text("card")
        monkeypatch.setitem(sys.modules, "torchcodec", None)
        monkeypatch.setattr(datagen, "_list_repo_audio_files", lambda repo: ["a.wav", "b.wav"])
        import huggingface_hub
        monkeypatch.setattr(huggingface_hub, "snapshot_download",
                            lambda repo, repo_type=None, allow_patterns=None, max_workers=None: str(src))
        monkeypatch.setitem(sys.modules, "datasets", _fake_datasets_module())

        files = datagen.download_hf_audio_dataset("org/folder", tmp_path / "out", max_samples=1, sr=16000)

        assert [f.name for f in files] == ["000000.wav"]
        audio, sr = sf.read(files[0], dtype="float32")
        assert sr == 16000 and abs(len(audio) - 16000) <= 2


# ---------------------------------------------------------------------------
# The pipeline refuses to finish with an empty class
# ---------------------------------------------------------------------------

class TestPipelineRefusesEmptyClasses:
    def test_no_downloads_is_an_error_not_a_dataset(self, tmp_path, monkeypatch) -> None:
        import pytest
        from ww_trainer import datagen

        monkeypatch.setattr(datagen, "download_hf_audio_dataset", lambda *a, **k: [])
        monkeypatch.setattr(datagen, "find_positive_dataset", lambda ww: "org/positives")
        cfg = datagen.DatagenConfig(wake_word="hey test", output_dir=tmp_path,
                                    n_positive=10, download_augmentation=False, adversarial=False,
                                    silence_windows=0)
        with pytest.raises(RuntimeError, match="0 positives and 0 negatives"):
            datagen.run_datagen_pipeline(cfg)
        assert not (tmp_path / "train" / "metadata.csv").exists()


# ---------------------------------------------------------------------------
# A capped download samples the whole repo
# ---------------------------------------------------------------------------

class TestCappedDownloadSamplesTheRepo:
    """A cap must draw from the repo, not from its first directory."""

    def test_cap_covers_more_than_one_class(self, monkeypatch, tmp_path) -> None:
        # A repo that groups clips per class, as TigreGotico/NAR does: the
        # sorted file list starts with every clip of the first class.
        classes = [f"class{c:02d}" for c in range(42)]
        listing = sorted(f"{c}/{c}_{i}.wav" for c in classes for i in range(20))
        captured: list = []

        monkeypatch.setitem(sys.modules, "datasets", _fake_datasets_module())
        monkeypatch.setattr(datagen, "_list_repo_audio_files", lambda repo: listing)
        monkeypatch.setattr(
            datagen, "_download_audio_files",
            lambda dataset_id, files, output_dir, sr: captured.extend(files) or [],
        )

        random.seed(42)
        datagen.download_hf_audio_dataset("org/folder", tmp_path / "out", max_samples=20)

        assert len(captured) == 20
        drawn = {f.split("/")[0] for f in captured}
        # The old slice gave exactly one class. Any spread beats that; require
        # a real one so a future regression to slicing fails here.
        assert len(drawn) > 1, f"cap took {len(drawn)} class(es): {sorted(drawn)}"
        assert listing[:20] != sorted(captured)

    def test_the_draw_is_reproducible_under_the_seed(self, monkeypatch, tmp_path) -> None:
        listing = sorted(f"c{c:02d}/f{i}.wav" for c in range(10) for i in range(10))
        seen: list = []

        monkeypatch.setitem(sys.modules, "datasets", _fake_datasets_module())
        monkeypatch.setattr(datagen, "_list_repo_audio_files", lambda repo: listing)
        monkeypatch.setattr(
            datagen, "_download_audio_files",
            lambda dataset_id, files, output_dir, sr: seen.append(list(files)) or [],
        )
        for _ in range(2):
            random.seed(7)
            datagen.download_hf_audio_dataset("org/folder", tmp_path / "out", max_samples=5)
        assert seen[0] == seen[1]

    def test_a_cap_at_or_above_the_repo_size_takes_everything(self, monkeypatch, tmp_path) -> None:
        listing = ["a/1.wav", "a/2.wav", "b/1.wav"]
        captured: list = []
        monkeypatch.setitem(sys.modules, "datasets", _fake_datasets_module())
        monkeypatch.setattr(datagen, "_list_repo_audio_files", lambda repo: listing)
        monkeypatch.setattr(
            datagen, "_download_audio_files",
            lambda dataset_id, files, output_dir, sr: captured.extend(files) or [],
        )
        datagen.download_hf_audio_dataset("org/folder", tmp_path / "out", max_samples=99)
        assert captured == listing


# ---------------------------------------------------------------------------
# Pipeline runs over fake folder repos
# ---------------------------------------------------------------------------

def _install_folder_repos(monkeypatch, tmp_path: Path, n_files: int = 30) -> None:
    """Make every repo a folder of *n_files* WAVs served from local disk."""
    import huggingface_hub

    src = tmp_path / "hub"

    def list_files(repo):
        return [f"clips/{i:03d}.wav" for i in range(n_files)]

    def snapshot(repo, repo_type=None, allow_patterns=None, max_workers=None):
        root = src / repo.replace("/", "__")
        for name in allow_patterns:
            path = root / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                _make_wav(path, duration=0.3)
        return str(root)

    monkeypatch.setitem(sys.modules, "datasets", _fake_datasets_module())
    monkeypatch.setattr(datagen, "_list_repo_audio_files", list_files)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(datagen, "NEGATIVE_DATASETS", {
        k: [datagen.split_source(v)[0] for v in vs] for k, vs in datagen.NEGATIVE_DATASETS.items()
    })


def _pipeline_config(out: Path, seed: int = 42) -> DatagenConfig:
    return DatagenConfig(
        wake_word="hey_mycroft", output_dir=out, n_positive=6, max_negative=6,
        vad_trim=False, download_augmentation=False, adversarial=False, seed=seed,
        silence_windows=0,  # synthetic silence negatives have their own tests
    )


def _rows(csv_path: Path) -> list:
    with open(csv_path) as f:
        return [tuple(r) for r in csv.reader(f) if r]


class TestSplitIsReproducibleWithCachedDownloads:
    def test_rerun_with_cached_downloads_gives_the_same_split(self, tmp_path, monkeypatch) -> None:
        _install_folder_repos(monkeypatch, tmp_path)
        out = tmp_path / "ds"

        first = run_datagen_pipeline(_pipeline_config(out))
        first_split = (_rows(first.train_csv), _rows(first.test_csv))

        # Every download directory now holds its WAVs, so the second run
        # reads them from disk instead of drawing a sample.
        second = run_datagen_pipeline(_pipeline_config(out))
        assert (_rows(second.train_csv), _rows(second.test_csv)) == first_split


class TestNegativesFromDifferentReposDoNotCollide:
    def test_every_negative_survives_preprocessing(self, tmp_path, monkeypatch) -> None:
        _install_folder_repos(monkeypatch, tmp_path)
        monkeypatch.setattr(datagen, "NEGATIVE_DATASETS", {
            "general": ["org/sounds", "org/alarms"], "speech": ["org/speech"],
        })

        result = run_datagen_pipeline(_pipeline_config(tmp_path / "ds"))

        negatives = [p for p, label in _rows(result.train_csv) + _rows(result.test_csv) if label == "0"]
        assert len(negatives) == 3 * 6
        assert len(set(negatives)) == len(negatives)
        assert all(Path(p).exists() for p in negatives)


class TestUndecodableItemsAreSkipped:
    def test_an_undecodable_row_is_skipped_not_fatal(self, tmp_path, monkeypatch) -> None:
        import io
        import soundfile as sf

        hf_datasets = _fake_datasets_module()
        monkeypatch.setitem(sys.modules, "datasets", hf_datasets)
        buf = io.BytesIO()
        sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
        # An m4a payload: soundfile has no decoder for it.
        rows = [
            {"audio": {"bytes": b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00", "path": "clip.m4a"}},
            {"audio": {"bytes": buf.getvalue(), "path": "clip.wav"}},
        ]

        class FakeDataset:
            features = {"audio": hf_datasets.Audio()}

            def cast_column(self, col, feature):
                return self

            def __iter__(self):
                return iter(rows)

        monkeypatch.setattr(datagen, "_list_repo_audio_files", lambda repo: [])
        monkeypatch.setattr(hf_datasets, "load_dataset", lambda *a, **k: FakeDataset())

        files = datagen.download_hf_audio_dataset("org/parquet", tmp_path / "out", sr=16000)

        assert len(files) == 1
        assert files[0].exists()


class TestCappedLoadStreamsAndStopsAtTheCap:
    def _run(self, tmp_path, monkeypatch, max_samples):
        import io
        import sys
        import soundfile as sf

        hf_datasets = _fake_datasets_module()
        monkeypatch.setitem(sys.modules, "datasets", hf_datasets)
        buf = io.BytesIO()
        sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
        row = {"audio": {"bytes": buf.getvalue(), "path": "clip.wav"}}
        calls: list = []
        pulled = [0]

        class FakeDataset:
            features = {"audio": hf_datasets.Audio()}

            def cast_column(self, col, feature):
                return self

            def __iter__(self):
                for _ in range(50):
                    pulled[0] += 1
                    yield row

        def fake_load(dataset_id, **kwargs):
            calls.append(kwargs)
            return FakeDataset()

        monkeypatch.setattr(datagen, "_list_repo_audio_files", lambda repo: [])
        monkeypatch.setattr(hf_datasets, "load_dataset", fake_load)
        files = datagen.download_hf_audio_dataset(
            "org/parquet", tmp_path / "out", max_samples=max_samples, sr=16000)
        return files, calls, pulled[0]

    def test_a_capped_load_streams_and_reads_no_more_than_the_cap(self, tmp_path, monkeypatch) -> None:
        files, calls, pulled = self._run(tmp_path, monkeypatch, 3)

        assert calls == [{"split": "train", "streaming": True}]
        assert len(files) == 3
        assert pulled == 3

    def test_an_uncapped_load_still_uses_the_cache_first(self, tmp_path, monkeypatch) -> None:
        _, calls, _ = self._run(tmp_path, monkeypatch, None)

        assert calls[0]["streaming"] is False


class TestFolderRepoFetchesOnlyTheDrawnFilesInOneCall:
    def test_a_capped_folder_repo_makes_one_batched_request_for_n_files(self, tmp_path, monkeypatch) -> None:
        import huggingface_hub

        listing = [f"clips/{i:05d}.wav" for i in range(10000)]
        requests: list = []
        singles: list = []

        def snapshot(repo, repo_type=None, allow_patterns=None, max_workers=None):
            requests.append(list(allow_patterns))
            root = tmp_path / "hub"
            for name in allow_patterns:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                _make_wav(path, duration=0.2)
            return str(root)

        monkeypatch.setitem(sys.modules, "datasets", _fake_datasets_module())
        monkeypatch.setattr(datagen, "_list_repo_audio_files", lambda repo: listing)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                            lambda *a, **k: singles.append(a) or "")

        files = datagen.download_hf_audio_dataset("org/folder", tmp_path / "out", max_samples=7, sr=16000)

        assert len(files) == 7
        assert len(requests) == 1 and len(requests[0]) == 7
        assert set(requests[0]) <= set(listing)
        assert singles == []


class TestSnapshotPatternsAreLiteral:
    def test_special_characters_in_names_match_only_the_drawn_files(self, tmp_path, monkeypatch) -> None:
        import huggingface_hub
        from huggingface_hub.utils import filter_repo_objects

        drawn_names = ["a[1].wav", "b*.wav", "c?.wav", "d e.wav"]
        decoys = ["a1.wav", "bxyz.wav", "cz.wav", "d  e.wav", "other.wav"]
        listing = sorted(drawn_names + decoys)
        patterns: list = []

        def snapshot(repo, repo_type=None, allow_patterns=None, max_workers=None):
            patterns.extend(allow_patterns)
            root = tmp_path / "hub"
            root.mkdir(exist_ok=True)
            for name in listing:
                _make_wav(root / name, duration=0.2)
            return str(root)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)

        (tmp_path / "out").mkdir()
        datagen._download_audio_files("org/folder", drawn_names, tmp_path / "out", 16000)

        matched = list(filter_repo_objects(listing, allow_patterns=patterns))
        assert sorted(matched) == sorted(drawn_names)

    def test_a_failed_snapshot_raises_instead_of_yielding_no_clips(self, tmp_path, monkeypatch) -> None:
        import huggingface_hub

        def snapshot(*a, **k):
            raise OSError("network down")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)

        with pytest.raises(RuntimeError, match="org/folder.*network down"):
            datagen._download_audio_files("org/folder", ["a.wav"], tmp_path / "out", 16000)


class TestUnreachableRepoRaises:
    def test_a_failed_listing_and_a_failed_load_raise_naming_both(self, tmp_path, monkeypatch) -> None:
        import huggingface_hub

        hf_datasets = _fake_datasets_module()
        monkeypatch.setitem(sys.modules, "datasets", hf_datasets)

        def broken_listing(self, *a, **k):
            raise OSError("listing refused")

        def broken_load(*a, **k):
            raise ValueError("load refused")

        monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", broken_listing)
        monkeypatch.setattr(hf_datasets, "load_dataset", broken_load)

        with pytest.raises(RuntimeError, match="org/folder.*listing refused.*load refused"):
            datagen.download_hf_audio_dataset("org/folder", tmp_path / "out", max_samples=5)


class TestNotebookForwardsMaxNegative:
    def test_load_generated_passes_the_cap_to_datagen(self, tmp_path, monkeypatch) -> None:
        monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "notebooks"))
        import nb_dataset

        seen: dict = {}

        def fake_pipeline(cfg):
            seen["max_negative"] = cfg.max_negative
            raise RuntimeError("stop after config")

        monkeypatch.setattr(datagen, "run_datagen_pipeline", fake_pipeline)
        with pytest.raises(RuntimeError, match="stop after config"):
            nb_dataset.load_generated(
                wake_word="hey_x", output_dir=str(tmp_path), n_positive=5, lang="en",
                adversarial=False, download_augmentation=False, seed=1, max_negative=7,
            )
        assert seen["max_negative"] == 7


# ---------------------------------------------------------------------------
# the split keeps both labels in both files
# ---------------------------------------------------------------------------

class TestSplitKeepsBothLabelsInBothFiles:
    """The split must not hand back a single-class test set."""

    @staticmethod
    def _labels(csv_path: Path) -> set:
        with open(csv_path) as f:
            return {row[1] for row in csv.reader(f) if row}

    @staticmethod
    def _fake_download(dataset_id, output_dir, **kwargs):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        n = kwargs.get("max_samples") or 5
        return [_make_wav(output_dir / f"{i:04d}.wav") for i in range(n)]

    @pytest.mark.parametrize("size", [2, 3, 5, 10])
    def test_both_splits_hold_both_labels(self, size, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(datagen, "download_hf_audio_dataset", self._fake_download)
        cfg = DatagenConfig(
            wake_word="hey_mycroft", output_dir=tmp_path / "ds",
            n_positive=size, max_negative=size, vad_trim=False,
            download_augmentation=False, seed=42,
        )
        result = datagen.run_datagen_pipeline(cfg)
        assert self._labels(result.train_csv) == {"0", "1"}
        assert self._labels(result.test_csv) == {"0", "1"}

    def test_a_dataset_too_small_to_split_is_refused(self, tmp_path, monkeypatch) -> None:
        # One clip of each label cannot give both labels to both files. The old
        # code wrote a single-class test set and the completeness guard called
        # it fine; now it raises.
        monkeypatch.setattr(datagen, "download_hf_audio_dataset", self._fake_download)
        cfg = DatagenConfig(
            wake_word="hey_mycroft", output_dir=tmp_path / "ds",
            n_positive=1, max_negative=1, vad_trim=False,
            download_augmentation=False, seed=42,
        )
        with pytest.raises(RuntimeError, match="single-class test set"):
            datagen.run_datagen_pipeline(cfg)


# ---------------------------------------------------------------------------
# Every training example is a fixed-length window
# ---------------------------------------------------------------------------


class TestFixedWindows:
    def test_every_example_is_window_long_and_negatives_include_a_short_fragment(self) -> None:
        import random
        import numpy as np
        from ww_trainer.datagen import fit_to_window, negative_windows

        rng = random.Random(0)
        sr, w = 16000, 1.5
        short = np.ones(int(0.6 * sr), dtype=np.float32)
        assert len(fit_to_window(short, sr, w, rng)) == int(w * sr)
        long = np.random.RandomState(0).randn(10 * sr).astype(np.float32)
        wins = negative_windows(long, sr, w, 3, rng)
        assert len(wins) == 4 and all(len(x) == int(w * sr) for x in wins)
        assert any((x == 0).mean() > 0.3 for x in wins)      # the short zero-padded fragment
        assert sum((x == 0).mean() < 0.01 for x in wins) == 3  # three full windows
        one_second = np.ones(sr, dtype=np.float32)
        wins = negative_windows(one_second, sr, w, 3, rng)
        assert len(wins) == 2 and all(len(x) == int(w * sr) for x in wins)


# ---------------------------------------------------------------------------
# A positive longer than the window is not blind-cropped
# ---------------------------------------------------------------------------


def _make_word_clip(path: Path, word_start: float, word_dur: float, total_dur: float,
                     sr: int = 16000) -> Path:
    """A clip of silence with one loud burst (the "word") at *word_start*."""
    wav = np.zeros(int(sr * total_dur), dtype=np.float32)
    start = int(sr * word_start)
    end = start + int(sr * word_dur)
    wav[start:end] = 0.8
    path.parent.mkdir(parents=True, exist_ok=True)
    from ww_trainer.dataset import _save_audio
    _save_audio(str(path), torch.tensor(wav).unsqueeze(0).float(), sr)
    return path


def _add_burst(path: Path, start: float, dur: float, sr: int = 16000) -> Path:
    import soundfile as sf
    wav, _ = sf.read(str(path), dtype="float32")
    wav[int(sr * start):int(sr * (start + dur))] = 0.8
    sf.write(str(path), wav, sr)
    return path


class TestLongPositiveIsNotBlindCropped:
    def test_off_centre_long_positive_is_skipped_not_corrupted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        def fake_download(dataset_id: str, output_dir: Path, **kwargs) -> list:
            output_dir.mkdir(parents=True, exist_ok=True)
            if dataset_id != "TigreGotico/synthetic-wakeword-hey_mycroft":
                # One short negative clip per negative dataset, so Stage 5's
                # empty-class check does not fire on an unrelated class.
                return [_make_word_clip(output_dir / "neg.wav", 0.0, 0.1, 1.0)]
            return [
                # Two clips short enough to survive as-is: PR #74's split
                # guard refuses a test set holding only one class, so the
                # intentional skip below needs at least two survivors.
                _make_word_clip(output_dir / "good.wav", word_start=0.5, word_dur=0.4, total_dur=1.0),
                _make_word_clip(output_dir / "good2.wav", word_start=0.3, word_dur=0.4, total_dur=1.0),
                # The word sits at 0.2-0.6s of a 5s clip, well off-centre: a
                # uniform random crop to the 1.5s window misses it most of
                # the time (the reviewer's probe lost it in 83% of trials).
                # A second burst at 4.5s keeps the clip long after the
                # silence trim, so only the skip can protect the word.
                _add_burst(_make_word_clip(output_dir / "bad_long.wav", word_start=0.2, word_dur=0.4,
                                           total_dur=5.0), 4.5, 0.4),
            ]

        monkeypatch.setattr(datagen, "download_hf_audio_dataset", fake_download)
        cfg = DatagenConfig(
            wake_word="hey_mycroft", output_dir=tmp_path / "ds", n_positive=3,
            vad_trim=False, download_augmentation=False, adversarial=False, seed=1,
            allow_positive_skips=True,
        )

        with caplog.at_level(logging.WARNING):
            result = run_datagen_pipeline(cfg)

        proc_pos = sorted((tmp_path / "ds" / "positives" / "processed").glob("*.wav"))
        assert [p.name for p in proc_pos] == ["good.wav", "good2.wav"]
        assert result.n_positive == 2
        assert any("bad_long.wav" in r.message and "window" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Windows of one source clip stay on one side of the split
# ---------------------------------------------------------------------------


def _source_of(path: str) -> str:
    import re
    return re.sub(r"_w\d+$", "", Path(path).stem)


class TestSplitKeepsSourceClipsApart:
    def _download(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_download(dataset_id: str, output_dir: Path, **kwargs) -> list:
            output_dir.mkdir(parents=True, exist_ok=True)
            if dataset_id == "TigreGotico/synthetic-wakeword-hey_mycroft":
                return [_make_word_clip(output_dir / f"pos{i}.wav", 0.2, 0.4, 1.0) for i in range(10)]
            return [_make_wav(output_dir / f"neg{i}.wav", duration=4.0 + i % 3) for i in range(12)]

        monkeypatch.setattr(datagen, "download_hf_audio_dataset", fake_download)

    def test_no_source_clip_is_in_both_train_and_test(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._download(monkeypatch)
        cfg = DatagenConfig(
            wake_word="hey_mycroft", output_dir=tmp_path / "ds", n_positive=10,
            vad_trim=False, download_augmentation=False, adversarial=False, seed=3,
        )

        result = run_datagen_pipeline(cfg)

        train = {_source_of(p) for p, _ in read_metadata_csv(result.train_csv)}
        test = {_source_of(p) for p, _ in read_metadata_csv(result.test_csv)}
        assert test and train
        assert train & test == set()

    def test_whole_groups_are_assigned_and_singletons_split_randomly(self) -> None:
        from ww_trainer.datagen import split_groups

        random.seed(0)
        groups = [[(f"a{i}_w{j}", 0) for j in range(4)] for i in range(5)]
        groups += [[(f"s{i}", 0)] for i in range(20)]
        train, test = split_groups(groups, 0.2)

        assert len(train) + len(test) == 40
        assert test and train
        sources = lambda es: {n.split("_w")[0] for n, _ in es if "_w" in n}
        assert sources(train) & sources(test) == set()

    def test_a_single_group_stays_in_train(self) -> None:
        from ww_trainer.datagen import split_groups

        train, test = split_groups([[("a_w0", 0), ("a_w1", 0)]], 0.2)
        assert len(train) == 2 and test == []


class TestNearMissPhraseStaysOnOneSide:
    def test_renderings_of_one_phrase_are_not_split_across_train_and_test(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stem_phrase: dict = {}

        def fake_download(dataset_id: str, output_dir: Path, **kwargs) -> list:
            output_dir.mkdir(parents=True, exist_ok=True)
            return [_make_wav(output_dir / f"neg{i}.wav") for i in range(4)]

        def fake_synth(text: str, output_dir: Path, n: int = 1, lang: str = "en", tts_config=None) -> list:
            output_dir.mkdir(parents=True, exist_ok=True)
            files = [_make_wav(output_dir / f"{uuid.uuid4().hex[:8]}.wav", duration=1.0) for _ in range(n)]
            stem_phrase.update({f.stem: text for f in files})
            return files

        monkeypatch.setattr(datagen, "download_hf_audio_dataset", fake_download)
        monkeypatch.setattr(datagen, "generate_adversarial_texts", lambda *a, **k: [])
        monkeypatch.setattr(datagen, "synthesize_positives", fake_synth)
        adv_file = tmp_path / "near_miss.txt"
        adv_file.write_text("".join(f"hey phrase{i}\n" for i in range(8)))
        cfg = DatagenConfig(
            wake_word="hey test", output_dir=tmp_path / "ds", n_positive=3,
            vad_trim=False, download_augmentation=False, seed=1,
            adversarial=True, adversarial_file=str(adv_file), adversarial_per_text=3,
        )

        result = run_datagen_pipeline(cfg)

        def phrases(csv_path: Path) -> set:
            out = set()
            for path, label in read_metadata_csv(csv_path):
                if label == 0 and "adversarial" in path:
                    out.add(stem_phrase[_source_of(path).split("adversarial_")[-1]])
            return out

        train, test = phrases(result.train_csv), phrases(result.test_csv)
        assert train and test
        assert train & test == set()


# ---------------------------------------------------------------------------
# Skipped positives are counted and a mostly-skipped run is refused
# ---------------------------------------------------------------------------


def _padded_clip(path: Path, pad_s: float, word_s: float, sr: int = 16000) -> Path:
    return _make_word_clip(path, word_start=pad_s, word_dur=word_s, total_dur=2 * pad_s + word_s, sr=sr)


class TestEnergyTrimOnNoisyClips:
    @pytest.mark.parametrize("snr_db", [30, 20])
    def test_padding_with_recording_noise_is_trimmed(self, snr_db: int) -> None:
        from ww_trainer.datagen import trim_silence_energy

        rng = np.random.RandomState(snr_db)
        sr = 16000
        word = rng.randn(int(0.8 * sr)).astype(np.float32) * 0.3
        noise_rms = 0.3 / 10 ** (snr_db / 20)
        wav = rng.randn(int(2.0 * sr)).astype(np.float32) * noise_rms
        wav[int(0.5 * sr):int(1.3 * sr)] += word

        trimmed = trim_silence_energy(wav, sr)

        assert len(trimmed) <= int(1.5 * sr)
        assert len(trimmed) >= int(0.75 * sr)


class TestPositiveSkipSummary:
    def _download(self, monkeypatch: pytest.MonkeyPatch, positives) -> None:
        def fake_download(dataset_id: str, output_dir: Path, **kwargs) -> list:
            output_dir.mkdir(parents=True, exist_ok=True)
            if dataset_id == "TigreGotico/synthetic-wakeword-hey_mycroft":
                return [make(output_dir / f"pos{i}.wav") for i, make in enumerate(positives)]
            return [_make_wav(output_dir / f"neg{i}.wav", duration=3.0) for i in range(6)]

        monkeypatch.setattr(datagen, "download_hf_audio_dataset", fake_download)

    def _cfg(self, tmp_path: Path, n: int, **kw) -> DatagenConfig:
        return DatagenConfig(
            wake_word="hey_mycroft", output_dir=tmp_path / "ds", n_positive=n,
            vad_trim=False, download_augmentation=False, adversarial=False, seed=1, **kw,
        )

    def test_padded_clips_are_trimmed_and_kept_without_vad(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._download(monkeypatch, [lambda p, i=i: _padded_clip(p, 0.4 + 0.05 * (i % 4), 0.6) for i in range(8)])

        result = run_datagen_pipeline(self._cfg(tmp_path, 8))

        assert result.n_positive == 8

    def test_summary_line_reports_counts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         caplog: pytest.LogCaptureFixture) -> None:
        import logging

        positives = [lambda p: _make_word_clip(p, 0.5, 0.4, 1.0)] * 9
        positives.append(lambda p: _make_word_clip(p, 0.5, 2.0, 3.0))
        self._download(monkeypatch, positives)

        with caplog.at_level(logging.INFO):
            run_datagen_pipeline(self._cfg(tmp_path, 10))

        assert any("Positives: 9 kept, 1 skipped too long, 0 skipped other (of 10)" in r.message
                   for r in caplog.records)

    def test_refuses_when_over_a_fifth_is_skipped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        positives = [lambda p: _make_word_clip(p, 0.5, 0.4, 1.0)] * 3
        positives += [lambda p: _make_word_clip(p, 0.5, 2.0, 3.0)] * 2
        self._download(monkeypatch, positives)

        with pytest.raises(RuntimeError, match="2 of 5 positives were skipped"):
            run_datagen_pipeline(self._cfg(tmp_path, 5))

    def test_allow_flag_overrides_the_refusal(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        positives = [lambda p: _make_word_clip(p, 0.5, 0.4, 1.0)] * 3
        positives += [lambda p: _make_word_clip(p, 0.5, 2.0, 3.0)] * 2
        self._download(monkeypatch, positives)

        result = run_datagen_pipeline(self._cfg(tmp_path, 5, allow_positive_skips=True))

        assert result.n_positive == 3


# ---------------------------------------------------------------------------
# --adversarial-file
# ---------------------------------------------------------------------------


class TestAdversarialFile:
    def _patch_pipeline(self, monkeypatch: pytest.MonkeyPatch):
        def fake_download(dataset_id: str, output_dir: Path, **kwargs) -> list:
            # One real negative per negative dataset, so Stage 5's
            # empty-class check never fires for a reason unrelated to the
            # adversarial-file handling under test here.
            output_dir.mkdir(parents=True, exist_ok=True)
            return [_make_wav(output_dir / "neg.wav")]

        monkeypatch.setattr(datagen, "download_hf_audio_dataset", fake_download)
        monkeypatch.setattr(datagen, "generate_adversarial_texts", lambda *a, **k: [])
        calls: list = []

        def fake_synth(text: str, output_dir: Path, n: int = 1, lang: str = "en", tts_config=None) -> list:
            calls.append((text, n))
            output_dir.mkdir(parents=True, exist_ok=True)
            return [_make_wav(output_dir / f"{uuid.uuid4().hex[:8]}.wav") for _ in range(n)]

        monkeypatch.setattr(datagen, "synthesize_positives", fake_synth)
        return calls

    def _cfg(self, tmp_path: Path, adv_file: str | None, adversarial: bool = True) -> DatagenConfig:
        return DatagenConfig(
            wake_word="hey test", output_dir=tmp_path / "ds", n_positive=3,
            vad_trim=False, download_augmentation=False, seed=1,
            adversarial=adversarial, adversarial_file=adv_file, adversarial_per_text=2,
        )

    def test_phrases_from_file_become_negatives(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._patch_pipeline(monkeypatch)
        adv_file = tmp_path / "near_miss.txt"
        adv_file.write_text("hey rest\nhey west\n")

        result = run_datagen_pipeline(self._cfg(tmp_path, str(adv_file)))

        texts_rendered_per_text = {text for text, n in calls if n == 2}
        assert {"hey rest", "hey west"} <= texts_rendered_per_text
        entries = read_metadata_csv(result.train_csv) + read_metadata_csv(result.test_csv)
        negatives = [p for p, label in entries if label == 0]
        assert any("adversarial" in p for p in negatives)

    def test_missing_file_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_pipeline(monkeypatch)
        cfg = self._cfg(tmp_path, str(tmp_path / "does_not_exist.txt"))
        with pytest.raises(FileNotFoundError):
            run_datagen_pipeline(cfg)

    def test_empty_or_comment_only_file_is_a_noop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._patch_pipeline(monkeypatch)
        adv_file = tmp_path / "near_miss.txt"
        adv_file.write_text("# nothing here\n\n   \n")

        run_datagen_pipeline(self._cfg(tmp_path, str(adv_file)))

        # Only Stage 1's positive synthesis call (n=3) happened; no
        # adversarial text reached synthesize_positives.
        assert all(n == 3 for _, n in calls)

    def test_file_without_adversarial_flag_is_a_noop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_pipeline(monkeypatch)
        # A missing file would raise if ever opened; adversarial=False must
        # keep it unopened (datagen.py gates the read behind `if config.adversarial`).
        cfg = self._cfg(tmp_path, str(tmp_path / "does_not_exist.txt"), adversarial=False)

        result = run_datagen_pipeline(cfg)

        assert result.n_positive == 3


# ---------------------------------------------------------------------------
# Silence and near-silence negative windows
# ---------------------------------------------------------------------------


class TestSilenceWindows:
    def test_three_classes_at_window_length_and_at_level(self, tmp_path: Path) -> None:
        import numpy as np
        import soundfile as sf
        from ww_trainer.datagen import _write_window, silence_windows

        sr, w = 16000, 1.5
        wins = silence_windows(sr, w, 7, seed=0)
        assert len(wins) == 21 and all(len(x) == int(w * sr) for _, x in wins)
        by_name = {}
        for name, x in wins:
            by_name.setdefault(name, []).append(x)
        assert sorted(by_name) == ["noise_m40dbfs", "noise_m60dbfs", "zeros"]
        assert all((x == 0).all() for x in by_name["zeros"])
        for name, level in (("noise_m60dbfs", -60.0), ("noise_m40dbfs", -40.0)):
            for x in by_name[name]:
                rms_db = 20 * np.log10(np.sqrt(np.mean(x ** 2)))
                assert abs(rms_db - level) < 1.0, (name, rms_db)
        # The level survives the write: no peak normalisation on silence windows.
        dst = tmp_path / "noise_m60dbfs_0000.wav"
        _write_window(by_name["noise_m60dbfs"][0], dst, sr, normalize=False)
        back, _ = sf.read(str(dst), dtype="float32")
        rms_db = 20 * np.log10(np.sqrt(np.mean(back ** 2)))
        assert abs(rms_db + 60.0) < 1.0, rms_db
        zeros = tmp_path / "zeros_0000.wav"
        _write_window(by_name["zeros"][0], zeros, sr, normalize=False)
        back, _ = sf.read(str(zeros), dtype="float32")
        assert (back == 0).all()

    @patch("ww_trainer.datagen.download_hf_audio_dataset")
    def test_pipeline_writes_silence_negatives_and_counts_them(
        self, mock_download: MagicMock, tmp_path: Path
    ) -> None:
        import json

        def fake_download(dataset_id: str, output_dir: Path, **kwargs) -> list:
            output_dir.mkdir(parents=True, exist_ok=True)
            return [_make_wav(output_dir / f"{i:04d}.wav") for i in range(3)]

        mock_download.side_effect = fake_download
        cfg = DatagenConfig(
            wake_word="hey_mycroft", output_dir=tmp_path / "dataset", n_positive=3,
            max_negative=3, vad_trim=False, download_augmentation=False,
            silence_windows=4, seed=1,
        )
        result = run_datagen_pipeline(cfg)
        silence = sorted((result.negatives_dir / "silence").glob("*.wav"))
        assert len(silence) == 12
        entries = read_metadata_csv(result.train_csv) + read_metadata_csv(result.test_csv)
        assert sum(1 for p, label in entries if "/silence/" in p and label == 0) == 12
        cfg_json = json.loads((tmp_path / "dataset" / "datagen_config.json").read_text())
        assert cfg_json["n_silence_windows"] == {"zeros": 4, "noise_m60dbfs": 4, "noise_m40dbfs": 4}


class TestSilenceWindowsSurviveAugmentation:
    def test_augmented_silence_windows_stay_quiet(self, tmp_path: Path) -> None:
        import numpy as np
        from ww_trainer.dataset import AudioDataset
        from ww_trainer.datagen import _write_window, silence_windows

        sr = 16000
        samples = []
        for name, win in silence_windows(sr, 1.5, 2, seed=0):
            dst = tmp_path / "negatives" / "silence" / f"{name}_{len(samples)}.wav"
            _write_window(win, dst, sr, normalize=False)
            samples.append((str(dst), "0"))
        ds = AudioDataset(samples, sample_rate=sr, aug_prob=1.0)
        for i in range(len(samples)):
            for _ in range(5):
                wav = ds[i][0].numpy()
                rms_db = 20 * np.log10(np.sqrt(np.mean(wav ** 2)) + 1e-12)
                assert rms_db < -35.0, (samples[i][0], rms_db)


    def test_a_positive_in_a_user_folder_named_silence_is_augmented(self, tmp_path: Path) -> None:
        import numpy as np
        import soundfile as sf
        from ww_trainer.dataset import AudioDataset

        sr = 16000
        path = tmp_path / "my_clips" / "silence" / "hey_jarvis_017.wav"
        path.parent.mkdir(parents=True)
        sf.write(str(path), (0.01 * np.sin(np.linspace(0, 400, int(1.5 * sr)))).astype(np.float32), sr)
        ds = AudioDataset([(str(path), "1")], sample_rate=sr, aug_prob=1.0)
        peaks = [float(np.abs(ds[0][0].numpy()).max()) for _ in range(5)]
        assert all(p > 0.5 for p in peaks), peaks


class TestSilenceWindowProperties:
    @staticmethod
    def _run(tmp_path: Path, n: int):
        def fake_download(dataset_id: str, output_dir: Path, **kwargs) -> list:
            output_dir.mkdir(parents=True, exist_ok=True)
            return [_make_wav(output_dir / f"{i:04d}.wav") for i in range(3)]

        with patch("ww_trainer.datagen.download_hf_audio_dataset", side_effect=fake_download):
            cfg = DatagenConfig(
                wake_word="hey_mycroft", output_dir=tmp_path / "dataset", n_positive=3,
                max_negative=3, vad_trim=False, download_augmentation=False,
                silence_windows=n, seed=1,
            )
            return run_datagen_pipeline(cfg)

    def test_pipeline_writes_silence_at_its_real_level(self, tmp_path: Path) -> None:
        import numpy as np
        import soundfile as sf

        result = self._run(tmp_path, 3)
        levels = {}
        for f in (result.negatives_dir / "silence").glob("*.wav"):
            x, _ = sf.read(str(f), dtype="float32")
            levels.setdefault(f.name.rsplit("_", 1)[0], []).append(
                20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))
        assert all(v < -120 for v in levels["zeros"])
        assert all(abs(v + 60.0) < 1.0 for v in levels["noise_m60dbfs"])
        assert all(abs(v + 40.0) < 1.0 for v in levels["noise_m40dbfs"])

    def test_silence_windows_are_independent_split_groups(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, 30)
        entries = {"train": read_metadata_csv(result.train_csv), "test": read_metadata_csv(result.test_csv)}
        for side, rows in entries.items():
            assert any("/silence/" in p for p, _ in rows), side
