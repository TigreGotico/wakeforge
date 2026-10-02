"""Tests for ww_trainer.datagen — synthetic data pipeline."""

import csv
import random
import sys
import tempfile
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
    def test_general_and_speech_sources(self) -> None:
        assert NEGATIVE_DATASETS["general"] == ["agkphysics/AudioSet"]
        assert NEGATIVE_DATASETS["speech"] == ["TigreGotico/not-wake-words-speech-en"]

    def test_every_pipeline_dataset_has_a_permissive_licence(self) -> None:
        """A dataset enters the pipeline only with a permissive licence on record."""
        from ww_trainer.datagen import DATASET_LICENSES, KNOWN_POSITIVE_DATASETS, PERMISSIVE_LICENSES
        used = {d for ds in NEGATIVE_DATASETS.values() for d in ds} | set(KNOWN_POSITIVE_DATASETS.values())
        missing = used - set(DATASET_LICENSES)
        assert not missing, missing
        non_permissive = {d: lic for d, lic in DATASET_LICENSES.items() if lic not in PERMISSIVE_LICENSES}
        assert not non_permissive, non_permissive

    def test_excluded_sources_stay_out(self) -> None:
        """NC and unlicensed sources ruled out of the training set never come back."""
        used = {d for ds in NEGATIVE_DATASETS.values() for d in ds}
        for banned in ("TigreGotico/ESC-50", "TigreGotico/NAR", "hf-internal-testing/librispeech_asr_demo",
                       "TigreGotico/ambient_noises", "TigreGotico/building_106_kitchen_3secs",
                       "TigreGotico/public_domain_sounds_3secs", "TigreGotico/FMA_3secs",
                       "davidscripka/MIT_environmental_impulse_responses"):
            assert banned not in used

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
        monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                            lambda repo, name, repo_type=None: str(src / name))
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
                                    n_positive=10, download_augmentation=False, adversarial=False)
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

    def hub_download(repo, name, repo_type=None):
        path = src / repo.replace("/", "__") / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            _make_wav(path, duration=0.3)
        return str(path)

    monkeypatch.setitem(sys.modules, "datasets", _fake_datasets_module())
    monkeypatch.setattr(datagen, "_list_repo_audio_files", list_files)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub_download)


def _pipeline_config(out: Path, seed: int = 42) -> DatagenConfig:
    return DatagenConfig(
        wake_word="hey_mycroft", output_dir=out, n_positive=6, max_negative=6,
        vad_trim=False, download_augmentation=False, adversarial=False, seed=seed,
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
