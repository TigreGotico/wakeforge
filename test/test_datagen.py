"""Tests for ww_trainer.datagen — synthetic data pipeline."""

import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torchaudio

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

class TestDownloadDecodesWithSoundfile:
    def test_encoded_bytes_become_16k_wavs(self, tmp_path, monkeypatch) -> None:
        import io
        import sys
        import numpy as np
        import soundfile as sf
        import datasets as hf_datasets
        from ww_trainer import datagen

        sr_in = 22050
        tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(sr_in) / sr_in)).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, tone, sr_in, format="WAV")
        row = {"audio": {"bytes": buf.getvalue(), "path": "clip.wav"}}

        class FakeDataset:
            features = {"audio": hf_datasets.Audio()}

            def cast_column(self, col, feature):
                assert col == "audio" and feature.decode is False
                return self

            def __iter__(self):
                return iter([row, row])

        monkeypatch.setitem(sys.modules, "torchcodec", None)
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

        src = tmp_path / "src"; src.mkdir()
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
        monkeypatch.setattr(datagen, "load_dataset", None, raising=False)

        files = datagen.download_hf_audio_dataset("org/folder", tmp_path / "out", max_samples=1, sr=16000)

        assert [f.name for f in files] == ["000000.wav"]
        audio, sr = sf.read(files[0], dtype="float32")
        assert sr == 16000 and abs(len(audio) - 16000) <= 2
