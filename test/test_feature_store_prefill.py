"""Features are stored in float16 and can be prefilled by worker processes.

A wakehubert dataset with several augmented variants per clip did not fit the
in-memory budget in float32, so variants were evicted and recomputed every
epoch, and each variant was computed lazily in the training process one clip at
a time. Stores now hold float16, the budget is configurable, and ``prefill``
computes every clip's variants in parallel before training.
"""
import numpy as np
import soundfile as sf
import torch

from conftest import export_stand_in_featurizer
from ww_trainer.feats import OnnxFeatureExtractor
from ww_trainer.feature_store import FeatureStore, prefill


def _clips(tmp_path, n):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        p = tmp_path / f"clip{i}.wav"
        sf.write(p, (rng.standard_normal(16000) * 0.1).astype(np.float32), 16000)
        out.append((str(p), "1" if i % 2 else "0"))
    return out


def _store(tmp_path):
    onnx = tmp_path / "feat.onnx"
    export_stand_in_featurizer(onnx, seed=0)
    ext = OnnxFeatureExtractor(str(onnx), sample_rate=16000, device="cpu", hop_samples=320)
    return FeatureStore(ext, str(tmp_path / "cache"))


def test_features_are_held_and_written_as_float16_and_read_back_as_float32(tmp_path):
    store = _store(tmp_path)
    a = _clips(tmp_path, 1)[0][0]
    feats = torch.randn(50, 128)
    store.put(a, 0, feats)
    assert store._memory[(a, 0)].dtype == torch.float16
    assert np.load(store._disk_path(a, 0)).dtype == np.float16
    got = store.get(a, 0)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, feats, rtol=1e-2, atol=1e-2)
    store.close()
    from_disk = store.get(a, 0)
    assert from_disk.dtype == torch.float32
    torch.testing.assert_close(from_disk, feats, rtol=1e-2, atol=1e-2)


def test_the_memory_budget_counts_float16_bytes(tmp_path):
    store = _store(tmp_path)
    store.put(_clips(tmp_path, 1)[0][0], 0, torch.randn(50, 128))
    assert store.nbytes == 50 * 128 * 2


def test_prefill_writes_every_variant_of_every_clip(tmp_path):
    store = _store(tmp_path)
    clips = _clips(tmp_path, 6)
    n = prefill(store, clips, variants=2, workers=2, dataset_kwargs={"aug_prob": 1.0, "snr_min": 0.0, "snr_max": 20.0},
                chunk=2)
    assert n == 6 * 3
    for path, _ in clips:
        for v in range(3):
            assert store._disk_path(path, v).exists(), (path, v)
    # variant 0 is the clip as recorded: exactly what the training process computes un-augmented
    path = clips[0][0]
    wav = torch.from_numpy(sf.read(path, dtype="float32")[0])
    torch.testing.assert_close(store.get(path, 0), store.featurize(wav), rtol=1e-2, atol=1e-2)


def test_prefill_skips_what_is_already_on_disk(tmp_path):
    store = _store(tmp_path)
    clips = _clips(tmp_path, 3)
    assert prefill(store, clips, variants=1, workers=1, dataset_kwargs={"aug_prob": 1.0}) == 6
    assert prefill(store, clips, variants=1, workers=1, dataset_kwargs={"aug_prob": 1.0}) == 0


def test_the_cli_sets_the_memory_budget_and_passes_the_workers(monkeypatch, tmp_path):
    from click.testing import CliRunner
    from ww_trainer import cli
    seen = {}

    class Stop(Exception):
        pass

    class FakeTrainer:
        def __init__(self, *a, **k):
            seen["max_bytes"] = FeatureStore.max_bytes
            self.model = type("M", (), {"feature_extractor": None})()

        def train(self, **kwargs):
            seen["train"] = kwargs
            raise Stop

    monkeypatch.setattr(cli, "WakeWordTrainer", FakeTrainer)
    monkeypatch.setattr("ww_trainer.feature_store.store_for", lambda *a, **k: None)
    meta = tmp_path / "m.csv"
    meta.write_text("".join(f"{p},{y}\n" for p, y in _clips(tmp_path, 4)))
    old = FeatureStore.max_bytes
    try:
        result = CliRunner().invoke(cli.train, ["--wake-word", "x", "--metadata", str(meta), "--featurizer-type", "mfcc",
                                                "--feature-cache-max-gb", "1.5", "--feature-cache-workers", "3",
                                                "--no-feature-cache"])
    finally:
        FeatureStore.max_bytes = old
    assert isinstance(result.exception, Stop), result.output
    assert seen["max_bytes"] == int(1.5 * 1024 ** 3)
    assert seen["train"]["feature_cache_workers"] == 3


def test_an_in_process_prefill_leaves_the_memory_budget_alone(tmp_path):
    """With one worker the prefill runs in the training process; it must not shrink that process's budget."""
    store = _store(tmp_path)
    before = FeatureStore.max_bytes
    prefill(store, _clips(tmp_path, 2), variants=1, workers=1)
    assert FeatureStore.max_bytes == before
    a = _clips(tmp_path, 1)[0][0]
    store.put(a, 0, torch.randn(50, 128))
    assert (a, 0) in store._memory


def test_prefill_draws_voice_conversion_and_wake_word_over_speech_like_training(tmp_path, monkeypatch):
    """A variant is filled by a training draw that may use wake-word-over-speech; prefill must draw the same way."""
    from ww_trainer.dataset import AudioDataset
    store = _store(tmp_path)
    clips = [(p, "1") for p, _ in _clips(tmp_path, 2)]
    speech = tmp_path / "speech"; speech.mkdir()
    sf.write(speech / "s.wav", (np.random.default_rng(1).standard_normal(16000) * 0.1).astype(np.float32), 16000)
    calls = []
    real = AudioDataset._waveform

    def spy(self, idx, will_augment, use_vc, use_wow):
        calls.append((will_augment, use_vc, use_wow))
        return real(self, idx, will_augment, use_vc, use_wow)

    monkeypatch.setattr(AudioDataset, "_waveform", spy)
    prefill(store, clips, variants=2, workers=1,
            dataset_kwargs={"aug_prob": 0.0, "wake_word_over_speech_folder": str(speech), "wow_prob": 1.0})
    variants = [c for c in calls if any(c)]
    assert variants and all(c == (False, False, True) for c in variants), calls


def test_an_in_process_prefill_keeps_the_training_process_threads(tmp_path):
    store = _store(tmp_path)
    before = torch.get_num_threads()
    torch.set_num_threads(3)
    try:
        prefill(store, _clips(tmp_path, 2), variants=1, workers=1, dataset_kwargs={"aug_prob": 1.0})
        assert torch.get_num_threads() == 3
    finally:
        torch.set_num_threads(before)


def test_a_dataset_that_never_augments_gets_no_augmented_variants(tmp_path):
    store = _store(tmp_path)
    clips = _clips(tmp_path, 2)
    assert prefill(store, clips, variants=2, workers=1, dataset_kwargs={"aug_prob": 0.0}) == 2
    assert not store._disk_path(clips[0][0], 1).exists()
