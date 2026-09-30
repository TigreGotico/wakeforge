"""Pretrained featurizers on every training path.

Every path — the heads, the training loop and its feature store, every loss,
SpecAugment, mining, evaluation, calibration, multi-stage, the infinite loop,
the quickstart, grid/genetic search, export and inference — runs on a stand-in
pretrained featurizer served by the ``fake_hub`` fixture (``conftest.py``):
waveform [B, N] -> features [B, N // 320, D] with D = 128, or 256 for the
``-wide`` repositories. No test touches the network.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
import soundfile as sf
import torch
from click.testing import CliRunner

from ww_trainer.augment import Mixup, SpectrogramAugment
from ww_trainer.calibration import calibrate_model
from ww_trainer.checkpoint import average_checkpoints, save_intermediate_checkpoint
from ww_trainer.cli import train
from ww_trainer.dataset import AudioDataset, collate_fn
from ww_trainer.evaluation import evaluate_model
from ww_trainer.factory import HEAD_REGISTRY, create_model
from ww_trainer.feats import OnnxFeatureExtractor
from ww_trainer.feature_store import FeatureStore
from ww_trainer.few_shot import FewShotDetector, generate_reference
from ww_trainer.inference import OnnxStreamingWakeWord, OnnxWakeWordInferencer
from ww_trainer.infinite_loop import StoppingGoal, infinite_training_loop
from ww_trainer.loss import LossManager
from ww_trainer.mining import mine_hard_negatives
from ww_trainer.model import BiGruClassifierHead, GruClassifierHead
from ww_trainer.multi_stage import run_multi_stage_training
from ww_trainer.pretrained import PRETRAINED_FEATURIZERS
from ww_trainer.quickstart import QuickstartConfig, _train_from_datagen_result
from ww_trainer.sweep import PRETRAINED_GENES, _build_search_space, run_genetic_search, \
    run_grid_search
from ww_trainer.tiers import get_tier
from ww_trainer.trainer import WakeWordTrainer

SR = 16000
TRAIN_KW = dict(epochs=1, batch_size=4, lr=1e-3, mine_fraction=0.0, pca_every=0,
                save_best=False, metrics_log="")


@pytest.fixture
def clips(tmp_path):
    """4 positives and 4 negatives of different lengths (0.5 to 1.2 s)."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(4):
        n = int(SR * (0.5 + 0.2 * i))
        t = np.arange(n) / SR
        pos, neg = tmp_path / f"pos_{i}.wav", tmp_path / f"neg_{i}.wav"
        sf.write(pos, (0.3 * np.sin(2 * np.pi * (440 + 40 * i) * t)).astype(np.float32), SR)
        sf.write(neg, (0.1 * rng.standard_normal(n)).astype(np.float32), SR)
        rows += [(str(pos), "1"), (str(neg), "0")]
    return rows


@pytest.fixture
def noise_dir(tmp_path):
    d = tmp_path / "noise"
    d.mkdir()
    sf.write(d / "n.wav", (0.05 * np.random.default_rng(1).standard_normal(SR)).astype(np.float32), SR)
    return str(d)


@pytest.fixture
def featurized(monkeypatch):
    """Count the waveforms each ONNX featurizer processes."""
    counter = {"clips": 0}
    original = OnnxFeatureExtractor.forward

    def counting(self, wavs):
        out = original(self, wavs)
        counter["clips"] += out.shape[0]
        return out

    monkeypatch.setattr(OnnxFeatureExtractor, "forward", counting)
    return counter


def _trainer(arch="gru", featurizer="wakehubert", **kwargs):
    kwargs.setdefault("losses_cfg", [{"name": "bce", "weight": 1.0}])
    return WakeWordTrainer(arch=arch, featurizer=None, featurizer_type=featurizer,
                           device="cpu", seed=0, **kwargs)


def _write_csv(path, rows):
    path.write_text("".join(f"{p},{l}\n" for p, l in rows))
    return path


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("featurizer,dim", [("wakehubert", 128), ("wakehubert-mel-tcn-wide", 256)])
@pytest.mark.parametrize("arch", sorted(HEAD_REGISTRY))
def test_every_head_trains_one_epoch_on_stored_features(fake_hub, clips, tmp_path, arch,
                                                        featurizer, dim):
    if arch == "phonmatch":  # conditioned on the keyword's phoneme IDs
        clips = [(path, label, [3, 5, 7]) for path, label in clips]
    trainer = _trainer(arch, featurizer)
    assert trainer.model.classifier.input_size == dim
    f1 = trainer.train(output_dir=tmp_path / "out", train_data=clips, test_data=clips,
                       **TRAIN_KW)
    assert 0.0 <= f1 <= 1.0
    store = FeatureStore.for_extractor(trainer.model.feature_extractor)
    feats = store.get(clips[0][0])
    assert feats is not None and feats.shape[1] == dim


def test_bigru_is_selectable_with_head_kwargs(fake_hub):
    model = create_model("bigru", None, featurizer_type="wakehubert", device="cpu",
                         hidden_dim=48, linear_dim=24, gru_n_layers=2, bidirectional=False)
    head = model.classifier
    assert isinstance(head, BiGruClassifierHead)
    assert head.gru.bidirectional and head.gru.num_layers == 2 and head.gru.hidden_size == 48
    assert model.embed(torch.zeros(2, SR)).shape == (2, 24)
    gru = create_model("gru", None, featurizer_type="wakehubert", device="cpu", linear_dim=40)
    assert gru.classifier.fc1.out_features == 40


@pytest.mark.parametrize("tier,head_type", [("wakehubert", GruClassifierHead),
                                            ("wakehubert-bigru", BiGruClassifierHead)])
def test_wakehubert_tiers(fake_hub, tier, head_type):
    tc = get_tier(tier)
    model = create_model(tc.head_arch, None, featurizer_type=tc.extractor_type, device="cpu",
                         hidden_dim=tc.hidden_dim, bidirectional=tc.bidirectional)
    assert type(model.classifier) is head_type
    assert model.classifier.gru.bidirectional == tc.bidirectional


@pytest.mark.parametrize("arch", ["gru", "bigru", "ffn", "ocsvm"])
def test_pooling_ignores_padded_frames(fake_hub, arch):
    model = create_model(arch, None, featurizer_type="wakehubert", device="cpu").eval()
    torch.manual_seed(0)
    short, long = 0.1 * torch.randn(12000), 0.1 * torch.randn(24000)
    padded = torch.zeros(2, 24000)
    padded[0, :12000], padded[1] = short, long
    with torch.no_grad():
        alone = model.embed([short])
        batch = model.embed(padded, lengths=torch.tensor([12000, 24000]))
        unmasked = model.embed(padded)
        logits = model(padded, lengths=torch.tensor([12000, 24000]))
    assert torch.allclose(batch[0], alone[0], atol=1e-5)
    assert not torch.allclose(unmasked[0], alone[0], atol=1e-4)
    assert torch.allclose(logits[0], model([short])[0], atol=1e-5)


def test_list_input_masks_by_its_own_lengths(fake_hub):
    model = create_model("gru", None, featurizer_type="wakehubert", device="cpu").eval()
    short, long = 0.1 * torch.randn(8000), 0.1 * torch.randn(20000)
    with torch.no_grad():
        assert torch.allclose(model([short, long])[0], model([short])[0], atol=1e-5)


# ---------------------------------------------------------------------------
# Feature store
# ---------------------------------------------------------------------------

def test_features_computed_once_per_clip_across_epochs(fake_hub, clips, tmp_path, featurized):
    trainer = _trainer()
    featurized["clips"] = 0
    trainer.train(output_dir=tmp_path / "out", train_data=clips, test_data=clips,
                  **{**TRAIN_KW, "epochs": 3, "mine_fraction": 0.5})
    # one pass per clip, plus the frame-rate probe
    assert featurized["clips"] <= len(clips) + 1


def test_augmented_variants_are_bounded_and_reused(fake_hub, clips, tmp_path, noise_dir,
                                                   monkeypatch):
    augmented = []
    original = AudioDataset.get_augmented

    def counting(self, wav):
        augmented.append(1)
        return original(self, wav)

    monkeypatch.setattr(AudioDataset, "get_augmented", counting)
    trainer = _trainer(bg_noise_folder=noise_dir)
    trainer.train(output_dir=tmp_path / "out", train_data=clips, test_data=clips,
                  **{**TRAIN_KW, "epochs": 4}, aug_prob=1.0, aug_warmup_epochs=1,
                  feature_cache_variants=2)
    store = FeatureStore.for_extractor(trainer.model.feature_extractor)
    variants = {k for k in store._memory if k[1] > 0}
    assert variants and {v for _, v in variants} <= {1, 2}
    assert len(augmented) <= 2 * len(clips)


def test_zero_variants_augments_on_the_fly(fake_hub, clips, tmp_path, noise_dir):
    trainer = _trainer(bg_noise_folder=noise_dir)
    trainer.train(output_dir=tmp_path / "out", train_data=clips, test_data=clips,
                  **{**TRAIN_KW, "epochs": 2}, aug_prob=1.0, aug_warmup_epochs=1)
    store = FeatureStore.for_extractor(trainer.model.feature_extractor)
    assert store._memory and all(v == 0 for _, v in store._memory)


def test_store_is_shared_across_heads_and_persists_to_disk(fake_hub, clips, tmp_path,
                                                           featurized):
    cache = tmp_path / "features"
    _trainer("gru").train(output_dir=tmp_path / "a", train_data=clips, test_data=clips,
                          feature_cache_dir=str(cache), **TRAIN_KW)
    featurized["clips"] = 0
    _trainer("bigru").train(output_dir=tmp_path / "b", train_data=clips, test_data=clips,
                            feature_cache_dir=str(cache), **TRAIN_KW)
    assert featurized["clips"] <= 1  # the frame-rate probe only
    assert len(list(cache.glob("*.npy"))) == len(clips)
    FeatureStore.close_all()
    featurized["clips"] = 0
    _trainer("ffn").train(output_dir=tmp_path / "c", train_data=clips, test_data=clips,
                          feature_cache_dir=str(cache), **TRAIN_KW)
    assert featurized["clips"] <= 1


def test_collate_pads_stored_features_with_frame_lengths():
    batch = collate_fn([(torch.ones(10, 128), 1, "a"), (torch.ones(25, 128), 0, "b")], "cpu")
    feats, labels, paths, kw = batch
    assert feats.shape == (2, 25, 128) and feats[0, 10:].abs().sum() == 0
    assert batch.lengths.tolist() == [10, 25]
    wavs = collate_fn([(torch.ones(100), 1, "a"), (torch.ones(300), 0, "b")], "cpu")
    assert wavs[0].shape == (2, 300) and wavs.lengths.tolist() == [100, 300]


def test_feature_mixup_tracks_lengths():
    feats, labels = torch.randn(4, 30, 128), torch.tensor([1.0, 0.0, 1.0, 0.0])
    before = torch.tensor([5, 30, 10, 20])
    mixed, mixed_labels, lengths = Mixup.mix_padded(feats, labels, before)
    assert mixed.shape == feats.shape and mixed_labels.shape == labels.shape
    assert (lengths >= before).all() and lengths.max() == 30


# ---------------------------------------------------------------------------
# SpecAugment and losses
# ---------------------------------------------------------------------------

def test_spec_augment_sized_to_features_and_kept_in_real_frames():
    spec = SpectrogramAugment.for_features(128, 50.0)
    assert (spec.max_time_width, spec.max_freq_width) == (5, 13)
    assert (SpectrogramAugment.for_features(40, 100.0).max_time_width,
            SpectrogramAugment.for_features(40, 100.0).max_freq_width) == (10, 4)
    spec = SpectrogramAugment(n_time_masks=4, max_time_width=5, n_freq_masks=0)
    feats = torch.ones(64, 50, 128)
    lengths = torch.full((64,), 12)
    out = spec(feats, lengths)
    assert torch.equal(out[:, 12:], feats[:, 12:])
    assert (out[:, :12] == 0).any()


@pytest.mark.parametrize("loss", [
    "bce", "triplet", "soft_triplet", "pair", "cn2pair", "lse", "contrastive", "angular",
    "focal", "label_smoothing_bce", "arcface", "center", "ntxent", "supcon", "proxy_nca",
    "multi_similarity", "rppl", "halo", "oc_softmax", "size_aware",
])
def test_every_loss_on_stored_features(fake_hub, loss):
    from ww_trainer.loop import with_embed_dim
    model = create_model("bigru", None, featurizer_type="wakehubert-mel-tcn-wide",
                         device="cpu", linear_dim=48)
    manager = LossManager(with_embed_dim([{"name": loss, "weight": 1.0}], model), device="cpu")
    manager.set_spec_augment(SpectrogramAugment.for_features(256, 50.0))
    model.train()
    feats = torch.randn(6, 40, 256)
    lengths = torch.tensor([40, 20, 35, 10, 40, 25])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    total, parts = manager.compute_loss(model, feats, labels, lengths=lengths)
    assert torch.isfinite(total) and loss in parts
    total.backward()


# ---------------------------------------------------------------------------
# Training paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("selector", [
    ["--featurizer-type", "wakehubert-mel-tcn-deep", "--arch", "bigru", "--linear-dim", "32"],
    ["--tier", "wakehubert-bigru"],
    ["--featurizer", "wakehubert-mel-tcn-wide", "--arch", "gru", "--feature-cache-variants", "2"],
])
def test_cli_train(fake_hub, clips, tmp_path, selector):
    csv_path = _write_csv(tmp_path / "train.csv", clips)
    out = tmp_path / "model"
    result = CliRunner().invoke(train, [
        "--wake-word", "hey_test", "--metadata", str(csv_path), "--test-metadata", str(csv_path),
        *selector, "--epochs", "1", "--batch-size", "4", "--device", "cpu",
        "--output-dir", str(out), "--feature-cache-dir", str(tmp_path / "fc"),
        "--mine-sample", "0", "--pca-every", "0", "--spec-augment",
    ])
    assert result.exit_code == 0, result.output + repr(result.exception)
    meta = json.loads((out / "hey_test_meta.json").read_text())
    assert meta["pretrained_featurizer"] in PRETRAINED_FEATURIZERS
    assert "featurizer_revision" in meta and "tier" in meta and "arch" in meta


def test_cli_meta_records_streaming_head(fake_hub, clips, tmp_path):
    csv_path = _write_csv(tmp_path / "train.csv", clips)
    out = tmp_path / "model"
    result = CliRunner().invoke(train, [
        "--wake-word", "hey_test", "--metadata", str(csv_path), "--test-metadata", str(csv_path),
        "--tier", "wakehubert", "--epochs", "1", "--batch-size", "4", "--device", "cpu",
        "--output-dir", str(out), "--no-feature-cache", "--mine-sample", "0", "--pca-every", "0",
    ])
    assert result.exit_code == 0, result.output + repr(result.exception)
    meta = json.loads((out / "hey_test_meta.json").read_text())
    assert meta["tier"] == "wakehubert" and meta["pretrained_featurizer"] == "wakehubert"
    assert meta["streaming_heads"] and meta["stream_window"] == 45  # median positive 0.9 s at 50 fps
    for name in meta["streaming_heads"]:
        assert (out / name).exists()


def test_quickstart_with_pretrained_featurizer(fake_hub, clips, tmp_path):
    class Result:
        train_csv = _write_csv(tmp_path / "train.csv", clips)
        test_csv = _write_csv(tmp_path / "test.csv", clips)

    cfg = QuickstartConfig(wake_word="hey test", output_dir=tmp_path, tier="small",
                           featurizer="wakehubert-mel-tcn-wide", epochs=1, batch_size=4,
                           device="cpu", feature_cache_variants=1)
    result = _train_from_datagen_result(cfg, Result())
    assert 0.0 <= result.metrics["f1"] <= 1.0
    head = onnx.load(str(result.best_onnx_path))
    meta = {p.key: p.value for p in head.metadata_props}
    assert meta["pretrained_featurizer"] == "wakehubert-mel-tcn-wide"


def test_search_space_has_pretrained_gene():
    space = _build_search_space(full=True)
    assert set(PRETRAINED_GENES) <= set(space["featurizer_type"])
    assert "bigru" in space["arch"]
    assert "wakehubert-mel-tcn-wide" in PRETRAINED_GENES
    assert not any(g.startswith("wakexeus") or g.endswith("-int8") for g in PRETRAINED_GENES)


def test_grid_search_heads_share_features(fake_hub, clips, tmp_path, featurized):
    csv_path = _write_csv(tmp_path / "data.csv", clips * 2)
    result = run_grid_search(
        str(csv_path), output_dir=str(tmp_path / "grid"), device="cpu", epochs_per_trial=1,
        search_space={"featurizer_type": ["wakehubert"], "arch": ["gru", "bigru", "ffn"],
                      "loss": ["arcface"], "batch_size": [4]},
    )
    assert len(result["all_results"]) == 3
    assert len(list((tmp_path / "grid").rglob("final_model.pt"))) == 3  # no trial failed
    # 8 distinct clips, each featurized once for all three heads, plus one probe per trial
    assert featurized["clips"] <= len(clips) + 3


def test_genetic_search_with_pretrained_gene(fake_hub, clips, tmp_path):
    csv_path = _write_csv(tmp_path / "data.csv", clips * 2)
    result = run_genetic_search(
        str(csv_path), population_size=2, generations=1, output_dir=str(tmp_path / "ga"),
        device="cpu", epochs_per_trial=1, feature_cache_dir=str(tmp_path / "fc"),
        search_space={"featurizer_type": ["wakehubert", "wakehubert-mel-tcn-wide"],
                      "arch": ["gru", "bigru"], "batch_size": [4]},
    )
    assert result["best_config"]["featurizer_type"] in ("wakehubert", "wakehubert-mel-tcn-wide")
    trained = list((tmp_path / "ga").rglob("final_model.pt"))
    assert trained and len(trained) == len(result["all_results"])  # no candidate failed
    assert list((tmp_path / "fc").glob("*.npy"))


def test_multi_stage_reuses_features(fake_hub, clips, tmp_path, featurized):
    trainer = _trainer()
    featurized["clips"] = 0
    run_multi_stage_training(trainer, [{"epochs": 1, "lr": 1e-3}, {"epochs": 1, "lr": 1e-4}],
                             clips, clips, tmp_path / "ms", batch_size=4, mine_fraction=0.0,
                             save_best=True, metrics_log="", pca_every=0)
    assert (tmp_path / "ms" / "averaged_model.pt").exists()
    assert featurized["clips"] <= len(clips) + 2


def test_infinite_loop_runs_on_pretrained_featurizer(fake_hub, clips, tmp_path, featurized):
    trainer = _trainer(losses_cfg=[{"name": "center", "weight": 1.0}])
    wakes = [c for c in clips if c[1] == "1"]
    pool = [c for c in clips if c[1] == "0"]
    featurized["clips"] = 0
    best = infinite_training_loop(trainer, tmp_path / "inf", wakes, pool, clips,
                                  goal=StoppingGoal(max_epochs=3, min_epochs=1), batch_size=4,
                                  aug_prob=0.0, eval_every=1)
    assert 0.0 <= best <= 1.0
    assert (tmp_path / "inf" / "final.onnx").exists()
    assert featurized["clips"] <= len(clips) + 1


def test_mining_evaluation_calibration_use_the_store(fake_hub, clips, tmp_path, featurized):
    model = create_model("gru", None, featurizer_type="wakehubert", device="cpu")
    store = FeatureStore.for_extractor(model.feature_extractor)
    negatives = [c for c in clips if c[1] == "0"]
    featurized["clips"] = 0
    for _ in range(2):
        mine_hard_negatives(model, negatives, "cpu", dataset_fraction=1.0,
                            wake_cache=[c for c in clips if c[1] == "1"], feature_store=store)
        evaluate_model(model, clips, "cpu", batch_size=4, feature_store=store)
        calibrate_model(model, clips, str(tmp_path / "cal"), feature_store=store)
    assert featurized["clips"] == len(clips)


def test_few_shot_reference_with_pretrained_featurizer(fake_hub, clips, tmp_path):
    model = create_model("gru", None, featurizer_type="wakehubert", device="cpu").eval()
    positives = [p for p, l in clips if l == "1"]
    generate_reference(model, positives, tmp_path / "ref.json")
    detector = FewShotDetector(tmp_path / "ref.json", model=model, threshold=0.5)
    audio, _ = sf.read(positives[0], dtype="float32")
    assert detector.detect(audio)["confidence"] > 0.99


def test_checkpoint_averaging_with_pretrained_featurizer(fake_hub, tmp_path):
    paths = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        model = create_model("bigru", None, featurizer_type="wakehubert", device="cpu")
        paths.append(tmp_path / f"m{seed}.pt")
        model.save_checkpoint(str(paths[-1]))
    avg = average_checkpoints(paths)
    fresh = create_model("bigru", None, featurizer_type="wakehubert", device="cpu")
    fresh.load_state_dict(avg)
    assert fresh(torch.zeros(1, SR)).shape == (1,)


# ---------------------------------------------------------------------------
# Export and inference
# ---------------------------------------------------------------------------

@pytest.fixture
def snapshot_hub(stand_in_repos, monkeypatch, tmp_path):
    """Serve the stand-ins from a Hugging Face cache layout (``snapshots/<commit>/``)."""
    import ww_trainer.pretrained as pretrained
    commit = "0123456789abcdef0123456789abcdef01234567"

    def fake_download(repo_id, filename, revision=None, **kwargs):
        snap = tmp_path / "hub" / repo_id.replace("/", "--") / "snapshots" / (revision or commit)
        snap.mkdir(parents=True, exist_ok=True)
        if not (snap / filename).exists():
            (snap / filename).symlink_to(stand_in_repos[repo_id] / filename)
        return str(snap / filename)

    monkeypatch.setattr(pretrained, "hf_hub_download", fake_download)
    return commit


def _meta(path):
    return {p.key: p.value for p in onnx.load(str(path)).metadata_props}


def test_export_records_featurizer_and_inference_rebuilds_it(snapshot_hub, clips, tmp_path):
    trainer = _trainer("gru", "wakehubert-mel-tcn-deep")
    out = tmp_path / "out"
    trainer.train(output_dir=out, train_data=clips, test_data=clips,
                  **{**TRAIN_KW, "save_best": True})
    for stem in ("best_f1", "final_model"):
        meta = _meta(out / f"{stem}.onnx")
        assert meta["pretrained_featurizer"] == "wakehubert-mel-tcn-deep"
        assert meta["featurizer_revision"] == snapshot_hub
        stream_meta = _meta(out / f"{stem}_streaming.onnx")
        assert stream_meta["pretrained_featurizer"] == "wakehubert-mel-tcn-deep"
        assert stream_meta["stream_window"] == str(trainer.stream_window)
    assert not (out / "final_model_featurizer.onnx").exists()

    audio, _ = sf.read(clips[0][0], dtype="float32")
    ext = trainer.model.feature_extractor
    explicit = OnnxWakeWordInferencer(ext.model_path, str(out / "final_model.onnx"), device="cpu")
    rebuilt = OnnxWakeWordInferencer(None, str(out / "final_model.onnx"), device="cpu")
    assert rebuilt.infer(audio) == pytest.approx(explicit.infer(audio), abs=1e-6)
    trainer.model.eval()
    with torch.no_grad():
        ref = float(torch.sigmoid(trainer.model([torch.from_numpy(audio)])))
    assert rebuilt.infer(audio) == pytest.approx(ref, abs=1e-4)

    streamer = OnnxStreamingWakeWord.from_head(str(out / "final_model_streaming.onnx"))
    assert (streamer.window, streamer.hidden_dim, streamer.hop) == (
        trainer.stream_window, 128, 320)
    assert 0.0 <= streamer.push(audio[:1600]) <= 1.0


@pytest.mark.parametrize("featurizer,arch", [("wakehubert-mel-bigru", "gru"),
                                             ("wakehubert", "bigru")])
def test_no_streaming_head_for_unstreamable_models(fake_hub, tmp_path, featurizer, arch):
    model = create_model(arch, None, featurizer_type=featurizer, device="cpu")
    assert not model.streamable
    save_intermediate_checkpoint(model, "hey", arch, 0, {}, None, tmp_path / "m.pt",
                                 stream_window=40)
    assert (tmp_path / "m.onnx").exists()
    assert not (tmp_path / "m_streaming.onnx").exists()


def test_export_does_not_need_onnxscript(fake_hub, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxscript", None)
    model = create_model("gru", None, featurizer_type="wakehubert", device="cpu")
    save_intermediate_checkpoint(model, "hey", "gru", 0, {}, None, tmp_path / "m.pt",
                                 stream_window=40)
    assert (tmp_path / "m.onnx").exists() and (tmp_path / "m_streaming.onnx").exists()


def test_inferencer_without_featurizer_refuses_plain_head(tmp_path):
    head = GruClassifierHead(hidden_dim=8, input_size=16, device="cpu")
    head.export_to_onnx(str(tmp_path / "h.onnx"))
    with pytest.raises(ValueError, match="does not record a pretrained featurizer"):
        OnnxWakeWordInferencer(None, str(tmp_path / "h.onnx"))


# ---------------------------------------------------------------------------
# When the store is used
# ---------------------------------------------------------------------------

def test_progressive_unfreezing_trains_the_extractor(clips, tmp_path):
    trainer = WakeWordTrainer(arch="gru", featurizer=None, featurizer_type="sincnet",
                              device="cpu", seed=0, n_filters=16, freeze_extractor=True,
                              unfreeze_at_epoch=1, losses_cfg=[{"name": "bce", "weight": 1.0}])
    before = {k: v.clone() for k, v in trainer.model.feature_extractor.state_dict().items()}
    trainer.train(output_dir=tmp_path / "out", train_data=clips, test_data=clips,
                  **{**TRAIN_KW, "epochs": 3, "lr": 1e-2})
    after = trainer.model.feature_extractor.state_dict()
    moved = max(float((after[k] - v).abs().max()) for k, v in before.items()
                if v.is_floating_point() and v.numel())
    assert moved > 0.0


def test_frozen_trainable_extractor_gets_no_store():
    from ww_trainer.feature_store import store_for
    model = create_model("gru", None, featurizer_type="sincnet", device="cpu", n_filters=16)
    for p in model.feature_extractor.parameters():
        p.requires_grad = False
    assert store_for(model.feature_extractor, requested=True) is None


def test_builtin_extractors_keep_waveform_augmentation_by_default(clips, tmp_path, monkeypatch):
    calls = []
    original = AudioDataset.get_augmented

    def counting(self, wav):
        calls.append(1)
        return original(self, wav)

    monkeypatch.setattr(AudioDataset, "get_augmented", counting)
    trainer = WakeWordTrainer(arch="gru", featurizer=None, featurizer_type="mfcc", device="cpu",
                              seed=0, losses_cfg=[{"name": "bce", "weight": 1.0},
                                                  {"name": "rppl", "weight": 1.0}])
    trainer.train(output_dir=tmp_path / "out", train_data=clips, test_data=clips, **TRAIN_KW)
    assert calls  # RPPL's consistency view is a waveform-augmented copy
    assert not FeatureStore._shared


def test_builtin_extractor_store_on_request():
    from ww_trainer.feature_store import store_for
    model = create_model("gru", None, featurizer_type="mfcc", device="cpu")
    assert store_for(model.feature_extractor) is None
    assert store_for(model.feature_extractor, requested=True) is not None


FEATS_BYTES = 50 * 128 * 4


def _store(fake_hub_name):
    return FeatureStore.for_extractor(OnnxFeatureExtractor.from_pretrained(fake_hub_name,
                                                                          device="cpu"))


def test_one_memory_budget_across_stores(fake_hub, monkeypatch):
    monkeypatch.setattr(FeatureStore, "max_bytes", 3 * FEATS_BYTES)
    a, b = _store("wakehubert"), _store("wakehubert-int8")
    a.put("a0", 0, torch.zeros(50, 128))
    a.put("a1", 0, torch.zeros(50, 128))
    b.put("b0", 0, torch.zeros(50, 128))
    assert (len(a), len(b)) == (2, 1)
    assert FeatureStore.used_bytes() == 3 * FEATS_BYTES
    b.put("b1", 0, torch.zeros(50, 128))  # over budget: the store used least recently yields
    assert len(a) + len(b) <= 3 and len(b) == 2
    assert FeatureStore.used_bytes() <= 3 * FEATS_BYTES


def test_budget_is_released_by_stores_of_earlier_runs(fake_hub, monkeypatch):
    monkeypatch.setattr(FeatureStore, "max_bytes", 2 * FEATS_BYTES)
    earlier = _store("wakehubert")
    earlier.put("x0", 0, torch.zeros(50, 128))
    earlier.put("x1", 0, torch.zeros(50, 128))
    kept_alive = earlier  # a run that ended but whose store is still referenced
    later = _store("wakehubert-int8")
    later.put("y0", 0, torch.zeros(50, 128))
    later.put("y1", 0, torch.zeros(50, 128))
    assert len(later) == 2 and len(kept_alive) == 0

    FeatureStore.close_all()
    del earlier, kept_alive, later
    fresh = _store("wakehubert-mel-tcn-deep")
    fresh.put("z0", 0, torch.zeros(50, 128))
    fresh.put("z1", 0, torch.zeros(50, 128))
    assert len(fresh) == 2 and FeatureStore.used_bytes() == 2 * FEATS_BYTES


def test_closed_store_frees_budget_and_refills(fake_hub, monkeypatch):
    monkeypatch.setattr(FeatureStore, "max_bytes", FEATS_BYTES)
    store = _store("wakehubert")
    store.put("a", 0, torch.zeros(50, 128))
    store.close()
    assert len(store) == 0 and FeatureStore.used_bytes() == 0
    store.put("b", 0, torch.zeros(50, 128))
    assert store.get("b") is not None


def test_clip_frames_survives_unreadable_header(tmp_path, monkeypatch):
    import soundfile
    from ww_trainer.loop import clip_frames
    path = tmp_path / "a.wav"
    sf.write(path, np.zeros(SR, dtype=np.float32), SR)

    def broken(_):
        raise RuntimeError("format not understood")

    monkeypatch.setattr(soundfile, "info", broken)
    assert clip_frames([(str(path), "1"), (str(tmp_path / "missing.wav"), "1")], 50.0) == 50
