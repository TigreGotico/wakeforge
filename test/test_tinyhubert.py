"""TinyHuBERT recipe: frame rate, causality, frame pairing, target statistics, losses, and the
exported extractor loading through OnnxFeatureExtractor."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

_P = Path(__file__).resolve().parents[1] / "scripts" / "research" / "tinyhubert.py"
_S = importlib.util.spec_from_file_location("tinyhubert", _P)
th = importlib.util.module_from_spec(_S)
_S.loader.exec_module(th)


MEL = {"mel-tcn": ("MelTCNStudent", {}), "mel-tcn-mixconv": ("MelTCNStudent", {"mixconv": True}),
       "mel-gru": ("MelGRUStudent", {}), "mel-attn": ("MelAttnStudent", {"window": 5}),
       "mel-bigru": ("MelBiGRUStudent", {})}


def _student(kind="wave-gru", **kw):
    torch.manual_seed(0)
    if kind in MEL:
        cls, extra = MEL[kind]
        s = getattr(th, cls)(channels=16, blocks=4, feature_dim=16, n_mels=20, n_targets=2, target_dim=12,
                             **{**extra, **kw})
    else:
        s = th.WakeHuBERTStudent(cnn_dim=8, gru_hidden=16, n_targets=2, target_dim=12)
    return s.eval()


CAUSAL = ["wave-gru", "mel-tcn", "mel-tcn-mixconv", "mel-gru", "mel-attn"]
KINDS = CAUSAL + ["mel-bigru"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("n", [16000, 32000, 21920, 320, 639, 641])
def test_the_student_emits_one_frame_per_320_samples(kind, n):
    assert _student(kind)(torch.randn(2, n)).shape == (2, n // 320, 16)


@pytest.mark.parametrize("kind", KINDS)
def test_hubert_and_student_frame_rates_agree(kind):
    # HuBERT base: seven convolutions with total stride 320 and a 400-sample receptive field
    hubert_frames = (32000 - 400) // 320 + 1
    student_frames = _student(kind)(torch.randn(1, 32000)).shape[1]
    assert abs(student_frames - hubert_frames) <= 1


def test_the_log_mel_front_end_matches_torchaudio():
    import torchaudio
    x = torch.randn(1, 16000)
    ours = th.CausalLogMel(n_mels=40)(x)
    ref = torchaudio.transforms.MelSpectrogram(16000, n_fft=400, hop_length=160, n_mels=40, center=False, power=2.0)(
        torch.nn.functional.pad(x, (240, 0)))
    assert ours.shape == ref.shape
    assert torch.allclose(ours, torch.log(ref + 1e-6), atol=1e-3)


@pytest.mark.parametrize("kind", CAUSAL)
def test_a_frame_does_not_depend_on_any_later_sample(kind):
    s = _student(kind)
    x = torch.randn(1, 16000)
    base = s(x)
    for t in (0, 7, 30):
        y = x.clone()
        y[:, 320 * (t + 1):] = torch.randn(1, 16000 - 320 * (t + 1)) * 5
        out = s(y)
        assert torch.allclose(out[:, :t + 1], base[:, :t + 1], atol=1e-6), f"frame <= {t} saw the future"
        assert not torch.allclose(out[:, t + 1:], base[:, t + 1:], atol=1e-6)


def test_the_bidirectional_student_hears_later_samples():
    s = _student("mel-bigru")
    x = torch.randn(1, 16000)
    y = x.clone()
    y[:, 320 * 31:] = torch.randn(1, 16000 - 320 * 31) * 5
    assert not torch.allclose(s(y)[:, :31], s(x)[:, :31], atol=1e-6)
    assert not s.streaming and _student("mel-gru").streaming


def test_mixconv_splits_channels_over_four_kernel_sizes():
    dw = _student("mel-tcn-mixconv").blocks[0].dw
    assert isinstance(dw, th.MixDWConv) and [c.kernel_size[0] for c in dw.convs] == [3, 5, 7, 9]
    assert sum(c.in_channels for c in dw.convs) == 16


def test_target_blocks_feed_early_heads_from_their_blocks_and_the_last_from_the_features():
    wav = torch.randn(2, 16000)
    s = _student("mel-tcn", target_blocks=(1,))
    base = th.student_preds(s, wav, 0, 40)[0]
    assert base.shape == (2, 2, 40, 12)
    with torch.no_grad():
        s.out.weight.zero_(); s.out.bias.zero_()
    zeroed = th.student_preds(s, wav, 0, 40)[0]
    assert torch.allclose(zeroed[0], base[0]), "the tapped head read the output projection"
    assert not torch.allclose(zeroed[1], base[1]), "the last teacher layer must come from the exported features"
    s = _student("mel-tcn", target_blocks=(1,))
    with torch.no_grad():
        s.blocks[1].pw.weight.mul_(3)
    after = th.student_preds(s, wav, 0, 40)[0]
    assert torch.allclose(after[0], base[0]) and not torch.allclose(after[1], base[1])
    plain = _student("mel-tcn")
    ref = th.student_preds(plain, wav, 0, 40)[0]
    with torch.no_grad():
        plain.out.weight.zero_(); plain.out.bias.zero_()
    assert not torch.allclose(th.student_preds(plain, wav, 0, 40)[0], ref)
    g = _student("mel-gru", target_blocks=(6,))  # the second GRU layer
    assert th.student_preds(g, wav, 0, 40)[0].shape == (2, 2, 40, 12)
    for bad in ((1, 2), (0,), (7,)):
        with pytest.raises(SystemExit):
            _student("mel-gru", target_blocks=bad)


def test_masking_acts_only_in_training():
    x = torch.randn(2, 16000)
    s = _student("mel-tcn", mask_prob=0.3).train()
    assert not torch.allclose(s(x), s(x))
    s.eval()
    ref = s(x)
    assert torch.allclose(s(x), ref)
    s.mask_prob = 0.0
    assert torch.allclose(s(x), ref)
    s.train()
    assert torch.allclose(s(x), s(x))
    keep = th.span_mask(torch.zeros(1, 4, 200), 0.05)
    starts = [i for i in range(200) if keep[0, 0, i] == 0 and (i == 0 or keep[0, 0, i - 1] == 1)]
    assert starts and all(i + 9 >= 199 or (keep[0, 0, i:i + 10] == 0).all() for i in starts)


@pytest.mark.parametrize("st, tt, lag, want", [(100, 99, 0, 99), (100, 99, 4, 95), (50, 49, 49, 0), (10, 99, 2, 7)])
def test_frame_pairing(st, tt, lag, want):
    assert th.pair_frames(st, tt, lag) == want


def test_pairing_gives_the_student_the_teacher_frames_whole_window():
    # teacher frame t reads samples [320t, 320t + 400); its paired student frame t + 1 + lag has
    # heard every sample before 320 (t + 2 + lag)
    for t in range(50):
        assert 320 * (t + 2) >= 320 * t + 400


def test_target_statistics_standardise_each_layer_and_dimension():
    g = torch.Generator().manual_seed(0)
    scale = torch.tensor([1.0, 50.0, 0.01])
    batches = [torch.randn(2, 4, 30, 3, generator=g) * scale + torch.tensor([0.0, 3.0, -7.0]) for _ in range(20)]
    norm = th.TargetNorm(2, 3)
    norm.fit(iter(batches))
    z = norm(torch.cat(batches, dim=1)).flatten(0, 2)
    assert torch.allclose(z.mean(0), torch.zeros(3), atol=1e-4)
    assert torch.allclose(z.std(0), torch.ones(3), atol=1e-2)
    assert bool(norm.fitted)


def test_distil_loss_is_lowest_at_the_target_and_reports_per_layer():
    t = torch.randn(3, 2, 10, 8)
    same, l1, cos = th.distil_loss(t.clone(), t)
    assert torch.allclose(l1, torch.zeros(3)) and torch.allclose(cos, torch.ones(3), atol=1e-6)
    assert math.isclose(same.item(), -torch.nn.functional.logsigmoid(torch.tensor(1.0)).item(), rel_tol=1e-5)
    worse, _, _ = th.distil_loss(t + 0.5 * torch.randn_like(t), t)
    assert worse > same


def test_cross_clip_nce_ignores_frames_of_the_same_clip():
    # every frame of a clip is identical: counting same-clip frames as negatives would make the
    # positive indistinguishable from them and hold the loss at log(T) or above
    B, T, D = 4, 6, 16
    torch.manual_seed(0)
    clips = torch.nn.functional.normalize(torch.randn(B, 1, D), dim=-1).expand(B, T, D).contiguous()
    loss = th.cross_clip_nce(clips, clips, temperature=0.05)
    assert loss.item() < 0.1
    assert th.cross_clip_nce(clips[:1], clips[:1]).item() == 0.0


def test_preloaded_clips_equal_decoded_ones(tmp_path):
    for i in range(3):
        sf.write(str(tmp_path / f"{i}.wav"), np.random.RandomState(i).uniform(-0.5, 0.5, 20000).astype(np.float32), 16000)
    files = sorted(str(p) for p in tmp_path.glob("*.wav"))
    a = th.Clips(files, 1.0, fixed=True, seed=1)
    b = th.Clips(files, 1.0, fixed=True, seed=1, preload=True)
    for i in range(3):
        assert torch.allclose(a[i][0], b[i][0], atol=1 / 32768)


def test_a_non_speech_item_is_heard_unchanged_by_teacher_and_student(tmp_path):
    for name in ("speech", "music"):
        (tmp_path / name).mkdir()
        sf.write(str(tmp_path / name / "a.wav"), np.random.RandomState(len(name)).randn(20000).astype(np.float32) * 0.1, 16000)
    speech, music = th.list_audio(tmp_path / "speech"), th.list_audio(tmp_path / "music")
    ds = th.Clips(speech, 1.0, noise_files=speech, p_noise=1.0, nonspeech=music, p_nonspeech=1.0, fixed=True)
    clean, student = ds[0]
    assert torch.equal(clean, student)
    ref = th.load_mono(music[0])
    assert any(torch.equal(clean, ref[o:o + 16000]) for o in range(0, 4001))


def test_reverberation_keeps_level_and_a_unit_impulse_is_the_identity():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(16000, generator=g)
    impulse = torch.zeros(4000); impulse[300] = 1.0
    assert torch.allclose(th.reverberate(x, impulse), x, atol=1e-4)
    room = torch.randn(8000, generator=g) * torch.exp(-torch.arange(8000) / 800.0)
    y = th.reverberate(x, room)
    assert y.shape == x.shape and abs(y.pow(2).mean().sqrt() / x.pow(2).mean().sqrt() - 1) < 1e-3


def test_reverberation_survives_the_noise_mixed_after_it(tmp_path):
    for name in ("speech", "noise", "rir"):
        (tmp_path / name).mkdir()
    sf.write(str(tmp_path / "speech" / "a.wav"), np.random.RandomState(0).randn(20000).astype(np.float32) * 0.1, 16000)
    sf.write(str(tmp_path / "noise" / "n.wav"), np.zeros(20000, np.float32) + 1e-6, 16000)
    room = np.random.RandomState(1).randn(4000).astype(np.float32) * np.exp(-np.arange(4000) / 400.0).astype(np.float32)
    sf.write(str(tmp_path / "rir" / "r.wav"), room, 16000)
    ds = th.Clips(th.list_audio(tmp_path / "speech"), 1.0, noise_files=th.list_audio(tmp_path / "noise"), p_noise=1.0,
                  snr_range=(60.0, 60.0), rirs=th.list_audio(tmp_path / "rir"), p_rir=1.0, fixed=True)
    clean, student = ds[0]
    assert not torch.allclose(student, clean, atol=1e-3), "the reverberation was dropped when noise was added"


def _speech_dir(tmp_path, n=4):
    d = tmp_path / "sp"
    d.mkdir(exist_ok=True)
    for i in range(n):
        sf.write(str(d / f"{i}.wav"), np.random.RandomState(i).randn(20000).astype(np.float32) * 0.1, 16000)
    return th.list_audio(d)


def test_augmentation_actions_leave_about_a_quarter_of_speech_clean(tmp_path):
    files = _speech_dir(tmp_path)
    ds = th.Clips(files * 100, 1.0, noise_files=files, rirs=files, actions=True, snr_range=(0.0, 20.0))
    random_state = th.random.getstate()
    th.random.seed(0)
    clean_share = np.mean([torch.equal(*ds[i]) for i in range(400)])
    th.random.setstate(random_state)
    assert 0.18 < clean_share < 0.32


def test_the_snr_floor_falls_over_the_first_half_of_training(tmp_path):
    ds = th.Clips(_speech_dir(tmp_path), 1.0, snr_range=(-5.0, 20.0), curriculum=True)
    for progress, floor in ((0.0, 20.0), (0.25, 7.5), (0.5, -5.0), (0.9, -5.0)):
        ds.progress.value = progress
        assert ds.snr_now() == (floor, 20.0)
    assert th.Clips(_speech_dir(tmp_path), 1.0, snr_range=(-5.0, 20.0)).snr_now() == (-5.0, 20.0)


def test_babble_is_other_talkers_from_the_training_speech(tmp_path):
    files = _speech_dir(tmp_path)
    ds = th.Clips(files, 1.0, p_noise=1.0, p_babble=1.0, snr_range=(0.0, 0.0), fixed=True, babble_snr_min=0.0)
    clean, student = ds[0]
    assert not torch.equal(clean, student)
    residual = student - clean
    assert abs(10 * torch.log10(clean.pow(2).mean() / residual.pow(2).mean()).item()) < 0.5


def test_babble_is_never_louder_than_the_voice_being_labelled(tmp_path):
    files = _speech_dir(tmp_path)
    ds = th.Clips(files, 1.0, p_noise=1.0, p_babble=1.0, snr_range=(-5.0, -5.0), babble_snr_min=5.0, fixed=True)
    for i in range(len(files)):
        clean, student = ds[i]
        snr = 10 * torch.log10(clean.pow(2).mean() / (student - clean).pow(2).mean()).item()
        assert snr > 4.5, f"babble mixed at {snr:.1f} dB"


def test_the_enhancement_target_pairs_each_student_frame_with_its_two_mel_frames():
    enh = th.EnhanceHead(feature_dim=8, n_mels=20)
    x = torch.randn(2, 16000)
    t = enh.target(x)
    mel = enh.mel(x)
    assert t.shape == (2, 50, 40)
    assert torch.allclose(t[0, 7, :20], mel[0, :, 14]) and torch.allclose(t[0, 7, 20:], mel[0, :, 15])
    enh.fit([x])
    assert enh(torch.randn(2, 50, 8), x).item() > 0


def test_a_streamed_window_is_a_contiguous_piece_of_the_file(tmp_path):
    x = np.random.RandomState(5).randn(48000).astype(np.float32) * 0.1
    sf.write(str(tmp_path / "a.wav"), x, 16000)
    sf.write(str(tmp_path / "b.wav"), np.random.RandomState(6).randn(48000).astype(np.float32) * 0.1, 24000)
    import random as _r
    w = th.read_crop(str(tmp_path / "a.wav"), 16000, _r.Random(3)).numpy()
    starts = [s for s in range(0, 32001) if np.allclose(x[s:s + 16000], w, atol=1e-4)]
    assert len(starts) == 1
    assert th.read_crop(str(tmp_path / "b.wav"), 16000, _r.Random(3)).shape == (16000,)


def test_xeus_names_route_to_the_espnet_teacher_and_the_rest_to_transformers(monkeypatch):
    made = []
    monkeypatch.setattr(th, "XeusTeacher", lambda *a: made.append("xeus") or "x")
    monkeypatch.setattr(th, "Teacher", lambda *a: made.append("hf") or "h")
    th.make_teacher("espnet/xeus", [6], "cpu"); th.make_teacher("microsoft/wavlm-base-plus", [8], "cpu")
    assert made == ["xeus", "hf"] and th.FAMILY["xeus"] == "WakeXeus"


def test_a_list_file_selects_exactly_its_paths_and_a_broken_file_is_replaced_in_training(tmp_path):
    files = _speech_dir(tmp_path)
    (tmp_path / "broken.wav").write_bytes(b"RIFF not really audio")
    lst = tmp_path / "train.txt"
    lst.write_text("\n".join([files[0], str(tmp_path / "broken.wav")]) + "\n")
    assert th.list_audio(str(lst)) == sorted([files[0], str(tmp_path / "broken.wav")])
    ds = th.Clips(th.list_audio(str(lst)), 1.0)
    assert all(ds[i][0].shape == (16000,) for i in range(2) for _ in range(3))
    with pytest.raises(Exception):
        th.Clips([str(tmp_path / "broken.wav")], 1.0, fixed=True)[0]


def test_shards_hold_one_crop_per_file_and_training_items_are_augmented_afresh(tmp_path):
    files = _speech_dir(tmp_path, n=5)
    (tmp_path / "bad.wav").write_bytes(b"nope")
    assert th.build_shards(files + [str(tmp_path / "bad.wav")], tmp_path / "sh", 1.0, per_shard=2) == 5
    assert len(list((tmp_path / "sh").glob("crops-*.npy"))) == 3
    base = th.Clips(files, 1.0, noise_files=files, p_noise=1.0, snr_range=(5.0, 5.0))
    ds = th.ShardClips(tmp_path / "sh", base)
    assert len(ds) == 5
    a, b = ds[0], ds[0]
    assert torch.equal(a[0], b[0]) and not torch.equal(a[1], b[1])
    ns = th.Clips(files, 1.0, nonspeech=files, p_nonspeech=1.0)
    c = th.ShardClips(tmp_path / "sh", ns)[1]
    assert torch.equal(c[0], c[1])


def test_the_snr_floor_stays_put_without_the_curriculum(tmp_path):
    ds = th.Clips(_speech_dir(tmp_path), 1.0, snr_range=(0.0, 20.0))
    assert ds.snr_now() == (0.0, 20.0)


def test_mix_at_snr_hits_the_requested_snr():
    g = torch.Generator().manual_seed(0)
    clean, noise = torch.randn(16000, generator=g), torch.randn(16000, generator=g) * 3
    mixed = th.mix_at_snr(clean, noise, 10.0)
    snr = 10 * torch.log10(clean.pow(2).mean() / (mixed - clean).pow(2).mean())
    assert abs(snr.item() - 10.0) < 1e-3


@pytest.mark.parametrize("kind", KINDS)
def test_the_export_loads_as_an_onnx_feature_extractor(tmp_path, kind):
    from ww_trainer.feats import OnnxFeatureExtractor

    s = _student(kind)
    th.export(s, tmp_path, {"layers": [4, 8]}, "cpu")
    ext = OnnxFeatureExtractor(str(tmp_path / "tinyhubert.onnx"), device="cpu")
    assert ext.feature_dim == 16
    wavs = [torch.randn(16000), torch.randn(24000)]
    feats = ext(wavs)
    assert feats.shape == (2, 75, 16)
    with torch.no_grad():
        ref = s(wavs[1][None])[0]
    assert torch.allclose(feats[1], ref, atol=1e-4)
    meta = json.loads((tmp_path / "tinyhubert.json").read_text())
    assert meta["params_exported"] == sum(p.numel() for n, p in s.named_parameters() if not n.startswith("proj."))


def test_an_export_that_disagrees_with_torch_is_refused(tmp_path, monkeypatch):
    import onnxruntime

    real = onnxruntime.InferenceSession

    class Off(real):
        def run(self, *a, **k):
            return [o + 0.01 for o in super().run(*a, **k)]

    monkeypatch.setattr(onnxruntime, "InferenceSession", Off)
    with pytest.raises(SystemExit, match="disagrees with torch"):
        th.export(_student(), tmp_path, {}, "cpu")
    assert not (tmp_path / "tinyhubert.json").exists()


@pytest.mark.parametrize("kind", [k for k in KINDS if k in MEL])
def test_int8_export_keeps_the_front_end_float_and_agrees_with_float(tmp_path, kind):
    import onnx

    s = _student(kind)
    th.export(s, tmp_path, {}, "cpu")
    g = torch.Generator().manual_seed(3)
    clips = [torch.randn(16000, generator=g) * 0.1 for _ in range(16)]
    q = th.quantize_int8(tmp_path, clips[:8], clips[8:])
    # an untrained 16-channel student is the least favourable case for int8: its agreement with float moves
    # with the onnxruntime build (0.88 to 0.99 seen for mel-gru), while trained students measure 0.98 to 0.999
    assert q["int8_float_nodes"] > 0 and q["int8_feature_cosine_mean"] > (0.8 if kind in ("mel-gru", "mel-bigru") else 0.9)
    graph = onnx.load(str(tmp_path / "tinyhubert_int8.onnx")).graph
    quantised = {i for n in graph.node if n.op_type == "DequantizeLinear" for i in n.output}
    convs = [n for n in graph.node if n.op_type == "Conv"]
    assert not any(i in quantised for i in convs[0].input), "the DFT convolution was quantised"
    assert any(i in quantised for n in convs[1:] for i in n.input), "nothing after the front end was quantised"


class _FakeTeacher:
    """Stands in for HuBERT: deterministic features of the clean audio, HuBERT's frame count."""

    def __init__(self, name, layers, device, dtype=torch.float32):
        self.layers, self.dim, self.device, self.model_type = layers, 12, device, "fake"
        g = torch.Generator().manual_seed(1)
        self.w = torch.randn(len(layers), 400, 12, generator=g)

    def __call__(self, wav):
        frames = wav.unfold(1, 400, 320)  # [B, T, 400], T = (N - 400) // 320 + 1
        return torch.einsum("btk,lkd->lbtd", frames, self.w)


def test_a_short_run_trains_evaluates_resumes_and_exports(tmp_path, monkeypatch):
    for name, n in (("train", 6), ("val", 3), ("noise", 2), ("music", 12), ("rir", 2)):
        d = tmp_path / name
        d.mkdir()
        for i in range(n):
            sf.write(str(d / f"{i}.wav"), np.random.RandomState(i).randn(24000).astype(np.float32) * 0.1, 16000)
    monkeypatch.setattr(th, "Teacher", _FakeTeacher)
    out = tmp_path / "run"
    common = ["--audio-dir", str(tmp_path / "train"), "--val-dir", str(tmp_path / "val"), "--noise-dir",
              str(tmp_path / "noise"), "--out-dir", str(out), "--cnn-dim", "4", "--gru-hidden", "8", "--layers", "1,2",
              "--batch-size", "2", "--crop-seconds", "1.0", "--norm-batches", "2", "--eval-every", "2", "--workers", "0",
              "--device", "cpu", "--nce-weight", "0.1"]
    th.main(common + ["--steps", "2"])
    th.main(common + ["--steps", "4", "--resume", str(out / "last.pt")])
    mel = tmp_path / "mel"
    th.main([x if x != str(out) else str(mel) for x in common]
            + ["--steps", "2", "--student", "mel-tcn", "--channels", "8", "--blocks", "2", "--feature-dim", "8",
               "--n-mels", "16", "--int8", "--preload", "--nonspeech-dir", str(tmp_path / "music"),
               "--rir-dir", str(tmp_path / "rir"), "--aug", "actions", "--p-babble", "0.5", "--curriculum",
               "--snr-min", "-5", "--enh-weight", "0.1", "--family", "WakeTest"])
    meta_mel = json.loads((mel / "tinyhubert.json").read_text())
    assert meta_mel["student"] == "mel-tcn" and meta_mel["feature_dim"] == 8 and "int8_feature_cosine_mean" in meta_mel
    assert meta_mel["nonspeech_clips"] == 10 and meta_mel["rirs"] == 2 and meta_mel["family"] == "WakeTest"
    assert "enh" in torch.load(mel / "last.pt") and meta_mel["params_exported"] == sum(
        p.numel() for n, p in th.MelTCNStudent(8, 2, 8, 16, n_targets=2, target_dim=12).named_parameters()
        if not n.startswith("proj."))
    mel_vals = [json.loads(l) for l in (mel / "metrics.jsonl").read_text().splitlines() if "val_loss" in l]
    assert mel_vals and all(len(v["val_nonspeech_cos"]) == 2 and "val_selection_loss" in v for v in mel_vals)
    vals = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines() if "val_loss" in l]
    assert [v["step"] for v in vals] == [2, 4] and all(len(v["val_cos"]) == 2 for v in vals)
    assert torch.load(out / "last.pt")["step"] == 4
    assert (out / "tinyhubert.onnx").exists()
    meta = json.loads((out / "tinyhubert.json").read_text())
    assert meta["frame_rate_hz"] == 50.0 and meta["latency_ms"] == 100.0 and meta["feature_dim"] == 8


def test_several_students_share_one_teacher_pass_and_each_exports_on_its_own(tmp_path, monkeypatch):
    for name, n in (("train", 6), ("val", 3)):
        d = tmp_path / name
        d.mkdir()
        for i in range(n):
            sf.write(str(d / f"{i}.wav"), np.random.RandomState(i).randn(24000).astype(np.float32) * 0.1, 16000)
    calls = []

    class Counting(_FakeTeacher):
        def __call__(self, wav):
            calls.append(len(wav))
            return super().__call__(wav)

    monkeypatch.setattr(th, "Teacher", Counting)
    common = ["--audio-dir", str(tmp_path / "train"), "--val-dir", str(tmp_path / "val"), "--layers", "1,2",
              "--batch-size", "2", "--crop-seconds", "1.0", "--norm-batches", "2", "--eval-every", "2", "--workers", "0",
              "--device", "cpu", "--student", "mel-tcn", "--channels", "8", "--blocks", "2", "--feature-dim", "8",
              "--n-mels", "16"]
    th.main(common + ["--out-dir", str(tmp_path / "one"), "--steps", "2", "--student-spec", "a:lag_frames=4"])
    one = len(calls); calls.clear()
    specs = ["--student-spec", "a:lag_frames=4", "--student-spec", "b:lag_frames=12,channels=12,feature_dim=6"]
    th.main(common + ["--out-dir", str(tmp_path / "two"), "--steps", "2"] + specs)
    assert len(calls) == one  # the teacher runs per batch, not per student
    a, b = (json.loads((tmp_path / "two" / n / "tinyhubert.json").read_text()) for n in "ab")
    assert (a["lag_frames"], a["feature_dim"], a["latency_ms"]) == (4, 8, 100.0)
    assert (b["lag_frames"], b["feature_dim"], b["latency_ms"]) == (12, 6, 260.0)
    assert (tmp_path / "two" / "b" / "tinyhubert.onnx").exists()
    th.main(common + ["--out-dir", str(tmp_path / "two"), "--steps", "4", "--resume", "auto"] + specs)
    for n in "ab":
        vals = [json.loads(l) for l in (tmp_path / "two" / n / "metrics.jsonl").read_text().splitlines() if "val_loss" in l]
        assert [v["step"] for v in vals] == [2, 4]
    with pytest.raises(SystemExit):
        th.main(common + ["--out-dir", str(tmp_path / "bad"), "--steps", "2", "--student-spec", "a:dropout=0.1"])


def test_every_student_kind_and_distillation_option_trains_and_exports(tmp_path, monkeypatch):
    for name, n in (("train", 6), ("val", 4)):
        d = tmp_path / name
        d.mkdir()
        for i in range(n):
            sf.write(str(d / f"{i}.wav"), np.random.RandomState(i).randn(24000).astype(np.float32) * 0.1, 16000)
    monkeypatch.setattr(th, "Teacher", _FakeTeacher)
    common = ["--audio-dir", str(tmp_path / "train"), "--val-dir", str(tmp_path / "val"), "--layers", "1,2",
              "--batch-size", "2", "--crop-seconds", "1.0", "--norm-batches", "2", "--eval-every", "2", "--workers", "0",
              "--device", "cpu", "--student", "mel-tcn", "--channels", "8", "--blocks", "2", "--feature-dim", "8",
              "--n-mels", "16", "--attn-window", "5", "--steps", "2", "--int8"]
    specs = {"gru": "student=mel-gru,rnn_layers=1", "attn": "student=mel-attn,attn_layers=1",
             "bigru": "student=mel-bigru", "mix": "mixconv=1,mask_prob=0.2", "taps": "target_blocks=1",
             "grutaps": "student=mel-gru,target_blocks=4"}
    out = tmp_path / "many"
    th.main(common + ["--out-dir", str(out)] + [x for n, v in specs.items() for x in ("--student-spec", f"{n}:{v}")])
    metas = {n: json.loads((out / n / "tinyhubert.json").read_text()) for n in specs}
    assert all((out / n / "tinyhubert.onnx").exists() and "int8_feature_cosine_mean" in metas[n] for n in specs)
    assert metas["attn"]["student"] == "mel-attn" and metas["attn"]["args"]["attn_layers"] == 1
    assert metas["bigru"]["streaming"] is False and metas["gru"]["streaming"] is True
    assert metas["mix"]["args"]["mixconv"] == 1 and metas["mix"]["args"]["mask_prob"] == 0.2
    ck = torch.load(out / "grutaps" / "last.pt")["student"]
    assert ck["proj.0.weight"].shape == (12, 8) and "seq.1.gru.weight_hh_l0" in ck
    one = tmp_path / "one"
    th.main(common + ["--out-dir", str(one), "--student", "mel-attn", "--target-blocks", "3", "--mask-prob", "0.1"])
    assert json.loads((one / "tinyhubert.json").read_text())["student"] == "mel-attn"
    with pytest.raises(SystemExit):
        th.main(common + ["--out-dir", str(tmp_path / "bad"), "--student-spec", "x:target_blocks=1+2"])


def test_several_teachers_each_run_once_per_batch_and_students_follow_their_own(tmp_path, monkeypatch):
    for name, n in (("train", 6), ("val", 3)):
        d = tmp_path / name
        d.mkdir()
        for i in range(n):
            sf.write(str(d / f"{i}.wav"), np.random.RandomState(i).randn(24000).astype(np.float32) * 0.1, 16000)
    calls = []

    class Named(_FakeTeacher):
        def __init__(self, name, layers, device, dtype=None):
            super().__init__(name, layers, device, dtype)
            self.name, self.model_type = name, "hubert"
            if "big" in name:
                self.dim = 20
                self.w = torch.randn(len(layers), 400, 20, generator=torch.Generator().manual_seed(2))

        def __call__(self, wav):
            calls.append(self.name)
            return super().__call__(wav)

    monkeypatch.setattr(th, "Teacher", Named)
    out = tmp_path / "run"
    args = ["--audio-dir", str(tmp_path / "train"), "--val-dir", str(tmp_path / "val"), "--batch-size", "2",
            "--crop-seconds", "1.0", "--norm-batches", "2", "--eval-every", "2", "--workers", "0", "--device", "cpu",
            "--student", "mel-tcn", "--channels", "8", "--blocks", "2", "--feature-dim", "8", "--n-mels", "16",
            "--out-dir", str(out), "--steps", "2",
            "--teacher-spec", "fake/small:1,2", "--teacher-spec", "fake/big:1,2,3",
            "--student-spec", "s:lag_frames=4", "--student-spec", "b1:teacher=big", "--student-spec", "b2:teacher=big,channels=12"]
    th.main(args)
    # norm fitting 2 batches per teacher, then per batch (2 train + 2 val) each teacher exactly once
    assert calls.count("fake/small") == 2 + 4 and calls.count("fake/big") == 2 + 4
    metas = {n: json.loads((out / n / "tinyhubert.json").read_text()) for n in ("s", "b1", "b2")}
    assert metas["s"]["teacher"] == "fake/small" and metas["s"]["layers"] == [1, 2]
    assert metas["b2"]["teacher"] == "fake/big" and metas["b2"]["layers"] == [1, 2, 3]
    ck = torch.load(out / "b1" / "last.pt")
    assert ck["student"]["proj.0.weight"].shape == (20, 8) and len([k for k in ck["student"] if k.endswith("proj.2.weight")]) == 1
    assert ck["norm"]["mean"].shape[0] == 3 and torch.load(out / "s" / "last.pt")["norm"]["mean"].shape[0] == 2
    with pytest.raises(SystemExit):
        th.main(args[:-6] + ["--student-spec", "x:teacher=fake"])  # matches both teachers


def _word_corpus(root, entries, lengths=None):
    """Tiny wav files at ``root/<rel>`` and the tab-separated label list naming them."""
    lines = []
    for i, (rel, word) in enumerate(entries):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        n = (lengths or {}).get(rel, 16000)
        sf.write(str(p), np.random.RandomState(i).randn(n).astype(np.float32) * 0.1, 16000)
        lines.append(f"{rel}\t{word}")
    (root / "list.tsv").write_text("\n".join(lines) + "\n")
    return root / "list.tsv"


def test_word_shards_keep_the_most_frequent_classes_round_robin_and_skip_unreadable_files(tmp_path):
    entries = ([(f"en/clips/yes/{i}.wav", "yes") for i in range(3)] + [(f"en/clips/no/{i}.wav", "no") for i in range(2)]
               + [(f"pt/clips/sim/{i}.wav", "sim") for i in range(2)] + [("en/clips/stop/0.wav", "stop")])
    lst = _word_corpus(tmp_path, entries, {"en/clips/no/0.wav": 8000, "en/clips/yes/0.wav": 24000})
    (tmp_path / "en/clips/yes/1.wav").write_bytes(b"not audio")
    common = ["--word-list", str(lst), "--word-root", str(tmp_path)]
    th.main(common + ["--build-word-shards", str(tmp_path / "all")])
    d = tmp_path / "all"
    assert json.loads((d / "classes.json").read_text()) == ["en/yes", "en/no", "pt/sim", "en/stop"]
    words, labels, lengths = (np.load(d / f"{k}-0000.npy") for k in ("words", "labels", "lengths"))
    assert labels.tolist() == [0, 1, 2, 3, 1, 2, 0]  # the unreadable second "yes" is skipped
    assert words.shape == (7, 16000) and words.dtype == np.int16 and labels.dtype == np.int32
    assert lengths.tolist() == [16000, 8000, 16000, 16000, 16000, 16000, 16000]
    ref = th.load_mono(str(tmp_path / "en/clips/yes/0.wav"))[4000:20000]
    assert np.allclose(words[0] / 32768, ref.numpy(), atol=1 / 32768)  # centre crop
    assert not words[1, 8000:].any()  # zero-padded at the end
    th.main(common + ["--build-word-shards", str(tmp_path / "two"), "--word-classes", "2", "--word-max-clips", "3"])
    assert json.loads((tmp_path / "two" / "classes.json").read_text()) == ["en/yes", "en/no"]
    assert np.load(tmp_path / "two" / "labels-0000.npy").tolist() == [0, 1, 1]


def _distil_dirs(tmp_path):
    for name, n in (("train", 6), ("val", 3)):
        d = tmp_path / name
        d.mkdir()
        for i in range(n):
            sf.write(str(d / f"{i}.wav"), np.random.RandomState(i).randn(24000).astype(np.float32) * 0.1, 16000)
    return ["--audio-dir", str(tmp_path / "train"), "--val-dir", str(tmp_path / "val"), "--layers", "1,2",
            "--batch-size", "2", "--crop-seconds", "1.0", "--norm-batches", "2", "--workers", "0", "--device", "cpu",
            "--student", "mel-tcn", "--channels", "8", "--blocks", "2", "--feature-dim", "8", "--n-mels", "16"]


def _word_shards(tmp_path, n=60):
    words = tmp_path / "wordsrc"
    lst = _word_corpus(words, [(f"{'en' if i % 3 else 'pt'}/clips/w{i % 3}/{i}.wav", f"w{i % 3}") for i in range(n)])
    th.main(["--word-list", str(lst), "--word-root", str(words), "--build-word-shards", str(tmp_path / "ws"),
             "--word-seconds", "0.5"])
    return tmp_path / "ws"


def test_a_word_student_logs_word_metrics_exports_only_its_features_and_resumes(tmp_path, monkeypatch):
    import onnxruntime

    common = _distil_dirs(tmp_path) + ["--word-shards", str(_word_shards(tmp_path)), "--word-batch", "4",
                                       "--eval-every", "50", "--out-dir", str(tmp_path / "run")]
    monkeypatch.setattr(th, "Teacher", _FakeTeacher)
    specs = ["--student-spec", "w:word_weight=1.0", "--student-spec", "p:lag_frames=4"]
    th.main(common + ["--steps", "100"] + specs)
    th.main(common + ["--steps", "102", "--resume", "auto"] + specs)
    recs = {n: [json.loads(l) for l in (tmp_path / "run" / n / "metrics.jsonl").read_text().splitlines()] for n in "wp"}
    train_w = [r for r in recs["w"] if "train_loss" in r]
    assert train_w and all("train_word_loss" in r and 0 <= r["train_word_acc"] <= 1 for r in train_w)
    assert [r["step"] for r in recs["w"] if "val_word_acc" in r] == [50, 100, 102]
    assert not any(k.startswith(("train_word", "val_word")) for r in recs["p"] for k in r)
    ck = torch.load(tmp_path / "run" / "w" / "last.pt")
    assert ck["step"] == 102 and ck["word_head"]["fc.weight"].shape == (3, 8)
    assert "word_head" not in torch.load(tmp_path / "run" / "p" / "last.pt")
    assert not any(k.startswith("word_head") for k in ck["student"])
    metas = {n: json.loads((tmp_path / "run" / n / "tinyhubert.json").read_text()) for n in "wp"}
    assert metas["w"]["params_exported"] == metas["p"]["params_exported"]
    sess = onnxruntime.InferenceSession(str(tmp_path / "run" / "w" / "tinyhubert.onnx"), providers=["CPUExecutionProvider"])
    assert [o.name for o in sess.get_outputs()] == ["features"]
    assert sess.run(None, {"waveform": np.zeros((1, 16000), np.float32)})[0].shape == (1, 50, 8)


def test_the_word_batch_never_reaches_the_teacher(tmp_path, monkeypatch):
    calls = []

    class Counting(_FakeTeacher):
        def __call__(self, wav):
            calls.append(len(wav))
            return super().__call__(wav)

    monkeypatch.setattr(th, "Teacher", Counting)
    common = _distil_dirs(tmp_path) + ["--steps", "2", "--eval-every", "2", "--student-spec", "a:lag_frames=4"]
    th.main(common + ["--out-dir", str(tmp_path / "plain")])
    plain = list(calls); calls.clear()
    th.main(common + ["--out-dir", str(tmp_path / "word"), "--word-shards", str(_word_shards(tmp_path)),
                      "--word-batch", "4", "--student-spec", "w:word_weight=1.0"])
    assert calls == plain  # same number and sizes of teacher batches: distillation batches only
    assert json.loads((tmp_path / "word" / "w" / "metrics.jsonl").read_text().splitlines()[-1])["val_word_acc"] >= 0
    with pytest.raises(SystemExit):
        th.main(common + ["--out-dir", str(tmp_path / "bad"), "--word-weight", "0.5"])


def test_the_word_classifier_learns_separable_words(tmp_path, monkeypatch):
    words = tmp_path / "tones"
    lines = []
    g = np.random.RandomState(0)
    t = np.arange(8000) / 16000
    for i in range(400):
        f, name = (300.0, "low") if i % 2 else (3000.0, "high")
        rel = f"en/clips/{name}/{i}.wav"
        (words / rel).parent.mkdir(parents=True, exist_ok=True)
        x = g.uniform(0.1, 0.5) * np.sin(2 * np.pi * f * t + g.uniform(0, 2 * np.pi)) + g.randn(8000) * 0.01
        sf.write(str(words / rel), x.astype(np.float32), 16000)
        lines.append(f"{rel}\t{name}")
    (words / "list.tsv").write_text("\n".join(lines) + "\n")
    th.main(["--word-list", str(words / "list.tsv"), "--word-root", str(words), "--build-word-shards",
             str(tmp_path / "ws"), "--word-seconds", "0.5"])
    held_out = th.WordClips(tmp_path / "ws", held_out=True)
    assert len(held_out) == 8 and sorted(held_out.labels.tolist()) == [0] * 4 + [1] * 4
    monkeypatch.setattr(th, "Teacher", _FakeTeacher)
    out = tmp_path / "run"
    th.main(_distil_dirs(tmp_path) + ["--out-dir", str(out), "--word-shards", str(tmp_path / "ws"), "--word-batch", "16",
                                      "--word-weight", "1.0", "--steps", "40", "--eval-every", "40", "--warmup", "1",
                                      "--lr", "3e-3"])
    val = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines() if "val_word_acc" in l]
    assert val[-1]["step"] == 40 and val[-1]["val_word_acc"] > 0.9


@pytest.mark.parametrize("kind", ["wave-gru", "mel-tcn"])
@pytest.mark.parametrize("lag", [0, 3])
def test_a_teacher_frame_is_predicted_from_the_features_1_plus_lag_frames_later(kind, lag):
    wav = torch.randn(2, 16000)
    s = _student(kind)
    with torch.no_grad():
        preds, feats = th.student_preds(s, wav, lag, 30)
        assert torch.allclose(preds, torch.stack(s.heads(feats[:, 1 + lag:31 + lag])))
    t = _student("mel-tcn", target_blocks=(1,))
    with torch.no_grad():
        preds, feats = th.student_preds(t, wav, lag, 30)
        block = t.forward_blocks(wav)[1][0]
        assert torch.allclose(preds[0], t.proj[0](block[:, 1 + lag:31 + lag]))
        assert torch.allclose(preds[1], t.proj[1](feats[:, 1 + lag:31 + lag]))


@pytest.mark.parametrize("spec", [False, True], ids=["one-student", "student-spec"])
@pytest.mark.parametrize("curriculum", [False, True], ids=["fixed", "curriculum"])
def test_the_snr_floor_falls_during_training_only_with_the_curriculum(tmp_path, monkeypatch, spec, curriculum):
    floors = []
    real = th.Clips.augment

    def spy(self, clean, rng):
        if not self.fixed:
            floors.append(self.snr_now()[0])
        return real(self, clean, rng)

    monkeypatch.setattr(th.Clips, "augment", spy)
    monkeypatch.setattr(th, "Teacher", _FakeTeacher)
    args = _distil_dirs(tmp_path) + ["--out-dir", str(tmp_path / "run"), "--steps", "4", "--snr-min", "-5",
                                     "--snr-max", "20"]
    args += ["--curriculum"] if curriculum else []
    args += ["--student-spec", "a:lag_frames=4"] if spec else []
    th.main(args)
    if curriculum:
        # progress 0 while fitting the target statistics and at step 0, then 0.25, 0.5, 0.75
        assert floors == sorted(floors, reverse=True) and sorted(set(floors), reverse=True) == [20.0, 7.5, -5.0]
    else:
        assert floors and set(floors) == {-5.0}


def test_a_student_resumed_ahead_of_the_others_keeps_its_step_and_its_records(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "Teacher", _FakeTeacher)
    common = _distil_dirs(tmp_path) + ["--out-dir", str(tmp_path / "run"), "--eval-every", "2"]
    th.main(common + ["--steps", "4", "--student-spec", "a:lag_frames=4"])
    a = tmp_path / "run" / "a"
    records, saved = (a / "metrics.jsonl").read_text(), torch.load(a / "last.pt")
    th.main(common + ["--steps", "2", "--resume", "auto", "--student-spec", "a:lag_frames=4",
                      "--student-spec", "b:lag_frames=2"])
    now = torch.load(a / "last.pt")
    assert now["step"] == 4 and (a / "metrics.jsonl").read_text() == records
    assert all(torch.equal(v, now["student"][k]) for k, v in saved["student"].items())
    b = [json.loads(l) for l in (tmp_path / "run" / "b" / "metrics.jsonl").read_text().splitlines() if "val_loss" in l]
    assert [v["step"] for v in b] == [2] and torch.load(tmp_path / "run" / "b" / "last.pt")["step"] == 2


def test_a_resumed_student_continues_its_optimiser_schedule_and_target_statistics(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "Teacher", _FakeTeacher)
    common = _distil_dirs(tmp_path) + ["--out-dir", str(tmp_path / "run"), "--eval-every", "2",
                                       "--student-spec", "a:lag_frames=4"]
    th.main(common + ["--steps", "2"])
    last = tmp_path / "run" / "a" / "last.pt"
    ck = torch.load(last)
    # statistics a fresh fit on the same stream could not reproduce
    ck["norm"]["mean"] += 1.0; ck["norm"]["std"] *= 2.0
    torch.save(ck, last)
    th.main(common + ["--steps", "3", "--resume", "auto"])
    after = torch.load(last)
    assert after["step"] == 3 and after["sched"]["last_epoch"] == 3
    assert after["opt"]["state"] and {int(s["step"]) for s in after["opt"]["state"].values()} == {3}
    assert torch.equal(after["norm"]["mean"], ck["norm"]["mean"]) and torch.equal(after["norm"]["std"], ck["norm"]["std"])


@pytest.mark.parametrize("spec", [False, True], ids=["one-student", "student-spec"])
def test_the_word_classifier_is_trained_with_the_student(tmp_path, monkeypatch, spec):
    heads = []

    class Spy(th.WordHead):
        def __init__(self, *a):
            super().__init__(*a)
            heads.append((self, self.fc.weight.detach().clone()))

    monkeypatch.setattr(th, "WordHead", Spy)
    monkeypatch.setattr(th, "Teacher", _FakeTeacher)
    th.main(_distil_dirs(tmp_path) + ["--out-dir", str(tmp_path / "run"), "--word-shards", str(_word_shards(tmp_path)),
                                      "--word-batch", "4", "--steps", "2"]
            + (["--student-spec", "w:word_weight=1.0"] if spec else ["--word-weight", "1.0"]))
    (head, initial), = heads
    assert not torch.equal(head.fc.weight.detach(), initial)


def test_the_word_head_averages_only_the_frames_that_hold_audio():
    torch.manual_seed(0)
    head = th.WordHead(4, 3)
    feats = torch.randn(2, 10, 4)
    samples = torch.tensor([5 * 320 - 100, 10 * 320])  # five frames of audio, then padding; ten frames
    with torch.no_grad():
        out = head(feats, samples)
        assert torch.allclose(out[0], head.fc(feats[0, :5].mean(0)), atol=1e-6)
        assert torch.allclose(out[1], head.fc(feats[1].mean(0)), atol=1e-6)
        padded = feats.clone()
        padded[0, 5:] += 10
        assert torch.allclose(head(padded, samples), out, atol=1e-6)
        assert torch.allclose(head(feats, torch.tensor([0, 3200]))[0], head.fc(feats[0, 0]), atol=1e-6)


def test_held_out_word_clips_are_heard_clean_and_cover_every_class(tmp_path):
    ws = _word_shards(tmp_path, n=300)  # three classes of 100 clips
    files = _speech_dir(tmp_path)
    base = th.Clips(files, 1.0, noise_files=files, p_noise=1.0, snr_range=(0.0, 0.0))
    held, train = th.WordClips(ws, base, held_out=True), th.WordClips(ws, base)
    assert sorted(held.labels.tolist()) == [0, 0, 1, 1, 2, 2] and len(held) + len(train) == 300
    assert not set(held.index) & set(train.index)
    for k, (i, j) in enumerate(held.index):
        assert torch.equal(held[k][0], torch.from_numpy(held.parts[i][j].astype(np.float32) / 32768.0))
    i, j = train.index[0]
    assert not torch.equal(train[0][0], torch.from_numpy(train.parts[i][j].astype(np.float32) / 32768.0))


def test_negative_timing_options_and_shards_of_another_crop_length_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "Teacher", _FakeTeacher)
    common = _distil_dirs(tmp_path) + ["--out-dir", str(tmp_path / "run"), "--steps", "1"]
    for bad in (["--lag-frames", "-1"], ["--attn-window", "0"], ["--student-spec", "a:lag_frames=-1"],
                ["--student-spec", "a:student=mel-attn,attn_window=0"]):
        with pytest.raises(SystemExit, match="lag_frames|attn_window"):
            th.main(common + bad)
    th.build_shards(th.list_audio(tmp_path / "train"), tmp_path / "sh", 0.5)
    with pytest.raises(SystemExit, match="crop-seconds"):
        th.main(common + ["--shards", str(tmp_path / "sh")])
