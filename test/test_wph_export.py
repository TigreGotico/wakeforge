"""scripts/research/wakephonehubert/export/build_wakephonehubert.py: random heads on a tiny quantised trunk become the
published ONNX layout, and the checks the exporter and the validator apply pass on it.

The trunk is a few-kilobyte QDQ graph with the structure the exporter locates (log-mel, BatchNorm, stride-2 stem,
eight residual blocks with ReLU folded into uint8 quantisation, a 1x1 output projection), so no model is downloaded.
"""
import importlib.util
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto as TP
from onnx import helper, numpy_helper
from scipy.fft import dct

_DIR = Path(__file__).resolve().parent.parent / "scripts" / "research" / "wakephonehubert"
_S = importlib.util.spec_from_file_location("build_wakephonehubert", _DIR / "export" / "build_wakephonehubert.py")
bw = importlib.util.module_from_spec(_S)
_S.loader.exec_module(bw)

N_IPA = 392
FRAMES = 10
SAMPLES = FRAMES * 320


def make_trunk():
    rng = np.random.default_rng(0)
    init, nodes = [], []

    def f32(name, a):
        init.append(numpy_helper.from_array(np.asarray(a, np.float32), name))
        return name

    def qdq(name, x, scale=0.05, dtype=np.uint8):
        init.append(numpy_helper.from_array(np.asarray(scale, np.float32), f"{name}_scale"))
        init.append(numpy_helper.from_array(np.asarray(0, dtype), f"{name}_zp"))
        nodes.append(helper.make_node("QuantizeLinear", [x, f"{name}_scale", f"{name}_zp"], [f"{name}_q"], name=f"{name}_Quantize"))
        nodes.append(helper.make_node("DequantizeLinear", [f"{name}_q", f"{name}_scale", f"{name}_zp"], [f"{name}_dq"], name=f"{name}_Dequantize"))
        return f"{name}_dq"

    init.append(numpy_helper.from_array(np.array([1], np.int64), "axes1"))
    nodes += [
        helper.make_node("Unsqueeze", ["waveform", "axes1"], ["wave3"], name="unsqueeze"),
        helper.make_node("Conv", ["wave3", f32("fb_w", rng.normal(0, 0.05, (64, 1, 320)))], ["fb"], strides=[160], pads=[80, 80], name="filterbank"),
        helper.make_node("Mul", ["fb", "fb"], ["power"], name="power"),
        helper.make_node("Add", ["power", f32("eps", 1e-3)], ["power_eps"], name="eps"),
        helper.make_node("Log", ["power_eps"], ["mel_log"], name="log"),
        helper.make_node("BatchNormalization", ["mel_log", f32("bn_scale", np.ones(64)), f32("bn_bias", np.zeros(64)),
                                                f32("bn_mean", np.full(64, -5.0)), f32("bn_var", np.full(64, 4.0))], ["mel_norm"], name="norm"),
        helper.make_node("Conv", ["mel_norm", f32("stem_w", rng.normal(0, 0.2, (256, 64, 3))), f32("stem_b", rng.normal(0, 0.1, 256))],
                         ["stem_out"], strides=[2], pads=[1, 1], name="stem"),
    ]
    x = qdq("tap0", "stem_out")
    for i in range(1, 9):
        nodes.append(helper.make_node("Conv", [x, f32(f"block{i}_w", rng.normal(0, 0.03, (256, 256, 1)))], [f"block{i}_out"], name=f"block{i}"))
        branch = qdq(f"branch{i}", f"block{i}_out", scale=0.02)
        nodes.append(helper.make_node("Add", [x, branch], [f"sum{i}"], name=f"residual{i}"))
        x = qdq(f"tap{i}", f"sum{i}")
    init.append(numpy_helper.from_array(rng.integers(-60, 60, (128, 256, 1)).astype(np.int8), "proj_q"))
    init.append(numpy_helper.from_array(np.asarray(0.01, np.float32), "proj_scale"))
    init.append(numpy_helper.from_array(np.asarray(0, np.int8), "proj_zp"))
    nodes += [
        helper.make_node("DequantizeLinear", ["proj_q", "proj_scale", "proj_zp"], ["proj_w"], name="proj_weight_Dequantize"),
        helper.make_node("Conv", [x, "proj_w"], ["proj_out"], name="proj"),
    ]
    y = qdq("out", "proj_out", scale=0.05, dtype=np.int8)
    nodes.append(helper.make_node("Transpose", [y], ["features"], perm=[0, 2, 1], name="to_frames_last"))
    graph = helper.make_graph(nodes, "tiny_quantised_trunk",
                              [helper.make_tensor_value_info("waveform", TP.FLOAT, ["batch", "samples"])],
                              [helper.make_tensor_value_info("features", TP.FLOAT, ["batch", "frames", 128])], init)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    return model


def make_heads(seed=0):
    from wph_model import TapHead
    torch.manual_seed(seed)
    heads = {"vad": TapHead(1, hidden=16), "ipa": TapHead(N_IPA, hidden=24, side=16)}
    for h in heads.values():
        for p in h.parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, 0, 0.08)
    return {k: v.eval() for k, v in heads.items()}


def run(path, wav, names=None, optimize=True):
    so = ort.SessionOptions()
    if not optimize:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    names = names or [o.name for o in s.get_outputs()]
    return dict(zip(names, s.run(names, {"waveform": wav})))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("export")
    trunk, heads = make_trunk(), make_heads()
    trunk_path = d / "trunk.onnx"
    onnx.save(trunk, str(trunk_path))
    loc = bw.locate(trunk.graph)
    tail = bw.export_tail(heads, d / "tail.onnx")
    full = bw.merge(trunk, tail, loc)
    onnx.checker.check_model(full, full_check=True)
    full_path = d / "full.onnx"
    onnx.save(full, str(full_path))
    return {"dir": d, "trunk": trunk, "trunk_path": trunk_path, "heads": heads, "loc": loc, "tail": tail, "full": full, "full_path": full_path}


def waves(n=3):
    rng = np.random.default_rng(1)
    return [(rng.standard_normal(SAMPLES * (k + 1)) * 0.2).astype(np.float32) for k in range(n)]


def test_locate_finds_the_nine_taps_the_output_and_the_mel(built):
    loc = built["loc"]
    assert [t[0] for t in loc["taps"]] == [f"tap{i}_q" for i in range(9)]
    assert loc["out"][0] == "out_q"
    assert (loc["mel_log"], loc["mel_norm"]) == ("mel_log", "mel_norm")
    assert len(loc["residual_adds"]) == 8


def test_merged_model_outputs_and_shapes(built):
    w = np.stack([waves()[0], waves()[0][::-1]])
    r = run(built["full_path"], w)
    assert list(r) == ["hubert", "layers", "logmel", "mfcc", "vad", "ipa", "features"]
    b, t = w.shape[0], FRAMES
    assert r["hubert"].shape == (b, t, 128)
    assert r["layers"].shape == (b, t, 9, 256)
    assert r["logmel"].shape == (b, 2 * t, 64)
    assert r["mfcc"].shape == (b, 2 * t, 20)
    assert r["vad"].shape == (b, t, 1)
    assert r["ipa"].shape == (b, t, N_IPA)
    assert r["features"].shape == (b, t, 128 + 1 + N_IPA)
    assert r["features"].shape[-1] == 521


def test_hubert_output_is_the_trunk_bit_for_bit_and_features_are_the_join(built):
    # Without the optimiser: its fusions differ between the two graphs and move a few values across a rounding tie.
    for w in waves():
        r = run(built["full_path"], w[None], optimize=False)
        ref = run(built["trunk_path"], w[None], optimize=False)["features"]
        assert np.array_equal(r["hubert"].view(np.uint32), ref.view(np.uint32))
        assert np.array_equal(r["features"], np.concatenate([r["hubert"], r["vad"], r["ipa"]], -1))


def test_head_outputs_are_probabilities(built):
    r = run(built["full_path"], waves()[1][None])
    assert ((r["vad"] > 0) & (r["vad"] < 1)).all()
    assert np.allclose(r["ipa"].sum(-1), 1.0, atol=1e-5)
    assert r["ipa"].std() > 1e-3


def test_onnx_heads_match_pytorch_heads_on_the_onnx_taps(built):
    r = run(built["full_path"], waves()[2][None])
    taps = torch.from_numpy(np.ascontiguousarray(r["layers"].transpose(0, 2, 3, 1)))
    out = torch.from_numpy(np.ascontiguousarray(r["hubert"].transpose(0, 2, 1)))
    mel = torch.from_numpy(r["logmel"]).transpose(1, 2)
    t = out.shape[-1]
    mel2 = mel.reshape(1, 64, t, 2).permute(0, 1, 3, 2).reshape(1, 128, t)
    with torch.no_grad():
        for name in ("vad", "ipa"):
            ref = bw.activate(name, built["heads"][name](taps, out, mel2)).numpy()
            assert np.abs(ref - r[name]).max() < 1e-4, name


def test_mfcc_matrix_is_the_orthonormal_dct():
    ref = dct(np.eye(64), type=2, norm="ortho", axis=0)[:20].T
    assert np.allclose(bw.dct_matrix(), ref, atol=1e-6)


def test_features_file_is_the_pruned_model(built):
    feat = bw.prune(built["full"], ["features"])
    onnx.checker.check_model(feat, full_check=True)
    path = built["dir"] / "features.onnx"
    onnx.save(feat, str(path))
    w = waves()[0][None]
    assert [o.name for o in feat.graph.output] == ["features"]
    assert np.array_equal(run(path, w)["features"], run(built["full_path"], w)["features"])
    assert path.stat().st_size < built["full_path"].stat().st_size


@pytest.mark.parametrize("transform", ["float16", "int8-weights"])
def test_reduced_precision_head_passes_the_acceptance_rule(built, transform):
    tail = built["tail"]
    cand = {"float16": bw.float16_heads, "int8-weights": bw.int8_weight_heads}[transform](tail, ["ipa"])
    cand_path = built["dir"] / f"{transform}.onnx"
    onnx.save(bw.merge(built["trunk"], cand, built["loc"]), str(cand_path))
    rule = {"tol": 0.02, "agree": 0.99}
    got = bw.acceptance(built["full_path"], cand_path, waves(), ["vad", "ipa"], rule)
    assert got["ipa"]["pass"] and got["vad"]["pass"], got
    assert got["ipa"]["decisions"] > 0
    assert cand_path.stat().st_size < built["full_path"].stat().st_size


def test_acceptance_rule_rejects_a_different_head(built):
    other = bw.merge(built["trunk"], bw.export_tail(make_heads(seed=7), built["dir"] / "other_tail.onnx"), built["loc"])
    other_path = built["dir"] / "other.onnx"
    onnx.save(other, str(other_path))
    got = bw.acceptance(built["full_path"], other_path, waves(), ["vad", "ipa"], {"tol": 0.02, "agree": 0.99})
    assert not got["ipa"]["pass"] and not got["vad"]["pass"], got


def test_acceptance_rule_rejects_on_decision_agreement_alone(built):
    other = bw.merge(built["trunk"], bw.export_tail(make_heads(seed=7), built["dir"] / "other_tail.onnx"), built["loc"])
    other_path = built["dir"] / "other.onnx"
    onnx.save(other, str(other_path))
    got = bw.acceptance(built["full_path"], other_path, waves(), ["vad", "ipa"], {"tol": 1.0, "agree": 0.99})
    for head in ("vad", "ipa"):
        assert got[head]["max_abs_prob_diff"] < 1.0, got
        assert got[head]["decision_agreement"] < 0.99, got
        assert not got[head]["pass"], got
    same = bw.acceptance(built["full_path"], built["full_path"], waves(), ["vad", "ipa"], {"tol": 1.0, "agree": 0.99})
    assert same["vad"]["pass"] and same["ipa"]["pass"], same


def test_load_head_round_trips_a_train_pool_checkpoint(tmp_path):
    heads = make_heads()
    torch.save(heads["vad"].state_dict(), tmp_path / "vad_head.pt")
    torch.save({"head": heads["ipa"].state_dict(), "symbols": [f"s{i}" for i in range(N_IPA)], "blank": 0,
                "dropped_ids": [0, 1, 2, 3], "delay": 5}, tmp_path / "ipa_head.pt")
    vad, vad_side, vad_meta = bw.load_head(tmp_path / "vad_head.pt")
    ipa, ipa_side, ipa_meta = bw.load_head(tmp_path / "ipa_head.pt")
    assert (vad_side, ipa_side, vad_meta) == (0, 16, {})
    assert ipa_meta["blank"] == 0 and ipa_meta["delay"] == 5 and len(ipa_meta["symbols"]) == N_IPA
    assert ipa.o.out_channels == N_IPA and ipa.c1.out_channels == 24
