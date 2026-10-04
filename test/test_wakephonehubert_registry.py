"""The WakePhoneHuBERT entry of the pretrained featurizer registry.

test_entry_resolves_its_config_and_gives_521_wide_features downloads from the Hugging Face repository
TigreGotico/wakephonehubert, so it needs network access and a Hugging Face token that can read that repository.
"""
import numpy as np
import onnxruntime as ort
import torch

from ww_trainer.pretrained import PRETRAINED_FEATURIZERS, load_pretrained_featurizer, resolve_pretrained

NAME = "wakephonehubert-int8"


def test_entry_fields():
    e = PRETRAINED_FEATURIZERS[NAME]
    assert (e.repo_id, e.variant, e.license, e.context_samples, e.revision) == (
        "TigreGotico/wakephonehubert", "features_int8", "apache-2.0", 41920, "343b2497cb0ef0310d163eb53e3f0f817260dbf7")


def test_entry_resolves_its_config_and_gives_521_wide_features():
    onnx_path, config = resolve_pretrained(NAME)
    assert onnx_path.endswith("wakephonehubert_features_int8.onnx")
    assert config["feature_dim"] == 521
    assert config["output"]["hop_samples"] == 320 and config["output"]["frame_rate_hz"] == 50.0
    assert config["receptive_field_samples"] == 41920
    assert config["output"]["layout"] == {"hubert": [0, 128], "vad": [128, 129], "ipa": [129, 521]}

    ext = load_pretrained_featurizer(NAME, device="cpu")
    assert (ext.feature_dim, ext.hop_samples, ext.context_samples, ext.streaming) == (521, 320, 41920, True)
    feats = ext(torch.zeros(1, 24000))
    assert tuple(feats.shape) == (1, 75, 521)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    out = session.run(None, {session.get_inputs()[0].name: (rng.standard_normal((1, 24000)) * 0.1).astype(np.float32)})[0]
    assert out.shape == (1, 75, 521)
    vad, ipa = out[..., 128], out[..., 129:]
    assert ((vad >= 0) & (vad <= 1)).all()
    assert np.allclose(ipa.sum(-1), 1.0, atol=1e-3)
