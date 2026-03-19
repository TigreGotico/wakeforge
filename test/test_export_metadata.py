"""Tests for export_to_onnx metadata propagation and export_featurizer path."""
import os
import pytest
import torch
import numpy as np
import onnx

from ww_trainer.feats import MfccExtractor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel
from ww_trainer.utils import embed_onnx_metadata


def test_base_extractor_export_accepts_metadata(tmp_path: str) -> None:
    """BaseExtractor.export_to_onnx should accept and embed metadata."""
    ext = MfccExtractor(n_mfcc=13, sr=16000)
    ext.to("cpu")
    ext.device = torch.device("cpu")

    out = str(tmp_path / "mfcc.onnx")
    metadata = {"wake_word": "hey_test", "version": "1.0"}
    ext.export_to_onnx(out, metadata=metadata)

    assert os.path.exists(out)
    model = onnx.load(out)
    props = {p.key: p.value for p in model.metadata_props}
    assert props["wake_word"] == "hey_test"
    assert props["version"] == "1.0"


def test_base_extractor_export_no_metadata(tmp_path: str) -> None:
    """BaseExtractor.export_to_onnx still works without metadata."""
    ext = MfccExtractor(n_mfcc=13, sr=16000)
    ext.to("cpu")
    ext.device = torch.device("cpu")

    out = str(tmp_path / "mfcc_nometa.onnx")
    ext.export_to_onnx(out)

    assert os.path.exists(out)
    model = onnx.load(out)
    assert len(model.metadata_props) == 0


def test_classifier_head_export_kwargs(tmp_path: str) -> None:
    """ClassifierHead.export_to_onnx works with keyword arguments."""
    head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")
    out = str(tmp_path / "head.onnx")
    metadata = {"arch": "ffn", "test": "true"}
    head.export_to_onnx(out, quantize=False, dynamo=False, metadata=metadata)

    assert os.path.exists(out)
    model = onnx.load(out)
    props = {p.key: p.value for p in model.metadata_props}
    assert props["arch"] == "ffn"


def test_base_wake_model_export_featurizer_with_metadata(tmp_path: str) -> None:
    """BaseWakeModel.export_to_onnx with export_featurizer=True and metadata.

    This is the path that was broken by BUG-2 before the fix.
    """
    ext = MfccExtractor(n_mfcc=13, sr=16000)
    ext.to("cpu")
    ext.device = torch.device("cpu")
    head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")
    model = BaseWakeModel(feature_extractor=ext, classifier=head, device="cpu")

    out = str(tmp_path / "model.onnx")
    metadata = {"wake_word": "hey_test", "epoch": "5"}
    model.export_to_onnx(out, export_featurizer=True, metadata=metadata)

    # Head ONNX should exist with metadata
    assert os.path.exists(out)
    head_model = onnx.load(out)
    head_props = {p.key: p.value for p in head_model.metadata_props}
    assert head_props["wake_word"] == "hey_test"

    # Featurizer ONNX should exist with metadata
    feat_out = out.replace(".onnx", "_featurizer.onnx")
    assert os.path.exists(feat_out)
    feat_model = onnx.load(feat_out)
    feat_props = {p.key: p.value for p in feat_model.metadata_props}
    assert feat_props["wake_word"] == "hey_test"


def test_base_wake_model_export_quantize_flag(tmp_path: str) -> None:
    """Verify quantize=True actually produces an int8 file (BUG-1 regression test)."""
    head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")
    out = str(tmp_path / "head.onnx")
    head.export_to_onnx(out, quantize=True)

    int8_path = str(tmp_path / "head_int8.onnx")
    assert os.path.exists(int8_path), "Quantized model should be created when quantize=True"
