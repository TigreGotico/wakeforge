"""Tests for advanced ONNX export features and metadata embedding."""
import os
import pytest
import torch
import numpy as np
from pathlib import Path
import onnx
import onnxruntime as ort

from ww_trainer.utils import embed_onnx_metadata
from ww_trainer.feats import (
    MfccExtractor,
    MarkovTransitionExtractor,
    HMMStateExtractor,
)
from ww_trainer.model import FfnClassifierHead
def test_markov_boundary_training():
    base = MfccExtractor(n_mfcc=13, sr=16000)
    # Use order 2, n_codes 4
    ext = MarkovTransitionExtractor(base, n_codes=4, order=2)
    ext.to("cpu")
    ext.device = torch.device("cpu")
    base.to("cpu")
    base.device = torch.device("cpu")

    # We'll mock base.forward to return something that will result in 
    # predictable k-means and tokens.
    # Actually, easier to mock _simple_kmeans and _quantize.

    ext._simple_kmeans = lambda data, k: torch.randn(k, 13)
    ext._fitted = True

    # Mock _quantize to return [0, 0, 1, 1, 1]
    ext._quantize = lambda feats: torch.tensor([[0, 0, 1, 1, 1]], device=feats.device)

    # Fit on 1 sample
    ext.fit([torch.randn(16000)])

    # Verify transition matrix
    tm = ext._transition_matrix

    # State 0 (context [0,0]) should have transitions to 0 and 1

    # Transitions observed:
    # t=0: state 0 -> token[1]=0. Count(0,0)=1
    # t=1: state 0 -> token[2]=1. Count(0,1)=1
    # Total for state 0 = 2.
    # tm[0] = [0.5, 0.5, 0, 0] (approx)

    assert tm[0, 0] > 0.4
    assert tm[0, 1] > 0.4
    # State 1 (context [0,1]) should have transition to 1
    assert tm[1, 1] > 0.9

    # State 5 (context [1,1]) should have transition to 1
    assert tm[5, 1] > 0.9

def test_multi_onnx_inferencer(tmp_path):
    # 1. Create a mock base extractor (Identity for MFCC-like shape)
    # wav [1, 16000] -> feats [1, 100, 13]
    base_path = str(tmp_path / "base.onnx")
    
    # Simple constant output generator for base
    # We'll use a constant node to simulate 100 frames of 13 features
    node_base = onnx.helper.make_node("Constant", [], ["Y"], 
                                     value=onnx.helper.make_tensor("value", onnx.TensorProto.FLOAT, [1, 100, 13], [0.5]*100*13))
    graph_base = onnx.helper.make_graph([node_base], "base", 
                                        [onnx.helper.make_tensor_value_info("input_values", onnx.TensorProto.FLOAT, [1, 16000])],
                                        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 100, 13])])
    model_base = onnx.helper.make_model(graph_base)
    onnx.save(model_base, base_path)
    
    # 2. Create a mock VAD extractor (Identity for chunk probabilities)
    # chunk [512] -> prob [1, 1]
    vad_path = str(tmp_path / "vad.onnx")
    # axes as input for opset 18+
    axes_vad = onnx.helper.make_tensor("axes", onnx.TensorProto.INT64, [1], [1])
    node_vad = onnx.helper.make_node("ReduceMean", ["X", "axes"], ["Y"], keepdims=1)
    graph_vad = onnx.helper.make_graph([node_vad], "vad", 
                                       [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["B_vad", 512])],
                                       [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, ["B_vad", 1])],
                                       initializer=[axes_vad])
    model_vad = onnx.helper.make_model(graph_vad)
    onnx.save(model_vad, vad_path)
    
    # 3. Create a mock head (Accepts 13 + 1 = 14 inputs)
    head_path = str(tmp_path / "head.onnx")
    axes_head = onnx.helper.make_tensor("axes", onnx.TensorProto.INT64, [2], [1, 2])
    node_head = onnx.helper.make_node("ReduceMean", ["input_features", "axes"], ["logits"], keepdims=0)
    graph_head = onnx.helper.make_graph([node_head], "head", 
                                        [onnx.helper.make_tensor_value_info("input_features", onnx.TensorProto.FLOAT, [1, 100, 14])],
                                        [onnx.helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1])],
                                        initializer=[axes_head])
    model_head = onnx.helper.make_model(graph_head)
    onnx.save(model_head, head_path)
    
    # 4. Test Inferencer
    from ww_trainer.inference import OnnxWakeWordInferencer
    inferencer = OnnxWakeWordInferencer(
        extractor_path=base_path,
        head_path=head_path,
        vad_path=vad_path
    )
    
    dummy_audio = np.random.randn(16000).astype(np.float32)
    prob = inferencer.infer(dummy_audio)
    
    assert 0 <= prob <= 1.0
    print(f"Multi-ONNX infer successful: prob={prob}")

def test_embed_onnx_metadata(tmp_path):
    # ...

    # Create a dummy ONNX model
    onnx_path = str(tmp_path / "dummy.onnx")
    
    # Simple graph: identity
    node = onnx.helper.make_node("Identity", ["X"], ["Y"])
    graph = onnx.helper.make_graph([node], "test", 
                                   [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1])],
                                   [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])])
    model = onnx.helper.make_model(graph)
    onnx.save(model, onnx_path)
    
    metadata = {
        "test_key": "test_value",
        "wake_word": "hey_mycroft",
        "version": "1.0.0"
    }
    
    embed_onnx_metadata(onnx_path, metadata)
    
    # Load and verify
    loaded_model = onnx.load(onnx_path)
    props = {p.key: p.value for p in loaded_model.metadata_props}
    
    for k, v in metadata.items():
        assert props[k] == v

def test_markov_transition_extractor_onnx_parity(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    base = MfccExtractor(n_mfcc=13, sr=16000)
    ext = MarkovTransitionExtractor(base, n_codes=8, order=1)
    ext.to("cpu")
    ext.device = torch.device("cpu")
    base.to("cpu")
    base.device = torch.device("cpu")
    
    # Fit on some random data
    ext.fit([torch.randn(16000) for _ in range(2)])
    
    onnx_path = str(tmp_path / "markov.onnx")
    metadata = {"type": "markov_test"}
    
    ext.export_to_onnx(onnx_path, metadata=metadata)
    
    assert os.path.exists(onnx_path)
    
    # Verify metadata
    loaded_model = onnx.load(onnx_path)
    props = {p.key: p.value for p in loaded_model.metadata_props}
    assert props["type"] == "markov_test"
    
    # Parity check
    sess = ort.InferenceSession(onnx_path)
    # 1 second of audio
    dummy_wav = torch.randn(1, 16000)
    
    # Python forward
    with torch.no_grad():
        py_out = ext(dummy_wav)
    
    ort_out = sess.run(None, {"input_values": dummy_wav.numpy()})[0]
    
    # Parity check - allow small tolerance due to STFT/float differences
    np.testing.assert_allclose(py_out.numpy(), ort_out, atol=1e-3)

def test_hmm_state_extractor_onnx_parity(tmp_path):
    pytest.importorskip("markovonnx")
    base = MfccExtractor(n_mfcc=13, sr=16000)
    ext = HMMStateExtractor(base, n_states=4, n_codes=8)
    ext.to("cpu")
    ext.device = torch.device("cpu")
    base.to("cpu")
    base.device = torch.device("cpu")
    
    # Fit
    ext.fit([torch.randn(16000) for _ in range(2)], n_iter=1)
    
    onnx_path = str(tmp_path / "hmm.onnx")
    ext.export_to_onnx(onnx_path, metadata={"type": "hmm_test"})
    
    assert os.path.exists(onnx_path)
    
    # Parity check
    sess = ort.InferenceSession(onnx_path)
    dummy_wav = torch.randn(1, 16000)
    
    with torch.no_grad():
        py_out = ext(dummy_wav)
        
    ort_out = sess.run(None, {"input_values": dummy_wav.numpy()})[0]
    
    np.testing.assert_allclose(py_out.numpy(), ort_out, atol=1e-3)
