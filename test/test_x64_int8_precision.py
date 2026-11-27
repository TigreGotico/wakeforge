"""Featurizer sessions use exact int8 kernels on every x86 CPU (ww_trainer.ort_options).

The QDQ convolution below makes every pair of u8 x s8 products sum to 2 * 255 * 127, past int16: the default AVX2
kernel saturates and returns about half the true value (8.1 instead of 16), while VNNI and ARM CPUs, and the exact
path, return 16. The reference is the same graph at ORT_ENABLE_BASIC, run unfused in float.
"""
import platform

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from ww_trainer.ort_options import exact_int8

C_IN, C_OUT, T = 16, 4, 8


def _qdq_conv(path):
    inits = [
        numpy_helper.from_array(np.array(1.0 / 255, np.float32), "x_scale"),
        numpy_helper.from_array(np.array(0, np.uint8), "x_zp"),
        numpy_helper.from_array(np.full((C_OUT, C_IN, 1), 127, np.int8), "w_q"),
        numpy_helper.from_array(np.array(1.0 / 127, np.float32), "w_scale"),
        numpy_helper.from_array(np.array(0, np.int8), "w_zp"),
        numpy_helper.from_array(np.array(0.1, np.float32), "y_scale"),
        numpy_helper.from_array(np.array(0, np.uint8), "y_zp"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zp"], ["x_q"]),
        helper.make_node("DequantizeLinear", ["x_q", "x_scale", "x_zp"], ["x_dq"]),
        helper.make_node("DequantizeLinear", ["w_q", "w_scale", "w_zp"], ["w_dq"]),
        helper.make_node("Conv", ["x_dq", "w_dq"], ["y"], kernel_shape=[1]),
        helper.make_node("QuantizeLinear", ["y", "y_scale", "y_zp"], ["y_q"]),
        helper.make_node("DequantizeLinear", ["y_q", "y_scale", "y_zp"], ["out"]),
    ]
    graph = helper.make_graph(nodes, "qdq_conv", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C_IN, T])],
                              [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, C_OUT, T])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


def _run(path, options, level):
    options.graph_optimization_level = level
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]).run(
        None, {"x": np.ones((1, C_IN, T), np.float32)})[0]


def test_exact_int8_conv(tmp_path):
    path = tmp_path / "qdq_conv.onnx"
    _qdq_conv(path)
    exact = _run(path, ort.SessionOptions(), ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
    assert np.allclose(exact, C_IN)
    fused = _run(path, exact_int8(), ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    assert np.array_equal(fused, exact), f"{platform.machine()}: {fused.ravel()[:4]} instead of {exact.ravel()[:4]}"
