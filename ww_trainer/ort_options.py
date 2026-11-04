"""onnxruntime session options for the featurizers and heads wakeforge trains and runs.

An int8 featurizer (QuantizeLinear/DequantizeLinear around its convolutions, as ``wakehubert-int8``) is fused into
``QLinearConv`` from ``ORT_ENABLE_EXTENDED`` on. On x86 CPUs without AVX-512 VNNI (AVX2: most desktops, laptops and
cloud VMs before 2020) onnxruntime's default u8s8 kernel sums pairs of products in 16 bits and saturates, so the
frames differ by several percent from a VNNI or ARM CPU's: a model trained on one kind of machine then scores up to
~0.1 off on the other. ``session.x64quantprecision`` selects the exact 32-bit path on those CPUs; it keeps int8 speed
and changes nothing on VNNI or ARM CPUs, nor for float models.
"""
from typing import Optional

import onnxruntime as ort


def exact_int8(options: Optional[ort.SessionOptions] = None) -> ort.SessionOptions:
    """``options`` (new ones by default) with exact int8 kernels on every x86 CPU."""
    options = options if options is not None else ort.SessionOptions()
    options.add_session_config_entry("session.x64quantprecision", "1")
    return options
