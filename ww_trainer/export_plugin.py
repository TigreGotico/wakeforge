"""Bundle one head, or an ensemble of heads, in the OVOS wake word plugin's model format.

The plugin loads a single ONNX file with input ``features`` [batch, frames, 128]
and one output ``logit_calibrated`` [batch]; the sigmoid of that output is the
probability. An ensemble averages the heads' logits before the calibration.
"""
import json
import sys
from pathlib import Path

import click
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, compose, helper, numpy_helper

INPUT_NAME = "features"
OUTPUT_NAME = "logit_calibrated"
FEATURE_DIM = 128
TOLERANCE = 1e-4
PLUGIN_FEATURIZERS = ("wakehubert", "wakehubert-int8")


def _head_io(model, path):
    graph = model.graph
    initializers = {i.name for i in graph.initializer}
    inputs = [i for i in graph.input if i.name not in initializers]
    if len(inputs) != 1:
        raise ValueError(f"{path}: expected one input, found {[i.name for i in inputs]}")
    dims = inputs[0].type.tensor_type.shape.dim
    if len(dims) != 3 or dims[2].dim_value != FEATURE_DIM:
        shape = [d.dim_value or d.dim_param for d in dims]
        raise ValueError(f"{path}: input must be [batch, frames, {FEATURE_DIM}], found {shape}")
    named = [o for o in graph.output if o.name == "logits"]
    if not named:
        raise ValueError(f"{path}: expected an output named logits, found "
                         f"{[o.name for o in graph.output]}")
    out = named[0]
    odims = out.type.tensor_type.shape.dim
    if len(odims) == 2 and odims[1].dim_value == 1:
        pass
    elif len(odims) != 1:
        raise ValueError(f"{path}: logits must be [batch] or [batch, 1]")
    return inputs[0].name, out.name


def _read_calibration(path):
    if path is None:
        return 1.0, 0.0
    data = json.loads(Path(path).read_text())
    return float(data["coef"]), float(data["intercept"])


def _merged_metadata(models):
    merged = {}
    conflicts = set()
    for m in models:
        for p in m.metadata_props:
            if p.key in merged and merged[p.key] != p.value:
                conflicts.add(p.key)
            merged.setdefault(p.key, p.value)
    return {k: v for k, v in merged.items() if k not in conflicts}


def build_model(heads, coef, intercept, metadata, window_frames):
    """Return the combined ModelProto for already loaded head models."""
    nodes, initializers, opsets = [], [], {}
    logit_names = []
    for i, head in enumerate(heads):
        in_name, out_name = _head_io(head, f"head {i}")
        prefix = f"h{i}/"
        head = compose.add_prefix(head, prefix)
        nodes.append(helper.make_node("Identity", [INPUT_NAME], [prefix + in_name],
                                      name=f"{prefix}share_input"))
        nodes.extend(head.graph.node)
        initializers.extend(head.graph.initializer)
        for op in head.opset_import:
            opsets[op.domain] = max(opsets.get(op.domain, 0), op.version)
        shape = f"{prefix}flat_shape"
        initializers.append(numpy_helper.from_array(np.array([-1], np.int64), shape))
        flat = f"{prefix}logit"
        nodes.append(helper.make_node("Reshape", [prefix + out_name, shape], [flat],
                                      name=f"{prefix}squeeze"))
        logit_names.append(flat)
    opsets[""] = max(opsets.get("", 0), 13)
    initializers.append(numpy_helper.from_array(np.array(coef, np.float32), "plugin/coef"))
    initializers.append(numpy_helper.from_array(np.array(intercept, np.float32), "plugin/intercept"))
    nodes += [
        helper.make_node("Mean", logit_names, ["plugin/mean"], name="plugin/mean"),
        helper.make_node("Mul", ["plugin/mean", "plugin/coef"], ["plugin/scaled"], name="plugin/scale"),
        helper.make_node("Add", ["plugin/scaled", "plugin/intercept"], [OUTPUT_NAME], name="plugin/shift"),
    ]
    graph = helper.make_graph(
        nodes, "wakeforge_plugin_head",
        [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, ["batch", "frames", FEATURE_DIM])],
        [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, ["batch"])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid(d, v) for d, v in opsets.items()])
    model.ir_version = max(h.ir_version for h in heads)
    for k, v in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = k, str(v)
    return model


def _head_logits(path, features):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.asarray(sess.run(None, {name: features})[0], np.float64).reshape(len(features))


def verify(model, head_paths, coef, intercept, window_frames):
    """Run the combined model against the heads computed one by one; raise on a mismatch."""
    onnx.checker.check_model(model)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for frames in (window_frames, max(1, window_frames // 2)):
        features = rng.normal(0, 1, (4, frames, FEATURE_DIM)).astype(np.float32)
        got = sess.run(None, {INPUT_NAME: features})[0]
        if got.shape != (4,):
            raise ValueError(f"combined output has shape {got.shape}, expected (4,)")
        mean = np.mean([_head_logits(p, features) for p in head_paths], axis=0)
        want = coef * mean + intercept
        diff = float(np.max(np.abs(got.astype(np.float64) - want) - TOLERANCE * np.abs(want)))
        if diff > TOLERANCE:
            raise ValueError(f"combined logit differs from the heads by {diff:.2e} "
                             f"beyond the tolerance (abs and rel {TOLERANCE:.0e}) at {frames} frames")


def export_plugin(head_paths, output, wake_word=None, license="Apache-2.0", training_data="",
                  threshold=None, calibration=None, window_frames=None):
    """Write the plugin-format model to ``output`` and return its path."""
    if not head_paths:
        raise ValueError("at least one head is required")
    if threshold is None or not 0.0 < float(threshold) < 1.0:
        raise ValueError(f"threshold must be a probability in (0, 1), got {threshold}")
    if not training_data:
        raise ValueError("training_data must name the data the heads were trained on")
    heads = [onnx.load(str(p)) for p in head_paths]
    for h, p in zip(heads, head_paths):
        _head_io(h, p)
    featurizers = {}
    for h, p in zip(heads, head_paths):
        found = {m.key: m.value for m in h.metadata_props}.get("pretrained_featurizer")
        if found not in PLUGIN_FEATURIZERS:
            raise ValueError(f"{p}: pretrained_featurizer is {found!r}; "
                             f"the plugin takes {' or '.join(PLUGIN_FEATURIZERS)}")
        featurizers[str(p)] = found
    if len(set(featurizers.values())) > 1:
        raise ValueError(f"heads disagree on pretrained_featurizer: {featurizers}")
    (featurizer,) = set(featurizers.values())
    declared = {{m.key: m.value for m in h.metadata_props}.get("window_frames") for h in heads} - {None}
    if len(declared) > 1:
        raise ValueError(f"heads disagree on window_frames: {sorted(declared)}")
    if declared:
        (head_window,) = declared
        if window_frames is not None and str(window_frames) != head_window:
            raise ValueError(f"--window-frames {window_frames} does not match the heads' {head_window}")
        window_frames = int(head_window)
    window_frames = window_frames or 75
    coef, intercept = _read_calibration(calibration)
    metadata = _merged_metadata(heads)
    wake_word = wake_word or metadata.get("wake_word", "").replace("_", " ")
    if not wake_word:
        raise ValueError("wake_word is not in the heads' metadata; pass it explicitly")
    metadata.update({
        "wake_word": wake_word,
        "pretrained_featurizer": featurizer,
        "window_frames": str(window_frames),
        "license": license,
        "training_data": training_data,
        "default_threshold": repr(float(threshold)),
    })
    model = build_model(heads, coef, intercept, metadata, window_frames)
    verify(model, head_paths, coef, intercept, window_frames)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    return output


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("heads", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--calibration", type=click.Path(exists=True, dir_okay=False),
              help="calibration.json with coef and intercept (default 1 and 0).")
@click.option("--threshold", required=True, type=float, help="Default trigger on the calibrated probability scale, in (0, 1). "
              "A threshold chosen on raw head scores must be mapped through the calibration first.")
@click.option("--wake-word", default=None, help="Spoken phrase; defaults to the heads' metadata.")
@click.option("--license", "license_", default="Apache-2.0", show_default=True)
@click.option("--training-data", required=True, help="Free text naming the training data.")
@click.option("--window-frames", default=None, type=int,
              help="Frames per window; taken from the heads when they declare it, else 75.")
@click.option("-o", "--output", required=True, type=click.Path(dir_okay=False))
def main(heads, calibration, threshold, wake_word, license_, training_data, window_frames, output):
    """Write HEADS (several heads form a mean-logit ensemble) as one plugin model."""
    try:
        path = export_plugin(list(heads), output, wake_word, license_, training_data,
                             threshold, calibration, window_frames)
    except ValueError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
