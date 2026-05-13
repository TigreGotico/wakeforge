# ww-trainer — Benchmarking Guide

How to measure feature extractor and model latency, and compare PyTorch vs ONNX Runtime performance.

---

## Overview

`ww_trainer/benchmark.py` provides:

- `bench_extractor` — latency, parameter count, output shape for a single extractor (`benchmark.py:70`)
- `bench_model` — end-to-end latency + real-time factor for extractor + head pairs (`benchmark.py:116`)
- `bench_onnx_vs_pytorch` — latency and numerical accuracy comparison between PyTorch and ONNX Runtime (`benchmark.py:166`)
- `run_benchmark` — runs the full suite, saves CSVs and plots (`benchmark.py:242`)

---

## Quick Start

```python
from ww_trainer.benchmark import run_benchmark

report = run_benchmark(output_dir="benchmark_results/", device="cpu")
```

Or from the CLI:

```bash
uv run python -m ww_trainer.benchmark --output-dir benchmark_results/ --device cpu --runs 50
```

---

## Result Dataclasses

### `ExtractorBenchResult`

Fields: `name`, `feature_dim`, `params`, `latency_ms_mean`, `latency_ms_std`, `latency_ms_p99`, `output_shape`, `onnx_exportable`.

Defined at `benchmark.py:28`.

### `ModelBenchResult`

Fields: `tier`, `extractor_name`, `head_arch`, `total_params`, `extractor_params`, `head_params`, `latency_ms_mean`, `latency_ms_std`, `rtf`.

The `rtf` (real-time factor) is `latency_ms / (audio_duration_s * 1000)`. Values below 0.1 are suitable for always-on wakeword detection.

Defined at `benchmark.py:41`.

### `OnnxVsPytorchResult`

Fields: `name`, `pytorch_latency_ms_mean`, `pytorch_latency_ms_std`, `onnx_latency_ms_mean`, `onnx_latency_ms_std`, `speedup`, `max_abs_diff`, `mean_abs_diff`, `numerically_equivalent`, `onnx_path`.

Defined at `benchmark.py:56`.

---

## ONNX vs PyTorch Comparison

### What it measures

`bench_onnx_vs_pytorch` (`benchmark.py:166`) takes a PyTorch extractor and the path to its ONNX export. It:

1. Runs `n_warmup` warmup passes through both backends.
2. Times `n_runs` inference passes through PyTorch (`torch.no_grad`) and ONNX Runtime separately.
3. Runs one additional inference pass through both and computes the absolute element-wise difference between outputs.

This measures two things independently: **latency** and **numerical accuracy**.

### How to interpret the speedup number

```
speedup = pytorch_latency_ms_mean / onnx_latency_ms_mean
```

- `speedup > 1.0` — ONNX Runtime is faster than PyTorch on this extractor.
- `speedup < 1.0` — PyTorch is faster (uncommon for small models; possible on GPU with batch sizes > 1).
- `speedup == 1.0` — identical performance.

For `MfccExtractor` and `FilterbankExtractor` on CPU, expected speedups are typically in the range of **1.2x to 3x** because ONNX Runtime applies constant folding and graph-level optimisations (e.g. fusing matmuls) that PyTorch's eager mode does not apply at runtime.

For `SincNetExtractor`, speedup is lower (often close to 1x) because the bottleneck is the learned conv1d layer, which PyTorch's native MKLDNN backend also optimises well.

### Numerical equivalence

```
numerically_equivalent = max_abs_diff < 1e-3
```

`MfccExtractor` and `FilterbankExtractor` are expected to be numerically equivalent (`max_abs_diff` on the order of 1e-6 to 1e-7) because they use only standard floating-point ops (matmul, log, stft). Any difference is due to floating-point ordering in fused ops.

`SincNetExtractor` may have slightly larger diffs (1e-5 to 1e-4) due to the sinc filter normalisation involving `abs().sum()`, which can accumulate small ordering differences during ONNX constant folding.

A `max_abs_diff` above 1e-3 should be investigated — it may indicate a shape mismatch, an unsupported op falling back to a different kernel, or a dynamic axis issue.

### Running the comparison programmatically

```python
import tempfile
from ww_trainer.feats import MfccExtractor
from ww_trainer.benchmark import bench_onnx_vs_pytorch

ext = MfccExtractor(sr=16000, n_mfcc=40)
with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
    onnx_path = f.name

ext.export_to_onnx(onnx_path)

result = bench_onnx_vs_pytorch(
    ext, onnx_path,
    sample_rate=16000,
    audio_duration_s=1.0,
    n_warmup=5,
    n_runs=50,
    device="cpu",
)

print(f"Speedup: {result.speedup:.2f}x")
print(f"Max abs diff: {result.max_abs_diff:.2e}")
print(f"Numerically equivalent: {result.numerically_equivalent}")
```

---

## Output Files

After `run_benchmark()` or `save_results()` + `plot_results()`:

| File | Contents |
| :--- | :--- |
| `extractor_benchmark.csv` | Per-extractor latency and params |
| `model_benchmark.csv` | Per-tier end-to-end latency and RTF |
| `onnx_vs_pytorch.csv` | ONNX vs PyTorch latency and numerical diff per extractor |
| `extractor_latency.png` | Bar chart: extractor latency |
| `extractor_params.png` | Bar chart: extractor parameter counts |
| `latency_vs_params.png` | Scatter: latency vs model size |
| `rtf_comparison.png` | Bar chart: real-time factor by tier |
| `benchmark_summary.png` | Summary table |
| `onnx_vs_pytorch_latency.png` | Grouped bar: PyTorch vs ONNX latency with speedup annotation |
| `onnx_speedup.png` | Bar chart: ONNX speedup ratio per extractor |
| `onnx_numerical_accuracy.png` | Bar chart: max absolute diff (log scale) |

---

## BenchmarkReport

`BenchmarkReport` (`benchmark.py:67`) holds all results:

```python
@dataclass
class BenchmarkReport:
    extractor_results: list[ExtractorBenchResult]
    model_results: list[ModelBenchResult]
    onnx_vs_pytorch: list[OnnxVsPytorchResult]
    device: str
    audio_duration_s: float
    n_warmup: int
    n_runs: int
```

All three result lists are populated by `run_benchmark()` and can be inspected programmatically after the call returns.
