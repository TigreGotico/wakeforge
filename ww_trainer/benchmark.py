"""Benchmarking utility for ww-trainer.

Measures extractor latency, model size, and accuracy across hardware tiers.
Generates publication-ready plots saved to an output directory.

Usage:
    python -m ww_trainer.benchmark --output-dir benchmark_results/
    python -m ww_trainer.benchmark --metadata dataset.csv --output-dir results/

    # Programmatic:
    from ww_trainer.benchmark import run_benchmark
    results = run_benchmark(output_dir="results/")
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ExtractorBenchResult:
    name: str
    feature_dim: int
    params: int
    latency_ms_mean: float
    latency_ms_std: float
    latency_ms_p99: float
    output_shape: tuple
    onnx_exportable: bool
    notes: str = ""


@dataclass
class ModelBenchResult:
    tier: str
    extractor_name: str
    head_arch: str
    total_params: int
    extractor_params: int
    head_params: int
    latency_ms_mean: float   # end-to-end: extractor + head
    latency_ms_std: float
    rtf: float               # real-time factor: latency / audio_duration
    notes: str = ""


@dataclass
class OnnxVsPytorchResult:
    name: str                          # e.g. "MfccExtractor(dim=40)"
    pytorch_latency_ms_mean: float
    pytorch_latency_ms_std: float
    onnx_latency_ms_mean: float
    onnx_latency_ms_std: float
    speedup: float                     # pytorch / onnx — >1 means ONNX faster
    max_abs_diff: float                # max absolute numerical difference
    mean_abs_diff: float               # mean absolute numerical difference
    numerically_equivalent: bool       # True if max_abs_diff < 1e-3
    onnx_path: str


@dataclass
class BenchmarkReport:
    extractor_results: list[ExtractorBenchResult] = field(default_factory=list)
    model_results: list[ModelBenchResult] = field(default_factory=list)
    onnx_vs_pytorch: list[OnnxVsPytorchResult] = field(default_factory=list)
    device: str = "cpu"
    audio_duration_s: float = 1.0
    n_warmup: int = 5
    n_runs: int = 50


def count_params(module: torch.nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def bench_extractor(
    extractor,
    sample_rate: int = 16000,
    audio_duration_s: float = 1.0,
    n_warmup: int = 5,
    n_runs: int = 50,
    device: str = "cpu",
) -> ExtractorBenchResult:
    """Benchmark a single extractor: latency, params, output shape."""
    extractor = extractor.to(device)
    extractor.eval()

    n_samples = int(sample_rate * audio_duration_s)
    dummy = torch.zeros(1, n_samples, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = extractor(dummy)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            out = extractor(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    params = count_params(extractor)
    onnx_ok = hasattr(extractor, 'export_to_onnx')

    return ExtractorBenchResult(
        name=type(extractor).__name__,
        feature_dim=extractor.feature_dim,
        params=params,
        latency_ms_mean=float(np.mean(times)),
        latency_ms_std=float(np.std(times)),
        latency_ms_p99=float(np.percentile(times, 99)),
        output_shape=tuple(out.shape),
        onnx_exportable=onnx_ok,
    )


def bench_model(
    tier_name: str,
    extractor,
    head,
    sample_rate: int = 16000,
    audio_duration_s: float = 1.0,
    n_warmup: int = 5,
    n_runs: int = 50,
    device: str = "cpu",
) -> ModelBenchResult:
    """Benchmark end-to-end extractor + head latency."""
    extractor = extractor.to(device).eval()
    head = head.to(device).eval()

    n_samples = int(sample_rate * audio_duration_s)
    dummy_wav = torch.zeros(1, n_samples, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            feats = extractor(dummy_wav)
            _ = head(feats)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            feats = extractor(dummy_wav)
            _ = head(feats)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    rtf = float(np.mean(times)) / (audio_duration_s * 1000)

    return ModelBenchResult(
        tier=tier_name,
        extractor_name=type(extractor).__name__,
        head_arch=type(head).__name__,
        total_params=count_params(extractor) + count_params(head),
        extractor_params=count_params(extractor),
        head_params=count_params(head),
        latency_ms_mean=float(np.mean(times)),
        latency_ms_std=float(np.std(times)),
        rtf=rtf,
    )


def bench_onnx_vs_pytorch(
    extractor,
    onnx_path: str,
    sample_rate: int = 16000,
    audio_duration_s: float = 1.0,
    n_warmup: int = 5,
    n_runs: int = 50,
    device: str = "cpu",
) -> OnnxVsPytorchResult:
    """Compare PyTorch extractor vs its ONNX export for latency and accuracy.

    Args:
        extractor: The PyTorch extractor (already exported to onnx_path).
        onnx_path: Path to the ONNX file.
        sample_rate: Audio sample rate.
        audio_duration_s: Duration of test audio.
        n_warmup: Warmup runs.
        n_runs: Timed runs.
        device: 'cpu' or 'cuda'.

    Returns:
        OnnxVsPytorchResult with latency comparison and numerical diff.
    """
    import onnxruntime as ort

    extractor = extractor.to(device).eval()
    n_samples = int(sample_rate * audio_duration_s)

    dummy_torch = torch.zeros(1, n_samples, device=device)
    dummy_np = np.zeros((1, n_samples), dtype=np.float32)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" \
                else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Warmup both
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = extractor(dummy_torch)
    for _ in range(n_warmup):
        _ = sess.run([output_name], {input_name: dummy_np})

    # Time PyTorch
    pt_times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            pt_out = extractor(dummy_torch)
            pt_times.append((time.perf_counter() - t0) * 1000)

    # Time ONNX
    ort_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        ort_out = sess.run([output_name], {input_name: dummy_np})[0]
        ort_times.append((time.perf_counter() - t0) * 1000)

    # Numerical comparison (single run)
    with torch.no_grad():
        pt_ref = extractor(dummy_torch).cpu().numpy()
    ort_ref = sess.run([output_name], {input_name: dummy_np})[0]

    diff = np.abs(pt_ref - ort_ref)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    pt_arr = np.array(pt_times)
    ort_arr = np.array(ort_times)
    speedup = float(np.mean(pt_arr) / np.mean(ort_arr))

    name = f"{type(extractor).__name__}(dim={extractor.feature_dim})"

    return OnnxVsPytorchResult(
        name=name,
        pytorch_latency_ms_mean=float(np.mean(pt_arr)),
        pytorch_latency_ms_std=float(np.std(pt_arr)),
        onnx_latency_ms_mean=float(np.mean(ort_arr)),
        onnx_latency_ms_std=float(np.std(ort_arr)),
        speedup=speedup,
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        numerically_equivalent=max_diff < 1e-3,
        onnx_path=onnx_path,
    )


def run_benchmark(
    output_dir: str = "benchmark_results",
    device: str = "cpu",
    sample_rate: int = 16000,
    audio_duration_s: float = 1.0,
    n_warmup: int = 5,
    n_runs: int = 50,
    include_tiers: Optional[list[str]] = None,
) -> BenchmarkReport:
    """Run the full benchmark suite and save results + plots.

    Args:
        output_dir: Directory to save plots and CSV results.
        device: 'cpu' or 'cuda'.
        sample_rate: Audio sample rate for dummy input.
        audio_duration_s: Duration of dummy audio clip.
        n_warmup: Warmup runs before timing.
        n_runs: Timed runs for statistics.
        include_tiers: List of tier names to benchmark. None = all local tiers
                       (skips 'medium' and 'large' which require ONNX/transformers).

    Returns:
        BenchmarkReport with all results.
    """
    from ww_trainer.feats import MfccExtractor
    from ww_trainer.model import FfnClassifierHead, GruClassifierHead, CnnClassifierHead

    # Try to import optional extractors
    extractors_to_bench = []

    # Always available
    extractors_to_bench.append(MfccExtractor(sr=sample_rate, n_mfcc=40))
    extractors_to_bench.append(MfccExtractor(sr=sample_rate, n_mfcc=13))

    try:
        from ww_trainer.feats import FilterbankExtractor
        extractors_to_bench.append(FilterbankExtractor(sr=sample_rate, n_mels=80))
        extractors_to_bench.append(FilterbankExtractor(sr=sample_rate, n_mels=40))
    except ImportError:
        pass

    try:
        from ww_trainer.feats import SincNetExtractor
        extractors_to_bench.append(SincNetExtractor(sr=sample_rate, n_filters=80))
    except ImportError:
        pass

    report = BenchmarkReport(device=device, audio_duration_s=audio_duration_s,
                             n_warmup=n_warmup, n_runs=n_runs)

    # Bench individual extractors
    logger.info("Benchmarking extractors...")
    for ext in extractors_to_bench:
        name = f"{type(ext).__name__}(dim={ext.feature_dim})"
        logger.info("  %s ...", name)
        try:
            result = bench_extractor(ext, sample_rate, audio_duration_s, n_warmup, n_runs, device)
            result.name = name
            report.extractor_results.append(result)
        except Exception as e:
            logger.warning("  FAILED: %s", e)

    # ONNX vs PyTorch comparison
    logger.info("Running ONNX vs PyTorch comparison...")
    import tempfile
    import os
    for ext in extractors_to_bench:
        name = f"{type(ext).__name__}(dim={ext.feature_dim})"
        logger.info("  Comparing %s ...", name)
        onnx_tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                onnx_tmp = f.name
            ext.export_to_onnx(onnx_tmp)
            result = bench_onnx_vs_pytorch(
                ext, onnx_tmp, sample_rate, audio_duration_s, n_warmup, n_runs, device
            )
            report.onnx_vs_pytorch.append(result)
            os.unlink(onnx_tmp)
        except Exception as e:
            logger.warning("  Comparison FAILED for %s: %s", name, e)
            if onnx_tmp is not None:
                try:
                    os.unlink(onnx_tmp)
                except Exception:
                    pass

    # Bench tier combinations (micro and small only — no ONNX/transformers needed)
    tiers_to_run = include_tiers or ["micro", "small", "filterbank_small", "sincnet_small"]

    tier_configs = {
        "micro": (MfccExtractor(sr=sample_rate, n_mfcc=40),
                  FfnClassifierHead(input_size=40, hidden_dim=128, device=device)),
        "small": (MfccExtractor(sr=sample_rate, n_mfcc=40),
                  GruClassifierHead(input_size=40, hidden_dim=128, device=device)),
    }

    # Add filterbank tiers if available
    try:
        from ww_trainer.feats import FilterbankExtractor
        tier_configs["filterbank_small"] = (
            FilterbankExtractor(sr=sample_rate, n_mels=80),
            GruClassifierHead(input_size=80, hidden_dim=128, device=device),
        )
        tier_configs["filterbank_cnn"] = (
            FilterbankExtractor(sr=sample_rate, n_mels=80),
            CnnClassifierHead(input_size=80, conv_dim=128, device=device),
        )
    except ImportError:
        pass

    try:
        from ww_trainer.feats import SincNetExtractor
        tier_configs["sincnet_small"] = (
            SincNetExtractor(sr=sample_rate, n_filters=80),
            GruClassifierHead(input_size=80, hidden_dim=128, device=device),
        )
    except ImportError:
        pass

    logger.info("Benchmarking tier models...")
    for tier_name, (ext, head) in tier_configs.items():
        if tiers_to_run and tier_name not in tiers_to_run:
            continue
        logger.info("  %s ...", tier_name)
        try:
            result = bench_model(tier_name, ext, head, sample_rate, audio_duration_s,
                                 n_warmup, n_runs, device)
            report.model_results.append(result)
        except Exception as e:
            logger.warning("  FAILED: %s", e)

    # Save results and plots
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_results(report, out_dir)
    plot_results(report, out_dir)

    logger.info("Benchmark complete. Results saved to %s", output_dir)
    return report


def save_results(report: BenchmarkReport, out_dir: Path) -> None:
    """Save benchmark results to CSV files."""
    import csv

    # Extractor results
    ext_path = out_dir / "extractor_benchmark.csv"
    with open(ext_path, "w", newline="") as f:
        if report.extractor_results:
            writer = csv.DictWriter(f, fieldnames=vars(report.extractor_results[0]).keys())
            writer.writeheader()
            for r in report.extractor_results:
                writer.writerow(vars(r))

    # Model results
    model_path = out_dir / "model_benchmark.csv"
    with open(model_path, "w", newline="") as f:
        if report.model_results:
            writer = csv.DictWriter(f, fieldnames=vars(report.model_results[0]).keys())
            writer.writeheader()
            for r in report.model_results:
                writer.writerow(vars(r))

    # ONNX vs PyTorch results
    if report.onnx_vs_pytorch:
        ovp_path = out_dir / "onnx_vs_pytorch.csv"
        with open(ovp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=vars(report.onnx_vs_pytorch[0]).keys())
            writer.writeheader()
            for r in report.onnx_vs_pytorch:
                writer.writerow(vars(r))

    logger.info("Results saved: %s, %s", ext_path, model_path)


def plot_results(report: BenchmarkReport, out_dir: Path) -> None:
    """Generate and save benchmark plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plots")
        return

    _plot_extractor_latency(report, out_dir)
    _plot_extractor_params(report, out_dir)
    _plot_model_latency_vs_params(report, out_dir)
    _plot_rtf_comparison(report, out_dir)
    _plot_summary_table(report, out_dir)
    _plot_onnx_vs_pytorch_latency(report, out_dir)
    _plot_speedup_bars(report, out_dir)
    _plot_numerical_accuracy(report, out_dir)


def _plot_extractor_latency(report: BenchmarkReport, out_dir: Path) -> None:
    """Bar chart: extractor latency mean +/- std."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not report.extractor_results:
        return

    names = [r.name for r in report.extractor_results]
    means = [r.latency_ms_mean for r in report.extractor_results]
    stds = [r.latency_ms_std for r in report.extractor_results]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4,
                  color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Feature Extractor Latency ({report.audio_duration_s}s audio, {report.device})")
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "extractor_latency.png", dpi=150)
    plt.close()


def _plot_extractor_params(report: BenchmarkReport, out_dir: Path) -> None:
    """Bar chart: extractor parameter counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not report.extractor_results:
        return

    names = [r.name for r in report.extractor_results]
    params = [r.params for r in report.extractor_results]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    bars = ax.bar(range(len(names)), params, color="coral", alpha=0.8, edgecolor="black")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Parameters")
    ax.set_title("Feature Extractor Parameter Count")
    ax.grid(axis="y", alpha=0.3)

    for bar, p in zip(bars, params):
        label = f"{p/1e6:.1f}M" if p >= 1e6 else (f"{p/1e3:.1f}K" if p >= 1000 else str(p))
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(params) * 0.01,
                label, ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "extractor_params.png", dpi=150)
    plt.close()


def _plot_model_latency_vs_params(report: BenchmarkReport, out_dir: Path) -> None:
    """Scatter plot: model latency vs total parameter count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not report.model_results:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for r in report.model_results:
        ax.scatter(r.total_params, r.latency_ms_mean, s=100, zorder=5)
        ax.annotate(r.tier, (r.total_params, r.latency_ms_mean),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel("Total Parameters")
    ax.set_ylabel("End-to-End Latency (ms)")
    ax.set_title(f"Latency vs Model Size ({report.device})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "latency_vs_params.png", dpi=150)
    plt.close()


def _plot_rtf_comparison(report: BenchmarkReport, out_dir: Path) -> None:
    """Bar chart: real-time factor for each tier model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not report.model_results:
        return

    names = [r.tier for r in report.model_results]
    rtfs = [r.rtf for r in report.model_results]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    colors = ["green" if r < 0.1 else ("orange" if r < 0.5 else "red") for r in rtfs]
    bars = ax.bar(range(len(names)), rtfs, color=colors, alpha=0.8, edgecolor="black")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="RTF = 1.0 (real-time limit)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Real-Time Factor (lower is better)")
    ax.set_title(f"Real-Time Factor by Tier ({report.device})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar, rtf in zip(bars, rtfs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(rtfs) * 0.01,
                f"{rtf:.3f}x", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "rtf_comparison.png", dpi=150)
    plt.close()


def _plot_summary_table(report: BenchmarkReport, out_dir: Path) -> None:
    """Render a summary table as a PNG using matplotlib table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not report.model_results:
        return

    cols = ["Tier", "Extractor", "Head", "Params", "Latency (ms)", "RTF"]
    rows = []
    for r in report.model_results:
        params_str = f"{r.total_params/1e6:.2f}M" if r.total_params >= 1e6 else f"{r.total_params/1e3:.1f}K"
        rows.append([
            r.tier,
            r.extractor_name,
            r.head_arch,
            params_str,
            f"{r.latency_ms_mean:.1f} +/- {r.latency_ms_std:.1f}",
            f"{r.rtf:.4f}x",
        ])

    fig, ax = plt.subplots(figsize=(12, max(3, len(rows) * 0.6 + 1)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Header styling
    for j in range(len(cols)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row colors
    for i in range(1, len(rows) + 1):
        color = "#EBF3FB" if i % 2 == 0 else "white"
        for j in range(len(cols)):
            table[i, j].set_facecolor(color)

    plt.title(f"ww-trainer Benchmark Summary ({report.device}, {report.audio_duration_s}s audio)",
              fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_onnx_vs_pytorch_latency(report: BenchmarkReport, out_dir: Path) -> None:
    """Grouped bar chart: PyTorch vs ONNX latency for each extractor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not report.onnx_vs_pytorch:
        return

    names = [r.name for r in report.onnx_vs_pytorch]
    pt_means = [r.pytorch_latency_ms_mean for r in report.onnx_vs_pytorch]
    pt_stds = [r.pytorch_latency_ms_std for r in report.onnx_vs_pytorch]
    ort_means = [r.onnx_latency_ms_mean for r in report.onnx_vs_pytorch]
    ort_stds = [r.onnx_latency_ms_std for r in report.onnx_vs_pytorch]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 6))
    bars1 = ax.bar(x - width/2, pt_means, width, yerr=pt_stds, capsize=4,
                   label='PyTorch', color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, ort_means, width, yerr=ort_stds, capsize=4,
                   label='ONNX Runtime', color='coral', alpha=0.8, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'PyTorch vs ONNX Runtime Latency ({report.audio_duration_s}s audio, {report.device})')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Annotate speedup
    for i, r in enumerate(report.onnx_vs_pytorch):
        color = 'green' if r.speedup > 1 else 'red'
        ax.text(x[i], max(pt_means[i], ort_means[i]) + max(pt_stds[i], ort_stds[i]) + 0.5,
                f'{r.speedup:.2f}x', ha='center', fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_dir / 'onnx_vs_pytorch_latency.png', dpi=150)
    plt.close()


def _plot_speedup_bars(report: BenchmarkReport, out_dir: Path) -> None:
    """Bar chart: ONNX speedup over PyTorch (>1 = ONNX faster)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not report.onnx_vs_pytorch:
        return

    names = [r.name for r in report.onnx_vs_pytorch]
    speedups = [r.speedup for r in report.onnx_vs_pytorch]
    numerical_ok = [r.numerically_equivalent for r in report.onnx_vs_pytorch]

    colors = ['green' if s > 1 else 'red' for s in speedups]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    bars = ax.bar(range(len(names)), speedups, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='1x (equal speed)')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Speedup (PyTorch latency / ONNX latency)')
    ax.set_title('ONNX Runtime Speedup vs PyTorch (higher = faster ONNX)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for i, (bar, s, ok) in enumerate(zip(bars, speedups, numerical_ok)):
        label = f'{s:.2f}x'
        if not ok:
            label += '\n[diff]'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                label, ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / 'onnx_speedup.png', dpi=150)
    plt.close()


def _plot_numerical_accuracy(report: BenchmarkReport, out_dir: Path) -> None:
    """Bar chart: max absolute difference between PyTorch and ONNX outputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not report.onnx_vs_pytorch:
        return

    names = [r.name for r in report.onnx_vs_pytorch]
    max_diffs = [r.max_abs_diff for r in report.onnx_vs_pytorch]

    colors = ['green' if r.numerically_equivalent else 'orange'
              for r in report.onnx_vs_pytorch]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    bars = ax.bar(range(len(names)), max_diffs, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(1e-3, color='red', linestyle='--', alpha=0.7, label='1e-3 threshold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Max Absolute Difference')
    ax.set_title('ONNX vs PyTorch Numerical Accuracy (lower = more accurate)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'onnx_numerical_accuracy.png', dpi=150)
    plt.close()


def main() -> None:
    """CLI entrypoint for the benchmark suite."""
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="ww-trainer benchmark suite")
    parser.add_argument("--output-dir", default="benchmark_results",
                        help="Directory to save results and plots")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--duration", type=float, default=1.0,
                        help="Audio duration in seconds for dummy input")
    parser.add_argument("--runs", type=int, default=50,
                        help="Number of timed runs per benchmark")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup runs before timing")
    parser.add_argument("--tiers", nargs="+", default=None,
                        help="Tier names to benchmark (default: all local tiers)")
    args = parser.parse_args()

    run_benchmark(
        output_dir=args.output_dir,
        device=args.device,
        audio_duration_s=args.duration,
        n_warmup=args.warmup,
        n_runs=args.runs,
        include_tiers=args.tiers,
    )


if __name__ == "__main__":
    main()
