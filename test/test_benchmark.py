"""Tests for ww_trainer/benchmark.py"""
import pytest
import numpy as np

from ww_trainer.feats import MfccExtractor
from ww_trainer.model import FfnClassifierHead, GruClassifierHead
from ww_trainer.benchmark import (
    count_params,
    bench_extractor,
    bench_model,
    bench_onnx_vs_pytorch,
    run_benchmark,
    save_results,
    plot_results,
    BenchmarkReport,
    ExtractorBenchResult,
    ModelBenchResult,
    OnnxVsPytorchResult,
)


def test_count_params():
    model = MfccExtractor(n_mfcc=13)
    p = count_params(model)
    assert isinstance(p, int)
    assert p >= 0


def test_bench_extractor_basic():
    ext = MfccExtractor(sr=16000, n_mfcc=13, n_mels=40, n_fft=400, hop_length=160)
    result = bench_extractor(ext, sample_rate=16000, audio_duration_s=0.5,
                             n_warmup=1, n_runs=3, device="cpu")
    assert isinstance(result, ExtractorBenchResult)
    assert result.latency_ms_mean > 0
    assert result.feature_dim == 13
    assert len(result.output_shape) == 3


def test_bench_model_basic():
    ext = MfccExtractor(sr=16000, n_mfcc=13, n_mels=40, n_fft=400, hop_length=160)
    head = FfnClassifierHead(input_size=13, hidden_dim=16, device="cpu")
    result = bench_model("test_tier", ext, head, sample_rate=16000,
                         audio_duration_s=0.5, n_warmup=1, n_runs=3, device="cpu")
    assert isinstance(result, ModelBenchResult)
    assert result.latency_ms_mean > 0
    assert result.rtf > 0
    assert result.total_params == result.extractor_params + result.head_params


def test_run_benchmark_smoke(tmp_path):
    report = run_benchmark(
        output_dir=str(tmp_path),
        device="cpu",
        audio_duration_s=0.5,
        n_warmup=1,
        n_runs=3,
        include_tiers=["micro", "small"],
    )
    assert isinstance(report, BenchmarkReport)
    assert len(report.extractor_results) > 0
    assert len(report.model_results) > 0


def test_save_results(tmp_path):
    report = BenchmarkReport(
        extractor_results=[
            ExtractorBenchResult("MfccExtractor", 40, 1000, 5.0, 0.5, 8.0, (1, 100, 40), True)
        ],
        model_results=[
            ModelBenchResult("micro", "MfccExtractor", "FfnClassifierHead",
                             50000, 1000, 49000, 6.0, 0.5, 0.006)
        ],
    )
    save_results(report, tmp_path)
    assert (tmp_path / "extractor_benchmark.csv").exists()
    assert (tmp_path / "model_benchmark.csv").exists()


def test_bench_onnx_vs_pytorch(tmp_path):
    ext = MfccExtractor(sr=16000, n_mfcc=13, n_mels=40, n_fft=400, hop_length=160)
    onnx_path = str(tmp_path / "mfcc.onnx")
    ext.export_to_onnx(onnx_path)

    result = bench_onnx_vs_pytorch(ext, onnx_path, sample_rate=16000,
                                    audio_duration_s=0.5, n_warmup=1,
                                    n_runs=3, device="cpu")
    assert isinstance(result, OnnxVsPytorchResult)
    assert result.pytorch_latency_ms_mean > 0
    assert result.onnx_latency_ms_mean > 0
    assert result.speedup > 0
    assert result.max_abs_diff >= 0
    assert result.numerically_equivalent  # MFCC should be numerically identical


def test_run_benchmark_includes_onnx_comparison(tmp_path):
    report = run_benchmark(
        output_dir=str(tmp_path),
        device="cpu",
        audio_duration_s=0.5,
        n_warmup=1,
        n_runs=3,
        include_tiers=["micro"],
    )
    assert len(report.onnx_vs_pytorch) > 0
    # CSV should be created
    assert (tmp_path / "onnx_vs_pytorch.csv").exists()


def test_onnx_vs_pytorch_plots(tmp_path):
    report = BenchmarkReport(
        onnx_vs_pytorch=[
            OnnxVsPytorchResult("MfccExtractor(dim=13)", 5.0, 0.3, 2.5, 0.2,
                                 2.0, 1e-5, 5e-6, True, "/tmp/mfcc.onnx"),
            OnnxVsPytorchResult("FilterbankExtractor(dim=40)", 4.0, 0.2, 3.5, 0.3,
                                 1.14, 2e-4, 1e-4, True, "/tmp/fbank.onnx"),
        ]
    )
    plot_results(report, tmp_path)
    assert (tmp_path / "onnx_vs_pytorch_latency.png").exists()
    assert (tmp_path / "onnx_speedup.png").exists()
    assert (tmp_path / "onnx_numerical_accuracy.png").exists()


def test_plot_results(tmp_path):
    report = BenchmarkReport(
        extractor_results=[
            ExtractorBenchResult("MfccExtractor(dim=40)", 40, 1000, 5.0, 0.5, 8.0, (1, 100, 40), True),
            ExtractorBenchResult("MfccExtractor(dim=13)", 13, 500, 3.0, 0.3, 5.0, (1, 100, 13), True),
        ],
        model_results=[
            ModelBenchResult("micro", "MfccExtractor", "FfnClassifierHead",
                             50000, 1000, 49000, 6.0, 0.5, 0.006),
            ModelBenchResult("small", "MfccExtractor", "GruClassifierHead",
                             200000, 1000, 199000, 10.0, 0.8, 0.010),
        ],
    )
    plot_results(report, tmp_path)
    # At least some plots should have been created
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) > 0
