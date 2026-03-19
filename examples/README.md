# ww-trainer Examples

Runnable demos covering every architecture in ww-trainer.

## Feature Extractors × Classifier Heads

| # | Example | Extractor | Head | Focus |
|---|---------|-----------|------|-------|
| 01 | `01_mfcc_ffn_micro.py` | MFCC | FFN | Smallest model, ONNX export |
| 02 | `02_mfcc_gru_small.py` | MFCC | GRU | Temporal modeling, bidirectional |
| 03 | `03_mfcc_bcresnet.py` | MFCC / FilterBank | BC-ResNet | 2D conv, tau scaling |
| 04 | `04_filterbank_cnn.py` | FilterBank | CNN | Log-mel + 1D convolutions |
| 05 | `05_sincnet_gru.py` | SincNet | GRU | Learnable filterbank |
| 06 | `06_gammatone_ffn.py` | Gammatone | FFN | Auditory-inspired features |
| 07 | `07_delta_mfcc_ffn.py` | Delta-MFCC | FFN | Velocity + acceleration features |
| 14 | `14_hubert_training.py` | HuBERT / Wav2Vec2 | GRU | Transformer extractors (requires `transformers`) |

## Inference & Deployment

| # | Example | Focus |
|---|---------|-------|
| 08 | `08_onnx_inference.py` | Zero-PyTorch inference with onnxruntime |
| 09 | `09_streaming_inference.py` | Real-time chunk-by-chunk detection |

## Training & Research

| # | Example | Focus |
|---|---------|-------|
| 10 | `10_knowledge_distillation.py` | Compress large model → small student |
| 11 | `11_hardware_tiers.py` | Pre-configured tier presets |
| 12 | `12_hyperparameter_sweep.py` | Optuna auto-tuning (requires `optuna`) |
| 13 | `13_benchmark_extractors.py` | Latency and throughput comparison |

## Quick Start

```bash
cd ww-trainer

# Run any example
.venv/bin/python examples/01_mfcc_ffn_micro.py

# Run all examples (skip those requiring optional deps)
for f in examples/[0-9]*.py; do
    echo "=== $f ==="
    .venv/bin/python "$f" || echo "(skipped)"
    echo
done
```
