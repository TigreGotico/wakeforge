# ww-trainer Documentation

Training and research suite for wake word detection models -- from microcontrollers to servers. Every extractor exports to ONNX; production inference requires only `onnxruntime` and `numpy`.

Version: `0.0.1a1` -- `ww_trainer/version.py`

---

## Component Counts

| Category | Count | Source |
|----------|-------|--------|
| Feature extractors (standalone) | 12 | `feats.py` |
| Feature enrichment wrappers | 5 | `feats.py:1318-1700` |
| Classifier heads | 11 | `model.py` |
| Loss functions | 17 | `loss.py` |
| Search strategies | 4 | `sweep.py` |
| Hardware tier presets | 8 | `tiers.py` |
| Examples | 28 | `examples/` |

---

## Documentation

### Core Architecture

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | System design: abstractions, data flow, ONNX pipeline, sliding cache, design decisions |
| [api.md](api.md) | Full API reference -- every public class and method with `file:line` citations |
| [data_contract.md](data_contract.md) | Dataset format, CSV contract, notebook pipeline, augmentation data sources |

### Components

| Document | Description |
|----------|-------------|
| [extractors.md](extractors.md) | All 17 feature extractors (12 standalone + 5 wrappers): parameters, when to use, hardware fit |
| [classifiers.md](classifiers.md) | All 11 classifier heads: architecture details, param counts, hardware recommendations |
| [losses.md](losses.md) | All 17 loss functions: classification, metric, contrastive, composite. Recommended combinations |
| [enrichment.md](enrichment.md) | Feature enrichment wrappers: VAD, Pitch, MultiResolution, SNRAware. Stacking patterns |

### Training & Deployment

| Document | Description |
|----------|-------------|
| [training.md](training.md) | Step-by-step training guide: dataset prep, tier selection, full CLI reference, augmentation |
| [export.md](export.md) | ONNX export: extractors, heads, quantization, verification |
| [inference.md](inference.md) | Single, batch, and streaming inference -- ONNX and PyTorch paths |
| [streaming.md](streaming.md) | Streaming inference deep-dive: `SlidingFeatureCacheTensor`, PyTorch vs ONNX paths |
| [distillation.md](distillation.md) | Knowledge distillation: `CnnLstmExtractor`, teacher-student workflow |
| [hardware_guide.md](hardware_guide.md) | Hardware-specific configurations: MCU, RPi, laptop, server |

### Optimization & Research

| Document | Description |
|----------|-------------|
| [sweep.md](sweep.md) | Hyperparameter sweeps: Optuna, Grid, Random, Genetic strategies |
| [search_strategies.md](search_strategies.md) | Decision tree for choosing a search strategy. Full architecture search with `full=True` |
| [benchmarking.md](benchmarking.md) | Latency measurement, ONNX vs PyTorch comparison, RTF analysis |
| [recipes.md](recipes.md) | End-to-end recipes: smallest model, best accuracy, noisy env, auto-search, RPi deploy |

---

## Quick Start

```bash
# Install
uv pip install ww_trainer

# Train a micro model (MFCC + FFN, ~50K params, no GPU needed)
ww_trainer-train \
  --wake-word hey_jarvis \
  --metadata dataset.csv \
  --tier micro \
  --output-dir ./models/hey_jarvis

# List available hardware tiers
ww_trainer-train --list-tiers
```

---

## Key Links

- GitHub: https://github.com/TigreGotico/ww_trainer
- Pre-exported MFCC ONNX: https://huggingface.co/TigreGotico/mfcc-onnx
- Examples: [examples/README.md](../examples/README.md)
- Funded by NGI0 Commons Fund / NLnet
