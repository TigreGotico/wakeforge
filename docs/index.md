# ww-trainer Documentation

Training and research suite for wake word detection — from microcontrollers to GPU servers. Every extractor exports to ONNX; production inference requires only `onnxruntime` and `numpy`.

Version: `0.0.1a1` — `ww_trainer/version.py`

---

## Module Reference

| Module | Description |
|--------|-------------|
| `ww_trainer.feats` | Feature extractors (12 standalone + 5 wrappers) and `SlidingFeatureCacheTensor` |
| `ww_trainer.model` | Classifier heads and `BaseWakeModel` |
| `ww_trainer.loss` | 17 loss functions via `LossManager` |
| `ww_trainer.sweep` | Optuna, Grid, Random, Genetic, Two-Stage Genetic search |
| `ww_trainer.quickstart` | `train_from_wakeword` — string → ONNX in one call |
| `ww_trainer.trainer` | `WakeWordTrainer` training loop |
| `ww_trainer.dataset` | `AudioDataset` with augmentation and feature caching |
| `ww_trainer.inference` | `OnnxWakeWordInferencer` — PyTorch-free inference |
| `ww_trainer.benchmark` | Latency and RTF measurement |
| `ww_trainer.tiers` | 11 hardware tier presets |
| `ww_trainer.datagen` | Synthetic dataset generation pipeline |
| `ww_trainer.export_c` | C header export for ESP32 FFN models |
| `ww_trainer.calibration` | Platt scaling confidence calibration |
| `ww_trainer.qat` | Quantization-aware training |
| `ww_trainer.ddp` | Multi-GPU (DDP) utilities |
| `ww_trainer.visualization` | PCA, t-SNE, ROC/PR/DET plot helpers |
| `ww_trainer.evaluation` | `compute_fitness_score`, `compute_readiness` |
| `ww_trainer.checkpoint` | `save_checkpoint`, `load_checkpoint` |
| `ww_trainer.cache` | `FeatureCache` — MD5-keyed feature caching |
| `ww_trainer.mining` | `mine_hard_negatives` |
| `ww_trainer.loop` | `training_loop` (extracted from `WakeWordTrainer`) |

---

## Notebook Reference

| Notebook | Description |
|----------|-------------|
| [notebooks/genetic_search.ipynb](../notebooks/genetic_search.ipynb) | End-to-end: datagen → genetic HP search → multi-tier training → ONNX export → benchmark. Kaggle/Colab/Paperspace-ready via env vars. |

### Notebook env vars (key subset — full table in Cell 1)

| Variable | Default | Purpose |
|----------|---------|---------|
| `WAKE_WORD` | `hey jarvis` | Target phrase |
| `OUTPUT_DIR` | `./ww_output` | Output root |
| `POPULATION` / `GENERATIONS` | 12 / 5 | Genetic search budget |
| `TIERS_TO_TRAIN` | `micro,small,filterbank_small` | Architecture tiers to train |
| `FINAL_EPOCHS` | 30 | Epochs for final per-tier training |
| `SEARCH_TWO_STAGE` | `false` | Enable two-stage genetic search |
| `SEARCH_DEMES` | 1 | Parallel island count |
| `SEARCH_FITNESS_FN` | `f1` | Selection pressure: `f1`, `exp_f1`, `double_exp_f1` |
| `CUSTOM_TRAIN_CSV` | — | BYO dataset; skips TTS synthesis |
| `CUSTOM_TEST_CSV` | — | BYO test split; omit for auto 80/20 |
| `DOWNLOAD_AUGMENT` | `false` | Download bg_noise/music/RIR from HF |

---

## Key Classes and Functions

| Name | Module | Description | Source |
|------|--------|-------------|--------|
| `run_genetic_search` | `sweep` | Island-model GA hyperparameter search | `sweep.py:543` |
| `run_two_stage_genetic_search` | `sweep` | Broad stage 1 + focused stage 2 GA | `sweep.py:662` |
| `run_sweep` | `sweep` | Optuna Bayesian search | `sweep.py:91` |
| `run_grid_search` | `sweep` | Exhaustive grid search | `sweep.py:202` |
| `run_random_search` | `sweep` | Uniform random sampling | `sweep.py:276` |
| `train_from_wakeword` | `quickstart` | String → trained ONNX in one call | `quickstart.py:205` |
| `QuickstartConfig` | `quickstart` | All quickstart knobs with defaults | `quickstart.py:34` |
| `QuickstartResult` | `quickstart` | Paths and metrics of a completed run | `quickstart.py:70` |
| `BaseWakeModel` | `model` | Extractor + head + ONNX export | `model.py` |
| `OnnxWakeWordInferencer` | `inference` | PyTorch-free inference (single/batch/streaming) | `inference.py` |
| `SlidingFeatureCacheTensor` | `feats` | Rolling feature cache for streaming | `feats.py:27` |
| `WakeWordTrainer` | `trainer` | Full training loop with AMP, DDP, MLflow | `trainer.py` |
| `AudioDataset` | `dataset` | `(path, label)` dataset with augmentation | `dataset.py` |
| `LossManager` | `loss` | Multi-loss composition and triplet mining | `loss.py` |
| `_validate_ga_params` | `sweep` | Raises `ValueError` on bad GA inputs | `sweep.py:25` |
| `_apply_fitness_fn` | `sweep` | Selection pressure transform (internal) | `sweep.py:73` |

---

## Documentation

### Core Architecture

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | System design: abstractions, data flow, ONNX pipeline, sliding cache |
| [api.md](api.md) | Full API reference — every public class and method with `file:line` citations |
| [data_contract.md](data_contract.md) | Dataset format, CSV contract, augmentation layout |

### Components

| Document | Description |
|----------|-------------|
| [extractors.md](extractors.md) | All 17 feature extractors: parameters, hardware fit |
| [classifiers.md](classifiers.md) | All 11 classifier heads: architecture, param counts |
| [losses.md](losses.md) | All 17 loss functions and recommended combinations |
| [enrichment.md](enrichment.md) | VAD, Pitch, SNRAware, MultiResolution wrappers |
| [markov_hmm.md](markov_hmm.md) | Markov and HMM extractors |
| [esp32.md](esp32.md) | ESP32 tiers, `SizeAwareLoss`, micro genetic search |

### Training & Deployment

| Document | Description |
|----------|-------------|
| [training.md](training.md) | Dataset prep, tier selection, full CLI reference |
| [export.md](export.md) | ONNX export, quantization, metadata embedding |
| [inference.md](inference.md) | Single, batch, and streaming inference |
| [streaming.md](streaming.md) | `SlidingFeatureCacheTensor` deep-dive |
| [distillation.md](distillation.md) | Knowledge distillation workflow |
| [hardware_guide.md](hardware_guide.md) | Hardware-specific configurations |
| [quickstart.md](quickstart.md) | `QuickstartConfig` API and CLI reference |

### Optimization & Research

| Document | Description |
|----------|-------------|
| [sweep.md](sweep.md) | Full search API: all functions, parameter tables, fitness functions, island model |
| [search_strategies.md](search_strategies.md) | Decision tree for choosing a search strategy |
| [benchmarking.md](benchmarking.md) | Latency measurement, ONNX vs PyTorch, RTF |
| [recipes.md](recipes.md) | End-to-end recipes per use case |

---

## Key Links

- [FAQ.md](../FAQ.md) — keyword-rich Q&A
- [AUDIT.md](../AUDIT.md) — known issues and tech debt
- [SUGGESTIONS.md](../SUGGESTIONS.md) — improvement proposals
- GitHub: https://github.com/TigreGotico/ww_trainer
- Pre-exported MFCC ONNX: https://huggingface.co/TigreGotico/mfcc-onnx
- Funded by NGI0 Commons Fund / NLnet
