# ww-trainer Documentation

Training and research suite for wake word detection — from microcontrollers to GPU servers. Every extractor exports to ONNX; production inference requires only `onnxruntime` and `numpy`.

Version: `0.0.1a1` — `ww_trainer/version.py`

---

## Module Reference

| Module | Description |
|--------|-------------|
| `ww_trainer.feats` | Feature extractors (12 standalone + 5 wrappers) and `SlidingFeatureCacheTensor` |
| `ww_trainer.model` | Classifier heads and `BaseWakeModel` |
| `ww_trainer.loss` | 17 loss functions via `LossManager`; RPPL with EMA prototype and warmup |
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
| `ww_trainer.visualization` | PCA, t-SNE, UMAP, ROC/PR/DET, RPPL dashboard plots |
| `ww_trainer.evaluation` | `compute_fitness_score`, `compute_readiness` |
| `ww_trainer.checkpoint` | `save_checkpoint`, `load_checkpoint` |
| `ww_trainer.cache` | `FeatureCache` — MD5-keyed feature caching |
| `ww_trainer.mining` | `mine_hard_negatives` |
| `ww_trainer.loop` | `training_loop` (extracted from `WakeWordTrainer`) |
| `ww_trainer.infinite_loop` | `infinite_training_loop`, `StoppingGoal` — goal-based open-ended training |
| `ww_trainer.vc_helpers` | `load_vc_backend` — pluggable TTS/VC abstraction (chatterbox-onnx / chatterbox / linacodec) |
| `ww_trainer.env` | `.env` file loader (`load_env`) — injects env vars for all scripts |

---

## Notebook Reference

| Notebook | Description |
|----------|-------------|
| [notebooks/kaggle_quickstart.ipynb](../notebooks/kaggle_quickstart.ipynb) | Zero-to-ONNX in one notebook: TTS datagen → single-tier training → ONNX export → inference test. Kaggle/Colab/Paperspace-ready. |
| [notebooks/kaggle_experiments.ipynb](../notebooks/kaggle_experiments.ipynb) | Systematic experiments: loop over tiers × loss functions × augmentation levels → results table + F1 bar chart. Resumable. |
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
| `infinite_training_loop` | `infinite_loop` | Goal-based training over unlimited NWW pool | `infinite_loop.py` |
| `StoppingGoal` | `infinite_loop` | Configures F1/EER/FAR targets and plateau detection | `infinite_loop.py` |
| `load_vc_backend` | `vc_helpers` | Factory for TTS/VC backends (onnx / torch / linacodec) | `vc_helpers.py` |
| `RobustProtoDiversityLoss` | `loss` | RPPL: EMA prototype + warmup + hard-div + proto-consistency | `loss.py:281` |
| `LossManager.step_epoch` | `loss` | Notifies epoch-aware criteria (RPPL warmup scheduling) | `loss.py` |
| `plot_rppl_dashboard` | `visualization` | 6-panel RPPL training dashboard → MLflow artifact | `visualization.py` |

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

## Scripts

All scripts live under `scripts/`. Run with `.venv/bin/python scripts/<subdir>/<script>.py`.

### Training (`scripts/train/`)

| Script | Purpose |
|--------|---------|
| `train_hey_mycroft.py` | Quick single-run training from a local dataset |
| `train_full.py` | Sequential multi-arch training (6 CPU-safe tiers) |
| `train_full_nww.py` | GA search + loss comparison with full NWW augmentation |
| `train_rppl.py` | RPPL experiment — PCA/t-SNE + 6-panel RPPL dashboard every N epochs |
| `train_infinite.py` | Goal-based open-ended training with large NWW pool mining + VC synthesis |
| `train_ablation.py` | Loss × augmentation ablation grid → MLflow heatmaps |
| `train_parallel.py` | Parallel loss-comparison training with shared waveform cache |
| `train_micro_genetic.py` | Three-stage GA focused on class imbalance / MCU targets |
| `train_sincnet_genetic.py` | GA over SincNet / Gammatone featurisers |
| `train_vc_ablation.py` | Measures impact of voice-conversion positives on F1/EER |
| `train_pilot_gpt.py` | Infinite training for the "pilot gee pee tee" wake word |
| `train_mfcc.sh` | Shell wrapper for a standard MFCC run |

### Evaluation & Inference (`scripts/eval/`)

| Script | Purpose |
|--------|---------|
| `eval_hey_mycroft.py` | Full evaluation: ROC/PR/DET, FP/FN lists, confidence histograms |
| `test_wakeword.py` | CLI: test an ONNX model on a file or live microphone |
| `listen_all.py` | Run N random models in parallel with a real-time confidence dashboard |
| `mic_test.py` | Minimal mic capture test (sounddevice) |

### Data Management (`scripts/data/`)

| Script | Purpose |
|--------|---------|
| `download_hdd4_datasets.py` | Download NWW/augmentation datasets to `/mnt/hdd4` via git-lfs |
| `select_training_subset.py` | Copy a size-capped subset from hdd4 into a local fast-access dir |
| `preprocess.py` | VAD-based silence trimming + normalisation of raw audio |
| `rebuild_dataset.py` | Rebuild dataset adding speech negatives |
| `localise_csvs.py` | Rewrite metadata CSVs to use local paths after copying from hdd4 |
| `generate_vc_positives.py` | Batch TTS/VC positive generation from donor voices |
| `generate_pilot_gpt_dataset.py` | Build the pilot_gpt dataset (TTS + negatives) |
| `generate_pilot_gpt_vc.py` | VC positives for pilot_gpt |

### Research / Utilities (`scripts/research/`, `scripts/`)

| Script | Purpose |
|--------|---------|
| `research/tinyhubert.py` | WakeHuBERT — streaming HuBERT distillation trainer |
| `research/tinyhuberta.py` | Alternative HuBERT distillation variant |
| `export_mfcc.py` | Export a trained model's MFCC featurizer to ONNX |
| `export_w2vbert.py` | Export a Wav2Vec2-BERT featurizer to ONNX via optimum |
| `train_markov_featurizer.py` | Fit and export a MarkovTransitionExtractor |

---

## Key Links

- [faq.md](faq.md) — keyword-rich Q&A
- [audit.md](audit.md) — known issues and tech debt
- [suggestions.md](suggestions.md) — feature backlog
- [changelog.md](changelog.md) — timestamped change history
- [references.md](references.md) — academic bibliography
- [rppl_whitepaper.md](rppl_whitepaper.md) — RPPL loss technical whitepaper
- [tinyhubert_whitepaper.md](tinyhubert_whitepaper.md) — TinyHuBERT distillation design
- GitHub: https://github.com/TigreGotico/ww-trainer
- Pre-exported MFCC ONNX: https://huggingface.co/TigreGotico/mfcc-onnx
- Funded by NGI0 Commons Fund / NLnet
