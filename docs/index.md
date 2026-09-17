# wakeforge Documentation

wakeforge is a research framework for wake-word detection — from
microcontrollers to GPU servers. It implements architectures, losses, and
featurizers from the published literature; some exist to reproduce a
paper's result, not because they are the best default. An easy training run
is not proof of a good detector: see
[guides/expectations.md](guides/expectations.md) for what "good" means in
false-accept and false-reject numbers, and how to tell a demonstration
preset from a deployable one. Every component exports to ONNX; production
inference requires only `onnxruntime` and `numpy`.

Version: `0.4.1a3` — `ww_trainer/version.py` (import path and CLI scripts
stay `ww_trainer`; the package, repo, and docs are `wakeforge`)

---

## Start here

| Document | For… |
|---|---|
| [learning_path.md](learning_path.md) | The **zero-to-hero curriculum** — staged path from first model to research surface, with literature anchors |
| [faq.md](faq.md) | Topic-ordered Q&A — 15 sections, jump to the one matching your task |
| [getting_started/quickstart.md](getting_started/quickstart.md) | One command, one ONNX pair |
| [getting_started/requirements.md](getting_started/requirements.md) | Disk, bandwidth, RAM, and time budget per quickstart preset |
| [examples/README.md](https://github.com/TigreGotico/wakeforge/blob/dev/examples/README.md) | 43 runnable examples |

---

## Guides — how to do a thing

| Document | Covers |
|---|---|
| [guides/datasets.md](guides/datasets.md) | `AudioDataset`, CSV format, augmentation folders, label convention |
| [guides/training.md](guides/training.md) | Full CLI, losses, augmentation, hard-negative mining, infinite training, VC backends |
| [guides/evaluation.md](guides/evaluation.md) | Latency, RTF, ONNX-vs-PyTorch benchmarking |
| [guides/expectations.md](guides/expectations.md) | What a deployable wake word looks like in FA/hour and FRR, data budget per tier, how to measure with `ww-benchmarks` |
| [guides/inference.md](guides/inference.md) | Single, batch, and streaming inference (PyTorch and ONNX) |
| [guides/export.md](guides/export.md) | ONNX export, quantization, metadata embedding |
| [guides/embedded.md](guides/embedded.md) | ESP32 / MCU tiers, `SizeAwareLoss`, micro genetic search, C header |
| [guides/search.md](guides/search.md) | Optuna, Grid, Random, Genetic, Two-Stage — strategies and parameter tables |
| [guides/distillation.md](guides/distillation.md) | Knowledge distillation workflow |
| [guides/recipes.md](guides/recipes.md) | End-to-end recipes per use case |
| [guides/notebooks.md](guides/notebooks.md) | Notebook curriculum, hardware-tier decision tree |

---

## Reference — exhaustive component lists

| Document | Covers |
|---|---|
| [reference/extractors.md](reference/extractors.md) | All featurizers (MFCC, SincNet, LEAF, HuBERT-ONNX, Markov, HMM, …) |
| [reference/classifiers.md](reference/classifiers.md) | All classifier heads (FFN, GRU, BC-ResNet, KWT, OCSVM, …) |
| [reference/losses.md](reference/losses.md) | All loss functions including RPPL component breakdown |
| [reference/enrichment.md](reference/enrichment.md) | VAD, Pitch, SNRAware, MultiResolution, Delta wrappers |
| [reference/phonmatch.md](reference/phonmatch.md) | PhonMatchNet — IPA phoneme-conditioned heads |
| [reference/api.md](reference/api.md) | Every public class and method with `file:line` citations |

---

## Research

| Document | Covers |
|---|---|
| [research/rppl.md](research/rppl.md) | Robust Prototype Diversity Loss — formulation, EMA, ablations |
| [research/tinyhubert.md](research/tinyhubert.md) | TinyHuBERT distillation design |
| [research/references.md](research/references.md) | Bibliography |

---

## Internals — contributor docs

| Document | Covers |
|---|---|
| [internals/architecture.md](internals/architecture.md) | System design: abstractions, data flow, ONNX pipeline, sliding cache |
| [internals/known_issues.md](internals/known_issues.md) | Currently open issues and ONNX-export constraints |
| [TODO.md](https://github.com/TigreGotico/wakeforge/blob/dev/TODO.md) | Open feature backlog |

---

## Key classes — quick lookup

| Name | Module | Source |
|---|---|---|
| `train_from_wakeword` | `quickstart` | `quickstart.py:231` |
| `QuickstartConfig`, `QuickstartResult` | `quickstart` | `quickstart.py:23,71` |
| `WakeWordTrainer` | `trainer` | `trainer.py` |
| `BaseWakeModel` | `model` | `model.py` |
| `AudioDataset` | `dataset` | `dataset.py` |
| `LossManager` | `loss` | `loss.py` |
| `RobustProtoDiversityLoss` | `loss` | `loss.py:282` |
| `OnnxWakeWordInferencer` | `inference` | `inference.py` |
| `SlidingFeatureCacheTensor` | `feats` | `feats.py:27` |
| `run_sweep` / `run_grid_search` / `run_random_search` | `sweep` | `sweep.py:91,299,374` |
| `run_genetic_search` / `run_two_stage_genetic_search` | `sweep` | `sweep.py:744,1029` |
| `infinite_training_loop`, `StoppingGoal` | `infinite_loop` | `infinite_loop.py` |
| `load_vc_backend` | `vc_helpers` | `vc_helpers.py` |

---

## Modules at a glance

| Module | Purpose |
|---|---|
| `ww_trainer.feats` | Feature extractors and `SlidingFeatureCacheTensor` |
| `ww_trainer.model` | Classifier heads and `BaseWakeModel` |
| `ww_trainer.loss` | Loss functions via `LossManager` |
| `ww_trainer.sweep` | Hyperparameter search (Optuna / Grid / Random / Genetic) |
| `ww_trainer.quickstart` | String → ONNX in one call |
| `ww_trainer.trainer` + `ww_trainer.loop` | Training driver and per-epoch loop |
| `ww_trainer.dataset` + `ww_trainer.cache` + `ww_trainer.mining` | Data loading, feature cache, hard-negative mining |
| `ww_trainer.inference` | PyTorch-free inference |
| `ww_trainer.benchmark` | Latency and RTF measurement |
| `ww_trainer.tiers` | Hardware tier presets |
| `ww_trainer.datagen` | Synthetic dataset generation |
| `ww_trainer.export_c` | C-header export for ESP32 FFN models |
| `ww_trainer.calibration` | Platt scaling |
| `ww_trainer.qat` | Quantization-aware training |
| `ww_trainer.ddp` | Multi-GPU (DDP) |
| `ww_trainer.visualization` | PCA, t-SNE, UMAP, ROC/PR/DET, RPPL dashboard |
| `ww_trainer.evaluation` | `compute_fitness_score`, `compute_readiness` |
| `ww_trainer.checkpoint` | Save / load / average checkpoints |
| `ww_trainer.infinite_loop` | Goal-based open-ended training |
| `ww_trainer.vc_helpers` | voiceclonnx VC delegation |
| `ww_trainer.env` | `.env` loader |

---

## External

- GitHub: <https://github.com/TigreGotico/wakeforge>
- Pre-exported MFCC ONNX: <https://huggingface.co/TigreGotico/mfcc-onnx>
- Funded by NGI0 Commons Fund / NLnet
