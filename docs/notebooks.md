# Notebook Curriculum Guide

A progressive 10-notebook curriculum for training, evaluating, and deploying
wake-word detectors — from a 241-parameter ESP32 model to a HuBERT-backed
production classifier.

---

## Overview

### Who this is for

| Persona | Goal | Start here |
|---------|------|------------|
| **Beginner** | Get a working ONNX model for a Raspberry Pi 4 as fast as possible | nb01 → nb03 |
| **ML Engineer** | Systematically find the best config for a target device | nb02 → nb05 or nb06 |
| **Researcher** | Understand trade-offs, distil a new featurizer, run ablations | nb07 → nb08 → nb09 |

---

## Hardware Tier Reference

| Tier | Extractor | Head | Params | RAM (inference) | Target device | Batch | Notes |
|------|-----------|------|--------|-----------------|---------------|-------|-------|
| `esp32_nano` | MFCC-13 | FFN-16 | ~241 | <1 KB | ESP32 sub-1 KB | 1 | Smallest possible |
| `esp32_sweet` | MFCC-13 | FFN-64 | ~1 K | ~4 KB | ESP32 sub-10 KB | 1 | Best ESP32 default |
| `esp32_max` | MFCC-13 | FFN-128 | ~2 K | ~10 KB | ESP32 sub-50 KB | 1 | Max for ESP32 |
| `micro` | MFCC-40 | FFN-128 | ~50 K | ~200 KB | MCU / RPi Zero | 4 | Good baseline MCU |
| `delta_micro` | delta-MFCC-13 | FFN-128 | ~55 K | ~220 KB | MCU / RPi Zero | 4 | Better temporal |
| `small` | MFCC-40 | GRU-128 | ~200 K | ~800 KB | RPi 3/4, SBC | 16 | Best CPU default |
| `filterbank_small` | FilterBank | GRU-128 | ~200 K | ~800 KB | RPi 3/4, SBC | 16 | Trainable filters |
| `sincnet_small` | SincNet | GRU-128 | ~300 K | ~1.2 MB | RPi, SBC | 16 | Learnable sinc |
| `gammatone_small` | Gammatone | GRU-128 | ~200 K | ~800 KB | RPi, SBC | 16 | Perceptual filters |
| `medium` | HuBERT-ONNX | FFN | ~90 M feat | ~350 MB | RPi 4, laptop | 8 | GPU recommended |
| `large` | HuBERT PyTorch | biGRU-256 | ~300 M feat | ~1.2 GB | Server/workstation | 16 | GPU required |

CPU-safe tiers: `esp32_*`, `micro`, `delta_micro`, `small`, `filterbank_small`,
`sincnet_small`, `gammatone_small`.
GPU required: `medium`, `large`, `hubert_*`, `wav2vec2_*`.

---

## Curriculum Map

```
Track A — Getting Started
  nb01  Zero-to-ONNX quickstart         Beginner / ML Eng   ~15 min T4 / ~2 h CPU
  nb02  Systematic experiments grid     ML Eng              ~60–80 min T4 / CPU varies

Track B — Production Training
  nb03  Infinite / goal-based training  ML Eng / Researcher ~30 min T4 / overnight CPU
  nb10  Genetic HP search               ML Eng / Researcher ~2–4 h T4

Track C — Hardware Tiers
  nb04  Micro scale (ESP32, RPi Zero)   Embedded Eng        ~20 min T4 / ~3 h CPU
  nb05  Embedded scale (RPi 3/4, x86)  Embedded Eng        ~40 min T4 / ~4 h CPU
  nb06  GPU scale (HuBERT featurizer)   ML Eng / Researcher ~60 min T4 (GPU required)

Track D — Advanced
  nb07  TinyHuBERT distillation         Researcher          ~90 min T4 (GPU required)
  nb08  WakeHuBERT inference            ML Eng / Researcher ~30 min T4 / ~2 h CPU
  nb09  Full ablation                   Researcher          ~3–5 h T4 / multi-day CPU
```

---

## Per-Notebook Descriptions

### nb01 — Zero-to-ONNX Quickstart
**File:** `kaggle_quickstart.ipynb` (symlinked as nb01)
**Purpose:** End-to-end pipeline in one click: datagen → train → ONNX export → inference test.
**Prerequisites:** None.
**What you learn:** The three required steps (datagen, train, export); reading ONNX output sizes;
running `OnnxWakeWordInferencer`.
**Runtime:** ~15 min Kaggle T4 · ~2 h local CPU (reduce `N_POSITIVE=100`).
**Key outputs:** `best_f1.onnx`, `best_f1_featurizer.onnx`, summary metrics.

---

### nb02 — Systematic Experiments Grid
**File:** `kaggle_experiments.ipynb` (symlinked as nb02)
**Purpose:** Grid search over tiers × losses × augmentation levels. Resumable.
**Prerequisites:** nb01 (understand the config vars).
**What you learn:** Which loss/augment combination works best; reading the F1 bar chart and
precision-recall scatter.
**Runtime:** ~8 min/cell on T4 · default grid (8 cells) ≈ 60–80 min.
**Key outputs:** `results/*.json`, `experiment_results.png`.

---

### nb03 — Infinite / Goal-Based Training
**File:** `kaggle_infinite.ipynb` (symlinked as nb03)
**Purpose:** Train until a target F1 / EER / FAR is reached, not for a fixed epoch count.
**Prerequisites:** nb01.
**What you learn:** Goal-driven stopping; FAR/FRR budgets; long-run convergence behaviour.
**Runtime:** ~30 min T4 · overnight CPU (depends on target).
**Key outputs:** Best checkpoint when goal is reached, convergence curve PNG.

---

### nb04 — Micro Scale (ESP32, RPi Zero)
**File:** `nb04_micro.ipynb`
**Purpose:** Train all MCU-class tiers, audit sizes, export a C header for ESP32 firmware.
**Prerequisites:** nb01.
**What you learn:** ESP32 FLASH/RAM budgets; QAT int8; C header export; latency on embedded
hardware.
**Runtime:** ~20 min T4 · ~3 h CPU (5 tiers × 20 epochs each).
**Key outputs:** Per-tier ONNX files, `model.h` C header, results DataFrame, size audit table.

---

### nb05 — Embedded Scale (RPi 3/4, x86 low-power)
**File:** `nb05_embedded.ipynb`
**Purpose:** Train and benchmark the four embedded-class tiers; compare featurizers with PCA;
demo streaming inference.
**Prerequisites:** nb01.
**What you learn:** SincNet vs Gammatone vs FilterBank vs MFCC for embedded deployment;
real-time factor (RTF); streaming confidence curves.
**Runtime:** ~40 min T4 · ~4 h CPU (4 tiers × 30 epochs each).
**Key outputs:** Benchmark table, PCA PNG, streaming confidence PNG, ONNX files.

---

### nb06 — GPU Scale (HuBERT Featurizer)
**File:** `nb06_gpu.ipynb`
**Purpose:** Train `medium` (HuBERT-ONNX featurizer) and optionally `large` (biGRU-256);
compare against CPU tiers; visualise t-SNE of HuBERT embeddings.
**Prerequisites:** nb01, GPU available.
**What you learn:** When HuBERT featurizer is worth the compute cost; t-SNE cluster separation;
RTF penalty of large featurizers (~100× MFCC).
**Runtime:** ~60 min T4 (GPU required — will skip large tier if VRAM < 8 GB).
**Key outputs:** medium/large ONNX files, t-SNE PNG, F1/EER comparison table.

---

### nb07 — TinyHuBERT Distillation
**File:** `nb07_distill.ipynb`
**Purpose:** Distil a streaming causal CNN+GRU student that mimics HuBERT's 768-d frame
representations; export as `tinyhubert.onnx`.
**Prerequisites:** nb06 (understand HuBERT featurizer), GPU required.
**What you learn:** Knowledge distillation with MSE + InfoNCE loss; causal student architecture;
how to evaluate representation quality with PCA overlap.
**Runtime:** ~90 min T4 (GPU required).
**Key outputs:** `tinyhubert.onnx`, training curve PNG, PCA overlap PNG, student vs teacher
param count.

---

### nb08 — WakeHuBERT Inference (CPU-friendly)
**File:** `nb08_wakehubert.ipynb`
**Purpose:** Use a pre-distilled `tinyhubert.onnx` as the featurizer for a lightweight
wake-word classifier — no GPU needed at training time.
**Prerequisites:** nb07 (provides `tinyhubert.onnx`).
**What you learn:** `OnnxFeatureExtractor` API; two-step pipeline (distil on GPU → classify on
CPU); size audit for the combined featurizer + head.
**Runtime:** ~30 min T4 · ~2 h CPU (featurizer forward is ONNX-accelerated).
**Key outputs:** Wake-word classifier ONNX, combined size audit, PCA comparison vs MFCC.

---

### nb09 — Full Ablation
**File:** `nb09_ablation.ipynb`
**Purpose:** 3×4×3 = 36-cell grid (featurizer × loss × augmentation) with heatmaps, Fisher
ratio, and silhouette scores. More thorough than nb02.
**Prerequisites:** nb02 (smaller grid), nb05 (featurizer intuition).
**What you learn:** Heatmap interpretation; Fisher ratio as a featurizer quality proxy;
silhouette score for embedding separability.
**Runtime:** ~3–5 h T4 · multi-day CPU (use `SKIP_COMPLETED=true` to spread across sessions).
**Key outputs:** `ablation_heatmap.png`, `augmentation_impact.png`, `embedding_quality.png`,
best config summary.

---

### nb10 — Genetic HP Search
**File:** `genetic_search.ipynb` (symlinked as nb10)
**Purpose:** Evolve the best hyperparameter configuration using a genetic algorithm; train the
top-N survivors across multiple tiers.
**Prerequisites:** nb02 (understand the search space).
**What you learn:** Genetic search vs grid search; population dynamics; multi-tier training from
a single best config.
**Runtime:** ~2–4 h T4.
**Key outputs:** Pareto-front plot, best HP JSON, trained multi-tier ONNX files.

---

## Choosing the Right Notebook

```
What is your target hardware?
├── ESP32 / MCU / RPi Zero  ──────────────────────────────→ nb04
├── RPi 3 / RPi 4 / x86 low-power (N100, Celeron, etc.)  → nb05
│   └── Want best possible accuracy on RPi 4?
│       └── GPU available for pre-training?  ─────────────→ nb07 → nb08
└── Server / laptop with discrete GPU  ──────────────────→ nb06
    └── Want to distil a portable featurizer?  ──────────→ nb07 → nb08

How much time do you have?
├── "I just want something working now"  ─────────────────→ nb01
├── "I want the best config without guessing"  ───────────→ nb02 (quick) or nb09 (thorough)
├── "I want to train until quality is good enough"  ──────→ nb03
└── "I want to explore the full search space"  ──────────→ nb10 (genetic)

Are you a researcher?
└── Want to understand what drives accuracy?  ────────────→ nb09 (ablation)
    └── Want to publish a new featurizer baseline?  ──────→ nb07 + nb09
```

---

## Platform Setup

### Kaggle Secrets for MLflow

1. Open **Add-ons → Secrets** in the Kaggle editor.
2. Add a secret named `MLFLOW_TOKEN` with your bearer token.
3. Set `MLFLOW_URI` in Cell 2 (or as a Kaggle Secret named `MLFLOW_URI`).
4. The notebooks inject the token automatically via `kaggle_secrets.UserSecretsClient`.

### GPU vs CPU notebooks

| Notebook | GPU required? | Notes |
|----------|---------------|-------|
| nb01–nb05 | No | CPU-only tiers; use `DEVICE=cpu` or `DEVICE=auto` |
| nb06 | Yes | `assert torch.cuda.is_available()` at Cell 3 |
| nb07 | Yes | Teacher (HuBERT) forward pass requires GPU |
| nb08 | No | Featurizer is ONNX; head training is lightweight |
| nb09 | No | Uses CPU-safe featurizers only |
| nb10 | No | Genetic search uses CPU tiers |

### Disk budget per notebook

| Notebook | Approximate disk use |
|----------|----------------------|
| nb01 | ~500 MB (dataset + 2 ONNX files) |
| nb02 | ~2–3 GB (shared dataset + N result dirs) |
| nb03 | ~500 MB |
| nb04 | ~1 GB (5 tier outputs) |
| nb05 | ~1.5 GB (4 tier outputs + plots) |
| nb06 | ~2 GB (HuBERT cache + 2 tier outputs) |
| nb07 | ~3 GB (HuBERT teacher cache + streaming dataset cache) |
| nb08 | ~500 MB (tinyhubert.onnx + head) |
| nb09 | ~5–8 GB (36-cell outputs) |
| nb10 | ~2–4 GB (population × survivors) |

Kaggle gives 20 GB of persistent storage per notebook session. All notebooks
include a disk-space guard that aborts early if less than 3–5 GB is free.

---

## Recommended Running Order

### Beginner
```
nb01  →  nb03  →  nb04 (if targeting embedded)
```

### ML Engineer
```
nb01  →  nb02  →  nb05 (RPi target) or nb06 (GPU target)  →  nb10 (optional)
```

### Researcher
```
nb01  →  nb02  →  nb06  →  nb07  →  nb08  →  nb09
```

---

## File Mapping (old name → new name)

| Old notebook | New name | Track |
|---|---|---|
| `kaggle_quickstart.ipynb` | nb01 | A |
| `kaggle_experiments.ipynb` | nb02 | A |
| `kaggle_infinite.ipynb` | nb03 | B |
| `genetic_search.ipynb` | nb10 | B |
| — | `nb04_micro.ipynb` | C |
| — | `nb05_embedded.ipynb` | C |
| — | `nb06_gpu.ipynb` | C |
| — | `nb07_distill.ipynb` | D |
| — | `nb08_wakehubert.ipynb` | D |
| — | `nb09_ablation.ipynb` | D |
