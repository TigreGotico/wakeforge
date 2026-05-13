# Wake Word Trainer

A research-grade training suite for **wake-word detection** — the always-on keyword spotter that wakes "Hey Siri", "OK Google", or your own custom phrase. Train, evaluate, and ship lightweight on-device detectors that run anywhere from an ESP32 to a GPU server. Every component exports to ONNX; production inference requires only `onnxruntime` and `numpy` — no PyTorch at runtime.

## What is a wake word?

A wake word is a short phrase ("hey jarvis", "computer", "alexa") that a device listens for continuously. When detected, downstream STT/NLU runs. The detector must:

- **Run on tiny hardware** — sub-100 KB models, <10 % CPU, no internet.
- **Tolerate noise, accents, distance, reverberation** — real-world far-field audio.
- **Almost never false-fire** — < 1 false alarm per hour is the bar.
- **Trigger reliably when spoken** — > 90 % recall at that false-alarm rate.

ww-trainer is the toolchain to build that detector from a single phrase — synthesise data, train, evaluate, export, deploy.

## Who is this for?

| You are… | Start here |
|---|---|
| **Hobbyist** who wants to wake a Pi with their own phrase | [`docs/quickstart.md`](docs/quickstart.md) — one command, ONNX in 5 minutes |
| **Embedded engineer** shipping to ESP32 / MCU | [`docs/esp32.md`](docs/esp32.md) + [`docs/hardware_guide.md`](docs/hardware_guide.md) |
| **Voice-assistant integrator** (OVOS, Rhasspy, custom) | [`docs/inference.md`](docs/inference.md) + [`docs/streaming.md`](docs/streaming.md) |
| **ML researcher** comparing architectures, losses, distillation | [`docs/sweep.md`](docs/sweep.md), [`docs/losses.md`](docs/losses.md), [`docs/rppl_whitepaper.md`](docs/rppl_whitepaper.md) |
| **Curious noob** who has never trained a model | [`notebooks/kaggle_quickstart.ipynb`](notebooks/kaggle_quickstart.ipynb) — runs free on Kaggle |

## Why this framework?

- **Single-string-to-ONNX** quickstart: `train_from_wakeword("hey jarvis", out)` and you have a deployable model.
- **17 feature extractors × 15 classifier heads × 17 losses** — a real research surface, not a toy.
- **Genetic & Bayesian hyperparameter search** with island-model parallelism and adaptive mutation.
- **Synthetic datagen** — TTS + voice conversion to bootstrap a dataset from zero recordings.
- **Hard-negative mining** and **infinite training** for industrial-scale negative pools.
- **ONNX-first**: every model — featurizer and head — exports cleanly. No CUDA-only kernels.
- **Hardware tiers from `esp32_nano` (sub-1 KB) to `hubert_medium`** — one preset per target.

### Trade-offs / honest limitations

- **CPU-only training is supported but slow** for the largest tiers. Best UX is a single mid-range GPU.
- **Synthetic-only datasets** are great smoke-tests but real users still need real recordings for top performance.
- **ONNX export is mandatory** — features that cannot trace (custom CUDA kernels, dynamic control flow) are excluded from the framework (see [`docs/audit.md`](docs/audit.md)).
- **Large SSL featurizers** (HuBERT, Wav2Vec2-BERT) are pre-exported to ONNX and used frozen — not fine-tuned at training time. This guarantees train/inference parity but limits SSL adaptation.

## Install

```bash
uv pip install -e ".[dev]"
# Optional extras
uv pip install -e ".[sweep]"        # Optuna Bayesian search
uv pip install -e ".[transformers]" # HuBERT / Wav2Vec2 / Wav2Vec2-BERT extractors
uv pip install -e ".[mlflow]"       # experiment tracking
uv pip install -e ".[datagen]"      # synthetic dataset generation (TTS + VAD)
uv pip install -e ".[vc-onnx]"     # voice conversion — CPU ONNX backend (default)
uv pip install -e ".[vc-torch]"    # voice conversion — GPU PyTorch backend
uv pip install -e ".[vc-linacodec]" # voice conversion — LinaCodec 48 kHz backend
uv pip install -e ".[vc]"          # all VC backends
uv pip install -e ".[mic]"         # live microphone testing (sounddevice)
uv pip install -e ".[viz]"         # UMAP embedding visualisation
uv pip install -e ".[markov]"      # Markov/HMM extractor training
```

## Quickstart

```python
from ww_trainer.quickstart import train_from_wakeword

result = train_from_wakeword("hey jarvis", "./hey_jarvis", tier="small", epochs=50)
print(result.best_onnx_path)   # Path to best_f1.onnx
print(result.metrics)          # {"f1": ..., "precision": ..., "recall": ...}
```

Or from the CLI:

```bash
ww_trainer-quickstart --wake-word "hey jarvis" --output-dir ./hey_jarvis
```

This synthesises a dataset (TTS + negatives) and trains. Pass `--reuse-dataset` to skip datagen if a dataset already exists. See [`docs/quickstart.md`](docs/quickstart.md).

## Genetic Hyperparameter Search

`run_genetic_search` — `ww_trainer/sweep.py:744`
`run_two_stage_genetic_search` — `ww_trainer/sweep.py:1029`

```python
from ww_trainer.sweep import run_two_stage_genetic_search

results = run_two_stage_genetic_search(
    "dataset.csv",
    population_size=20, generations=10,
    stage2_population=10, stage2_generations=5,
    n_demes=2, fitness_fn="exp_f1",
)
print(results["best_config"], results["best_score"])
```

Key parameters — full tables in [`docs/sweep.md`](docs/sweep.md):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `population_size` | 20 | Individuals per generation |
| `generations` | 10 | Max generations |
| `n_demes` | 1 | Parallel island count |
| `migration_interval` | 5 | Generations between ring-topology deme migrations |
| `fitness_fn` | `"f1"` | Selection pressure: `"f1"`, `"exp_f1"`, `"double_exp_f1"` |
| `mutation_decay` | 0.0 | Per-generation multiplicative decay on mutation rate |
| `on_generation` | None | Callback after each generation |
| `full` | False | Expand space to include featurizer / arch / loss |
| `target_f1` | None | Early stop when F1 reaches this value |
| `timeout_minutes` | None | Wall-clock stop per deme |

## Notebooks

| Notebook | Purpose |
|----------|---------|
| [`notebooks/kaggle_quickstart.ipynb`](notebooks/kaggle_quickstart.ipynb) | Zero-to-ONNX: TTS datagen → single-tier training → ONNX export → inference test |
| [`notebooks/kaggle_experiments.ipynb`](notebooks/kaggle_experiments.ipynb) | Grid: tiers × loss functions × augmentation → results table + plots. Resumable. |
| [`notebooks/genetic_search.ipynb`](notebooks/genetic_search.ipynb) | Datagen → genetic HP search → multi-tier training → ONNX export → benchmark |
| [`notebooks/kaggle_infinite.ipynb`](notebooks/kaggle_infinite.ipynb) | Infinite training: smoke test, full run, VC synthesis, NWW pool ablation |
| [`notebooks/distill.ipynb`](notebooks/distill.ipynb) | Knowledge distillation experiments |
| [`notebooks/nb04_micro.ipynb`](notebooks/nb04_micro.ipynb) | MCU/ESP32 tiers: size audit, C header export, latency benchmark |
| [`notebooks/nb05_embedded.ipynb`](notebooks/nb05_embedded.ipynb) | RPi/x86 tiers: RTF benchmark, streaming sliding-window demo |
| [`notebooks/nb06_gpu.ipynb`](notebooks/nb06_gpu.ipynb) | GPU server (HuBERT): t-SNE embeddings, RTF comparison |
| [`notebooks/nb07_distill.ipynb`](notebooks/nb07_distill.ipynb) | TinyHuBERT distillation: MSE + InfoNCE student training |
| [`notebooks/nb08_wakehubert.ipynb`](notebooks/nb08_wakehubert.ipynb) | WakeHuBERT classifier on distilled ONNX featurizer, CPU-only |
| [`notebooks/nb09_ablation.ipynb`](notebooks/nb09_ablation.ipynb) | Resumable 36-cell ablation grid: featurizers × losses × augmentation |

All notebooks are Kaggle/Colab/Paperspace-ready via env vars and Kaggle Secrets for MLflow credentials.

Key env vars (all have safe defaults):

| Variable | Default | Purpose |
|----------|---------|---------|
| `WAKE_WORD` | `hey jarvis` | Target phrase |
| `OUTPUT_DIR` | `./ww_output` | Output root |
| `POPULATION` / `GENERATIONS` | 12 / 5 | Search budget |
| `TIERS_TO_TRAIN` | `micro,small,filterbank_small` | Which architectures to train |
| `FINAL_EPOCHS` | 30 | Epochs for final per-tier training |
| `SEARCH_TWO_STAGE` | `false` | Enable two-stage search |
| `SEARCH_DEMES` | 1 | Parallel islands |
| `CUSTOM_TRAIN_CSV` | — | BYO dataset path (skips datagen) |

See [docs/index.md](docs/index.md) for full env var reference.

## Scripts

All runnable scripts live under `scripts/`. Run any of them from the project root with `.venv/bin/python scripts/<subdir>/<script>.py`.

### Training (`scripts/train/`)

| Script | Purpose |
|--------|---------|
| `train_hey_mycroft.py` | Quick 30-epoch CPU run (micro tier, mfcc-40 + GRU). Good first smoke test. |
| `train_full.py` | Sequential training across 6 architectures (FFN/GRU/CNN/BCResNet/TCResNet/DSCNN). |
| `train_full_nww.py` | Genetic search + multi-loss comparison with the full NWW pool. |
| `train_ablation.py` | Sweeps all (loss × augmentation) pairs, logs heatmaps to MLflow. `--resume` skips completed cells. |
| `train_rppl.py` | RPPL loss experiment with per-epoch embedding visualisation dashboard. |
| `train_infinite.py` | Goal-based open-ended loop — trains until F1/EER targets are met. |
| `train_vc_ablation.py` | Measures the impact of voice conversion positives on F1/EER. |
| `train_parallel.py` | Trains multiple loss configs in parallel using a shared waveform cache. |
| `train_micro_genetic.py` | Genetic search for the smallest viable model (class-imbalance focus). |
| `train_sincnet_genetic.py` | Genetic search over SincNet/Gammatone featurizers. |
| `train_hey_computer.py` | Infinite training for the "hey computer" wake word. |
| `train_mfcc.sh` | Shell wrapper for a standard MFCC run. |

```bash
# Quick single-run
.venv/bin/python scripts/train/train_hey_mycroft.py

# Ablation (resumable)
.venv/bin/python scripts/train/train_ablation.py --epochs 30 --resume

# Infinite training until F1 ≥ 0.92
.venv/bin/python scripts/train/train_infinite.py --target-f1 0.92 --target-eer 0.08

# RPPL experiment with embedding plots
.venv/bin/python scripts/train/train_rppl.py --epochs 50 --arch gru --viz-every 5
```

### Evaluation & Live Inference (`scripts/eval/`)

| Script | Purpose |
|--------|---------|
| `eval_hey_mycroft.py` | Full evaluation suite — ROC/PR/DET plots, confusion matrix, FP/hour estimate. |
| `test_wakeword.py` | Test a trained model on an audio file or live mic. |
| `listen_all.py` | Run N random models in parallel with a real-time confidence dashboard. |
| `mic_test.py` | Minimal mic capture test (sounddevice). |

```bash
# Evaluate best checkpoint
.venv/bin/python scripts/eval/eval_hey_mycroft.py

# Test on a file
.venv/bin/python scripts/eval/test_wakeword.py \
    --featurizer experiments/hey_mycroft/models/best_f1_featurizer.onnx \
    --model      experiments/hey_mycroft/models/best_f1.onnx \
    --audio sample.wav

# Live dashboard (5 random models from a directory)
.venv/bin/python scripts/eval/listen_all.py \
    --models-dir experiments/hey_mycroft/ablation --max-models 5
```

### Data Management (`scripts/data/`)

| Script | Purpose |
|--------|---------|
| `download_hdd4_datasets.py` | Download NWW/augmentation datasets to `/mnt/hdd4` via git-lfs. |
| `select_training_subset.py` | Copy a size-capped subset from hdd4 into a local fast-access dir. |
| `preprocess.py` | VAD-based silence trimming + normalisation of raw audio. |
| `rebuild_dataset.py` | Rebuild dataset adding speech negatives (required for good FAR). |
| `localise_csvs.py` | Rewrite metadata CSVs to use local paths after copying from hdd4. |
| `generate_vc_positives.py` | Generate VC positives for hey_mycroft using NWW clips as speaker donors. |
| `generate_hey_computer_dataset.py` | Build the "hey computer" dataset (TTS + negatives). |
| `generate_hey_computer_vc.py` | VC positives for "hey computer". |

```bash
# Typical first-time setup for hey_mycroft
.venv/bin/python scripts/data/download_hdd4_datasets.py
.venv/bin/python scripts/data/select_training_subset.py
.venv/bin/python scripts/data/rebuild_dataset.py
```

### Research / Experimental (`scripts/research/`)

| Script | Purpose |
|--------|---------|
| `tinyhubert.py` | WakeHuBERT — streaming HuBERT distillation trainer. |
| `tinyhuberta.py` | Alternative HuBERT distillation variant. |

### Utilities (`scripts/`)

| Script | Purpose |
|--------|---------|
| `export_mfcc.py` | Export a trained model's MFCC featurizer to ONNX. |
| `export_w2vbert.py` | Export a Wav2Vec2-BERT featurizer to ONNX via optimum. |
| `train_markov_featurizer.py` | Fit and export a MarkovTransitionExtractor. |

---

## RPPL Loss + Infinite Training

Two research-focused workflows for squeezing maximum accuracy from large NWW pools:

### RPPL experiment

Trains with the Robust Prototype Diversity Loss and logs rich embedding visualisations to MLflow every N epochs:

```bash
.venv/bin/python scripts/train/train_rppl.py --epochs 50 --arch gru --viz-every 5
```

What you see in MLflow per epoch: `rppl_bce`, `rppl_proto`, `rppl_div`, `rppl_center`, `rppl_cons`, `rppl_geo_scale`, `rppl_proto_ema_norm`, plus a 6-panel dashboard PNG showing sub-loss trajectories, warmup ramp, EMA prototype stability, Fisher ratio, and silhouette score.

### Infinite training

Goal-based open-ended loop — trains until F1/EER targets are met, not until a fixed epoch count. Designed for very large NWW pools (millions of files):

```bash
# Run until F1 ≥ 0.92 and EER ≤ 0.08
.venv/bin/python scripts/train/train_infinite.py

# With on-the-fly voice conversion positives
.venv/bin/python scripts/train/train_infinite.py --vc-per-epoch 5

# Stricter targets, larger NWW scan
.venv/bin/python scripts/train/train_infinite.py \
    --target-f1 0.95 --target-eer 0.05 \
    --scan-size 20000 --arch bcresnet --loss rppl
```

Each epoch: infer on a random subset of the NWW pool → keep hard negatives → (optionally) synthesise VC positives → train → check stopping goals.

### Voice conversion backends (`ww_trainer/vc_helpers.py`)

| Backend | Mode | Notes |
|---------|------|-------|
| `chatterbox-onnx` | CPU TTS + VC | Default — cross-platform |
| `chatterbox` | GPU TTS + VC | Best quality with CUDA |
| `linacodec` | CPU/GPU VC only | 48 kHz codec-based conversion |

```bash
export WW_VC_BACKEND=chatterbox-onnx   # or chatterbox / linacodec / auto
```

---

## Key Features

- **17 feature extractors**: MFCC, FilterBank, SincNet, Gammatone, LEAF, PLP, PNCC, CQT, HuBERT, Wav2Vec2, TorchAudio-HuBERT, Markov, HMM, plus enrichment wrappers (VAD, Pitch, SNRAware, MultiResolution, Delta)
- **15 classifier heads**: FFN, OCSVM, GRU, CNN, BC-ResNet, TC-ResNet, DS-CNN, MatchboxNet, Res15, KWT, Conformer, MixConv, CRNN, EfficientNet, ConvAttention
- **17 loss functions**: BCE, Focal, ArcFace, SupCon, NTXent, RPPL (with EMA prototype + hard-neg diversity + proto-consistency), Triplet, and more
- **5 search strategies**: Optuna (Bayesian), Grid, Random, Genetic, Two-Stage Genetic
- **Island model**: `n_demes` parallel populations with ring-topology migration — `sweep.py:785`
- **Adaptive mutation**: `mutation_decay` parameter reduces mutation rate each generation — `sweep.py:599`
- **Progress callbacks**: `on_generation` fires after every generation with live stats — `sweep.py:576`
- **ONNX-first deployment**: extractor and head export independently; inference via `OnnxWakeWordInferencer`
- **Streaming inference**: `SlidingFeatureCacheTensor` for chunk-by-chunk real-time detection
- **Knowledge distillation**, **QAT**, **multi-GPU (DDP)**, **confidence calibration**
- **ESP32 tiers**: sub-1 KB models with C header export via `export_c.export_to_c_header`
- **11 hardware tier presets**: from `esp32_nano` to `hubert_medium`
- **Infinite training mode**: goal-based loop with large NWW mining + on-the-fly VC synthesis
- **RPPL dashboard**: 6-panel per-epoch embedding diagnostic, auto-logged to MLflow
- **41 runnable examples** in [`examples/`](examples/)

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/learning_path.md](docs/learning_path.md) | **Zero-to-hero curriculum** — staged learning path with literature anchors |
| [examples/README.md](examples/README.md) | 43 runnable examples — covers every featurizer / head / loss |
| [docs/index.md](docs/index.md) | Navigation hub — all modules, scripts, and key functions |
| [docs/notebooks.md](docs/notebooks.md) | Notebook curriculum guide — hardware tiers, decision tree, recommended running order |
| [docs/sweep.md](docs/sweep.md) | Full search API: parameter tables, fitness functions, island model |
| [docs/quickstart.md](docs/quickstart.md) | `QuickstartConfig` API reference |
| [docs/training.md](docs/training.md) | Step-by-step training guide, full CLI reference, infinite training, VC backends |
| [docs/extractors.md](docs/extractors.md) | All 17 feature extractors |
| [docs/classifiers.md](docs/classifiers.md) | All 15 classifier heads |
| [docs/losses.md](docs/losses.md) | All 17 loss functions including RPPL component breakdown |
| [docs/hardware_guide.md](docs/hardware_guide.md) | MCU → server tier selection |
| [docs/faq.md](docs/faq.md) | Common questions and error resolutions |
| [docs/audit.md](docs/audit.md) | Known issues and tech debt |
| [docs/suggestions.md](docs/suggestions.md) | Feature backlog and improvement proposals |
| [docs/references.md](docs/references.md) | Academic references and bibliography |
| [docs/rppl_whitepaper.md](docs/rppl_whitepaper.md) | RPPL loss technical whitepaper |
| [docs/tinyhubert_whitepaper.md](docs/tinyhubert_whitepaper.md) | TinyHuBERT distillation design |

## Contributing

Issues and pull requests welcome. Target the `dev` branch (not `master`). For
larger changes, open an issue first to discuss scope. Tests live in `test/`;
run with `uv run pytest`.

## Citation

If you use ww-trainer in academic work, please cite:

```bibtex
@software{ww_trainer,
  title  = {ww-trainer: a research suite for on-device wake-word detection},
  author = {TigreGotico contributors},
  year   = {2026},
  url    = {https://github.com/TigreGotico/ww-trainer},
  note   = {Funded by NGI0 Commons Fund / NLnet, grant 101135429}
}
```

## Credits

Funded by [NGI0 Commons Fund](https://nlnet.nl/project/OpenVoiceOS) / NLnet under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).

![](./ngi.png)

## License

Apache 2.0
