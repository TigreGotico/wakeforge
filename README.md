# Wake Word Trainer

Full training and research suite for wake word detection — from microcontrollers to GPU servers. All components export to ONNX; production inference requires only `onnxruntime` and `numpy`.

## Install

```bash
uv pip install -e ".[dev]"
# Optional extras
uv pip install -e ".[sweep]"        # Optuna Bayesian search
uv pip install -e ".[transformers]" # HuBERT / Wav2Vec2 extractors
uv pip install -e ".[mlflow]"       # experiment tracking
uv pip install -e ".[datagen]"      # synthetic dataset generation
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

`run_genetic_search` — `ww_trainer/sweep.py:543`
`run_two_stage_genetic_search` — `ww_trainer/sweep.py:662`

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

## Notebook

`notebooks/genetic_search.ipynb` — end-to-end: dataset synthesis → genetic search → multi-tier training → ONNX export → benchmark. Kaggle/Colab/Paperspace-ready via env vars.

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

## RPPL Loss + Infinite Training

Two research-focused workflows for squeezing maximum accuracy from large NWW pools:

### RPPL experiment (`train_rppl.py`)

Trains with the Robust Prototype Diversity Loss and logs rich embedding visualisations to MLflow every N epochs:

```bash
.venv/bin/python train_rppl.py --epochs 50 --arch gru --viz-every 5
```

What you see in MLflow per epoch: `rppl_bce`, `rppl_proto`, `rppl_div`, `rppl_center`, `rppl_cons`, `rppl_geo_scale`, `rppl_proto_ema_norm`, plus a 6-panel dashboard PNG (`rppl/rppl_dashboard.png`) showing sub-loss trajectories, warmup ramp, EMA prototype stability, Fisher ratio, and silhouette score.

### Infinite training (`train_infinite.py`)

Goal-based open-ended loop — trains until F1/EER targets are met, not until a fixed epoch count. Designed for very large NWW pools (millions of files):

```bash
# Run until F1 ≥ 0.92 and EER ≤ 0.08
.venv/bin/python train_infinite.py

# With on-the-fly voice conversion positives
.venv/bin/python train_infinite.py --vc-per-epoch 5 --vc-text "hey mycroft"

# Stricter targets, larger NWW scan
.venv/bin/python train_infinite.py \
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
- **11 classifier heads**: FFN, GRU, CNN, BC-ResNet, TC-ResNet, DS-CNN, MatchboxNet, Res15, KWT, Conformer, CRNN
- **17 loss functions**: BCE, Focal, ArcFace, SupCon, NTXent, RPPL (with EMA prototype + hard-neg diversity + proto-consistency), Triplet, and more
- **5 search strategies**: Optuna (Bayesian), Grid, Random, Genetic, Two-Stage Genetic
- **Island model**: `n_demes` parallel populations with ring-topology migration — `sweep.py:620`
- **Adaptive mutation**: `mutation_decay` parameter reduces mutation rate each generation — `sweep.py:530`
- **Progress callbacks**: `on_generation` fires after every generation with live stats — `sweep.py:551`
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
| [docs/index.md](docs/index.md) | Navigation hub — all modules, scripts, and key functions |
| [docs/sweep.md](docs/sweep.md) | Full search API: parameter tables, fitness functions, island model |
| [docs/quickstart.md](docs/quickstart.md) | `QuickstartConfig` API reference |
| [docs/training.md](docs/training.md) | Step-by-step training guide, full CLI reference, infinite training, VC backends |
| [docs/extractors.md](docs/extractors.md) | All 17 feature extractors |
| [docs/classifiers.md](docs/classifiers.md) | All 11 classifier heads |
| [docs/losses.md](docs/losses.md) | All 17 loss functions including RPPL component breakdown |
| [docs/hardware_guide.md](docs/hardware_guide.md) | MCU → server tier selection |
| [FAQ.md](FAQ.md) | Common questions and error resolutions |
| [AUDIT.md](AUDIT.md) | Known issues and tech debt |

## Credits

Funded by [NGI0 Commons Fund](https://nlnet.nl/project/OpenVoiceOS) / NLnet under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).

![](./ngi.png)

## License

Apache 2.0
