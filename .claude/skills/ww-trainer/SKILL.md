---
name: ww-trainer
description: "Train, evaluate, and export wake-word detection models using the ww-trainer toolkit (pip: ww_trainer). Covers dataset generation, single-run training, genetic HP search, multi-tier experiments, ONNX export, and live inference testing."
---

Check availability:
```bash
python -c "import ww_trainer; print(ww_trainer.__version__)"
```

Install:
```bash
pip install ww_trainer                                      # minimal
pip install "ww_trainer[datagen]"                          # +TTS dataset generation
pip install "ww_trainer[datagen,mlflow,vc-onnx,sweep,viz]" # full research stack
```

---

## Core concept — two ONNX files always required

Every trained model exports two files. Both are required for inference:
- `*_featurizer.onnx` — feature extractor (MFCC, FilterBank, SincNet, …)
- `*.onnx` — classifier head (GRU, FFN, EfficientNet, …)

```python
from ww_trainer.inference import OnnxWakeWordInferencer
model = OnnxWakeWordInferencer("featurizer.onnx", "head.onnx")
score = model.infer(wav_float32_array)  # float in [0, 1]
```

---

## CLI entry points

All installed by `pip install ww_trainer`:

| Command | Purpose |
|---------|---------|
| `ww_trainer-quickstart` | Full pipeline: datagen → train → ONNX export |
| `ww_trainer-train` | Train on an existing CSV dataset |
| `ww_trainer-datagen` | Generate a dataset only (no training) |
| `ww_trainer-infer` | Score a WAV file with a trained ONNX model |
| `ww_trainer-benchmark` | Latency/RTF benchmark across tiers |

### ww_trainer-quickstart (most common)
```bash
ww_trainer-quickstart \
  --wake-word "hey jarvis" \
  --output-dir ./hey_jarvis \
  --tier small \
  --epochs 50 \
  --n-positive 1000 \
  --reuse-dataset          # skip datagen if dataset exists
```

### ww_trainer-train (BYO dataset)
```bash
ww_trainer-train \
  --wake-word "hey jarvis" \
  --metadata dataset/train.csv \
  --test-metadata dataset/test.csv \
  --tier small \
  --epochs 50 \
  --arch gru \
  --loss-type focal \
  --export-onnx \
  --output-dir ./model
```

Key flags: `--arch` (gru/ffn/cnn/bcresnet/tcresnet/dscnn/matchboxnet/res15/kwt/conformer/crnn/efficientnet), `--loss-type` (bce/focal/arcface/supcon/ntxent/rppl/triplet), `--device` (auto/cpu/cuda), `--calibrate`, `--use-vad`, `--export-c` (ESP32 C header).

### ww_trainer-datagen
```bash
ww_trainer-datagen \
  --wake-word "hey jarvis" \
  --output-dir ./dataset \
  --n-positive 1000 \
  --lang en \
  --vc-refs ./donor_voices/   # optional voice conversion positives
```

### ww_trainer-infer
```bash
ww_trainer-infer \
  --featurizer model/best_f1_featurizer.onnx \
  --model      model/best_f1.onnx \
  --audio      sample.wav \
  --threshold  0.5
# exits 0 if detected, 1 if not
```

### ww_trainer-benchmark
```bash
ww_trainer-benchmark --tiers micro small filterbank_small --device cpu
```

---

## Python API

### 1. Zero-to-ONNX
```python
from ww_trainer.quickstart import train_from_wakeword

result = train_from_wakeword(
    "hey jarvis",
    "./hey_jarvis",
    tier="small",
    epochs=50,
    n_positive=1000,
    lang="en",
    reuse_dataset=True,    # skip datagen if dataset exists
    device="auto",
)
print(result.best_onnx_path)  # path to head ONNX
print(result.metrics)         # {"f1": ..., "precision": ..., "recall": ...}
```

### 2. Dataset generation only
```python
from ww_trainer.datagen import DatagenConfig, run_datagen_pipeline

result = run_datagen_pipeline(DatagenConfig(
    wake_word="hey jarvis",
    output_dir="./dataset",
    n_positive=1000,
    lang="en",
    adversarial=True,           # phonetically-similar hard negatives
    vad_trim=True,
    download_augmentation=True, # bg-noise/music/RIR from HF
))
# result.train_csv, result.test_csv, result.bg_noise_dir
```

Pre-built HF datasets (skips TTS synthesis):
`alexa`, `hey_mycroft`, `hey_siri`, `wake_up`, `hey_computer`, `voice_assistant`, `home_assistant`

### 3. Inference
```python
from ww_trainer.inference import OnnxWakeWordInferencer
import numpy as np

model = OnnxWakeWordInferencer("featurizer.onnx", "head.onnx")
wav = np.zeros(16000, dtype=np.float32)
score = model.infer(wav)              # float [0,1]

# Streaming
for chunk in audio_stream:
    score, cache = model.infer_streaming(chunk, cache)
    if score > 0.5:
        print("detected")
```

### 4. Genetic hyperparameter search
```python
from ww_trainer.sweep import run_genetic_search, run_two_stage_genetic_search

results = run_genetic_search(
    "dataset/train.csv",
    population_size=20, generations=10,
    fitness_fn="exp_f1",  # "f1" | "exp_f1" | "double_exp_f1"
    n_demes=2,
    target_f1=0.90,
)
print(results["best_config"], results["best_score"])
```

---

## Hardware tiers

Pass `tier=` to quickstart/train. Controls both extractor and head architecture.

| Tier | Extractor | Approx params | Target |
|------|-----------|---------------|--------|
| `esp32_nano` | MFCC-13 + FFN-16 | ~241 | ESP32 / ATmega |
| `esp32_sweet` | MFCC-13 + FFN-64 | ~1 K | ESP32 |
| `esp32_max` | MFCC-13 + FFN-128 | ~2 K | ESP32-S3 |
| `micro` | MFCC-40 + FFN | ~50 K | RPi Zero / MCU |
| `delta_micro` | MFCC-13 Δ+ΔΔ + FFN | ~55 K | MCU |
| `small` | MFCC-40 + GRU | ~200 K | RPi 3/4 |
| `filterbank_small` | FilterBank + GRU | ~200 K | Embedded SBC |
| `gammatone_small` | Gammatone + GRU | ~200 K | Embedded SBC |
| `sincnet_small` | SincNet + GRU | ~300 K | Low-power x86 |
| `efficientnet_small` | FilterBank + EfficientNet-B0 | ~4 M | RPi 4 / x86 |
| `ssl_small` | ONNX featurizer + FFN | ~200 K head | x86 / GPU server |
| `ssl_medium` | ONNX featurizer + GRU | ~1 M head | GPU server |

**CPU-safe** (no GPU required): all tiers except `ssl_small` and `ssl_medium` (which
depend on the loaded ONNX featurizer size).

**SSL tiers** require a pre-exported featurizer ONNX. Export once, then use for both training and inference:
```bash
.venv/bin/python scripts/export_hubert.py --model voidful/hubert-tiny-v2 --output hubert-tiny-v2.onnx
# then: --tier ssl_small --featurizer-type onnx --featurizer hubert-tiny-v2.onnx
```
Pre-exported variants: https://huggingface.co/TigreGotico/onnx-feature-extractors

---

## Loss functions

| Name | When to use |
|------|-------------|
| `bce` | Default baseline |
| `focal` | Class-imbalanced datasets |
| `arcface` | Metric learning; large NWW pools |
| `supcon` | Supervised contrastive |
| `ntxent` | NT-Xent contrastive |
| `rppl` | Best for large NWW pools + hard-negative mining |
| `triplet` | Triplet margin |
| `size_aware` | ESP32 — penalises model size |

Combine: `losses_cfg=[{"name": "bce", "weight": 0.5}, {"name": "arcface", "weight": 0.5}]`

---

## Pitfalls

- **Two ONNX files required** — featurizer path first, head path second. One arg → `TypeError`.
- **16 kHz mono float32** — resample before `infer()`.
- **`reuse_dataset=True`** — always set on re-runs; TTS synthesis is slow.
- **`ssl_small`/`ssl_medium` tiers**: must export featurizer ONNX first (`scripts/export_hubert.py` etc.) — HuBERT/Wav2Vec2 training wrappers were removed; use `OnnxFeatureExtractor` for training too (guarantees feature parity with inference).
- **`efficientnet` head needs `torchvision`**: `pip install torchvision`.
- **`trust_remote_code` removed in datasets ≥ 3.x** — do not pass to `load_dataset`.
