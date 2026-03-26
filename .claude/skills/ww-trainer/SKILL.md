---
name: ww-trainer
description: "Train, evaluate, and export wake-word detection models using the ww-trainer toolkit (pip: ww_trainer). Covers dataset generation, single-run training, genetic HP search, multi-tier experiments, ONNX export, and live inference testing."
---

You have access to the `ww_trainer` Python package. It may be installed as a pip package or available as a local git clone. Check availability with:

```bash
python -c "import ww_trainer; print(ww_trainer.__version__)"
```

If not installed, install it:

```bash
# Minimal (training only)
pip install ww_trainer

# With dataset generation (TTS + VAD)
pip install "ww_trainer[datagen]"

# With MLflow tracking
pip install "ww_trainer[mlflow]"

# With voice conversion (CPU)
pip install "ww_trainer[vc-onnx]"

# With hyperparameter search
pip install "ww_trainer[sweep]"

# Full research stack
pip install "ww_trainer[datagen,mlflow,vc-onnx,sweep,viz]"
```

---

## Core Concepts

**Every model produces two ONNX files — both required for inference:**
- `<name>_featurizer.onnx` — feature extractor (MFCC, SincNet, FilterBank, …)
- `<name>.onnx` — classifier head (GRU, FFN, CNN, …)

```python
from ww_trainer.inference import OnnxWakeWordInferencer
model = OnnxWakeWordInferencer("featurizer.onnx", "head.onnx")
score = model.infer(wav_float32_array)  # float in [0, 1]
```

---

## 1. Zero-to-ONNX (quickstart)

The fastest path from a wake-word string to a deployable ONNX model:

```python
from ww_trainer.quickstart import train_from_wakeword

result = train_from_wakeword(
    "hey jarvis",          # wake-word phrase
    "./hey_jarvis",        # output directory
    tier="small",          # hardware tier (see §Tiers)
    epochs=50,
    device="auto",         # "auto" picks cuda > mps > cpu
    n_positive=500,        # TTS samples to synthesise
    lang="en",             # BCP-47 language code
    download_augmentation=True,  # download bg-noise/music/RIR from HF
    reuse_dataset=True,    # skip datagen if dataset already exists
)

print(result.best_onnx_path)   # path to head ONNX
print(result.metrics)          # {"f1": ..., "precision": ..., "recall": ...}
```

`train_from_wakeword` handles everything: dataset download/synthesis → training → ONNX export. It is safe to re-run — if the dataset already exists it is reused.

**CLI equivalent:**
```bash
ww_trainer-quickstart --wake-word "hey jarvis" --output-dir ./hey_jarvis --tier small --epochs 50
```

---

## 2. Dataset generation only

```python
from ww_trainer.datagen import DatagenConfig, run_datagen_pipeline

result = run_datagen_pipeline(DatagenConfig(
    wake_word="hey jarvis",
    output_dir="./dataset",
    n_positive=500,
    lang="en",
    adversarial=True,           # add phonetically-similar hard negatives
    vad_trim=True,              # strip silence with Silero VAD
    download_augmentation=True, # bg-noise, music, RIR from HF
    seed=42,
))

# result.train_csv, result.test_csv  — metadata CSVs
# result.bg_noise_dir, result.music_dir, result.rir_dir  — augmentation dirs
```

Known wake words with pre-built HF datasets (downloaded instead of synthesised):
`alexa`, `hey_mycroft`, `hey_siri`, `wake_up`, `hey_computer`, `voice_assistant`, `home_assistant`

---

## 3. Hardware tiers

Pick based on target deployment. Tiers control both the feature extractor and classifier architecture.

| Tier | Extractor | Params | Target |
|------|-----------|--------|--------|
| `esp32_nano` | MFCC-13 | ~5 K | ESP32 / ATmega |
| `esp32_sweet` | MFCC-20 | ~15 K | ESP32 with more flash |
| `esp32_max` | MFCC-26 | ~30 K | ESP32-S3 |
| `micro` | MFCC-20 | ~50 K | RPi Zero / MCU |
| `delta_micro` | MFCC+Δ+ΔΔ | ~75 K | MCU with more flash |
| `small` | MFCC-40 | ~200 K | RPi 3/4 |
| `filterbank_small` | FilterBank | ~250 K | Embedded SBC |
| `gammatone_small` | Gammatone | ~250 K | Embedded SBC |
| `sincnet_small` | SincNet | ~350 K | Low-power x86 |
| `hubert_small` | HuBERT (frozen) | ~100 M | GPU server |
| `hubert_medium` | HuBERT (frozen) | ~300 M | GPU server |

**CPU-safe tiers** (no GPU required): `esp32_*`, `micro`, `delta_micro`, `small`, `filterbank_small`, `gammatone_small`, `sincnet_small`

---

## 4. Genetic hyperparameter search

When a single training run isn't enough — searches over lr, batch_size, hidden_dim, and optionally featurizer/arch/loss:

```python
from ww_trainer.sweep import run_genetic_search, run_two_stage_genetic_search

# Single-stage
results = run_genetic_search(
    "dataset/train/metadata.csv",
    population_size=20,
    generations=10,
    fitness_fn="exp_f1",   # "f1" | "exp_f1" | "double_exp_f1"
    n_demes=2,             # parallel island populations
    target_f1=0.90,        # early stop when reached
)

# Two-stage (broad coarse → fine-tune)
results = run_two_stage_genetic_search(
    "dataset/train/metadata.csv",
    population_size=20, generations=10,
    stage2_population=10, stage2_generations=5,
)

print(results["best_config"], results["best_score"])
```

---

## 5. Multi-tier experiment grid

Compare tiers, losses, and augmentation levels systematically — each cell writes a JSON result and is resumable:

```python
import json
from pathlib import Path
from ww_trainer.quickstart import train_from_wakeword

results = []
for tier in ["micro", "small", "filterbank_small"]:
    for loss in ["bce", "focal", "rppl"]:
        result_file = Path(f"results/{tier}_{loss}.json")
        if result_file.exists():
            results.append(json.loads(result_file.read_text()))
            continue
        r = train_from_wakeword(
            "hey jarvis", f"./models/{tier}_{loss}",
            tier=tier, epochs=30,
            losses_cfg=[{"name": loss, "weight": 1.0}],
            reuse_dataset=True,
        )
        row = {"tier": tier, "loss": loss, **r.metrics}
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(json.dumps(row))
        results.append(row)
```

---

## 6. Loss functions

| Name | When to use |
|------|-------------|
| `bce` | Default baseline |
| `focal` | Class-imbalanced datasets |
| `arcface` | Metric-learning style; improves with large NWW pools |
| `supcon` | Supervised contrastive; good positive diversity helps |
| `ntxent` | NT-Xent contrastive |
| `rppl` | RPPL — best for large NWW pools + hard-negative mining |
| `triplet` | Triplet margin loss |
| `size_aware` | ESP32 targets — penalises model size |

Configure in trainer:
```python
losses_cfg=[{"name": "focal", "weight": 1.0}]
# or combine:
losses_cfg=[{"name": "bce", "weight": 0.5}, {"name": "arcface", "weight": 0.5}]
```

---

## 7. Inference

```python
from ww_trainer.inference import OnnxWakeWordInferencer
import numpy as np

model = OnnxWakeWordInferencer("featurizer.onnx", "head.onnx")

# Single clip
wav = np.zeros(16000, dtype=np.float32)  # 1 s of silence at 16 kHz
score = model.infer(wav)                 # float in [0, 1]

# Streaming (chunk by chunk)
for chunk in audio_stream:
    score = model.infer_stream(chunk)
    if score > 0.5:
        print("Wake word detected!")
```

---

## 8. Evaluation

```python
from ww_trainer.evaluation import compute_fitness_score, compute_readiness

fitness = compute_fitness_score(model, test_data)   # F1, EER, AUC
readiness = compute_readiness(model, test_data)     # pass/fail deployment check
```

Key metrics: `f1`, `precision`, `recall`, `eer` (Equal Error Rate), `auc`, `far` (False Accept Rate per hour).

---

## 9. ONNX export (manual)

```python
from ww_trainer.trainer import WakeWordTrainer

trainer = WakeWordTrainer(arch="gru", featurizer_type="mfcc", ...)
trainer.train(output_dir="./model", train_data=..., test_data=..., epochs=50)

# Export featurizer separately (always required for inference)
trainer.model.load_checkpoint("./model/best_f1.pt")
trainer.model.feature_extractor.export_to_onnx("./model/best_f1_featurizer.onnx")
```

---

## 10. Notebooks (if repo cloned)

| Notebook | Purpose |
|----------|---------|
| `notebooks/kaggle_quickstart.ipynb` | Zero-to-ONNX, Kaggle/Colab-ready |
| `notebooks/kaggle_experiments.ipynb` | Resumable experiment grid with plots |
| `notebooks/kaggle_infinite.ipynb` | Infinite training: smoke test, full run, VC synthesis, pool ablation |
| `notebooks/genetic_search.ipynb` | Genetic HP search + multi-tier benchmark |
| `notebooks/distill.ipynb` | HuBERT → TinyHuBERT distillation |

---

## Common pitfalls

- **Two ONNX files required**: `OnnxWakeWordInferencer` takes two positional args — featurizer path first, then head path. Passing one raises `TypeError`.
- **16 kHz mono float32**: all audio must be resampled to 16 000 Hz, mono, float32 before calling `infer()`.
- **HuBERT / Wav2Vec2 tiers need GPU**: `hubert_small` / `hubert_medium` are impractical on CPU.
- **`reuse_dataset=True`**: always set this when re-running training; datagen re-synthesis is slow.
- **`trust_remote_code` removed in datasets ≥ 3.x**: do not pass it to `load_dataset`.
