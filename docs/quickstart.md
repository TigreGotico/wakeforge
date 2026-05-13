# Quickstart — Single String to Trained ONNX Model

`ww_trainer-quickstart` generates a synthetic dataset and trains a wake-word detector in one command.

## CLI

```bash
ww_trainer-quickstart \
  --wake-word "hey jarvis" \
  --output-dir ./hey_jarvis \
  --tier small \          # default
  --epochs 50 \           # default
  --n-positive 1000 \     # default
  --no-augmentation-data  # skip HF downloads (faster smoke test)
```

On completion:
```
✓ Dataset: ./hey_jarvis/dataset
✓ Model:   ./hey_jarvis/model/best_f1.onnx  (F1=0.923)
```

All options:

| Flag | Default | Description |
|------|---------|-------------|
| `--wake-word` | required | Wake-word phrase |
| `--output-dir` | required | Root output directory |
| `--tier` | `small` | Hardware tier (see `ww_trainer-tiers`) |
| `--epochs` | `50` | Training epochs |
| `--batch-size` | `16` | Batch size |
| `--lr` | `5e-4` | Learning rate |
| `--n-positive` | `1000` | Positive samples to synthesise |
| `--lang` | `en` | BCP-47 language for TTS |
| `--adversarial/--no-adversarial` | on | Grapheme hard-negatives |
| `--augmentation-data/--no-augmentation-data` | on | Download bg_noise/music/RIR |
| `--reuse-dataset` | off | Skip datagen if dataset exists |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--export-onnx/--no-export-onnx` | on | Export ONNX after training |
| `--seed` | `42` | Random seed |

## Python API

```python
from ww_trainer.quickstart import train_from_wakeword

result = train_from_wakeword(
    "hey jarvis",
    "./hey_jarvis",
    tier="micro",
    epochs=2,
    download_augmentation=False,
)
print(result.best_onnx_path)   # Path to best_f1.onnx
print(result.metrics)          # {"f1": 0.923}
```

`train_from_wakeword` accepts any `QuickstartConfig` field as a keyword argument.

## Two-File ONNX Contract

Production inference always requires **two** ONNX files:

| File | Role |
|------|------|
| `best_f1_featurizer.onnx` | Feature extractor — input: `[B, T]` float32 waveform; output: `[B, T_frames, F]` |
| `best_f1.onnx` | Classifier head — input: `[B, T_frames, F]`; output: `[B]` logit |

The CLI exports both automatically when `--export-onnx` is on (default). Pass both paths to `OnnxWakeWordInferencer` — `ww_trainer/inference.py:8`:

```python
from ww_trainer.inference import OnnxWakeWordInferencer

model = OnnxWakeWordInferencer(
    "hey_jarvis/model/best_f1_featurizer.onnx",
    "hey_jarvis/model/best_f1.onnx",
)
score = model.infer(wav_float32_array)  # float in [0, 1]
```

## Output Layout

```
<output_dir>/
  dataset/
    train/metadata.csv
    test/metadata.csv
    <slug>/positives/
    <slug>/negatives/
    augmentation/bg_noise/   # if download_augmentation=True
    augmentation/music/
    augmentation/rir/
  model/
    best_f1.pt
    best_f1_featurizer.onnx  # if export_onnx=True (featurizer)
    best_f1.onnx             # if export_onnx=True (head)
    metrics_log.csv
```

## Skipping Datagen

If you already have a dataset from `ww_trainer-datagen`, pass `--reuse-dataset` to skip the generation step and jump straight to training:

```bash
ww_trainer-quickstart \
  --wake-word "hey jarvis" \
  --output-dir ./hey_jarvis \
  --reuse-dataset
```

The dataset directory must follow the layout above.

## Source References

- `train_from_wakeword` — `ww_trainer/quickstart.py:231`
- `QuickstartConfig` — `ww_trainer/quickstart.py:23`
- `QuickstartResult` — `ww_trainer/quickstart.py:71`
- `_run_or_load_datagen` — `ww_trainer/quickstart.py:97`
- `_train_from_datagen_result` — `ww_trainer/quickstart.py:144`
