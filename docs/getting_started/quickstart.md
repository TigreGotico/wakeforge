# Quickstart — Single String to Trained ONNX Model

> **Before you start — resource budget.** The default run downloads
> several HF datasets and trains for 50 epochs. Plan for **≈ 3–8 GB disk,
> 2–5 GB download, 20–40 min on CPU** (much less on a GPU, much less
> with `--no-augmentation-data --n-positive 200`). Full breakdown by
> preset, dataset, tier, and cache location:
> [`requirements.md`](requirements.md).

Goal: go from typing a phrase like `"hey jarvis"` to a deployable ONNX model in **one command**, on a laptop, in under 10 minutes for a small smoke test.

How it works under the hood — `train_from_wakeword` — `ww_trainer/quickstart.py:231`:

1. **Synthesise positives** — TTS produces 1000 utterances of the phrase in varied voices/speeds/pitches.
2. **Mine negatives** — short common-speech clips that do *not* contain the phrase, optionally adversarial graphemes ("hay janice", "hey jarvi…").
3. **Optionally download augmentation** — background noise, music, and Room Impulse Responses to simulate distance and reverberation.
4. **Train** a chosen hardware tier (defaults to `small` — MFCC + GRU, ~50 K params).
5. **Export two ONNX files** — featurizer + head — for PyTorch-free runtime.

### When to use the quickstart vs. the full pipeline

| Use the quickstart when… | Use the full pipeline when… |
|---|---|
| You want a working detector *today* | You need < 0.5 FA/hour or > 95 % recall |
| You have no real recordings | You have real far-field user audio |
| You're prototyping a new phrase | You're shipping to production |
| You're benchmarking the framework | You're sweeping architectures / losses |

`ww_trainer-quickstart` generates a synthetic dataset and trains a wake-word detector in one command.

## Install

The quickstart needs the `datagen` extra (TTS plugins + HF `datasets`) and an
audio codec backend (`torchcodec`) on top of the core install:

```bash
uv pip install -e ".[dev,datagen,torchcodec]"
```

`[dev]` alone is **not** enough — datagen will fail with
`ModuleNotFoundError: ovos_plugin_manager` / `datasets` / `torchcodec`.

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
| `--lang` | `en` | BCP-47 language for TTS (e.g. `en-us`, `nl-nl`, `pt-br` — pass a region for best voice selection) |
| `--adversarial/--no-adversarial` | on | Grapheme hard-negatives |
| `--vc-refs` | unset | Dir of reference WAVs for voice conversion (needs `[vc-onnx]`/`[vc-torch]`) |
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

## Voice cloning (optional, recommended for quality)

TTS alone gives limited voice diversity — especially with only one plugin
installed. **Voice conversion** (VC) re-renders each synthesised positive in
the timbre of a reference speaker, multiplying effective diversity.

VC is opt-in and requires:

1. The VC extra: `uv pip install -e ".[vc-onnx]"` (CPU ONNX, recommended) or
   `".[vc-torch]"` (GPU PyTorch).
2. A directory of short reference WAVs (3–10 s of clean speech each), one
   per target speaker.

Then pass `--vc-refs`:

```bash
ww_trainer-quickstart \
  --wake-word "hey jarvis" \
  --output-dir ./hey_jarvis \
  --vc-refs ./my_voices/
```

Without `--vc-refs`, the VC step is skipped entirely — no `chatterbox_*`
dependency is loaded. With `--vc-refs` but missing extras, the quickstart
fails fast with a pointer to `[vc-onnx]`.

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
