# Quickstart — Single String to Trained ONNX Model

> **Before you start — resource budget.** The default run downloads
> several HF datasets (≈ 2.9 GB upstream) plus another ~5 GB if you opt
> into voice cloning. Plan for **≈ 6–8 GB free disk and 5 GB download**
> on the default preset, **~30–60 min on CPU** (or ~10–15 min on a
> mid-range GPU). Smoke runs with `--no-augmentation-data --n-positive
> 200` need ≈ 1.5 GB of disk. With a novel phrase they take about **70 min
> on an 8-core CPU**, almost all of it in datagen rather than training: one
> HTTP round trip per downloaded file (ESC-50 ≈ 2 000, NAR ≈ 850,
> `not-wake-words-speech-en` up to 10 000), AudioSet streamed as parquet
> and capped at 5 000 rows, and about 11 min of sequential edge-tts
> synthesis. Each negative source is capped at `--n-positive` clips by
> default (`--max-negative`; `0` takes every clip), so the file counts
> above apply only to a larger cap or `--max-negative 0`. Synthesis needs internet access, because
> edge-tts calls an online service. A phrase with a prebuilt positives
> dataset (such as `"hey mycroft"`) skips the synthesis. Full per-dataset
> breakdown: [`requirements.md`](requirements.md).

Goal: go from typing a phrase like `"hey toaster"` to a deployable ONNX model in **one command**, on a laptop. Training a micro-tier model takes minutes; the wall-clock time of a first run is set by the dataset downloads and the TTS synthesis described above.

How it works under the hood — `train_from_wakeword` — `ww_trainer/quickstart.py:290`:

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
  --wake-word "hey toaster" \
  --output-dir ./hey_toaster \
  --tier small \
  --epochs 50 \
  --n-positive 1000 \
  --no-augmentation-data
```

`--tier small`, `--epochs 50` and `--n-positive 1000` repeat the defaults and
can be dropped. `--no-augmentation-data` skips the background-noise, music
and RIR downloads; the negative datasets are still downloaded.

On completion:
```
✓ Dataset: ./hey_toaster/dataset
✓ Model:   ./hey_toaster/model/best_f1.onnx  (F1=0.923)
```

That F1 score is a training-set metric. It is not a measured FA/hour or
real-world recall figure — see
[`../guides/expectations.md`](../guides/expectations.md) for what those
numbers mean and how to measure them before shipping this model.

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
| `--vc-refs` | unset | Dir of reference WAVs for voice conversion (needs the `[vc]` extra) |
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

The CLI exports both automatically when `--export-onnx` is on (default). Pass both paths to `OnnxWakeWordInferencer` — `ww_trainer/inference.py:137`:

```python
from ww_trainer.inference import OnnxWakeWordInferencer

model = OnnxWakeWordInferencer(
    "hey_toaster/model/best_f1_featurizer.onnx",
    "hey_toaster/model/best_f1.onnx",
)
score = model.infer(wav_float32_array)  # float in [0, 1]
```

The input is 16 kHz mono float32. `OnnxWakeWordInferencer.infer` does not
resample. `ww_trainer-infer` resamples only when `soundfile` is missing
(through `torchaudio`); with `soundfile` installed it prints a warning for
another sample rate and scores the clip as if it were 16 kHz, which gives a
wrong score. Pass 16 kHz mono, for example after
`ffmpeg -i in.wav -ar 16000 -ac 1 clip.wav`.

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

1. The VC extra: `uv pip install -e ".[vc]"` (pure-ONNX `voiceclonnx`,
   zero PyTorch at runtime). `[vc-onnx]` is an alias of `[vc]`; there is no
   `[vc-torch]` extra.
2. A directory of short reference WAVs (3–10 s of clean speech each), one
   per target speaker.

Then pass `--vc-refs`:

```bash
ww_trainer-quickstart \
  --wake-word "hey toaster" \
  --output-dir ./hey_toaster \
  --vc-refs ./my_voices/
```

Without `--vc-refs`, the VC step is skipped entirely — no `voiceclonnx`
dependency is loaded. With `--vc-refs` but missing extras, the quickstart
fails fast with a pointer to `[vc]`.

## Skipping Datagen

If you already have a dataset from `ww_trainer-datagen`, pass `--reuse-dataset` to skip the generation step and jump straight to training:

```bash
ww_trainer-quickstart \
  --wake-word "hey toaster" \
  --output-dir ./hey_toaster \
  --reuse-dataset
```

The dataset directory must follow the layout above.

## Source References

- `train_from_wakeword` — `ww_trainer/quickstart.py:290`
- `QuickstartConfig` — `ww_trainer/quickstart.py:23`
- `QuickstartResult` — `ww_trainer/quickstart.py:97`
- `_run_or_load_datagen` — `ww_trainer/quickstart.py:123`
- `_train_from_datagen_result` — `ww_trainer/quickstart.py:189`
