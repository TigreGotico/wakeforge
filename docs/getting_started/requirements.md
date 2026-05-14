# Resource Requirements — Disk, Bandwidth, RAM, Time

What `ww_trainer-quickstart` will actually cost you before you run it.
Numbers marked `TBD` are empirical and will be filled in by the maintainer
at first release — please [open an issue](https://github.com/TigreGotico/ww-trainer/issues)
if your measured values differ by more than 2×.

## TL;DR — pick a preset

| Preset | Flags | Disk | Network (first run) | Peak RAM | Time, CPU | Time, GPU |
|---|---|---|---|---|---|---|
| **Smoke** | `--no-augmentation-data --n-positive 200 --epochs 10 --tier micro` | ~50 MB | ~100 MB (TTS plugin only) | ~1 GB | ~5 min | ~2 min |
| **Default** | (no flags) | TBD (≈ 3–8 GB) | TBD (≈ 2–5 GB) | ~2 GB | TBD (≈ 20–40 min) | TBD (≈ 5–10 min) |
| **Default + VC** | `--vc-refs <dir>` | + ~1 GB | + ~500 MB (Chatterbox model) | ~3 GB | + ~10 min | + ~3 min |
| **Production** | `--n-positive 5000 --epochs 100 --tier large` | TBD (≈ 15–30 GB) | TBD (≈ 5–10 GB) | ~8 GB | not recommended | TBD (≈ 1–2 h) |

CPU times assume an 8-core x86-64 laptop. GPU times assume a mid-range
consumer card (RTX 3060 or similar).

## Datasets downloaded

Sources are declared in `NEGATIVE_DATASETS` —
[`ww_trainer/datagen.py:47-73`](../../ww_trainer/datagen.py).

| Dataset | Role | Default cap | On-disk size |
|---|---|---|---|
| [`TigreGotico/ESC-50`](https://huggingface.co/datasets/TigreGotico/ESC-50) | General negatives (env. sounds) | none | ~600 MB |
| [`TigreGotico/NAR`](https://huggingface.co/datasets/TigreGotico/NAR) | General negatives | none | TBD |
| [`agkphysics/AudioSet`](https://huggingface.co/datasets/agkphysics/AudioSet) | General negatives | **5 000 samples** | TBD (≈ 1–3 GB) |
| [`TigreGotico/not-wake-words-speech-en`](https://huggingface.co/datasets/TigreGotico/not-wake-words-speech-en) | Speech negatives (primary) | none | TBD |
| [`hf-internal-testing/librispeech_asr_demo`](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_demo) | Speech negatives | ~70 clips | ~50 MB |
| [`Anton-Bushuiev/speech-commands-v2-resampled`](https://huggingface.co/datasets/Anton-Bushuiev/speech-commands-v2-resampled) | Short-command negatives | none | TBD |
| [`TigreGotico/ambient_noises`](https://huggingface.co/datasets/TigreGotico/ambient_noises) | `bg_noise/` augmentation | none | TBD |
| [`TigreGotico/building_106_kitchen_3secs`](https://huggingface.co/datasets/TigreGotico/building_106_kitchen_3secs) | `bg_noise/` augmentation | none | TBD |
| [`TigreGotico/public_domain_sounds_3secs`](https://huggingface.co/datasets/TigreGotico/public_domain_sounds_3secs) | `bg_noise/` augmentation | none | TBD |
| [`TigreGotico/FMA_3secs`](https://huggingface.co/datasets/TigreGotico/FMA_3secs) | `music/` augmentation | none | TBD |
| [`davidscripka/MIT_environmental_impulse_responses`](https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses) | `rir/` augmentation | none | TBD |

Pass `--no-augmentation-data` to skip every row in the bottom four
categories — that alone saves several GB on the default preset.

## Synthetic positives — derivable from defaults

TTS produces 16 kHz mono WAVs of ~1.5 s each. Per-sample size:

```
16 000 Hz × 1.5 s × 2 B/sample ≈ 48 KB
```

So for the default `n_positive=1000`
(see [`QuickstartConfig` — `ww_trainer/quickstart.py:50`](../../ww_trainer/quickstart.py)):

| n_positive | Pre-VAD | Post-VAD trim |
|---|---|---|
| 200 (smoke) | ~10 MB | ~7 MB |
| 1 000 (default) | ~48 MB | ~35 MB |
| 5 000 (production) | ~240 MB | ~175 MB |

VAD trim is [`trim_silence_vad` — `ww_trainer/datagen.py:111`](../../ww_trainer/datagen.py),
typically 10–30 % smaller after silence removal.

## Voice conversion (optional, opt-in via `--vc-refs`)

The first call to `voice_convert_batch` downloads the Chatterbox-ONNX model
from HuggingFace and caches it under `~/.cache/huggingface/hub/`. VC is
**only triggered when you pass `--vc-refs <dir>`** — there is no implicit
download otherwise.

| Backend | Extra | First-time download | RAM during VC |
|---|---|---|---|
| `chatterbox-onnx` (CPU, recommended) | `[vc-onnx]` | ~500 MB | ~2 GB |
| `chatterbox-tts` (GPU) | `[vc-torch]` | ~1 GB | ~3 GB + VRAM |

See [`voice_convert_batch` — `ww_trainer/datagen.py:349`](../../ww_trainer/datagen.py).

## Trained-model footprint

Tier definitions live in [`ww_trainer/tiers.py:29-185`](../../ww_trainer/tiers.py).
ONNX sizes are estimates for the **head** only; SSL tiers also ship a
~300 MB–2 GB frozen featurizer ONNX alongside.

| Tier | Params | Head ONNX (int8) | Featurizer ONNX |
|---|---|---|---|
| `esp32_nano` | ~240 | < 1 KB | n/a (MFCC built-in) |
| `esp32_sweet` | ~1 K | ~5 KB | n/a |
| `esp32_max` | ~2 K | ~10 KB | n/a |
| `micro` | ~50 K | ~200 KB | n/a |
| `small` (default) | ~200 K | ~800 KB | n/a |
| `efficientnet_small` | ~4 M | ~16 MB | n/a |
| `ssl_small` | ~200 K | ~800 KB | ~300 MB (TinyHuBERT) |
| `large` | ~2 M | ~8 MB | ~1–2 GB (HuBERT base) |

Run `ww_trainer-tiers` to print the live table.

## Where caches live

| Cache | Default path | Override |
|---|---|---|
| HuggingFace datasets + models | `~/.cache/huggingface/` | `HF_HOME` |
| PyTorch hub | `~/.cache/torch/hub/` | `TORCH_HOME` |
| OVOS plugin TTS audio | `~/.cache/mycroft/` | `XDG_CACHE_HOME` |

These caches are **shared across all `ww_trainer-quickstart` runs**. The
first run pays the full download cost; subsequent runs reuse everything
that hasn't been pruned.

## How to slim down a run

In order of impact:

1. `--no-augmentation-data` — drops bg_noise/music/RIR. Saves most of the
   network + disk on the default preset.
2. `--reuse-dataset` — skip datagen entirely on subsequent runs; train
   on a previously generated `dataset/` directory.
3. `--n-positive 200` — fewer TTS calls (and fewer negatives, since
   the trainer matches the ratio).
4. `--tier esp32_nano` / `micro` — smallest models, fastest training,
   no SSL featurizer.
5. Skip `--vc-refs` — no Chatterbox download, no VC pass.

## Verifying these numbers yourself

```bash
du -sh ./hey_jarvis/dataset/*
du -sh ~/.cache/huggingface/datasets/
du -sh ~/.cache/huggingface/hub/
/usr/bin/time -v ww_trainer-quickstart --wake-word "hey jarvis" \
    --output-dir ./hey_jarvis --no-augmentation-data --n-positive 200
```

If your numbers differ materially from the table at the top of this page,
please open an issue with the four `du` outputs and your OS — the docs
should match reality, not the maintainer's laptop.
