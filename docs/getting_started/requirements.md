# Resource Requirements — Disk, Bandwidth, RAM, Time

What `ww_trainer-quickstart` will actually cost you before you run it.
Sizes below come from the **HuggingFace API tree endpoint**, queried
2026-05-14. They will not change unless the upstream repos are re-uploaded.
RAM and time figures are empirical estimates from the maintainer's test
runs — please [open an issue](https://github.com/TigreGotico/ww-trainer/issues)
if your measured values differ by more than 2×.

## TL;DR — pick a preset

| Preset | Flags | Disk (HF cache + outputs) | Network, first run | Peak RAM | Time, CPU | Time, GPU |
|---|---|---|---|---|---|---|
| **Smoke** | `--no-augmentation-data --n-positive 200 --epochs 10 --tier micro` | ~1.5 GB | ~1.3 GB | ~1.5 GB | ~10 min | ~3 min |
| **Default** | (no flags) | **~6–8 GB** | **~5 GB** | ~3 GB | ~30–60 min | ~10–15 min |
| **Default + VC** | `--vc-refs <dir>` | **+ ~5 GB** | **+ ~5 GB** (Chatterbox ONNX) | ~5 GB | + ~15 min | + ~5 min |
| **Production** | `--n-positive 5000 --epochs 100 --tier large` | **~15–20 GB** | ~7 GB | ~8 GB | not recommended | ~1–2 h |

Why "HF cache + outputs": HF caches the raw dataset under
`~/.cache/huggingface/` **and** ww-trainer re-writes each used sample as
a 16 kHz mono WAV into `<output-dir>/dataset/`. The two are not the same
bytes, so plan for both. CPU times assume an 8-core x86-64 laptop;
GPU times a mid-range consumer card (RTX 3060 or similar).

## Datasets downloaded

Sources are declared in `NEGATIVE_DATASETS` —
[`ww_trainer/datagen.py:47-73`](../../ww_trainer/datagen.py).
"Upstream" is the full repo size on HF (what ends up in the HF cache if
loaded non-streaming). "Local output" is what ww-trainer writes under
`<output-dir>/dataset/` after sampling and resampling to 16 kHz WAV.

| Dataset | Role | Default cap | Upstream | Local output |
|---|---|---|---|---|
| [`TigreGotico/ESC-50`](https://huggingface.co/datasets/TigreGotico/ESC-50) | General negatives (env. sounds) | none (~2 000 clips) | **883 MB** | ~200 MB |
| [`TigreGotico/NAR`](https://huggingface.co/datasets/TigreGotico/NAR) | General negatives | none | **45 MB** | ~30 MB |
| [`agkphysics/AudioSet`](https://huggingface.co/datasets/agkphysics/AudioSet) | General negatives | **5 000 samples** | **2.4 TB total** — streamed, only the 5 000 capped samples are kept | ~500 MB |
| [`TigreGotico/not-wake-words-speech-en`](https://huggingface.co/datasets/TigreGotico/not-wake-words-speech-en) | Speech negatives (primary) | none | **320 MB** | ~250 MB |
| [`hf-internal-testing/librispeech_asr_demo`](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_demo) | Speech negatives | ~70 clips | **9 MB** | ~7 MB |
| [`Anton-Bushuiev/speech-commands-v2-resampled`](https://huggingface.co/datasets/Anton-Bushuiev/speech-commands-v2-resampled) | Short-command negatives | none | gated (HF login required) | varies |
| [`TigreGotico/ambient_noises`](https://huggingface.co/datasets/TigreGotico/ambient_noises) | `bg_noise/` augmentation | none | **297 MB** | ~250 MB |
| [`TigreGotico/building_106_kitchen_3secs`](https://huggingface.co/datasets/TigreGotico/building_106_kitchen_3secs) | `bg_noise/` augmentation | none | **432 MB** | ~400 MB |
| [`TigreGotico/public_domain_sounds_3secs`](https://huggingface.co/datasets/TigreGotico/public_domain_sounds_3secs) | `bg_noise/` augmentation | none | **902 MB** | ~850 MB |
| [`TigreGotico/FMA_3secs`](https://huggingface.co/datasets/TigreGotico/FMA_3secs) | `music/` augmentation | none | **5 MB** | ~5 MB |
| [`davidscripka/MIT_environmental_impulse_responses`](https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses) | `rir/` augmentation | none | **8 MB** | ~8 MB |

**Sum of "Upstream", default preset (excluding AudioSet stream and the
gated speech-commands repo): ~2.9 GB.** Add the locally written outputs
(~2 GB) plus HF cache overhead (Parquet/Arrow intermediates roughly
double the raw audio bytes) to land around 6–8 GB for the default run.

Pass `--no-augmentation-data` to skip the bottom five rows — saves
~1.6 GB upstream + ~1.5 GB local.

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

| Backend | Extra | HF repo (queried 2026-05-14) | First-time download | RAM during VC |
|---|---|---|---|---|
| `chatterbox-onnx` (CPU, recommended) | `[vc-onnx]` | [`onnx-community/chatterbox-onnx`](https://huggingface.co/onnx-community/chatterbox-onnx) | **~5.0 GB** | ~3 GB |
| `chatterbox-tts` (GPU) | `[vc-torch]` | [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox) | **~11.7 GB** | ~6 GB + VRAM |

These are big — Chatterbox is a full TTS+VC stack, not a small VC head.
If you don't need voice cloning, skip the extra entirely; the quickstart
runs fine without it.

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
