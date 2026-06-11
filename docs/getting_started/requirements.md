# Resource Requirements — Disk, Bandwidth, RAM, Time

What `ww_trainer-quickstart` will actually cost you before you run it.

## TL;DR — pick a preset

| Preset | Flags | Disk (HF cache + outputs) | Network, first run | Peak RAM | Time, CPU | Time, GPU |
|---|---|---|---|---|---|---|
| **Smoke** | `--no-augmentation-data --n-positive 200 --epochs 10 --tier micro` | ~1.5 GB | ~1.3 GB | ~1.5 GB | ~10 min | ~3 min |
| **Known phrase** | `--wake-word "hey jarvis"` (any phrase in [§ Pre-built positives](#pre-built-positives--skip-the-synth-step)) | ~3 GB | ~2.5 GB | ~2 GB | ~10 min | ~3 min |
| **Default** | (no flags, novel phrase) | ~6–8 GB | ~5 GB | ~3 GB | ~30–60 min | ~10–15 min |
| **Default + VC** | `--vc-refs <dir>` | + ~5 GB | + ~5 GB (Chatterbox ONNX) | ~5 GB | + ~15 min | + ~5 min |
| **Production** | `--n-positive 5000 --epochs 100 --tier large` | ~15–20 GB | ~7 GB | ~8 GB | not recommended | ~1–2 h |
| **Kitchen sink** | every dataset + both VC backends + all SSL featurizers — see [§ All-in](#all-in-budget) | ~30–35 GB | ~25 GB | ~8 GB | — | — |

HF caches the raw dataset under `~/.cache/huggingface/` and ww-trainer
re-writes each used sample as a 16 kHz mono WAV into
`<output-dir>/dataset/`. Plan for both. CPU times assume an 8-core x86-64
laptop; GPU times a mid-range consumer card (RTX 3060 or similar).

## Datasets downloaded

Sources are declared in `NEGATIVE_DATASETS` —
[`ww_trainer/datagen.py:47-73`](../../ww_trainer/datagen.py).
"Upstream" is the full repo size on HF (what ends up in the HF cache if
loaded non-streaming). "Local output" is what ww-trainer writes under
`<output-dir>/dataset/` after sampling and resampling to 16 kHz WAV.

| Dataset | Role | Default cap | Upstream | Local output |
|---|---|---|---|---|
| [`TigreGotico/ESC-50`](https://huggingface.co/datasets/TigreGotico/ESC-50) | General negatives (env. sounds) | none (~2 000 clips) | 883 MB | ~200 MB |
| [`TigreGotico/NAR`](https://huggingface.co/datasets/TigreGotico/NAR) | General negatives | none | 45 MB | ~30 MB |
| [`agkphysics/AudioSet`](https://huggingface.co/datasets/agkphysics/AudioSet) | General negatives | **5 000 samples** | 2.4 TB total — streamed, only the 5 000 capped samples are kept | ~500 MB |
| [`TigreGotico/not-wake-words-speech-en`](https://huggingface.co/datasets/TigreGotico/not-wake-words-speech-en) | Speech negatives (primary), also a voice-donor pool for VC | none | 320 MB | ~250 MB |
| [`hf-internal-testing/librispeech_asr_demo`](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_demo) | Speech negatives | ~70 clips | 9 MB | ~7 MB |
| [`TigreGotico/ambient_noises`](https://huggingface.co/datasets/TigreGotico/ambient_noises) | `bg_noise/` augmentation | none | 297 MB | ~250 MB |
| [`TigreGotico/building_106_kitchen_3secs`](https://huggingface.co/datasets/TigreGotico/building_106_kitchen_3secs) | `bg_noise/` augmentation | none | 432 MB | ~400 MB |
| [`TigreGotico/public_domain_sounds_3secs`](https://huggingface.co/datasets/TigreGotico/public_domain_sounds_3secs) | `bg_noise/` augmentation | none | 902 MB | ~850 MB |
| [`TigreGotico/FMA_3secs`](https://huggingface.co/datasets/TigreGotico/FMA_3secs) | `music/` augmentation | none | 5 MB | ~5 MB |
| [`davidscripka/MIT_environmental_impulse_responses`](https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses) | `rir/` augmentation | none | 8 MB | ~8 MB |

Sum of upstream sizes, default preset (excluding AudioSet which is
streamed): ~2.9 GB. Add locally written outputs (~2 GB) plus HF cache
Parquet/Arrow intermediates (~30–50 % overhead) → ~6–8 GB for the
default run.

Pass `--no-augmentation-data` to skip the bottom five rows — saves
~1.6 GB upstream + ~1.5 GB local.

### If you already cached the full AudioSet by mistake

Versions prior to this fix would attempt a non-streaming load of
AudioSet first and only fall back to streaming on failure — which meant
HF began downloading the full 2.4 TB upstream into the cache before the
5 000-sample iteration cap ever kicked in. To recover:

```bash
du -sh ~/.cache/huggingface/datasets/datasets--agkphysics--AudioSet
rm -rf ~/.cache/huggingface/datasets/datasets--agkphysics--AudioSet
```

This release forces streaming for AudioSet
([`STREAMING_ONLY_DATASETS` — `ww_trainer/datagen.py`](../../ww_trainer/datagen.py)),
so the cache will not balloon again.

## Pre-built positives — skip the synth step

When your wake word is one of the phrases in the
[**Synthetic WakeWord Datasets** collection](https://huggingface.co/collections/TigreGotico/synthetic-wakeword-datasets-68ee52b6976ed8a20c8cf98f),
`run_datagen_pipeline` downloads them instead of running TTS. The
mapping lives in
[`KNOWN_POSITIVE_DATASETS` — `ww_trainer/datagen.py:37`](../../ww_trainer/datagen.py).
Just type the phrase — no `--vc-refs`, no TTS plugins, no language tag.

| Wake word phrase | HF dataset | Size |
|---|---|---|
| `hey jarvis` | [`TigreGotico/synthetic-wakeword-hey_jarvis`](https://huggingface.co/datasets/TigreGotico/synthetic-wakeword-hey_jarvis) | ~85 MB |
| `ok google` | [`TigreGotico/synthetic-wakeword-ok_google`](https://huggingface.co/datasets/TigreGotico/synthetic-wakeword-ok_google) | ~85 MB |
| `alexa` | [`TigreGotico/synthetic-wakeword-alexa`](https://huggingface.co/datasets/TigreGotico/synthetic-wakeword-alexa) | 73 MB |
| `hey mycroft` | [`TigreGotico/synthetic-wakeword-hey_mycroft`](https://huggingface.co/datasets/TigreGotico/synthetic-wakeword-hey_mycroft) | 90 MB |
| `hey siri` | [`TigreGotico/synthetic-wakeword-hey_siri`](https://huggingface.co/datasets/TigreGotico/synthetic-wakeword-hey_siri) | 81 MB |
| `wake up` | [`TigreGotico/synthetic-wakeword-wake_up`](https://huggingface.co/datasets/TigreGotico/synthetic-wakeword-wake_up) | 73 MB |
| `hey computer` | [`TigreGotico/synthetic-wakeword-hey_computer`](https://huggingface.co/datasets/TigreGotico/synthetic-wakeword-hey_computer) | 87 MB |
| `voice assistant` | [`TigreGotico/synthetic-wakeword-voice_assistant`](https://huggingface.co/datasets/TigreGotico/synthetic-wakeword-voice_assistant) | 95 MB |
| `home assistant` | [`TigreGotico/synthetic-wakeword-home_assistant`](https://huggingface.co/datasets/TigreGotico/synthetic-wakeword-home_assistant) | 88 MB |

Older bundled megapack (raw TTS, no VC pass, lower quality — kept for
completeness and cross-keyword benchmarking, not recommended for
training a new detector):
[`OpenVoiceOS/synthetic-wakewords`](https://huggingface.co/datasets/OpenVoiceOS/synthetic-wakewords)
— 2.09 GB.

## Synthesised positives — TTS path

When the phrase is novel, datagen falls back to TTS. Per-sample size:

```
16 000 Hz × 1.5 s × 2 B/sample ≈ 48 KB
```

For the default `n_positive=1000`
([`QuickstartConfig` — `ww_trainer/quickstart.py:50`](../../ww_trainer/quickstart.py)):

| n_positive | Pre-VAD | Post-VAD trim |
|---|---|---|
| 200 (smoke) | ~10 MB | ~7 MB |
| 1 000 (default) | ~48 MB | ~35 MB |
| 5 000 (production) | ~240 MB | ~175 MB |

VAD trim is [`trim_silence_vad` — `ww_trainer/datagen.py:111`](../../ww_trainer/datagen.py),
typically 10–30 % smaller after silence removal.

## Voice conversion — essential for no-real-data runs

> TTS plugins alone give 1–3 voices and a robotic prosody distribution;
> models trained on that overfit hard. VC re-renders every synthesised
> positive in the timbre of a random reference speaker, turning 1 000
> TTS clips into 1 000 *uniquely voiced* training samples. For any
> from-zero training run on a novel phrase, **plan to enable VC**.

VC is opt-in via `--vc-refs <dir>`. There is no implicit Chatterbox
download otherwise.

| Backend | Extra | HF repo | First-time download | RAM during VC |
|---|---|---|---|---|
| `chatterbox-onnx` (CPU, recommended) | `[vc-onnx]` | [`onnx-community/chatterbox-onnx`](https://huggingface.co/onnx-community/chatterbox-onnx) | ~5.0 GB | ~3 GB |
| `chatterbox-tts` (GPU) | `[vc-torch]` | [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox) | ~11.7 GB | ~6 GB + VRAM |

### Where to get reference voices (zero extra disk)

[`TigreGotico/not-wake-words-speech-en`](https://huggingface.co/datasets/TigreGotico/not-wake-words-speech-en)
is already downloaded as a negatives source and contains short clean
speech from many speakers — point `--vc-refs` straight at it:

```bash
ww_trainer-quickstart \
  --wake-word "hey acme" \
  --output-dir ./hey_acme \
  --vc-refs ./hey_acme/dataset/negatives/not-wake-words-speech-en
```

Other donors: any directory of short (3–10 s) clean speech WAVs at any
sample rate (Chatterbox resamples internally). LibriSpeech, phone
memos, or a single friend's voice all work.

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
| `ssl_small` | ~200 K | ~800 KB | ~300 MB (TinyHuBERT / DistilHuBERT) |
| `large` | ~2 M | ~8 MB | ~1–2 GB (HuBERT base / Wav2Vec2-BERT) |

Run `ww_trainer-tiers` to print the live table.

Pre-exported feature-extractor ONNX files for SSL tiers live in the
[`onnx-feature-extractors` collection](https://huggingface.co/collections/TigreGotico/onnx-feature-extractors-690a3d2bced386cc3c338c77):

| Extractor | HF repo | Size |
|---|---|---|
| DistilHuBERT | [`TigreGotico/distillhubert-onnx`](https://huggingface.co/TigreGotico/distillhubert-onnx) | 144 MB |
| MFCC / Δ-MFCC / filterbank / gammatone / CQT / PNCC / PLP | (one repo each in the collection) | < 5 MB each |

## All-in budget

Pulling **every** HF asset ww-trainer can use, with both Chatterbox
backends installed and one cached SSL featurizer per tier:

| Category | Subtotal |
|---|---|
| Negative + augmentation HF datasets (table above, AudioSet capped) | ~3.4 GB |
| All pre-built positive wake-word datasets + OpenVoiceOS megapack | ~2.9 GB |
| Pre-exported ONNX feature extractors (full collection) | ~155 MB |
| Chatterbox ONNX (`vc-onnx`) | ~5.0 GB |
| Chatterbox PyTorch (`vc-torch`) | ~11.7 GB |
| Auto-downloaded HuggingFace SSL weights for SSL tier training (HuBERT-base + Wav2Vec2-base + Wav2Vec2-BERT) | ~3 GB |
| Local outputs (one training run, default tier) | ~2 GB |
| HF cache Parquet/Arrow overhead | ~2 GB |
| **Total** | **~30–35 GB** |

Per-tier SSL featurizer weights are downloaded lazily by `transformers`
on first use of a tier — only the tiers you actually train pay this
cost. Running many experiments multiplies the local-outputs row.

## Where caches live

| Cache | Default path | Override |
|---|---|---|
| HuggingFace datasets + models | `~/.cache/huggingface/` | `HF_HOME` |
| PyTorch hub | `~/.cache/torch/hub/` | `TORCH_HOME` |
| OVOS plugin TTS audio | `~/.cache/mycroft/` | `XDG_CACHE_HOME` |

Caches are shared across all `ww_trainer-quickstart` runs — the first
run pays the full download cost.

## How to slim down a run

In order of impact:

1. Use a phrase from the pre-built collection above — drops TTS and
   reduces the positive footprint to a known number.
2. `--no-augmentation-data` — drops bg_noise/music/RIR.
3. `--reuse-dataset` — skip datagen on subsequent runs.
4. `--n-positive 200` — fewer TTS calls and fewer mined negatives.
5. `--tier esp32_nano` / `micro` — smallest models, no SSL featurizer.
6. Skip `--vc-refs` — no Chatterbox download.
