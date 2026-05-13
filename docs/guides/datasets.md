# Notebook → ww-trainer Data Contract

This document describes the exact interface between dataset-generation notebooks and the
`ww_trainer` training pipeline.  Every claim about runtime behaviour cites the source file
and line number it was read from.

---

## 1. What `AudioDataset` Expects

### 1.1 The `samples` argument

`AudioDataset.__init__` (`ww_trainer/dataset.py:104`) takes a positional argument `samples`
which must be an **iterable of 2-tuples**:

```
(path: str, label: str)
```

- `path` — absolute path to an audio file.  All paths are validated at construction time;
  missing files are logged with `logger.warning` (`dataset.py:155-162`).
- `label` — a **string**, either `"1"` (wake word / positive) or `"0"` (non-wake / negative).
  The string is cast to `int` at item access time (`dataset.py:298`: `return wav, int(label), path`).
  Voice conversion is gated on `label == "1"` (`dataset.py:273`), so the string comparison is
  literal — any value other than `"1"` is treated as negative.

The trainer CLI reads the metadata CSV into this format directly (`trainer.py:830-832`):

```python
entries = [tuple(line.strip().split(",", 1)) for line in f if line.strip()]
```

The split uses `maxsplit=1` so paths that contain commas are handled correctly provided the
label is the last field.

### 1.2 Metadata CSV format

A plain-text CSV with **no header row**, two columns, no quoting:

```
/absolute/path/to/audio.wav,1
/absolute/path/to/audio.wav,0
```

Rules:
- One record per line.
- Column 1: absolute path to the audio file.
- Column 2: label string `"1"` or `"0"`.
- No header line.
- Empty lines are skipped (`trainer.py:831`: `if line.strip()`).

Concrete example:

```
/data/ww/hey_mycroft/hey_mycroft_00001.wav,1
/data/ww/hey_mycroft/hey_mycroft_00002.wav,1
/data/ww/not_wake_word/notww_00001.wav,0
/data/ww/not_wake_word/notww_00002.wav,0
```

### 1.3 Full `__init__` signature

```python
AudioDataset(
    samples,                        # list of (path, label) tuples — required
    sample_rate: int = 16000,       # target sample rate; auto-resamples if source differs
    aug_prob: float = 0.0,          # probability of applying augmentation per sample (0 = off)
    vc_prob: float = 0.3,           # probability of voice conversion on positive samples
    bg_noise_folder: str = None,    # general background noise files
    music_folder: str = None,       # music files for mixing
    bg_speech_folder: str = None,   # competing speaker / babble files
    mic_noise_folder: str = None,   # microphone hiss / silence noise
    rir_folder: str = None,         # room impulse responses for reverb
    vc_folder: str = None,          # reference voice WAVs for Chatterbox voice conversion
    snr_min: float = 0.0,           # minimum SNR (dB) for noise mixing
    snr_max: float = 20.0,          # maximum SNR (dB) for noise mixing
    pitch_min: float = -1.0,        # minimum pitch shift (semitones)
    pitch_max: float = 1.0,         # maximum pitch shift (semitones)
    speed_min: float = 0.95,        # minimum speed perturbation factor
    speed_max: float = 1.05,        # maximum speed perturbation factor
    device="auto"                   # device for Chatterbox VC model ("auto" → cuda/cpu)
)
```

Source: `ww_trainer/dataset.py:104-121`.

### 1.4 Audio format support

`AudioDataset.__getitem__` loads files with `torchaudio.load(path)` (`dataset.py:281`).
torchaudio supports: **WAV, FLAC, MP3, M4A (AAC), OGG/Vorbis**.

For augmentation folders `_collect_audio_files` (`dataset.py:78-92`) accepts:
`.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`.

### 1.5 Sample rate

Default target is **16 000 Hz**.  If a loaded file has a different sample rate, it is
resampled automatically with `torchaudio.functional.resample` and a `logger.warning` is
emitted (`dataset.py:284-289`).  Pre-resampling to 16 kHz before training saves CPU time
at the cost of disk space.

### 1.6 Minimum / recommended dataset size

There is no hard minimum enforced by code. Practical guidance:

| Scenario | Positive samples | Negative samples |
|---|---|---|
| Proof of concept | 100 | 300 |
| Usable model | 500 | 1 500 |
| Production quality | 1 000+ | 3 000+ |

Keep the negative-to-positive ratio between **2:1 and 5:1**.  The `AudioDataset` constructor
logs the label distribution at `INFO` level (`dataset.py:163-167`) so you can verify balance
before training starts.

---

## 2. The Notebook Pipeline

Two notebooks live in
`notebooks/ww/` (relative to the Machine Learning Workspace root).

### 2.1 `tts2ww.ipynb` — 8-stage factory

A fully-featured pipeline producing large synthetic datasets.

| Stage | What it does | Key output |
|---|---|---|
| 0 — Config | Sets all paths via `os.environ.setdefault` | Environment variables |
| 1 — Adversarial generation | GraphemeAug + LLM-based phonetically confusable hard-negatives | `$ADV_OUTPUT_DIR/<ww>.txt` |
| 2 — Normalize | Lowercase / sort / deduplicate adversarial word lists | Updated `.txt` files |
| 3 — TTS Synthesis + VC | Edge TTS / Google TTS / Piper → optional Chatterbox VC | `$SYNTH_OUTPUT_DIR/<ww>/*.wav` |
| 4 — Bulk VC augmentation | Revoice Stage 3 output with `chatterbox_bulk_vc` CLI | `$VC_AUG_OUTPUT_DIR/<ww>/*.wav` |
| 5 — VC TTS | Direct Chatterbox TTS with varied exaggeration via `chatterbox_bulk_tts` | `$VC_TTS_OUTPUT_DIR/<ww>/*.wav` |
| 6 — Training augmentation | Reads `metadata.csv`, applies noise/reverb/pitch/speed, writes new CSV | `$AUG_OUTPUT_DIR/metadata.csv` + audio |
| 7 — Benchmark generation | Deterministic test sets at fixed SNRs | `$BENCH_OUTPUT_DIR/` sub-folders |

**Stage 6 is the critical integration point.**  It reads a `metadata.csv` from
`AUG_INPUT_DIR` and writes a new `metadata.csv` to `AUG_OUTPUT_DIR`.  Both CSVs use the
two-column format described in §1.2.

Stage 6 output folder layout:

```
$AUG_OUTPUT_DIR/
    metadata.csv          ← path,label rows (absolute paths)
    wakes_aug/            ← augmented positive samples
        <stem>_aug.wav
    negatives_aug/        ← augmented negative samples
        <stem>_aug.wav
```

The `path` column in the output CSV contains absolute paths to files under
`$AUG_OUTPUT_DIR`.

**Prerequisite `metadata.csv` for Stage 6** — you must create this file before running
Stage 6.  It should list the clean (un-augmented) audio files you want augmented.  Format
is identical: absolute path + comma + label.

### 2.2 `ww_dataset_generator_ovos_vc.ipynb` — integrated single-notebook pipeline

A simpler alternative that runs TTS + VC + augmentation in one go using OVOS TTS plugins
(Edge, Google, Phoonnx) and Chatterbox ONNX.

**Output directory layout:**

```
$WW_BASE_DIR/dataset/$WW_LANG/
    wake_word/
        <word>/
            <word>_00001.wav   ← label "1"
            <word>_00002.wav
            …
    not_wake_word/
        notww_00001.wav        ← label "0"
        notww_00002.wav
        …
```

This notebook does **not** produce a `metadata.csv` automatically.  You must generate it
yourself after the notebook finishes (see §3 below).

**To run either notebook end-to-end:**

```bash
# 1. Install dependencies
pip install ovos-tts-plugin-edge-tts ovos-tts-plugin-google-tx \
            chatterbox_onnx librosa soundfile tqdm

# 2. Set key env vars before opening Jupyter
export WW_BASE_DIR=/data/ww
export WW_WORDS="hey_mycroft"
export WW_LANG=en

# 3. Open and run all cells in order
jupyter notebook notebooks/ww/tts2ww.ipynb
```

---

## 3. Manual Dataset Preparation

If you have your own audio recordings or want to skip the notebooks entirely:

### 3.1 Folder structure (recommended)

```
/data/hey_mycroft/
    wake/
        sample_001.wav
        sample_002.wav
        …
    non_wake/
        background_001.wav
        background_002.wav
        …
```

### 3.2 Generating `metadata.csv`

```bash
# Positive samples
find /data/hey_mycroft/wake -name "*.wav" | \
    awk '{print $0",1"}' >> /data/metadata.csv

# Negative samples
find /data/hey_mycroft/non_wake -name "*.wav" | \
    awk '{print $0",0"}' >> /data/metadata.csv
```

Or in Python:

```python
from pathlib import Path

wake_dir    = Path("/data/hey_mycroft/wake")
nonwake_dir = Path("/data/hey_mycroft/non_wake")

rows = []
for p in sorted(wake_dir.rglob("*.wav")):
    rows.append(f"{p.absolute()},1")
for p in sorted(nonwake_dir.rglob("*.wav")):
    rows.append(f"{p.absolute()},0")

Path("/data/metadata.csv").write_text("\n".join(rows) + "\n")
```

### 3.3 Using the `--metadata` and `--test-metadata` CLI flags

```bash
# Single CSV — trainer auto-splits 80/20
ww_trainer-train \
    --metadata /data/metadata.csv \
    --wakeword "hey mycroft" \
    --model-type gru

# Explicit train / test split
ww_trainer-train \
    --metadata /data/train.csv \
    --test-metadata /data/test.csv \
    --wakeword "hey mycroft" \
    --model-type gru
```

`--metadata` is required; `--test-metadata` is optional.  When `--test-metadata` is
omitted the trainer uses `--split` (default 0.8) to divide `--metadata` into train and
validation sets (`trainer.py:838-841`).

---

## 4. Augmentation Data

All augmentation folders are optional.  When a folder is omitted or empty the corresponding
augmentation type is silently skipped (`dataset.py:136-140`).

| CLI flag | `AudioDataset` parameter | Content | Application probability |
|---|---|---|---|
| `--bg-noise-folder` | `bg_noise_folder` | General ambient noise (wind, traffic, animals) | 60 % per sample |
| `--music-folder` | `music_folder` | Music tracks | 30 % per sample; SNR 0–10 dB (louder) |
| `--bg-speech-folder` | `bg_speech_folder` | Competing speaker / babble | 50 % per sample; SNR 10–25 dB (quieter) |
| `--mic-noise-folder` | `mic_noise_folder` | Microphone hiss / silence noise | 80 % per sample |
| `--rir-folder` | `rir_folder` | Room impulse responses | 30 % per sample |
| `--vc-folder` | `vc_folder` | Reference voice WAVs for Chatterbox VC | `vc_prob` (default 30 %) on positives only |

Source: `dataset.py:186-235` (augmentation probabilities hardcoded in `get_augmented`).

### 4.1 Expected format

All augmentation folders are scanned recursively for files with extensions
`.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg` (`dataset.py:80-92`).  Any sample rate is
accepted — files are resampled to `sample_rate` on load.

**RIR files** must be impulse responses (short mono recordings, typically < 2 s).

**VC reference voices** must be clean speech recordings of individual speakers.  Longer
clips (5–30 s) give better voice conversion quality.

### 4.2 Where to download suitable data

| Type | Source | Notes |
|---|---|---|
| Background noise | [MUSAN](https://www.openslr.org/17/) | noise/ subdirectory |
| Background noise | [ESC-50](https://github.com/karolpiczak/ESC-50) | 2 000 environmental clips |
| Background speech | [MUSAN](https://www.openslr.org/17/) | speech/ subdirectory |
| Music | [MUSAN](https://www.openslr.org/17/) | music/ subdirectory |
| Music | [FMA](https://github.com/mdeff/fma) | Free Music Archive |
| RIRs | [MIT Acoustical Reverberation Scene Statistics Survey](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html) | 271 real-room IRs |
| RIRs | [OpenSLR RIR + Noise](https://www.openslr.org/28/) | Simulated + real |
| VC reference voices | [LibriSpeech](https://www.openslr.org/12) | `dev-clean` or `test-clean` splits |
| VC reference voices | [MLCommons Multilingual Spoken Words](https://mlcommons.org/datasets/multilingual-spoken-words/) | Near-infinite speaker variety |
| HuggingFace datasets (used by `ww_dataset_generator_ovos_vc.ipynb`) | `TigreGotico/public_domain_sounds_3secs`, `TigreGotico/ESC-50` | Pre-cut 3 s clips |

---

## 5. Label Convention

`AudioDataset` treats the label as a string throughout most of its code.  The exact check
that determines whether voice conversion is applied (`dataset.py:273`):

```python
if label == "1" and self.vc is not None and random.random() < self.vc_prob:
```

The label is converted to `int` only at the point of returning a batch item
(`dataset.py:298`):

```python
return wav, int(label), path
```

Summary:

| String value | Interpretation | `int` value returned |
|---|---|---|
| `"1"` | Wake word / positive | `1` |
| `"0"` | Non-wake / negative | `0` |
| Any other string | Negative (no VC applied) | Runtime `ValueError` from `int()` |

**Use only `"1"` and `"0"`.** Any other string will pass through the dataset's internal
string comparisons as a negative but will raise `ValueError` when the batch is assembled.

---

## 6. End-to-End Example

This example shows the complete path from notebook output to a trained model.

### Step 1 — Run `ww_dataset_generator_ovos_vc.ipynb`

```bash
export WW_BASE_DIR=/data/ww
export WW_WORDS="hey_mycroft"
export WW_LANG=en
export WW_TARGET_SAMPLES=1000
export USE_PHOONNX_TTS=1
export ENABLE_VOICE_CLONE=1

jupyter nbconvert --to notebook --execute \
    notebooks/ww/ww_dataset_generator_ovos_vc.ipynb
```

After execution the following directories exist:

```
/data/ww/dataset/en/
    wake_word/hey_mycroft/*.wav     (~1 000 files, label "1")
    not_wake_word/*.wav             (~631 files, label "0")
```

### Step 2 — Build `metadata.csv`

```python
from pathlib import Path

base = Path("/data/ww/dataset/en")
rows = []

for p in sorted((base / "wake_word").rglob("*.wav")):
    rows.append(f"{p.absolute()},1")

for p in sorted((base / "not_wake_word").glob("*.wav")):
    rows.append(f"{p.absolute()},0")

Path("/data/ww/metadata.csv").write_text("\n".join(rows) + "\n")
print(f"Wrote {len(rows)} rows")
```

### Step 3 — Train

```bash
ww_trainer-train \
    --metadata /data/ww/metadata.csv \
    --wakeword "hey mycroft" \
    --model-type gru \
    --tier small \
    --epochs 30 \
    --batch-size 64 \
    --aug-prob 0.8 \
    --bg-noise-folder /data/musan/noise \
    --bg-speech-folder /data/musan/speech \
    --music-folder /data/musan/music \
    --rir-folder /data/rirs \
    --output-dir /data/ww/model
```

### Step 4 — Export to ONNX and run inference

```python
from ww_trainer.inference import OnnxWakeWordInferencer
import numpy as np

inferencer = OnnxWakeWordInferencer(
    extractor_path="/data/ww/model/mfcc.onnx",
    head_path="/data/ww/model/classifier_head.onnx",
    sample_rate=16000,
)

# One-shot clip
audio = np.zeros(16000, dtype=np.float32)   # replace with real audio
prob = inferencer.infer(audio)
print(f"Wake word probability: {prob:.3f}")

# Streaming
cache = None
chunk_size = 1600   # 100 ms at 16 kHz
for i in range(0, len(audio), chunk_size):
    chunk = audio[i:i + chunk_size]
    prob, cache = inferencer.infer_streaming(chunk, cache)
    if prob > 0.5:
        print("Wake word detected!")
```
