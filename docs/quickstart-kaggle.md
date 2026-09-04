# Kaggle / Colab Quickstart

> **For people who prefer reading docs before running code.**  
> This page mirrors [`notebooks/kaggle_quickstart.ipynb`](../notebooks/kaggle_quickstart.ipynb)
> step by step.  Open the notebook link when you are ready to run.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TigreGotico/ww-trainer/blob/dev/notebooks/kaggle_quickstart.ipynb)

---

## What you will end up with

Two ONNX files you drop onto an OpenVoiceOS (or Rhasspy) device:

```
ww_output/model/
├── best_f1_featurizer.onnx    ← audio feature extractor
└── best_f1.onnx               ← classifier head
```

Together these are typically under 1 MB.  Runtime inference needs only
`onnxruntime` + `numpy` — no PyTorch on the device.

---

## Prerequisites

### Kaggle (recommended — free GPU)

1. Create a free [Kaggle account](https://www.kaggle.com).
2. Open the notebook:
   **Code → + New Notebook → Import from GitHub URL**  
   `https://github.com/TigreGotico/ww-trainer/blob/dev/notebooks/kaggle_quickstart.ipynb`
3. Set **Session options → Accelerator → GPU T4 × 1** (free, no credit card).
4. Edit Cell 2: set `WAKE_WORD` to your phrase.
5. Click **Run All**.

Full run time: **≈ 25–40 min** on GPU T4.

### Google Colab

Click the badge at the top of this page.

- Set runtime to **T4 GPU** (Runtime → Change runtime type).
- Cell 4 detects Colab and mounts Google Drive automatically so outputs survive
  session resets.

### Local CPU

```bash
pip install "wakeforge[datagen,torchcodec]"
jupyter notebook notebooks/kaggle_quickstart.ipynb
```

Use `TIER=micro`, `N_POSITIVE=100`, `DOWNLOAD_AUGMENT=false` — expect 30–60 min.

---

## Step-by-step walkthrough

### Cell 2 — Configure

The only cell you need to edit.  Set `WAKE_WORD` to your phrase, e.g. `"hey computer"`.
All other variables have safe defaults.

| Variable | Default | Notes |
|----------|---------|-------|
| `WAKE_WORD` | `hey jarvis` | Any phrase, any language |
| `TIER` | `small` | See tier table below |
| `EPOCHS` | `50` | More = better accuracy, more time |
| `BATCH_SIZE` | `16` | Reduce to `8` for OOM errors |
| `N_POSITIVE` | `500` | TTS utterances synthesised |
| `LANG` | `en` | BCP-47 language code |
| `ADVERSARIAL` | `true` | Hard-negative confusable phrases |
| `DOWNLOAD_AUGMENT` | `true` | Download background noise / reverb |
| `REUSE_DATASET` | `true` | Skip datagen if dataset exists |

### Cell 3 — Install

Installs `wakeforge[datagen,torchcodec]`.  Takes 3–5 min on Kaggle.

### Cell 4 — Platform setup

- **Colab:** mounts Google Drive; redirects `OUTPUT_DIR` so files persist.
- **Kaggle:** injects `MLFLOW_TOKEN` secret (silently skipped if absent).
- Any platform: sets `MLFLOW_TRACKING_URI` if `MLFLOW_URI` was provided.

### Cell 5 — Generate dataset

Synthesises positive utterances via edge-tts, downloads negative speech from
HuggingFace public datasets, and optionally downloads background noise / music /
room impulse responses for augmentation.

Expected time: **15–20 min** on Kaggle.  
`REUSE_DATASET=true` (default): completes instantly on subsequent runs.

Troubleshooting:
- *`datasets` not found* → re-run Cell 3.
- *Not enough disk space* → set `DOWNLOAD_AUGMENT=false`.
- *VAD fails* → `!pip install -q "git+https://github.com/TigreGotico/vadonnx.git"` then re-run.

### Cell 6 — Train

Calls `train_from_wakeword()`.  The augmentation assets from Cell 5 are picked up
automatically.  Best checkpoint and both ONNX files are saved to
`OUTPUT_DIR/model/`.

Expected time: **5–15 min** on Kaggle GPU T4 (50 epochs, `small` tier).

Troubleshooting:
- *CUDA OOM* → set `BATCH_SIZE=8` in Cell 2, re-run from Cell 6.
- *F1 < 0.5* → raise `N_POSITIVE` to 1000, re-run from Cell 5.
- *Session timeout* → checkpoint is saved every epoch; re-run Cells 2–5 (instant),
  then Cell 6 — training resumes from `best_f1.pt`.

### Cell 7 — Verify ONNX

Checks that `best_f1_featurizer.onnx` and `best_f1.onnx` exist and are non-empty.
Both are required for inference.

### Cell 8 — Inference sanity check

Scores a positive sample from the test set.  Score ≥ 0.8 = confident.
Score < 0.5 = needs more data or training.

### Cell 9 — Ship it

Prints the OVOS `mycroft.conf` snippet and download instructions.

---

## Tier reference

| Tier | Architecture | Params | Target device |
|------|-------------|--------|---------------|
| `micro` | MFCC-40 + FFN | ~50 K | MCU, RPi Zero |
| **`small`** | MFCC-40 + GRU | ~200 K | **RPi 3/4, most SBCs** |
| `sincnet_small` | SincNet + GRU | ~300 K | SBC with more headroom |
| `filterbank_small` | FilterBank + GRU | ~250 K | Embedded SBC |
| `gammatone_small` | Gammatone + GRU | ~250 K | Embedded SBC |

---

## Deploying to OpenVoiceOS

Install the wakeword plugin on your OVOS device:

```bash
pip install ovos-ww-plugin-precise-onnx
```

Copy the two ONNX files to the device:

```bash
scp best_f1_featurizer.onnx best_f1.onnx \
    ovos@mydevice:~/.local/share/mycroft/precise/
```

Add to `~/.config/mycroft/mycroft.conf` (replace `hey_jarvis` and paths):

```json
{
  "listener": {
    "wake_word": "hey_jarvis"
  },
  "hotwords": {
    "hey_jarvis": {
      "module": "ovos-ww-plugin-precise-onnx",
      "model": "/home/ovos/.local/share/mycroft/precise/best_f1.onnx",
      "sensitivity": 0.5,
      "trigger_level": 3,
      "listen": true
    }
  }
}
```

Restart the listener and verify:

```bash
systemctl --user restart ovos-listener
journalctl --user -u ovos-listener -f   # look for "WW activated"
```

---

## Further reading

- [`docs/getting_started/quickstart.md`](getting_started/quickstart.md) — local CLI quickstart
- [`docs/guides/inference.md`](guides/inference.md) — `OnnxWakeWordInferencer` API
- [`docs/guides/embedded.md`](guides/embedded.md) — C export for ESP32 / MCU
- [`notebooks/kaggle_experiments.ipynb`](../notebooks/kaggle_experiments.ipynb) — compare
  architectures and loss functions systematically
- [`docs/faq.md`](faq.md) — common questions across 15 topics
