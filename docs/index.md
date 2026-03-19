# ww-trainer — Documentation Index

Full training and research suite for wake word detection models — from microcontrollers to servers.
Every feature extractor exports to ONNX; production inference requires only `onnxruntime` and `numpy`.

Version: `0.0.1a1` — see `ww_trainer/version.py`

---

## Architecture Overview

```
Raw Audio [B, T]
      |
  BaseExtractor                ww_trainer/feats.py:57
  ├── MfccExtractor            classical, ONNX-exportable, no transformers
  ├── OnnxFeatureExtractor     loads pre-exported extractor ONNX at runtime
  ├── HubertExtractor          neural (train-time), requires transformers
  └── Wav2Vec2Extractor        neural (train-time), requires transformers
      |  [B, T_frames, F]
  ClassifierHead               ww_trainer/model.py:13
  ├── FfnClassifierHead        mean-pool → two-layer MLP
  ├── CnnClassifierHead        Conv1d × 2 → AdaptiveAvgPool → FC
  └── GruClassifierHead        GRU → mean-pool → FC
      |  [B]  logits
  sigmoid → probability in [0, 1]

  Training:  BaseWakeModel  ww_trainer/model.py:65
  Inference: OnnxWakeWordInferencer  ww_trainer/inference.py:8
             (onnxruntime only — no torch at runtime)
```

---

## Table of Contents

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | System design: abstractions, data flow, extractor taxonomy, hardware tiers, sliding cache, key design decisions |
| [api.md](api.md) | Full API reference — every public class and method with file:line citations |
| [training.md](training.md) | Step-by-step training guide: dataset prep, tier selection, full CLI reference, loss functions, augmentation, checkpointing |
| [export.md](export.md) | ONNX export guide: exporting extractors and heads, quantization, verification |
| [inference.md](inference.md) | Inference guide: single, batch, and streaming inference; ONNX and PyTorch paths |
| [sweep.md](sweep.md) | Hyperparameter sweep guide with Optuna |

---

## Quick Start

```bash
# Install
pip install ww_trainer

# Install with HuBERT/Wav2Vec2 training support
pip install "ww_trainer[transformers]"

# Train a micro model (MFCC + FFN, ~50K params, no GPU needed)
ww_trainer-train \
  --wake-word hey_jarvis \
  --metadata dataset.csv \
  --tier micro \
  --output-dir ./models/hey_jarvis

# List available hardware tiers
ww_trainer-train --list-tiers

# Full CLI help
ww_trainer-train --help
```

---

## Key Links

- GitHub: https://github.com/TigreGotico/ww_trainer
- Pre-exported MFCC ONNX: https://huggingface.co/TigreGotico/mfcc-onnx
- Funded by NGI0 Commons Fund / NLnet
