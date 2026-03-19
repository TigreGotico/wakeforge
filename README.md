# Wake Word Trainer

A full training and research suite for wake word detection models of all sizes — from microcontrollers to servers. Every feature extractor exports to ONNX; production inference requires only `onnxruntime`.

---

## Project Description

`ww-trainer` implements the `feature-extractor + classifier-head` architecture for wake word detection:

- **Training time:** Classical (MFCC) or neural (HuBERT, Wav2Vec2) extractors feed into lightweight classifier heads (FFN, GRU, CNN). Both components are trained together and exported to ONNX separately.
- **Inference time:** Two ONNX sessions (extractor + head) are loaded by `OnnxWakeWordInferencer`. No PyTorch required at runtime — only `onnxruntime` and `numpy`.
- **Hardware tiers:** Four preset configurations targeting everything from MCUs to workstations.

---

## Hardware Tier Table

| Tier | Extractor | Head | Input | Params | Target Hardware |
|------|-----------|------|-------|--------|----------------|
| micro | MFCC (40 coeff) | FFN (128d) | raw audio | ~50K | MCU, RPi Zero |
| small | MFCC (40 coeff) | GRU (128d) | raw audio | ~200K | RPi, small SBC |
| medium | HuBERT-base ONNX | FFN (128d) | raw audio | ~90M feat + 200K head | RPi 4, laptop |
| large | W2VBert/HuBERT | GRU (256d, bidir) | raw audio | ~300M feat + 1M head | Server/workstation |

Use `ww_trainer-train --list-tiers` to print this table at any time.

---

## Quickstart

```bash
pip install ww_trainer
# or with transformers support (HuBERT/Wav2Vec2 training-time extractors):
pip install "ww_trainer[transformers]"

# Train a micro model (MFCC + FFN, ~50K params)
ww_trainer-train \
  --ww-name hey_jarvis \
  --metadata dataset.csv \
  --tier micro \
  --output-dir ./models/hey_jarvis

# Train with a pre-exported HuBERT ONNX extractor
ww_trainer-train \
  --ww-name hey_jarvis \
  --metadata dataset.csv \
  --featurizer distilhubert.onnx \
  --arch gru \
  --output-dir ./models/hey_jarvis
```

---

## Dataset Format

The training CSV has two columns with no header:

```
/path/to/audio/hey_jarvis_001.wav,wake
/path/to/audio/background_noise_042.wav,not_wake
```

| Column | Values |
|--------|--------|
| `path` | Absolute or relative path to a WAV/FLAC/MP3/OGG/M4A file |
| `label` | `wake` or `not_wake` (or `1` / `0`) |

---

## Extractor Matrix

| Extractor class | Type | ONNX-exportable | Requires |
|----------------|------|-----------------|---------|
| `MfccExtractor` | Classical | Yes | `torch` |
| `OnnxFeatureExtractor` | Runtime loader | Yes (loads pre-exported) | `onnxruntime` |
| `HubertExtractor` | Neural (train-time) | Yes | `transformers` |
| `Wav2Vec2Extractor` | Neural (train-time) | Yes | `transformers` |

All extractors subclass `BaseExtractor` and share the uniform contract:
- `forward(wavs) -> Tensor[B, T, F]`
- `export_to_onnx(out_path)` — exports to ONNX with dynamic batch/time axes

---

## ONNX-Only Inference

After training, inference requires only `numpy` and `onnxruntime`:

```python
from ww_trainer.inference import OnnxWakeWordInferencer
import numpy as np

inferencer = OnnxWakeWordInferencer("extractor.onnx", "head.onnx")
audio = np.zeros(16000, dtype=np.float32)  # replace with real audio
prob = inferencer.infer(audio)
print(f"Wake word probability: {prob:.3f}")

# Streaming (rolling feature cache, no full-window recomputation)
cache = None
chunk = np.zeros(4000, dtype=np.float32)
prob, cache = inferencer.infer_streaming(chunk, cache)
```

---

## Architecture

The system is a two-stage pipeline:

```
[Feature Extractor]   →   [Classifier Head]   →   [Wake Word Model]
  BaseExtractor              ClassifierHead           BaseWakeModel
  ├── MfccExtractor           ├── FfnClassifierHead    feature_extractor + classifier
  ├── OnnxFeatureExtractor    ├── CnnClassifierHead
  ├── HubertExtractor (train) └── GruClassifierHead
  └── Wav2Vec2Extractor (train)
         ↓ export_to_onnx()        ↓ export_to_onnx()
     extractor.onnx            classifier_head.onnx
              \                         /
               └── inference.py ───────
                   OnnxWakeWordInferencer
                   (onnxruntime ONLY, no torch)
```

**Key invariants:**
- `BaseExtractor.forward(wavs) -> [B, T, F]` — uniform output for all extractors
- `ClassifierHead.forward(feats: [B, T, F]) -> [B]` — uniform input for all heads
- Both `export_to_onnx()` methods are always available (not optional)

---

## Credits

This work was made possible by the generous grant from [NGI0 Commons Fund](https://nlnet.nl/project/OpenVoiceOS)

![](./ngi.png)

> This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund), a fund established by [NLnet](https://nlnet.nl) with financial support from the European Commission's [Next Generation Internet](https://ngi.eu) programme, under the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en) under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429). Additional funding is made available by the [Swiss State Secretariat for Education, Research and Innovation](https://www.sbfi.admin.ch/sbfi/en/home.html) (SERI).
