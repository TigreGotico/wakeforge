# ww-trainer — ONNX Export Guide

How to export trained models to ONNX for deployment.

---

## 4. Metadata Embedding

Exported models automatically include rich metadata describing the training run. This allows deployment tools to identify the model and its expected performance without external config files.

Embedded keys include:
- `wake_word`: Name of the keyword (e.g. "hey_mycroft")
- `arch`: Head architecture (GRU, CNN, etc.)
- `epoch`: Training epoch
- `featurizer`: Featurizer class name
- `metric_f1`: Best F1 score achieved

To inspect metadata:
```python
import onnx
model = onnx.load("model.onnx")
for prop in model.metadata_props:
    print(f"{prop.key}: {prop.value}")
```

Implementation details:
- `embed_onnx_metadata` utility in `utils.py:295`.
- Integrated into `ClassifierHead.export_to_onnx` (`model.py:40`).
- Integrated into `BaseExtractor.export_to_onnx` (`feats.py:80`).

---

## 5. Hybrid Pipeline Export (Markov/HMM)

`MarkovTransitionExtractor` and `HMMStateExtractor` support "Hybrid Export" — they export the entire chain from raw audio to features into a single ONNX file.

```python
from ww_trainer.feats import MfccExtractor, MarkovTransitionExtractor

base = MfccExtractor()
markov = MarkovTransitionExtractor(base, n_codes=16)
# Train the extractor
markov.fit(audio_samples)
# Export the full pipeline
markov.export_to_onnx("markov_pipeline.onnx")
```

The exported graph takes raw waveforms (`input_values`) and outputs enriched features (`features`). The transition matrices and VQ codebooks are baked into the ONNX graph as constants.

---

## 1. Why ONNX

Wake word detection typically runs on embedded or constrained hardware where installing PyTorch is impractical (RPi Zero, MCUs, home assistants). ONNX Runtime is a lightweight, cross-platform inference engine with C, Python, and .NET bindings.

The design separates the extractor from the classifier head (`architecture.md`) so each can be:
- Exported to ONNX independently.
- Loaded independently at inference time.
- Replaced or updated without re-exporting the other.

After export, `OnnxWakeWordInferencer` (`inference.py:8`) runs inference with only `numpy` and `onnxruntime` — no PyTorch import anywhere.

---

## 2. Exporting `MfccExtractor`

`MfccExtractor` is a pure-PyTorch module. Its `forward` method uses `return_complex=False` in `torch.stft` (`feats.py:228`) specifically to remain ONNX-exportable.

```python
from ww_trainer.feats import MfccExtractor

extractor = MfccExtractor(sr=16000, n_mfcc=40, n_mels=40, n_fft=400, hop_length=160)
extractor.export_to_onnx("mfcc.onnx")
```

`BaseExtractor.export_to_onnx` (`feats.py:77`) uses `torch.onnx.export` with:
- Opset 18.
- Dynamic axes: batch size (`batch_size`) and time (`time`) on both input and output.
- `do_constant_folding=True`.
- `TrainingMode.EVAL`.
- Verifies the exported model with `onnx.checker.check_model`.

Expected ONNX shapes:
- Input `input_values`: `[batch_size, time]` float32.
- Output `features`: `[batch_size, time, n_mfcc]` float32.

A pre-exported version (40 coefficients, 16 kHz) is available at https://huggingface.co/TigreGotico/mfcc-onnx.

---

## 3. Exporting HuBERT and Wav2Vec2

These extractors require `transformers` and should be exported once and then loaded via `OnnxFeatureExtractor` for all subsequent training and inference.

```python
from ww_trainer.feats import HubertExtractor

extractor = HubertExtractor(
    model_name="voidful/hubert-tiny-v2",
    sample_rate=16000,
    device="cpu",
)
extractor.export_to_onnx("hubert_tiny.onnx")
```

The `scripts/export_w2vbert.py` script shows the same pattern for Wav2Vec2 (`scripts/export_w2vbert.py:84`–`87`).

After export, load via `OnnxFeatureExtractor` for training:

```python
from ww_trainer.feats import OnnxFeatureExtractor

extractor = OnnxFeatureExtractor("hubert_tiny.onnx", sample_rate=16000, device="cuda")
print(extractor.feature_dim)  # auto-detected from ONNX output shape
```

Then pass `featurizer_type="onnx"` to `WakeWordTrainer`.

---

## 4. Exporting a Classifier Head

`ClassifierHead.export_to_onnx` (`model.py:30`) exports the head only.

```python
from ww_trainer.model import GruClassifierHead

head = GruClassifierHead(input_size=40, hidden_dim=128)
# load weights first
head.load_state_dict(...)
head.export_to_onnx("head.onnx")
```

Expected ONNX shapes:
- Input `input_features`: `[1, T_features, input_size]` float32 (time dimension is dynamic).
- Output `logits`: scalar float32.

---

## 5. Exporting a Full Model

`BaseWakeModel.export_to_onnx` (`model.py:136`) is a convenience method that exports the head, and optionally the extractor.

```python
from ww_trainer.feats import MfccExtractor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel

extractor = MfccExtractor()
head = FfnClassifierHead(input_size=40, hidden_dim=128)
model = BaseWakeModel(extractor, head)

# Load trained weights
model.load_checkpoint("best_f1.pt")

# Export head only
model.export_to_onnx("head.onnx")

# Export both head and extractor
model.export_to_onnx("head.onnx", export_featurizer=True)
# writes: head.onnx, head_featurizer.onnx
```

When `--export-onnx` is passed to the CLI, `WakeWordTrainer.save_intermediate_ckpt` (`trainer.py:630`–`645`) calls `model.export_to_onnx(onnx_path)` after every checkpoint save.

---

## 6. Quantization

Both export methods accept `quantize=True`.

**`BaseExtractor.export_to_onnx(out, quantize=True)` — `feats.py:96`–`102`:**

Writes two additional files:
- `<stem>_int16.onnx` — `QInt16` via `onnxruntime.quantization.quantize_dynamic`.
- `<stem>_int8.onnx` — `QInt8` via `quantize_dynamic`.

Both quantize `MatMul` and `Gemm` operations.

**`ClassifierHead.export_to_onnx(out, quantize=True)` — `model.py:57`–`62`:**

Writes one additional file:
- `<stem>_int8.onnx` — `QInt8`.

```python
extractor.export_to_onnx("mfcc.onnx", quantize=True)
# creates: mfcc.onnx, mfcc_int16.onnx, mfcc_int8.onnx

head.export_to_onnx("head.onnx", quantize=True)
# creates: head.onnx, head_int8.onnx
```

**When to use quantization:**
- INT8 is suitable for CPU inference on modern x86 and ARM targets.
- INT16 is a lighter quantization that usually preserves more accuracy.
- MFCC extractor benefits less from quantization (computationally cheap already).
- HuBERT/Wav2Vec2 heads benefit most — large matrix multiplications.

---

## 7. Verifying ONNX Exports

`export_to_onnx` calls `onnx.checker.check_model(onnx_model)` automatically (`feats.py:92`–`93`, `model.py:53`–`54`). A passing check means the graph is structurally valid.

To verify numerically (check that ONNX and PyTorch agree):

```python
import numpy as np
import torch
import onnxruntime as ort
from ww_trainer.feats import MfccExtractor

extractor = MfccExtractor()
extractor.export_to_onnx("mfcc.onnx")

# PyTorch output
wav = torch.randn(1, 16000)
with torch.no_grad():
    pt_out = extractor(wav).numpy()

# ONNX output
sess = ort.InferenceSession("mfcc.onnx", providers=["CPUExecutionProvider"])
onnx_out = sess.run(None, {"input_values": wav.numpy()})[0]

diff = np.abs(pt_out - onnx_out).max()
print(f"Max absolute difference: {diff:.6f}")  # should be < 1e-5
```

---

## 8. Loading in `OnnxFeatureExtractor`

After exporting, load the ONNX extractor back into training with `OnnxFeatureExtractor` (`feats.py:105`):

```python
from ww_trainer.feats import OnnxFeatureExtractor

ext = OnnxFeatureExtractor("mfcc.onnx", sample_rate=16000, device="cpu")
print(ext.feature_dim)  # 40
```

Or pass directly to `WakeWordTrainer`:

```python
from ww_trainer.trainer import WakeWordTrainer

trainer = WakeWordTrainer(
    arch="gru",
    featurizer="mfcc.onnx",
    featurizer_type="onnx",
    losses_cfg=[{"name": "bce", "weight": 1.0}],
)
```

---

## 9. Pre-exported Models

A pre-exported MFCC ONNX model (40 coefficients, 16 kHz, with INT8 and INT16 variants) is available at:

https://huggingface.co/TigreGotico/mfcc-onnx

This is the same model built by `scripts/export_mfcc.py` (`scripts/export_mfcc.py:111`).

Download and use directly:

```python
from ww_trainer.feats import OnnxFeatureExtractor

ext = OnnxFeatureExtractor("mfcc_sr16000_mfcc40_mels40_fft400_hop160.onnx")
print(ext.feature_dim)  # 40
```
