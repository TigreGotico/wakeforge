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
- `embed_onnx_metadata` utility in `ww_trainer/utils.py`.
- Integrated into `ClassifierHead.export_to_onnx` (`model.py:52`).
- Integrated into `BaseExtractor.export_to_onnx` (`feats.py:107`).

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

`MfccExtractor` is a pure-PyTorch module. Its `forward` method uses `return_complex=False` in `torch.stft` (`feats.py:380`) specifically to remain ONNX-exportable.

```python
from ww_trainer.feats import MfccExtractor

extractor = MfccExtractor(sr=16000, n_mfcc=40, n_mels=40, n_fft=400, hop_length=160)
extractor.export_to_onnx("mfcc.onnx")
```

`BaseExtractor.export_to_onnx` (`feats.py:107`) uses `torch.onnx.export` with:
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

`HubertExtractor` and `Wav2Vec2Extractor` training wrappers were removed in 0.2.0. Use the standalone export scripts to produce ONNX files once, then load with `OnnxFeatureExtractor` for all subsequent training and inference runs.

```bash
# Export HuBERT tiny to ONNX
.venv/bin/python scripts/export_hubert.py --model voidful/hubert-tiny-v2 --out hubert_tiny.onnx

# Export Wav2Vec2-BERT to ONNX
.venv/bin/python scripts/export_w2vbert.py --out w2vbert.onnx
```

After export, load with `OnnxFeatureExtractor` — `ww_trainer/feats.py:157`:

```python
from ww_trainer.feats import OnnxFeatureExtractor

extractor = OnnxFeatureExtractor("hubert_tiny.onnx", sample_rate=16000, device="cpu")
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

sess = ort.InferenceSession("mfcc.onnx", providers=["CPUExecutionProvider"])

for length in [8000, 16000, 32000]:          # 0.5 s, 1 s, 2 s
    torch.manual_seed(42)
    wav = torch.randn(1, length)

    with torch.no_grad():
        pt_out = extractor(wav).numpy()

    onnx_out = sess.run(None, {"input_values": wav.numpy()})[0]

    diff = np.abs(pt_out - onnx_out)
    corr = np.corrcoef(pt_out.flatten(), onnx_out.flatten())[0, 1]
    print(f"length={length:>6}  MAE={diff.mean():.6f}  MaxAE={diff.max():.6f}  corr={corr:.6f}")
```

Expected output for fixed-transform extractors (MFCC, Filterbank, Gammatone, PLP, PNCC, CQT):

```
length=  8000  MAE=0.000001  MaxAE=0.000012  corr=1.000000
length= 16000  MAE=0.000001  MaxAE=0.000013  corr=1.000000
length= 32000  MAE=0.000001  MaxAE=0.000015  corr=1.000000
```

**Threshold:** MaxAE < 1e-3 is acceptable. Float32 arithmetic divergence between PyTorch and ONNX Runtime is typically < 1e-5. Larger values indicate a tracing issue (dynamic shape, unsupported op, or wrong exporter).

**Dynamo vs TorchScript exporter:** Extractors that use `torch.stft` with `return_complex=True` (Filterbank, PLP, PNCC, CQT, DeltaMFCC) force `dynamo=True` in their `export_to_onnx` override. The dynamo exporter decomposes complex ops differently, which can produce slightly larger but still acceptable numerical differences.

### Bulk export and validate

`scripts/export_and_push_all.py` exports all fixed-transform extractors, runs the three-length divergence check on each before pushing, and prints a summary table:

```
.venv/bin/python scripts/export_and_push_all.py
```

```
[mfcc-mfcc40-mels40-fft400-hop160-onnx] Exporting ...
[mfcc-mfcc40-mels40-fft400-hop160-onnx] Export OK — 44.8 KB
[mfcc-mfcc40-mels40-fft400-hop160-onnx] Validating (PyTorch vs ONNX divergence) ...
    Length           MAE          MaxAE      Corr
  --------  ------------  ------------  --------
      8000      0.000001      0.000012  1.000000
     16000      0.000001      0.000013  1.000000
     32000      0.000001      0.000015  1.000000
```

Any model with `*** HIGH ***` on a row has MaxAE ≥ 1e-3 and is flagged before upload.

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

All fixed-transform extractors are pre-exported and available on HuggingFace under the
[onnx-feature-extractors](https://huggingface.co/collections/TigreGotico/onnx-feature-extractors) collection:

| Repo | Variants | Notes |
|------|----------|-------|
| `TigreGotico/mfcc-onnx` | mfcc13/20/30/40 × mels23/40/64/80 + int8/int16 | Multiple configs in one repo |
| `TigreGotico/filterbank-mels{N}-fft{F}-hop{H}-onnx` | mels=40/64/80/128 | Log-mel spectrogram |
| `TigreGotico/delta-mfcc-mfcc{N}-mels{M}-fft{F}-hop{H}-onnx` | 4 configs | MFCC + Δ + ΔΔ |
| `TigreGotico/gammatone-filters{N}-frame{F}-hop{H}-onnx` | 4 configs | ERB-scale auditory filterbank |
| `TigreGotico/plp-plp{N}-fft{F}-bark{B}-lp{L}-onnx` | 2 configs | Perceptual LP |
| `TigreGotico/pncc-pncc{N}-fft{F}-filters{M}-onnx` | 2 configs | Power-normalized cepstral |
| `TigreGotico/cqt-bins{N}-oct{O}-fmin{F}-onnx` | 3 configs | Constant-Q transform |

**Learnable extractors** (`SincNetExtractor`, `LEAFExtractor`) are not pre-exported because their
parameters are trained per wake word. Export them after training:

```python
trainer.model.load_checkpoint("best_f1.pt")
trainer.model.feature_extractor.export_to_onnx("sincnet_featurizer.onnx")
```

Download and use a pre-exported model directly:

```python
from ww_trainer.feats import OnnxFeatureExtractor

ext = OnnxFeatureExtractor("mfcc_sr16000_mfcc40_mels40_fft400_hop160.onnx")
print(ext.feature_dim)  # 40
```
