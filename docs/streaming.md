# Streaming Inference

Processing audio in real-time chunks rather than full utterances. Two paths: PyTorch (development) and ONNX (production).

---

## Core Concept: Sliding Feature Cache

Wake word detection on live audio receives small chunks (e.g. 250ms) continuously. Each chunk produces new feature frames that must be combined with recent history before classification.

Both paths maintain a rolling window of the most recent feature frames (default: 50 frames = ~500ms at 10ms hop).

---

## ONNX Path (Production)

`OnnxWakeWordInferencer.infer_streaming` -- `inference.py:74`

No PyTorch dependency. Cache is a plain numpy array.

**Per-call flow:**
1. Extract features from chunk: `extractor.run(audio_chunk)` -> `[T_new, F]` (`inference.py:86-88`)
2. Concatenate with cache: `np.concatenate([cache, new_feats])` (`inference.py:93`)
3. Truncate to last 50 frames (`inference.py:96-98`)
4. Run head on cached features -> sigmoid probability (`inference.py:100-105`)
5. Return `(probability, updated_cache)`

```python
from ww_trainer.inference import OnnxWakeWordInferencer
import numpy as np

inf = OnnxWakeWordInferencer("mfcc.onnx", "head.onnx")
cache = None

for chunk in audio_stream:  # float32 arrays, e.g. 4000 samples (250ms)
    prob, cache = inf.infer_streaming(chunk, cache)
    if prob > 0.5:
        print("Wake word detected!")
        cache = None  # reset after detection
```

**Window size:** Hardcoded at 50 frames (`inference.py:96`). To change, modify the source or use the PyTorch path.

---

## PyTorch Path (Development)

`BaseWakeModel.forward_streaming` -- `model.py:110`

Uses `SlidingFeatureCacheTensor` (`feats.py:27`) -- an `nn.Module` with an in-place buffer.

**Per-call flow:**
1. Extract features: `self.feature_extractor([audio_chunk])` -> `[1, T_new, F]` (`model.py:122`)
2. Update cache: `cache(feats.squeeze(0))` -> `[T_window, F]` (`model.py:123`)
3. Classify: `self.classifier.forward(cached.unsqueeze(0))` -> logit (`model.py:124`)
4. Return `sigmoid(logit)` as float (`model.py:125`)

### `SlidingFeatureCacheTensor` -- `feats.py:27`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `feature_dim` | 768 | Feature dimension F |
| `window_size` | 50 | Max frames retained |

Internal state: `feature_cache` buffer `[window_size, F]` and `current_len` counter. Updated in-place via `forward(new_feats)` (`feats.py:36`). When buffer overflows, old frames shift out (`feats.py:46`).

```python
import torch
from ww_trainer.feats import MfccExtractor, SlidingFeatureCacheTensor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel

extractor = MfccExtractor(n_mfcc=40)
head = FfnClassifierHead(input_size=40, hidden_dim=128)
model = BaseWakeModel(extractor, head)
model.load_checkpoint("best_f1.pt")
model.eval()

cache = SlidingFeatureCacheTensor(feature_dim=40, window_size=50)

for chunk_np in audio_stream:
    chunk = torch.tensor(chunk_np, dtype=torch.float32)
    with torch.no_grad():
        prob = model.forward_streaming(chunk, cache)
    if prob > 0.5:
        cache = SlidingFeatureCacheTensor(feature_dim=40, window_size=50)  # reset
```

---

## PyTorch vs ONNX Comparison

| Aspect | PyTorch | ONNX |
|--------|---------|------|
| Dependency | `torch` | `numpy` + `onnxruntime` |
| Cache type | `SlidingFeatureCacheTensor` (nn.Module buffer) | `np.ndarray` |
| Window size | Configurable via constructor | Hardcoded 50 frames |
| Reset | Create new `SlidingFeatureCacheTensor` | Set `cache = None` |
| Use case | Development, testing | Production deployment |
| Latency | Higher (PyTorch overhead) | Lower |

---

## Chunk Size Guidelines

| Chunk duration | Samples (16kHz) | Latency tradeoff |
|---------------|-----------------|------------------|
| 100ms | 1600 | Lower latency, more CPU overhead per second |
| 250ms | 4000 | Good balance for most deployments |
| 500ms | 8000 | Lower CPU, higher detection latency |

With MFCC extractor and 250ms chunks, per-chunk latency is well under 1ms on any hardware including RPi Zero.
