# ww-trainer — Inference Guide

How to run wake word detection at runtime using exported ONNX models or PyTorch during development.

---

## 1. ONNX-Only Inference

After exporting the extractor and head to ONNX, runtime inference requires only `numpy` and `onnxruntime`. PyTorch is not imported at all.

This matters for deployment targets (RPi Zero, embedded Linux, Raspberry Pi) where PyTorch is heavy or unavailable.

**What you need (two files — both required):**
1. `best_f1_featurizer.onnx` — exported feature extractor. Input: `[B, T]` float32. Output: `[B, T_frames, F]` float32.
2. `best_f1.onnx` — exported classifier head. Input: `[B, T_frames, F]` float32. Output: `[B]` scalar logit.
3. `vad.onnx` (Optional) — e.g. Silero VAD. Input: `[B, 512]` float32. Output: `[B, 1]` probability.

Pass `featurizer_path` first, `head_path` second — order is mandatory. Swapping the arguments raises a shape mismatch at runtime.

See [export.md](export.md) for how to produce these files.

---

## 2. Multi-ONNX Pipeline (VAD Requirement)

If your model was trained with `SileroVadWrapper`, the classifier head expects an extra input channel for the VAD probability. The `OnnxWakeWordInferencer` handles this by running a second ONNX session for the VAD and concatenating the result to the features.

```python
from ww_trainer.inference import OnnxWakeWordInferencer

inferencer = OnnxWakeWordInferencer(
    extractor_path="mfcc.onnx",
    head_path="head.onnx",
    vad_path="silero_vad.onnx",  # Optional VAD model
    sample_rate=16000
)

# Inference works the same way; alignment and concatenation are internal
prob = inferencer.infer(audio)
```

**Implementation Details:**
- **Alignment**: Neural VADs often operate on fixed-size chunks (e.g. 512 samples). The inferencer automatically pads the audio, runs batch inference, and uses `numpy.interp` to linearly align the probabilities with the extractor's timeframe.
- **Dependency-Free**: This multi-model pipeline uses only `numpy` and `onnxruntime`.

---

## 3. Single Inference

`OnnxWakeWordInferencer.infer` — `inference.py:36`

```python
from ww_trainer.inference import OnnxWakeWordInferencer
import numpy as np
import soundfile as sf

inferencer = OnnxWakeWordInferencer(
    extractor_path="mfcc.onnx",
    head_path="head.onnx",
    sample_rate=16000,
    device="cpu",
)

# Load a WAV file
audio, sr = sf.read("test.wav")
if audio.ndim > 1:
    audio = audio.mean(axis=1)  # to mono
audio = audio.astype(np.float32)

prob = inferencer.infer(audio)
print(f"Wake word probability: {prob:.4f}")
if prob > 0.5:
    print("Wake word detected!")
```

`infer` (`inference.py:36`–`55`):
1. Adds a batch dimension: `audio[np.newaxis, :]` → `[1, T]`.
2. Runs extractor ONNX session → `[1, T_frames, F]`.
3. Runs head ONNX session → scalar logit.
4. Returns `sigmoid(logit)` as a Python float.

---

## 4. Batch Inference

`OnnxWakeWordInferencer.infer_batch` — `inference.py:57`

For higher throughput when processing many clips at once.

```python
import numpy as np
from ww_trainer.inference import OnnxWakeWordInferencer

inferencer = OnnxWakeWordInferencer("mfcc.onnx", "head.onnx")

# audio_batch: [B, T] — all clips must be the same length
audio_batch = np.zeros((8, 16000), dtype=np.float32)

probs = inferencer.infer_batch(audio_batch)  # np.ndarray shape [8], float32
print(probs)
```

Pads to equal length before passing to `infer_batch`. If your clips have different lengths, pad them manually first.

---

## 5. Streaming Inference

`OnnxWakeWordInferencer.infer_streaming` — `inference.py:74`

Streaming inference processes audio in small chunks and maintains a rolling feature cache. This is the correct approach for live microphone input — you never have a full utterance, only small frames arriving in real time.

**How it works:**
1. Each call passes a new audio chunk to the extractor.
2. New features are appended to the cache.
3. The cache is truncated to the last 50 frames (`inference.py:96`–`98`).
4. The head runs on the cached features.
5. Returns the probability and the updated cache.

```python
from ww_trainer.inference import OnnxWakeWordInferencer
import numpy as np

inferencer = OnnxWakeWordInferencer(
    extractor_path="mfcc.onnx",
    head_path="head.onnx",
    sample_rate=16000,
    device="cpu",
)

# Simulate microphone: 250 ms chunks at 16 kHz = 4000 samples per chunk
CHUNK_SIZE = 4000
THRESHOLD = 0.5

cache = None  # None on first call

def process_audio_stream(stream):
    """stream is an iterator of float32 numpy arrays of shape [CHUNK_SIZE]"""
    global cache
    for chunk in stream:
        prob, cache = inferencer.infer_streaming(chunk, cache)
        if prob > THRESHOLD:
            print(f"Wake word detected! (prob={prob:.3f})")
            cache = None  # optionally reset after detection

# Example with simulated audio
stream = [np.zeros(CHUNK_SIZE, dtype=np.float32) for _ in range(20)]
process_audio_stream(stream)
```

**Window size:** The default cache size is 50 frames. For a 10 ms hop (160 samples at 16 kHz), that covers 500 ms of audio — enough for a typical wake word. The window size is hardcoded in `infer_streaming` (`inference.py:97`). To change it, use the PyTorch path with `SlidingFeatureCacheTensor(window_size=N)`.

---

## 6. PyTorch Inference (Development)

During development or testing you can use the PyTorch model directly without exporting to ONNX.

### Single waveform

`BaseWakeModel.infer` — `model.py:127`

```python
import numpy as np
from ww_trainer.feats import MfccExtractor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel

extractor = MfccExtractor(sr=16000, n_mfcc=40)
head = FfnClassifierHead(input_size=40, hidden_dim=128)
model = BaseWakeModel(extractor, head)
model.load_checkpoint("best_f1.pt")
model.eval()

audio = np.zeros(16000, dtype=np.float32)
prob = model.infer(audio)
print(f"Probability: {prob:.4f}")
```

### Streaming with `SlidingFeatureCacheTensor`

`BaseWakeModel.forward_streaming` — `model.py:110`

```python
import torch
from ww_trainer.feats import MfccExtractor, SlidingFeatureCacheTensor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel

extractor = MfccExtractor(sr=16000, n_mfcc=40)
head = FfnClassifierHead(input_size=40, hidden_dim=128)
model = BaseWakeModel(extractor, head)
model.load_checkpoint("best_f1.pt")
model.eval()

FEATURE_DIM = extractor.feature_dim   # 40
WINDOW_SIZE = 50

cache = SlidingFeatureCacheTensor(feature_dim=FEATURE_DIM, window_size=WINDOW_SIZE)

CHUNK_SAMPLES = 4000
THRESHOLD = 0.5

def process_chunk(chunk_np):
    chunk = torch.tensor(chunk_np, dtype=torch.float32)
    with torch.no_grad():
        prob = model.forward_streaming(chunk, cache)
    return prob

# Simulate 20 chunks of 250 ms
for i in range(20):
    chunk = torch.zeros(CHUNK_SAMPLES).numpy()
    prob = process_chunk(chunk)
    if prob > THRESHOLD:
        print(f"Wake word detected at chunk {i}! prob={prob:.3f}")
```

`SlidingFeatureCacheTensor` (`feats.py:27`) is updated **in-place** every call. Reset it by creating a new instance after a detection event.

---

## 7. Device Selection

For `OnnxWakeWordInferencer`:

| `device` | Behavior |
|----------|---------|
| `"auto"` | Checks `ort.get_all_providers()` for `CUDAExecutionProvider`; uses CUDA if available |
| `"cuda"` | Forces `CUDAExecutionProvider, CPUExecutionProvider` |
| `"cpu"` | Forces `CPUExecutionProvider` only |

For PyTorch `BaseWakeModel`:

| `device` | Behavior |
|----------|---------|
| `"auto"` | `torch.cuda.is_available()` check at construction |
| `"cuda"` | Forces CUDA |
| `"cpu"` | Forces CPU |

---

## 8. Latency Considerations

Feature extraction dominates latency. Rough comparisons for a single 1-second clip:

| Extractor | Runtime | CPU Latency (approx) |
|-----------|---------|---------------------|
| `MfccExtractor` (PyTorch) | PyTorch | < 1 ms |
| MFCC ONNX | onnxruntime | < 1 ms |
| HuBERT tiny ONNX | onnxruntime | 10–50 ms |
| HuBERT base ONNX | onnxruntime | 100–500 ms |

MFCC extractors are 2–3 orders of magnitude faster than neural extractors. On embedded targets, always use MFCC (`micro` or `small` tier).

For streaming, the relevant number is the per-chunk latency, not full-waveform latency. With MFCC and a 250 ms chunk:
- Feature extraction: << 1 ms.
- GRU head: << 1 ms.
- Total: well within real-time budget even on RPi Zero.

---


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
