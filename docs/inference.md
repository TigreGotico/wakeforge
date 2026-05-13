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
