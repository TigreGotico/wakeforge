# ww-trainer — Architecture

Deep-dive into the system design: why it is structured the way it is, how data flows through it, and what every major component does.

---

## Core Abstraction: Three Layers

The entire system rests on three abstract types defined in `ww_trainer/feats.py` and `ww_trainer/model.py`.

### `BaseExtractor` — `ww_trainer/feats.py:70`

Converts raw audio `[B, T]` float32 waveforms into dense frame-level feature tensors `[B, T_frames, F]`.

Every concrete extractor must:
1. Implement the `feature_dim` property (`feats.py:97`) — the integer `F` that downstream heads need to know their `input_size`.
2. Implement `forward(wavs: WavInput) -> Tensor[B, T_frames, F]` (`feats.py:101`).
3. Inherit `export_to_onnx(out, quantize=False, dynamo=False)` from the base class (`feats.py:107`).

The split between extractor and head exists so that a heavy neural extractor (HuBERT, ~90M parameters) can be:
- Exported to ONNX **once** and reused across many training runs without re-exporting.
- Shared by multiple heads simultaneously via the `shared_extractor` argument to `WakeWordTrainer`.
- Replaced at inference time without changing the head.

### `ClassifierHead` — `ww_trainer/model.py:27`

Maps feature frames `[B, T_frames, F]` to a scalar logit `[B]` (pre-sigmoid).

Every concrete head must implement:
- `forward(feats) -> Tensor[B]` (`model.py:45`) — full forward pass returning logits.
- `embed(feats) -> Tensor[B, D]` (`model.py:49`) — returns the pre-logit embedding for metric learning and visualization.

The `embed()` method is what makes metric losses (triplet, RPPL, cn2pair) possible.

### `BaseWakeModel` — `ww_trainer/model.py:93`

Combines one extractor and one head. Holds no learnable parameters directly; they all live in the extractor and head submodules.

`BaseWakeModel.forward(wavs)` (`model.py:147`) calls `feature_extractor(wavs)` then `classifier.forward(feats)`. `BaseWakeModel.embed(wavs)` (`model.py:168`) calls `feature_extractor(wavs)` then `classifier.embed(feats)`.

---

## Hybrid & Multi-ONNX Pipelines

While the standard architecture uses a single feature extractor and a single head, `ww-trainer` supports two advanced patterns for robust edge deployment:

### 1. Hybrid Pipeline Export (Markov/HMM)
For classical sequential extractors like `MarkovTransitionExtractor` or `HMMStateExtractor`, the entire processing chain—from raw waveforms to enriched features—is symbolic and can be traced into a single ONNX graph. This includes the base featurizer (e.g., MFCC), the vector quantization (VQ) codebook, and the Markov/HMM logic.

### 2. Multi-ONNX Pipeline (VAD Requirement)
When using pre-trained neural models like **Silero VAD**, the system transitions from a single-graph model to a multi-model pipeline.
- **Training**: Silero VAD is treated as a parallel feature stream that is interpolated and concatenated to the base features.
- **Inference**: The `OnnxWakeWordInferencer` manages separate ONNX sessions for the base featurizer, the VAD model, and the classifier head. It performs the necessary temporal alignment and stream fusion at runtime using only NumPy.

This modularity allows for "Best-of-Breed" component swapping (e.g., swapping a TinyVAD for Silero) without retraining the core primary classifier head.

---

## Data Flow Diagram

### Training path (audio → loss)

```
CSV file: /path/audio.wav,1
     |
     v
AudioDataset.__getitem__          dataset.py:269
  FeatureCache.get(path)?         cache.py (if cache hit + no augment → skip load)
  torchaudio.load(path)
  resample to 16 kHz if needed    dataset.py:284-289
  optional augmentation           dataset.py:295-296
  FeatureCache.put(path) if no augment
     |  wav: Tensor[T]  float32
     v
collate_fn(batch, device)         dataset.py:301
  zero-pad all wavs to max_len
  stack into Tensor[B, T]
  move to device
     |  wavs: Tensor[B, T]
     |  labels: Tensor[B]  float32
     v
BaseWakeModel.forward(wavs)       model.py:86
  BaseExtractor.forward(wavs)
     |  feats: Tensor[B, T_frames, F]
  ClassifierHead.forward(feats)
     |  logits: Tensor[B]
     v
BaseWakeModel.embed(wavs)         model.py:91
  (same extractor call, head.embed)
     |  embeds: Tensor[B, D]
     v
LossManager.compute_loss(         loss.py:454
  model, wavs, labels, dataset_ref)
  → total scalar loss
     v
optimizer.step()
```

### Inference path (audio → probability, ONNX only)

```
audio: np.ndarray  shape [T]  float32
     |
     v
OnnxWakeWordInferencer.infer(audio)    inference.py:36
  audio[np.newaxis, :]  →  shape [1, T]
  extractor ONNX session.run()
     |  feats: np.ndarray  [1, T_frames, F]
  head ONNX session.run()
     |  logit: scalar
  sigmoid(logit)
     |
     v
float probability in [0, 1]
```

### Streaming inference path

```
audio_chunk: np.ndarray  shape [chunk_T]
cache: np.ndarray | None  shape [T_cached, F]
     |
     v
OnnxWakeWordInferencer.infer_streaming    inference.py:74
  extractor ONNX session on [1, chunk_T]
     |  new_feats: [T_new, F]
  cache = concat(cache, new_feats)
  cache = cache[-50:]   (last 50 frames)
  head ONNX session on [1, T_cached, F]
     |
     v
(probability, updated_cache)
```

---

## ONNX Pipeline: Training to Runtime

```
Training (PyTorch)                      Runtime (onnxruntime only)

BaseExtractor (MfccExtractor)           extractor.onnx
  .export_to_onnx("extractor.onnx")  →  OnnxFeatureExtractor (optional, for training)
                                         or directly in OnnxWakeWordInferencer.extractor

ClassifierHead (GruClassifierHead)      head.onnx
  .export_to_onnx("head.onnx")       →  OnnxWakeWordInferencer.head

No PyTorch at runtime.
```

`BaseExtractor.export_to_onnx` (`feats.py:77`): uses `torch.onnx.export` with opset 18, dynamic axes for both batch and time dimensions, `do_constant_folding=True`, `TrainingMode.EVAL`. Verifies with `onnx.checker.check_model`.

`ClassifierHead.export_to_onnx` (`model.py:30`): same approach. Dynamic axis on the time dimension (`T_features`) of the feature input. Input name `input_features`; output name `logits`.

When `quantize=True`:
- `BaseExtractor.export_to_onnx`: writes both `_int16.onnx` and `_int8.onnx` via `quantize_dynamic` (`feats.py:96`–`102`).
- `ClassifierHead.export_to_onnx`: writes `_int8.onnx` only (`model.py:57`–`62`).

---

## Extractor Taxonomy

| Class | Type | `feature_dim` source | Output shape | ONNX-exportable | Requires |
|-------|------|---------------------|--------------|-----------------|---------|
| `MfccExtractor` | Classical DSP | `n_mfcc` parameter (`feats.py:181`) | `[B, T, n_mfcc]` | Yes (pure PyTorch) | `torch` only |
| `OnnxFeatureExtractor` | Runtime ONNX loader | ONNX output shape or dummy run (`feats.py:126`) | `[B, T, F]` | Already ONNX | `onnxruntime` |
| `HubertExtractor` | Neural (HuBERT) | `hubert.config.hidden_size` (`feats.py:264`) | `[B, T, hidden]` | Yes (via base class) | `transformers` |
| `Wav2Vec2Extractor` | Neural (Wav2Vec2) | `model.config.hidden_size` (`feats.py:296`) | `[B, T, hidden]` | Yes (via base class) | `transformers` |

`MfccExtractor` uses `return_complex=False` in `torch.stft` (`feats.py:228`) specifically to remain ONNX-exportable (complex return values are not yet supported by the ONNX exporter).

`HubertExtractor` and `Wav2Vec2Extractor` mark `_REQUIRES_TRANSFORMERS = True` (`feats.py:249`, `feats.py:283`) as a documentation convention. Import is guarded with `try/except ImportError` in both `__init__` methods and in `WakeWordTrainer.create_model` (`trainer.py:140`–`150`).

---

## Hardware Tiers

Defined in `ww_trainer/tiers.py`. Each tier is a `TierConfig` dataclass (`tiers.py:12`) specifying a complete extractor+head configuration.

| Tier | Extractor | Head | Hidden | n_mfcc | Bidirectional | Layers | Params | Target |
|------|-----------|------|--------|--------|---------------|--------|--------|--------|
| `micro` | `mfcc` | `ffn` | 128 | 40 | No | 1 | ~50K | MCU, RPi Zero |
| `small` | `mfcc` | `gru` | 128 | 40 | No | 1 | ~200K | RPi, small SBC |
| `medium` | `onnx` | `ffn` | 128 | — | No | 1 | ~90M feat + 200K head | RPi 4, laptop |
| `large` | `hubert` | `gru` | 256 | — | Yes | 2 | ~300M feat + 1M head | Server/workstation |

`HARDWARE_TIERS` dict: `tiers.py:26`. `get_tier(name)`: `tiers.py:83`. `list_tiers()` formatted table: `tiers.py:70`.

When `--tier` is passed to the CLI (`trainer.py:795`–`805`), it sets `arch`, `featurizer_type`, `hidden_dim`, `bidirectional`, `gru_n_layers`, and optionally `n_mfcc`.

---

## Shared Extractor Pattern

When training multiple wake words (or ablations), you can build one extractor and pass it to all trainers to avoid loading it multiple times:

```python
from ww_trainer.feats import OnnxFeatureExtractor
from ww_trainer.trainer import WakeWordTrainer

extractor = OnnxFeatureExtractor("distilhubert.onnx", device="cuda")

trainer_a = WakeWordTrainer(
    arch="gru", featurizer="", losses_cfg=[{"name": "bce", "weight": 1.0}],
    shared_extractor=extractor,
)
trainer_b = WakeWordTrainer(
    arch="ffn", featurizer="", losses_cfg=[{"name": "bce", "weight": 1.0}],
    shared_extractor=extractor,
)
```

`WakeWordTrainer.create_model` checks `if shared_extractor is not None` first (`trainer.py:132`) and skips constructing a new extractor if one is provided.

---

## Sliding Feature Cache

`SlidingFeatureCacheTensor` — `ww_trainer/feats.py:27`

Used in streaming inference to accumulate the most recent `window_size` feature frames without recomputing the entire history.

**Internal state:**
- `feature_cache: Tensor[window_size, feature_dim]` — registered as a buffer (`feats.py:33`).
- `current_len: int` — number of valid frames currently in the buffer.

**`forward(new_feats: Tensor[T_new, F])` — `feats.py:36`:**

1. Computes shift = `max(0, current_len + T_new - window_size)`.
2. If shift > 0, copies `feature_cache[shift:current_len]` to `feature_cache[:new_len]` using `.clone()` to avoid aliasing (`feats.py:46`).
3. Appends `new_feats` at position `current_len`.
4. Returns `feature_cache[:current_len]` — always a `[T_current, F]` slice.

The cache is updated **in-place** (it is a `nn.Module` buffer). The ONNX inference equivalent is the numpy rolling array in `OnnxWakeWordInferencer.infer_streaming` (`inference.py:93`–`98`), which keeps the last 50 frames by simple array concatenation and slicing.

`BaseWakeModel.forward_streaming` (`model.py:110`) uses this cache:

```python
feats = self.feature_extractor([audio_chunk])   # [1, T_new, F]
cached = cache(feats.squeeze(0))                # [T_window, F]
logit = self.classifier.forward(cached.unsqueeze(0))  # [1]
return torch.sigmoid(logit).item()
```

---

## Key Design Decisions

### ONNX-only inference

PyTorch is not required at runtime. This is intentional: wake word detection runs on embedded devices (RPi Zero, MCUs) where installing PyTorch is impractical. `OnnxWakeWordInferencer` (`inference.py:8`) imports only `numpy` and `onnxruntime`.

### `feature_dim` as a property, not a constructor argument

If `feature_dim` were a constructor argument to the head, callers would have to know it upfront. By making it a property on every `BaseExtractor` subclass, `WakeWordTrainer.create_model` (`trainer.py:155`–`156`) can auto-detect it:

```python
if feature_dim is None:
    feature_dim = extractor.feature_dim
```

This allows the tier presets to work without specifying the dimension explicitly.

### Separate `.pt` and `.ts` checkpoint files

`save_checkpoint` (`checkpoint.py:10`) writes two files:
- `name.pt` — model weights only (via `model.save_checkpoint`).
- `name.ts` — trainer state: epoch, metrics dict, optimizer state.

This separation means you can load just the model weights without carrying a large optimizer state, and the trainer state can be inspected or modified independently.

### Adaptive hard-negative sampling

The training loop does not use a fixed hard-negative ratio. Instead, it computes a `readiness` score from embedding statistics via `compute_readiness()` (`evaluation.py`) and uses an exponential moving average of that score to modulate how many hard vs easy vs random negatives are sampled each epoch via `_build_epoch_data()` (`loop.py`). This prevents gradient collapse from overexposure to hard examples early in training.

### Feature vectorization cache

`FeatureCache` (`cache.py`) stores un-augmented waveforms as `.npy` files keyed by MD5(file content + extractor class name + feature_dim + sample_rate). This avoids redundant I/O and resampling for test/validation sets across epochs. The cache is bypassed when augmentation is active for a given sample, ensuring augmented variants are always freshly computed. Integrated into `AudioDataset.__getitem__` (`dataset.py`).

### Layer freezing for transfer learning

`WakeWordTrainer._freeze()` / `_unfreeze()` (`trainer.py`) set `requires_grad=False` on the feature extractor and/or first N classifier parameters. Progressive unfreezing (`unfreeze_at_epoch`) recreates the optimizer at the specified epoch to include newly-thawed parameters.

### Epoch-level data replacement

After assembling epoch_data from wake + hard/easy/random negatives, `replacement_ratio` fraction is dropped and replaced with random samples from the full pool. Implemented in `_build_epoch_data()` — `loop.py`. Optional balanced mode ensures 50/50 pos/neg in the replaced portion. Complements hard-negative mining by introducing additional diversity.

### Composite fitness score

`compute_fitness_score()` (`evaluation.py`) produces a single training quality metric: `(1 - 0.8*FP_rate - 0.2*FN_rate) * size_penalty`. FP is penalized 4× more than FN. `size_penalty = max(0, 1 - 0.1 * max(0, params/budget - 1))`. When `--fitness-checkpoint` is enabled, `best_fitness.pt` is saved alongside other metric checkpoints.
