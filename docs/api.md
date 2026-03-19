# ww-trainer — API Reference

Complete reference for every public class and method. All line numbers reference the current source.

---

## `ww_trainer.feats`

Source: `ww_trainer/feats.py`

### Type alias

```python
WavInput: TypeAlias = Union[torch.Tensor, List[torch.Tensor]]
```

Defined at `feats.py:11`. Accepted by all extractor `forward` methods.

---

### `ensure_wav_list` — `feats.py:14`

```python
def ensure_wav_list(wavs: WavInput) -> List[torch.Tensor]
```

Normalizes the `wavs` argument to a list of 1-D tensors.

| Input | Behaviour |
|-------|-----------|
| 1-D `Tensor[T]` | Returns `[wavs]` |
| 2-D `Tensor[B, T]` | Returns `[wavs[0], wavs[1], ...]` |
| `list` of tensors | Returns as-is |

Raises `ValueError` for unsupported tensor shapes, `TypeError` for non-tensor inputs.

---

### `SlidingFeatureCacheTensor` — `feats.py:27`

```python
class SlidingFeatureCacheTensor(torch.nn.Module)
```

Rolling buffer that accumulates the most recent `window_size` feature frames for streaming inference.

#### `__init__` — `feats.py:28`

```python
def __init__(self, feature_dim: int = 768, window_size: int = 50)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature_dim` | `int` | `768` | Feature dimension F of each frame |
| `window_size` | `int` | `50` | Maximum number of frames to retain |

Registers `feature_cache: Tensor[window_size, feature_dim]` as a buffer (zero-initialized).

#### `forward` — `feats.py:36`

```python
def forward(self, new_feats: Tensor) -> Tensor
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `new_feats` | `[T_new, F]` | Newly extracted feature frames |

Returns `Tensor[T_current, F]` — the current window (up to `window_size` frames). Updates the cache in-place. When the buffer overflows, old frames are discarded from the front.

---

### `BaseExtractor` — `feats.py:57`

```python
class BaseExtractor(torch.nn.Module)
```

Abstract base for all feature extractors.

#### `__init__` — `feats.py:59`

```python
def __init__(self, sample_rate: int = 16000, device: str = "auto") -> None
```

`device="auto"` selects CUDA if available, otherwise CPU (`feats.py:61`–`62`).

#### `feature_dim` property — `feats.py:66`

```python
@property
def feature_dim(self) -> int
```

Abstract. Subclasses must return the integer feature dimension F. Raises `NotImplementedError` in the base class.

#### `forward` — `feats.py:71`

```python
@abc.abstractmethod
def forward(self, wavs: WavInput, **kwargs) -> torch.Tensor
```

Abstract. Must return `Tensor[B, T_frames, F]`.

#### `export_to_onnx` — `feats.py:77`

```python
def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False) -> None
```

Exports the extractor to ONNX.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `out` | `str` | required | Output file path (e.g. `"extractor.onnx"`) |
| `quantize` | `bool` | `False` | If True, also writes INT16 and INT8 quantized variants |
| `dynamo` | `bool` | `False` | Use TorchDynamo exporter instead of TorchScript |

Dynamic axes: batch size and time dimension. Opset 18. Verifies the exported model with `onnx.checker`. If `quantize=True`, writes `<stem>_int16.onnx` and `<stem>_int8.onnx` (`feats.py:97`–`102`).

---

### `OnnxFeatureExtractor` — `feats.py:105`

```python
class OnnxFeatureExtractor(BaseExtractor)
```

Wraps a pre-exported extractor ONNX file as a `BaseExtractor`. Used for training when the extractor has already been exported (e.g. HuBERT exported once, reused across many training runs).

#### `__init__` — `feats.py:106`

```python
def __init__(self, model_path: str, sample_rate: int = 16000, device: str = "auto")
```

Loads `model_path` into an `onnxruntime.InferenceSession`. Selects `CUDAExecutionProvider` then `CPUExecutionProvider` when device is CUDA (`feats.py:110`–`118`).

#### `feature_dim` property — `feats.py:125`

Reads from the ONNX output shape. If the last dimension is dynamic (None), runs a dummy `[1, sample_rate]` inference to determine the dimension at runtime (`feats.py:130`–`133`).

#### `forward` — `feats.py:136`

```python
def forward(self, wavs: WavInput) -> torch.Tensor
```

Processes each waveform individually (batch size 1 per ONNX run), pads results to equal length, stacks. Returns `Tensor[B, T_frames, F]`.

---

### `MfccExtractor` — `feats.py:159`

```python
class MfccExtractor(BaseExtractor)
```

Pure-PyTorch MFCC extractor. Fully ONNX-exportable. Uses `return_complex=False` in `torch.stft` to remain compatible with the ONNX exporter (`feats.py:228`).

A pre-exported version is available at https://huggingface.co/TigreGotico/mfcc-onnx.

#### `__init__` — `feats.py:165`

```python
def __init__(
    self,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_mels: int = 40,
    n_fft: int = 400,
    hop_length: int = 160,
    f_min: float = 0.0,
    f_max: float = None,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sr` | `16000` | Audio sample rate |
| `n_mfcc` | `40` | Number of MFCC coefficients (= `feature_dim`) |
| `n_mels` | `40` | Number of mel filterbank bins |
| `n_fft` | `400` | FFT window size in samples |
| `hop_length` | `160` | STFT hop size in samples |
| `f_min` | `0.0` | Minimum frequency for mel filterbank |
| `f_max` | `None` | Maximum frequency (defaults to `sr/2`) |

Registers `mel_fb` and `dct_mat` as buffers so they move with the model's device.

#### `feature_dim` property — `feats.py:181`

Returns `self.n_mfcc`.

#### `forward` — `feats.py:213`

```python
def forward(self, wavs: WavInput) -> torch.Tensor
```

Computes MFCC: STFT → power spectrum → mel filterbank → log → DCT. Returns `Tensor[B, T, n_mfcc]`.

#### `_mel_filterbank` — `feats.py:184`

Builds the `[n_mels, n_fft//2+1]` triangular mel filterbank matrix using Hz-to-mel and mel-to-Hz conversions.

#### `_dct_matrix` — `feats.py:206`

Builds the orthonormal `[n_mfcc, n_mels]` DCT-II matrix.

---

### `HubertExtractor` — `feats.py:245`

```python
class HubertExtractor(BaseExtractor)
```

HuBERT encoder wrapper for training. Requires `transformers`. Sets `_REQUIRES_TRANSFORMERS = True` (`feats.py:249`).

#### `__init__` — `feats.py:251`

```python
def __init__(
    self,
    model_name: str = "voidful/hubert-tiny-v2",
    sample_rate: int = 16000,
    device: str = "auto",
) -> None
```

Loads `HubertModel.from_pretrained(model_name)` and puts it in eval mode. Raises `ImportError` if `transformers` is not installed (`feats.py:255`–`259`).

#### `feature_dim` property — `feats.py:263`

Returns `self.hubert.config.hidden_size`.

#### `forward` — `feats.py:267`

```python
def forward(self, wavs: WavInput) -> torch.Tensor
```

Pads to equal length, normalizes by peak amplitude (`feats.py:272`–`273`), runs `hubert(wav_tensor).last_hidden_state` under `torch.no_grad()`. Returns `Tensor[B, T, hidden_size]`.

---

### `Wav2Vec2Extractor` — `feats.py:279`

```python
class Wav2Vec2Extractor(BaseExtractor)
```

Wav2Vec2 encoder wrapper for training. Requires `transformers`. Sets `_REQUIRES_TRANSFORMERS = True` (`feats.py:283`).

#### `__init__` — `feats.py:285`

```python
def __init__(
    self,
    model_name: str = "patrickvonplaten/tiny-wav2vec2-no-tokenizer",
    sample_rate: int = 16000,
    device: str = "auto",
) -> None
```

#### `feature_dim` property — `feats.py:297`

Returns `self.model.config.hidden_size`.

#### `forward` — `feats.py:301`

```python
def forward(self, wavs: torch.Tensor) -> torch.Tensor
```

Mean/std normalizes the input (`feats.py:307`), runs `model(wavs).last_hidden_state` under `torch.no_grad()`. Returns `Tensor[B, T', hidden_size]`.

---

## `ww_trainer.model`

Source: `ww_trainer/model.py`

---

### `ClassifierHead` — `model.py:13`

```python
class ClassifierHead(torch.nn.Module)
```

Abstract base for all classifier heads.

#### `__init__` — `model.py:14`

```python
def __init__(self, input_size: int, sample_rate: int = 16000, device: str = "auto") -> None
```

| Parameter | Description |
|-----------|-------------|
| `input_size` | Feature dimension F from the extractor — must match `extractor.feature_dim` |
| `sample_rate` | Audio sample rate (informational) |
| `device` | `"auto"`, `"cuda"`, or `"cpu"` |

#### `forward` — `model.py:23`

```python
@abc.abstractmethod
def forward(self, feats: torch.Tensor) -> torch.Tensor
```

Abstract. Must return scalar logits `Tensor[B]`.

#### `embed` — `model.py:27`

```python
@abc.abstractmethod
def embed(self, feats: torch.Tensor) -> torch.Tensor
```

Abstract. Must return pre-logit embeddings `Tensor[B, D]` for metric learning.

#### `export_to_onnx` — `model.py:30`

```python
def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False) -> None
```

Exports the head to ONNX with a dummy `[1, 200, input_size]` input. Dynamic axis on the time dimension. Input name `input_features`, output name `logits`. If `quantize=True`, writes `<stem>_int8.onnx` (`model.py:57`–`62`).

---

### `FfnClassifierHead` — `model.py:150`

```python
class FfnClassifierHead(ClassifierHead)
```

Mean-pool over the time dimension, then a two-layer feed-forward network.

#### `__init__` — `model.py:153`

```python
def __init__(
    self,
    sample_rate: int = 16000,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    device: str = "auto",
    input_size: int = None,
) -> None
```

Architecture: `Linear(input_size → hidden_dim) → ReLU → Dropout(dropout) → Linear(hidden_dim → 1)`.

#### `forward` — `model.py:166`

```python
def forward(self, feats: torch.Tensor) -> torch.Tensor
```

`pooled = feats.mean(dim=1)` then sequential forward. Returns `Tensor[B]`.

#### `embed` — `model.py:170`

Returns `F.relu(linear_0(pooled))` — the hidden layer activation, shape `[B, hidden_dim]`.

---

### `CnnClassifierHead` — `model.py:175`

```python
class CnnClassifierHead(ClassifierHead)
```

Two Conv1d layers over the feature dimension, adaptive average pooling, then two FC layers.

**Note:** Conv1d operates on `[B, C, T]` so the head transposes features internally — feats arrive as `[B, T, F]` and `self.conv` receives `[B, F, T]` (`model.py:196`: `self.conv(feats)` — the `Conv1d` in `self.conv` has `in_channels=self.input_size`, which matches `F`).

#### `__init__` — `model.py:177`

```python
def __init__(
    self,
    sample_rate: int = 16000,
    conv_dim: int = 256,
    linear_dim: int = 128,
    kernel_size: int = 3,
    stride: int = 1,
    device: str = "auto",
    input_size: int = None,
) -> None
```

Architecture: `Conv1d(input_size, conv_dim, kernel_size) → ReLU → Conv1d(conv_dim, conv_dim, kernel_size) → ReLU → AdaptiveAvgPool1d(1) → FC(conv_dim, linear_dim) → FC(linear_dim, 1)`.

#### `forward` — `model.py:195`

Returns `Tensor[B]`.

#### `embed` — `model.py:200`

Returns `F.relu(fc1(conv_out))` — shape `[B, linear_dim]`.

---

### `GruClassifierHead` — `model.py:205`

```python
class GruClassifierHead(ClassifierHead)
```

GRU recurrent network over the time dimension, mean-pooled, then two FC layers.

#### `__init__` — `model.py:207`

```python
def __init__(
    self,
    hidden_dim: int = 128,
    linear_dim: int = 128,
    dropout: float = 0.0,
    bidirectional: bool = False,
    gru_n_layers: int = 1,
    sample_rate: int = 16000,
    device: str = "auto",
    input_size: int = None,
) -> None
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | `128` | GRU hidden size |
| `linear_dim` | `128` | FC hidden size |
| `dropout` | `0.0` | GRU inter-layer dropout (only applied when `gru_n_layers > 1`) |
| `bidirectional` | `False` | Whether to use a bidirectional GRU |
| `gru_n_layers` | `1` | Number of stacked GRU layers |

FC size after GRU: `hidden_dim * 2` if bidirectional, else `hidden_dim`.

#### `_ensure_correct_shape` — `model.py:226`

```python
def _ensure_correct_shape(self, feats: torch.Tensor) -> torch.Tensor
```

Auto-detects `[B, F, T]` vs `[B, T, F]` input orientation by comparing `D1` and `D2` to `input_size`. Transposes if `D1 == input_size and D2 != input_size`. Raises `ValueError` when both dimensions equal `input_size` (ambiguous).

#### `forward` — `model.py:245`

```python
def forward(self, feats: torch.Tensor) -> torch.Tensor
```

Calls `_ensure_correct_shape`, runs GRU, mean-pools GRU outputs, then two FC layers. Returns `Tensor[B]`.

#### `embed` — `model.py:252`

Returns `F.relu(fc1(pooled))` — shape `[B, linear_dim]`.

---

### `BaseWakeModel` — `model.py:65`

```python
class BaseWakeModel(nn.Module)
```

Combines an extractor and a head into a single trainable module.

#### `__init__` — `model.py:71`

```python
def __init__(
    self,
    feature_extractor: BaseExtractor,
    classifier: ClassifierHead,
    sample_rate: int = 16000,
    device: str = "auto",
) -> None
```

Moves both submodules to `device` via `self.to(self.device)` (`model.py:83`).

#### `forward` — `model.py:86`

```python
def forward(self, wavs: WavInput) -> torch.Tensor
```

Runs `feature_extractor(wavs)` then `classifier.forward(feats)`. Returns logits `Tensor[B]`.

#### `embed` — `model.py:91`

```python
def embed(self, wavs: WavInput) -> torch.Tensor
```

Returns `classifier.embed(feature_extractor(wavs))`. Used by metric losses and visualization.

#### `forward_streaming` — `model.py:110`

```python
def forward_streaming(
    self,
    audio_chunk: torch.Tensor,
    cache: SlidingFeatureCacheTensor,
) -> float
```

Processes one audio chunk with a rolling feature cache.

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio_chunk` | `Tensor[T]` | One chunk of raw audio at `self.sample_rate` |
| `cache` | `SlidingFeatureCacheTensor` | Updated in-place each call |

Returns sigmoid probability as a Python `float`.

#### `infer` — `model.py:127`

```python
def infer(self, audio: np.ndarray) -> float
```

Single-waveform inference. Wraps the audio in a tensor, calls `forward`, applies sigmoid. Returns `float` in `[0, 1]`. Raises `ValueError` if `audio.ndim != 1`.

#### `load_checkpoint` — `model.py:97`

```python
def load_checkpoint(self, ckpt_path: str) -> None
```

Loads a `.pt` file into the model with `strict=False`. Falls back to `state["model"]` key if a full dict is detected.

#### `save_checkpoint` — `model.py:107`

```python
def save_checkpoint(self, ckpt_path: str) -> None
```

Saves `self.state_dict()` via `torch.save`.

#### `export_to_onnx` — `model.py:136`

```python
def export_to_onnx(
    self,
    out: str,
    simplify: bool = False,
    quantize: bool = False,
    export_featurizer: bool = False,
) -> None
```

Exports the classifier head to `out`. If `export_featurizer=True`, also exports the extractor to `out.replace(".onnx", "") + "_featurizer.onnx"` (`model.py:143`).

---

## `ww_trainer.inference`

Source: `ww_trainer/inference.py`

No PyTorch imports at module level. Requires only `numpy` and `onnxruntime`.

---

### `OnnxWakeWordInferencer` — `inference.py:8`

```python
class OnnxWakeWordInferencer
```

Chains two ONNX sessions (extractor + head) for wake word detection.

#### `__init__` — `inference.py:20`

```python
def __init__(
    self,
    extractor_path: str,
    head_path: str,
    sample_rate: int = 16000,
    device: str = "auto",
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extractor_path` | `str` | required | Path to extractor ONNX — input `[B, T]`, output `[B, T_frames, F]` |
| `head_path` | `str` | required | Path to head ONNX — input `[B, T_frames, F]`, output scalar logit |
| `sample_rate` | `int` | `16000` | Expected audio sample rate |
| `device` | `str` | `"auto"` | `"cpu"`, `"cuda"`, or `"auto"` |

`device="auto"` checks `ort.get_all_providers()` for `CUDAExecutionProvider` (`inference.py:23`–`24`).

#### `infer` — `inference.py:36`

```python
def infer(self, audio: np.ndarray) -> float
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `audio` | `[T]` float32 | Single waveform at `sample_rate` |

Returns sigmoid probability in `[0, 1]`. Raises `ValueError` if `audio.ndim != 1`.

**Example:**

```python
from ww_trainer.inference import OnnxWakeWordInferencer
import numpy as np

inf = OnnxWakeWordInferencer("mfcc.onnx", "head.onnx")
audio = np.zeros(16000, dtype=np.float32)
prob = inf.infer(audio)
print(f"Wake probability: {prob:.4f}")
```

#### `infer_batch` — `inference.py:57`

```python
def infer_batch(self, audio_batch: np.ndarray) -> np.ndarray
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `audio_batch` | `[B, T]` float32 | Batch of equal-length waveforms |

Returns float32 array `[B]` of sigmoid probabilities. Raises `ValueError` if `audio_batch.ndim != 2`.

**Example:**

```python
batch = np.zeros((8, 16000), dtype=np.float32)
probs = inf.infer_batch(batch)  # shape [8]
```

#### `infer_streaming` — `inference.py:74`

```python
def infer_streaming(
    self,
    audio_chunk: np.ndarray,
    cache: np.ndarray | None,
) -> tuple[float, np.ndarray]
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `audio_chunk` | `[chunk_T]` float32 | One audio chunk |
| `cache` | `[T_cached, F]` or `None` | Feature cache from previous call; `None` on first call |

Returns `(probability: float, updated_cache: np.ndarray)`. Internally keeps the last 50 frames (`inference.py:96`–`98`).

**Example:**

```python
cache = None
chunks = [np.zeros(4000, dtype=np.float32)] * 10  # simulate mic chunks
for chunk in chunks:
    prob, cache = inf.infer_streaming(chunk, cache)
    if prob > 0.5:
        print("Wake word detected!")
```

---

## `ww_trainer.trainer`

Source: `ww_trainer/trainer.py`

---

### `EXTRACTOR_REGISTRY` — `trainer.py:24`

```python
EXTRACTOR_REGISTRY = {
    "onnx": OnnxFeatureExtractor,
    "mfcc": MfccExtractor,
}
```

Maps string extractor type names to classes. Used by `create_model` for the two directly importable extractors. `hubert` and `wav2vec2` are handled inline via guarded imports.

---

### `WakeWordTrainer` — `trainer.py:40`

```python
class WakeWordTrainer
```

Full training orchestrator: model construction, loss setup, epoch loop, hard-negative mining, evaluation, checkpointing, and MLflow logging.

#### `__init__` — `trainer.py:57`

```python
def __init__(
    self,
    arch: str,
    featurizer: str,
    feature_dim: int = None,
    device: str = "auto",
    mlflow_uri: str = None,
    resume: str = None,
    export_onnx: bool = False,
    losses_cfg: List[Dict[str, str]] = None,
    featurizer_type: str = "onnx",
    shared_extractor = None,
    use_amp: bool = False,
    **model_kwargs,
) -> None
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `arch` | required | Head architecture: `"ffn"`, `"gru"`, `"cnn"` |
| `featurizer` | required | Path to extractor ONNX (or `""` for `mfcc`/`hubert`/`wav2vec2`) |
| `feature_dim` | `None` | Feature dimension F; auto-detected from extractor if None |
| `device` | `"auto"` | `"cpu"`, `"cuda"`, or `"auto"` |
| `mlflow_uri` | `None` | MLflow tracking URI; disables MLflow if None |
| `resume` | `None` | Path to `.pt` checkpoint to resume from |
| `export_onnx` | `False` | Export to ONNX after every checkpoint save |
| `losses_cfg` | `None` | List of loss config dicts; see `LossManager` |
| `featurizer_type` | `"onnx"` | One of `"onnx"`, `"mfcc"`, `"hubert"`, `"wav2vec2"` |
| `shared_extractor` | `None` | Pre-built extractor to share across trainers |
| `use_amp` | `False` | Enable mixed-precision training |
| `**model_kwargs` | | Passed to `create_model` (arch-specific params) and augmentation |

When `mlflow_uri` is provided, starts an MLflow run named `<featurizer_slug>-<arch>` and logs all params (`trainer.py:99`–`116`).

#### `create_model` — `trainer.py:128`

```python
@staticmethod
def create_model(
    arch_name: str,
    featurizer: str,
    feature_dim: int = None,
    featurizer_type: str = "onnx",
    sample_rate: int = 16000,
    device: str = "auto",
    shared_extractor = None,
    **kwargs,
) -> BaseWakeModel
```

Builds and returns a `BaseWakeModel`. Extractor selection:

| `featurizer_type` | Extractor class |
|-------------------|----------------|
| `"onnx"` | `OnnxFeatureExtractor(featurizer, ...)` |
| `"mfcc"` | `MfccExtractor(sr=sample_rate)` |
| `"hubert"` | `HubertExtractor(featurizer, ...)` |
| `"wav2vec2"` | `Wav2Vec2Extractor(featurizer, ...)` |

Architecture-specific kwargs forwarded to the head:

| `arch_name` | Head class | Accepted kwargs |
|-------------|-----------|-----------------|
| `"ffn"` | `FfnClassifierHead` | `hidden_dim`, `dropout` |
| `"gru"` | `GruClassifierHead` | `hidden_dim`, `dropout`, `bidirectional`, `gru_n_layers` |
| `"cnn"` | `CnnClassifierHead` | `conv_dim`, `linear_dim`, `kernel_size`, `stride` |

#### `save_checkpoint` — `trainer.py:119`

```python
def save_checkpoint(
    self,
    epoch: int,
    metrics: dict,
    optimizer: torch.optim.Optimizer,
    out_path: Path | str,
) -> None
```

Delegates to `ww_trainer.checkpoint.save_checkpoint`.

#### `load_checkpoint` — `trainer.py:123`

```python
def load_checkpoint(
    self,
    path: Path | str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, dict]
```

Delegates to `ww_trainer.checkpoint.load_checkpoint`. Returns `(start_epoch, metrics)`.

#### `train` — `trainer.py:307`

```python
def train(
    self,
    output_dir: str | Path,
    train_data: List[Tuple[str, str]],
    test_data: List[Tuple[str, str]],
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    neg_threshold: float = 0.5,
    mine_fraction: float = 0.2,
    mining_type: str = "semihard",
    patience: int = 2,
    save_best: bool = True,
    metrics_log: str = "metrics_log.csv",
    pca_every: int = 0,
    tsne_every: int = 0,
    umap_every: int = 0,
    blend_ratio: float = 0.7,
    base_hard: float = 0.5,
    max_hard: float = 5.0,
    base_easy: float = 1.5,
    min_easy: float = 0.2,
    base_random: float = 0.1,
    total_ratio: float = 5,
    use_amp: bool = False,
    accumulate_grad_batches: int = 1,
    resume: str = None,
) -> None
```

Main training loop. Each epoch:
1. Computes embedding readiness score for adaptive negative sampling.
2. Samples hard, easy, and random negatives.
3. Trains with Adam optimizer and cosine annealing LR schedule.
4. Evaluates accuracy, precision, recall, F1, AUC on test set.
5. Saves best-metric checkpoints (when `save_best=True`): `best_loss.pt`, `best_precision.pt`, `best_recall.pt`, `best_f1.pt`.
6. Mines hard negatives from the nonwake pool.
7. Triggers early stopping when no new hard negatives are found for `patience` epochs.

Final model saved as `output_dir/final_model.pt`. Mining cache saved as `output_dir/hardneg_cache.pt`.

---

### CLI: `train` command — `trainer.py:668`

Entry point: `ww_trainer-train` (defined in `pyproject.toml`, implemented in `cli.py`).

See [training.md](training.md) for the full option table.

---

## `ww_trainer.tiers`

Source: `ww_trainer/tiers.py`

---

### `TierConfig` — `tiers.py:12`

```python
@dataclass
class TierConfig
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Tier identifier (e.g. `"micro"`) |
| `extractor_type` | `str` | required | One of `"mfcc"`, `"onnx"`, `"hubert"`, `"wav2vec2"` |
| `head_arch` | `str` | required | One of `"ffn"`, `"gru"`, `"cnn"` |
| `hidden_dim` | `int` | required | Head hidden dimension |
| `n_mfcc` | `int` | `40` | MFCC coefficients (only used when `extractor_type == "mfcc"`) |
| `bidirectional` | `bool` | `False` | GRU bidirectional flag (only used when `head_arch == "gru"`) |
| `gru_n_layers` | `int` | `1` | Number of GRU layers |
| `description` | `str` | `""` | Human-readable description |
| `approx_params` | `str` | `""` | Approximate parameter count string |
| `target_hardware` | `str` | `""` | Target deployment hardware |

---

### `HARDWARE_TIERS` — `tiers.py:26`

```python
HARDWARE_TIERS: dict[str, TierConfig]
```

Keys: `"micro"`, `"small"`, `"medium"`, `"large"`. See [architecture.md](architecture.md) for the full tier table.

---

### `get_tier` — `tiers.py:83`

```python
def get_tier(name: str) -> TierConfig
```

Returns `HARDWARE_TIERS[name]`. Raises `ValueError` with available names if `name` is unknown.

---

### `list_tiers` — `tiers.py:70`

```python
def list_tiers() -> str
```

Returns a formatted multi-line string table of all tiers. Used by `--list-tiers` CLI flag.

---

## `ww_trainer.dataset`

Source: `ww_trainer/dataset.py`

---

### `AudioDataset` — `dataset.py:99`

```python
class AudioDataset(Dataset)
```

PyTorch `Dataset` for loading audio files with optional on-the-fly augmentation.

#### `__init__` — `dataset.py:104`

```python
def __init__(
    self,
    samples: List[Tuple[str, str]],
    sample_rate: int = 16000,
    aug_prob: float = 0.0,
    vc_prob: float = 0.3,
    bg_noise_folder: str = None,
    music_folder: str = None,
    bg_speech_folder: str = None,
    mic_noise_folder: str = None,
    rir_folder: str = None,
    vc_folder: str = None,
    snr_min: float = 0.0,
    snr_max: float = 20.0,
    pitch_min: float = -1.0,
    pitch_max: float = 1.0,
    speed_min: float = 0.95,
    speed_max: float = 1.05,
    device: str = "auto",
)
```

`samples` is a list of `(path, label)` tuples where `label` is `"1"` (wake) or `"0"` (non-wake).

On construction: validates files (warns on missing), logs label distribution (`dataset.py:154`–`167`).

When `vc_folder` and `vc_prob > 0`: loads `chatterbox_onnx.ChatterboxOnnx` (`dataset.py:145`–`148`), raising `ImportError` if not installed.

#### `__len__` — `dataset.py:169`

Returns `len(self.samples)`.

#### `__getitem__` — `dataset.py:269`

Returns `(wav: Tensor[T], label: int, path: str)`.

Load order: optionally voice-converts wake samples (`label=="1"`, prob `vc_prob`), then loads with `torchaudio.load`, resamples if needed, applies augmentation with probability `aug_prob`.

---

### `collate_fn` — `dataset.py:301`

```python
def collate_fn(batch, device="auto") -> Tuple[Tensor, Tensor, Tuple[str, ...]]
```

Pads all waveforms in a batch to the same length (zero-padding), stacks into `Tensor[B, T]`, moves to `device`. Returns `(wavs, labels, paths)`.

---

## `ww_trainer.loss`

Source: `ww_trainer/loss.py`

---

### `CN2Plus1PairLoss` — `loss.py:13`

(C_N,2 + 1)-pair loss from López-Espejo et al., TASLP 2021.

```python
class CN2Plus1PairLoss(nn.Module)
def __init__(self, d_max: float = 2.0, margin: float = 0.0)
def forward(self, anchor: Tensor, positive: Tensor, negatives: Tensor) -> Tensor
```

Inputs: `anchor [B, D]`, `positive [B, D]`, `negatives [N_neg, D]`. Encourages anchor-positive distances to be small relative to anchor-negative and negative-negative distances. Uses softplus.

---

### `SoftTripletLoss` — `loss.py:86`

Soft triplet loss: `log(1 + exp(D(a,p) - D(a,n)))`.

```python
class SoftTripletLoss(nn.Module)
def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor
```

L2-normalizes inputs, computes squared L2 distances, applies softplus. Gradient flows even when the hard margin constraint is satisfied.

---

### `ContrastiveLoss` — `loss.py:126`

Classic contrastive (Siamese) loss.

```python
class ContrastiveLoss(nn.Module)
def __init__(self, margin: float = 1.0)
def forward(self, embeds: Tensor, labels: Tensor) -> Tensor
```

Pulls positive pairs together (squared distance), pushes negative pairs beyond `margin` with `max(0, margin - D)^2`.

---

### `LiftedStructureLoss` — `loss.py:170`

Lifted Structured Embedding loss. Uses all positive and negative pairs via log-sum-exp.

```python
class LiftedStructureLoss(nn.Module)
def __init__(self, margin: float = 1.0)
def forward(self, embeds: Tensor, labels: Tensor) -> Tensor
```

---

### `AngularLoss` — `loss.py:221`

Cosine-based margin loss. Enforces `cos(A,P) > cos(A,N) + margin`.

```python
class AngularLoss(nn.Module)
def __init__(self, margin: float = 0.5)
def forward(self, embeds: Tensor, labels: Tensor) -> Tensor
```

Mines semi-hard triplets using Euclidean distance, then applies the angular (cosine) condition.

---

### `RobustProtoDiversityLoss` (RPPL) — `loss.py:281`

Composite loss: BCE + prototype contrastive + positive center loss + negative diversity + consistency.

```python
class RobustProtoDiversityLoss(nn.Module)
def __init__(
    self,
    tau: float = 0.1,
    K_neg_proto: int = 0,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    delta: float = 0.1,
    eta: float = 0.5,
)
def forward(
    self,
    logits: Tensor,
    labels: Tensor,
    embeds: Tensor,
    aug_embeds: Tensor = None,
) -> Tensor
```

| Parameter | Description |
|-----------|-------------|
| `tau` | Temperature for prototype contrastive loss |
| `K_neg_proto` | Number of negative prototypes (0/1 = single mean prototype) |
| `alpha` | BCE weight |
| `beta` | Prototype contrastive weight |
| `gamma` | Negative diversity weight |
| `delta` | Positive center loss weight |
| `eta` | Consistency loss weight (requires `aug_embeds`) |

When `aug_embeds` is provided, adds a consistency term `||z - z_aug||^2` (`loss.py:382`–`383`).

---

### `LossManager` — `loss.py:395`

```python
class LossManager
def __init__(
    self,
    loss_configs: List[Dict[str, Any]],
    mining_type: str = "semihard",
    device: str = "cpu",
) -> None
```

Manages multiple weighted loss functions. `loss_configs` is a list of dicts:

```python
[
    {"name": "bce", "weight": 1.0},
    {"name": "triplet", "weight": 0.5, "margin": 1.0},
]
```

Supported names: `"bce"`, `"triplet"`, `"soft_triplet"`, `"pair"`, `"cn2pair"`, `"lse"`, `"contrastive"`, `"angular"`, `"rppl"`.

#### `compute_loss` — `loss.py:454`

```python
def compute_loss(
    self,
    model: nn.Module,
    wavs: torch.Tensor,
    labels: torch.Tensor,
    dataset_ref: AudioDataset = None,
) -> Tuple[torch.Tensor, Dict[str, float]]
```

Calls `model(wavs)` for logits and `model.embed(wavs)` for embeddings, then computes each configured loss. Returns `(total_loss, {"loss_name": value, "total": value})`.

`dataset_ref` is only used by `rppl` for consistency loss — it calls `dataset_ref.get_augmented(w)` to produce augmented embeddings.

---

## `ww_trainer.utils`

Source: `ww_trainer/utils.py`

---

### `timed` — `utils.py:10`

```python
def timed(func)
```

Decorator that logs function execution time in milliseconds when the `DEBUG` environment variable is set to `1`, `true`, `yes`, or `on`. No-op otherwise.

```python
@timed
def my_function():
    ...
```

---

### `compute_metric_statistics` — `utils.py:31`

```python
def compute_metric_statistics(
    labels: torch.Tensor,
    embeds: torch.Tensor,
    margin: float = 1.0,
) -> Tuple[float, float, float]
```

Returns `(mean_ap_dist, mean_an_dist, violation_fraction)` — mean anchor-positive distance, mean anchor-negative distance, and fraction of valid triplets that violate the margin.

---

### `sample_triplets` — `utils.py:77`

```python
def sample_triplets(
    labels: torch.Tensor,
    embeds: torch.Tensor,
    margin: float,
    mining_type: str = "semihard",
    max_triplets: int = 100,
) -> Tuple[Tensor, Tensor, Tensor, float]
```

Returns `(anchor_indices, positive_indices, negative_indices, violation_fraction)` or `(None, None, None, frac)` if no valid triplets found.

Mining strategies:
- `"semihard"`: hardest positive + semi-hard negative (beyond hardest positive; falls back to hard if none exist).
- `"hard"`: hardest positive + hardest negative (closest).
- `"random"`: hardest positive + random negative.

---

### `pairwise_distance` — `utils.py:179`

```python
def pairwise_distance(embeddings: torch.Tensor, eps: float = 1e-12) -> torch.Tensor
```

Computes the `[B, B]` pairwise Euclidean distance matrix. Uses the dot-product formulation for numerical stability; clamps to zero before sqrt.

---

### `pairwise_cosine_similarity` — `utils.py:187`

```python
def pairwise_cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor
```

Returns `[B, B]` cosine similarity matrix. Assumes embeddings are already L2-normalized.

---

### `triplet_violation_fraction` — `utils.py:199`

```python
def triplet_violation_fraction(
    labels: torch.Tensor,
    embeds: torch.Tensor,
    margin: float,
) -> Tuple[int, int, float]
```

Returns `(n_violating, n_valid, violation_fraction)`.

---

### `get_hard_pair_distances` — `utils.py:215`

```python
def get_hard_pair_distances(
    labels: torch.Tensor,
    embeddings: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]
```

Returns `(dist_ap_hard, dist_an_hard)` — the hardest anchor-positive distance (farthest positive) and hardest anchor-negative distance (closest negative) for each sample.

---

### `sample_semihard_triplets` — `utils.py:239`

```python
def sample_semihard_triplets(
    labels: torch.Tensor,
    embeds: torch.Tensor,
    margin: float = 1.0,
    max_triplets: int = 1024,
)
```

Efficient semi-hard mining. Returns `(a_idx, p_idx, n_idx, violation_fraction)`. L2-normalizes embeddings before distance computation.

---

## `ww_trainer.visualization`

Source: `ww_trainer/visualization.py`

All functions are standalone (not methods). They accept `mlflow` as an optional last argument and log artifacts when provided.

---

### `plot_roc` — `visualization.py:27`

```python
def plot_roc(
    targets, probs, epoch: int, plot_dir: Path, auc: float, mlflow=None
) -> Optional[Path]
```

Plots ROC curve, saves to `plot_dir/roc_epoch_{epoch}.png`.

---

### `plot_pr` — `visualization.py:50`

```python
def plot_pr(
    targets, probs, epoch: int, plot_dir: Path, mlflow=None
) -> Optional[Path]
```

Plots Precision-Recall curve, saves to `plot_dir/pr_epoch_{epoch}.png`.

---

### `plot_det` — `visualization.py:73`

```python
def plot_det(
    targets, probs, epoch: int, plot_dir: Path, mlflow=None
) -> Optional[Path]
```

Plots DET curve (log-log axes), saves to `plot_dir/det_epoch_{epoch}.png`.

---

### `log_confidence_histogram` — `visualization.py:98`

```python
def log_confidence_histogram(
    targets: list,
    probs: list,
    epoch: int,
    outdir: Path,
    bins: int = 50,
    mlflow=None,
) -> Optional[str]
```

Plots two overlapping histograms — wake vs non-wake confidence distributions. Saves to `outdir/confidence_hist_epoch_{epoch}.png`.

---

### `log_pca` — `visualization.py:138`

```python
def log_pca(
    model, dataset, outdir: Path, epoch: int, device,
    sample_size: int = 200, mlflow=None,
) -> str
```

Runs PCA on embeddings of up to `sample_size` samples, plots 2-D scatter by class.

---

### `log_tsne` — `visualization.py:175`

```python
def log_tsne(
    model, dataset, outdir: Path, epoch: int, device,
    sample_size: int = 200, mlflow=None,
) -> Optional[str]
```

Runs t-SNE (perplexity=30, init="pca") on embeddings, plots 2-D scatter.

---

### `log_umap` — `visualization.py:218`

```python
def log_umap(
    model, dataset, outdir: Path, epoch: int, device,
    sample_size: int = 200, mlflow=None,
) -> Optional[str]
```

Runs UMAP if `umap-learn` is installed; falls back to t-SNE with a `logging.warning` (`visualization.py:234`–`241`).

---

### `log_embeddings_stats` — `visualization.py:265`

```python
def log_embeddings_stats(
    model, dataset, epoch: int, device, batch_size: int = 128, mlflow=None
) -> dict
```

Computes and returns a stats dict:

| Key | Description |
|-----|-------------|
| `embed_norm_mean` | Mean L2 norm of embeddings |
| `embed_norm_std` | Std of L2 norms |
| `embed_var_total` | Mean per-dimension variance across all samples |
| `intra_pos_var` | Mean per-dimension variance within wake samples |
| `intra_neg_var` | Mean per-dimension variance within non-wake samples |

Used by `WakeWordTrainer._compute_readiness` (`trainer.py:183`) to modulate hard-negative sampling.

---

## `ww_trainer.mining`

Source: `ww_trainer/mining.py`

---

### `mine_hard_negatives` — `mining.py:14`

```python
def mine_hard_negatives(
    model,
    nonwakes: List[Tuple[str, str]],
    device,
    hardness_cache: Dict[str, float] = None,
    neg_threshold: float = 0.5,
    dataset_fraction: float = 0.2,
    cache_decay: float = 0.9,
    max_cache_size: int = 10000,
    use_embedding_mining: bool = True,
    embed_top_k: int = 1000,
    wake_cache: List[Tuple[str, str]] = None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, float]]
```

Mines hard negatives from `nonwakes` using model confidence and optionally embedding similarity.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `neg_threshold` | `0.5` | Confidence above which a sample is considered hard |
| `dataset_fraction` | `0.2` | Fraction of nonwakes to evaluate each call |
| `cache_decay` | `0.9` | EMA decay for hardness scores across calls |
| `max_cache_size` | `10000` | Maximum entries in the rolling hardness cache |
| `use_embedding_mining` | `True` | Refine with cosine similarity to wake prototype |
| `embed_top_k` | `1000` | Keep top-K by embedding similarity |
| `wake_cache` | `None` | List of wake samples for prototype computation |

Returns `(hard_negatives, easy_negatives, updated_hardness_cache)`.

Hardness score: `new = cache_decay * old + (1 - cache_decay) * max(0, prob - neg_threshold)` (`mining.py:74`).

---

### `save_mining_cache` — `mining.py:126`

```python
def save_mining_cache(cache: dict, path: str) -> None
```

Saves `cache` dict via `torch.save`.

---

### `load_mining_cache` — `mining.py:132`

```python
def load_mining_cache(path: str) -> dict
```

Loads `cache` dict from `path`. Returns `{}` if the file does not exist.

---

## `ww_trainer.checkpoint`

Source: `ww_trainer/checkpoint.py`

---

### `save_checkpoint` — `checkpoint.py:10`

```python
def save_checkpoint(
    model,
    epoch: int,
    metrics: dict,
    optimizer: Optional[torch.optim.Optimizer],
    out_path: Path | str,
) -> None
```

Writes two files:
- `out_path.with_suffix(".pt")` — model weights via `model.save_checkpoint()`.
- `out_path.with_suffix(".ts")` — trainer state dict: `{"epoch", "metrics", "optimizer_state"}`.

Creates parent directories if needed.

---

### `load_checkpoint` — `checkpoint.py:42`

```python
def load_checkpoint(
    model,
    path: Path | str,
    device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, dict]
```

Loads model weights from `path`. If `path.with_suffix(".ts")` exists, restores epoch and metrics and optionally the optimizer state. Returns `(start_epoch, metrics)`.

---

## `ww_trainer.sweep`

Source: `ww_trainer/sweep.py`

---

### `run_sweep` — `sweep.py:20`

```python
def run_sweep(
    metadata_csv: str,
    n_trials: int = 20,
    output_dir: str = "sweep_results",
    featurizer: str = None,
    featurizer_type: str = "mfcc",
    sample_rate: int = 16000,
    device: str = "auto",
    epochs_per_trial: int = 5,
    study_name: str = "ww_trainer_sweep",
    storage: str = None,
) -> None
```

Runs an Optuna hyperparameter sweep. Requires `pip install optuna`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `metadata_csv` | required | Dataset CSV path (path,label per line) |
| `n_trials` | `20` | Number of trials |
| `output_dir` | `"sweep_results"` | Root directory for per-trial checkpoints |
| `featurizer` | `None` | ONNX extractor path (for `featurizer_type="onnx"`) |
| `featurizer_type` | `"mfcc"` | Extractor type |
| `epochs_per_trial` | `5` | Epochs per trial (keep short) |
| `study_name` | `"ww_trainer_sweep"` | Optuna study name |
| `storage` | `None` | Optuna storage URL; `None` = in-memory (not resumable) |

Search space: `arch` ∈ {ffn, gru, cnn}, `hidden_dim` ∈ {64, 128, 256}, `lr` log-uniform in `[1e-4, 1e-2]`, `batch_size` ∈ {16, 32, 64}, `dropout` ∈ {0.0, 0.1, 0.2, 0.3, 0.4}.

Saves best params to `output_dir/best_params.json`.

---

## `ww_trainer.cache`

Source: `ww_trainer/cache.py`

### `FeatureCache` — `cache.py:19`

```python
class FeatureCache(cache_dir: str, extractor_name: str, extractor_params_hash: str)
```

Disk-backed cache for extracted audio features. Stores `.npy` files keyed by MD5(file content + extractor identity).

| Method | Signature | Description |
|--------|-----------|-------------|
| `get` | `(audio_path: str) -> Optional[np.ndarray]` | Return cached features or `None` on miss |
| `put` | `(audio_path: str, features: np.ndarray) -> None` | Store features in cache |
| `clear` | `() -> int` | Remove all cached entries; returns count |
| `size` | `() -> int` | Number of cached `.npy` files |

### `make_extractor_params_hash` — `cache.py:115`

```python
def make_extractor_params_hash(extractor_name: str, feature_dim: int, sample_rate: int) -> str
```

Builds a deterministic hash string from extractor configuration. Used as the `extractor_params_hash` argument to `FeatureCache`.

---

## `ww_trainer.evaluation` — `compute_fitness_score`

### `compute_fitness_score` — `evaluation.py:118`

```python
def compute_fitness_score(
    f1: float, fp_rate: float, fn_rate: float,
    param_count: int, param_budget: int,
    fp_weight: float = 0.8, fn_weight: float = 0.2,
    size_weight: float = 0.1,
) -> float
```

Composite training fitness score. Penalizes FP 4× more than FN, with a model-size penalty for exceeding `param_budget`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fp_weight` | `0.8` | Weight for false positive rate penalty |
| `fn_weight` | `0.2` | Weight for false negative rate penalty |
| `size_weight` | `0.1` | Weight for size penalty |

Formula: `(1 - fp_weight * FP_rate - fn_weight * FN_rate) * max(0, 1 - size_weight * max(0, params/budget - 1))`

Returns a float in `[0, 1]`. Higher is better.

---

## `ww_trainer.version`

Source: `ww_trainer/version.py`

```python
VERSION_MAJOR = 0
VERSION_MINOR = 0
VERSION_BUILD = 1
VERSION_ALPHA = 1
VERSION_STR = "0.0.1a1"
```
