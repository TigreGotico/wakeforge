# ww-trainer — Quick Facts

| Field | Value |
|-------|-------|
| Package name | `ww_trainer` |
| Version | 0.0.1a1 |
| License | Apache 2.0 |
| Python | ≥ 3.10 |
| Entry point | `ww_trainer-train` → `ww_trainer.train:train` |

---

## Key Classes

| Class | Module | Description |
|-------|--------|-------------|
| `AudioDataset` | `ww_trainer.dataset` | PyTorch Dataset with on-the-fly augmentation |
| `collate_fn` | `ww_trainer.dataset` | Pads variable-length waveforms in a batch |
| `OnnxFeatureExtractor` | `ww_trainer.feats` | HuBERT feature extraction via ONNX runtime |
| `SlidingFeatureCacheTensor` | `ww_trainer.feats` | Rolling feature buffer for streaming inference |
| `FfnClassifierHead` | `ww_trainer.model` | Feedforward classifier head |
| `CnnClassifierHead` | `ww_trainer.model` | 1-D convolutional classifier head |
| `GruClassifierHead` | `ww_trainer.model` | GRU-based sequence classifier head |
| `BaseWakeModel` | `ww_trainer.model` | Combines feature extractor + classifier head |
| `RobustProtoDiversityLoss` | `ww_trainer.loss` | RPPL composite loss (recommended) |
| `LossManager` | `ww_trainer.loss` | Assembles and weights multiple loss functions |
| `WakeWordTrainer` | `ww_trainer.trainer` | Full training loop with MLflow support |

---

## Supported Audio Formats

Handled by `_load_audio_mono` / torchaudio: `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`

---

## ONNX Outputs

| File | Contents |
|------|----------|
| `<name>.onnx` | Classifier head (FP32) |
| `<name>_int8.onnx` | INT8 quantized classifier head |
| `<name>_int16.onnx` | INT16 quantized classifier head |
| `<name>_featurizer.onnx` | HuBERT feature extractor (FP32) |

---

## Feature Tensor Shape

`OnnxFeatureExtractor.forward(wavs)` returns `[B, T, 768]`

- `B` = batch size
- `T` = number of HuBERT frames (≈ 50 per second of audio at 16 kHz)
- `768` = HuBERT feature dimension

---

## Dataset Format

`AudioDataset` expects a list of `(path: str, label: str)` tuples where label is `"1"` (wake) or `"0"` (non-wake).
