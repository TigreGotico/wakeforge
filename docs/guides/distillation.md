# Knowledge Distillation

Compress a large teacher extractor (HuBERT, Wav2Vec2) into a compact `CnnLstmExtractor` student that runs on constrained hardware.

Source: `ww_trainer/distill.py`

---

## Architecture

### `CnnLstmExtractor` -- `distill.py:32`

4-layer strided Conv1D front-end + 2-layer bidirectional LSTM + linear projection.

| Layer | Kernel | Stride | Output |
|-------|--------|--------|--------|
| Conv1d(1, C, 10) | 10 | 10 | `[B, C, T/10]` |
| Conv1d(C, C, 3) | 3 | 2 | `[B, C, T/20]` |
| Conv1d(C, C, 3) | 3 | 4 | `[B, C, T/80]` |
| Conv1d(C, C, 3) | 3 | 2 | `[B, C, T/160]` |
| LSTM(C, H, bidir) | -- | -- | `[B, T/160, 2H]` |
| Linear(2H, D) | -- | -- | `[B, T/160, D]` |

Total stride: 10 x 2 x 4 x 2 = 160 (10 ms frames at 16 kHz). Each Conv1d layer uses GELU activation and GroupNorm (`distill.py:74-90`).

**Parameters:** `output_dim` (256), `conv_channels` (256), `lstm_hidden` (256), `lstm_layers` (2), `dropout` (0.1). Typical param count: 500K-2M.

---

## `KnowledgeDistillationTrainer` -- `distill.py:143`

Trains student to mimic teacher features while classifying wake words.

**Loss:** `alpha * MSE(student_feats, teacher_feats) + (1 - alpha) * BCE(logits, labels)` (`distill.py:353`).

- `alpha=0.7` (default): 70% distillation, 30% task loss.
- Teacher is frozen (`distill.py:305-307`).
- If teacher and student have different `feature_dim`, a linear projection aligns them (`distill.py:203-206`).
- Time-length mismatch handled by interpolation (`distill.py:230-240`).

**Training loop** (`distill.py:247`):
- AdamW optimizer with cosine annealing LR schedule.
- Gradient clipping at 1.0 (`distill.py:356`).
- Early stopping with configurable patience (`distill.py:441-443`).
- Best student exported to ONNX automatically (`distill.py:453-458`).

---

## Convenience Function

`distill_hubert_to_cnn_lstm()` -- `distill.py:463`

```python
from ww_trainer.distill import distill_hubert_to_cnn_lstm

trainer = distill_hubert_to_cnn_lstm(
    teacher_onnx_path="hubert.onnx",
    train_data=[("audio/hey_jarvis_001.wav", "1"), ...],
    val_data=[...],
    output_dir="models/student/",
    student_dim=256,
    alpha=0.7,
    epochs=50,
)
# Output: models/student/student_extractor.onnx
```

Internally creates `OnnxFeatureExtractor` (teacher), `CnnLstmExtractor` (student), and `FfnClassifierHead` (`distill.py:508-519`).

---

## Workflow

1. Export teacher to ONNX (e.g. `HubertExtractor.export_to_onnx("hubert.onnx")`).
2. Run `distill_hubert_to_cnn_lstm("hubert.onnx", train_data, val_data)`.
3. Student ONNX saved to `output_dir/student_extractor.onnx`.
4. Deploy with `OnnxFeatureExtractor("student_extractor.onnx")` on target device.

---

## When to Use

- You trained a high-accuracy model with HuBERT/Wav2Vec2 but need to deploy on RPi or similar.
- The student (500K-2M params) is 100-600x smaller than HuBERT base (90M).
- Student ONNX runs in <10ms on RPi 4 (vs 100-500ms for HuBERT ONNX).

**When NOT to use:** If MFCC + a good head already meets your accuracy target, distillation adds unnecessary complexity.
