# ww-trainer — Suggestions

Improvement proposals for future development.

---

## S-001 — Implement `forward_streaming` for edge deployment

**Problem:** `SlidingFeatureCacheTensor` (`feats.py:27`) exists but is never called. Edge devices need chunk-by-chunk inference without reprocessing the full audio window every frame.

**Solution:**
```python
# In BaseWakeModel (model.py)
def forward_streaming(self, audio_chunk: torch.Tensor, cache: SlidingFeatureCacheTensor) -> float:
    feats = self.feature_extractor([audio_chunk])   # [1, T_new, 768]
    cached = cache(feats.squeeze(0))                # [T_total, 768]
    logit = self.classifier.forward(cached.unsqueeze(0))  # [1]
    return torch.sigmoid(logit).item()
```

**Impact:** Enables real-time wake word detection on microcontrollers and single-board computers without batching overhead.

---

## S-002 — Optuna hyperparameter sweep

**Problem:** Key hyperparameters (learning rate, hidden dim, loss weights, batch size) are currently tuned manually.

**Solution:** Add `ww_trainer/sweep.py` with Optuna integration. Expose as `ww_trainer-sweep` CLI entry point.

**Impact:** Automated tuning typically yields 5–15% relative AUC improvement with minimal human effort.

---

## S-003 — Mixed-precision training (`torch.amp.autocast`)

**Problem:** Training on CUDA is memory-bound at FP32. Feature extraction alone occupies significant GPU memory.

**Solution:** Wrap the training step in `torch.amp.autocast("cuda")` and use `GradScaler`. ~2× throughput improvement expected on Ampere GPUs.

**Impact:** Faster iteration, larger effective batch sizes.

---

## S-004 — Gradient accumulation

**Problem:** Memory constraints on consumer GPUs limit batch size, which hurts metric learning losses that need diverse negatives per batch.

**Solution:** Add `--accumulate-grad-batches N` CLI flag. Accumulate gradients over N steps before calling `optimizer.step()`. Effective batch = `batch_size × N`.

**Impact:** Metric learning quality scales with batch size diversity.

---

## S-005 — Persist hard-negative mining cache across epochs

**Problem:** The hard-negative mining cache (`trainer.py:~525-629`) is rebuilt from scratch each epoch. For large datasets this is expensive.

**Solution:** Serialize `cache.pt` to the checkpoint directory at end of each epoch. Reload on resume. Use cosine similarity staleness check to invalidate entries after N epochs.

**Impact:** ~30% training step speedup on large datasets.

---

## S-006 — PyTorch feature backend alternative to ONNX

**Problem:** `OnnxFeatureExtractor` requires `onnxruntime`, which has platform-specific installation challenges. There is no pure-PyTorch fallback.

**Solution:** Add `TorchFeatureExtractor(BaseExtractor)` using `torchaudio.pipelines.HUBERT_BASE` or a lightweight custom encoder. Both backends expose the same `forward(wavs) -> [B, T, 768]` interface.

**Impact:** Removes `onnxruntime` as a hard dependency for training; simplifies CI.

---

## S-007 — Dataset validation at `__init__` time

**Problem:** `AudioDataset` discovers bad files (missing, wrong sample rate, corrupted) only at `__getitem__` time during training, causing cryptic mid-epoch errors.

**Solution:** Add a `validate=True` kwarg that checks all file paths exist, are readable, and have a detectable audio format. Log label distribution and warn on severe class imbalance (>10:1 ratio).

**Impact:** Shift fail-fast: catch data issues before spending GPU time.

---

## S-008 — Replace UMAP with optional import + graceful fallback

**Problem:** `umap-learn` is a heavy optional dependency used only for embedding visualization. If absent, the entire visualization step crashes rather than falling back gracefully.

**Solution:** Lazy import `umap` inside `_log_embeddings_to_mlflow`. If import fails, fall back to t-SNE (always available via scikit-learn) and emit a warning.

**Impact:** Cleaner optional dependency handling; no crash for users without UMAP.

---

## S-009 — Optuna hyperparameter sweep

**Problem:** Key hyperparameters (learning rate, hidden dim, loss weights, batch size, GRU layers) are manually tuned. Small changes to these can have large effects on AUC and false-positive rate.

**Solution:** Add `ww_trainer/sweep.py` with Optuna integration and expose as `ww_trainer-sweep` CLI entry point.

```python
import optuna
from ww_trainer.trainer import WakeWordTrainer

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    gru_n_layers = trial.suggest_int("gru_n_layers", 1, 3)

    trainer = WakeWordTrainer(
        arch="gru",
        featurizer="extractor.onnx",
        hidden_dim=hidden_dim,
        gru_n_layers=gru_n_layers,
        losses_cfg=[{"name": "bce", "weight": 1.0}],
    )
    trainer.train(
        train_data=train_data,
        test_data=test_data,
        epochs=10,
        batch_size=batch_size,
        lr=lr,
        output_dir=f"sweep/{trial.number}",
    )
    return best_f1  # returned from trainer

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(study.best_params)
```

**Impact:** Automated tuning typically yields 5–15% relative AUC improvement with minimal human effort.

---

## S-010 — DDP multi-GPU training

**Problem:** Training with large neural extractors (HuBERT, Wav2Vec2) is slow on a single GPU. The codebase has no multi-GPU support.

**Solution:** Wrap `BaseWakeModel` in `torch.nn.parallel.DistributedDataParallel` and launch with `torchrun`. Key changes needed:

1. Replace `DataLoader` with a `DistributedSampler`.
2. Wrap model: `model = DDP(model, device_ids=[local_rank])`.
3. Only rank 0 logs metrics and saves checkpoints.
4. Add `--ddp` CLI flag; detect `LOCAL_RANK` environment variable.

**Impact:** Near-linear speedup across GPUs for large-extractor tiers (medium, large).

---

## S-011 — Confidence calibration (Platt scaling)

**Problem:** Raw sigmoid outputs from `BaseWakeModel` are not calibrated probabilities. High-confidence false positives are common when the training set is class-imbalanced.

**Solution:** Add post-training Platt scaling (`sklearn.calibration.CalibratedClassifierCV` or a simple logistic regression fit on the validation set logits).

```python
from sklearn.calibration import calibration_curve
import numpy as np

# After training, collect validation logits and labels
logits = np.array(all_logits)
labels = np.array(all_labels)

# Fit a 1-D logistic regression on the logits
from sklearn.linear_model import LogisticRegression
cal = LogisticRegression()
cal.fit(logits.reshape(-1, 1), labels)

# Save calibration parameters (slope + intercept) alongside model
calibration_params = {"coef": float(cal.coef_[0][0]), "intercept": float(cal.intercept_[0])}
```

At inference time, apply the calibration transform to ONNX logits before returning the probability.

**Impact:** Dramatically reduces false-positive rate for production deployments where probability thresholding matters.

---

## S-012 — Quantization-aware training (QAT) for MCU targets

**Problem:** Post-training INT8 quantization (currently available via `quantize=True` in `export_to_onnx`) can degrade accuracy for the `micro` tier where the model is already small.

**Solution:** Add quantization-aware training using `torch.ao.quantization`:

```python
import torch.ao.quantization as tq

# Before training
model.qconfig = tq.get_default_qat_qconfig("x86")
model_prepared = tq.prepare_qat(model.train())

# After training
model_quantized = tq.convert(model_prepared.eval())

# Export quantized model to ONNX
torch.onnx.export(model_quantized, dummy_input, "model_qat_int8.onnx", opset_version=18)
```

This is most impactful for the `micro` tier (MFCC + FFN, ~50K params) targeting MCU deployment where INT8 arithmetic is native.

**Impact:** 4× model size reduction and 2–4× inference speedup on MCUs with minimal accuracy loss when QAT is used instead of post-training quantization.
