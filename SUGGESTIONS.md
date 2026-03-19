# ww-trainer — Suggestions

Improvement proposals for future development.

---

## S-005a — Native ONNX implementation for HMM Forward Pass

**Problem:** `HMMStateExtractor` currently relies on ONNX tracing of a vectorized PyTorch loop. While functional, it is less efficient than a native ONNX `Scan` operator which can be further optimized by compilers like TensorRT or OpenVINO.

**Solution:** Implement the HMM forward algorithm using `onnx.helper.make_node("Scan", ...)` with a custom step sub-graph. This avoids the overhead of unrolling the time loop during export.

**Impact:** Improved inference latency on specialized edge hardware and reduced ONNX file size.

---

## S-006a — Sparse transition matrix lookup for Markov Extractor

**Problem:** The `MarkovTransitionExtractor` currently exports a dense transition matrix. For large vocabularies (`n_codes`) and high orders, this matrix grows exponentially ($V^{order+1}$), leading to huge ONNX files and RAM usage.

**Solution:** Implement a sparse lookup mechanism in the ONNX graph using `Gather` and `Where`. Store only the rows that were actually visited during training and use a uniform distribution fallback for unseen contexts.

**Impact:** Significant reduction in model footprint for high-order Markov models (e.g., $O(3)$ with 64 codes).

---

## S-001 — Implement `forward_streaming` for edge deployment ✅ DONE

Implemented in `BaseWakeModel.forward_streaming` — `ww_trainer/model.py:119-134`.

---

## S-002 — Optuna hyperparameter sweep ✅ DONE

Implemented in `ww_trainer/sweep.py`: Optuna, grid, random, genetic, and micro search strategies.

---

## S-003 — Mixed-precision training ✅ DONE

Implemented in `ww_trainer/trainer.py`: `use_amp` flag, `GradScaler`, `torch.amp.autocast`.

---

## S-004 — Gradient accumulation ✅ DONE

Implemented in `ww_trainer/trainer.py`: `accumulate_grad_batches` parameter with scaled loss and conditional optimizer step.

---

## S-005b — Persist hard-negative mining cache across epochs ✅ DONE

Implemented in `ww_trainer/mining.py`: `save_mining_cache`/`load_mining_cache` with `hardneg_cache.pt` persistence.

---

## S-006b — PyTorch feature backend alternative to ONNX ✅ DONE

Implemented as `TorchAudioHubertExtractor` — `ww_trainer/feats.py`. Uses `torchaudio.pipelines` (HUBERT_BASE, HUBERT_LARGE, etc.) with no `transformers` dependency.

---

## S-007 — Dataset validation at `__init__` time ✅ DONE

Implemented in `AudioDataset.__init__` — `ww_trainer/dataset.py`. `validate=True` checks files are readable via `soundfile.info()`. Class imbalance warning (>10:1 ratio) always active.

---

## S-008 — Replace UMAP with optional import + graceful fallback ✅ DONE

Implemented in `ww_trainer/visualization.py`: `_HAS_UMAP` flag with t-SNE fallback.

---

## S-009 — Optuna hyperparameter sweep ✅ DONE

Duplicate of S-002.

---

## S-010 — DDP multi-GPU training ✅ DONE

Implemented in `ww_trainer/ddp.py`: `setup_ddp`, `wrap_model_ddp`, `create_distributed_loader`, `is_main_process`. Launch with `torchrun`.

---

## S-011 — Confidence calibration (Platt scaling) ✅ DONE

Implemented in `ww_trainer/calibration.py`: `fit_platt_scaling`, `apply_platt_scaling`, `save_calibration`, `load_calibration`, `calibrate_model`.

---

## S-012 — Quantization-aware training (QAT) for MCU targets ✅ DONE

Implemented in `ww_trainer/qat.py`: `prepare_qat` and `convert_qat`. Uses `torch.ao.quantization`.

---

## S-013 — ESP-IDF C export for ESP32 tiers ✅ DONE

Implemented in `ww_trainer/export_c.py`: `export_to_c_header` generates a self-contained `.h` file with int8 weights and a C inference function for FFN models.

---

## S-014 — BaseExtractor.device should track nn.Module.to() ✅ DONE

Implemented via `_apply` override in both `BaseExtractor` (`feats.py`) and `ClassifierHead` (`model.py`). Device attribute auto-updates on `.to()`/`.cuda()`/`.cpu()`.
