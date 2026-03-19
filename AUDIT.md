# ww-trainer — Audit

Known issues, tech debt, and limitations. All claims are evidence-based with `file:line` citations.

---

## Confirmed Bugs (as of v0.0.1a1)

### BUG-001 — feats.py:87 — Wrong argument to ONNX checker ✅ FIXED
**Severity:** Medium — silently skips model validation
**File:** `ww_trainer/feats.py:87`
**Description:** `onnx.checker.check_model(out)` was called with the file path string instead of the loaded ONNX model object. The checker accepts a path string in some versions but the intent (and correct API) is to pass the model object returned by `onnx.load()` on line 86.
**Fix:** Changed to `onnx.checker.check_model(onnx_model)`.

---

### BUG-002 — dataset.py:237 — Hardcoded /tmp/ path for voice conversion ✅ FIXED
**Severity:** Low-Medium — breaks on Windows; race condition on concurrent workers
**File:** `ww_trainer/dataset.py` (formerly line 237)
**Description:** `vc_path = f"/tmp/vc_{uuid4()}.wav"` hardcoded the Linux `/tmp/` directory. This is non-portable and creates a temporary file that is never explicitly cleaned up.
**Fix:** Replaced with `tempfile.NamedTemporaryFile(suffix=".wav", delete=False)`.

---

### BUG-003 — dataset.py:253 — Bare `except:` swallows all exceptions ✅ FIXED
**Severity:** Medium — hides voice conversion errors, making debugging impossible
**File:** `ww_trainer/dataset.py` (formerly line 253)
**Description:** A bare `except:` clause caught `BaseException` (including `KeyboardInterrupt`, `SystemExit`). Errors during voice conversion were silently discarded.
**Fix:** Changed to `except Exception as exc:` with `logger.warning(...)`.

---

### BUG-004 — model.py:219 — GRU shape heuristic silently wrong ✅ FIXED
**Severity:** Medium — silent incorrect results when feature_dim == sequence_len
**File:** `ww_trainer/model.py:219`
**Description:** `GruClassifierHead._ensure_correct_shape` auto-transposes `[B, F, T]` to `[B, T, F]` when `D1 == input_size`. When `D1 == D2 == input_size`, the heuristic cannot determine orientation and falls through silently with potentially wrong results.
**Fix:** Added explicit `ValueError` for the ambiguous case.

---

### BUG-005 — feats.py:44 — SlidingFeatureCacheTensor shift logic out-of-bounds ✅ FIXED
**Severity:** High — `RuntimeError` at runtime when `T_new > current_len`
**File:** `ww_trainer/feats.py:44`
**Description:** `self.feature_cache[:-shift]` selects `window_size - shift` slots but only `current_len - shift` frames are valid to copy. When `T_new > current_len` the slice target is larger than the source, causing a shape mismatch `RuntimeError`.
**Fix:** Changed to `self.feature_cache[:new_len] = self.feature_cache[shift:self.current_len]`.

---

### BUG-007 — utils.py:290 — Return statement overwritten by utility ✅ FIXED
**Severity:** High — breaks all triplet-based loss functions
**File:** `ww_trainer/utils.py:290`
**Description:** During the addition of `embed_onnx_metadata`, the return statement of `sample_semihard_triplets` was accidentally overwritten, causing it to return `None` instead of the mined triplets.
**Fix:** Restored the multi-value return statement before the new utility.

---

### BUG-008 — feats.py:317 — STFT numerical drift in ONNX backends
**Severity:** Low — minor parity mismatch (~1e-4) between PyTorch and ONNX Runtime
**File:** `ww_trainer/feats.py:317`
**Description:** `torch.stft` used in MFCC and Filterbank extractors produces slightly different results in the exported ONNX graph compared to the native PyTorch implementation. This is due to internal differences in FFT windowing and floating-point optimizations in ONNX Runtime.
**Status:** Documented in `FAQ.md`. Recommended workaround: increase tolerance in unit tests to `1e-3`.

---

### BUG-006 — dataset.py — Top-level import of optional dependency
**Severity:** Low — breaks any import of `ww_trainer.dataset` when `chatterbox_onnx` is not installed
**File:** `ww_trainer/dataset.py` (formerly line 15)
**Description:** `from chatterbox_onnx import ChatterboxOnnx` was at module top-level. Since `chatterbox_onnx` is an optional voice-conversion dependency, this caused `ImportError` for users who haven't installed it — even when they never use voice conversion.
**Fix:** Moved import inside the `if vc_folder and vc_prob > 0:` branch with a comment.

---

### BUG-007 — loss.py:451 — margin not stored in loss_entry dict ✅ FIXED
**Severity:** Low — margin config for triplet mining always defaults to 1.0, ignoring user config
**File:** `ww_trainer/loss.py:451`
**Description:** `LossManager.__init__` stores each loss as `{"name", "weight", "criterion"}` but not `"margin"`. In `compute_loss`, `loss_entry.get("margin", 1.0)` always returns the default because the key was never stored.
**Fix:** Added `"margin": cfg.get("margin", 1.0)` to the stored dict.

---

### BUG-009 — model.py:151 — export_to_onnx positional arg mismatch ✅ FIXED
**Severity:** High — silently swaps quantize/dynamo flags
**File:** `ww_trainer/model.py:151`
**Description:** `BaseWakeModel.export_to_onnx` called `self.classifier.export_to_onnx(out, simplify, quantize, metadata=metadata)` positionally, but `ClassifierHead.export_to_onnx` signature is `(out, quantize, dynamo, metadata)`. This passed `simplify` as `quantize` and `quantize` as `dynamo`. Quantization was silently skipped when requested, and dynamo export was unexpectedly enabled.
**Fix:** Switched to keyword arguments: `export_to_onnx(out, quantize=quantize, dynamo=simplify, metadata=metadata)`.

---

### BUG-010 — model.py:154 — metadata kwarg passed to extractors that don't accept it ✅ FIXED
**Severity:** Medium — TypeError at runtime
**File:** `ww_trainer/model.py:154`
**Description:** `self.feature_extractor.export_to_onnx(f_out, quantize, metadata=metadata)` passed `metadata` as a keyword argument, but `BaseExtractor.export_to_onnx` did not accept it. This caused a TypeError when using `export_featurizer=True` with metadata on any non-Markov/HMM extractor.
**Fix:** Added `metadata: dict = None` parameter to `BaseExtractor.export_to_onnx` and all overrides.

---

## Tech Debt

### TD-001 — trainer.py monolith ✅ RESOLVED
`trainer.py` reduced from 1170 → 246 lines. Training loop extracted to `ww_trainer/loop.py:training_loop()`. `compute_readiness()` moved to `evaluation.py`. `save_intermediate_checkpoint()` moved to `checkpoint.py`.

### TD-002 — No streaming inference wired up
`SlidingFeatureCacheTensor` (`feats.py:27-53`) exists and is now tested, but `BaseWakeModel.forward_streaming` is not yet implemented. Required for edge deployment.

### TD-003 — setup.py instead of pyproject.toml ✅ RESOLVED
Migrated to `pyproject.toml` with optional dependency groups: `dev`, `transformers`, `vc`, `mlflow`, `sweep`, `markov`, `datagen`.

### TD-004 — 0% test coverage before this audit
No tests existed in v0.0.1a1. Test suite added in this sprint (45 tests, see `test/`).

### TD-005 — SileroVadWrapper downloads from internet at init
`feats.py:SileroVadWrapper.__init__` calls `torch.hub.load('snakers4/silero-vad', ...)` which downloads from GitHub. Fails in air-gapped/CI environments. The `onnx_path` alternative exists but the PyTorch Hub path has no offline fallback or cache control.

### TD-006 — Dataset generation scripts have zero test coverage
`scripts/dataset_generation/` (5 scripts, ~976 lines) have no tests. Some use `os.system()` for shell commands. These were ported from Jupyter notebooks and should be validated.

### TD-007 — Manual `self.device` attribute pattern ✅ FIXED
Fixed via `_apply` override in `BaseExtractor` and `ClassifierHead`. Device now auto-syncs on `.to()`/`.cuda()`/`.cpu()`.

### TD-008 — Flaky `test_hmm_fit_updates_parameters`
`test/test_hmm_extended.py::test_hmm_fit_updates_parameters` fails intermittently in full suite runs but passes in isolation. Likely a seed/ordering issue — the sinusoidal training data may not always produce sufficiently non-uniform HMM parameters depending on K-means initialization.

### TD-009 — ClassifierHead ONNX batch axis was fixed ✅ FIXED
`ClassifierHead.export_to_onnx` (`model.py:47`) previously only set dynamic axes for the time dimension, not batch. Batch>1 ONNX inference failed. Fixed by adding `{0: "batch_size"}` to dynamic_axes for both input and output.
