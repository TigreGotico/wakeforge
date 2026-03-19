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

## Tech Debt

### TD-001 — trainer.py — 1170-line monolith
`ww_trainer/trainer.py` handles training loop, checkpointing, hard-negative mining, embedding visualization, MLflow logging, and CLI — six distinct responsibilities in one file. Planned refactor in Phase 3 (see `PLAN.md`).

### TD-002 — No streaming inference wired up
`SlidingFeatureCacheTensor` (`feats.py:27-53`) exists and is now tested, but `BaseWakeModel.forward_streaming` is not yet implemented. Required for edge deployment.

### TD-003 — setup.py instead of pyproject.toml
Legacy packaging. Migration planned in Phase 5 (see `PLAN.md`).

### TD-004 — 0% test coverage before this audit
No tests existed in v0.0.1a1. Test suite added in this sprint (45 tests, see `test/`).
