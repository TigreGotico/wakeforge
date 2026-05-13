# AI Transparency Log

## 2026-05-13 — CI fix for PR #9 (feat/ocsvm-head)

**AI Model**: claude-sonnet-4-6
**Actions Taken**:
- Rebased `feat/ocsvm-head` onto `origin/dev` (clean, no conflicts).
- Diagnosed CI failures from run logs (runs 24913170931, 24913170978, 24913170802).
- Moved `torchcodec` from core dependencies to optional `[torchcodec]` extra — fixes `RuntimeError: Could not load libtorchcodec` in CI environments without FFmpeg.
- Added `onnxscript` to `[dev]` and new `[test]` extras — fixes `ModuleNotFoundError: No module named 'onnxscript'` in ONNX export tests (PyTorch ≥ 2.x).
- Added `[test]` extra alias — CI workflow installs `[test]` but only `[dev]` existed.
- Fixed license check workflow: added `exclude_packages` for `soxr` (LGPL-2.1, transitive via librosa) and `tqdm` (MPL-2.0, transitive via core deps).
- Fixed malformed `notebooks/genetic_search.ipynb` — the file was a single raw cell wrapping the entire notebook JSON; unwrapped it to 20 proper cells.
- Updated `docs/faq.md` with entries for `torchcodec` and `onnxscript`.

**Oversight**: Human review required before push. Tests pass locally (876 passed, 5 skipped).
