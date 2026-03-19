# ww-trainer — Production Readiness Plan

## Current State (v0.0.1a1)

Research-quality prototype. Core concepts are sound; the codebase needs hardening before production
use on edge devices. Key concerns:

| Area | State |
|------|-------|
| Tests | 45 passing (from previous sprint) |
| Documentation | FAQ.md, QUICK_FACTS.md, AUDIT.md, SUGGESTIONS.md, MAINTENANCE_REPORT.md |
| Bugs | 7 fixed in previous sprint |
| trainer.py | 1170-line monolith doing 6 jobs |
| Packaging | legacy setup.py |
| Streaming inference | Designed but not wired up |
| CLI entry point | Broken (`ww_trainer.train:train` module doesn't exist) |
| MfccExtractor | Only in scripts/ — not in main package |
| HuBERT/W2VBert extractors | Only in scripts/ — not in main package |
| ONNX-only inference path | Missing — inference currently requires PyTorch |

---

## Architecture Guideline

> **Wake words are always modeled as feature-extractor + classifier head. The feature extractor can
> be shared across models. Everything is exported to ONNX. Inference requires only ONNX runtime.
> This is a full training and research suite for wake words of all sizes and accuracies for all
> hardware requirements — including MFCC and other classical extractors, all also exported to ONNX.**

---

## Architecture Map

```
[Feature Extractor]   →   [Classifier Head]   →   [Wake Word Model]
  BaseExtractor              ClassifierHead           BaseWakeModel
  ├── MfccExtractor           ├── FfnClassifierHead    feature_extractor + classifier
  ├── OnnxFeatureExtractor    ├── CnnClassifierHead
  ├── HubertExtractor (train) └── GruClassifierHead
  └── W2VBertExtractor (train)
         ↓ export_to_onnx()        ↓ export_to_onnx()
     extractor.onnx            classifier_head.onnx
              \                         /
               └── inference.py ───────
                   OnnxWakeWordInferencer
                   (onnxruntime ONLY, no torch)
```

**Key invariants:**
- `BaseExtractor.forward(wavs) -> [B, T, F]` — uniform output contract
- `ClassifierHead.forward(feats: [B, T, F]) -> [B]` — uniform input contract
- Both `export_to_onnx()` methods are required (not optional)
- Inference: load extractor ONNX + head ONNX → run both with `ort.InferenceSession`

---

## Hardware Tier Definitions

| Tier | Extractor | Head | Input | Params | Target Hardware |
|------|-----------|------|-------|--------|----------------|
| micro | MFCC (40 coeff) | FFN (128d) | raw audio | ~50K | MCU, RPi Zero |
| small | MFCC (40 coeff) | GRU (128d) | raw audio | ~200K | RPi, small SBC |
| medium | HuBERT-base ONNX | FFN (128d) | raw audio | ~90M feat + 200K head | RPi 4, laptop |
| large | W2VBert/HuBERT | GRU (256d, bidir) | raw audio | ~300M feat + 1M head | Server/workstation |

---

## Phase 0 — Fix Existing Issues (DONE)

- ✅ 7 bugs fixed (feats.py, dataset.py, model.py, loss.py)
- ✅ 45 tests passing
- ✅ FAQ.md, QUICK_FACTS.md, AUDIT.md, SUGGESTIONS.md, MAINTENANCE_REPORT.md created

---

## Phase 1 — Extractor Consolidation (feats.py)

**Goal:** All feature extractors in `ww_trainer/feats.py` as `BaseExtractor` subclasses.

**1a. Move `MfccExtractor` from `scripts/export_mfcc.py` → `ww_trainer/feats.py`**
- Class already complete; copy with minor cleanup (absolute imports, type hints)
- Verify `export_to_onnx()` works (the script already demos this)
- Output shape must be `[B, T, n_mfcc]` (transpose from `[B, n_mfcc, T]` for head compatibility)
- Add `feature_dim` property so trainer can query it without instantiating the head

**1b. Move `HubertExtractor` / `W2VBertExtractor` from `scripts/export_w2vbert.py` → `ww_trainer/feats.py`**
- These are training-time extractors (require `transformers` library)
- Mark with `_REQUIRES_TRANSFORMERS = True` class attribute
- `export_to_onnx()` exports to a file that `OnnxFeatureExtractor` then loads for inference/training
- Guard import: `try: from transformers import ... except ImportError: raise ImportError("Install transformers for HubertExtractor")`

**1c. Add `feature_dim` property to `BaseExtractor`**
- Abstract property or attribute set in `__init__`
- Enables trainer to create matching head automatically without knowing the extractor type

**Affected files:**
- `ww_trainer/feats.py` — add 3 new classes, add `feature_dim` property
- `test/test_feats.py` — new test file: MfccExtractor forward pass, output shape, export_to_onnx smoke

---

## Phase 2 — ONNX-Only Inference Module

**Goal:** `ww_trainer/inference.py` — zero PyTorch imports, onnxruntime only.

```python
# ww_trainer/inference.py
import numpy as np
import onnxruntime as ort

class OnnxWakeWordInferencer:
    def __init__(self, extractor_path: str, head_path: str,
                 sample_rate: int = 16000, device: str = "auto"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" \
                    else ["CPUExecutionProvider"]
        self.extractor = ort.InferenceSession(extractor_path, providers=providers)
        self.head = ort.InferenceSession(head_path, providers=providers)
        self.sample_rate = sample_rate

    def infer(self, audio: np.ndarray) -> float:
        """audio: 1-D float32 numpy array at self.sample_rate. Returns sigmoid probability."""
        wav = audio[np.newaxis, :]  # [1, T]
        feats = self.extractor.run(None, {"input_values": wav})[0]  # [1, T, F]
        logit = self.head.run(None, {"input_features": feats})[0]   # [1]
        return float(1.0 / (1.0 + np.exp(-logit[0])))  # sigmoid, no torch

    def infer_streaming(self, audio_chunk: np.ndarray,
                        cache: np.ndarray) -> tuple[float, np.ndarray]:
        """Process one chunk; returns (prob, updated_cache)."""
        ...
```

**Affected files:**
- `ww_trainer/inference.py` — new file
- `test/test_inference.py` — test with pre-exported ONNX dummy models (use `torch.onnx.export` of trivial nets in test fixtures)

---

## Phase 3 — Trainer Extractor Registry & Shared Extractor

**Goal:** `WakeWordTrainer` supports all extractor types; extractor can be shared across models.

**3a. Extractor factory in `WakeWordTrainer.create_model()`**
```python
EXTRACTOR_REGISTRY = {
    "onnx": OnnxFeatureExtractor,
    "mfcc": MfccExtractor,
    "hubert": HubertExtractor,
    "w2vbert": W2VBertExtractor,
}
extractor_cls = EXTRACTOR_REGISTRY[featurizer_type]
feature_extractor = extractor_cls(featurizer_arg, ...)
```

**3b. `feature_dim` auto-detection** — query `extractor.feature_dim` to set `input_size` on head automatically

**3c. Shared extractor support** — `WakeWordTrainer` accepts an optional pre-built extractor; multiple `WakeWordTrainer` instances can share one `OnnxFeatureExtractor` session

**3d. Fix broken CLI entry point** (setup.py line 62: `ww_trainer.train:train` → doesn't exist)
- Either: add `train()` function to `trainer.py` and expose it, or rename setup.py entry point
- Must work: `uv run ww_trainer-train --help`

**Affected files:**
- `ww_trainer/trainer.py` — extractor registry, feature_dim auto-detect, shared extractor, fix CLI
- `setup.py` (or new `pyproject.toml`) — fix entry point

---

## Phase 4 — trainer.py Refactor (Monolith Split)

**Goal:** Split 1170-line `trainer.py` into focused modules.

| New File | Contents from trainer.py | ~Lines |
|----------|--------------------------|--------|
| `ww_trainer/visualization.py` | `_log_embeddings_to_mlflow`, `_plot_roc`, `_plot_pr`, `_plot_det` | ~208 |
| `ww_trainer/mining.py` | `_mine_hard_negatives`, hard-negative cache | ~104 |
| `ww_trainer/checkpoint.py` | `save_checkpoint`, `load_checkpoint`, best-metric tracking | ~45 |
| `trainer.py` (residual) | training loop only | ≤400 |

**Affected files:** `ww_trainer/trainer.py`, 3 new modules, tests for each

---

## Phase 5 — Streaming Inference (Wired Up)

**Goal:** `BaseWakeModel.forward_streaming()` implemented and `OnnxWakeWordInferencer.infer_streaming()` working.

```python
# model.py — BaseWakeModel
def forward_streaming(self, audio_chunk: torch.Tensor,
                      cache: SlidingFeatureCacheTensor) -> float:
    feats = self.feature_extractor([audio_chunk])      # [1, T_new, F]
    cached = cache(feats.squeeze(0))                   # [T_window, F]
    logit = self.classifier.forward(cached.unsqueeze(0))
    return torch.sigmoid(logit).item()
```

**Affected files:** `ww_trainer/model.py`, `ww_trainer/inference.py`, `test/test_streaming.py`

---

## Phase 6 — Packaging & Documentation

- Migrate `setup.py` → `pyproject.toml`
- `[project.optional-dependencies]`: `dev`, `transformers` (for HuBERT/W2VBert training), `vc` (chatterbox_onnx)
- Update `README.md` with hardware tier table, quickstart, extractor matrix
- Update `docs/index.md` with full architecture narrative
- Update `PLAN.md` and `TODO.md` in ww-trainer/ to reflect this architecture

---

## Critical Files

| File | Role | Status |
|------|------|--------|
| `ww_trainer/feats.py` | All feature extractors | Extend with MFCC, HuBERT, W2VBert |
| `ww_trainer/model.py` | Classifier heads + BaseWakeModel | Add `forward_streaming` |
| `ww_trainer/trainer.py` | Training loop + CLI | Fix CLI, add extractor registry, split |
| `ww_trainer/inference.py` | ONNX-only inference | **Create new** |
| `scripts/export_mfcc.py` | MfccExtractor reference | Source for migration |
| `scripts/export_w2vbert.py` | HuBERT/W2VBert reference | Source for migration |
| `setup.py` → `pyproject.toml` | Packaging | Fix CLI entry point |
| `test/test_feats.py` | Extractor tests | **Create new** |
| `test/test_inference.py` | ONNX inference tests | **Create new** |
| `test/test_streaming.py` | Streaming tests | **Create new** |

---

## Verification

After each phase:
```bash
uv run --python .venv pytest test/ -v --cov=ww_trainer
# Must: 0 failures, coverage ≥ 80%

uv run ww_trainer-train --help
# Must: display CLI help without error

python -c "from ww_trainer.inference import OnnxWakeWordInferencer; print('ok')"
# Must: import succeeds with only onnxruntime installed (no torch)
```

Phase 1 specific: verify MFCC extractor exports valid ONNX and OnnxFeatureExtractor can load it.
Phase 2 specific: verify inference.py has zero `import torch` at module level.

---

## Definition of Done (v1.0)

- [ ] All extractors in `ww_trainer/feats.py` with `feature_dim` property
- [ ] `ww_trainer/inference.py` exists with zero PyTorch imports
- [ ] CLI entry point fixed and working
- [ ] `uv run pytest test/ -v --cov=ww_trainer` passes with ≥ 80% coverage
- [ ] trainer.py ≤ 400 lines
- [ ] `forward_streaming` implemented and tested
- [ ] All mandated docs exist and are accurate
- [ ] pyproject.toml replaces setup.py
- [ ] MAINTENANCE_REPORT.md has an entry for every change
- [ ] FAQ.md updated
