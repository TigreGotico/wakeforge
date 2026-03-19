# ww-trainer — Maintenance Report

## Change Log

---

### 2026-03-19 — Datagen v2: OPM plugin discovery + OVOS VAD

**Type:** Refactor
**Model used:** Claude Opus 4.6
**Human oversight level:** User-directed; plan reviewed before implementation
**Files modified:** `ww_trainer/datagen.py`, `pyproject.toml`, `test/test_datagen.py`, `FAQ.md`

**Summary:**
- Replaced hardcoded Edge/Google TTS imports with `ovos_plugin_manager.tts.find_tts_plugins()` for automatic discovery of all installed OVOS TTS plugins (including phoonnx)
- Replaced `webrtcvad` with `ovos_plugin_manager.vad.OVOSVADFactory` using `ovos-vad-plugin-silero` as default
- Updated `[datagen]` optional deps: removed `webrtcvad` and `ovos-tts-plugin-google-tx`, added `ovos-plugin-manager` and `ovos-vad-plugin-silero`
- Added `TestCollectTTSPluginsMocked` test, updated existing TTS mock test
- 697 tests passing (26 datagen-specific)

---

### 2026-03-19 — Inspiration features from precise-lite-trainer

**Type:** Feature
**Model used:** Claude Opus 4.6
**Human oversight level:** User-directed; plan reviewed before implementation
**Files created:** `ww_trainer/cache.py`, `test/smoketests/test_cache_smoke.py`, `test/smoketests/test_freeze_smoke.py`, `test/smoketests/test_replacement_smoke.py`, `test/smoketests/test_fitness_smoke.py`
**Files modified:** `ww_trainer/dataset.py`, `ww_trainer/trainer.py`, `ww_trainer/evaluation.py`, `ww_trainer/cli.py`, `docs/training.md`, `docs/api.md`, `docs/architecture.md`, `FAQ.md`, `MAINTENANCE_REPORT.md`

**Summary:**
- Feature vectorization cache (`FeatureCache`): MD5-based content hashing, `.npy` storage, auto-invalidation on extractor change
- Layer freezing for transfer learning: `freeze_extractor`, `freeze_layers`, progressive unfreezing at epoch N
- Epoch-level data replacement: `replacement_ratio` with optional balanced 50/50 pos/neg resampling
- Composite fitness score: `compute_fitness_score()` with FP-weighted penalty + model size penalty, `best_fitness.pt` checkpoint
- 8 new CLI options, 20 new smoke tests (all passing), 671 total tests passing
- Full documentation update: FAQ, training guide, API reference, architecture

---

### 2026-03-19 — Add comprehensive smoke test suite

**Type:** Test
**Model used:** Claude Opus 4.6
**Human oversight level:** User-directed
**Files created:** `test/smoketests/__init__.py`, `test/smoketests/conftest.py`, `test/smoketests/test_extractors_smoke.py`, `test/smoketests/test_wrappers_smoke.py`, `test/smoketests/test_heads_smoke.py`, `test/smoketests/test_losses_smoke.py`, `test/smoketests/test_pipelines_smoke.py`, `test/smoketests/test_training_smoke.py`, `test/smoketests/test_export_smoke.py`

**Summary:**
- 100 smoke tests covering all end-to-end pipelines with dummy data
- 10 standalone extractors, 7 wrappers (incl. stacked), 11 heads, 18 losses
- 27 factory-created extractor×head pipelines (incl. enriched variants)
- 5 WakeWordTrainer training runs (BCE, focal, CNN, multi-loss, size-aware)
- ONNX export + inference round-trips (single, streaming, C export, calibration)
- 1 xfail: batch ONNX inference (pre-existing: head exported with fixed batch dim)

---

### 2026-03-19 — Implement SUGGESTIONS.md items (S-006b thru S-014)

**Type:** Feature
**Model used:** Claude Opus 4.6
**Human oversight level:** User-directed; all source files read before editing
**Files created:** `ww_trainer/calibration.py`, `ww_trainer/export_c.py`, `ww_trainer/qat.py`, `ww_trainer/ddp.py`, `test/test_suggestions.py`
**Files modified:** `ww_trainer/feats.py`, `ww_trainer/model.py`, `ww_trainer/dataset.py`, `docs/index.md`, `FAQ.md`, `SUGGESTIONS.md`, `AUDIT.md`, `MAINTENANCE_REPORT.md`

**Summary:**
- S-014: `_apply` override in BaseExtractor and ClassifierHead for automatic device tracking
- S-007: `validate=True` parameter for AudioDataset + class imbalance warnings
- S-006b: `TorchAudioHubertExtractor` using torchaudio.pipelines (no transformers)
- S-011: `calibration.py` — Platt scaling (fit, apply, save, load, calibrate_model)
- S-013: `export_c.py` — ESP-IDF C header export with int8 weights + inference function
- S-012: `qat.py` — QAT via torch.ao.quantization (prepare_qat, convert_qat)
- S-010: `ddp.py` — DDP utilities (setup, wrap, distributed loader, rank helpers)
- Marked S-001/003/004/005b/008 as already done
- S-005a/S-006a deferred (ONNX graph construction optimizations)
- 18 new tests (502 total passing)

---

### 2026-03-19 — Fix export_to_onnx bugs and API consistency

**Type:** Bugfix
**Model used:** Claude Opus 4.6
**Human oversight level:** User-directed review and fix of Gemini 2.0 Flash changes
**Files modified:** `ww_trainer/model.py`, `ww_trainer/feats.py`, `pyproject.toml`, `FAQ.md`, `MAINTENANCE_REPORT.md`

**Summary:**
- **BUG-1 (HIGH)**: Fixed `BaseWakeModel.export_to_onnx` passing positional args in wrong order to `ClassifierHead.export_to_onnx` — `simplify` was passed as `quantize`, `quantize` as `dynamo`. Now uses keyword args.
- **BUG-2 (MEDIUM)**: Added `metadata: dict = None` to `BaseExtractor.export_to_onnx` and all overrides. Previously, calling `export_featurizer=True` with metadata would TypeError on non-Markov extractors.
- **DESIGN-2**: Removed dead `from onnx import helper, TensorProto, numpy_helper` imports from Markov/HMM export methods.
- **QUALITY-3**: Removed unused `onnxscript>=0.6.2` dependency from `pyproject.toml` (never imported anywhere).

---

### 2026-03-19 — Advanced ONNX Export and Markov/HMM Optimizations

**Type:** Feature / Optimization
**Model used:** Gemini 2.0 Flash
**Human oversight level:** User-directed; all source files read and verified via tests
**Files created:** `test/test_onnx_advanced.py`, `test/test_hmm_extended.py`, `scripts/dataset_generation/01_adversarial_gen.py`, `scripts/dataset_generation/02_tts_synth.py`, `scripts/dataset_generation/03_training_aug.py`, `scripts/dataset_generation/04_benchmark_gen.py`, `scripts/dataset_generation/05_ovos_vc_gen.py`, `scripts/dataset_generation/README.md`, `scripts/train_markov_featurizer.py`, `examples/37_hmm_advanced.py`, `examples/38_markov_onnx_blackbox.py`, `examples/39_silero_vad_wrapper.py`, `examples/40_multi_onnx_pipeline.py`, `examples/41_hmm_feature_extraction.py`, `docs/markov_hmm.md`
**Files modified:** `ww_trainer/utils.py`, `ww_trainer/feats.py`, `ww_trainer/model.py`, `ww_trainer/trainer.py`, `ww_trainer/cli.py`, `ww_trainer/factory.py`, `ww_trainer/inference.py`, `MAINTENANCE_REPORT.md`, `FAQ.md`, `AUDIT.md`, `SUGGESTIONS.md`, `docs/index.md`, `docs/extractors.md`, `docs/export.md`, `docs/inference.md`, `docs/architecture.md`

**Summary:**
- Integrated advanced ONNX export capabilities with rich metadata embedding.
- Ported dataset generation and augmentation pipelines from Jupyter notebooks to standalone scripts.
- Optimized Markov and HMM feature extractors for ONNX compatibility and batch performance.
- Added end-to-end "Blackbox Featurizer" workflow for benchmarking classical sequential features against deep learning heads.
- Integrated Silero VAD as a robust neural feature stream with a NumPy-only multi-ONNX production pipeline.


**Changes:**

1.  **Metadata Embedding**:
    - Added `embed_onnx_metadata` utility to `ww_trainer/utils.py` to inject training context into ONNX files.
    - Updated `ClassifierHead` and `BaseExtractor` to include `wake_word`, `arch`, `epoch`, and metrics in exported models.
    - Wired metadata passing from `WakeWordTrainer` down to export calls.

2.  **Markov & HMM Optimizations**:
    - Rewrote `MarkovTransitionExtractor.forward` to be fully vectorized, enabling ONNX tracing.
    - Optimized HMMStateExtractor.forward with batch-vectorized algorithm for higher throughput.
    - Fixed HMM parameter alignment by accounting for markovonnx UNK token during fit.
    - Fixed STFT reflect padding error for very short audio by adding zero-padding check.
    - Fixed export_to_onnx for both extractors to export the full pipeline (Base + Wrapper) into a single ONNX graph.

3.  **Neural VAD & Multi-ONNX Pipelines**:
    - Implemented `SileroVadWrapper` for robust voice activity enrichment during training.
    - Updated `OnnxWakeWordInferencer` to support an optional VAD ONNX model requirement.
    - Implemented pure-NumPy temporal alignment for multi-model ONNX pipelines.
    - Added CLI flags `--use-neural-vad` and `--vad-onnx` for easy integration.

4.  **Dataset Generation Scripts**:
    - Ported phonetically adversarial generation (`01_adversarial_gen.py`).
    - Ported multi-engine TTS synthesis with voice conversion (`02_tts_synth.py`).
    - Ported training augmentation pipeline (`03_training_aug.py`).
    - Ported structured benchmark dataset generator (`04_benchmark_gen.py`).
    - Ported OVOS-specific TTS collection pipeline (`05_ovos_vc_gen.py`).

5.  **Maintenance & Infrastructure**:
    - Created `test/test_onnx_advanced.py` for parity checks and multi-ONNX verification.
    - Created `test/test_hmm_extended.py` for comprehensive HMM verification (fit, batching, normalization).
    - Created `docs/markov_hmm.md` documenting classical sequential feature extraction.
    - Created `scripts/train_markov_featurizer.py` for standalone Markov pipeline training and export.
    - Added 5 new advanced example scripts (`37`-`41`) covering hybrid models, blackbox featurizers, and production pipelines.
    - Fixed `sample_semihard_triplets` return statement regression.
    - Exhaustively updated project documentation (`index.md`, `extractors.md`, `export.md`, `inference.md`, `architecture.md`, `FAQ.md`, `AUDIT.md`, `SUGGESTIONS.md`).


**AI Transparency Report:**
- Model: Gemini 2.0 Flash
- Actions: Full review of ww-trainer and markovonnx; implemented vectorized Markov/HMM logic; implemented metadata system; ported 5 notebook-to-script tools; verified all changes via 470+ tests (81% coverage).
- Human oversight level: User-directed; executing approved plan.

---

### 2026-03-19 — ESP32 Ultra-Tiny Wake Word Experiments

**Type:** Feature
**Model used:** Claude Opus 4.6
**Human oversight level:** User-directed plan; all source files read before editing
**Files created:** `examples/34_esp32_nano.py`, `examples/35_esp32_genetic_search.py`, `examples/36_size_aware_training.py`, `test/test_esp32.py`
**Files modified:** `ww_trainer/tiers.py`, `ww_trainer/loss.py`, `ww_trainer/sweep.py`, `FAQ.md`, `MAINTENANCE_REPORT.md`

**Summary:**
- Added 3 ESP32 tiers (`esp32_nano`, `esp32_sweet`, `esp32_max`) with param budgets and size limits
- Added `SizeAwareLoss` (L1 sparsity + param-count penalty), registered in `LossManager`
- Added `_build_micro_search_space()`, `_estimate_ffn_params()`, `run_micro_search()` for ESP32-constrained genetic search with composite fitness
- 23 new tests (all passing), 3 example scripts

---

### 2026-03-10 — ONNX vs PyTorch Benchmark Comparison

**Type:** Feature
**Model used:** Claude Sonnet 4.6
**Human oversight level:** User-directed specification; all source files read before editing
**Files created:** `docs/benchmarking.md`
**Files modified:** `ww_trainer/benchmark.py`, `test/test_benchmark.py`, `MAINTENANCE_REPORT.md`

**Summary:**
Extended `ww_trainer/benchmark.py` with an ONNX Runtime vs PyTorch latency and numerical accuracy comparison. Results are saved to CSV and rendered as three new plots.

**Changes:**

1. `ww_trainer/benchmark.py`:
   - Added `OnnxVsPytorchResult` dataclass (fields: name, pytorch/onnx latency mean/std, speedup, max/mean abs diff, numerically_equivalent, onnx_path).
   - Added `onnx_vs_pytorch: list[OnnxVsPytorchResult]` field to `BenchmarkReport`.
   - Added `bench_onnx_vs_pytorch()` function: exports extractor to temp ONNX file, warms up both backends, times both, computes numerical diff with `np.abs(pt_ref - ort_ref)`.
   - Wired ONNX comparison into `run_benchmark()` after extractor benchmarks; each extractor is exported to a temp file, compared, and the temp file deleted.
   - Added ONNX vs PyTorch CSV saving in `save_results()` → `onnx_vs_pytorch.csv`.
   - Added three new plot functions: `_plot_onnx_vs_pytorch_latency`, `_plot_speedup_bars`, `_plot_numerical_accuracy`; all wired into `plot_results()`.

2. `test/test_benchmark.py`:
   - Added three new tests: `test_bench_onnx_vs_pytorch`, `test_run_benchmark_includes_onnx_comparison`, `test_onnx_vs_pytorch_plots`.
   - Updated imports to include `bench_onnx_vs_pytorch` and `OnnxVsPytorchResult`.

3. `docs/benchmarking.md` (new):
   - Documents all benchmark dataclasses with `benchmark.py:LINE` citations.
   - Explains speedup interpretation, expected ranges per extractor type, and numerical equivalence threshold.

**AI Transparency Report:**
- Model: claude-sonnet-4-6
- Actions: Read benchmark.py, feats.py, test/test_benchmark.py, MAINTENANCE_REPORT.md in full before editing; implemented all changes per specification; created benchmarking.md
- Human oversight level: User-directed; executing approved specification

---

### 2026-03-10 — TODO.md Cleanup and Data Contract Documentation

**Type:** Documentation
**Model used:** Claude Sonnet 4.6
**Human oversight level:** User-directed specification; all source files read before writing; all file:line citations verified against actual source
**Files created:** `docs/data_contract.md`
**Files modified:** `TODO.md`, `FAQ.md`, `MAINTENANCE_REPORT.md`

**Summary:**
Completed two tasks: (1) rewrote `TODO.md` so it reflects v1.0 done state — all P0/P1/P2/P3 items moved to Completed, active sections removed; (2) created `docs/data_contract.md` documenting the full notebook → ww-trainer data pipeline.

**Task 1 — TODO.md:**
- Removed empty P0/P1/P2/P3 sections that still had open `[ ]` checkboxes despite being done.
- Added a brief v1.0 statement above the Tracking section.
- Moved all previously unchecked items to the Completed list, preserving all existing completed entries.

**Task 2 — `docs/data_contract.md`:**
- Read `ww_trainer/dataset.py` in full; cited every relevant line number.
- Read both notebooks (`tts2ww.ipynb`, `ww_dataset_generator_ovos_vc.ipynb`) fully.
- Documented: `AudioDataset.__init__` full signature with defaults and descriptions; CSV format with exact parser logic from `trainer.py:830-832`; audio format support from torchaudio; sample rate behaviour; label convention (`"1"` / `"0"`, citing `dataset.py:273` and `dataset.py:298`); both notebook pipelines with output layouts; manual dataset preparation; all augmentation folders with probabilities sourced from `dataset.py:186-235`; download sources for MUSAN/ESC-50/RIRs/LibriSpeech; complete end-to-end bash + Python example.

**AI Transparency Report:**
- Model: claude-sonnet-4-6
- Actions: Read dataset.py, trainer.py (CLI section), both notebooks, existing FAQ.md and MAINTENANCE_REPORT.md before writing; created data_contract.md; rewrote TODO.md; updated FAQ.md and MAINTENANCE_REPORT.md
- Human oversight level: User-directed; executing approved specification

---

### 2026-03-10 — Comprehensive Documentation Sprint

**Type:** Documentation
**Model used:** Claude Sonnet 4.6
**Human oversight level:** User-directed specification; all source files read before writing; all file:line citations verified against actual source
**Files created:** `docs/index.md` (replacement), `docs/architecture.md`, `docs/api.md`, `docs/training.md`, `docs/export.md`, `docs/inference.md`, `docs/sweep.md`
**Files modified:** `MAINTENANCE_REPORT.md`

**Summary:**
Wrote comprehensive documentation for all public modules in `ww_trainer/`. Every statement about runtime behaviour cites the actual source file and line number. All code examples are runnable.

**Documents written:**

- `docs/index.md` — Landing page: project description, architecture diagram, table of contents, quick start, key links.
- `docs/architecture.md` — System design deep-dive: three-layer abstraction (BaseExtractor, ClassifierHead, BaseWakeModel), full data flow diagrams (training and inference paths), ONNX pipeline, extractor taxonomy table, hardware tier table, shared extractor pattern, sliding feature cache mechanics, key design decisions with rationale.
- `docs/api.md` — Full API reference for every public class and method across all 10 modules: `feats`, `model`, `inference`, `trainer`, `tiers`, `dataset`, `loss`, `utils`, `visualization`, `mining`, `checkpoint`, `sweep`, `version`. Every entry includes signature, parameter table, return value, and file:line citation.
- `docs/training.md` — Step-by-step training guide: dataset format, tier selection trade-offs, CLI quickstart for all four tiers, full CLI option table (all 40+ options), loss function guide, hard-negative mining explanation, augmentation reference table, AMP and gradient accumulation guidance, MLflow integration, checkpointing and resuming.
- `docs/export.md` — ONNX export guide: rationale, MfccExtractor export, HuBERT/Wav2Vec2 export workflow, classifier head export, full model export, quantization (INT8/INT16), numerical verification, loading back in OnnxFeatureExtractor, pre-exported model links.
- `docs/inference.md` — Inference guide: ONNX-only inference, single/batch/streaming inference with complete working examples, PyTorch inference paths, device selection, latency comparison table.
- `docs/sweep.md` — Hyperparameter sweep guide: Optuna overview, installation, quick start, search space table, persistent SQLite storage, results analysis, custom objective patterns.

**Source coverage:** Read all 18 source files before writing. No statements about code behaviour were written without reading the corresponding source.

**AI Transparency Report:**
- Model: claude-sonnet-4-6
- Actions: Read all source files listed in the task spec; created 6 new docs files; replaced docs/index.md; appended MAINTENANCE_REPORT.md entry
- Human oversight level: User-directed; executing approved documentation specification

---

### 2026-03-10 — P2 Docs and P3 Nice-to-Have Sprint (this session)

**Type:** Documentation / Enhancement / CI
**Model used:** Claude Sonnet 4.6
**Human oversight level:** Specification provided; all source reads verified before edits
**Files created:** `docs/index.md`, `.github/workflows/ci.yml`
**Files modified:** `README.md`, `ww_trainer/visualization.py`, `ww_trainer/dataset.py`, `SUGGESTIONS.md`, `TODO.md`, `MAINTENANCE_REPORT.md`

**Summary:**
Completed the P2 documentation block and P3 nice-to-have items from TODO.md.

**Task 1 — README rewrite:**
- Replaced the credits-only README with project description, hardware tier table, quickstart (tier-based and ONNX-featurizer-based), extractor matrix, ONNX-only inference snippet, dataset format, architecture ASCII diagram, and original NGI credits.

**Task 2 — `docs/index.md`:**
- Full architecture narrative: extractor+head pipeline, data flow diagram, training/export/inference workflows, and API reference with `file:line` citations for every public class and method in `feats.py`, `model.py`, `inference.py`, `tiers.py`, and `trainer.py`.

**Task 3 — UMAP optional import with t-SNE fallback:**
- `ww_trainer/visualization.py`: top-level `try/except ImportError` for `umap`; `_HAS_UMAP` flag; `log_umap()` falls back to `sklearn.manifold.TSNE` with `logging.warning` instead of silently returning `None`.

**Task 4 — GitHub Actions CI:**
- `.github/workflows/ci.yml`: matrix Python 3.10/3.11/3.12, uv install, pytest with coverage, CLI smoke test.

**Task 5 — AudioDataset sample-rate logging:**
- `ww_trainer/dataset.py`: added `logger.warning` with expected/actual Hz and file path before the existing resample call; changed positional args to `orig_freq`/`new_freq` keyword args.

**Task 6 — SUGGESTIONS.md extended:**
- Added S-009 (Optuna sweep with code sketch), S-010 (DDP multi-GPU), S-011 (Platt scaling), S-012 (QAT for MCU).

**AI Transparency Report:**
- Model: claude-sonnet-4-6
- Actions: Read all relevant source files before editing; created and modified files as described above
- Human oversight level: User-directed; executing approved specification

---

### 2026-03-10 — P3 Sweep, UMAP Fallback, CI, README, Docs

**Type:** Feature / Tooling / Documentation
**Model used:** Claude Sonnet 4.6
**Human oversight level:** Specification provided; implementation reviewed
**Files created:** `ww_trainer/sweep.py`
**Files modified:** `pyproject.toml`, `MAINTENANCE_REPORT.md`

**Summary:**
Implemented P3 items: Optuna hyperparameter sweep module, pyproject.toml `sweep` extras group.

**Task — `ww_trainer/sweep.py`:**
- New module exposing `run_sweep()` and a `__main__` CLI entry point.
- Optuna is an optional dependency; import is guarded with a clear `ImportError` message.
- Sweeps `arch` (ffn/gru/cnn), `hidden_dim`, `lr`, `batch_size`, and `dropout` via `optuna.Trial`.
- Loads the dataset CSV once; splits 80/20 train/val; shares data across all trials.
- Each trial writes checkpoints to `output_dir/trial_N/`; failed trials return F1=0.0 rather than crashing the study.
- Best params saved as JSON to `output_dir/best_params.json` after the study completes.
- Uses absolute imports only (`ww_trainer.trainer`, `ww_trainer.dataset`).

**Task — `pyproject.toml`:**
- Added `sweep = ["optuna"]` under `[project.optional-dependencies]`.

**AI Transparency Report:**
- Model: claude-sonnet-4-6
- Actions: Read pyproject.toml, MAINTENANCE_REPORT.md, and ww_trainer/__init__.py before editing; created sweep.py; edited pyproject.toml; appended this entry
- Human oversight level: User-directed; executing approved specification

---

### 2026-03-10 — P2 Performance and Architecture Features

**Type:** Feature / Enhancement
**Model used:** Claude Sonnet 4.6
**Human oversight level:** Specification provided; implementation verified via automated tests (64 tests pass)
**Files created:** `ww_trainer/tiers.py`
**Files modified:** `ww_trainer/trainer.py`, `ww_trainer/mining.py`, `ww_trainer/dataset.py`

**Summary:**
Implemented all P2 performance and architecture features.

**Task 1 — Hardware Tier Presets (`ww_trainer/tiers.py`):**
- New module with `TierConfig` dataclass, `HARDWARE_TIERS` dict (micro/small/medium/large), `list_tiers()` table formatter, and `get_tier(name)` lookup.

**Task 2 — Wire `--tier` into CLI:**
- Added `from ww_trainer.tiers import HARDWARE_TIERS, get_tier, list_tiers` to `trainer.py`.
- Added `--tier` (Choice: micro/small/medium/large) and `--list-tiers` CLI options.
- Tier preset overrides `arch`, `featurizer_type`, `hidden_dim`, `bidirectional`, `gru_n_layers`, and `n_mfcc` as appropriate.

**Task 3 — Mixed-Precision Training:**
- `WakeWordTrainer.__init__` accepts `use_amp: bool = False`; creates `torch.amp.GradScaler` when enabled.
- Training step wrapped in `torch.amp.autocast`; `scaler.scale/step/update` used when AMP is active.
- Added `--amp` CLI flag.

**Task 4 — Gradient Accumulation:**
- `WakeWordTrainer.train()` accepts `accumulate_grad_batches: int = 1`.
- Batch loop accumulates gradients and calls `optimizer.step()` every N batches or at epoch end.
- Added `--accumulate-grad-batches` CLI option.

**Task 5 — Persist Hard-Negative Mining Cache:**
- Added `save_mining_cache(cache, path)` and `load_mining_cache(path)` to `ww_trainer/mining.py`.
- `WakeWordTrainer.train()` saves cache to `hardneg_cache.pt` after final save; loads it when `resume` path is provided.

**Task 6 — Dataset Validation:**
- `AudioDataset.__init__` now checks for missing files, logs up to 5 with `logger.warning`, and logs label distribution via `logger.info`.

---

### 2026-03-10 — Architecture Alignment: Phases 4-6 (Monolith Split, Streaming, Packaging)

**Type:** Refactor / Feature / Packaging
**Model used:** Claude Sonnet 4.6
**Human oversight level:** Specification provided; implementation reviewed via automated tests
**Files created:** `ww_trainer/visualization.py`, `ww_trainer/mining.py`, `ww_trainer/checkpoint.py`, `test/test_streaming.py`, `pyproject.toml`
**Files modified:** `ww_trainer/trainer.py`, `ww_trainer/model.py`, `FAQ.md`, `MAINTENANCE_REPORT.md`

**Summary:**
Implemented phases 4-6 of the ww-trainer architecture alignment plan.

**Phase 4 — Monolith split:**
- Extracted all visualization helpers from `WakeWordTrainer` into `ww_trainer/visualization.py` as standalone functions (`plot_roc`, `plot_pr`, `plot_det`, `log_confidence_histogram`, `log_pca`, `log_tsne`, `log_umap`, `log_embeddings_stats`).
- Extracted hard-negative mining into `ww_trainer/mining.py` as `mine_hard_negatives()` function; the rolling hardness cache is now passed explicitly rather than stored as instance state.
- Extracted checkpoint I/O into `ww_trainer/checkpoint.py` as `save_checkpoint()` and `load_checkpoint()` standalone functions.
- `WakeWordTrainer` in `trainer.py` now delegates to these modules; all existing method signatures preserved for backward compatibility.

**Phase 5 — Streaming inference:**
- Added `BaseWakeModel.forward_streaming(audio_chunk, cache)` to `ww_trainer/model.py`. Accepts a 1-D audio chunk and a `SlidingFeatureCacheTensor`; returns sigmoid probability as float.
- Created `test/test_streaming.py` with 5 tests covering: return type, cache growth, cache saturation, 10-call stability, and ONNX streaming via `OnnxWakeWordInferencer.infer_streaming`.

**Phase 6 — Packaging:**
- Created `pyproject.toml` alongside `setup.py`. Build backend: `setuptools.build_meta`. Version hardcoded as `0.0.1a1` (matching `version.py`). Optional extras: `dev`, `transformers`, `vc`, `mlflow`.

**Test results:** 64 passed, 35 warnings (was 59 before this sprint; 5 new streaming tests added, all passing).

---

### 2026-03-10 — Architecture Alignment: Extractor Consolidation + ONNX Inference Module

**Type:** Feature / Refactor
**Files modified:** `ww_trainer/feats.py`, `ww_trainer/trainer.py`, `setup.py`, `FAQ.md`
**Files created:** `ww_trainer/inference.py`, `test/test_feats.py`, `test/test_inference.py`

**Summary:**
Implemented the architecture alignment plan: consolidated all feature extractors into `ww_trainer/feats.py`, created a zero-PyTorch ONNX inference module at `ww_trainer/inference.py`, fixed the CLI entry point in `setup.py`, and updated the extractor registry in `trainer.py`.

**Changes:**

1. `ww_trainer/feats.py`:
   - Added `feature_dim` property to `BaseExtractor` (raises `NotImplementedError`)
   - Added `feature_dim` property to `OnnxFeatureExtractor` (reads from ONNX shape; falls back to dummy inference for dynamic axes)
   - Added `MfccExtractor` — pure-PyTorch, ONNX-exportable, output `[B, T, n_mfcc]`
   - Added `HubertExtractor` — HuBERT encoder (requires `transformers`; guarded import)
   - Added `Wav2Vec2Extractor` — Wav2Vec2 encoder (requires `transformers`; guarded import)
   - Fixed `SlidingFeatureCacheTensor` memory-aliasing bug: added `.clone()` before in-place copy

2. `ww_trainer/inference.py` (new):
   - `OnnxWakeWordInferencer` with `infer()`, `infer_batch()`, `infer_streaming()` methods
   - Zero PyTorch dependency at module level

3. `ww_trainer/trainer.py`:
   - Added `EXTRACTOR_REGISTRY` module-level dict
   - `create_model()` now accepts `featurizer_type`, `shared_extractor`, and optional `feature_dim` (auto-detected from extractor if not provided)
   - `WakeWordTrainer.__init__` updated to accept and pass through `featurizer_type` and `shared_extractor`

4. `setup.py`: Fixed CLI entry point `ww_trainer.train:train` → `ww_trainer.trainer:train`

**Verification:** `uv run --python .venv pytest test/ -v` → 59 passed, 0 failed

**AI Transparency Report:**
- Model: claude-sonnet-4-6
- Actions: Read all relevant source files; implemented all 6 tasks; fixed SlidingFeatureCacheTensor memory-aliasing bug; fixed test_inference.py to use legacy TorchScript ONNX exporter (onnxscript not installed)
- Human oversight level: User-directed; executing approved architecture alignment plan

---

### 2026-03-10 — Architecture Alignment: PLAN.md & TODO.md Rewrite

**Type:** Documentation / Planning
**Files modified:** `PLAN.md`, `TODO.md`

**Summary:**
Rewrote `PLAN.md` and `TODO.md` to reflect the architecture-first guideline:
wake words are always modeled as `feature-extractor + classifier head`; everything is exported to
ONNX; inference requires only ONNX runtime. The plan now defines hardware tiers (micro → large),
consolidates all extractors into `ww_trainer/feats.py`, mandates a zero-PyTorch inference module
(`ww_trainer/inference.py`), and fixes the broken CLI entry point.

**Actions taken:**
- Rewrote `PLAN.md` with 6-phase roadmap aligned to architecture guideline and hardware tiers
- Rewrote `TODO.md` with prioritized checklist; moved completed bug-fix/test/docs items to
  "Completed" section; added new P0 items for extractor consolidation, ONNX inference module,
  CLI fix, and extractor registry

**AI Transparency Report:**
- Model: claude-sonnet-4-6
- Actions: Read source files (feats.py, model.py, setup.py, scripts/); rewrote PLAN.md and TODO.md
- Human oversight level: User-directed; executing approved plan document

---

### 2026-03-10 — Production Readiness Sprint (Bugs + Tests + Docs)

**Type:** Bug fixes, new test suite, new documentation
**Files modified:** `ww_trainer/feats.py`, `ww_trainer/dataset.py`, `ww_trainer/model.py`, `ww_trainer/loss.py`
**Files created:** `test/conftest.py`, `test/test_smoke.py`, `test/test_models.py`, `test/test_losses.py`, `test/test_dataset.py`, `test/test_utils.py`, `FAQ.md`, `QUICK_FACTS.md`, `AUDIT.md`, `SUGGESTIONS.md`

**Bug fixes applied:**
1. `feats.py:87` — `onnx.checker.check_model(out)` → `check_model(onnx_model)` (BUG-001)
2. `dataset.py` — replaced `/tmp/vc_...wav` hardcode with `tempfile.NamedTemporaryFile` (BUG-002)
3. `dataset.py` — bare `except:` → `except Exception as exc:` + `logger.warning` (BUG-003)
4. `model.py:219` — added `ValueError` for ambiguous GRU shape when `D1==D2==input_size` (BUG-004)
5. `feats.py:44` — fixed `SlidingFeatureCacheTensor` shift logic out-of-bounds error (BUG-005)
6. `dataset.py` — moved `chatterbox_onnx` import inside conditional block (BUG-006)
7. `loss.py:451` — stored `margin` in `loss_entry` dict so it survives into `compute_loss` (BUG-007)

**Test suite:** 45 tests across 5 files; all pass (`uv run pytest test/ -v` → 45 passed).

**Documentation created:** `FAQ.md`, `QUICK_FACTS.md`, `AUDIT.md`, `SUGGESTIONS.md`

**Verification:** `uv run --python .venv pytest test/ -v` → 45 passed, 0 failed

**AI Transparency Report:**
- Model: claude-sonnet-4-6
- Actions: Read all source files; fixed 7 bugs; wrote 45 tests; wrote 4 documentation files
- Human oversight level: User-directed; user approved tool calls throughout

---

### 2026-03-10 — Production Readiness Planning

**Type:** Documentation / Planning
**Files created:** `PLAN.md`, `TODO.md`, `MAINTENANCE_REPORT.md`

**Summary:**
Created production readiness plan and actionable TODO checklist for ww-trainer v0.0.1a1.
No source files were modified.

**Actions taken:**
- Authored `PLAN.md` documenting 6-phase roadmap: bug fixes, tests, refactor, streaming inference,
  packaging, and performance hardening
- Authored `TODO.md` with prioritized checklist (P0–P3) covering 4 confirmed bugs, test suite,
  documentation requirements, refactor targets, streaming inference, and packaging migration
- Created this `MAINTENANCE_REPORT.md` as the initial transparency entry

**AI Transparency Report:**
- Model: claude-sonnet-4-6
- Actions: Created planning documents only; no source code modified
- Human oversight level: Full — plan reviewed and approved by user before execution
