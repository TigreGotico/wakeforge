# ww-trainer — FAQ

## Genetic Search — Advanced (wakegp-inspired)

**Q: What does `SEARCH_TWO_STAGE` do?**

Runs a coarse global search (stage 1) then seeds a focused fine-tune round (stage 2) with the top-K configs from stage 1. This improves final quality by first exploring broadly then refining the best region. See `run_two_stage_genetic_search` — `ww_trainer/sweep.py:662`. Each history entry in the merged result carries a `stage` field (1 or 2).

**Q: When should I use `SEARCH_DEMES > 1`?**

On multi-core CPUs or multi-GPU machines. Each deme is an independent GA population run in parallel, preventing premature convergence. Use 2–4 demes; each deme uses roughly one CPU core. Each deme writes to `output_dir/deme_{id}/` so trial files are isolated. See `run_genetic_search` (`n_demes` parameter) — `ww_trainer/sweep.py:543`.

**Q: What is `SEARCH_FITNESS_FN` and which value should I pick?**

Controls selection pressure. `f1` (default) uses raw F1. `exp_f1` steepens the gradient and helps when F1 plateaus above 0.9. `double_exp_f1` applies extreme pressure near the optimum. Start with `f1`; switch to `exp_f1` if search stagnates at high F1. The fitness transform is applied **only for selection**; `best_score` in the result is always raw F1. Unknown `fitness_fn` values raise `ValueError` — valid values: `"f1"`, `"exp_f1"`, `"double_exp_f1"`. See `_validate_ga_params` — `ww_trainer/sweep.py:24`.

**Q: What fields does each history entry contain?**

Each entry in `run_genetic_search` / `_run_deme` history: `{"generation": int, "best": float, "avg": float, "elapsed_seconds": float}`. In two-stage results, an additional `"stage"` key (1 or 2) is added — `sweep.py:774-778`.

---

## Notebooks & Cloud Training

**Q: How do I run the genetic search notebook on Kaggle?**

1. Upload or link the repo to a Kaggle dataset/notebook.
2. Add secrets in **Add-ons → Secrets**: `WAKE_WORD`, `OUTPUT_DIR` (e.g. `/kaggle/working/ww_output`), and any other config vars.
3. Open `notebooks/genetic_search.ipynb`; the Config cell reads all vars via `os.environ.get(...)`.
4. Run all cells. The notebook auto-detects the Kaggle platform and installs dependencies.
5. Outputs (ONNX models, evolution plot, benchmark PNGs) are written to `OUTPUT_DIR` and available in the Kaggle output panel.

**Q: Which env vars control the genetic search notebook?**

See the configuration table in `notebooks/genetic_search.ipynb` cell 1, or `docs/index.md`. Key vars: `WAKE_WORD`, `POPULATION`, `GENERATIONS`, `TIERS_TO_TRAIN`, `FINAL_EPOCHS`. All have safe defaults so the notebook runs without any env vars set.

**Q: How do I skip dataset regeneration on repeated runs (resume safety)?**

Cell 4 is fully resume-safe across all three dataset modes:
- **BYO CSV**: no datagen at all; the optional 80/20 split is written once to `OUTPUT_DIR/dataset_split/` and reused.
- **HF / Auto**: `reuse_dataset=True` is always passed — if `dataset/train/metadata.csv` and `dataset/test/metadata.csv` exist under `OUTPUT_DIR`, datagen is skipped. See `_run_or_load_datagen` — `quickstart.py:115`.

**Q: How do I use my own dataset instead of generating one?**

Set `CUSTOM_TRAIN_CSV=/absolute/path/to/metadata.csv` (format: `path,label`, no header). Optionally set `CUSTOM_TEST_CSV`; if absent, the notebook splits 80/20 and writes the split to `OUTPUT_DIR/dataset_split/` (idempotent). No TTS or HF downloads occur in this mode.

**Q: How do I force a specific HuggingFace dataset for positives?**

Set `HF_DATASET=org/repo-name` (e.g. `HF_DATASET=OpenVoiceOS/hey-jarvis-dataset`). This overrides `find_positive_dataset` auto-detection and downloads positives from the specified repo. Negatives and augmentation proceed normally.

**Q: How do I supply my own negative (not-wake-word) audio?**

Set `NEGATIVES_DIR=/path/to/neg_audio/`. The directory is used directly as the negatives source — no HF download for general negatives occurs. Works in all three dataset modes.

**Q: How do I add extra HF repos to the negatives without replacing the built-ins?**

Set `EXTRA_NEGATIVES_HF=org/repo1,org/repo2`. The listed repos are appended to `NEGATIVE_DATASETS["general"]` before datagen runs, so built-in repos are preserved. The same pattern applies to augmentation categories: `EXTRA_BG_NOISE_HF`, `EXTRA_MUSIC_HF`, `EXTRA_RIR_HF`.

**Q: How do I use local audio for bg-noise / music / RIR augmentation?**

Set one or more of `BG_NOISE_DIR`, `MUSIC_DIR`, `RIR_DIR` to local directory paths. After datagen, the corresponding fields on `DatagenResult` are overridden to point at those directories. HF downloads for that category are skipped. The trainer then picks up the local files for augmentation. You can mix: e.g. local `BG_NOISE_DIR` with HF `EXTRA_MUSIC_HF`.

## Inspiration Features (precise-lite-trainer)

**Q: How do I cache extracted features between training runs?**

Use `--feature-cache-dir .feature_cache/` (default). Un-augmented waveforms are cached as `.npy` files keyed by MD5(file content + extractor identity). Cache is bypassed when augmentation is active. Disable with `--no-feature-cache`. See `FeatureCache` — `cache.py`.

**Q: How do I freeze layers for transfer learning / fine-tuning?**

Use `--freeze-extractor` to freeze the entire feature extractor, and/or `--freeze-layers N` to freeze the first N classifier parameters. Add `--unfreeze-at-epoch N` for progressive unfreezing (frozen for N epochs, then all layers train). See `WakeWordTrainer._freeze` — `trainer.py`.

**Q: What is epoch-level data replacement?**

`--replacement-ratio 0.4` randomly drops 40% of epoch data and replaces it with fresh samples from the full pool each epoch, reducing overfitting. `--balanced-replacement` ensures 50/50 wake/nonwake in the replaced portion. Complements hard-negative mining.

**Q: What is the composite fitness score?**

`compute_fitness_score()` (`evaluation.py`) combines detection quality and model size into a single metric: `(1 - 0.8*FP_rate - 0.2*FN_rate) * size_penalty`. FP is penalized 4× more than FN. Enable best-fitness checkpointing with `--fitness-checkpoint`. Set `--fitness-param-budget` for the size penalty threshold.

**Q: Is `--feature-cache-dir` actually wired up?**

Yes (as of 2026-03-19). After the trainer is created, a `FeatureCache` is instantiated from the extractor's name/feature_dim/sample_rate and stored in `trainer.augment_opts["feature_cache"]`. Every training-epoch `AudioDataset` receives it automatically via the loop. Disable with `--no-feature-cache`. See `cli.py` and `ww_trainer/cache.py`.

**Q: Does `SileroVadWrapper.__init__` still download the model at construction?**

No. The `torch.hub.load()` call was moved to the first `forward()` call (lazy init). Construction is now always fast and offline-safe. If you provide `onnx_path`, the ONNX session is still loaded eagerly (local file only). See `feats.py:SileroVadWrapper`.

---

**Q: Where does the epoch training loop live?**

The full training loop was extracted from `WakeWordTrainer.train()` into `ww_trainer/loop.py:training_loop()`. `WakeWordTrainer.train()` is now a thin 4-line delegate. `trainer.py` is 246 lines. Related helpers in `loop.py`: `_build_epoch_data`, `_run_batch_loop`, `_update_best_checkpoints`, `_log_fp_fn_artifacts`.

**Q: Where is `compute_readiness` (hard-negative readiness score)?**

Moved to `ww_trainer/evaluation.py:compute_readiness()`. Previously it was a private method on `WakeWordTrainer`.

**Q: Where is `save_intermediate_checkpoint`?**

Moved to `ww_trainer/checkpoint.py:save_intermediate_checkpoint()`. `WakeWordTrainer.save_intermediate_ckpt()` is now a thin wrapper for backwards compatibility.

---

## New Modules (v1.2)

**Q: How do I export an FFN model to C for ESP32?**

Use `ww_trainer.export_c.export_to_c_header(model, "model.h")`. Generates a self-contained `.h` with int8 weights and a `ww_model_infer()` function. Only supports FFN heads.

**Q: How do I calibrate model probabilities?**

Use `ww_trainer.calibration.calibrate_model(model, val_data, output_dir)` after training. Fits Platt scaling on validation logits and saves `calibration.json`. Apply with `apply_platt_scaling(logits, params)`.

**Q: How do I use QAT (quantization-aware training)?**

`from ww_trainer.qat import prepare_qat, convert_qat`. Call `prepare_qat(model)` before training and `convert_qat(model)` after. Note: `torch.ao.quantization` is deprecated in PyTorch 2.10+; migration to `torchao` is planned.

**Q: How do I use multi-GPU training?**

Use `ww_trainer.ddp` utilities and launch with `torchrun --nproc_per_node=N`. Call `setup_ddp()`, `wrap_model_ddp(model, local_rank)`, `create_distributed_loader(dataset, batch_size)`.

**Q: How do I use HuBERT without the transformers library?**

Use `TorchAudioHubertExtractor(bundle_name="HUBERT_BASE")` from `ww_trainer.feats`. Only requires `torchaudio`.

**Q: How do I validate my dataset before training?**

Pass `validate=True` to `AudioDataset(samples, validate=True)`. Checks all files are readable audio. Class imbalance warnings (>10:1 ratio) are always active.

---

**Q: How do I run the smoke tests?**

```bash
.venv/bin/python -m pytest test/smoketests/ -v
```

100 tests covering all extractor×head×loss pipelines with dummy data. Runs in ~25 seconds.

---

## ESP32 / Ultra-Tiny Models

**Q: What are the ESP32 tiers?**

Three tiers targeting ESP32 (520 KB RAM, 4 MB flash): `esp32_nano` (sub-1KB, ≤1024 params), `esp32_sweet` (sub-10KB, ≤10240 params), `esp32_max` (sub-50KB, ≤51200 params). All use MFCC + FFN. See `tiers.py`.

**Q: What is the smallest possible model?**

MFCC-13 + FFN-8 = 97 params = 0.1 KB int8. The MFCC-13 + FFN-16 = 241 params (0.2 KB) is the nano tier default.

**Q: What is SizeAwareLoss?**

A loss wrapper (`loss.py:SizeAwareLoss`) that adds L1 sparsity + param-count penalties to any base loss. Use `{"name": "size_aware", "param_budget": 1024}` in `losses_cfg`.

**Q: How does the micro genetic search work?**

`sweep.run_micro_search()` uses composite fitness = `accuracy_weight * f1 + size_weight * (1 - params/budget)`. Models exceeding the tier's param budget are hard-rejected. Search space is auto-constrained per tier.

---

## Installation

**Q: How do I install ww-trainer?**

```bash
uv pip install -e .
```

For development (tests + coverage):

```bash
uv venv .venv
uv pip install --python .venv -e . torch torchaudio onnx onnxruntime librosa soundfile pytest pytest-cov
```

**Q: What Python version is required?**

Python 3.10 or newer. Tested on 3.13.

**Q: What CUDA version is required for GPU training?**

CUDA 11.8+ is recommended. The code auto-selects CUDA if available; falls back to CPU otherwise. All CI tests run on CPU.

---

## ONNX Export & Metadata

**Q: How do I export my model to ONNX?**

Use the `--export-onnx` flag during training. The trainer will automatically export the best model (F1, Loss, etc.) and the final model to `.onnx` files in your output directory. You can also manually call `model.export_to_onnx("model.onnx")`. All `export_to_onnx` methods accept an optional `metadata: dict` parameter to embed key-value pairs into the ONNX file.

**Q: What metadata is embedded in the ONNX files?**

All exported models now contain training context in their `metadata_props`:
- `wake_word`: The target keyword name.
- `arch`: Head architecture (GRU, CNN, FFN).
- `epoch`: Save checkpoint epoch.
- `featurizer`: Class name of the feature extractor.
- `metric_*`: Performance metrics (F1, Precision, Recall, Loss) at save time.

Use `onnx.load("model.onnx").metadata_props` to inspect.

**Q: Are Markov and HMM extractors ONNX-compatible?**

Yes. `MarkovTransitionExtractor` uses a vectorized state lookup that traces correctly to ONNX. `HMMStateExtractor` also supports ONNX export. Both export the entire pipeline (Base Extractor + Markov/HMM Wrapper) as a single symbolic graph.

**Q: Why is there a small numerical difference between PyTorch and ONNX?**

The STFT implementation in PyTorch and its ONNX equivalent may have minor precision differences (~1e-4 absolute) due to internal windowing and FFT algorithms. This is normal and rarely affects wake-word detection accuracy.

---

## Blackbox ONNX Featurizers

**Q: Can I use a pre-trained ONNX model as a feature extractor?**

Yes. Use the `OnnxFeatureExtractor` class or the `--featurizer-type onnx` CLI flag. This treats the ONNX model as a "blackbox" that takes raw audio and outputs features.

**Q: How do I create a Markov blackbox featurizer?**

Use the provided script:
```bash
python scripts/train_markov_featurizer.py --wake-folder ./my_wakes --out markov.onnx
```
Then use it for training different heads:
```bash
ww_trainer-train --featurizer markov.onnx --featurizer-type onnx --model-type gru ...
```

---

## Dataset Generation

**Q: What format does `AudioDataset` expect?**

Pass a list of `(path, label)` tuples where:
- `path` is an absolute path to a `.wav`, `.flac`, `.mp3`, `.m4a`, or `.ogg` file
- `label` is a string `"1"` (wake word) or `"0"` (non-wake)

```python
samples = [
    ("/data/hey_jarvis_001.wav", "1"),
    ("/data/background_001.wav", "0"),
]
```

**Q: How do I prepare a dataset from a folder?**

Use the notebooks in `notebooks/ww/tts2ww.ipynb` (8-stage synthetic dataset factory) or `ww_dataset_generator_ovos_vc.ipynb`. Both produce CSV/folder layouts compatible with `AudioDataset`.

**Q: What audio sample rate does the model expect?**

16 000 Hz. `AudioDataset` resamples automatically using torchaudio if the source differs.

---

## Training

**Q: How do I start a training run?**

```bash
uv run ww_trainer-train --help
```

Key flags: `--wake-folder`, `--non-wake-folder`, `--model-type {ffn,cnn,gru}`, `--epochs`, `--batch-size`.

**Q: What loss functions are supported?**

| Name | Class | Notes |
|------|-------|-------|
| `bce` | `BCEWithLogitsLoss` | Standard binary cross-entropy |
| `triplet` | `TripletMarginLoss` | PyTorch built-in |
| `soft_triplet` | `SoftTripletLoss` | Log(1 + exp(D_ap - D_an)) |
| `pair` | `MarginRankingLoss` | Hard-pair distances |
| `cn2pair` | `CN2Plus1PairLoss` | López-Espejo et al. TASLP 2021 |
| `lse` | `LiftedStructureLoss` | Log-sum-exp over all pairs |
| `contrastive` | `ContrastiveLoss` | Classic Siamese loss |
| `angular` | `AngularLoss` | Cosine-based margin |
| `rppl` | `RobustProtoDiversityLoss` | Composite prototype loss (recommended) |

**Q: Which model architecture should I use?**

- `ffn` — fastest, good baseline
- `gru` — best temporal modelling, recommended for production
- `cnn` — good for fixed-length inputs

---

## ONNX Export

**Q: How do I export a trained model to ONNX?**

```python
model.export_to_onnx("classifier_head.onnx")
# For featurizer too:
model.export_to_onnx("classifier_head.onnx", export_featurizer=True)
```

**Q: Can I quantize the exported model?**

Yes — pass `quantize=True` to `export_to_onnx`. Both INT8 and INT16 variants are written automatically.

---

## Common Errors

**Q: `ModuleNotFoundError: No module named 'chatterbox_onnx'`**

`chatterbox_onnx` is an optional dependency for voice-conversion augmentation. It is only required when you set `vc_folder` in `AudioDataset`. Install it separately if needed:

```bash
pip install chatterbox-onnx
```

**Q: `ModuleNotFoundError: No module named 'torchcodec'`**

Required by torchaudio >= 2.5 for audio loading. Install:

```bash
pip install torchcodec
```

**Q: `ValueError: Ambiguous feature shape` in GRU**

The `GruClassifierHead` cannot auto-detect orientation when both spatial dimensions equal `input_size`. Pass features as `[B, T, F]` explicitly instead of `[B, F, T]`.

**Q: Training loss is NaN**

Common causes: (1) learning rate too high, (2) audio files contain silence/very short clips, (3) batch has only one class (no valid triplets). Enable `--debug` for per-step loss logging.

---

## Feature Extractors

**Q: What feature extractors are available?**

| Type | Class | Notes |
|------|-------|-------|
| `onnx` | `OnnxFeatureExtractor` | Load any pre-exported ONNX extractor (default) |
| `mfcc` | `MfccExtractor` | Pure-PyTorch MFCC, fully ONNX-exportable |
| `hubert` | `HubertExtractor` | HuBERT encoder; requires `transformers` |
| `wav2vec2` | `Wav2Vec2Extractor` | Wav2Vec2 encoder; requires `transformers` |

All extractors expose a `feature_dim` property; `WakeWordTrainer.create_model` auto-detects `feature_dim` from the extractor when the argument is omitted.

**Q: How do I use MfccExtractor instead of an ONNX extractor?**

Pass `featurizer_type="mfcc"` to `WakeWordTrainer`:

```python
trainer = WakeWordTrainer(arch="gru", featurizer="", featurizer_type="mfcc")
```

Or use `create_model` directly:

```python
model = WakeWordTrainer.create_model("gru", featurizer="", featurizer_type="mfcc")
```

**Q: How do I query the feature dimension of any extractor?**

```python
from ww_trainer.feats import MfccExtractor, OnnxFeatureExtractor
print(MfccExtractor(n_mfcc=40).feature_dim)      # 40
print(OnnxFeatureExtractor("model.onnx").feature_dim)  # from ONNX shape or dummy run
```

---

## ONNX-Only Inference

**Q: How do I run inference without PyTorch installed?**

Use `OnnxWakeWordInferencer` from `ww_trainer.inference`. It requires only `numpy` and `onnxruntime`:

```python
from ww_trainer.inference import OnnxWakeWordInferencer
import numpy as np

inferencer = OnnxWakeWordInferencer(
    extractor_path="mfcc.onnx",
    head_path="classifier_head.onnx",
    sample_rate=16000,
    device="cpu",
)

audio = np.zeros(16000, dtype=np.float32)
prob = inferencer.infer(audio)          # single clip → float in [0, 1]
probs = inferencer.infer_batch(batch)   # [B, T] array → [B] probabilities

# Streaming with a rolling feature cache
cache = None
for chunk in audio_chunks:
    prob, cache = inferencer.infer_streaming(chunk, cache)
```

**Q: What ONNX files does OnnxWakeWordInferencer need?**

Two files:
1. **Extractor ONNX** — the feature extractor (e.g. exported `MfccExtractor` or a HuBERT/Wav2Vec2 ONNX). Input: `[B, T]` float32. Output: `[B, T_frames, F]` float32.
2. **Head ONNX** — the classifier head exported via `BaseWakeModel.export_to_onnx()`. Input: `[B, T_frames, F]` float32. Output: scalar logit.

---

## MLflow

**Q: How do I enable MLflow tracking?**

Pass `--mlflow-uri http://localhost:5000` (or any valid tracking URI). Metrics, artifacts, and ONNX checkpoints are logged automatically.

---

## Streaming Inference

**Q: How do I run streaming (chunk-by-chunk) inference with PyTorch?**

Use `BaseWakeModel.forward_streaming` together with `SlidingFeatureCacheTensor`:

```python
from ww_trainer.feats import MfccExtractor, SlidingFeatureCacheTensor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel

extractor = MfccExtractor(sr=16000, n_mfcc=13)
head = FfnClassifierHead(input_size=13, hidden_dim=64)
model = BaseWakeModel(feature_extractor=extractor, classifier=head, sample_rate=16000)
model.eval()

cache = SlidingFeatureCacheTensor(feature_dim=13, window_size=50)
for chunk in audio_chunks:           # chunk: 1-D float32 torch.Tensor
    prob = model.forward_streaming(chunk, cache)
    if prob > 0.5:
        print("Wake word detected!")
```

`SlidingFeatureCacheTensor` is updated in-place each call and holds the last `window_size` frames.
`forward_streaming` is defined in `BaseWakeModel` — `ww_trainer/model.py`.

**Q: How do I run ONNX streaming inference without PyTorch?**

Use `OnnxWakeWordInferencer.infer_streaming`. It manages a rolling numpy cache internally:

```python
from ww_trainer.inference import OnnxWakeWordInferencer
import numpy as np

inferencer = OnnxWakeWordInferencer("mfcc.onnx", "head.onnx", sample_rate=16000)
cache = None
for chunk in audio_chunks:           # chunk: 1-D float32 numpy array
    prob, cache = inferencer.infer_streaming(chunk, cache)
    if prob > 0.5:
        print("Wake word detected!")
```

The cache is a numpy array of shape `[T_cached, F]` grown and windowed automatically (last 50 frames).
`infer_streaming` is defined in `OnnxWakeWordInferencer` — `ww_trainer/inference.py`.

---

## Packaging

**Q: Does ww-trainer have a pyproject.toml?**

Yes. `pyproject.toml` was added alongside `setup.py` (both coexist for backward compatibility). The pyproject.toml declares the build system, project metadata, optional extras, and the `ww_trainer-train` CLI entry point. `setup.py` remains as the authoritative version source.

**Q: What optional dependency groups are available?**

| Extra | Packages | Use case |
|-------|----------|----------|
| `dev` | pytest, pytest-cov | Running tests |
| `transformers` | transformers | HubertExtractor, Wav2Vec2Extractor |
| `vc` | chatterbox_onnx | Voice-cloning augmentation |
| `mlflow` | mlflow | Experiment tracking |
| `datagen` | datasets, ovos-plugin-manager, ovos-tts-plugin-edge-tts, ovos-vad-plugin-silero | Synthetic dataset generation |

Install with: `uv pip install -e ".[transformers,mlflow]"`

---

## Architecture / Module Layout

**Q: Where is the code for visualization (PCA, t-SNE, ROC/PR/DET plots)?**

All visualization helpers were extracted into `ww_trainer/visualization.py` as standalone functions:
- `plot_roc`, `plot_pr`, `plot_det` — curve plots saved to disk and optionally logged to MLflow
- `log_confidence_histogram` — confidence distribution histogram
- `log_pca`, `log_tsne`, `log_umap` — embedding projections
- `log_embeddings_stats` — embedding norm/variance statistics

**Q: Where is the hard-negative mining logic?**

Extracted to `ww_trainer/mining.py` as `mine_hard_negatives()` — a standalone function that accepts the model, nonwake list, device, and a mutable hardness cache dict.

**Q: Where is the checkpoint save/load logic?**

Extracted to `ww_trainer/checkpoint.py` as `save_checkpoint()` and `load_checkpoint()` — standalone functions that accept the model and optimizer directly.

---

## Documentation

**Q: Where is the full documentation?**

Six docs files in `docs/`:

| File | Contents |
|------|---------|
| `docs/index.md` | Landing page and table of contents |
| `docs/architecture.md` | System design: extractor/head split, data flow, hardware tiers, sliding cache |
| `docs/api.md` | Full API reference for all public classes and methods with `file:line` citations |
| `docs/training.md` | Training guide: dataset prep, CLI reference, loss functions, augmentation |
| `docs/export.md` | ONNX export: extractors, heads, quantization, verification |
| `docs/inference.md` | Inference: single, batch, streaming (ONNX and PyTorch) |
| `docs/sweep.md` | Hyperparameter sweep with Optuna |

**Q: What documentation was added in the 2026-03-10 sprint?**

All six `docs/` files were created (or replaced for `index.md`) in a single documentation sprint. Every class and method entry in `api.md` cites the actual source file and line number. See `docs/changelog.md` for the full transparency report.

**Q: Where is the notebook → ww-trainer data contract documented?**

`docs/data_contract.md`. It covers: `AudioDataset.__init__` parameters, CSV format, audio format support, augmentation folder conventions, label convention (`"1"` / `"0"`), and an end-to-end example from notebook output to a trained ONNX model.

**Q: Is TODO.md up to date?**

Yes. As of 2026-03-10 all P0, P1, P2, and P3 items are complete and the active sections have been removed. Only the Completed section remains.

---

## Datagen (ww_trainer-datagen)

**Q: How does TTS plugin discovery work in datagen?**

`_collect_tts_plugins()` calls `ovos_plugin_manager.tts.find_tts_plugins()` to discover all installed OVOS TTS plugins via entry points. Any plugin (edge-tts, google-tx, phoonnx, piper, etc.) is automatically used if installed. No hardcoded imports.

**Q: What VAD engine does datagen use?**

OVOS VAD via `OVOSVADFactory` with `ovos-vad-plugin-silero` as default. Replaces the previous `webrtcvad` dependency. The plugin exposes `is_silence(chunk)` on 16kHz PCM frames.

**Q: How do I add a new TTS engine to datagen?**

Install any OVOS TTS plugin (`uv pip install ovos-tts-plugin-<name>`). It will be auto-discovered via OPM entry points — no code changes needed.

---

## Quickstart (ww_trainer-quickstart)

**Q: How do I go from a wake-word string to a trained ONNX model in one command?**

```bash
ww_trainer-quickstart --wake-word "hey jarvis" --output-dir ./hey_jarvis
```

This runs datagen (TTS synthesis + negatives) then training automatically. See `docs/quickstart.md` for all options.

**Q: How do I use the Python API?**

```python
from ww_trainer.quickstart import train_from_wakeword
result = train_from_wakeword("hey jarvis", "./hey_jarvis", tier="micro", epochs=2)
print(result.best_onnx_path)
```

**Q: How do I skip datagen if I already have a dataset?**

Pass `--reuse-dataset` (CLI) or `reuse_dataset=True` (Python). The dataset directory must contain `train/metadata.csv` and `test/metadata.csv`.

**Q: What tier should I use?**

`small` (default) targets RPi-class hardware (~200K params). Use `micro` for MCU/RPi Zero or `medium`/`large` for server-side accuracy. Run `ww_trainer-tiers` to see all options.

**Q: How is augmentation data wired from datagen to training?**

`_train_from_datagen_result` reads `bg_noise_dir`, `music_dir`, and `rir_dir` from `DatagenResult` and passes them as `bg_noise_folder`, `music_folder`, `rir_folder` kwargs to `WakeWordTrainer` — `ww_trainer/quickstart.py:168`.
