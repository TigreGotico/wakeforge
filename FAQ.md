# ww-trainer — FAQ

## Inspiration Features (precise-lite-trainer)

**Q: How do I cache extracted features between training runs?**

Use `--feature-cache-dir .feature_cache/` (default). Un-augmented waveforms are cached as `.npy` files keyed by MD5(file content + extractor identity). Cache is bypassed when augmentation is active. Disable with `--no-feature-cache`. See `FeatureCache` — `cache.py`.

**Q: How do I freeze layers for transfer learning / fine-tuning?**

Use `--freeze-extractor` to freeze the entire feature extractor, and/or `--freeze-layers N` to freeze the first N classifier parameters. Add `--unfreeze-at-epoch N` for progressive unfreezing (frozen for N epochs, then all layers train). See `WakeWordTrainer._freeze` — `trainer.py`.

**Q: What is epoch-level data replacement?**

`--replacement-ratio 0.4` randomly drops 40% of epoch data and replaces it with fresh samples from the full pool each epoch, reducing overfitting. `--balanced-replacement` ensures 50/50 wake/nonwake in the replaced portion. Complements hard-negative mining.

**Q: What is the composite fitness score?**

`compute_fitness_score()` (`evaluation.py`) combines detection quality and model size into a single metric: `(1 - 0.8*FP_rate - 0.2*FN_rate) * size_penalty`. FP is penalized 4× more than FN. Enable best-fitness checkpointing with `--fitness-checkpoint`. Set `--fitness-param-budget` for the size penalty threshold.

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

All six `docs/` files were created (or replaced for `index.md`) in a single documentation sprint. Every class and method entry in `api.md` cites the actual source file and line number. See `MAINTENANCE_REPORT.md` for the full transparency report.

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
