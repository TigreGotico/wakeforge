# ww-trainer — Training Guide

Step-by-step guide for training a wake word model from raw audio data to a deployed ONNX checkpoint.

---

## 1. Dataset Preparation

### CSV format

The training metadata is a plain-text CSV with two columns and no header:

```
/absolute/path/to/hey_jarvis_001.wav,1
/absolute/path/to/background_noise_042.wav,0
/absolute/path/to/hey_jarvis_002.wav,1
```

- Column 1: absolute or relative path to an audio file (WAV, FLAC, MP3, OGG, M4A).
- Column 2: `1` for wake word, `0` for non-wake.

`AudioDataset` (`dataset.py:104`) accepts this as a list of `(path, label)` tuples. Files are validated on construction; missing files are logged with `logger.warning`. Label distribution is logged via `logger.info`.

### Directory layout

```
dataset/
├── wake/
│   ├── hey_jarvis_001.wav
│   └── hey_jarvis_002.wav
└── nonwake/
    ├── background_001.wav
    └── background_002.wav
```

### Recommended balance

The trainer handles class imbalance via adaptive negative sampling — you do not need a 50/50 split. A practical starting point is:

- Wake samples: 200–2000 utterances (synthetic or recorded).
- Non-wake samples: 5×–20× more than wake.

### Generating synthetic wake word data

Use a TTS engine to generate phonetically diverse recordings of your wake phrase. The `notebooks/ww/tts2ww.ipynb` notebook in the repository implements an 8-stage synthetic data factory. Diversity in speaker, speed, pitch, and noise conditions is more important than raw quantity.

### Sample rate

All audio is expected at 16 000 Hz. `AudioDataset.__getitem__` (`dataset.py:284`) resamples automatically using `torchaudio.functional.resample` if the source differs, but resampling at load time is slower than pre-converting your dataset.

---

## 2. Choosing a Hardware Tier

| Tier | Extractor | Head | Params | Target Hardware | When to Use |
|------|-----------|------|--------|----------------|-------------|
| `micro` | MFCC (40 coeff) | FFN (128d) | ~50K | MCU, RPi Zero | Embedded, no GPU, minimum latency |
| `small` | MFCC (40 coeff) | GRU (128d) | ~200K | RPi, small SBC | Constrained device, better temporal modelling |
| `medium` | HuBERT ONNX | FFN (128d) | ~90M feat + 200K head | RPi 4, laptop | High accuracy, extractor pre-exported |
| `large` | HuBERT (PyTorch) | GRU bidir 2-layer (256d) | ~300M feat + 1M head | Server/workstation | Best accuracy, GPU available |

**Rules of thumb:**
- Start with `small` — it works offline, has temporal modeling, and is fast enough to iterate.
- Use `medium` when you have a pre-exported HuBERT ONNX and need higher accuracy on constrained hardware.
- Use `large` only when accuracy is the primary concern and you have a GPU.
- Use `micro` only for truly constrained targets (MCUs, RPi Zero).

```bash
ww_trainer-train --list-tiers
```

---

## 3. CLI Quickstart

### Micro tier (MFCC + FFN, no GPU)

```bash
ww_trainer-train \
  --wake-word hey_jarvis \
  --metadata dataset.csv \
  --tier micro \
  --epochs 50 \
  --batch-size 32 \
  --output-dir ./models/hey_jarvis_micro
```

### Small tier (MFCC + GRU)

```bash
ww_trainer-train \
  --wake-word hey_jarvis \
  --metadata dataset.csv \
  --tier small \
  --epochs 50 \
  --save-best \
  --output-dir ./models/hey_jarvis_small
```

### Medium tier (HuBERT ONNX + FFN)

```bash
ww_trainer-train \
  --wake-word hey_jarvis \
  --metadata dataset.csv \
  --featurizer distilhubert.onnx \
  --arch ffn \
  --epochs 30 \
  --save-best \
  --export-onnx \
  --output-dir ./models/hey_jarvis_medium
```

### Large tier (HuBERT PyTorch + bidirectional GRU, GPU)

```bash
ww_trainer-train \
  --wake-word hey_jarvis \
  --metadata dataset.csv \
  --tier large \
  --epochs 30 \
  --amp \
  --batch-size 64 \
  --save-best \
  --device cuda \
  --output-dir ./models/hey_jarvis_large
```

---

## 4. Full CLI Reference

All options for `ww_trainer-train` (`trainer.py:668`):

### Hardware tier preset

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--tier` | choice | `None` | Preset: `micro`, `small`, `medium`, `large`. Overrides `--arch` and `--featurizer-type`. |
| `--list-tiers` | flag | `False` | Print tier table and exit. |

### Dataset

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--wake-word` | str | required | Wake word name (used in output file naming) |
| `--metadata` | str | required | Path to training CSV (`path,label`) |
| `--test-metadata` | str | `None` | Optional separate test CSV. If absent, dataset is split. |
| `--split` | float | `0.8` | Train/test split ratio when `--test-metadata` is not provided |

### Training loop

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--epochs` | int | `50` | Number of training epochs |
| `--batch-size` | int | `16` | Mini-batch size |
| `--lr` | float | `5e-4` | Initial learning rate (Adam) |
| `--resume` | str | `None` | Resume from a `.pt` checkpoint |
| `--output-dir` | str | `trained_models/<arch>/<wake_word>` | Output directory |
| `--save-best` | flag | `False` | Save separate checkpoints for best loss/precision/recall/F1 |

### Architecture

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--featurizer` | str | `None` | Path to extractor ONNX file |
| `--feature-dim` | int | `None` | Feature dimension F (auto-detected from extractor if omitted) |
| `--arch` | str | `"gru"` | Head architecture: `gru`, `cnn`, `ffn` |
| `--device` | choice | `"auto"` | `"auto"`, `"cpu"`, or `"cuda"` |
| `--sample-rate` | int | `16000` | Audio sample rate |
| `--export-onnx` | flag | `False` | Export classifier head to ONNX after each checkpoint save |

### Loss configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--loss-type` | str | `"bce"` | Loss type(s): `bce`, `triplet`, `pair`, `cn2pair`, `rppl`, or comma-separated combination |
| `--loss-weight` | str | `"1.0"` | Comma-separated weights matching `--loss-type` |
| `--triplet-margin` | float | `1.0` | Margin for triplet-based losses |
| `--mining-type` | choice | `"semihard"` | Triplet mining: `semihard`, `hard`, or `random` |

### Hard-negative mining

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--neg-threshold` | float | `0.5` | Model confidence cutoff for identifying hard negatives |
| `--mine-sample` | float | `0.2` | Fraction of non-wake pool to evaluate per epoch |
| `--patience` | int | `2` | Epochs with no new hard negatives before early stopping |
| `--base-hard` | float | `0.5` | Hard-negative ratio at epoch 0 |
| `--max-hard` | float | `5.0` | Hard-negative ratio at final epoch |
| `--base-easy` | float | `1.5` | Easy-negative ratio at epoch 0 |
| `--min-easy` | float | `0.2` | Easy-negative ratio at final epoch |
| `--base-random` | float | `0.1` | Random-negative baseline ratio |
| `--total-ratio` | float | `5.0` | Target negatives per wake sample |
| `--blend-ratio` | float | `0.7` | Blend weight between progress and LR for adaptation |

### Augmentation

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--aug-prob` | float | `0.8` | Probability of applying any augmentation to a training sample |
| `--vc-prob` | float | `0.1` | Probability of voice-cloning augmentation (wake samples only) |
| `--bg-noise-folder` | str | `None` | Folder with background noise clips (applied with p=0.6) |
| `--mic-noise-folder` | str | `None` | Folder with microphone/silence clips (p=0.8) |
| `--music-folder` | str | `None` | Folder with music clips (p=0.3, SNR 0–10 dB) |
| `--bg-speech-folder` | str | `None` | Folder with background speech clips (p=0.5, SNR 10–25 dB) |
| `--rir-folder` | str | `None` | Folder with Room Impulse Responses (p=0.3) |
| `--vc-folder` | str | `None` | Folder with speaker voices for voice cloning |
| `--snr-min` | float | `0.0` | Minimum SNR for noise mixing (dB) |
| `--snr-max` | float | `20.0` | Maximum SNR for noise mixing (dB) |
| `--pitch-min` | float | `-1.0` | Minimum pitch shift in semitones (p=0.3) |
| `--pitch-max` | float | `1.0` | Maximum pitch shift in semitones |
| `--speed-min` | float | `0.95` | Minimum speed perturbation factor (p=0.3) |
| `--speed-max` | float | `1.05` | Maximum speed perturbation factor |

### Feature Cache

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--feature-cache-dir` | str | `".feature_cache/"` | Directory for cached feature vectors (`.npy` files) |
| `--no-feature-cache` | flag | `False` | Disable feature vectorization cache |

`FeatureCache` (`cache.py`) stores un-augmented waveforms keyed by MD5(file content + extractor class + feature_dim + sample_rate). Cache is automatically invalidated when the extractor changes. Bypassed when augmentation is applied to a sample.

### Layer Freezing (Transfer Learning)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--freeze-extractor` | flag | `False` | Freeze entire feature extractor (`requires_grad=False`) |
| `--freeze-layers` | int | `0` | Freeze first N classifier layer parameters |
| `--unfreeze-at-epoch` | int | `None` | Unfreeze all layers at this epoch (progressive unfreezing) |

Useful for fine-tuning a pre-trained model on a new wake word. `_freeze()` and `_unfreeze()` methods on `WakeWordTrainer` (`trainer.py`).

### Data Replacement (Epoch Resampling)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--replacement-ratio` | float | `0.0` | Fraction of training data to resample each epoch (0 = disabled) |
| `--balanced-replacement` | flag | `False` | Ensure equal pos/neg in the resampled portion |

Each epoch, `replacement_ratio` fraction of the epoch data is dropped and replaced with random samples from the full pool. Complements hard-negative mining. Implemented in `WakeWordTrainer.train()` (`trainer.py`).

### Fitness Checkpoint

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--fitness-checkpoint` | flag | `False` | Save `best_fitness.pt` based on composite fitness score |
| `--fitness-param-budget` | int | `100000` | Parameter budget for the size penalty component |

Composite fitness: `(1 - fp_weight*FP_rate - fn_weight*FN_rate) * size_penalty`. See `compute_fitness_score()` — `evaluation.py`.

### Performance

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--amp` | flag | `False` | Enable mixed-precision training (requires CUDA) |
| `--accumulate-grad-batches` | int | `1` | Accumulate gradients over N batches before optimizer step |

### Logging

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--metrics-log` | str | `"metrics_log.csv"` | Per-epoch metrics CSV file |
| `--mlflow-uri` | str | `None` | MLflow tracking URI |

### Visualization

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--pca-every` | int | `1` | Run PCA embedding plot every N epochs (0 = disable) |
| `--tsne-every` | int | `0` | Run t-SNE embedding plot every N epochs (0 = disable) |
| `--umap-every` | int | `0` | Run UMAP embedding plot every N epochs (0 = disable) |
| `--rppl-every` | int | `0` | RPPL dashboard plot every N epochs (0 = auto: every 5 epochs when loss=rppl) |

All three embedding viz functions (`log_pca`, `log_tsne`, `log_umap`) now log the same scalar embedding metrics to MLflow: `embed_centroid_dist`, `embed_fisher_ratio`, `embed_silhouette`, `embed_intra_pos`, `embed_intra_neg`, `embed_centroid_cos_sim`. These appear in the MLflow metrics tab and accumulate per epoch so you can plot them over training.

---

## 5. Loss Functions

| Name | Class | When to Use |
|------|-------|-------------|
| `bce` | `BCEWithLogitsLoss` | Default. Simple binary classification. Start here. |
| `triplet` | `TripletMarginLoss` | Hard metric learning. Good when BCE plateaus. |
| `soft_triplet` | `SoftTripletLoss` | Triplet without hard margin — gradient always flows. |
| `pair` | `MarginRankingLoss` | Pairwise (anchor-positive vs anchor-negative). |
| `cn2pair` | `CN2Plus1PairLoss` | López-Espejo et al. — uses all negative-negative pairs. |
| `lse` | `LiftedStructureLoss` | Log-sum-exp over all pairs in batch. |
| `contrastive` | `ContrastiveLoss` | Classic Siamese. |
| `angular` | `AngularLoss` | Cosine-based margin. |
| `rppl` | `RobustProtoDiversityLoss` | Composite: BCE + prototype + diversity + center loss. Recommended for production. |

Multiple losses can be combined:

```bash
--loss-type bce,triplet --loss-weight 0.7,0.3
```

The `rppl` loss is the most comprehensive: it enforces both classification accuracy (BCE component) and embedding quality (EMA prototype contrastive, hard-negative diversity, center loss, proto-ranked consistency). Use it when you want the best possible embedding separation and per-epoch visibility into embedding geometry via MLflow.

Use `train_rppl.py` for a dedicated RPPL experiment — it enables PCA + t-SNE viz every N epochs and generates a 6-panel RPPL dashboard logged as an MLflow artifact.

---

## 6. Hard-Negative Mining

Hard-negative mining identifies non-wake samples that the model confuses with the wake word (false positives) and adds them to the training set for the next epoch.

**What it does** (`mining.py:14`):
1. Evaluates `dataset_fraction` of the non-wake pool.
2. Computes a hardness score: `hardness = max(0, prob - neg_threshold)`.
3. Updates a rolling cache with EMA: `new_score = 0.9 * old + 0.1 * hardness`.
4. If `use_embedding_mining=True`, refines by computing cosine similarity to the wake prototype embedding and blending with the confidence score.
5. Returns hard (high-hardness) and easy (low-hardness) splits.

**`--mine-sample 0.2`**: evaluate 20% of the non-wake pool each epoch. Lower values are faster but less thorough.

**`--mining-type`**: affects triplet selection within the `LossManager`, not the outer hard-negative mining loop. `semihard` is the most stable; `hard` can cause instability early in training.

**Disable mining:** `--mine-sample 0` disables mining; training uses a random negative sample each epoch.

**Early stopping:** if no new hard negatives are found for `--patience` epochs, training stops early.

---

## 7. Augmentation Options

Augmentation is applied on-the-fly in `AudioDataset.get_augmented` (`dataset.py:173`). Each type has a fixed probability independent of `--aug-prob`:

| Type | Probability | Config |
|------|-------------|--------|
| Background noise | 0.6 (if folder set) | `--bg-noise-folder`, `--snr-min/max` |
| Microphone noise | 0.8 (if folder set) | `--mic-noise-folder`, `--snr-min/max` |
| Music | 0.3 (if folder set) | `--music-folder`, SNR 0–10 dB fixed |
| Background speech | 0.5 (if folder set) | `--bg-speech-folder`, SNR 10–25 dB fixed |
| Reverb (RIR) | 0.3 (if folder set) | `--rir-folder` |
| Pitch shift | 0.3 (always) | `--pitch-min/max` in semitones |
| Speed perturbation | 0.3 (always) | `--speed-min/max` as factor |
| Voice cloning | `--vc-prob` (wake only) | `--vc-folder`, requires `wakeforge[vc]` |

`--aug-prob` is the master gate: a sample is augmented only if `random() < aug_prob`. Set `--aug-prob 0` to disable all augmentation.

Post-augmentation: peak normalization to `[-1, 1]` (`dataset.py:239`–`241`).

---

## 8. Mixed Precision and Gradient Accumulation

### Mixed precision (`--amp`)

Enables `torch.amp.autocast` + `GradScaler` (`trainer.py:339`–`342`, `trainer.py:447`). Requires CUDA. Reduces memory usage and speeds up training on modern GPUs (Ampere and newer benefit most). On CPU it has no effect.

```bash
ww_trainer-train ... --amp --device cuda
```

### Gradient accumulation (`--accumulate-grad-batches N`)

Accumulates gradients over N batches before calling `optimizer.step()` (`trainer.py:457`–`463`). Simulates a larger effective batch size without increasing memory:

```bash
# Effective batch size = 16 * 4 = 64
ww_trainer-train ... --batch-size 16 --accumulate-grad-batches 4
```

---

## 9. MLflow Integration

Pass a tracking URI to enable MLflow logging:

```bash
ww_trainer-train ... --mlflow-uri http://localhost:5000
```

What gets logged (`trainer.py:99`–`116`, `trainer.py:479`–`534`):

- **Params**: arch, device, featurizer, feature_dim, loss types/weights, all model kwargs.
- **Per-epoch metrics**: total loss, per-loss breakdown, accuracy, precision, recall, F1, AUC, mean confidence (wake/nonwake), learning rate, hard/easy/random ratios, embedding stats (norm, variance), readiness score.
- **Artifacts**: ROC/PR/DET plots, confidence histograms, false-positive and false-negative CSV files, ONNX checkpoints (when `--export-onnx` is set).

Start a local MLflow server:

```bash
mlflow server --host 127.0.0.1 --port 5000
# Then open http://localhost:5000 in a browser
```

---

## 10. Checkpointing

When `--save-best` is set, the trainer saves a separate checkpoint whenever a metric improves:

| File | Metric |
|------|--------|
| `best_loss.pt` + `best_loss.ts` | Lowest average training loss |
| `best_precision.pt` + `best_precision.ts` | Highest eval precision |
| `best_recall.pt` + `best_recall.ts` | Highest eval recall |
| `best_f1.pt` + `best_f1.ts` | Highest eval F1 |
| `best_fitness.pt` | Highest composite fitness score (requires `--fitness-checkpoint`) |

Without `--save-best`, a checkpoint `ep{N}.pt` is saved every epoch.

A final checkpoint is always saved as `final_model.pt` at the end of training.

The `.pt` file contains model weights only. The `.ts` (trainer state) file contains epoch number, best metrics, and optimizer state. Both are needed to fully resume training.

---

## 11. Infinite Training Mode

`train_infinite.py` — goal-based training loop designed for very large NWW pools (millions of files). Rather than running a fixed number of epochs, it trains until performance targets are met.

### How it works

Each epoch:
1. A random `--scan-size` subset of the NWW pool is inferred.
2. Clips with confidence ≥ `--neg-threshold` are retained as hard negatives.
3. (Optional) `--vc-per-epoch` new positives are synthesised via voice conversion.
4. Model trains on wake positives + hard negatives.
5. Stopping goals are checked — training continues or terminates.

### Stopping goals

| Flag | Default | Meaning |
|------|---------|---------|
| `--target-f1` | 0.92 | Stop when F1 ≥ this (after `--min-epochs`) |
| `--target-eer` | 0.08 | Stop when EER ≤ this |
| `--target-far-frr1` | None | Stop when FAR at FRR=1% ≤ this |
| `--patience-after` | 10 | Epochs of no improvement after goals are met |
| `--hard-plateau` | 20 | Stop if no new hard negatives for this many epochs |
| `--min-epochs` | 20 | Minimum epochs regardless of goals |
| `--max-epochs` | None | Hard ceiling (unlimited by default) |

### Usage

```bash
# Basic — run until F1 ≥ 0.92 and EER ≤ 0.08
.venv/bin/python train_infinite.py

# With VC synthesis (5 new positives per epoch via voiceclonnx)
.venv/bin/python train_infinite.py --vc-per-epoch 5 --vc-text "hey mycroft"

# Larger NWW scan, stricter targets
.venv/bin/python train_infinite.py \
    --scan-size 20000 \
    --target-f1 0.95 --target-eer 0.05 \
    --arch bcresnet --loss rppl
```

All CLI defaults read from `.env` via `WW_` env var prefix (e.g. `WW_TARGET_F1`, `WW_SCAN_SIZE`).

### Mining cache

Hard-negative inference scores are persisted to `hardneg_cache.pt` and reloaded on resume (`--resume-cache`, on by default). This means the first epoch after a restart benefits from previously computed hardness scores.

---

## 12. Voice Conversion Backends

`ww_trainer/vc_helpers.py` is a thin delegation to the pure-ONNX
[voiceclonnx](https://github.com/TigreGotico/voiceclonnx) library — there is no
bespoke backend hierarchy. Voice conversion is audio-to-audio only; pick one of
voiceclonnx's 14 engines. Text→speech is handled separately by the OVOS TTS
datagen pipeline.

| Engine | SR | Notes |
|--------|----|-------|
| `knnvc` | 16 kHz | **Default.** Zero-shot any-to-any, pure-numpy kNN matching. |
| `facodec` / `freevc` / `openvoice` | 16–24 kHz | Zero-shot any-to-any neural converters. |
| `rvc` | 40 kHz | Any-to-one (reference is a trained `.onnx` model). |
| `linacodec` | 48 kHz | Codec-quality (formerly vendored, now a voiceclonnx engine). |
| `chatterbox` | 24 kHz | Chatterbox AR codec-LM, VC path. |

See `voiceclonnx` for the full list; `from ww_trainer.vc_helpers import list_engines`.

**Select an engine:**

```bash
# Environment variable (persists across scripts)
export WW_VC_ENGINE=knnvc        # any of the 14 voiceclonnx engines

# Or per-script CLI flag
.venv/bin/python generate_vc_positives.py --vc-engine facodec
.venv/bin/python train_infinite.py --vc-backend linacodec
```

**Python API:**

```python
from ww_trainer.vc_helpers import load_vc_backend

backend = load_vc_backend()                    # auto-detect
backend.tts("hey mycroft", donor.wav, out.wav, exaggeration=0.4)
backend.vc(source.wav, donor.wav, out.wav)
print(backend.sample_rate)                     # 24000 or 48000
```

---

## 13. Resuming Training

```bash
ww_trainer-train \
  --wake-word hey_jarvis \
  --metadata dataset.csv \
  --tier small \
  --resume ./models/hey_jarvis_small/best_f1.pt \
  --output-dir ./models/hey_jarvis_small
```

On resume (`trainer.py:94`–`95`, `trainer.py:344`–`350`):
- Model weights are loaded from `resume`.
- If a `hardneg_cache.pt` file exists in the checkpoint directory, the hard-negative cache is restored.
- The optimizer state is loaded from the `.ts` file if it exists.
- The LR scheduler restarts from epoch 0 (cosine annealing restarts — intentional).
