# AGENTS.md — Resource Guidelines for Local Experiments

This file governs how automated agents (Claude Code, CI scripts, notebooks) must
behave when running wake-word experiments on this machine.  Violating these rules
risks OOM kills, swap exhaustion, kernel stalls, or filling the disk.

---

## Machine Profile

| Resource | Total | Typical headroom |
|----------|-------|-----------------|
| RAM | 30 GB | ~18 GB available, but swap already 66% full at idle |
| Swap | 9.2 GB | Only ~3 GB free at idle — treat as emergency-only |
| CPU | 24 cores (AMD Ryzen AI 9 HX 370) | Cap workload at 4–6 cores |
| GPU | None (integrated Radeon 890M, no CUDA) | All training runs on CPU |
| Disk (`/`) | 906 GB, 321 GB free | Keep ≥ 50 GB free; clean up after runs |
| `/tmp` | 16 GB tmpfs | Do not write large files here |

**The swap being pre-loaded at idle is the single biggest risk.**
Any experiment that pushes RAM usage above ~16 GB will spill into the remaining
3 GB of swap, causing severe thrashing and potentially a system freeze.

---

## Hard Limits — Never Exceed

### Memory
- **Do not load HuBERT, Wav2Vec2, or Wav2Vec2-BERT extractors** locally.
  These require 2–10 GB of RAM just to load, plus gradient storage during training.
  Use MFCC, FilterBank, SincNet, or LEAF instead.
- **Do not use `tier` presets above `cortex_m7`** for local experiments.
  `rpi_zero`, `rpi_4`, and anything above are too large for CPU-only training.
- **Maximum batch size: 16.** Larger batches hold more audio tensors in RAM
  simultaneously. Use gradient accumulation (`accumulate_grad_batches`) if you
  need effective larger batches.
- **Maximum DataLoader workers: 2.** Each worker forks the process and duplicates
  the dataset object in memory. Use `num_workers=0` when in doubt.

### CPU
- **Cap PyTorch threads at 12** (leaving headroom for the OS and other processes):
  ```python
  import torch
  torch.set_num_threads(12)
  torch.set_num_interop_threads(4)
  ```
  Or set before launching any script:
  ```bash
  OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 python train.py ...
  ```
  Without this, PyTorch will use all 24 cores and make the machine unresponsive.
- **Do not use `torch.compile()` locally.** Compilation is CPU-intensive and
  generates large cached artifacts.

### Genetic / Hyperparameter Sweep
- **Maximum 1 deme** (`n_demes=1`). Multiple demes launch separate
  `ProcessPoolExecutor` worker processes; on a memory-constrained machine this
  multiplies RAM usage by the number of demes.
- **Maximum 20 trials per sweep** locally (`n_trials ≤ 20`).
- **Do not run island-model migration** (`migration_interval=0`) unless you have
  verified available RAM exceeds 10 GB before starting.
- **Set `timeout_minutes`** on every sweep call. Recommended: 30 minutes locally.

### Distributed Training
- **Never use DDP (`ddp.py`) locally.** There is no GPU and it will spawn
  multiple CPU-bound processes, exhausting all memory.

---

## Recommended Defaults for Local Runs

```python
import torch
torch.set_num_threads(12)
torch.set_num_interop_threads(4)

trainer_kwargs = dict(
    arch="mfcc_gru_small",   # or any tier <= cortex_m7
    batch_size=8,
    epochs=10,
    use_amp=False,           # AMP requires CUDA
    accumulate_grad_batches=4,
)

dataloader_kwargs = dict(
    num_workers=0,
    pin_memory=False,        # pin_memory only helps with CUDA
)

sweep_kwargs = dict(
    n_demes=1,
    n_trials=10,
    timeout_minutes=20,
    migration_interval=0,
)
```

For `preprocess.py`, always pass `--max-files` to cap dataset size:
```bash
OMP_NUM_THREADS=12 python preprocess.py --max-files 500 ...
```

---

## Disk Hygiene

- **MLflow runs accumulate artifacts fast.** After an experiment, either:
  - Delete the run: `mlflow runs delete --run-id <id>`
  - Or disable MLflow entirely for scratch runs: pass `mlflow=None` to the trainer.
- **Checkpoint files:** The training loop saves `best_*.pt` and `final_model.pt`.
  A full training run with `save_best=True` can produce 5+ checkpoint files per
  experiment. Delete checkpoints from failed or exploratory runs promptly.
- **`mlruns/` directory** at the project root already contains experiment data.
  Run `du -sh mlruns/` periodically. If it exceeds 5 GB, clean it up.
- **ONNX exports and quantized models** (e.g., `_int8.onnx`, `_int16.onnx`) are
  generated per export call. Do not export repeatedly without cleaning up.
- **Before starting any experiment, verify headroom:**
  ```bash
  free -h && df -h /home
  ```
  If available RAM < 10 GB or disk free < 50 GB, do not proceed without cleanup.

---

## External Drive — Pre-built Data Assets

The external drive (`/run/media/miro/endeavouros/`) holds large pre-built audio
assets that should be used **instead of re-downloading** when they are available.
Mount the drive before running experiments that need these datasets.

```bash
# Check if mounted
ls /run/media/miro/endeavouros/ww/ 2>/dev/null && echo "mounted" || echo "NOT MOUNTED"
```

### Available Paths

| Path | Contents | Use for |
|------|----------|---------|
| `/run/media/miro/endeavouros/ww/hf_datasets/` | HuggingFace audio datasets (pre-downloaded) | Positives, negatives, augmentation |
| `/run/media/miro/endeavouros/ww/vc_output/` | Voice-cloned samples (donor voices → target phrase) | Positive augmentation, speaker diversity |
| `/run/media/miro/endeavouros/ww/synth_output/` | Raw TTS samples (synthesized wake-word utterances) | Positive augmentation |

### How to Wire Them Into Training

Pass the VC and TTS directories as extra positives, and symlink or pass
the HF datasets as augmentation sources:

```python
# In train_hey_mycroft.py / train_full.py — example
cfg = QuickstartConfig(
    wake_word="hey mycroft",
    # Point at pre-cloned voices instead of re-downloading
    vc_refs_dir="/run/media/miro/endeavouros/ww/vc_output",
    # Additional positives from TTS
    extra_positive_dirs=["/run/media/miro/endeavouros/ww/synth_output"],
    # Use pre-downloaded HF datasets for negatives / augmentation
    hf_datasets_cache="/run/media/miro/endeavouros/ww/hf_datasets",
    download_augmentation=False,  # skip re-download; use cached
    ...
)
```

Or for `WakeWordTrainer` directly:

```python
trainer = WakeWordTrainer(
    ...
    bg_noise_folder="/run/media/miro/endeavouros/ww/hf_datasets/ambient_noises",
    music_folder="/run/media/miro/endeavouros/ww/hf_datasets/FMA_3secs",
    rir_folder="/run/media/miro/endeavouros/ww/hf_datasets/mit_rirs",
)
```

### Agent Rules for External Drive Data

- **Always check the drive is mounted before referencing these paths.**
  Do not hard-fail if unmounted; log a warning and fall back to downloading.
- **Never write to the external drive paths.** Treat them as read-only sources.
  Write all outputs (checkpoints, ONNX, metrics) under `experiments/` on the
  local disk.
- **Prefer these cached assets.** Downloading the same datasets from HuggingFace
  repeatedly wastes bandwidth and time. If a matching directory exists on the
  drive, use it directly.
- **VC samples are high-value positives.** They represent real speaker diversity
  that TTS cannot replicate. Always include them when available.

---

## ONNX-Exportability — Hard Requirement

**Every feature extractor and classifier head added to ww-trainer MUST export cleanly
via `export_to_onnx()` using only standard ONNX ops (opset 18).**

This is a core design constraint. Production inference uses only `onnxruntime` + `numpy`
— no PyTorch, no custom CUDA kernels, no HuggingFace transformers at runtime.

### What this means when adding new components

- **No custom CUDA kernels** — rules out `mamba-ssm`, any torch extension with `.cu` ops.
- **No non-traceable control flow** — avoid Python loops over dynamic lengths in `forward()`;
  use `torch.stft` (not `torchaudio.compliance.kaldi`), buffer-registered filterbanks, etc.
- **No in-place ops on inputs** — ONNX tracer cannot handle aliased mutation.
- **Test export before merging**: run `extractor.export_to_onnx("test.onnx")` and
  `onnx.checker.check_model(onnx.load("test.onnx"))` as part of any new component PR.

### Accepted exceptions (large SSL models)

`HubertExtractor`, `Wav2Vec2Extractor`, `Wav2Vec2BertExtractor`, `TorchAudioHubertExtractor`
raise `NotImplementedError` on `export_to_onnx()` by design — these are training-only
wrappers. Their ONNX equivalents are obtained via `optimum-cli` and loaded as
`OnnxFeatureExtractor`. This exception applies **only** to models ≥ 90 M params that
have a documented `optimum` export path.

### Components dropped for ONNX violation

- `MambaHead` — removed; `mamba-ssm` CUDA kernels not ONNX-registerable.

---

## Safe Extractor / Tier Choices

| Extractor | RAM at load | Safe locally? |
|-----------|------------|---------------|
| MFCC | < 50 MB | Yes |
| FilterBank | < 50 MB | Yes |
| SincNet | < 100 MB | Yes |
| LEAF | < 100 MB | Yes |
| PLP / PNCC / CQT / Gammatone | < 100 MB | Yes |
| DistilHuBERT (ONNX, scripts/) | ~300 MB | Caution — monitor RAM |
| HuBERT base | ~1.5 GB | No |
| HuBERT large | ~3 GB | No |
| Wav2Vec2 base | ~1.5 GB | No |
| Wav2Vec2-BERT | ~2 GB | No |

| Tier preset | Params | Safe locally? |
|-------------|--------|---------------|
| `esp32_nano` | ~241 | Yes |
| `esp32_small` | ~2 K | Yes |
| `cortex_m4` | ~10 K | Yes |
| `cortex_m7` | ~50 K | Yes |
| `rpi_zero` | ~200 K | Caution |
| `rpi_4` | ~1 M | No (with many epochs) |
| `hubert_*` | 90 M–300 M | Never |

---

## Monitoring During a Run

Open a second terminal and watch resources while training:

```bash
# Memory pressure (refresh every 5s)
watch -n 5 'free -h && echo "---" && cat /proc/meminfo | grep -E "Swap(Free|Used)"'

# Disk usage
watch -n 30 'df -h / && du -sh mlruns/ checkpoints/ 2>/dev/null'

# CPU load (ensure training is not pinning all cores)
watch -n 3 'top -bn1 | head -20'
```

If swap free drops below 1 GB, kill the training process immediately:
```bash
# Find and kill training
ps aux | grep python
kill -9 <pid>
```

---

## What Agents Must Do Before Starting Any Experiment

1. Run `free -h` — abort if available RAM < 10 GB.
2. Run `df -h /home` — abort if disk free < 50 GB.
3. Set `OMP_NUM_THREADS=4` and `torch.set_num_threads(4)`.
4. Confirm extractor is in the "safe locally" list above.
5. Confirm `batch_size ≤ 16` and `num_workers ≤ 2`.
6. If running a sweep, confirm `n_demes=1` and `n_trials ≤ 20`.
7. After the experiment, delete checkpoint files that are no longer needed and
   verify `mlruns/` is not growing unboundedly.
