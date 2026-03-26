# ww-trainer — Claude Code Project Context

Wake-word detection training toolkit. Trains lightweight on-device keyword
spotters (MFCC/SincNet/FilterBank → GRU/FFN → ONNX export).

## Quick Orientation

```
ww_trainer/          Core library
  feats.py           Feature extractors (MFCC, SincNet, FilterBank, Gammatone, …)
  dataset.py         Dataset loading, augmentation, hard-negative mining
  datagen.py         HuggingFace dataset downloader (positives + negatives)
  trainer.py         WakeWordTrainer — main training entry point
  loop.py            Training loop (epoch logic, threshold adaptation)
  tiers.py           Hardware tier presets (micro → sincnet_small → …)
  metrics.py         EER, AUC, FAR/FRR, DET curve, FP/hour estimation
  inference.py       OnnxWakeWordInferencer(featurizer_onnx, head_onnx)
  quickstart.py      train_from_wakeword() — one-call pipeline
scripts/
  train_hey_mycroft.py   Quick 30-epoch CPU training (micro tier)
  train_full.py          Multi-arch 50-epoch run (6 architectures)
  eval_hey_mycroft.py    Full evaluation + plots
  test_wakeword.py       CLI: --audio file.wav or --mic live testing
experiments/hey_mycroft/ Output: dataset/, models/<arch>/, eval/
```

## ONNX Export — Two Files Required

Every model exports two ONNX files. Both are required for inference:

- `best_f1_featurizer.onnx` — feature extractor (MFCC / SincNet / …)
- `best_f1.onnx` — classifier head (FFN / GRU)

```python
from ww_trainer.inference import OnnxWakeWordInferencer
model = OnnxWakeWordInferencer("best_f1_featurizer.onnx", "best_f1.onnx")
score = model.infer(wav_float32_array)  # returns float in [0, 1]
```

The trainer's `export_onnx=True` only saves the head. Export the featurizer
separately after training:

```python
trainer.model.load_checkpoint("best_f1.pt")
trainer.model.feature_extractor.export_to_onnx("best_f1_featurizer.onnx")
```

## Data Sources

### Local (pre-built, read-only)

The external drive at `/run/media/miro/endeavouros/ww/` contains pre-built assets.
**Check mount status before use. Never write to this drive.**

| Path | Contents |
|------|----------|
| `…/hf_datasets/` | HF audio datasets (ambient noise, music, RIRs, positives) |
| `…/vc_output/` | Voice-cloned wake-word samples (high-value positives) |
| `…/synth_output/` | Raw TTS synthesized wake-word samples |

```bash
# Check mounted
ls /run/media/miro/endeavouros/ww/ 2>/dev/null && echo "OK" || echo "NOT MOUNTED"
```

When available, pass these to the trainer instead of re-downloading:

```python
trainer = WakeWordTrainer(
    bg_noise_folder="/run/media/miro/endeavouros/ww/hf_datasets/ambient_noises",
    music_folder="/run/media/miro/endeavouros/ww/hf_datasets/FMA_3secs",
    rir_folder="/run/media/miro/endeavouros/ww/hf_datasets/mit_rirs",
    ...
)
```

### HuggingFace (download if drive not available)

`datagen.py` downloads and caches WAVs under `experiments/<ww>/dataset/`.
Arrow cache lives in `~/.cache/huggingface/datasets/` (non-streaming mode).
Streaming mode bypasses the Arrow cache — prefer non-streaming.

## Known Issues / Quirks

- **datagen.py AudioDecoder (datasets ≥ 3.x):** `samples.data` is
  `(channels, samples)` — use `.mean(dim=0)` to mono-mix, not `squeeze(0)`.
- **`torchaudio.save` expects 2D tensor** `(1, samples)`. Passing 3D crashes it.
- **`train_from_wakeword` uses `**kwargs`**, not a config object. Unpack fields.
- **`OnnxWakeWordInferencer` requires two positional args** — featurizer path,
  then head path. One arg raises `TypeError`.
- **No MLflow in project venv.** Pass `mlflow=None` (default). Trainer handles it.
- **`trust_remote_code` removed in datasets ≥ 3.x.** Do not pass it to `load_dataset`.

## Resource Constraints — READ BEFORE ANY EXPERIMENT

This machine has constrained swap. See `AGENTS.md` for full rules. Summary:

- CPU only — no CUDA, no `use_amp=True`, no `torch.compile()`
- `torch.set_num_threads(12)` + `OMP_NUM_THREADS=12` — always
- `batch_size ≤ 16`, `num_workers ≤ 4`
- No HuBERT / Wav2Vec2 — too large for RAM
- Abort if RAM available < 10 GB or disk free < 50 GB
- Safe tiers: `micro`, `delta_micro`, `small`, `filterbank_small`,
  `gammatone_small`, `sincnet_small`

## Python Environment

```bash
# Project venv (Python 3.13)
.venv/bin/python script.py

# Install packages
/usr/bin/uv pip install <pkg> --python .venv/bin/python
```

## Typical Workflow

```bash
# 1. Quick training (30 epochs, micro tier)
.venv/bin/python train_hey_mycroft.py

# 2. Full multi-arch training (50 epochs × 6 archs, background)
.venv/bin/python train_full.py >> train_full.log 2>&1 &
tail -f train_full.log

# 3. Evaluate with plots
.venv/bin/python eval_hey_mycroft.py

# 4. Test a specific model
.venv/bin/python test_wakeword.py --dir experiments/hey_mycroft/models/small --audio sample.wav
.venv/bin/python test_wakeword.py --dir experiments/hey_mycroft/models/small --mic
```
