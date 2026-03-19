# ww-trainer — Hyperparameter Sweep Guide

Automated hyperparameter search with Optuna.

---

## 1. Overview

`ww_trainer.sweep` (`sweep.py`) provides `run_sweep()` — an Optuna-based hyperparameter search over architecture and training parameters. Each trial trains a model for a small number of epochs on an 80/20 train/val split and returns the validation F1 score.

**When to use a sweep:**
- You do not know which head architecture (FFN / GRU / CNN) works best for your wake word.
- You want to tune learning rate, batch size, hidden size, or dropout without manual grid search.
- You have a GPU or fast CPU and can afford to run 20–100 short trials.

---

## 2. Installation

Optuna is an optional dependency:

```bash
pip install "ww_trainer[sweep]"
# or
pip install ww_trainer optuna
```

The `run_sweep` function raises `ImportError` with a clear message if `optuna` is not installed (`sweep.py:46`–`52`).

---

## 3. Quick Start

```python
from ww_trainer.sweep import run_sweep

run_sweep(
    metadata_csv="dataset.csv",
    n_trials=30,
    output_dir="sweep_results",
    featurizer_type="mfcc",
    epochs_per_trial=10,
    device="auto",
)
```

Or from the command line:

```bash
python -m ww_trainer.sweep \
  --metadata dataset.csv \
  --trials 30 \
  --output-dir sweep_results \
  --featurizer-type mfcc \
  --epochs 10
```

After the sweep completes, `best_params.json` is written to `output_dir/`:

```json
{
  "best_value": 0.8721,
  "best_params": {
    "arch": "gru",
    "hidden_dim": 256,
    "lr": 0.000312,
    "batch_size": 32,
    "dropout": 0.1
  }
}
```

---

## 4. Search Space

Defined in the `objective` function at `sweep.py:70`–`75`:

| Hyperparameter | Type | Range |
|----------------|------|-------|
| `arch` | categorical | `"ffn"`, `"gru"`, `"cnn"` |
| `hidden_dim` | categorical | `64`, `128`, `256` |
| `lr` | log-uniform float | `[1e-4, 1e-2]` |
| `batch_size` | categorical | `16`, `32`, `64` |
| `dropout` | step float | `0.0`, `0.1`, `0.2`, `0.3`, `0.4` |

All trials use a single BCE loss with weight 1.0 (`sweep.py:76`). The data is loaded once and shared across all trials (`sweep.py:62`–`67`).

---

## 5. Persistent Storage (Resumable Sweeps)

By default `storage=None` uses an in-memory Optuna database — the sweep is lost if the process crashes. Use a SQLite file to make it resumable:

```python
from ww_trainer.sweep import run_sweep

run_sweep(
    metadata_csv="dataset.csv",
    n_trials=100,
    output_dir="sweep_results",
    study_name="hey_jarvis_mfcc",
    storage="sqlite:///hey_jarvis_sweep.db",
    featurizer_type="mfcc",
    epochs_per_trial=5,
)
```

Or from CLI:

```bash
python -m ww_trainer.sweep \
  --metadata dataset.csv \
  --trials 100 \
  --storage "sqlite:///hey_jarvis_sweep.db"
```

Optuna uses `load_if_exists=True` (`sweep.py:123`) — if the study already exists in storage, new trials are appended to it. Restart the command to resume from where it stopped.

---

## 6. Analysing Results

### `best_params.json`

Written to `output_dir/best_params.json` after the sweep (`sweep.py:133`–`135`). Contains the best trial's value (F1) and the corresponding hyperparameters.

### Per-trial checkpoints

Each trial writes its checkpoints to `output_dir/trial_{N}/`. You can inspect the metrics CSV for each trial:

```bash
cat sweep_results/trial_5/metrics.csv
```

### Optuna dashboard

If using a SQLite storage URL, launch the Optuna dashboard:

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///hey_jarvis_sweep.db
```

Open http://localhost:8080 for an interactive visualization of all trials.

### Loading the best model

After finding the best params, retrain with the full dataset and more epochs:

```bash
ww_trainer-train \
  --wake-word hey_jarvis \
  --metadata dataset.csv \
  --arch gru \
  --tier small \
  --epochs 50 \
  --save-best \
  --output-dir ./models/hey_jarvis_final
```

---

## 7. Custom Objectives

The `objective` function inside `run_sweep` (`sweep.py:69`–`117`) is not directly overridable from the public API — it is a nested function. To customize the search space or objective metric, copy `sweep.py`, modify the `objective` function, and run your custom version:

```python
import optuna
from ww_trainer.trainer import WakeWordTrainer

def objective(trial):
    arch = trial.suggest_categorical("arch", ["gru"])  # fix arch
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512, step=64)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
    # add custom params
    gru_n_layers = trial.suggest_int("gru_n_layers", 1, 3)

    trainer = WakeWordTrainer(
        arch=arch,
        featurizer="",
        featurizer_type="mfcc",
        losses_cfg=[{"name": "bce", "weight": 1.0}],
        hidden_dim=hidden_dim,
        dropout=dropout,
        gru_n_layers=gru_n_layers,
    )
    # ... run trainer.train(...) and return your metric
    return 0.0

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(study.best_params)
```
