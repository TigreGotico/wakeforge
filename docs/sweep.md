# Hyperparameter Sweep Guide

Four search strategies in `ww_trainer/sweep.py`: Optuna (Bayesian), Grid, Random, and Genetic.

---

## Strategies Overview

| Strategy | Function | Line | Trials | Best For |
|----------|----------|------|--------|----------|
| Optuna (Bayesian) | `run_sweep()` | `sweep.py:20` | User-specified | General purpose, adaptive |
| Grid | `run_grid_search()` | `sweep.py:202` | All combos | Small spaces (<100 combos) |
| Random | `run_random_search()` | `sweep.py:276` | User-specified | High-dimensional spaces |
| Genetic | `run_genetic_search()` | `sweep.py:345` | pop_size * generations | Complex spaces, evolution |

All strategies use the same evaluation function `_evaluate_config()` -- `sweep.py:150` and default search space `_build_search_space()` -- `sweep.py:139`.

---

## Default Search Space

Defined in `_build_search_space()` -- `sweep.py:139`:

| Parameter | Values |
|-----------|--------|
| `arch` | `"ffn"`, `"gru"`, `"cnn"` |
| `hidden_dim` | 64, 128, 256 |
| `lr` | 1e-4, 5e-4, 1e-3, 5e-3, 1e-2 |
| `batch_size` | 16, 32, 64 |
| `dropout` | 0.0, 0.1, 0.2, 0.3 |

Total grid size: 3 * 3 * 5 * 3 * 4 = 540 combinations.

All strategies use BCE loss with weight 1.0. Data is loaded once and split 80/20 (`sweep.py:62-67`).

---

## 1. Optuna (Bayesian Optimization)

`run_sweep()` -- `sweep.py:20`

Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler. Learns which regions of the search space are promising and allocates more trials there.

**Python API:**
```python
from ww_trainer.sweep import run_sweep

run_sweep(
    metadata_csv="dataset.csv",
    n_trials=50,
    output_dir="sweep_results",
    featurizer_type="mfcc",
    epochs_per_trial=5,
    device="auto",
    study_name="my_sweep",
    storage="sqlite:///sweep.db",  # resumable
)
```

**CLI:**
```bash
python -m ww_trainer.sweep \
  --metadata dataset.csv --trials 50 \
  --featurizer-type mfcc --epochs 5 \
  --storage "sqlite:///sweep.db"
```

**Optuna-specific search space** (different from default grid -- uses continuous/log distributions): `sweep.py:70-74`. `lr` is log-uniform `[1e-4, 1e-2]`, `dropout` is step 0.1 in `[0.0, 0.4]`.

**Output:** `best_params.json` in output_dir (`sweep.py:133-135`). Per-trial checkpoints in `trial_{N}/`.

**Resumable:** Uses `load_if_exists=True` (`sweep.py:123`). Pass `storage` URL to resume crashed sweeps.

**When to use:** Default recommendation. Most efficient for 20-100 trials. Requires `pip install optuna`.

**When NOT to use:** When you want exhaustive coverage (use Grid). When Optuna is not installable.

---

## 2. Grid Search

`run_grid_search()` -- `sweep.py:202`

Exhaustive evaluation of every combination in the search space.

```python
from ww_trainer.sweep import run_grid_search

results = run_grid_search(
    metadata_csv="dataset.csv",
    output_dir="grid_results",
    featurizer_type="mfcc",
    epochs_per_trial=5,
    search_space={  # optional custom space
        "arch": ["gru", "cnn"],
        "hidden_dim": [64, 128],
        "lr": [1e-3, 5e-3],
        "batch_size": [32],
        "dropout": [0.1, 0.2],
    },
)
```

**Output:** `grid_results.json` with `best_config`, `best_score`, `all_results` (`sweep.py:268-270`).

**When to use:** Small search spaces (<100 combos). When you need guaranteed coverage. When comparing a handful of specific configurations.

**When NOT to use:** Default search space has 540 combos -- too many for grid. Use Random or Optuna instead.

---

## 3. Random Search

`run_random_search()` -- `sweep.py:276`

Samples configurations uniformly from the search space. Bergstra & Bengio (2012) showed random search finds good configs faster than grid search when not all hyperparameters matter equally.

```python
from ww_trainer.sweep import run_random_search

results = run_random_search(
    metadata_csv="dataset.csv",
    n_trials=50,
    output_dir="random_results",
    featurizer_type="mfcc",
    epochs_per_trial=5,
)
```

**Output:** `random_results.json` (`sweep.py:338`).

**When to use:** Large search spaces. When Optuna is not available. Quick exploration before focused Optuna sweep.

**When NOT to use:** When you have budget for Optuna (it's strictly more efficient).

---

## 4. Genetic Search

`run_genetic_search()` -- `sweep.py:345`

Evolves a population through selection, crossover, and mutation.

Algorithm per generation (`sweep.py:431-463`):
1. Evaluate all individuals (fitness = F1 score)
2. Select elite (top `elite_frac` fraction)
3. Breed new population via uniform crossover (`_crossover` -- `sweep.py:409`)
4. Mutate genes with probability `mutation_rate` (`_mutate` -- `sweep.py:416`)

```python
from ww_trainer.sweep import run_genetic_search

results = run_genetic_search(
    metadata_csv="dataset.csv",
    population_size=20,
    generations=10,
    output_dir="genetic_results",
    featurizer_type="mfcc",
    epochs_per_trial=5,
    mutation_rate=0.3,
    elite_frac=0.2,
)
```

**Output:** `genetic_results.json` with `best_config`, `best_score`, `history` (per-generation best/avg) (`sweep.py:465-470`).

Total evaluations: `population_size * generations` (200 with defaults).

**When to use:** Complex, multi-modal search spaces. When Bayesian optimization gets stuck in local optima. When you want to explore diverse regions.

**When NOT to use:** Small budgets (<50 evaluations) -- not enough generations to converge. Simple spaces where Optuna works well.

---

## Strategy Selection Guide

| Budget | Space Size | Recommendation |
|--------|-----------|----------------|
| <20 trials | Any | Random |
| 20-100 trials | Any | Optuna |
| Full coverage needed | <100 combos | Grid |
| 100+ trials, complex space | >1000 combos | Genetic (pop=20, gen=5+) |
| No Optuna available | Any | Random or Genetic |

---

## Post-Sweep: Retrain Best Config

After finding best params, retrain with full dataset and more epochs:

```bash
ww_trainer-train \
  --wake-word hey_jarvis \
  --metadata dataset.csv \
  --arch gru --hidden-dim 256 \
  --epochs 50 --save-best \
  --output-dir ./models/hey_jarvis_final
```

---

## Optuna Dashboard

With SQLite storage, launch the Optuna dashboard for interactive visualization:

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///sweep.db
```
