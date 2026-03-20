# Hyperparameter Sweep Guide

Four search strategies in `ww_trainer/sweep.py`: Optuna (Bayesian), Grid, Random, and Genetic.

---

## Strategies Overview

| Strategy | Function | Line | Trials | Best For |
|----------|----------|------|--------|----------|
| Optuna (Bayesian) | `run_sweep()` | `sweep.py:41` | User-specified | General purpose, adaptive |
| Grid | `run_grid_search()` | `sweep.py:249` | All combos | Small spaces (<100 combos) |
| Random | `run_random_search()` | `sweep.py:324` | User-specified | High-dimensional spaces |
| Genetic | `run_genetic_search()` | `sweep.py:543` | pop_size × generations × n_demes | Complex spaces, island model |
| Two-Stage Genetic | `run_two_stage_genetic_search()` | `sweep.py:662` | (stage1 + stage2) × n_demes | Best quality, broad then focused |

All strategies share `_evaluate_config()` — `sweep.py:180` and `_build_search_space()` — `sweep.py:134`.

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

`run_genetic_search()` — `sweep.py:543`

Evolves a population through selection, crossover, and mutation. Supports island-model parallelism, early stopping, and configurable selection pressure.

Algorithm per generation (`sweep.py:484-533` in `_run_deme`):
1. Evaluate all individuals; raw F1 is stored; `fitness_fn` transform applied for selection only.
2. Select elite (top `elite_frac` fraction by transformed fitness).
3. Breed new population via uniform crossover (`_crossover` — `sweep.py:459`).
4. Mutate genes with probability `mutation_rate` (`_mutate` — `sweep.py:463`).

### Parameters — `run_genetic_search`

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `metadata_csv` | `str` | required | Dataset CSV path (`path,label` per line) |
| `population_size` | `int` | 20 | Individuals per generation |
| `generations` | `int` | 10 | Maximum generations |
| `output_dir` | `str` | `"genetic_results"` | Directory for results and per-trial checkpoints |
| `featurizer` | `str\|None` | `None` | ONNX extractor path |
| `featurizer_type` | `str` | `"mfcc"` | Extractor type |
| `device` | `str` | `"auto"` | `"cpu"`, `"cuda"`, or `"auto"` |
| `epochs_per_trial` | `int` | 5 | Training epochs per candidate |
| `search_space` | `dict\|None` | `None` | Custom param→values map; defaults to `_build_search_space()` |
| `mutation_rate` | `float` | 0.3 | Per-gene mutation probability |
| `elite_frac` | `float` | 0.2 | Fraction of population kept as elite |
| `full` | `bool` | `False` | Expand search space to include featurizer/arch/loss |
| `n_demes` | `int` | 1 | Parallel island count (each deme = one process) |
| `timeout_minutes` | `float\|None` | `None` | Wall-clock stop per deme |
| `target_f1` | `float\|None` | `None` | Stop once best raw F1 >= this value |
| `fitness_fn` | `str` | `"f1"` | Selection pressure: `"f1"`, `"exp_f1"`, `"double_exp_f1"` |
| `seed_population` | `list\|None` | `None` | Pre-built configs to seed the initial population |

**Returns:** `dict` with `best_config`, `best_score` (raw F1), `all_results`, `history`.
Each `history` entry: `{"generation": int, "best": float, "avg": float, "elapsed_seconds": float}`.

**Output:** `genetic_results.json` — `sweep.py:654`.

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
    n_demes=2,
    timeout_minutes=30.0,
    target_f1=0.95,
    fitness_fn="exp_f1",
)
```

Total evaluations: `population_size * generations` (200 with defaults).

**When to use:** Complex, multi-modal search spaces. When Bayesian optimization gets stuck in local optima.

**When NOT to use:** Small budgets (<50 evaluations) — not enough generations to converge.

---

## 5. Two-Stage Genetic Search

`run_two_stage_genetic_search()` — `sweep.py:662`

Stage 1 runs a broad GA to find promising regions. Stage 2 seeds a tighter GA with the top-K configs from stage 1 using lower mutation rate and higher elite fraction.

### Parameters — `run_two_stage_genetic_search`

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `metadata_csv` | `str` | required | Dataset CSV path |
| `population_size` | `int` | 20 | Stage 1 population size |
| `generations` | `int` | 10 | Stage 1 generation count |
| `stage2_population` | `int` | 10 | Stage 2 population size |
| `stage2_generations` | `int` | 5 | Stage 2 generation count |
| `stage2_elite_frac` | `float` | 0.5 | Elite fraction for stage 2 |
| `stage2_mutation_rate` | `float` | 0.05 | Mutation rate for stage 2 (lower = more focused) |
| `top_k_seed` | `int` | 5 | Best stage-1 configs used to seed stage 2 |
| `output_dir` | `str` | `"sweep_results"` | Base output directory |
| `featurizer` | `str\|None` | `None` | ONNX extractor path |
| `featurizer_type` | `str` | `"mfcc"` | Extractor type |
| `device` | `str` | `"auto"` | Device |
| `epochs_per_trial` | `int` | 5 | Epochs per candidate |
| `full` | `bool` | `False` | Expand search space |
| `n_demes` | `int` | 1 | Parallel island count (applied to both stages) |
| `timeout_minutes` | `float\|None` | `None` | Wall-clock timeout per stage |
| `target_f1` | `float\|None` | `None` | Early-stop F1 target per stage |
| `fitness_fn` | `str` | `"f1"` | Selection pressure transform for both stages |

**Returns:** `dict` with `best_config`, `best_score` (raw F1), `stage1`, `stage2`, `history`.
Each `history` entry carries a `stage` key (1 or 2) — `sweep.py:774-778`.

**Output:** `two_stage_results.json` — `sweep.py:795`.

---

## 6. Fitness Transforms (`_apply_fitness_fn`)

`_apply_fitness_fn(f1, fitness_fn)` — `sweep.py:23`

Internal helper; applied only for GA selection comparisons. **Reported `best_score` is always raw F1.**

| `fitness_fn` | Formula | Use case |
|---|---|---|
| `"f1"` | identity | Default; no selection pressure boost |
| `"exp_f1"` | `exp(f1)` | Steepens gradient above 0.9; helps when search plateaus at high F1 |
| `"double_exp_f1"` | `exp(exp(f1) - 1)` | Extreme pressure near the optimum |

Unknown `fitness_fn` values fall through to the identity branch (same as `"f1"`).

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
