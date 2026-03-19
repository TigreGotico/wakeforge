# Search Strategy Guide

When to use each of the four hyperparameter search strategies in `ww_trainer/sweep.py`.

---

## Decision Tree

```
Do you need exhaustive coverage of a small space?
  YES → Grid Search (run_grid_search, sweep.py:202)

Is Optuna installed and your budget is 20-100 trials?
  YES → Optuna / Bayesian (run_sweep, sweep.py:20)

Is your search space large or multi-modal?
  YES, budget >100 → Genetic Search (run_genetic_search, sweep.py:345)
  YES, budget <100 → Random Search (run_random_search, sweep.py:276)

No strong preference?
  → Optuna (default recommendation)
```

---

## Strategy Comparison

| | Grid | Random | Optuna | Genetic |
|---|---|---|---|---|
| Function | `run_grid_search` | `run_random_search` | `run_sweep` | `run_genetic_search` |
| Line | `sweep.py:202` | `sweep.py:276` | `sweep.py:20` | `sweep.py:345` |
| Trials | All combos | User-specified | User-specified | pop_size * generations |
| Sampling | Exhaustive | Uniform | TPE (Bayesian) | Evolve + mutate |
| Resumable | No | No | Yes (SQLite) | No |
| Best for | <100 combos | Quick exploration | General purpose | Complex/multi-modal spaces |
| Output file | `grid_results.json` | `random_results.json` | `best_params.json` | `genetic_results.json` |
| Requires | -- | -- | `optuna` | -- |

---

## Default Search Space

`_build_search_space(full=False)` -- `sweep.py:113`:

| Parameter | Values | Count |
|-----------|--------|-------|
| `arch` | ffn, gru, cnn | 3 |
| `hidden_dim` | 64, 128, 256 | 3 |
| `lr` | 1e-4, 5e-4, 1e-3, 5e-3, 1e-2 | 5 |
| `batch_size` | 16, 32, 64 | 3 |
| `dropout` | 0.0, 0.1, 0.2, 0.3 | 4 |

Total: 540 combinations. Too large for grid; use Optuna or Random.

---

## Full Architecture Search (`full=True`)

`_build_search_space(full=True)` -- `sweep.py:131`:

Adds to the default space:

| Parameter | Values | Count |
|-----------|--------|-------|
| `featurizer_type` | mfcc, filterbank, sincnet, gammatone | 4 |
| `arch` | ffn, gru, cnn, bcresnet, tcresnet, dscnn, matchboxnet, res15, conformer, crnn | 10 |
| `loss` | bce, focal, label_smoothing_bce, supcon, arcface, ntxent | 6 |
| `n_features` | 13, 40, 64, 80 | 4 |

Combined with hyperparameters: ~500K+ combinations. Only feasible with Genetic or Random search.

```python
from ww_trainer.sweep import run_genetic_search

results = run_genetic_search(
    metadata_csv="dataset.csv",
    population_size=20,
    generations=10,
    output_dir="full_search",
    epochs_per_trial=5,
    full=True,
)
```

The evaluation function `_evaluate_config()` (`sweep.py:159`) maps each config to the correct `WakeWordTrainer` kwargs via `_FEAT_KWARGS` (`sweep.py:151-156`).

---

## Genetic Search Details

Algorithm per generation (`sweep.py:431-463`):

1. Evaluate all individuals (fitness = F1 score)
2. Select elite (top `elite_frac` fraction, default 0.2)
3. Breed via uniform crossover (`_crossover` -- `sweep.py:409`)
4. Mutate with probability `mutation_rate` (default 0.3) (`_mutate` -- `sweep.py:416`)

Total evaluations: `population_size * generations`.

**Tuning tips:**
- `population_size=20, generations=10` (200 evals) is a reasonable default.
- Increase `elite_frac` (e.g. 0.3) if convergence is too slow.
- Decrease `mutation_rate` (e.g. 0.1) in later runs to refine near a known good region.

---

## Post-Search Workflow

All strategies output the best config as JSON. Retrain with full epochs:

```bash
# Read best_params.json / genetic_results.json, then:
ww_trainer-train \
  --wake-word hey_jarvis \
  --metadata dataset.csv \
  --featurizer-type <best> --arch <best> \
  --hidden-dim <best> --lr <best> \
  --loss-type <best> \
  --epochs 50 --save-best --export-onnx \
  --output-dir ./models/final
```
