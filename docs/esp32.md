# ESP32 Wake Word Deployment

Ultra-tiny wake word models targeting ESP32 (520 KB RAM, 4 MB flash).

## Tiers

All ESP32 tiers use MFCC + FFN (no recurrence, no attention).

| Tier | Hidden | n_mfcc | Max Params | Max Size (int8) | Use Case |
|------|--------|--------|-----------|-----------------|----------|
| `esp32_nano` | 16 | 13 | 1,024 | 1.0 KB | Absolute minimum; keyword presence only |
| `esp32_sweet` | 64 | 13 | 10,240 | 10.0 KB | Best accuracy/size tradeoff |
| `esp32_max` | 128 | 13 | 51,200 | 50.0 KB | Maximum ESP32 budget |

Source: `TierConfig` — `ww_trainer/tiers.py:105-140`

### Param Count Formula

`FfnClassifierHead(n_mfcc, hidden_dim)` = `n_mfcc * hidden_dim + hidden_dim + hidden_dim + 1`

- Linear(n_mfcc, hidden_dim): `n_mfcc * hidden_dim + hidden_dim`
- Linear(hidden_dim, 1): `hidden_dim + 1`

`_estimate_ffn_params` — `ww_trainer/sweep.py`

### Smallest Viable Model

MFCC-10 + FFN-8 = **97 params** = 0.1 KB int8. See `examples/34_esp32_nano.py`.

## SizeAwareLoss

`SizeAwareLoss` — `ww_trainer/loss.py`

Wraps any base loss with two penalties:

1. **L1 sparsity**: `l1_weight * sum(|params|)` — pushes weights toward zero
2. **Size penalty**: `size_weight * max(0, n_params/budget - 1)` — penalizes exceeding budget

```python
from ww_trainer.loss import SizeAwareLoss

loss_fn = SizeAwareLoss(
    base_loss=nn.BCEWithLogitsLoss(),
    l1_weight=1e-5,
    size_weight=0.1,
    param_budget=1024,
)
loss = loss_fn(logits, labels, model)
```

Also available via `LossManager` as `{"name": "size_aware", "param_budget": 1024}`.

## Micro Genetic Search

`run_micro_search` — `ww_trainer/sweep.py`

Genetic algorithm with composite fitness constrained to ESP32-viable architectures.

**Fitness**: `accuracy_weight * f1 + size_weight * (1 - params/budget)`

- Models exceeding the tier's `max_params` are hard-rejected (fitness = -1)
- Search space auto-constrained per tier via `_build_micro_search_space`
- Only FFN architectures and MFCC features explored

```python
from ww_trainer.sweep import run_micro_search

result = run_micro_search(
    metadata_csv="dataset.csv",
    tier_name="esp32_sweet",
    population_size=16,
    generations=8,
    accuracy_weight=0.7,
    size_weight=0.3,
)
```

## Examples

| Script | Description |
|--------|-------------|
| `examples/34_esp32_nano.py` | Grid of sub-1KB configs with param counts and smoke tests |
| `examples/35_esp32_genetic_search.py` | CLI for genetic search across ESP32 tiers |
| `examples/36_size_aware_training.py` | SizeAwareLoss training loop demo |
