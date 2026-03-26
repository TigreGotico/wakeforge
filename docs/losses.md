# Loss Functions

All losses are managed by `LossManager` -- `ww_trainer/loss.py:761`. Multiple losses can be weighted and combined. The manager handles triplet mining, embedding extraction, and dispatch.

---

## Overview

| Loss | Class | Line | Input Type | Category |
|------|-------|------|-----------|----------|
| BCE | `nn.BCEWithLogitsLoss` | (PyTorch built-in) | logits, labels | Classification |
| Focal | `FocalLoss` | `loss.py:395` | logits, labels | Classification |
| LabelSmoothing | `LabelSmoothingBCE` | `loss.py:441` | logits, labels | Classification |
| Triplet | `nn.TripletMarginLoss` | (PyTorch built-in) | anchor, pos, neg | Metric |
| SoftTriplet | `SoftTripletLoss` | `loss.py:86` | anchor, pos, neg | Metric |
| Pair | `nn.MarginRankingLoss` | (PyTorch built-in) | dist_ap, dist_an | Metric |
| CN2+1Pair | `CN2Plus1PairLoss` | `loss.py:13` | anchor, pos, negatives | Metric |
| Contrastive | `ContrastiveLoss` | `loss.py:126` | embeds, labels | Metric |
| LiftedStructure | `LiftedStructureLoss` | `loss.py:170` | embeds, labels | Metric |
| Angular | `AngularLoss` | `loss.py:221` | embeds, labels | Metric |
| ArcFace | `ArcFaceLoss` | `loss.py:471` | embeds, labels | Metric (learnable) |
| Center | `CenterLoss` | `loss.py:521` | embeds, labels | Metric (learnable) |
| NTXent | `NTXentLoss` | `loss.py:551` | embeds, labels | Contrastive |
| SupCon | `SupConLoss` | `loss.py:597` | embeds, labels | Contrastive |
| ProxyNCA | `ProxyNCALoss` | `loss.py:645` | embeds, labels | Metric (learnable) |
| MultiSimilarity | `MultiSimilarityLoss` | `loss.py:686` | embeds, labels | Metric |
| RPPL | `RobustProtoDiversityLoss` | `loss.py:281` | logits, labels, embeds | Composite |

---

## Classification Losses

### BCE (Binary Cross-Entropy)

Standard binary classification loss. Applied to raw logits via `BCEWithLogitsLoss`.

**Config:** `{"name": "bce", "weight": 1.0}`

**When to use:** Always include as baseline. Works alone for simple tasks.

**When NOT to use:** Severe class imbalance (use Focal instead). Never remove entirely -- most combos benefit from BCE as a stabilizer.

---

### FocalLoss -- `loss.py:395`

Down-weights easy examples: `FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)`.

`alpha` controls positive class weight. `gamma` controls focusing -- higher values focus more on hard examples. At `gamma=0`, this is weighted BCE.

**Config:** `{"name": "focal", "weight": 1.0, "alpha": 0.25, "gamma": 2.0}`

**When to use:** Imbalanced datasets (many more negatives than positives). When the model quickly learns easy negatives but struggles with hard ones.

**When NOT to use:** Balanced datasets. `gamma > 3` can make training unstable.

---

### LabelSmoothingBCE -- `loss.py:441`

Replaces hard labels {0, 1} with soft labels {smoothing, 1 - smoothing}. Prevents overconfident predictions.

**Config:** `{"name": "label_smoothing_bce", "weight": 1.0, "smoothing": 0.1}`

**When to use:** When the model overfits or produces overly confident predictions. Good default replacement for plain BCE.

**When NOT to use:** When you have very clean labels and want maximum discrimination.

---

## Triplet/Pair Metric Losses

These operate on embeddings (from `model.embed()`). `LossManager` handles triplet mining via `sample_triplets()` or `sample_semihard_triplets()` from `ww_trainer/utils.py`.

### Triplet -- `loss.py:788` (LossManager config)

Standard triplet margin loss: `max(0, d(a,p) - d(a,n) + margin)`.

**Config:** `{"name": "triplet", "weight": 0.5, "margin": 1.0}`

Mining type is set on `LossManager` init: `mining_type="semihard"` (default) or `"hard"`.

**When to use:** Improving embedding quality for few-shot or open-set wake word detection. Combine with BCE.

**When NOT to use:** Very small batches (< 8) where mining finds few valid triplets. Use SupCon instead for more stable training.

---

### SoftTripletLoss -- `loss.py:86`

Soft margin variant: `log(1 + exp(d(a,p) - d(a,n)))` via softplus. Gradient flows even when the hard margin is satisfied.

**Config:** `{"name": "soft_triplet", "weight": 0.5}`

**When to use:** When standard triplet loss plateaus too early. Smoother gradient landscape.

**When NOT to use:** Same batch size constraints as triplet loss.

---

### Pair (MarginRankingLoss) -- `loss.py:792`

Margin ranking loss on hard-mined anchor-positive vs anchor-negative distances. Uses `get_hard_pair_distances()`.

**Config:** `{"name": "pair", "weight": 0.5, "margin": 1.0}`

**When to use:** Simpler alternative to triplet loss. Pairs are easier to mine than triplets.

**When NOT to use:** When you need the anchor-positive-negative relationship explicitly.

---

### CN2Plus1PairLoss -- `loss.py:13`

(C_N,2 + 1)-pair loss (Lopez-Espejo et al., TASLP 2021). Considers all negative-negative pair distances in addition to anchor-positive and anchor-negative distances. Regularizes negative embedding spread.

**Config:** `{"name": "cn2pair", "weight": 0.5, "d_max": 2.0, "margin": 0.0}`

**When to use:** When negative samples are too similar to each other (collapsed embeddings). Encourages diversity among negatives.

**When NOT to use:** Small negative sample counts (< 3). Computationally heavier than triplet due to N^2 negative distances.

---

## Batch Contrastive Losses

These use all pairs in the batch -- no explicit mining needed.

### ContrastiveLoss -- `loss.py:126`

Classic contrastive loss: pull same-label pairs together, push different-label pairs apart beyond margin.

Positive term: `d(i,j)^2` for same-label pairs. Negative term: `max(0, margin - d(i,j))^2` for different-label pairs.

**Config:** `{"name": "contrastive", "weight": 0.5, "margin": 1.0}`

**When to use:** Simple metric learning baseline. Requires both positives and negatives in batch.

**When NOT to use:** When SupCon or NTXent is available (they are strictly better).

---

### LiftedStructureLoss -- `loss.py:170`

Uses all positive and negative pairs via log-sum-exp. More informative gradients than contrastive loss by considering all negatives simultaneously.

**Config:** `{"name": "lse", "weight": 0.5, "margin": 1.0}`

**When to use:** When you want richer gradient signal than contrastive loss. Medium-sized batches (16-64).

**When NOT to use:** Very large batches (log-sum-exp can overflow). Small batches with few positives.

---

### AngularLoss -- `loss.py:221`

Cosine-space margin loss: enforces `cos(A,P) > cos(A,N) + margin`. Uses semi-hard triplet mining on Euclidean distances, then applies angular constraint.

**Config:** `{"name": "angular", "weight": 0.5, "margin": 0.5}`

**When to use:** When angular separation matters more than Euclidean distance. Speaker verification-style tasks.

**When NOT to use:** When embeddings are not L2-normalized.

---

### NTXentLoss -- `loss.py:551`

SimCLR loss (Chen et al. 2020). Temperature-scaled cross-entropy treating each sample's same-label partners as positives and all others as negatives.

**Config:** `{"name": "ntxent", "weight": 0.5, "temperature": 0.07}`

**When to use:** Large batches (32+). Strong contrastive learning signal. Good with data augmentation.

**When NOT to use:** Small batches. `temperature` is sensitive -- too low collapses embeddings, too high weakens the signal.

---

### SupConLoss -- `loss.py:597`

Supervised Contrastive Loss (Khosla et al., NeurIPS 2020). Extension of SimCLR to supervised setting. All same-class samples are positives.

**Config:** `{"name": "supcon", "weight": 0.5, "temperature": 0.07}`

**When to use:** Best general-purpose metric loss. More stable than triplet. Works with any batch size >= 4 (needs at least 1 positive and 1 negative).

**When NOT to use:** Self-supervised settings (use NTXent instead).

---

### MultiSimilarityLoss -- `loss.py:686`

Multi-Similarity Loss (Wang et al., CVPR 2019). Mines informative pairs using three similarity criteria. Computes weighted log-sum-exp over hard positives and hard negatives separately.

**Config:** `{"name": "multi_similarity", "weight": 0.5, "alpha": 2.0, "beta": 50.0, "base": 0.5}`

`alpha` weights positive pairs, `beta` weights negative pairs, `base` is the margin for mining.

**When to use:** When you want state-of-the-art pair mining without manual tuning. Handles the mining internally.

**When NOT to use:** Very small batches where mining finds no hard pairs.

---

## Learnable Proxy/Center Losses

These losses have learnable parameters (class centers or proxies) updated during training.

### ArcFaceLoss -- `loss.py:471`

Additive angular margin in cosine space (Deng et al., CVPR 2019). Adds angular penalty `margin` (radians) to the target class logit. Uses 2 class centers (wake/not-wake).

**Config:** `{"name": "arcface", "weight": 0.5, "embed_dim": 128, "margin": 0.5, "scale": 30.0}`

**When to use:** When you want maximum inter-class separation. Gold standard for verification tasks.

**When NOT to use:** Without proper embedding normalization. When embed_dim doesn't match your model's embedding size.

---

### CenterLoss -- `loss.py:521`

Penalizes distance from embeddings to their class center (Wen et al., ECCV 2016). Centers are learnable parameters.

**Config:** `{"name": "center", "weight": 0.3, "embed_dim": 128, "num_classes": 2}`

**When to use:** Combine with BCE or ArcFace to reduce intra-class variation. Simple and effective regularizer.

**When NOT to use:** Alone (no inter-class push -- must combine with a discriminative loss).

---

### ProxyNCALoss -- `loss.py:645`

Learnable proxies replace triplet mining (Movshovitz-Attias et al., ICCV 2017). Each sample is compared to all class proxies via softmax. Converges faster than triplet loss.

**Config:** `{"name": "proxy_nca", "weight": 0.5, "embed_dim": 128, "num_classes": 2, "scale": 8.0}`

**When to use:** When triplet mining is unreliable (small batches, imbalanced data). Proxy-based learning is more stable.

**When NOT to use:** When you want fine-grained embedding structure (proxies collapse intra-class variation).

---

## Composite Loss

### RobustProtoDiversityLoss (RPPL) -- `loss.py:281`

Five-component loss designed for binary fixed-keyword wake-word detection under severe class imbalance (~94 % NWW).

**Components:**

| Term | Weight param | Description | Prior work |
|------|-------------|-------------|-----------|
| BCE | `alpha` | Standard binary cross-entropy | — |
| Proto-softmax | `beta` | Each sample classified against EMA wake prototype and NWW mean prototype | Snell et al. 2017 (Prototypical Networks) |
| Center loss | `delta` | Pulls wake embeddings toward the wake prototype | Wen et al. 2016 |
| Hard-neg diversity | `gamma` | Spreads apart negatives with cos-sim to wake prototype above `hard_div_threshold` | Boudiaf et al. 2020 |
| Proto-ranked consistency | `eta` | Under acoustic augmentation, each sample must still be correctly classified against the class prototypes (requires dataset `get_augmented`) | BYOL / MeanTeacher |

**Novel features vs. standard metric learning:**
- **EMA wake prototype** (`proto_ema_alpha=0.05`): the batch mean of ~2–5 wake embeddings is too noisy to use directly; the EMA stabilises over ~20 batches.
- **Hard-negative targeting** (`hard_div_threshold=0.1`): only spreads confusable negatives, not easy ones far from the wake cluster.
- **Proto-ranked consistency**: replaces L2 augmentation invariance with a stricter objective — the augmented embedding must remain on the correct side of the prototype boundary.
- **Warmup scheduling** (`warmup_epochs=5`): geometric terms ramp from 0 → 1 so they don't train against random-epoch-0 prototypes.

**Full config:**

```python
{"name": "rppl", "weight": 1.0,
 "tau": 0.1,              # prototype similarity temperature
 "K_neg_proto": 0,        # 0 = single mean NWW prototype; >1 splits into K groups
 "alpha": 1.0,            # BCE
 "beta": 1.0,             # proto-softmax
 "gamma": 0.5,            # diversity
 "delta": 0.1,            # center
 "eta": 0.5,              # consistency
 "warmup_epochs": 5,
 "proto_ema_alpha": 0.05,
 "hard_div_threshold": 0.1,
 "consistency_mode": "proto"}   # "proto" or "l2" (legacy ablation)
```

**MLflow metrics logged automatically when using RPPL:**

| Metric | Meaning |
|--------|---------|
| `rppl_bce` | BCE sub-loss per epoch |
| `rppl_proto` | Prototype contrastive sub-loss |
| `rppl_div` | Diversity sub-loss (negative = good, maximising distance) |
| `rppl_center` | Center loss sub-loss |
| `rppl_cons` | Consistency sub-loss |
| `rppl_geo_scale` | Warmup ramp (0 → 1 over `warmup_epochs`) |
| `rppl_proto_ema_norm` | L2 norm of the EMA wake prototype (should stabilise quickly) |

A 6-panel **RPPL dashboard** PNG is auto-generated every 5 epochs and at end-of-training, logged to the MLflow `rppl/` artifact folder. Use `train_rppl.py` as the dedicated experiment script.

**When to use:** Hard-negative mining regime with a large NWW pool; infinite training mode; any scenario where you want per-epoch visibility into embedding structure.

**When NOT to use:** Very small datasets where there are consistently < 2 wake samples per batch (EMA has no data to stabilise from). Use Focal or SupCon instead. The 5 weight hyperparameters can be hard to tune — `train_rppl.py` provides sane defaults.

---

## Recommended Combinations

| Scenario | Loss Config | Notes |
|----------|-------------|-------|
| Simple baseline | BCE alone | Start here |
| Imbalanced data | Focal (weight=1.0) | Replace BCE entirely |
| Better generalization | LabelSmoothing (1.0) + SupCon (0.3) | Smooth labels + embedding quality |
| Few-shot | BCE (0.5) + ArcFace (0.5) + Center (0.2) | Maximize separation with few examples |
| Production | Focal (0.5) + SupCon (0.3) + Center (0.2) | Robust to imbalance + good embeddings |
| Noisy labels | LabelSmoothing (1.0) + MultiSimilarity (0.3) | Tolerates label noise |
| Maximum accuracy | BCE (0.5) + ArcFace (0.3) + SupCon (0.2) | Heavy but effective |
| Large NWW pool + infinite training | RPPL (1.0) | Best when mining loop provides hard negatives every epoch |
