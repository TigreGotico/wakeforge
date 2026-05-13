# Robust Prototype Diversity Loss (RPPL) for Binary Wake-Word Detection

> **Status:** engineering experiment specification, not a results paper. The method below is well-defined and implemented (`ww_trainer/loss.py:RobustProtoDiversityLoss`); the per-component ablations and baseline comparisons in §6 list *expected* effects. Numbers will be filled in as runs complete. Treat any quantitative claim as a hypothesis until it appears in a results table.

## 1. Motivation and intuitions

Wake-word detection (WWD) is a binary, fixed-keyword problem with severe class imbalance: in a realistic data mix roughly 94 % of samples are non-wake-word (NWW), so a model that always predicts "not wake" achieves ~94 % accuracy. Accuracy is therefore a misleading metric; F1, EER, and FAR/FRR at an operating threshold are appropriate.

Standard binary cross-entropy (BCE) optimises the decision boundary but does not shape the embedding space. Two trained models can have identical BCE loss yet very different embedding geometries — one may cluster wake utterances tightly, the other may scatter them. In a hard-negative mining regime (where the NWW pool is very large and only confusable negatives are trained on), a well-structured embedding space directly helps: the model's uncertainty about confusable sounds is visible in the geometry, and the mining loop exploits this signal.

RPPL is a recipe that combines five well-known objectives so that the decision boundary *and* the embedding structure are optimised jointly. The intuitions driving each term:

1. **Anchor wake samples to a stable point.** With only 2–5 positives per batch, every batch-mean wake prototype is a different point. Train against an EMA-smoothed prototype instead, so center loss and proto-softmax see a consistent target across batches.
2. **Push the boundary, not the easy mass.** Easy negatives are already far from the wake cluster; only confusable negatives carry useful gradient. Apply diversity pressure only above a cosine-similarity threshold to the wake prototype.
3. **Make augmentation a classification constraint, not a distance constraint.** Per-sample L2 invariance can be satisfied by collapsing all embeddings; classifying the augmented view against the shared prototypes cannot.
4. **Don't train geometry against random epoch-0 prototypes.** Warm up the geometric terms linearly so they only become significant once the EMA prototype has stabilised.
5. **Keep the classifier honest.** BCE on logits stays at full weight from epoch 0 — the geometric terms regularise it, they do not replace it.

Every component below comes from prior work (§4). The contribution of this document is the *recipe and the experiment spec*, not the individual losses.

---

## 2. Loss Formulation

Let `z = L2_norm(embed(x))` be the normalised embedding of input `x`, and `logit = classifier(embed(x))` be the binary logit.

**Total loss:**

```
L_RPPL = alpha * L_bce
       + geo_scale * (beta  * L_proto
                    + gamma * L_div
                    + delta * L_center)
       + eta * L_cons
```

`geo_scale = min(1.0, epoch / warmup_epochs)` ramps the geometric terms from 0 to 1 over the first `warmup_epochs` epochs (default 5). BCE and consistency are active from epoch 0.

### 2.1 BCE

```
L_bce = BCE_with_logits(logit, label)
```

Standard binary cross-entropy. Optimises the decision boundary directly.

### 2.2 Proto-softmax (L_proto)

```
p_w = EMA_wake_prototype          (see §3 — stable across batches)
p_n = mean(z[neg])                (single negative prototype; K>1 splits into K groups)

sim_i = [z_i · p_w / tau,  log-sum-exp(z_i · p_n / tau)]
L_proto = cross_entropy(sim_i, label_i_binary)
```

Each sample is classified against the two prototypes. Pulls wake embeddings toward `p_w` and NWW embeddings toward `p_n` without requiring explicit triplet sampling.

Prior work: Snell et al. 2017 (Prototypical Networks) — here applied in a binary, online-prototype setting. In multi-class KWS, prototypical approaches are used for enrollment-based detection; applying a single shared wake prototype to a fixed-keyword binary problem is the specific adaptation.

### 2.3 Negative diversity (L_div) — hard-negative targeted, hinge form

```
# Only spread negatives that are confusable with wake
cos_to_wake = z[neg] · p_w
hard_neg = z[neg][cos_to_wake > hard_div_threshold]
# Hinge: pairs already > div_margin apart contribute zero gradient
L_div = mean(max(0, div_margin - pairwise_sq_dist(hard_neg)))
```

The hinge form bounds the gradient: once a pair of confusable negatives is further than `div_margin` apart in squared L2, it produces no further pressure. The earlier unbounded `-mean(pairwise_sq_dist)` formulation kept pushing already-separated pairs and could destabilise late-stage training.

If fewer than two negatives clear the threshold in a given batch, `L_div = 0` for that step (no fallback to easy negatives — falling back would defeat the targeting).

Prior work: spread-out regularisation (Boudiaf et al. 2020), standard DML hinge losses. The hard-negative targeting via cosine threshold is the implementation-specific bit; nothing about the formulation is novel.

### 2.4 Center loss (L_center)

```
L_center = mean(||z[pos] - p_w||^2)
```

Pulls wake embeddings toward their prototype. Encourages intra-class compactness, complementing the inter-class separation from `L_proto`.

Prior work: Wen et al. 2016 (Center Loss for face verification). Here applied to the binary wake/nonwake problem.

### 2.5 Proto-ranked consistency (L_cons)

```
z_aug = L2_norm(embed(augment(x)))
aug_sim = [z_aug · p_w / tau,  log-sum-exp(z_aug · p_n / tau)]
L_cons = cross_entropy(aug_sim, label_binary)
```

Under acoustic augmentation (noise, RIR, pitch shift), each sample's augmented embedding must still be *correctly classified* against the shared class prototypes. This is stricter than L2 invariance: a model can satisfy L2 consistency by moving both embeddings together without improving separability, whereas proto-ranked consistency requires the augmented embedding to remain on the correct side of the prototype boundary.

Prior work: BYOL (Grill et al. 2020), MeanTeacher (Tarvainen & Valpola 2017) use representation invariance under augmentation. The proto-ranked formulation (classification against shared prototypes rather than L2 distance to original embedding) is the specific novel contribution of this work.

---

## 3. EMA Wake Prototype

The batch mean `mean(z[wake_in_batch])` is computed over 2–5 wake samples per batch (given batch_size=16 and ~25 % positives). With 5 samples in 128-D space the estimate is unreliable — the direction changes significantly between batches, and `L_proto` and `L_center` train against an unstable target.

The EMA prototype smooths this noise:

```python
if not initialised:
    p_w_ema = mean(z[wake])     # cold start
else:
    p_w_ema = (1 - alpha) * p_w_ema + alpha * mean(z[wake]).detach()

p_w_stable = L2_norm(p_w_ema)
```

With `alpha=0.05` (the default), the EMA has a time constant of ~20 batches. The prototype drifts with the model as training progresses (because `detach()` means it doesn't directly participate in gradient computation), but does not oscillate at the batch frequency. `L_proto` and `L_center` both use `p_w_stable`.

The batch mean is only used to update the EMA; it is never directly used as a loss target.

---

## 4. Honest framing — what is and is not new

Every individual component of RPPL comes from prior work. The contribution of this work is a *recipe*: a packaging of five well-known terms tuned for the binary, fixed-keyword, severely-imbalanced KWS regime, with two small engineering adaptations and a warmup schedule.

**Components and their sources:**

| Component | Source |
|---|---|
| BCE | standard |
| Proto-softmax (online, 2-class) | Prototypical Networks (Snell et al. 2017) |
| Center loss | Wen et al. 2016 |
| Hinge negative-diversity (spread-out) | Boudiaf et al. 2020 and the DML hinge tradition |
| Representation consistency under augmentation | MeanTeacher (Tarvainen & Valpola 2017), BYOL (Grill et al. 2020), SupCon (Khosla et al. 2020) |
| EMA target for prototype/teacher | MoCo (He et al. 2020), MeanTeacher |
| Curriculum / warmup of metric terms | ArcFace and successors |

**Engineering adaptations specific to this implementation (not paper-worthy on their own):**

1. **EMA-stabilised single wake prototype.** With 2–5 positives per batch the batch-mean prototype is too noisy to use as a center-loss target; the EMA smooths it. This is the same idea as a momentum encoder, narrowed to a single class prototype.
2. **Proto-ranked consistency.** The augmented view is forced to classify correctly against the shared (clean) prototypes — a 2-class supervised reduction of SupCon-with-augmentation. Functionally stricter than per-sample L2 invariance because it cannot be satisfied by representation collapse, but conceptually nothing new.
3. **Hard-negative diversity targeting via cosine threshold.** A standard thresholded mining choice, motivated by the upstream hard-negative mining loop in this codebase.
4. **Linear warmup of geometric terms.** Standard curriculum to avoid training against the random epoch-0 prototype.

**Not yet demonstrated.** This document describes the method; the per-component ablation table in §6 lists *expected* effects, not measured ones. Treat RPPL as a working engineering recipe pending an ablation study against `BCE`, `Focal`, `BCE+SupCon`, and `BCE+ArcFace` on identical data and seeds. Until those numbers exist, no claim of superiority is made.

**Closest prior work:** CN₂⁺¹ (Zeng et al., "Contrastive Metric Learning for Small-Sample Wake Word Detection") uses metric learning for a similar task but is multi-class/enrollment-based. The binary, fixed-keyword setting addressed here has no enrollment step.

---

## 5. Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `alpha` | 1.0 | BCE weight — primary classification signal |
| `beta` | 1.0 | Proto-softmax weight |
| `gamma` | 0.5 | Diversity weight |
| `delta` | 0.1 | Center loss weight (small to avoid over-squishing) |
| `eta` | 0.5 | Consistency weight |
| `tau` | 0.1 | Prototype similarity temperature |
| `warmup_epochs` | 5 | Geometric term ramp-up duration |
| `proto_ema_alpha` | 0.05 | EMA decay (~20 batch time constant) |
| `hard_div_threshold` | 0.1 | Cos-sim threshold for hard-negative targeting |
| `div_margin` | 1.0 | Squared-L2 margin for hinge diversity (pairs further apart → 0 grad) |
| `consistency_mode` | "proto" | "proto" = proto-ranked CE; "l2" = MSE ablation |

---

## 6. MLflow Metrics

`train_rppl.py` logs these metrics per epoch in addition to standard F1/EER/AUC:

| Metric | Meaning | Good value |
|---|---|---|
| `embed_centroid_dist` | Euclidean distance between wake/NWW centroids | Increasing |
| `embed_fisher_ratio` | centroid_dist / mean intra-class spread | > 1.0, increasing |
| `embed_silhouette` | Cosine silhouette score [-1, 1] | > 0.5, increasing |
| `embed_pca_var_explained` | Variance in first 2 PCs | Diagnostic |
| `rppl_bce` | BCE sub-loss | Decreasing |
| `rppl_proto` | Proto-softmax sub-loss | Decreasing |
| `rppl_div` | Hinge diversity sub-loss | Decreasing toward 0 once hard negatives are > `div_margin` apart |
| `rppl_center` | Center loss sub-loss | Decreasing |
| `rppl_cons` | Consistency sub-loss | Decreasing |
| `rppl_geo_scale` | Warmup ramp value (0 → 1 over `warmup_epochs`) | Linear ramp, then 1.0 |
| `rppl_proto_ema_norm` | L2 norm of EMA wake prototype | Stabilises within ~20 batches |

### Experiment spec — ablations and baselines

This section is a *protocol*, not results. Each row defines a run; the rightmost columns are filled in once the run completes. Use identical seed, splits, and training schedule across all rows; vary only the loss config.

**Baselines (no RPPL):**

| Run | Loss config | F1 | EER | FAR @ FRR=1% | Notes |
|---|---|---|---|---|---|
| B1 | `bce` | — | — | — | reference |
| B2 | `focal(alpha=0.9, gamma=2)` | — | — | — | imbalance-only baseline |
| B3 | `bce + supcon(0.3)` | — | — | — | classifier + embedding baseline |
| B4 | `bce + arcface(0.5) + center(0.2)` | — | — | — | margin-based baseline |

**RPPL ablations** (drop one term at a time from full RPPL):

| Run | Config | F1 | EER | FAR @ FRR=1% | `embed_fisher_ratio` | Expected vs. full |
|---|---|---|---|---|---|---|
| A0 | Full RPPL (defaults) | — | — | — | — | reference |
| A1 | `beta=0` (no proto-softmax) | — | — | — | — | wake cluster less compact |
| A2 | `gamma=0` (no diversity) | — | — | — | — | NWW spread reduced |
| A3 | `delta=0` (no center) | — | — | — | — | intra-class variance up |
| A4 | `eta=0` (no consistency) | — | — | — | — | aug-robustness drops |
| A5 | `consistency_mode="l2"` | — | — | — | — | weaker than proto-ranked |
| A6 | `warmup_epochs=0` | — | — | — | — | early instability expected |
| A7 | `proto_ema_alpha=1.0` (no EMA) | — | — | — | — | proto noise up |
| A8 | `K_neg_proto=4` (k-means NWW) | — | — | — | — | possibly helps multi-modal NWW |

**Decision rules** (set in advance to avoid post-hoc rationalisation):

- If full RPPL (A0) does not beat the best of {B1, B2, B3, B4} on F1 *and* FAR@FRR=1% on the held-out test set, the recipe is not justified as a default.
- If A1–A7 each match or beat A0, the corresponding term should be dropped.
- If `rppl_div` is 0 for > 50 % of training steps, lower `hard_div_threshold` and rerun A0.

---

## 7. Known limitations and open questions

Limitations (true regardless of results):

1. **Augmented forward pass cost.** Proto-ranked consistency requires a second forward pass per batch. On CPU this roughly doubles batch time. Disable with `eta=0` for faster ablations.
2. **`hard_div_threshold` sensitivity.** If the threshold is too high, no hard negatives are targeted and `L_div` is 0. Monitor `rppl_div` in MLflow; if it stays at 0 reduce the threshold (see §6 decision rules).
3. **EMA cold start.** For the first ~20 batches the EMA prototype is noisy. Warmup of geometric terms mitigates this; the EMA itself updates only in `model.train()` mode so validation batches do not drift the target.
4. **Single wake-word per model.** A single shared wake prototype is appropriate for fixed-keyword detection only. For multi-keyword or open-vocabulary KWS, a prototype per keyword is needed (standard Prototypical Networks regime).
5. **Hyperparameter count.** Eleven knobs is a lot. Defaults are educated guesses, not tuned values; expect the experiment in §6 to collapse this surface (e.g. by dropping a term entirely).

Open questions to resolve with results (revise this document once answered):

- Does RPPL beat `bce + supcon` on F1/EER at fixed compute? If not, RPPL is not justified as a default.
- Is the EMA prototype meaningfully different from a per-batch mean *after* warmup? Compare A0 vs A7.
- Does proto-ranked consistency actually beat L2 consistency in practice, or only in theory? Compare A0 vs A5.
- Do k-means negative prototypes help when the NWW pool is multi-modal (music vs speech vs noise)? Compare A0 vs A8.

This document is intended to be updated, not replaced, once results land — sections 1–5 are stable; §6 will gain numbers and §7 will shrink as questions are answered.
