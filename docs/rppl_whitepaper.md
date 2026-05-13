# Robust Prototype Diversity Loss (RPPL) for Binary Wake-Word Detection

## 1. Introduction

Wake-word detection (WWD) is a binary, fixed-keyword problem with severe class imbalance: in a realistic data mix roughly 94 % of samples are non-wake-word (NWW), so a model that always predicts "not wake" achieves ~94 % accuracy. Accuracy is therefore a misleading metric; F1, EER, and FAR/FRR at an operating threshold are appropriate.

Standard binary cross-entropy (BCE) optimises the decision boundary but does not shape the embedding space. Two trained models can have identical BCE loss yet very different embedding geometries — one may cluster wake utterances tightly, the other may scatter them. In a hard-negative mining regime (where the NWW pool is very large and only confusable negatives are trained on), a well-structured embedding space directly helps: the model's uncertainty about confusable sounds is visible in the geometry, and the mining loop exploits this signal.

RPPL combines five objectives to jointly optimise the decision boundary and the embedding structure. Most components are drawn from prior work; the novelty is their combination for *binary fixed-keyword KWS with severe class imbalance*, plus two specific adaptations described in §4.

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

### 2.3 Negative diversity (L_div) — hard-negative targeted

```
# Only spread negatives that are confusable with wake
cos_to_wake = z[neg] · p_w
hard_neg = z[neg][cos_to_wake > hard_div_threshold]
L_div = -mean(pairwise_sq_dist(hard_neg))     (maximise pairwise distance)
```

Easy negatives (silence, music, speech far from wake) are already distant from `p_w`; pushing them further wastes gradient. Targeting only confusable negatives (those with positive cosine similarity to `p_w`) concentrates the diversity pressure where it matters: the confusable region around the wake cluster.

Prior work: spread-out regularisation (Boudiaf et al. 2020), DML literature. The hard-negative targeting is specific to this implementation.

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

## 4. What Is and Is Not Novel

**Not novel (prior work applied):**
- BCE: standard
- Proto-softmax: Prototypical Networks (Snell et al. 2017), applied online
- Center loss: Wen et al. 2016
- Diversity / spread-out regularisation: Boudiaf et al. 2020 and DML literature
- Representation consistency under augmentation: BYOL (Grill 2020), MeanTeacher (Tarvainen 2017)

**Genuine novelty (specific to this work):**
1. **EMA wake prototype for small-batch binary KWS.** The noise problem with batch-mean prototypes is specific to small-batch, severely imbalanced data (2–5 positive samples per batch). EMA stabilisation is not commonly described in the KWS metric-learning literature.
2. **Proto-ranked consistency.** Replacing L2 augmentation invariance with prototype-classification consistency is a stricter and more semantically grounded objective. Under L2 consistency a model can "cheat" by collapsing all embeddings together; proto-ranked consistency prevents this by requiring correct prototype classification. This formulation is not found in prior KWS work.
3. **Hard-negative diversity targeting.** Applying diversity regularisation only to confusable negatives (those with positive cosine similarity to the wake prototype) is specific to the mining-loop setting of this system.
4. **Warmup scheduling for geometric terms.** Ramping geometric losses avoids the well-known instability of training against random-epoch-0 prototypes.

**Closest prior work:** CN₂⁺¹ (Zeng et al., "Contrastive Metric Learning for Small-Sample Wake Word Detection") uses metric learning for a similar task but is multi-class/enrollment-based. RPPL targets the binary fixed-keyword setting where no enrollment exists.

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
| `rppl_div` | Diversity sub-loss | Negative, magnitude increasing |
| `rppl_center` | Center loss sub-loss | Decreasing |
| `rppl_cons` | Consistency sub-loss | Decreasing |

### Ablation expectations

| Config | Expected effect on `embed_fisher_ratio` |
|---|---|
| BCE only (`--loss bce`) | Low, flat after convergence |
| RPPL full | Higher than BCE, increasing through training |
| RPPL `--no-proto` (beta=0) | Fisher ratio lower; wake cluster less compact |
| RPPL `--no-div` (gamma=0) | Fisher ratio lower; NWW cluster not spread |
| RPPL `consistency_mode=l2` | Similar to proto but slightly lower silhouette |

---

## 7. Limitations

1. **Augmented forward pass cost.** Proto-ranked consistency requires a second forward pass per batch (the augmented embeddings). On CPU this roughly doubles batch time. Disable with `--eta 0` for faster ablations.
2. **`hard_div_threshold` sensitivity.** If the threshold is too high, no hard negatives are targeted and `L_div` goes to zero. Monitor `rppl_div` in MLflow; if it stays at 0 reduce the threshold.
3. **EMA cold start.** For the first ~20 batches the EMA prototype is noisy. The warmup schedule mitigates this — geometric terms are scaled down while the prototype stabilises.
4. **Single wake-word per model.** RPPL's single shared wake prototype is appropriate for fixed-keyword detection only. For multi-keyword or open-vocabulary KWS, a prototype per keyword is needed (standard Prototypical Networks regime).
