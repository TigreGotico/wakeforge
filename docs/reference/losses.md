# Loss Functions

All losses are managed by `LossManager` — `ww_trainer/loss.py:1078`. Multiple losses can be weighted and combined; the manager handles triplet mining, embedding extraction, and dispatch.

Each section below follows the same shape:

- **Intuition** — what the loss is trying to do in plain terms.
- **Theory** — the formula and the key reference.
- **Config** — the dict you pass to `LossManager`.
- **When to use / When NOT to use** — practical guidance.

For background on the two big families:

- **Classification losses** operate on the model's scalar logit and shape the *decision boundary*. They answer "is this wake or not?".
- **Metric / contrastive losses** operate on the model's embedding vector and shape the *geometry of the representation*. They answer "are these two clips the same word?". A well-shaped embedding space makes downstream hard-negative mining and few-shot adaptation easier.

A typical wake-word recipe combines one classification loss with one metric loss — the classifier sharpens the boundary, the metric loss keeps the embedding from collapsing.

---

## Overview

| Loss | Class | Line | Input Type | Category |
|------|-------|------|-----------|----------|
| BCE | `nn.BCEWithLogitsLoss` | (PyTorch built-in) | logits, labels | Classification |
| Focal | `FocalLoss` | `loss.py:482` | logits, labels | Classification |
| LabelSmoothing | `LabelSmoothingBCE` | `loss.py:530` | logits, labels | Classification |
| Triplet | `nn.TripletMarginLoss` | (PyTorch built-in) | anchor, pos, neg | Metric |
| SoftTriplet | `SoftTripletLoss` | `loss.py:87` | anchor, pos, neg | Metric |
| Pair | `nn.MarginRankingLoss` | (PyTorch built-in) | dist_ap, dist_an | Metric |
| CN2+1Pair | `CN2Plus1PairLoss` | `loss.py:14` | anchor, pos, negatives | Metric |
| Contrastive | `ContrastiveLoss` | `loss.py:127` | embeds, labels | Metric |
| LiftedStructure | `LiftedStructureLoss` | `loss.py:171` | embeds, labels | Metric |
| Angular | `AngularLoss` | `loss.py:222` | embeds, labels | Metric |
| ArcFace | `ArcFaceLoss` | `loss.py:562` | embeds, labels | Metric (learnable) |
| Center | `CenterLoss` | `loss.py:617` | embeds, labels | Metric (learnable) |
| NTXent | `NTXentLoss` | `loss.py:652` | embeds, labels | Contrastive |
| SupCon | `SupConLoss` | `loss.py:698` | embeds, labels | Contrastive |
| ProxyNCA | `ProxyNCALoss` | `loss.py:746` | embeds, labels | Metric (learnable) |
| MultiSimilarity | `MultiSimilarityLoss` | `loss.py:792` | embeds, labels | Metric |
| HALO | `HALOLoss` | `loss.py:867` | embeds, labels | Classification |
| SizeAware | `SizeAwareLoss` | `loss.py:987` | logits, labels, model | MCU-regularizer |
| RPPL | `RobustProtoDiversityLoss` | `loss.py:282` | logits, labels, embeds | Composite |

---

## Classification Losses

These act on the scalar logit and directly minimise classification error.

### BCE (Binary Cross-Entropy)

**Intuition.** The textbook "is it wake or not?" loss. For each clip, the model outputs a probability; BCE penalises the gap between that probability and the true {0, 1} label. Symmetric — positive and negative errors are weighted equally.

**Theory.** `L = -[y log σ(z) + (1-y) log(1-σ(z))]`, applied through `BCEWithLogitsLoss` for numerical stability (log-sum-exp trick on the logits rather than calling `sigmoid` first). Equivalent to maximum-likelihood under a Bernoulli model.

**Config:** `{"name": "bce", "weight": 1.0}`

**When to use:** Always include as a baseline. Most other recipes benefit from BCE as a stabiliser — it provides a steady gradient on the actual classification objective while metric losses shape the embedding.

**When NOT to use:** With severe class imbalance and *no other weighting* — BCE alone will collapse onto "always predict negative". Reach for Focal, weighted BCE via `pos_weight`, or RPPL in that case. Don't remove it entirely from combos unless you have a strong replacement.

---

### FocalLoss — `loss.py:482`

**Intuition.** Standard BCE wastes gradient on easy examples. Once the model is confident-and-correct on the millions of obvious non-wake clips, those gradients add noise without improving the decision boundary. Focal loss multiplies BCE by `(1 - p_t)^γ` — a confident-correct prediction (p_t close to 1) gets its loss almost zeroed out, while a misclassified hard example keeps full weight. The `alpha` term separately re-weights the positive class to counter prior imbalance.

**Theory.** `FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)` (Lin et al., ICCV 2017, "Focal Loss for Dense Object Detection"). At `γ=0` reduces to weighted BCE. `α` in this implementation defaults to 0.90 (positive-class up-weight) — the paper's 0.25 default down-weights positives and is wrong for binary KWS with negative-heavy data.

**Config:** `{"name": "focal", "weight": 1.0, "alpha": 0.90, "gamma": 2.0}`

**When to use:** Severe class imbalance (>10:1 negative:positive); when the loss curve plateaus despite the model still missing rare hard cases; when you cannot afford explicit hard-negative mining.

**When NOT to use:** Balanced datasets — focal then just slows learning. Very high `γ > 3` makes training spiky and unstable. If you already mine hard negatives every epoch, Focal's hard-example focus becomes redundant.

---

### LabelSmoothingBCE — `loss.py:530`

**Intuition.** Hard labels {0, 1} push the model to drive logits to ±∞ to minimise loss, even when the data doesn't justify that certainty. Soft labels (e.g. {0.1, 0.9}) cap the gradient: once the model is "smoothing-confident", it stops sharpening. Result: better-calibrated probabilities and less overconfident wake triggers on weird audio.

**Theory.** `y_smooth = y · (1 - ε) + 0.5 · ε`; standard BCE on the smoothed target. Originally proposed for ImageNet classification (Szegedy et al., 2016, "Rethinking the Inception Architecture"). Equivalent to adding a small KL penalty toward the uniform distribution.

**Config:** `{"name": "label_smoothing_bce", "weight": 1.0, "smoothing": 0.1}`

**When to use:** When the model overfits to training labels, produces overconfident scores, or shows poor probability calibration in deployment. Good default replacement for plain BCE when production logging shows P(wake)=0.999 on borderline clips.

**When NOT to use:** When labels are very clean and you want maximum logit separation (e.g. for a downstream threshold sweep). When combined with strong margin losses (ArcFace etc.) that already cap confidence — double smoothing slows convergence.

---

## Triplet / Pair Metric Losses

These act on embeddings and require *positive* and *negative* examples per anchor. `LossManager` handles mining via `sample_triplets()` / `sample_semihard_triplets()` from `ww_trainer/utils.py`.

The shared intuition: pull same-class embeddings together, push different-class embeddings apart, but only by a *margin* — once a triplet is "comfortably correct", contribute zero gradient. This concentrates training on the hard cases at the boundary, which is exactly where errors happen.

### Triplet — PyTorch `nn.TripletMarginLoss`

**Intuition.** Pick an anchor wake clip, one other wake clip (positive), one non-wake clip (negative). Penalise the model whenever the anchor is closer to the negative than the positive by less than `margin`. Geometrically, you carve a margin-sized buffer around every positive pair.

**Theory.** `L = max(0, d(a, p) - d(a, n) + margin)` (Schroff et al., CVPR 2015, "FaceNet"). Mining matters: random triplets are usually trivially correct and yield no gradient. "Semi-hard" mining picks negatives that are inside the margin but still further than the positive — the sweet spot for stable learning.

**Config:** `{"name": "triplet", "weight": 0.5, "margin": 1.0}` (mining set at `LossManager` init: `mining_type="semihard"` or `"hard"`).

**When to use:** When you want explicit control over the margin and have batches large enough (≥ 16) for mining to find valid triplets. Useful for few-shot or open-set scenarios where you may later add new wake words.

**When NOT to use:** Very small batches (< 8) — too few valid triplets, gradient becomes noisy. Use SupCon or contrastive losses instead, which use all batch pairs.

---

### SoftTripletLoss — `loss.py:87`

**Intuition.** The hard hinge `max(0, ·)` of standard triplet has a discontinuity: a triplet that is "just satisfied" produces zero gradient even if it could easily be improved. The soft variant replaces the hinge with `softplus`, giving a smooth gradient everywhere.

**Theory.** `L = log(1 + exp(d(a, p) - d(a, n)))` — softplus of the triplet violation, no explicit margin. Equivalent to a logistic ranking loss over `d(a, n) - d(a, p)`.

**Config:** `{"name": "soft_triplet", "weight": 0.5}`

**When to use:** When hard triplet plateaus too early or the loss curve goes flat with many "satisfied" triplets but mediocre test accuracy. Smoother gradients help small batches.

**When NOT to use:** Same batch size limits as standard triplet. When you actually *want* the model to ignore already-correct triplets (e.g. very large batches with mining that finds many valid hard negatives) — the hard hinge is preferable.

---

### Pair (`nn.MarginRankingLoss`)

**Intuition.** Like triplet but you don't need a fixed anchor — just supply a positive-pair distance and a negative-pair distance and ask "is the positive closer by at least `margin`?". Easier to mine because you don't need three correlated examples.

**Theory.** `L = max(0, d_ap - d_an + margin)` evaluated on hard-mined pairs (`get_hard_pair_distances`). Mathematically the same hinge as triplet — the difference is *what gets sampled* upstream.

**Config:** `{"name": "pair", "weight": 0.5, "margin": 1.0}`

**When to use:** As a simpler, sturdier alternative to triplet when triplet mining is fragile. Common in speaker-verification style pipelines.

**When NOT to use:** When you genuinely need anchor-relative structure (e.g. specific wake clip as a reference). Pair loss treats all positives symmetrically.

---

### CN2Plus1PairLoss — `loss.py:14`

**Intuition.** Most pair/triplet losses only worry about pulling positives together and pushing one negative away. CN₂⁺¹ additionally pushes *negatives apart from each other* — preventing the model from collapsing all non-wake clips onto a single point. For wake-word detection that matters: if "TV chatter" and "fridge hum" embed to the same vector, your model has no headroom to learn fine-grained NWW structure.

**Theory.** (C_N,2 + 1)-pair loss (López-Espejo et al., TASLP 2021, "Improved External Speaker-Robust Keyword Spotting for Hearing Assistive Devices"). Adds a regulariser over all `C(N, 2)` negative-negative pair distances on top of the standard anchor-positive / anchor-negative terms. The `d_max` parameter caps the negative repulsion so it doesn't dominate.

**Config:** `{"name": "cn2pair", "weight": 0.5, "d_max": 2.0, "margin": 0.0}`

**When to use:** When inspecting embeddings (e.g. PCA in MLflow) shows negatives collapsed into one cluster; when the NWW pool is acoustically diverse (music + speech + noise) and you want the embedding to reflect that.

**When NOT to use:** Tiny batches (< 3 negatives) — there are no negative-negative pairs to regularise. Computationally heavier than triplet (`O(N²)` negative distances).

---

## Batch Contrastive Losses

These use *all* pairs in the batch — no explicit mining needed. Cheaper in code complexity, often better in practice.

### ContrastiveLoss — `loss.py:127`

**Intuition.** The original metric-learning loss. Two embeddings of the same class? Pull them together. Two of different classes? Push them apart, but only up to `margin` — past that, leave them alone (no need to push to infinity).

**Theory.** `L_pos = d(i, j)²` for same-label pairs, `L_neg = max(0, margin - d(i, j))²` for different-label pairs (Hadsell et al., CVPR 2006, "Dimensionality Reduction by Learning an Invariant Mapping"). The squared form gives smoother gradients than absolute distance.

**Config:** `{"name": "contrastive", "weight": 0.5, "margin": 1.0}`

**When to use:** Simple, robust baseline for metric learning. Works at any batch size that contains both classes.

**When NOT to use:** When SupCon or NTXent are options — they are strictly more informative (use log-sum-exp over all negatives rather than per-pair hinges).

---

### LiftedStructureLoss — `loss.py:171`

**Intuition.** Contrastive loss only considers one pair at a time. Lifted Structure considers *all* negatives per positive pair simultaneously via log-sum-exp — the gradient knows about the hardest negative for each positive, automatically.

**Theory.** `L = max(0, log(Σ exp(margin - d_an)) + d_ap)²` (Song et al., CVPR 2016, "Deep Metric Learning via Lifted Structured Feature Embedding"). The log-sum-exp acts as a smooth max over negatives.

**Config:** `{"name": "lse", "weight": 0.5, "margin": 1.0}`

**When to use:** Medium batches (16–64) when you want richer gradient signal than contrastive without paying the mining cost of triplet.

**When NOT to use:** Very large batches — log-sum-exp can overflow without careful temperature scaling. Very small batches with few positives provide insufficient signal.

---

### AngularLoss — `loss.py:222`

**Intuition.** Cosine similarity, not Euclidean distance, often matches how voice embeddings actually behave — scale doesn't matter, direction does. Angular loss enforces a margin in *angle* between positive and negative pairs.

**Theory.** Enforces `cos(a, p) > cos(a, n) + margin` on semi-hard mined triplets (Wang et al., ICCV 2017, "Deep Metric Learning with Angular Loss"). Margin is in cosine units, typically 0.3–0.5.

**Config:** `{"name": "angular", "weight": 0.5, "margin": 0.5}`

**When to use:** Speaker-verification-style tasks; when embeddings are L2-normalised and you want the geometry to live on the unit sphere.

**When NOT to use:** If the model does not L2-normalise its embeddings — angular constraints become meaningless on unnormalised vectors.

---

### NTXentLoss — `loss.py:652`

**Intuition.** Treat the batch as a self-supervised problem: each sample's same-label partners are "positives", everyone else is a negative. A temperature-scaled cross-entropy then sharpens the contrast. Used heavily in self-supervised pretraining (SimCLR).

**Theory.** `L_i = -log( exp(s_i,j / τ) / Σ_k exp(s_i,k / τ) )` for positive pair (i, j) (Chen et al., ICML 2020, "SimCLR"). Temperature `τ` controls how sharply hard negatives dominate.

**Config:** `{"name": "ntxent", "weight": 0.5, "temperature": 0.07}`

**When to use:** Large batches (32+); when you have data augmentation generating positive pairs cheaply; when you want a strong "spread everything apart" pressure.

**When NOT to use:** Small batches — too few negatives, signal collapses. Temperature is finicky: too low → embeddings collapse, too high → no gradient. Start at 0.07 and only change if metrics demand it.

---

### SupConLoss — `loss.py:698`

**Intuition.** SimCLR but supervised: instead of "augmented views of me are my positives", *all same-class samples in the batch* are positives. For binary KWS this means "all wake clips in the batch attract each other; all non-wake clips attract each other; the two clusters repel".

**Theory.** Supervised Contrastive Loss (Khosla et al., NeurIPS 2020). Generalises NTXent to multiple positives per anchor. Empirically more stable than triplet loss and works at small batches as long as both classes are present.

**Config:** `{"name": "supcon", "weight": 0.5, "temperature": 0.07}`

**When to use:** Default "good metric loss" — pair with BCE and you have a strong baseline. Works at any batch size ≥ 4 with both classes present.

**When NOT to use:** True self-supervised pretraining without labels (use NTXent instead). When you specifically want anchor-relative structure (use triplet).

---

### MultiSimilarityLoss — `loss.py:792`

**Intuition.** Pair-based losses depend heavily on which pairs you sample. MS-Loss removes the mining knob: it weights every pair according to three similarity criteria (self-similarity, positive-relative, negative-relative) and computes a weighted log-sum-exp over hard positives and hard negatives separately. Effectively self-mining.

**Theory.** Multi-Similarity Loss (Wang et al., CVPR 2019, "Multi-Similarity Loss with General Pair Weighting"). `α` controls positive-pair pulling strength, `β` controls negative-pair pushing strength, `base` is the soft margin.

**Config:** `{"name": "multi_similarity", "weight": 0.5, "alpha": 2.0, "beta": 50.0, "base": 0.5}`

**When to use:** When you want strong metric-learning behaviour without tuning triplet mining; when batches are 32+ and contain many pair candidates.

**When NOT to use:** Very small batches — the mining criteria find nothing useful. The three hyperparameters interact non-trivially; start from defaults.

---

## Learnable Proxy / Center Losses

These keep *learnable parameters* (class centers or proxies) updated during training. Conceptually they cache "what does a wake embedding look like, on average" so you don't have to recompute it from a batch.

### ArcFaceLoss — `loss.py:562`

**Intuition.** Standard softmax classifiers separate classes by a flat hyperplane. ArcFace adds an *angular margin* — the wake class doesn't just need to win, it needs to win by a few degrees of angular separation on the unit sphere. Produces dramatically more compact clusters and is the de-facto standard in face/speaker verification.

**Theory.** `L = -log( exp(s · cos(θ_y + m)) / [exp(s · cos(θ_y + m)) + Σ exp(s · cos(θ_j))] )` (Deng et al., CVPR 2019, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"). `s` is a scale (logit temperature), `m` is the angular margin in radians (0.5 ≈ 28°). Class centers are learnable, one per class.

**Config:** `{"name": "arcface", "weight": 0.5, "embed_dim": 128, "margin": 0.5, "scale": 30.0}`

**When to use:** When you want maximum inter-class separation (verification-style tasks); when you plan to threshold cosine similarity at inference for a yes/no decision.

**When NOT to use:** Without L2-normalised embeddings — the angular interpretation breaks. If `embed_dim` doesn't match the model's embedding output, training raises at the first forward pass.

---

### CenterLoss — `loss.py:617`

**Intuition.** BCE/softmax shape inter-class separation but ignore intra-class variance. Center loss adds the missing piece: keep each sample close to its class's learnable center. Combined with a discriminative loss, you get tight clusters separated by clean boundaries.

**Theory.** `L_center = ½ Σ ||x_i - c_{y_i}||²` (Wen et al., ECCV 2016, "A Discriminative Feature Learning Approach for Deep Face Recognition"). Class centers `c` are learnable parameters, updated via EMA-like averaging in practice.

**Config:** `{"name": "center", "weight": 0.3, "embed_dim": 128, "num_classes": 2}`

**When to use:** Combine with BCE or ArcFace to reduce intra-class scatter. Cheap, robust regulariser.

**When NOT to use:** Alone — no inter-class push, so embeddings will just collapse to one point. Always pair with a discriminative loss.

---

### ProxyNCALoss — `loss.py:746`

**Intuition.** Triplet mining is fragile: many triplets are uninformative, mining heuristics differ, batch composition matters. Proxy-NCA sidesteps the problem by keeping one *learnable proxy* per class — each sample is compared to all class proxies via softmax. No per-batch mining; convergence is faster.

**Theory.** Each sample's loss is the NCA objective over class proxies: `L = -log( exp(-d(x, p_y)) / Σ exp(-d(x, p_j)) )` (Movshovitz-Attias et al., ICCV 2017, "No Fuss Distance Metric Learning using Proxies"). `scale` warms up the softmax temperature.

**Config:** `{"name": "proxy_nca", "weight": 0.5, "embed_dim": 128, "num_classes": 2, "scale": 8.0}`

**When to use:** When triplet mining is unreliable (small batches, imbalanced data); when you want fast, stable metric learning without mining knobs.

**When NOT to use:** When you need fine-grained intra-class structure — proxies tend to collapse each class to a single point, which is a feature for verification but a bug if you want to detect sub-modes of "wake" (different speakers, etc.).

---

### HALOLoss — `loss.py:867`

**Intuition.** Standard softmax classifies by linear similarity to a weight vector. HALO replaces that with *squared Euclidean distance to a learnable centroid*, and adds an extra "abstain" sink class at the origin. The result: clips that look like nothing the model has seen get strongly pushed toward "abstain" rather than confidently misclassified. Better calibration and out-of-distribution rejection.

**Theory.** `p(y | x) ∝ exp(-||e - c_y||² / γ)` over `K+1` classes including an abstain centroid at the origin. Optional self-distillation (`distill=True`) regularises with soft targets; learnable `γ` adapts the softmax temperature.

**Config:**

```python
{"name": "halo", "weight": 1.0,
 "emb_dims": 64,        # must match model.embed() output dim
 "num_classes": 2,
 "learn_gamma": True,
 "distill": True,
 "label_smoothing": 0.1}
```

**When to use:** When you need well-calibrated confidence scores and aggressive rejection of non-wake-word audio (e.g. far-field deployments hearing arbitrary noise). Strong alternative to BCE in production.

**When NOT to use:** When `emb_dims` doesn't match the model's embedding size — fails at first forward pass. When the model has no `embed()` method.

---

## Composite Loss

### RobustProtoDiversityLoss (RPPL) — `loss.py:282`

**Intuition.** No single loss handles all the failure modes of severely imbalanced binary KWS, so RPPL stacks five complementary objectives: BCE pins the boundary, a prototype-softmax shapes inter-class geometry, center loss tightens the wake cluster, hinge diversity stops confusable negatives from collapsing onto each other, and proto-ranked consistency forces augmented views to classify correctly. An EMA-smoothed wake prototype stabilises the small-batch noise inherent to 2–5 positives per batch.

**Theory.** See [`docs/research/rppl.md`](../research/rppl.md) for the full formulation, prior-work attribution, and the experiment spec (ablations + baselines + decision rules). In short: each component is a known idea (Prototypical Networks, center loss, spread-out regularisation, MeanTeacher-style consistency); the contribution is the recipe for binary fixed-keyword KWS.

**Engineering choices specific to this implementation:**

- **EMA wake prototype** (`proto_ema_alpha=0.05`): batch mean of 2–5 wake embeddings is too noisy as a center-loss target; the EMA stabilises within ~20 batches. Updated only in `model.train()` mode.
- **Hard-negative targeting** (`hard_div_threshold=0.1`): only spreads confusable negatives. If fewer than two negatives clear the threshold, the diversity term contributes 0 (no fallback to easy negatives).
- **Hinge diversity** (`div_margin=1.0`): pairs > `div_margin` apart in squared L2 produce zero gradient — bounded, unlike the earlier unbounded `-mean(distance)` form.
- **Proto-ranked consistency**: augmented view must classify correctly against the shared prototypes. Stricter than per-sample L2 invariance because representation collapse cannot satisfy it.
- **Warmup scheduling** (`warmup_epochs=5`): geometric terms ramp 0 → 1 so they don't train against random-epoch-0 prototypes.

**Components:**

| Term | Weight param | Description | Source |
|------|-------------|-------------|--------|
| BCE | `alpha` | Standard binary cross-entropy | — |
| Proto-softmax | `beta` | Classify against EMA wake prototype + NWW mean prototype | Snell et al. 2017 |
| Center loss | `delta` | Pull wake embeddings toward the wake prototype | Wen et al. 2016 |
| Hard-neg diversity | `gamma` | Spread confusable negatives apart (hinge form) | Boudiaf et al. 2020 |
| Proto-ranked consistency | `eta` | Augmented view must classify correctly against shared prototypes | BYOL / MeanTeacher / SupCon |

**Full config:**

```python
{"name": "rppl", "weight": 1.0,
 "tau": 0.1,              # prototype similarity temperature
 "K_neg_proto": 0,        # 0/1 = single mean NWW prototype; >1 = cosine k-means
 "alpha": 1.0,            # BCE
 "beta": 1.0,             # proto-softmax
 "gamma": 0.5,            # diversity
 "delta": 0.1,            # center
 "eta": 0.5,              # consistency
 "warmup_epochs": 5,
 "proto_ema_alpha": 0.05,
 "hard_div_threshold": 0.1,
 "div_margin": 1.0,             # hinge margin (squared L2) for diversity
 "consistency_mode": "proto"}   # "proto" (proto-ranked CE) or "l2" (MSE ablation)
```

**MLflow metrics logged automatically when using RPPL:**

| Metric | Meaning |
|--------|---------|
| `rppl_bce` | BCE sub-loss per epoch |
| `rppl_proto` | Prototype contrastive sub-loss |
| `rppl_div` | Hinge diversity sub-loss (0 when hard negatives are already > `div_margin` apart) |
| `rppl_center` | Center loss sub-loss |
| `rppl_cons` | Consistency sub-loss |
| `rppl_geo_scale` | Warmup ramp (0 → 1 over `warmup_epochs`) |
| `rppl_proto_ema_norm` | L2 norm of the EMA wake prototype (should stabilise within ~20 batches) |

`train_rppl.py` is the dedicated experiment script.

**When to use:** Hard-negative mining regime with a large NWW pool; infinite training mode; any scenario where you want per-epoch visibility into embedding structure.

**When NOT to use:** Very small datasets where there are consistently < 2 wake samples per batch (EMA has no signal to stabilise from). Use Focal or SupCon instead. The 11 hyperparameters are not all tuned — start from defaults and consult the experiment spec in [`docs/research/rppl.md`](../research/rppl.md) before tweaking.

---

## MCU / Compute Regulariser

### SizeAwareLoss — `loss.py:987`

**Intuition.** When searching architectures for a microcontroller target, you don't just want the best accuracy — you want the best *accuracy-per-parameter*. SizeAware adds a soft penalty proportional to `log(n_params)` so genetic search prefers smaller models without an explicit hard constraint.

**Theory.** `L = BCE(logits, labels) + λ · log(n_params)`. The log term grows slowly so a small model gets a meaningful bonus without overriding the classification signal. Conceptually a Bayesian Occam's-razor prior on model size.

**Config:** `{"name": "size_aware", "weight": 1.0, "lambda": 0.01}`

**When to use:** MCU-target genetic search where parameter count is a first-class objective. See [`docs/esp32.md`](../esp32.md) for the full ESP32 workflow.

**When NOT to use:** When model size is fixed — no benefit over BCE. When `λ` is set too aggressively (> 0.05) it dominates classification and the search collapses to the smallest possible model regardless of accuracy. Start at `0.001` and ramp up.

---

## Recommended Combinations

| Scenario | Loss Config | Notes |
|----------|-------------|-------|
| Simple baseline | BCE alone | Start here |
| Imbalanced data | Focal (weight=1.0) | Replace BCE entirely |
| Better generalisation | LabelSmoothing (1.0) + SupCon (0.3) | Smooth labels + embedding quality |
| Few-shot | BCE (0.5) + ArcFace (0.5) + Center (0.2) | Maximise separation with few examples |
| Production | Focal (0.5) + SupCon (0.3) + Center (0.2) | Robust to imbalance + good embeddings |
| Noisy labels | LabelSmoothing (1.0) + MultiSimilarity (0.3) | Tolerates label noise |
| Maximum accuracy | BCE (0.5) + ArcFace (0.3) + SupCon (0.2) | Heavy but effective |
| Large NWW pool + infinite training | RPPL (1.0) | Best when mining loop provides hard negatives every epoch |
| MCU / genetic architecture search | SizeAware (1.0, λ=0.001) | Biases search toward smaller models |
