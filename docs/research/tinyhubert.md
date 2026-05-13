# TinyHuBERT: Cross-Architecture Distillation for Streaming Wake Word Detection

**Version:** 1.0
**Status:** Engineering experiment specification, not a results paper. The recipe below is well-defined and implemented (`scripts/research/tinyhubert.py`); the downstream impact on wake-word F1 has **not** been measured yet. Treat any quantitative claim as a hypothesis until it appears in the results table in §7.
**Requires:** CUDA GPU for the teacher forward pass; `datasets==3.6.0`

---

## 1. Motivation and intuitions

Large self-supervised speech models such as HuBERT (Hsu et al., 2021) learn rich phonetic
representations from unlabelled audio, but their size (~95M parameters, non-causal Transformer
attention) makes them impractical for on-device wake-word detection: too big, too slow, and
fundamentally incompatible with streaming because attention needs full sequence context.

TinyHuBERT is a *cross-architecture distillation recipe* that tries to keep the useful part —
HuBERT's contextual frame embeddings — while paying for none of the architectural cost. The
intuitions driving the design:

1. **Phonetic embeddings beat hand-crafted features for KWS.** MFCC throws away most of the
   phonetic structure that distinguishes "hey mycroft" from "hey marcy". A small model with
   HuBERT-flavoured embeddings should outperform the same model fed raw MFCC, even if the
   embedding model is itself tiny.
2. **The student's *architecture* can differ wildly from the teacher.** HuBERT is a
   Transformer; the student is a CNN + GRU. Distillation only needs the teacher's output
   embedding, not its internals — so we can swap a non-causal Transformer for a causal,
   stream-friendly RNN, as long as the embeddings line up frame-by-frame.
3. **Causality is non-negotiable for on-device wake detection.** A unidirectional GRU and a
   causal CNN stack give us a model that produces one new embedding per incoming audio chunk,
   with no look-ahead.
4. **MSE alone causes representation collapse.** If the student is only asked to match the
   teacher's vectors, it can satisfy the loss by producing a low-variance blob near the
   teacher's mean. A contrastive (InfoNCE) auxiliary term forces the student to keep
   *different* teacher frames *separable*, not just close on average.
5. **Match the teacher's frame rate, not its parameter count.** The CNN stack is sized so its
   total downsampling matches HuBERT's (×160 at 16 kHz). Frame-aligned outputs make the
   distillation loss meaningful at every timestep.

The resulting student can be used as a drop-in `OnnxFeatureExtractor` for wake-word training
inside ww-trainer.

This document describes the recipe. Whether it actually beats the alternatives (MFCC, SincNet,
DistilHuBERT off-the-shelf) for downstream wake-word F1 is an open question — see §7.

---

## 2. Architecture

### Teacher — frozen HuBERT Base

`facebook/hubert-base-ls960` loaded via HuggingFace `transformers`. All parameters frozen.
Outputs `[B, T', 768]` contextual embeddings at ~50 frames/second for 16 kHz audio.

### Student — `WakeHuBERTStudent`

```
Input: [B, 1, T]  (raw waveform, single channel)

CNN encoder (4 residual CnnBlocks):
  CnnBlock(1  → base_dim,    k=10, s=10)   # ×10 downsampling
  CnnBlock(base_dim → 2×base_dim, k=4,  s=4)    # ×4
  CnnBlock(2× → 4×base_dim, k=3,  s=2, p=1)   # ×2
  CnnBlock(4× → 4×base_dim, k=3,  s=2, p=1)   # ×2
  Total downsampling: ×160 → matches HuBERT's frame rate at 16 kHz

Unidirectional GRU  (gru_layers=1, gru_hidden=256 by default)
LayerNorm
Linear projection → out_dim  (default 768, matching teacher)

Output: [B, T', out_dim]
```

`CnnBlock` is a residual block: Conv1d → BN → GELU → Conv1d → BN, with a skip connection.
The GRU is **unidirectional** — no future context is used, making inference strictly causal.

If `out_dim != teacher.hidden_size` (768), a linear adapter layer is added automatically.

**Default parameter count** (base_dim=64, gru_hidden=256, gru_layers=1, out_dim=768):
approximately 3–4M parameters — roughly 25× smaller than HuBERT Base.

---

## 3. Training

### Dataset

Streams `MLCommons/ml_spoken_words` via HuggingFace `datasets` in streaming mode — no full
download required. Audio is cropped or zero-padded to a fixed length (default 1 second).
An epoch is defined as 5 000 training steps; validation runs for 500 steps.

Requires `datasets==3.6.0` due to a known streaming compatibility issue
([huggingface/datasets#7693](https://github.com/huggingface/datasets/issues/7693)).

### Loss

```
L = L_MSE + 0.1 × L_InfoNCE
```

**Intuition.** MSE provides coarse "go to this location in 768-D space" supervision — easy to
optimise, but on its own collapses to a low-variance blob. InfoNCE acts as a contrastive
counterweight: different teacher frames must remain distinguishable in the student's space.
MSE pulls; InfoNCE prevents collapse.

**MSE** (`F.mse_loss`): aligns the student's embedding vector with the teacher's at each frame.
Provides coarse positional supervision in the 768-d space.

**InfoNCE** (contrastive, temperature=0.07): flattens all `[B × T']` frame embeddings into a
matrix, sub-samples up to 2 048 frames, then computes cross-entropy over the cosine-similarity
matrix. The diagonal is the positive (student frame *i* ↔ teacher frame *i*); all off-diagonal
entries are negatives.

The 0.1 weight is an educated guess: MSE dominates early training while InfoNCE prevents
representation collapse. The exact ratio has *not* been ablated — see §7 for the experiment that
would justify it.

### Time alignment

HuBERT's CNN feature extractor and the student's CNN stack may produce slightly different
sequence lengths for the same input. Teacher features are linearly interpolated
(`F.interpolate(..., mode="linear")`) to match the student's `T'` before computing the loss.

### Optimiser and schedule

Adam (lr=1e-3), StepLR (step=5, γ=0.5). Gradient clipping at 5.0.
Best checkpoint saved to MLflow by validation loss; final model exported to ONNX.

---

## 4. Visualisation

Every `--plot-interval` epochs the script logs to MLflow:

- **PCA 2D / 3D**: joint PCA of student and teacher frame embeddings (blue = student, red = teacher).
  When distillation is working, the two point clouds overlap.
- **t-SNE 2D**: non-linear projection of up to 2 000 combined frames.

---

## 5. ONNX Export

The student is exported at the end of training with dynamic axes on the time dimension:

```python
torch.onnx.export(
    student, dummy,           # dummy: [1, 1, 16000]
    input_names=["waveform"], output_names=["embedding"],
    dynamic_axes={"waveform": {2: "n_samples"}, "embedding": {1: "time"}},
    opset_version=18,
)
```

The exported `tinyhubert.onnx` can then be loaded as an `OnnxFeatureExtractor` in ww-trainer
for downstream wake-word classification training.

---

## 6. Usage

```bash
# Requires GPU for the teacher pass
.venv/bin/python scripts/research/tinyhubert.py \
    --lang en \
    --epochs 50 \
    --batch-size 32 \
    --cnn-dim 64 \
    --gru-hidden 256 \
    --mlflow-uri http://localhost:5000

# Resume from checkpoint
.venv/bin/python scripts/research/tinyhubert.py \
    --lang en \
    --resume last_checkpoint.pt
```

After training, load the exported embedding model in ww-trainer:

```python
from ww_trainer.feats import OnnxFeatureExtractor
extractor = OnnxFeatureExtractor("tinyhubert.onnx")

from ww_trainer.trainer import WakeWordTrainer
trainer = WakeWordTrainer(
    arch="gru",
    featurizer="tinyhubert.onnx",
    featurizer_type="onnx",
    feature_dim=768,
)
```

---

## 6.5. Honest framing — what is and is not new

Every individual ingredient comes from prior work. The contribution of this document is a
*recipe and an experiment spec*, not a new technique.

| Ingredient | Source |
|---|---|
| HuBERT teacher | Hsu et al. 2021 |
| Self-supervised distillation of HuBERT into a smaller model | DistilHuBERT (Chang et al. 2022) |
| MSE + InfoNCE on frame embeddings | standard distillation loss family; InfoNCE from van den Oord et al. 2018 |
| Causal CNN feature extractor + RNN head | streaming-ASR student architecture, well-established |
| Cross-architecture (Transformer → CNN/RNN) distillation | standard in speech, e.g. streaming RNN-T students of bidirectional teachers |

**Engineering choices specific to this implementation** (none are paper-worthy on their own):

1. **Frame-rate matching via fixed-stride CNN.** Sizing the CNN stack so that total
   downsampling equals HuBERT's ×160 at 16 kHz, then using linear interpolation only for
   residual length mismatch. Makes the frame-by-frame distillation loss meaningful.
2. **Single-layer distillation target.** DistilHuBERT distils from *multiple* teacher layers
   (layer-wise prediction heads); this recipe uses only the final hidden state. Simpler,
   cheaper, may give up some structure — to be measured.
3. **Tiny student.** ~3–4 M parameters vs DistilHuBERT's ~24 M. Whether the resulting
   embeddings remain useful for downstream KWS is the open question.

**Not yet demonstrated.** Whether this student's embeddings improve wake-word F1 over MFCC,
SincNet, or off-the-shelf DistilHuBERT is unknown. Until those numbers exist, this is a
plausible engineering recipe rather than a validated technique.

---

## 7. Known limitations and open questions

**Limitations (true regardless of results):**

- **GPU required during distillation.** The HuBERT teacher forward pass is too slow on CPU
  for practical training. Student-only inference is CPU-compatible after ONNX export.
- **`datasets==3.6.0` pin.** Streaming from `MLCommons/ml_spoken_words` breaks on newer
  versions of `datasets`. See [issue #7693](https://github.com/huggingface/datasets/issues/7693).
- **Linear interpolation for frame alignment.** When student and teacher disagree by a few
  frames, `F.interpolate(mode="linear")` smears the teacher targets. Acceptable for small
  offsets; may obscure fine phonetic detail.
- **Not portable to this project's CPU-only dev machine.** Train on a GPU box, copy
  `tinyhubert.onnx` back.
- **Distils a single teacher layer.** No layer-wise supervision (cf. DistilHuBERT). May
  underuse the teacher's structure.

**Experiment spec — what would justify this recipe.** Identical wake-word training pipeline,
varying only the feature extractor. Same seed, same splits, same epochs.

| Run | Feature extractor | Params (extractor) | F1 | EER | FAR @ FRR=1% | Notes |
|---|---|---|---|---|---|---|
| F1 | MFCC (40 coeffs) | 0 | — | — | — | hand-crafted baseline |
| F2 | SincNet | ~few K | — | — | — | learnable filterbank baseline |
| F3 | DistilHuBERT (off-the-shelf, frozen) | ~24 M | — | — | — | strong upper baseline |
| F4 | TinyHuBERT (this work) | ~3–4 M | — | — | — | the proposal |
| F5 | TinyHuBERT, MSE only | ~3–4 M | — | — | — | InfoNCE ablation |
| F6 | TinyHuBERT, InfoNCE only | ~3–4 M | — | — | — | MSE ablation |
| F7 | TinyHuBERT, base_dim=32 (half size) | ~1 M | — | — | — | size sensitivity |

**Decision rules** (set in advance to avoid post-hoc rationalisation):

- If F4 does not beat F1 (MFCC) on F1 *and* FAR@FRR=1%, the recipe is not justified — the
  distillation cost is wasted.
- If F4 matches F3 (DistilHuBERT) within 1 % absolute F1, the size win (3–4 M vs 24 M) is
  worth shipping.
- If F5 or F6 match F4, drop the redundant loss term.
- If F7 matches F4, ship the smaller model.

**Open questions to resolve with results (revise this document once answered):**

- Does a 25× smaller student retain enough of HuBERT's phonetic signal for binary KWS, or does
  capacity become the bottleneck?
- Is the 0.1 InfoNCE weight roughly right, or does the optimum sit at 0.01 or 1.0?
- Does layer-wise distillation (DistilHuBERT-style) actually help downstream F1, or is the
  final layer sufficient?
- Does linear interpolation of teacher frames materially harm distillation quality vs. exact
  alignment?

This document is intended to be updated, not replaced, once results land — sections 1–6 are
stable; §7 will gain numbers and shrink as questions are answered.

---

## 8. References

- Hsu et al. (2021) — *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction
  of Hidden Units.* [arXiv:2106.07447](https://arxiv.org/abs/2106.07447)
- van den Oord et al. (2018) — *Representation Learning with Contrastive Predictive Coding.*
  [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
- Chang et al. (2022) — *DistilHuBERT: Speech Representation Learning by Layer-wise Distillation
  of Hidden-unit BERT.* [arXiv:2110.01900](https://arxiv.org/abs/2110.01900)

See [references.md](references.md) for the full project bibliography.
