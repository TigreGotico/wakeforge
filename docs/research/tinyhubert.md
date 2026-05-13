# TinyHuBERT: Cross-Architecture Distillation for Streaming Wake Word Detection

**Version:** 1.0
**Status:** Research prototype (`scripts/research/tinyhubert.py`)
**Requires:** CUDA GPU for the teacher forward pass; `datasets==3.6.0`

---

## 1. Overview

Large self-supervised speech models such as HuBERT (Hsu et al., 2021) learn rich phonetic
representations from unlabelled audio, but their size (~95M parameters, non-causal attention)
makes direct deployment on edge hardware impractical.

TinyHuBERT distils HuBERT's representations into a compact **causal CNN + GRU** student that:

- processes audio as a stream (one chunk at a time, no look-ahead),
- exports to a single ONNX file,
- produces 768-dimensional embeddings that track the teacher's representation space.

The resulting student can be used as a drop-in `OnnxFeatureExtractor` for wake-word training
inside ww-trainer.

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

**MSE** (`F.mse_loss`): aligns the student's embedding vector with the teacher's at each frame.
Provides coarse positional supervision in the 768-d space.

**InfoNCE** (contrastive, temperature=0.07): flattens all `[B × T']` frame embeddings into a
matrix, sub-samples up to 2 048 frames, then computes cross-entropy over the cosine-similarity
matrix. The diagonal is the positive (student frame *i* ↔ teacher frame *i*); all off-diagonal
entries are negatives. This forces the student's frames to be *discriminatively* placed — similar
frames must not collapse to the same point.

The 0.1 weight means MSE dominates early training while InfoNCE prevents representation collapse.

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

## 7. Limitations

- **GPU required during distillation**: the HuBERT teacher forward pass is too slow on CPU for
  practical training. The student-only inference is CPU-compatible after export.
- **datasets==3.6.0 pin**: streaming from `MLCommons/ml_spoken_words` breaks on newer versions
  of `datasets`. See [issue #7693](https://github.com/huggingface/datasets/issues/7693).
- **No downstream evaluation**: this script only minimises distillation loss. Whether the student
  embeddings improve wake-word F1 over MFCC or SincNet has not been benchmarked in this repo.
- **Not suitable for this project's CPU-only dev machine**: the teacher requires significant RAM
  and benefits from CUDA. Train on a GPU machine and copy `tinyhubert.onnx` locally.

---

## 8. References

- Hsu et al. (2021) — *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction
  of Hidden Units.* [arXiv:2106.07447](https://arxiv.org/abs/2106.07447)
- van den Oord et al. (2018) — *Representation Learning with Contrastive Predictive Coding.*
  [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
- Chang et al. (2022) — *DistilHuBERT: Speech Representation Learning by Layer-wise Distillation
  of Hidden-unit BERT.* [arXiv:2110.01900](https://arxiv.org/abs/2110.01900)

See [references.md](references.md) for the full project bibliography.
