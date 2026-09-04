# Classifier Heads

All heads inherit from `ClassifierHead` — `ww_trainer/model.py:27`. Input: `[B, T, F]` features. Output: `[B]` logits (apply sigmoid for probability). All heads provide `embed()` for metric learning losses.

---

## Overview

| Head | Class | Line | Architecture | Approx Params | Hardware Fit |
|------|-------|------|-------------|---------------|-------------|
| FFN | `FfnClassifierHead` | `model.py:246` | Mean pool + 2-layer MLP | ~10-35K | MCU |
| OCSVM | `OCSVMHead` | `model.py:271` | FFN backbone + One-Class SVM | ~20-50K + SVs | RPi/laptop |
| CNN | `CnnClassifierHead` | `model.py:451` | 2x Conv1d + pool + FC | ~70-200K | MCU/RPi |
| GRU | `GruClassifierHead` | `model.py:483` | GRU + mean pool + FC | ~50-500K | RPi |
| BCResNet | `BCResNetHead` | `model.py:678` | 2D broadcasted residual CNN | 6K-280K (tau) | MCU/RPi |
| TCResNet | `TCResNetHead` | `model.py:859` | 1D temporal residual CNN | ~30-120K | MCU/RPi |
| DSCNN | `DSCNNHead` | `model.py:921` | Depthwise-separable 2D CNN | 20-250K (S/M/L) | MCU |
| MatchboxNet | `MatchboxNetHead` | `model.py:1006` | 1D time-channel separable CNN | 25-90K | MCU/RPi |
| Res15 | `Res15Head` | `model.py:1092` | 1D dilated residual (dil 1-32) | ~50K | RPi |
| KWT | `KWTHead` | `model.py:1136` | Vision Transformer on patches | ~50-200K | RPi4/laptop |
| Conformer | `ConformerHead` | `model.py:1251` | Conv-augmented transformer | ~100-500K | RPi4/laptop |
| MixConv | `MixConvHead` | `model.py:1352` | Mixed depthwise convolutions | ~20-100K | MCU/RPi |
| CRNN | `CRNNHead` | `model.py:1426` | 2D CNN + GRU | ~50-200K | RPi |
| EfficientNet | `EfficientNetHead` | `model.py:1479` | EfficientNet-B0 on log-mel | ~4M (backbone) | Laptop/server |
| ConvAttention | `ConvAttentionHead` | `model.py:1541` | 1D conv stack + self-attention + mean-pool | ~5-30K | MCU/RPi |

Supporting module: `AttentionPooling` — `model.py:812`. Used internally by `ConformerHead`.

`ConvAttentionHead` is ported from [livekit/livekit-wakeword](https://github.com/livekit/livekit-wakeword) (Apache-2.0). Their docs report "60x lower AUT and 100x fewer FPs/h vs openWakeWord" with this head over a frozen embedding front-end. The port relaxes the original fixed `LayerNorm([D, T=16])` to `LayerNorm(D)` over the channel axis, so the head accepts any `T` and exports with a dynamic time axis like the rest of `ww-trainer`.

---

## Detailed Descriptions

### FfnClassifierHead — `model.py:246`

Mean-pool over time, then Linear(F, hidden) -> ReLU -> Dropout -> Linear(hidden, 1).

**Parameters:** `hidden_dim` (128), `dropout` (0.2), `input_size`.

**When to use:** Simplest baseline. Fastest inference. Use with powerful extractors (HuBERT, Wav2Vec2) where features are already rich.

**When NOT to use:** With low-dimensional extractors (MFCC) -- mean-pooling discards temporal structure that matters.

**Param count:** `input_size * hidden_dim + hidden_dim + hidden_dim + 1` (e.g., 768*128 + 128 + 128 + 1 = ~98K for HuBERT).

---

### CnnClassifierHead — `model.py:451`

Two Conv1d layers (kernel_size=3) operating on `[B, F, T]`, AdaptiveAvgPool1d(1), then FC layers.

**Parameters:** `conv_dim` (256), `linear_dim` (128), `kernel_size` (3), `stride` (1), `input_size`.

**When to use:** When local temporal patterns matter but you want to stay lightweight. Good with MFCC/filterbank features.

**When NOT to use:** When you need long-range temporal context (use GRU or Conformer instead).

---

### GruClassifierHead — `model.py:483`

GRU RNN with optional bidirectional mode, mean-pool, then FC layers. Auto-detects `[B, F, T]` vs `[B, T, F]` input shape — `GruClassifierHead._ensure_correct_shape` — `model.py:504`.

**Parameters:** `hidden_dim` (128), `linear_dim` (128), `dropout` (0.0), `bidirectional` (False), `gru_n_layers` (1), `input_size`.

**When to use:** Captures temporal dynamics. Good for multi-word wake phrases. Bidirectional mode improves accuracy at 2x cost.

**When NOT to use:** Streaming inference (bidirectional requires full utterance). MCUs (too many parameters with large hidden_dim).

---

### BCResNetHead — `model.py:678`

BC-ResNet (Kim et al., Interspeech 2021, Qualcomm). Broadcasted residual blocks with 2D pathway (frequency-temporal) and 1D pathway (temporally-pooled). Uses SubSpectralNorm.

**Parameters:** `tau` controls width. `input_size` (mel bins, typically 40).

| tau | base_c | Approx Params |
|-----|--------|---------------|
| 1 | 8 | ~6K |
| 1.5 | 12 | ~12K |
| 2 | 16 | ~20K |
| 3 | 24 | ~43K |
| 6 | 48 | ~160K |
| 8 | 64 | ~280K |

4 stages of 2/2/4/4 blocks — `model.py:735`. Stages at index 1 and 2 use stride — `model.py:726`.

**When to use:** State-of-the-art KWS accuracy at any param budget. Best with FilterBank or MFCC input. Scale tau to match hardware.

**When NOT to use:** With high-dimensional transformer features (designed for 2D spectrograms).

---

### TCResNetHead — `model.py:859`

TC-ResNet (Choi et al., Interspeech 2019). Purely 1D temporal convolutions with residual connections.

**Parameters:** `variant` (8 or 14), `channels` (64), `kernel_size` (9), `input_size`.

- `variant=8`: 3 blocks (TC-ResNet8)
- `variant=14`: 6 blocks (TC-ResNet14)

**When to use:** Simple, effective 1D temporal modeling. Good baseline for any spectrogram feature.

**When NOT to use:** When 2D spectral structure matters (use BCResNet or DSCNN).

---

### DSCNNHead — `model.py:921`

DS-CNN (Zhang et al., "Hello Edge", 2017). Depthwise-separable 2D CNN. The ARM/Google benchmark standard for microcontroller KWS.

**Parameters:** `size` ("S"/"M"/"L"), `input_size`.

| Size | Channels | Approx Params |
|------|----------|---------------|
| S | [64]*4 | ~20K |
| M | [64]*5 | ~80K |
| L | [128]*6 | ~250K |

**When to use:** MCU deployment. Well-studied, quantization-friendly, proven on ARM Cortex-M.

**When NOT to use:** When you need state-of-the-art accuracy (BCResNet or Conformer outperform).

---

### MatchboxNetHead — `model.py:1006`

MatchboxNet (Majumdar & Ginsburg, NVIDIA 2020). 1D time-channel separable convolutions with residual connections. Architecture: B blocks x R sub-blocks x C channels.

**Parameters:** `B` (3), `R` (2), `C` (64), `kernel_sizes` ((11, 13, 15)), `input_size`.

| Config | B | R | C | Approx Params |
|--------|---|---|---|---------------|
| 3x1x64 | 3 | 1 | 64 | ~25K |
| 3x2x64 | 3 | 2 | 64 | ~45K |
| 6x2x64 | 6 | 2 | 64 | ~90K |

**When to use:** Proven NVIDIA architecture for KWS. Good accuracy/size tradeoff. Large receptive field via big kernels.

**When NOT to use:** When 2D spectral features are important.

---

### Res15Head — `model.py:1092`

Res15 (Tang & Lin, Interspeech 2018). 1D residual blocks with exponentially increasing dilation: [1, 2, 4, 8, 16, 32]. Very large receptive field in 6 blocks.

**Parameters:** `channels` (45), `input_size`.

**When to use:** Long wake phrases where large temporal receptive field matters. Compact and effective.

**When NOT to use:** Very short wake words where large dilation is wasteful.

---

### KWTHead — `model.py:1136`

Keyword Transformer (Berg et al. 2021). Vision Transformer adapted for spectrograms: patches along time, positional embedding, transformer encoder, CLS token classification.

**Parameters:** `patch_len` (5), `d_model` (64), `n_heads` (4), `n_layers` (4), `dim_ff` (128), `dropout` (0.1), `input_size`.

**When to use:** When you have enough training data and want transformer-level accuracy without a transformer extractor. Good with MFCC/filterbank features.

**When NOT to use:** MCUs. Small datasets (transformers need more data). Streaming inference.

---

### ConformerHead — `model.py:1251`

Conformer (Gulati et al. 2020). Convolution-augmented transformer: FFN -> MHSA -> Conv -> FFN per block — `_ConformerBlock` — `model.py:1206`. Uses `AttentionPooling` — `model.py:812` — for time aggregation.

**Parameters:** `d_model` (64), `n_heads` (4), `n_layers` (4), `conv_kernel` (15), `dim_ff` (128), `dropout` (0.1), `input_size`.

**When to use:** Best accuracy among all heads when compute allows. Combines attention's global context with convolution's local patterns.

**When NOT to use:** MCUs, RPi Zero. Streaming inference (attention over full sequence).

---

### CRNNHead — `model.py:1426`

2D CNN frontend (2 conv layers with MaxPool) followed by GRU. The CNN extracts local spectro-temporal patterns; the GRU models sequence dynamics.

**Parameters:** `conv_channels` (32), `gru_hidden` (64), `gru_layers` (1), `dropout` (0.1), `input_size`.

**When to use:** Good middle ground between CNN-only and RNN-only. Works well with mel features. Production-proven in many KWS systems.

**When NOT to use:** MCUs (MaxPool + GRU combo is heavier than pure CNN). When you want streaming (GRU is fine but CNN frontend adds latency).

---

### OCSVMHead — `model.py:271`

Two-stage head: FFN backbone maps pooled frame features to a fixed-size embedding, then a One-Class SVM decision boundary classifies embeddings as inlier (wake-word) or outlier. Inspired by anomaly-detection framing of keyword spotting.

**Theory:** One-Class SVM (Scholkopf et al., 2001) finds a hyperplane in kernel space that separates the origin from positive-class embeddings with maximum margin, controlled by `nu` (upper bound on fraction of outliers). At inference, the signed distance from the hyperplane is the wake-word score.

**Architecture:** `[B,T,F]` → mean-pool → Linear(F→hidden) → ReLU → Dropout → Linear(hidden→embed_dim) → kernel OCSVM → `[B]` score — `OCSVMHead.forward` — `model.py:365`.

**Training stages:**
1. Stage 1 — backbone trains via BCE/focal loss alongside all other heads (no OCSVM involved).
2. Stage 2 — call `OCSVMHead.fit_ocsvm(dataloader)` — `model.py:392` — after training. Fits `sklearn.svm.OneClassSVM` on positive-class embeddings. Support vectors, dual coefficients, bias, and resolved kernel parameters are stored as torch buffers.

**ONNX export:** All four kernels are implemented in pure PyTorch (`OCSVMHead._kernel_vals` — `model.py:343`); no sklearn at inference time. Always call `export_to_onnx` **after** `fit_ocsvm` — exporting before fitting bakes the all-zero sentinel buffers.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 128 | FFN intermediate width |
| `embed_dim` | 64 | Embedding dimension fed to OCSVM |
| `dropout` | 0.1 | Backbone dropout |
| `nu` | 0.1 | OCSVM `nu`: upper bound on outlier fraction |
| `kernel` | `"rbf"` | `"rbf"`, `"linear"`, `"poly"`, or `"sigmoid"` |
| `gamma` | `"scale"` | Kernel coefficient; `"scale"` → `1 / (n_features × X.var())` computed from training embeddings, stored as buffer after fitting |
| `degree` | 3 | Degree for `"poly"` kernel only |
| `coef0` | 0.0 | Independent term for `"poly"` and `"sigmoid"` |

**Kernel guide:**
- `rbf` — default; good general-purpose choice; works well with normalized embeddings.
- `linear` — fastest; no kernel trick; good when embeddings are already high-dimensional.
- `poly` — captures interactions; use with small `degree` (2-3); slow for large SV counts.
- `sigmoid` — resembles a two-layer neural network; sensitive to `gamma` and `coef0`.

**Optional dependency:** `fit_ocsvm()` requires `scikit-learn`. Install with `pip install wakeforge[ocsvm]`.

**Tier preset:** `ocsvm_small` — MFCC-40 + OCSVMHead (hidden=128, embed=64) — `ww_trainer/tiers.py:174`.

```python
from ww_trainer.model import OCSVMHead

head = OCSVMHead(input_size=40, nu=0.05, kernel="rbf")
# ... Stage 1 training loop using BCE loss on head.forward() ...

# Stage 2: fit OCSVM on positives
head.fit_ocsvm(train_dataloader)  # model.py:392

# Export AFTER fitting
head.export_to_onnx("best_f1.onnx")
```

**When to use:** Few-shot / positive-only training; tight FAR control via `nu`; anomaly-detection framing of KWS; small positive datasets where binary cross-entropy overfits.

**When NOT to use:** When negatives are plentiful (binary heads train faster and typically score higher F1). Not for MCUs — the SV buffer size scales with training data.

---

### MixConvHead — `model.py:1352`

Mixed depthwise convolution head inspired by micro-wake-word's MixedNet. Each block splits channels into groups and applies different temporal kernel sizes per group (e.g., `[[3], [5,7], [9,13]]`), capturing multi-scale patterns without depth overhead.

**Architecture:** `[B,T,F]` → transpose → Conv1d stem → N × MixConvBlock → AdaptiveAvgPool1d(1) → Linear → `[B]` logits.

**Parameters:** `n_blocks` (3), `filters` (64), `kernel_groups` ([[3],[5,7],[9,13]]), `input_size`.

**When to use:** When multi-scale temporal patterns matter and you want an efficient alternative to BCResNet. MCU/RPi-friendly.

**When NOT to use:** When frequency-domain 2D structure is more important than temporal multi-scale.

---

### EfficientNetHead — `model.py:1479`

EfficientNet-B0 (Tan & Le, 2019) applied to log-mel spectrograms treated as single-channel 2D images (`[B,1,F,T]`). Trained from scratch (no pretrained weights). Requires `torchvision`.

**Architecture:** `[B,T,F]` → transpose → unsqueeze(1) → EfficientNet-B0 (1-ch input) → Dropout → Linear(1280, 1) → `[B]` logits — `EfficientNetHead.forward` — `model.py:1526`.

**Parameters:** `dropout` (0.2), `input_size` (mel bins, e.g. 40 or 80).

**Dependency:** `pip install torchvision`

**Tier preset:** `efficientnet_small` — FilterBank-40 + EfficientNetHead — `ww_trainer/tiers.py:107`.

**When to use:** When you have abundant data and a server/laptop, and want a 2D-CNN baseline without hand-tuning architecture.

**When NOT to use:** MCU or RPi (4M+ params). Streaming inference. When you need ONNX export (EfficientNet's SqueezeExcitation blocks export but verify with your ONNX runtime version).

---

### ConvAttentionHead — `model.py:1541`

Ported from [livekit/livekit-wakeword](https://github.com/livekit/livekit-wakeword) (Apache-2.0). Their docs report "60x lower AUT and 100x fewer FPs/hour vs openWakeWord" using this head over a frozen embedding front-end.

**Architecture:** Conv1d(F→D, k=3) → (Conv1d(D→D, k=3))×n_blocks → MultiheadAttention(D, n_heads) + residual + LayerNorm → mean-pool(T) → Linear(D, 1).

The original implementation fixes `LayerNorm([D, T=16])`; this port uses `LayerNorm(D)` over the channel axis so any `T` is accepted and the head exports with a dynamic time axis.

**Parameters:** `layer_dim` (96), `n_heads` (8), `n_blocks` (2), `input_size`.

**When to use:** When you need a very small, fast head with attention-level accuracy. Pairs well with frozen SSL embeddings.

**When NOT to use:** When T varies wildly across inference calls (attention is O(T²)).

---

## Hardware Fit Summary

| Hardware | Recommended Heads | Max Params |
|----------|-------------------|------------|
| MCU (Cortex-M) | FFN, DSCNN-S, BCResNet (tau=1-2), MixConv | <30K |
| RPi Zero | DSCNN-M, BCResNet (tau=3), MatchboxNet-3x1x64, TCResNet8, ConvAttention | <50K |
| RPi 3/4 | BCResNet (tau=6-8), GRU, CRNN, Res15, OCSVMHead, MixConv | <300K |
| Laptop | KWT, Conformer, EfficientNet, any | <500K |
| Server | Conformer, KWT with large configs, EfficientNet | Unlimited |
