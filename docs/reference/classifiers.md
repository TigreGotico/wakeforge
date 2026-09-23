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
| BCResNet | `BCResNetHead` | `model.py:776` | 2D broadcasted residual CNN | 6K-280K (tau) | MCU/RPi |
| TCResNet | `TCResNetHead` | `model.py:957` | 1D temporal residual CNN | ~30-120K | MCU/RPi |
| DSCNN | `DSCNNHead` | `model.py:1019` | Depthwise-separable 2D CNN | 20-250K (S/M/L) | MCU |
| MatchboxNet | `MatchboxNetHead` | `model.py:1104` | 1D time-channel separable CNN | 25-90K | MCU/RPi |
| Res15 | `Res15Head` | `model.py:1190` | 1D dilated residual (dil 1-32) | ~50K | RPi |
| KWT | `KWTHead` | `model.py:1234` | Vision Transformer on patches | ~50-200K | RPi4/laptop |
| Conformer | `ConformerHead` | `model.py:1349` | Conv-augmented transformer | ~100-500K | RPi4/laptop |
| MixConv | `MixConvHead` | `model.py:1448` | Mixed depthwise convolutions | ~20-100K | MCU/RPi |
| CRNN | `CRNNHead` | `model.py:1522` | 2D CNN + GRU | ~50-200K | RPi |
| EfficientNet | `EfficientNetHead` | `model.py:1575` | EfficientNet-B0 on log-mel | ~4M (backbone) | Laptop/server |
| ConvAttention | `ConvAttentionHead` | `model.py:1637` | 1D conv stack + self-attention + mean-pool | ~5-30K | MCU/RPi |

Supporting module: `AttentionPooling` — `model.py:910`. Used internally by `ConformerHead`.

`ConvAttentionHead` is ported from [livekit/livekit-wakeword](https://github.com/livekit/livekit-wakeword) (Apache-2.0). Their docs report "60x lower AUT and 100x fewer FPs/h vs openWakeWord" with this head over a frozen embedding front-end — LiveKit's own reported figure, not reproduced or measured by `wakeforge`; see the head's own section below. The port relaxes the original fixed `LayerNorm([D, T=16])` to `LayerNorm(D)` over the channel axis, so the head accepts any `T` and exports with a dynamic time axis like the rest of `wakeforge`.

---

## Detailed Descriptions

### FfnClassifierHead — `model.py:246`

Mean-pool over time, then Linear(F, hidden) -> ReLU -> Dropout -> Linear(hidden, 1).

**Source:** a generic MLP baseline, not a specific published architecture — no paper to cite.

**Parameters:** `hidden_dim` (128), `dropout` (0.2), `input_size`.

**When to use:** Simplest baseline. Fastest inference. Use with pretrained extractors (HuBERT, Wav2Vec2) whose features already carry the signal.

**When NOT to use:** With low-dimensional extractors (MFCC) -- mean-pooling discards temporal structure that matters.

**Data appetite:** not measured by this project. Because it discards temporal structure, it is the head most likely to need real (not only synthetic) positives to separate the wake phrase from similar-sounding speech — treat it as a fast baseline to compare other heads against, not a first choice for production.

**Param count:** `input_size * hidden_dim + hidden_dim + hidden_dim + 1` (e.g., 768*128 + 128 + 128 + 1 = ~98K for HuBERT).

---

### CnnClassifierHead — `model.py:451`

Two Conv1d layers (kernel_size=3) operating on `[B, F, T]`, AdaptiveAvgPool1d(1), then FC layers.

**Source:** a generic small CNN baseline, not a specific published architecture — no paper to cite.

**Parameters:** `conv_dim` (256), `linear_dim` (128), `kernel_size` (3), `stride` (1), `input_size`.

**When to use:** When local temporal patterns matter but you want to stay lightweight. Good with MFCC/filterbank features.

**When NOT to use:** When you need long-range temporal context (use GRU or Conformer instead).

**Data appetite:** not measured by this project.

---

### GruClassifierHead — `model.py:483`

GRU RNN with optional bidirectional mode, mean-pool, then FC layers. Auto-detects `[B, F, T]` vs `[B, T, F]` input shape — `GruClassifierHead._ensure_correct_shape` — `model.py:504`.

**Source:** a generic RNN baseline (GRU is Cho et al., 2014, ArXiv <https://arxiv.org/abs/1406.1078> — a general sequence-modeling architecture, not proposed for keyword spotting specifically). No dedicated KWS paper to cite.

**Parameters:** `hidden_dim` (128), `linear_dim` (128), `dropout` (0.0), `bidirectional` (False), `gru_n_layers` (1), `input_size`.

**When to use:** Captures temporal dynamics. Good for multi-word wake phrases. Bidirectional mode improves accuracy at 2x cost.

**When NOT to use:** Streaming inference (bidirectional requires full utterance). MCUs (too many parameters with large hidden_dim).

**Data appetite:** not measured by this project.

---

### BCResNetHead — `model.py:776`

BC-ResNet (Kim et al., *Broadcasted Residual Learning for Efficient Keyword Spotting*, Interspeech 2021, Qualcomm). ArXiv <https://arxiv.org/abs/2106.04140>. Broadcasted residual blocks with 2D pathway (frequency-temporal) and 1D pathway (temporally-pooled). Uses SubSpectralNorm.

**Status:** this is `wakeforge`'s own implementation of the published architecture. The parameter counts below are measured from this implementation; the accuracy figures in the paper are **not** reproduced or verified here.

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

**When to use:** Strong KWS accuracy at any param budget, per the paper's own reported results (not verified against those exact numbers by this project). Best with FilterBank or MFCC input. Scale tau to match hardware.

**When NOT to use:** With high-dimensional transformer features (designed for 2D spectrograms).

**Data appetite:** not measured by this project. The original paper trains and evaluates on Google Speech Commands, a public multi-word command dataset — the design assumes that scale of labeled data, not a single-phrase few-shot setting.

---

### TCResNetHead — `model.py:957`

TC-ResNet (Choi et al., *Temporal Convolution for Real-time Keyword Spotting on Mobile Devices*, Interspeech 2019). ArXiv <https://arxiv.org/abs/1904.03814>. Purely 1D temporal convolutions with residual connections.

**Status:** this is `wakeforge`'s own implementation of the published architecture; accuracy figures in the paper are not reproduced or verified here.

**Parameters:** `variant` (8 or 14), `channels` (64), `kernel_size` (9), `input_size`.

- `variant=8`: 3 blocks (TC-ResNet8)
- `variant=14`: 6 blocks (TC-ResNet14)

**When to use:** Simple, effective 1D temporal modeling. Good baseline for any spectrogram feature.

**When NOT to use:** When 2D spectral structure matters (use BCResNet or DSCNN).

**Data appetite:** not measured by this project. Like BC-ResNet, the paper's own evaluation is on Google Speech Commands, a public multi-word command dataset.

---

### DSCNNHead — `model.py:1019`

DS-CNN (Zhang et al., *Hello Edge: Keyword Spotting on Microcontrollers*, 2017). ArXiv <https://arxiv.org/abs/1711.07128>. Depthwise-separable 2D CNN. An ARM/Google microcontroller-KWS benchmark reference.

**Status:** this is `wakeforge`'s own implementation of the published architecture; accuracy figures in the paper are not reproduced or verified here.

**Parameters:** `size` ("S"/"M"/"L"), `input_size`.

| Size | Channels | Approx Params |
|------|----------|---------------|
| S | [64]*4 | ~20K |
| M | [64]*5 | ~80K |
| L | [128]*6 | ~250K |

**When to use:** MCU deployment. Well-studied, quantization-friendly, documented on ARM Cortex-M in the source paper.

**When NOT to use:** When you need the highest achievable accuracy for the param budget (BCResNet or Conformer report higher accuracy in their own papers, not verified against DS-CNN by this project).

**Data appetite:** not measured by this project. The source paper evaluates on Google Speech Commands.

---

### MatchboxNetHead — `model.py:1104`

MatchboxNet (Majumdar & Ginsburg, *MatchboxNet: 1D Time-Channel Separable Convolutional Neural Network Architecture for Speech Commands Recognition*, Interspeech 2020, NVIDIA). ArXiv <https://arxiv.org/abs/2004.08531>; full entry in [`../research/references.md`](../research/references.md). 1D time-channel separable convolutions with residual connections. Architecture: B blocks x R sub-blocks x C channels.

**Status:** this is `wakeforge`'s own implementation of the published architecture; accuracy figures in the paper are not reproduced or verified here.

**Parameters:** `B` (3), `R` (2), `C` (64), `kernel_sizes` ((11, 13, 15)), `input_size`.

| Config | B | R | C | Approx Params |
|--------|---|---|---|---------------|
| 3x1x64 | 3 | 1 | 64 | ~25K |
| 3x2x64 | 3 | 2 | 64 | ~45K |
| 6x2x64 | 6 | 2 | 64 | ~90K |

**When to use:** A published architecture with a stated accuracy/size tradeoff (not verified against this project's own runs). Large receptive field via big kernels.

**When NOT to use:** When 2D spectral features are important.

**Data appetite:** not measured by this project. The source paper evaluates on Google Speech Commands.

---

### Res15Head — `model.py:1190`

Res15 (Tang & Lin, *Deep Residual Learning for Small-Footprint Keyword Spotting*, ICASSP 2018). ArXiv <https://arxiv.org/abs/1710.10361>. 1D residual blocks with exponentially increasing dilation: [1, 2, 4, 8, 16, 32]. Large receptive field in 6 blocks.

**Status:** this is `wakeforge`'s own implementation of the published architecture; accuracy figures in the paper are not reproduced or verified here.

**Parameters:** `channels` (45), `input_size`.

**When to use:** Long wake phrases where a large temporal receptive field matters. Compact.

**When NOT to use:** Very short wake words where large dilation is wasteful.

**Data appetite:** not measured by this project. The source paper evaluates on Google Speech Commands.

---

### KWTHead — `model.py:1234`

Keyword Transformer (Berg et al., *Keyword Transformer: A Self-Attention Model for Keyword Spotting*, Interspeech 2021). ArXiv <https://arxiv.org/abs/2104.00769>; full entry in [`../research/references.md`](../research/references.md). Vision Transformer adapted for spectrograms: patches along time, positional embedding, transformer encoder, CLS token classification.

**Status:** this is `wakeforge`'s own implementation of the published architecture; accuracy figures in the paper are not reproduced or verified here.

**Parameters:** `patch_len` (5), `d_model` (64), `n_heads` (4), `n_layers` (4), `dim_ff` (128), `dropout` (0.1), `input_size`.

**When to use:** When you have enough training data and want transformer-level accuracy without a transformer extractor. Good with MFCC/filterbank features.

**When NOT to use:** MCUs. Small datasets — transformers are the least sample-efficient architecture in this list and need more data than the CNN/RNN heads above.

**Data appetite:** not measured by this project. The source paper evaluates on Google Speech Commands; transformer architectures in general need more labeled examples than a comparable CNN to reach the same accuracy, which is exactly why "when NOT to use" above calls out small datasets.

---

### ConformerHead — `model.py:1349`

Conformer (Gulati et al., *Conformer: Convolution-augmented Transformer for Speech Recognition*, Interspeech 2020). ArXiv <https://arxiv.org/abs/2005.08100>. Convolution-augmented transformer: FFN -> MHSA -> Conv -> FFN per block — `_ConformerBlock` — `model.py:1304`. Uses `AttentionPooling` — `model.py:910` — for time aggregation.

**Status:** the source paper targets full-sentence speech recognition, not keyword spotting; this head is `wakeforge`'s own adaptation of the block design to a binary wake-word classifier, not a reproduction of a KWS result from the paper.

**Parameters:** `d_model` (64), `n_heads` (4), `n_layers` (4), `conv_kernel` (15), `dim_ff` (128), `dropout` (0.1), `input_size`.

**When to use:** The largest-capacity head in this list when compute allows. Combines attention's global context with convolution's local patterns.

**When NOT to use:** MCUs, RPi Zero. Streaming inference (attention over full sequence). Small datasets — like KWT, this is a transformer-family head and needs more labeled data than a CNN/RNN head to earn its extra capacity.

**Data appetite:** not measured by this project, and the source paper's own results are on large-vocabulary ASR corpora, not a wake-word task — there is no directly applicable data-scale figure to cite here.

---

### CRNNHead — `model.py:1522`

2D CNN frontend (2 conv layers with MaxPool) followed by GRU. The CNN extracts local spectro-temporal patterns; the GRU models sequence dynamics.

**Source:** CNN+RNN hybrids are a widely used pattern in keyword spotting rather than one specific paper this implementation follows — no single citation to give. `docs/research/references.md` has entries that use CRNN-shaped models in their own experiments if you want prior art.

**Parameters:** `conv_channels` (32), `gru_hidden` (64), `gru_layers` (1), `dropout` (0.1), `input_size`.

**When to use:** Good middle ground between CNN-only and RNN-only. Works well with mel features.

**When NOT to use:** MCUs (MaxPool + GRU combo is heavier than pure CNN). When you want streaming (GRU is fine but CNN frontend adds latency).

**Data appetite:** not measured by this project.

---

### OCSVMHead — `model.py:271`

Two-stage head: FFN backbone maps pooled frame features to a fixed-size embedding, then a One-Class SVM decision boundary classifies embeddings as inlier (wake-word) or outlier. Inspired by anomaly-detection framing of keyword spotting.

**Theory:** One-Class SVM (Schölkopf et al., *Estimating the Support of a High-Dimensional Distribution*, Neural Computation 2001. <https://doi.org/10.1162/089976601750264965>) finds a hyperplane in kernel space that separates the origin from positive-class embeddings with maximum margin, controlled by `nu` (upper bound on fraction of outliers). At inference, the signed distance from the hyperplane is the wake-word score.

**Status:** the One-Class SVM decision rule is the cited paper's; the FFN backbone, the two-stage training procedure, and their application to wake-word embeddings are `wakeforge`'s own design, not from that paper or any other cited source.

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

**When NOT to use:** When negatives are plentiful (binary heads train faster; the relative F1 tradeoff between the two is not measured by this project). Not for MCUs — the SV buffer size scales with training data.

**Data appetite:** not measured by this project. The design intent is few-shot/positive-only, which is a different data-scale assumption than every other head in this file — treat the "few-shot" framing as this project's design goal, not a validated result.

---

### MixConvHead — `model.py:1448`

Mixed depthwise convolution head, using multiple kernel sizes per block after the mixed-kernel idea in Tan & Le, *MixConv: Mixed Depthwise Convolutional Kernels*, BMVC 2019 (ArXiv <https://arxiv.org/abs/1907.09595>), and following the block layout of the [micro-wake-word](https://github.com/kahrendt/microWakeWord) project's MixedNet. Each block splits channels into groups and applies different temporal kernel sizes per group (e.g., `[[3], [5,7], [9,13]]`), capturing multi-scale patterns without depth overhead.

**Status:** `wakeforge`'s own implementation, adapted from micro-wake-word's design rather than a reproduction of either cited source's own reported result.

**Architecture:** `[B,T,F]` → transpose → Conv1d stem → N × MixConvBlock → AdaptiveAvgPool1d(1) → Linear → `[B]` logits.

**Parameters:** `n_blocks` (3), `filters` (64), `kernel_groups` ([[3],[5,7],[9,13]]), `input_size`.

**When to use:** When multi-scale temporal patterns matter and you want an efficient alternative to BCResNet. MCU/RPi-friendly.

**When NOT to use:** When frequency-domain 2D structure is more important than temporal multi-scale.

**Data appetite:** not measured by this project.

---

### EfficientNetHead — `model.py:1575`

EfficientNet-B0 (Tan & Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019. ArXiv <https://arxiv.org/abs/1905.11946>) applied to log-mel spectrograms treated as single-channel 2D images (`[B,1,F,T]`). Trained from scratch (no pretrained weights). Requires `torchvision`.

**Status:** the source paper targets ImageNet image classification, not audio or keyword spotting; treating a log-mel spectrogram as a 1-channel image and training from scratch is `wakeforge`'s own adaptation, not a reproduction of a result from the paper.

**Architecture:** `[B,T,F]` → transpose → unsqueeze(1) → EfficientNet-B0 (1-ch input) → Dropout → Linear(1280, 1) → `[B]` logits — `EfficientNetHead.forward` — `model.py:1619`.

**Parameters:** `dropout` (0.2), `input_size` (mel bins, e.g. 40 or 80).

**Dependency:** `pip install torchvision`

**Tier preset:** `efficientnet_small` — FilterBank-40 + EfficientNetHead — `ww_trainer/tiers.py:107`.

**When to use:** When you have a server/laptop and want a 2D-CNN baseline without hand-tuning architecture. At ~4M backbone parameters, this is the heaviest head in this list, so treat "abundant data" as a real prerequisite, not a nice-to-have.

**When NOT to use:** MCU or RPi (4M+ params). Streaming inference. When you need ONNX export (EfficientNet's SqueezeExcitation blocks export but verify with your ONNX runtime version).

**Data appetite:** not measured by this project. The source paper's own scaling results are on ImageNet-scale image data, not wake-word audio — there is no directly applicable data-scale figure to cite. Being the largest head here (~4M params vs. tens of thousands for the MCU-tier heads), it is the one most likely to overfit on a small synthetic-only dataset.

---

### ConvAttentionHead — `model.py:1637`

Ported from [livekit/livekit-wakeword](https://github.com/livekit/livekit-wakeword) (Apache-2.0). Their docs report "60x lower AUT and 100x fewer FPs/hour vs openWakeWord" using this head over a frozen embedding front-end.

**Status:** that "60x / 100x" figure is LiveKit's own reported result on their own setup and evaluation data — `wakeforge` has not reproduced or measured it. This port also changes the normalization layer (below), so the figure may not transfer exactly even if you trusted it as-is.

**Architecture:** Conv1d(F→D, k=3) → (Conv1d(D→D, k=3))×n_blocks → MultiheadAttention(D, n_heads) + residual + LayerNorm → mean-pool(T) → Linear(D, 1).

The original implementation fixes `LayerNorm([D, T=16])`; this port uses `LayerNorm(D)` over the channel axis so any `T` is accepted and the head exports with a dynamic time axis.

**Parameters:** `layer_dim` (96), `n_heads` (8), `n_blocks` (2), `input_size`.

**When to use:** When you need a very small, fast head with attention-level accuracy. Pairs well with frozen SSL embeddings.

**When NOT to use:** When T varies wildly across inference calls (attention is O(T²)).

**Data appetite:** not measured by this project. LiveKit's own reported figure above is not a data-budget number and should not be read as one.

---

## Hardware Fit Summary

| Hardware | Recommended Heads | Max Params |
|----------|-------------------|------------|
| MCU (Cortex-M) | FFN, DSCNN-S, BCResNet (tau=1-2), MixConv | <30K |
| RPi Zero | DSCNN-M, BCResNet (tau=3), MatchboxNet-3x1x64, TCResNet8, ConvAttention | <50K |
| RPi 3/4 | BCResNet (tau=6-8), GRU, CRNN, Res15, OCSVMHead, MixConv | <300K |
| Laptop | KWT, Conformer, EfficientNet, any | <500K |
| Server | Conformer, KWT with large configs, EfficientNet | Unlimited |
