# TinyHuBERT: Distilling HuBERT into a Streaming Wake-Word Feature Extractor

**Status:** Engineering experiment specification. The recipe is implemented in
`scripts/research/tinyhubert.py`. Whether its features improve wake-word detection over MFCC
is measured by the experiment in §7; until a row of that table is filled, every claim about
downstream benefit is a hypothesis.
**Requires:** a GPU for the teacher forward pass, `transformers`, local audio on disk.

---

## 1. Motivation

Self-supervised speech models such as HuBERT (Hsu et al., 2021) learn phonetic frame
representations from unlabelled audio. A wake-word head trained on HuBERT features separates
"hey mycroft" from "hey marcy" far more easily than one trained on MFCC, but HuBERT Base has
about 95M parameters and bidirectional attention: too large for an always-on device and unable
to stream.

TinyHuBERT keeps the representation and drops the architecture. A small causal network is
trained to predict HuBERT's hidden states frame by frame; the trained network is then used as a
frozen `OnnxFeatureExtractor` under an ordinary wake-word head.

The design decisions, each of which the implementation encodes:

1. **Match HuBERT's frame rate.** HuBERT emits one frame every 20 ms: its convolutional front
   end has a total stride of 320 samples and a 400-sample receptive field. The student's five
   causal convolutions have strides 5, 4, 4, 2, 2, a total of 320, so every student frame has a
   teacher frame to learn from and no target is an interpolation of two.
2. **Causal, with a fixed lag.** HuBERT frame t depends on the whole clip, past and future; a
   streaming student cannot see the future, so part of that target is unpredictable to it. The
   student instead predicts teacher frame t at its own frame t + 1 + lag: by then it has heard
   the teacher frame's full 400-sample window and `lag` further 20 ms frames of audio. The
   default lag of 4 gives 100 ms of latency, which a wake-word decision tolerates.
3. **Distil middle layers, not only the last.** The last layer of a pretrained (not fine-tuned)
   HuBERT is shaped by its masked-prediction objective; phonetic content, and keyword-spotting
   accuracy, peak in the middle layers. Following DistilHuBERT (Chang et al., 2022), a shared
   backbone feeds one linear head per distilled layer (default 4, 8 and 12).
4. **Standardise the targets.** A few HuBERT dimensions carry magnitudes far above the rest, and
   an unnormalised distance spends its gradient on them. Targets are standardised per layer and
   per dimension with statistics measured on the teacher before training.
5. **DistilHuBERT's loss.** Per layer: L1 distance plus `-log sigmoid(cosine similarity)`. The L1
   term fixes the values, the cosine term the direction.
6. **Contrastive negatives from other clips only (optional).** An InfoNCE term that treats every
   other frame of the batch as a negative asks the student to separate neighbouring frames of one
   clip, which the teacher itself barely separates. `cross_clip_nce` excludes same-clip frames
   from the negatives. It is off by default (`--nce-weight 0`) and is one of the ablations in §7.
7. **Noise on the student's side only.** Wake words are spoken over television, music and
   kitchens. With `--noise-dir` the student hears each clip mixed with noise at a random SNR
   (0–20 dB by default) while the teacher hears it clean, so the student learns to report the
   speech rather than the mixture.

## 2. Architecture

**Teacher:** `facebook/hubert-base-ls960`, frozen, with `output_hidden_states=True`. For
16 kHz input of N samples it emits `(N - 400) // 320 + 1` frames of 768 dimensions per layer.

**Student (`WakeHuBERTStudent`):**

```
waveform [B, N]
  5 residual causal blocks: Conv1d(k, s) → BN → GELU → causal Conv1d(3) → BN, + strided skip
     channels 64, 128, 256, 256, 256; kernels 10, 8, 8, 4, 4; strides 5, 4, 4, 2, 2
  unidirectional GRU (256 hidden, 1 layer)
  LayerNorm                                  → features [B, N // 320, 256]   (exported)
  one Linear(256 → 768) head per teacher layer                             (training only)
```

Every convolution is padded on the left only, so frame t is computed from samples before
320 (t + 1) and nothing later; `test/test_tinyhubert.py` checks this by perturbing the future.

**Size:** the exported backbone has 2.33M parameters at the defaults (`cnn_dim=64`,
`gru_hidden=256`); `tinyhubert.json` records the exact count. The training heads add
0.6M and are not exported.

## 3. Training

**Data:** any directory of `.wav`, `.flac` or `.ogg` files, searched recursively, cropped at
random to `--crop-seconds` (default 2 s). Validation uses fixed, seeded crops of a held-out
directory, so validation losses are comparable across steps. LibriSpeech `train-clean-100`
with `dev-clean` for validation is a sufficient start; the noise directory can be any
non-speech audio (AudioSet, MUSAN).

**Objective:** the mean over distilled layers of `L1 - log sigmoid(cos)` on standardised
targets, plus `--nce-weight` times the cross-clip InfoNCE on the last distilled layer.

**Optimiser:** AdamW (lr 1e-3, weight decay 0.01), linear warm-up over 2,000 steps then cosine
decay to zero over `--steps`; gradients clipped at 5.

**Outputs:** `metrics.jsonl` (training loss, gradient norm, learning rate and throughput every
100 steps; validation loss, L1 and cosine per layer every `--eval-every` steps), `last.pt` and
`best.pt` (resumable with `--resume`), `tinyhubert.onnx` from the best checkpoint, and
`tinyhubert.json` with the teacher, layers, lag, latency, frame rate, feature dimension and
parameter count.

## 4. Export

`tinyhubert.onnx` takes `waveform` `[batch, samples]` and returns `features`
`[batch, frames, 256]`, the input layout `OnnxFeatureExtractor` feeds. The export is checked
against the PyTorch model on a clip whose length differs from the export's dummy input and
refused if they disagree.

The exported graph processes a whole window. The GRU's state is not an input or output of the
graph, so a device re-runs the extractor over a sliding window rather than feeding it 20 ms at a
time; a stateful export is future work and changes no result of the experiment below.

## 5. Usage

```bash
python scripts/research/tinyhubert.py \
    --audio-dir data/LibriSpeech/train-clean-100 \
    --val-dir data/LibriSpeech/dev-clean \
    --noise-dir data/noise \
    --out-dir runs/tinyhubert \
    --steps 60000

# continue an interrupted run
python scripts/research/tinyhubert.py ... --resume runs/tinyhubert/last.pt
```

Train a wake-word model on the extractor:

```bash
ww-trainer train --wake-word hey_mycroft \
    --metadata train/metadata.csv --test-metadata test/metadata.csv \
    --featurizer runs/tinyhubert/tinyhubert.onnx --featurizer-type onnx --feature-dim 256 \
    --arch gru --save-best
```

## 6. What is and is not new

Every ingredient comes from prior work; the contribution is the combination and the experiment.

| Ingredient | Source |
|---|---|
| HuBERT teacher | Hsu et al. 2021 |
| Layer-wise distillation, L1 + log-sigmoid-cosine loss | DistilHuBERT, Chang et al. 2022 |
| InfoNCE | van den Oord et al. 2018 |
| Streaming student of a bidirectional teacher, with look-ahead budget | standard in streaming ASR distillation |
| Clean teacher, distorted student input | Cross-Distortion Mapping, Huang et al. 2022 |

DistilHuBERT's student is a 24M-parameter Transformer with the teacher's full context; this
student is about ten times smaller, causal, and pays a fixed 100 ms of latency instead.

## 7. Experiment

Identical wake-word training (same data, splits, head, epochs and seed), varying only the
feature extractor. The wake word is "hey mycroft"; metrics are F1 on the held-out split and
false activations per hour on held-out negative audio at a fixed recall.

| Run | Feature extractor | Extractor params | F1 | FA/h at recall 0.9 |
|---|---|---|---|---|
| F1 | MFCC (40) | 0 | — | — |
| F2 | HuBERT Base, layer 8, frozen (upper bound) | 95M | — | — |
| F2b | DistilHuBERT, frozen ([`TigreGotico/distillhubert-onnx`](https://huggingface.co/TigreGotico/distillhubert-onnx)) | 24M | — | — |
| F3 | TinyHuBERT, defaults | ~2.3M | — | — |
| F4 | TinyHuBERT, lag 0 | ~2.3M | — | — |
| F5 | TinyHuBERT, no noise | ~2.3M | — | — |
| F6 | TinyHuBERT, last layer only | ~2.3M | — | — |
| F7 | TinyHuBERT, `cnn_dim=32` | ~0.8M | — | — |

**Decision rules, fixed before the runs:**

- If F3 does not beat F1 on both F1 score and FA/h, the recipe does not earn its training cost.
- If F3 matches F2b, the student is a drop-in replacement for DistilHuBERT at a tenth of its size.
- If F3 reaches most of the F2 − F1 gap, the size saving (about forty times) justifies it.
- F4, F5 and F6 each test one design decision from §1; a decision whose ablation matches F3 is
  removed.
- If F7 matches F3, the smaller student is the default.

The distillation's own validation cosine per layer is recorded beside each row; it says how
well the student imitates the teacher, not whether that helps detection, and is never used in
place of the downstream columns.

## 8. References

- Hsu et al. (2021). *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction
  of Hidden Units.* [arXiv:2106.07447](https://arxiv.org/abs/2106.07447)
- Chang, Yang, Lee (2022). *DistilHuBERT: Speech Representation Learning by Layer-wise
  Distillation of Hidden-unit BERT.* [arXiv:2110.01900](https://arxiv.org/abs/2110.01900)
- Huang et al. (2022). *Improving Generalizability of Distilled Self-supervised Speech Processing
  Models under Distorted Settings.* [arXiv:2210.07978](https://arxiv.org/abs/2210.07978)
- van den Oord et al. (2018). *Representation Learning with Contrastive Predictive Coding.*
  [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)

See [references.md](references.md) for the full project bibliography.
