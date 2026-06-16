# Zero to Hero — ww-trainer Learning Path

A staged curriculum that takes you from "I have never trained a wake-word
detector" to "I am shipping a custom model to embedded hardware and running
research-grade ablations". Each stage has one concrete deliverable.

If you only have an hour, do **Stage 1** and stop. The rest is depth, not
prerequisites.

---

## Stage 0 — Concepts (15 minutes, reading only)

**Goal:** understand what a wake-word detector is and what we are optimising.

Read in order:
1. The "What is a wake word?" section in [`../README.md`](../README.md).
2. The "Newcomer questions" section in [`faq.md`](faq.md) — metric glossary
   (F1, EER, FAR, FRR, FA/hour), the two-ONNX-file contract, deployment scope.
3. The pipeline diagram in [`architecture.md`](internals/architecture.md).

You should now be able to explain: why a wake word must run on-device, why
recall and FA/hour are in tension, and why the featurizer and head are
exported separately.

---

## Stage 1 — Your first model (10 minutes, one command)

**Goal:** produce a working `best_f1.onnx` + `best_f1_featurizer.onnx` pair
for a phrase of your choice.

```bash
ww_trainer-quickstart \
  --wake-word "hey jarvis" \
  --output-dir ./hey_jarvis \
  --tier micro \
  --epochs 10 \
  --no-augmentation-data
```

Then verify it loads and scores audio:

```bash
.venv/bin/python examples/08_onnx_inference.py
```

Where to look in the docs: [`quickstart.md`](getting_started/quickstart.md) explains every CLI
flag; [`inference.md`](guides/inference.md) covers the `OnnxWakeWordInferencer` API.

---

## Stage 2 — Understand the components (1–2 hours)

**Goal:** be able to swap featurizers and heads on purpose.

Run these examples in order and read the matching doc for each:

| Example | Pairs with | What you learn |
|---|---|---|
| `examples/01_mfcc_ffn_micro.py` | [`extractors.md`](reference/extractors.md), [`classifiers.md`](reference/classifiers.md) | MFCC, FFN, model sizing |
| `examples/02_mfcc_gru_small.py` | [`classifiers.md`](reference/classifiers.md) | Temporal modelling with GRU |
| `examples/04_filterbank_cnn.py` | [`extractors.md`](reference/extractors.md) | Log-mel filterbanks, 1D CNN |
| `examples/05_sincnet_gru.py` | [`extractors.md`](reference/extractors.md) | Learnable filterbank (SincNet) |
| `examples/11_hardware_tiers.py` | [`hardware_guide.md`](guides/embedded.md), [`tiers.py`](../ww_trainer/tiers.py) | Tier presets and when to use each |
| `examples/09_streaming_inference.py` | [`streaming.md`](guides/inference.md) | `SlidingFeatureCacheTensor` — real-time chunks |

At the end you should be able to look at any `WakeWordTrainer(...)`
construction and predict the parameter count and ONNX size before running it.

---

## Stage 3 — Real datasets and real metrics (2–4 hours)

**Goal:** train on real audio, measure the model honestly, and react to the
numbers.

1. **Data contract** — read [`data_contract.md`](guides/datasets.md). Build a
   real metadata CSV from your own recordings.
2. **Augmentation** — point the trainer at background noise, music, and RIR
   folders. See the "Augmentation" section in [`training.md`](guides/training.md).
3. **Evaluation** — run `scripts/eval/eval_hey_mycroft.py` (or your own
   variant). Look at ROC, PR, DET curves and the FP/hour estimate.
4. **Threshold tuning** — pick a threshold from the PR curve that matches your
   product's FA/hour budget. See [`benchmarking.md`](guides/evaluation.md).

You should now have a model with measurable FAR/FRR on real data, not just a
synthetic F1.

---

## Stage 4 — Hard negatives and infinite training (half a day)

**Goal:** push the false-fire rate below 1 / hour without sacrificing recall.

| Example / script | What it does |
|---|---|
| `scripts/train/train_full_nww.py` | Train with the full Not-Wake-Word negative pool |
| `scripts/train/train_infinite.py` | Goal-based loop: mine hard negatives + optionally synth VC positives until F1/EER targets hit |
| `examples/30_loss_combinations.py` | BCE + Focal + SupCon multi-loss composition |
| `examples/31_arcface_training.py` | ArcFace angular margin for tight clusters |

Docs to read: [`losses.md`](reference/losses.md), [`rppl_whitepaper.md`](research/rppl.md)
(the in-house RPPL loss), and the "Hard-negative mining" section of
[`training.md`](guides/training.md).

---

## Stage 5 — Search the architecture space (half a day, optional)

**Goal:** stop hand-picking architectures; let a search find them.

| Example / script | What it does |
|---|---|
| `examples/12_hyperparameter_sweep.py` | Optuna Bayesian search |
| `examples/28_full_architecture_search.py` | Genetic search over featurizer × head × loss |
| `scripts/train/train_sincnet_genetic.py` | GA over SincNet/Gammatone featurizers |
| `notebooks/genetic_search.ipynb` | End-to-end Kaggle-ready GA notebook |

Pair with [`sweep.md`](guides/search.md) and [`search_strategies.md`](guides/search.md).

---

## Stage 6 — Squeeze to embedded (varies)

**Goal:** fit on an ESP32 / Cortex-M without losing more than 2 F1 points.

| Example | What it does |
|---|---|
| `examples/34_esp32_nano.py` | Sub-1 KB grid search |
| `examples/35_esp32_genetic_search.py` | Sub-10 KB GA |
| `examples/36_size_aware_training.py` | `SizeAwareLoss` — train with size penalty |

Pair with [`esp32.md`](guides/embedded.md) and [`hardware_guide.md`](guides/embedded.md),
and `ww_trainer.export_c.export_to_c_header` for the actual C header output.

---

## Stage 7 — Research surface (open-ended)

**Goal:** run ablations, distil large models, contribute new components.

| Topic | Entry point |
|---|---|
| Knowledge distillation | `examples/10_knowledge_distillation.py`, [`distillation.md`](guides/distillation.md), [`tinyhubert_whitepaper.md`](research/tinyhubert.md) |
| RPPL loss internals | [`rppl_whitepaper.md`](research/rppl.md), `scripts/train/train_rppl.py` |
| Pretrained ONNX featurizers | `examples/38_markov_onnx_blackbox.py`, `examples/39_silero_vad_wrapper.py`, `examples/40_multi_onnx_pipeline.py` |
| PhonMatchNet (phoneme-conditioned) | `examples/42_phonmatch_training.py`, [`phonmatch.md`](reference/phonmatch.md) |
| OCSVM anomaly head | `examples/43_ocsvm_head.py`, [`classifiers.md`](reference/classifiers.md) |
| Markov / HMM features | `examples/33_markov_features.py`, `examples/37_hmm_advanced.py`, [`markov_hmm.md`](reference/extractors.md) |
| Ablation grids | `notebooks/nb09_ablation.ipynb`, `scripts/train/train_ablation.py` |

For framework contributions, read [`architecture.md`](internals/architecture.md),
[`audit.md`](internals/known_issues.md) (known issues), and [`../TODO.md`](../TODO.md) (open
backlog) before starting.

---

## Notebook curriculum (cloud / Kaggle / Colab)

If you prefer notebooks over scripts, the same curriculum is mirrored in
[`notebooks.md`](guides/notebooks.md) with a hardware-tier decision tree.

---

## Literature anchors

Every component in ww-trainer has an academic ancestor. The list below ties
the code to the paper and (when available) a canonical reference
implementation. Full bibliography in [`references.md`](research/references.md).

### Featurizers
- **MFCC** — Davis & Mermelstein, IEEE TASSP 1980. <https://ieeexplore.ieee.org/document/1163420>
- **Log-mel filterbank** — Stevens, Volkmann, Newman, JASA 1937 (mel scale).
- **SincNet** — Ravanelli & Bengio, *Speaker Recognition from Raw Waveform with SincNet*, SLT 2018. ArXiv <https://arxiv.org/abs/1808.00158> · GitHub <https://github.com/mravanelli/SincNet>
- **Gammatone filterbank** — Patterson & Holdsworth, 1996 (auditory model).
- **LEAF** — Zeghidour et al., *LEAF: A Learnable Frontend for Audio Classification*, ICLR 2021. ArXiv <https://arxiv.org/abs/2101.08596> · GitHub <https://github.com/google-research/leaf-audio>
- **PNCC** — Kim & Stern, *Power-Normalized Cepstral Coefficients*, IEEE/ACM TASLP 2016. <https://ieeexplore.ieee.org/document/7439789>
- **PLP** — Hermansky, *Perceptual Linear Predictive Analysis*, JASA 1990.
- **CQT** — Brown, *Calculation of a constant Q spectral transform*, JASA 1991.
- **HuBERT** — Hsu et al., *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units*, ArXiv <https://arxiv.org/abs/2106.07447> · GitHub <https://github.com/facebookresearch/fairseq/tree/main/examples/hubert>
- **Wav2Vec2** — Baevski et al., ArXiv <https://arxiv.org/abs/2006.11477> · GitHub <https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec>
- **Wav2Vec2-BERT** — Meta Seamless team, ArXiv <https://arxiv.org/abs/2312.05187>
- **DistilHuBERT** — Chang et al., ICASSP 2022. ArXiv <https://arxiv.org/abs/2110.01900> · GitHub <https://github.com/s3prl/s3prl>
- **Silero VAD** — GitHub <https://github.com/snakers4/silero-vad>

### Classifier heads
- **BC-ResNet** — Kim et al., *Broadcasted Residual Learning for Efficient Keyword Spotting*, Interspeech 2021. ArXiv <https://arxiv.org/abs/2106.04140> · GitHub <https://github.com/Qualcomm-AI-research/bcresnet>
- **TC-ResNet** — Choi et al., *Temporal Convolution for Real-time KWS on Mobile Devices*, Interspeech 2019. ArXiv <https://arxiv.org/abs/1904.03814> · GitHub <https://github.com/hyperconnect/TC-ResNet>
- **DS-CNN** — Zhang et al., *Hello Edge: Keyword Spotting on Microcontrollers*, ArXiv <https://arxiv.org/abs/1711.07128> · GitHub <https://github.com/ARM-software/ML-KWS-for-MCU>
- **Res15** — Tang & Lin, *Deep Residual Learning for Small-Footprint Keyword Spotting*, ICASSP 2018. ArXiv <https://arxiv.org/abs/1710.10361> · GitHub <https://github.com/castorini/honk>
- **MatchboxNet** — Majumdar & Ginsburg, *MatchboxNet — 1D Time-Channel Separable Convolutional Neural Network for Speech Commands Recognition*, Interspeech 2020. ArXiv <https://arxiv.org/abs/2004.08531> · GitHub <https://github.com/NVIDIA/NeMo>
- **KWT (Keyword Transformer)** — Berg et al., *Keyword Transformer: A Self-Attention Model for Keyword Spotting*, Interspeech 2021. ArXiv <https://arxiv.org/abs/2104.00769> · GitHub <https://github.com/ARM-software/keyword-transformer>
- **Conformer** — Gulati et al., *Conformer: Convolution-augmented Transformer for Speech Recognition*, Interspeech 2020. ArXiv <https://arxiv.org/abs/2005.08100>
- **MixConv** — Tan & Le, *MixConv: Mixed Depthwise Convolutional Kernels*, BMVC 2019. ArXiv <https://arxiv.org/abs/1907.09595> · GitHub <https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet>
- **EfficientNet** — Tan & Le, ICML 2019. ArXiv <https://arxiv.org/abs/1905.11946>
- **PhonMatchNet** — Lee et al., Interspeech 2023. ArXiv <https://arxiv.org/abs/2308.16511>
- **ConvAttentionHead** — ported from LiveKit. GitHub <https://github.com/livekit/livekit-wakeword> (Apache-2.0)
- **OCSVM head** — Schölkopf et al., *Estimating the Support of a High-Dimensional Distribution*, Neural Computation 2001. <https://doi.org/10.1162/089976601750264965>

### Losses
- **Focal loss** — Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017. ArXiv <https://arxiv.org/abs/1708.02002>
- **ArcFace** — Deng et al., CVPR 2019. ArXiv <https://arxiv.org/abs/1801.07698> · GitHub <https://github.com/deepinsight/insightface>
- **SupCon** — Khosla et al., NeurIPS 2020. ArXiv <https://arxiv.org/abs/2004.11362> · GitHub <https://github.com/HobbitLong/SupContrast>
- **NT-Xent / SimCLR** — Chen et al., ICML 2020. ArXiv <https://arxiv.org/abs/2002.05709>
- **Triplet loss** — Schroff et al., *FaceNet*, CVPR 2015. ArXiv <https://arxiv.org/abs/1503.03832>
- **HALO** — Hou et al., *Regional Hard-Example mining for KWS*, ICASSP 2020. <https://ieeexplore.ieee.org/document/9053009>
- **RPPL** — Robust Prototype Diversity Loss, ww-trainer in-house. See [`rppl_whitepaper.md`](research/rppl.md).

### Search & training
- **Optuna** — Akiba et al., KDD 2019. ArXiv <https://arxiv.org/abs/1907.10902> · GitHub <https://github.com/optuna/optuna>
- **Island-model GA** — Cantú-Paz, *Efficient and Accurate Parallel Genetic Algorithms*, 2000.
- **Knowledge distillation** — Hinton, Vinyals, Dean, NeurIPS 2014 workshop. ArXiv <https://arxiv.org/abs/1503.02531>
- **Synthesized speech for KWS** — Lin et al., *Training Keyword Spotters with Limited and Synthesized Speech Data*, ArXiv <https://arxiv.org/abs/2002.01322>
- **Voice conversion (voiceclonnx)** — GitHub <https://github.com/TigreGotico/voiceclonnx>

### Data & deployment
- **AudioSet** — Gemmeke et al., ICASSP 2017. <https://research.google.com/audioset/>
- **Speech Commands** — Warden, 2018. ArXiv <https://arxiv.org/abs/1804.03209>
- **MIT IR Survey (RIRs)** — Traer & McDermott, 2016. <https://mcdermottlab.mit.edu/Reverb/IR_Survey.html>
- **ONNX Runtime** — GitHub <https://github.com/microsoft/onnxruntime>
- **onnxruntime-web** — same repo, browser/WASM bindings.
- **OpenVoiceOS** — downstream consumer of ww-trainer models. GitHub <https://github.com/OpenVoiceOS>

