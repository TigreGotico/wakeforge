# ww-trainer Examples

Runnable demos covering every architecture in ww-trainer.

## Feature Extractors x Classifier Heads

| # | Example | Extractor | Head | Focus |
|---|---------|-----------|------|-------|
| 01 | `01_mfcc_ffn_micro.py` | MFCC | FFN | Smallest model, ONNX export |
| 02 | `02_mfcc_gru_small.py` | MFCC | GRU | Temporal modeling, bidirectional |
| 03 | `03_mfcc_bcresnet.py` | MFCC / FilterBank | BC-ResNet | 2D conv, tau scaling |
| 04 | `04_filterbank_cnn.py` | FilterBank | CNN | Log-mel + 1D convolutions |
| 05 | `05_sincnet_gru.py` | SincNet | GRU | Learnable filterbank |
| 06 | `06_gammatone_ffn.py` | Gammatone | FFN | Auditory-inspired features |
| 07 | `07_delta_mfcc_ffn.py` | Delta-MFCC | FFN | Velocity + acceleration features |
| 14 | `14_hubert_training.py` | HuBERT / Wav2Vec2 | GRU | Transformer extractors (requires `transformers`) |
| 15 | `15_tcresnet.py` | MFCC | TC-ResNet | 1D temporal conv residual (Choi 2019) |
| 16 | `16_dscnn.py` | MFCC | DS-CNN | Depthwise-separable 2D CNN (Zhang 2017) |
| 17 | `17_matchboxnet.py` | FilterBank | MatchboxNet | Time-channel separable 1D (NVIDIA 2020) |
| 18 | `18_res15.py` | MFCC | Res15 | Dilated residual 1D (Tang & Lin 2018) |
| 19 | `19_kwt_transformer.py` | FilterBank | KWT | Keyword Transformer (Berg 2021) |
| 20 | `20_conformer.py` | FilterBank | Conformer | Conv-augmented transformer (Gulati 2020) |
| 21 | `21_crnn.py` | MFCC | CRNN | CNN + GRU hybrid |
| 22 | `22_leaf_extractor.py` | LEAF | FFN / GRU | Learnable audio frontend (Google 2021) |

## Classical & Specialized Extractors

| # | Example | Extractor | Focus |
|---|---------|-----------|-------|
| 23 | `23_plp_pncc_cqt.py` | PLP / PNCC / CQT | Classical perceptual extractors (Hermansky 1990, Kim & Stern 2016) |
| 24 | `24_vad_enriched.py` | MFCC + VoiceActivity | VAD-enriched features: energy, ZCR, spectral flatness, VAD prob |
| 25 | `25_pitch_enriched.py` | MFCC + Pitch | Pitch features: F0, voicing probability, F0 delta |
| 26 | `26_multi_resolution.py` | MultiResolution(MFCC, MFCC) | Fine + coarse temporal resolution concatenation |
| 27 | `27_snr_aware.py` | MFCC + SNRAware | Per-frame SNR estimation for noisy environments |

## Inference & Deployment

| # | Example | Focus |
|---|---------|-------|
| 08 | `08_onnx_inference.py` | Zero-PyTorch inference with onnxruntime |
| 09 | `09_streaming_inference.py` | Real-time chunk-by-chunk detection |

## Training & Research

| # | Example | Focus |
|---|---------|-------|
| 10 | `10_knowledge_distillation.py` | Compress large model into small student |
| 11 | `11_hardware_tiers.py` | Pre-configured tier presets |
| 12 | `12_hyperparameter_sweep.py` | Optuna auto-tuning (requires `optuna`) |
| 13 | `13_benchmark_extractors.py` | Latency and throughput comparison |
| 28 | `28_full_architecture_search.py` | Genetic search over full architecture space: extractors + heads + losses |
| 29 | `29_stacked_enrichment.py` | Stack all enrichment wrappers: MFCC + VAD + Pitch + SNR + Delta |
| 30 | `30_loss_combinations.py` | LossManager multi-loss configs: BCE + Focal + SupCon |
| 31 | `31_arcface_training.py` | ArcFace angular margin loss for embedding separation |
| 32 | `32_full_pipeline.py` | End-to-end: build → train → ONNX export → inference → streaming |

## Specialised Extractors & Embedded

| # | Example | Focus |
|---|---------|-------|
| 33 | `33_markov_features.py` | Markov transition extractors — temporal modelling without neural nets |
| 34 | `34_esp32_nano.py` | ESP32 nano tier: sub-1 KB models via grid search |
| 35 | `35_esp32_genetic_search.py` | ESP32 GA: find best sub-10 KB model via evolutionary optimisation |
| 36 | `36_size_aware_training.py` | `SizeAwareLoss` — train a tiny model with sparsity + size penalties |
| 37 | `37_hmm_advanced.py` | Hybrid MFCC + HMM + GRU |
| 38 | `38_markov_onnx_blackbox.py` | Use a Markov ONNX featurizer as a black-box `OnnxFeatureExtractor` |
| 39 | `39_silero_vad_wrapper.py` | Wrap pre-trained Silero VAD as an ONNX feature stream |
| 40 | `40_multi_onnx_pipeline.py` | Chain multiple ONNX models in a single inference pipeline |
| 41 | `41_hmm_feature_extraction.py` | Classical HMM feature extraction |
| 42 | `42_phonmatch_training.py` | PhonMatchNet training with IPA phoneme conditioning |
| 43 | `43_ocsvm_head.py` | `OCSVMHead` — FFN backbone + One-Class SVM classifier (anomaly-style) |

## Browser tester

`wakeword_tester.html` — single-page HTML that loads a featurizer + head ONNX
pair via onnxruntime-web and tests them from the microphone. No build step.

## Recommended order — Zero to Hero

If you are learning the framework end-to-end, work through examples in this
order rather than numerically:

1. **01 → 02** — simplest model, then a small recurrent one.
2. **08 → 09** — load your trained ONNX and run it (offline + streaming).
3. **11** — see the tier presets.
4. **04 → 05 → 06 → 07** — alternative featurizers.
5. **15 → 17 → 18 → 16** — production-grade CNN/TCN heads.
6. **30 → 31** — multi-loss training and ArcFace.
7. **24 → 25 → 27 → 29** — enrichment wrappers and stacking.
8. **12 → 28** — Optuna and full GA architecture search.
9. **34 → 35 → 36** — squeeze to ESP32.
10. **14 → 10** — large SSL features, then distil to small.
11. **40 → 38 → 39** — multi-ONNX pipelines and external pretrained features.
12. **43** — OCSVM anomaly-style classifier head.
13. **32** — the full end-to-end pipeline.

## Quick Start

```bash
cd ww-trainer

# Run any example
.venv/bin/python examples/01_mfcc_ffn_micro.py

# Run all examples (skip those requiring optional deps)
for f in examples/[0-9]*.py; do
    echo "=== $f ==="
    .venv/bin/python "$f" || echo "(skipped)"
    echo
done
```
