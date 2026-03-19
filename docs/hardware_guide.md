# Hardware Targeting Guide

Configurations for deploying wake word models from microcontrollers to servers. Based on hardware tier presets in `TierConfig` -- `ww_trainer/tiers.py:12` and the full set of extractors/heads.

---

## Predefined Tiers

Defined in `HARDWARE_TIERS` -- `ww_trainer/tiers.py:26`.

| Tier | Extractor | Head | Hidden | Params | Target |
|------|-----------|------|--------|--------|--------|
| `micro` | MFCC (40) | FFN | 128 | ~50K | MCU, RPi Zero |
| `small` | MFCC (40) | GRU | 128 | ~200K | RPi, small SBC |
| `medium` | ONNX (HuBERT) | FFN | 128 | ~90M feat + 200K head | RPi 4, laptop |
| `large` | HuBERT | GRU (bidir, 2-layer) | 256 | ~300M feat + 1M head | Server |
| `sincnet_small` | SincNet | GRU | 128 | ~300K | RPi, small SBC |
| `filterbank_small` | FilterBank | GRU | 128 | ~200K | RPi, small SBC |
| `delta_micro` | Delta+MFCC (13x3) | FFN | 128 | ~55K | MCU, RPi Zero |
| `gammatone_small` | Gammatone | GRU | 128 | ~200K | RPi, small SBC |

Use `get_tier("micro")` (`tiers.py:120`) or `--tier micro` on CLI.

---

## Hardware Profiles

### MCU (ARM Cortex-M4/M7, ESP32)

**Constraints:** 256KB-1MB RAM, no floating point (use INT8 quantized ONNX), <10ms inference.

| Component | Recommendation |
|-----------|---------------|
| Extractor | `MfccExtractor(n_mfcc=13)` + `DeltaExtractor` (39-dim) or plain `MfccExtractor(n_mfcc=40)` |
| Head | `FfnClassifierHead(hidden_dim=64)` or `DSCNNHead(size="S")` or `BCResNetHead(tau=1)` |
| Loss | BCE or Focal |
| Total params | <30K (head only; MFCC has no learnable params) |
| Quantization | INT8 via `export_to_onnx(quantize=True)` |

**Best combos:**
1. MFCC(13) + Delta + FFN(64) -- ~10K params, simplest
2. MFCC(40) + DSCNN-S -- ~20K params, better accuracy
3. MFCC(40) + BCResNet(tau=1) -- ~6K params, smallest CNN option

---

### RPi Zero / RPi Zero 2W

**Constraints:** 512MB RAM, single-core ARM (Zero) or quad-core (Zero 2W), no GPU. Target <50ms inference.

| Component | Recommendation |
|-----------|---------------|
| Extractor | `MfccExtractor(n_mfcc=40)` or `PNCCExtractor` (noisy) |
| Head | `BCResNetHead(tau=2-3)`, `MatchboxNetHead(B=3,R=1,C=64)`, `TCResNetHead(variant=8)` |
| Loss | Focal + SupCon |
| Total params | <50K |
| Quantization | INT8 ONNX |

**Best combos:**
1. MFCC(40) + BCResNet(tau=3) -- ~43K params
2. MFCC(40) + MatchboxNet-3x1x64 -- ~25K params
3. PNCC(13) + SNRAware + TCResNet8 -- ~35K params (noisy environments)

---

### RPi 3/4 / Jetson Nano

**Constraints:** 1-4GB RAM, quad-core ARM or ARM+GPU. Target <100ms inference.

| Component | Recommendation |
|-----------|---------------|
| Extractor | `SincNetExtractor`, `FilterbankExtractor`, `GammatoneExtractor`, or `OnnxFeatureExtractor` (quantized HuBERT) |
| Head | `BCResNetHead(tau=6-8)`, `GruClassifierHead`, `CRNNHead`, `Res15Head` |
| Loss | Focal + SupCon + Center |
| Total params | <300K head (extractor varies) |

**Best combos:**
1. SincNet(80) + GRU(128) -- ~300K, learnable frontend
2. FilterBank(80) + BCResNet(tau=8) -- ~280K head, fast
3. ONNX-HuBERT(quantized) + FFN(128) -- ~98K head, best accuracy
4. Gammatone(64) + CRNN -- ~150K, noise-robust

---

### Laptop / Desktop (CPU)

**Constraints:** 8-32GB RAM, x86 CPU. Target <200ms inference.

| Component | Recommendation |
|-----------|---------------|
| Extractor | `OnnxFeatureExtractor` (HuBERT/Wav2Vec2 ONNX) or `LEAFExtractor` |
| Head | `ConformerHead`, `KWTHead`, `BCResNetHead(tau=8)` |
| Loss | BCE + ArcFace + SupCon |
| Total params | Unlimited for head |

**Best combos:**
1. ONNX-HuBERT + Conformer(d_model=64, n_layers=4) -- best accuracy
2. ONNX-HuBERT + KWT(d_model=64) -- transformer on transformer
3. LEAF(40) + BCResNet(tau=8) -- fully learnable, no pretrained model

---

### Server / Workstation (GPU)

**Constraints:** None. Maximize accuracy for training; distill for deployment.

| Component | Recommendation |
|-----------|---------------|
| Extractor | `HubertExtractor`, `Wav2Vec2Extractor`, `Wav2Vec2BertExtractor` |
| Head | `ConformerHead`, `KWTHead`, `GruClassifierHead(bidirectional=True, gru_n_layers=2)` |
| Loss | BCE + ArcFace + SupCon + Center |

**Training strategy:**
1. Train large model: HuBERT + Conformer
2. Distill to `CnnLstmExtractor` via `KnowledgeDistillationTrainer` -- `ww_trainer/distill.py:143`
3. Export student to ONNX, quantize
4. Deploy student ONNX on target hardware

---

## Knowledge Distillation Pipeline

`CnnLstmExtractor` -- `distill.py:32`: 4-layer strided Conv1D (total stride 160) + 2-layer bidirectional LSTM + linear projection. 500K-2M params depending on config.

`KnowledgeDistillationTrainer` -- `distill.py:143`: Trains student to mimic teacher features (MSE loss) while also classifying wake words (BCE loss). Weight controlled by `alpha` (default 0.7 for distillation, 0.3 for task).

Convenience function: `distill_hubert_to_cnn_lstm()` -- `distill.py:463`.

**Distillation workflow:**
1. Export teacher (HuBERT) to ONNX
2. Train student: `distill_hubert_to_cnn_lstm("hubert.onnx", train_data, val_data)`
3. Student ONNX exported automatically to `output_dir/student_extractor.onnx`
4. Use `OnnxFeatureExtractor("student_extractor.onnx")` on target device

---

## ONNX Inference

`OnnxWakeWordInferencer` -- `ww_trainer/inference.py:8`: Two-session ONNX inference (extractor + head). No PyTorch dependency at runtime.

Supports:
- Single inference: `infer(audio_1d)` -- `inference.py:36`
- Batch inference: `infer_batch(audio_2d)` -- `inference.py:57`
- Streaming inference: `infer_streaming(chunk, cache)` -- `inference.py:74` with rolling 50-frame cache

---

## Latency Budget Guidelines

| Hardware | Max Total Latency | Extractor Budget | Head Budget |
|----------|-------------------|-----------------|-------------|
| MCU | 10ms | 5ms (MFCC) | 5ms |
| RPi Zero | 50ms | 20ms | 30ms |
| RPi 4 | 100ms | 60ms (ONNX HuBERT quantized) | 40ms |
| Laptop | 200ms | 100ms | 100ms |
| Server | Unlimited | Unlimited | Unlimited |

Quantize ONNX models for MCU/RPi: `export_to_onnx(out, quantize=True)` produces INT8 and INT16 variants (`feats.py:96-102`).
