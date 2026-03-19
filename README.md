# Wake Word Trainer

A full training and research suite for wake word detection — from MCUs to servers. All components export to ONNX; inference needs only `onnxruntime` + `numpy`.

## Architecture

```
[Feature Extractor]  →  [Classifier Head]  →  [Binary Decision]
  raw audio → [B,T,F]     [B,T,F] → [B]        sigmoid → 0/1
```

Both components are independently exportable to ONNX. At inference time, no PyTorch is needed.

## What's Included

| Component | Count | Examples |
|-----------|-------|---------|
| **Feature Extractors** | 17 | MFCC, FilterBank, SincNet, Gammatone, LEAF, PLP, PNCC, CQT, HuBERT, Wav2Vec2, Delta, VAD, Pitch, SNR, MultiRes |
| **Classifier Heads** | 11 | FFN, GRU, CNN, BC-ResNet, TC-ResNet, DS-CNN, MatchboxNet, Res15, KWT, Conformer, CRNN |
| **Loss Functions** | 17 | BCE, Focal, ArcFace, SupCon, NTXent, ProxyNCA, MultiSimilarity, Triplet, RPPL, ... |
| **Search Strategies** | 4 | Optuna (Bayesian), Grid, Random, Genetic Algorithm |
| **Hardware Tiers** | 8 | micro, small, medium, large, sincnet_small, filterbank_small, delta_micro, gammatone_small |
| **Examples** | 32 | End-to-end runnable scripts for every architecture |
| **Tests** | 400+ | 76%+ coverage |

## Install

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Optional
uv pip install transformers  # HuBERT/Wav2Vec2
uv pip install optuna        # Bayesian sweep
uv pip install mlflow        # Experiment tracking
```

## Quick Start

```python
from ww_trainer import MfccExtractor, GruClassifierHead, BaseWakeModel

extractor = MfccExtractor(n_mfcc=40)
head = GruClassifierHead(input_size=40, hidden_dim=128, device="cpu")
model = BaseWakeModel(extractor, head, device="cpu")

# Export to ONNX
extractor.export_to_onnx("extractor.onnx")
head.export_to_onnx("head.onnx")

# Inference (no PyTorch needed)
from ww_trainer import OnnxWakeWordInferencer
inf = OnnxWakeWordInferencer("extractor.onnx", "head.onnx")
prob = inf.infer(audio_numpy)  # → float in [0, 1]
```

## CLI Training

```bash
ww_trainer-train \
  --wake-word "hey jarvis" \
  --metadata dataset.csv \
  --arch gru \
  --featurizer-type mfcc \
  --loss-type bce,focal \
  --epochs 50 \
  --export-onnx
```

## Full Architecture Search

```python
from ww_trainer.sweep import run_genetic_search

run_genetic_search(
    "dataset.csv",
    full=True,           # search extractors + heads + losses
    population_size=30,
    generations=20,
)
```

## Feature Enrichment (Novel)

Stack VAD, pitch, and SNR signals onto any base extractor:

```python
from ww_trainer import MfccExtractor, VoiceActivityExtractor, PitchExtractor, SNRAwareExtractor

ext = MfccExtractor(n_mfcc=13)              # 13 dims
ext = VoiceActivityExtractor(ext)            # +4 = 17
ext = PitchExtractor(ext)                    # +3 = 20
ext = SNRAwareExtractor(ext)                 # +2 = 22
```

## Documentation

See [`docs/`](docs/) for comprehensive guides:

| Doc | Contents |
|-----|----------|
| [extractors.md](docs/extractors.md) | All 17 feature extractors |
| [classifiers.md](docs/classifiers.md) | All 11 classifier heads |
| [losses.md](docs/losses.md) | All 17 loss functions |
| [hardware_guide.md](docs/hardware_guide.md) | MCU → server targeting |
| [recipes.md](docs/recipes.md) | End-to-end recipes |
| [sweep.md](docs/sweep.md) | Search strategies |
| [streaming.md](docs/streaming.md) | Real-time inference |
| [distillation.md](docs/distillation.md) | Model compression |
| [enrichment.md](docs/enrichment.md) | VAD/Pitch/SNR wrappers |

## Examples

32 runnable scripts in [`examples/`](examples/). See [examples/README.md](examples/README.md).

## License

Apache 2.0
