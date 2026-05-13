# End-to-End Recipes

Complete configurations for common deployment scenarios. Each recipe specifies extractor, head, loss, training command, and export.

---

## 1. Smallest Possible Model (Microcontroller)

**Target:** ARM Cortex-M4/M7, ESP32. <30K params, INT8 ONNX, <10ms inference.

**Config:** MFCC(13) + Delta + BCResNet(tau=1) -- ~6K head params.

```bash
ww_trainer-train \
  --wake-word hey_jarvis \
  --metadata dataset.csv \
  --featurizer-type mfcc --n-mfcc 13 \
  --arch bcresnet --tau 1 \
  --loss-type focal --loss-weight 1.0 \
  --epochs 50 --batch-size 32 \
  --export-onnx --save-best \
  --output-dir ./models/mcu
```

**Export with quantization:**
```python
from ww_trainer.feats import MfccExtractor
ext = MfccExtractor(n_mfcc=13)
ext.export_to_onnx("mfcc13.onnx", quantize=True)
# Produces: mfcc13.onnx, mfcc13_int8.onnx, mfcc13_int16.onnx
```

**Alternative:** MFCC(40) + DSCNN-S (~20K params) or MFCC(40) + FFN(64) (~10K params).

---

## 2. Maximum Accuracy (GPU Available)

**Target:** Server/workstation. Accuracy is the only concern.

**Config:** HuBERT + Conformer + BCE+ArcFace+SupCon.

```bash
ww_trainer-train \
  --wake-word hey_jarvis \
  --metadata dataset.csv \
  --tier large \
  --loss-type bce,arcface,supcon \
  --loss-weight 0.5,0.3,0.2 \
  --epochs 30 --batch-size 64 \
  --amp --device cuda \
  --save-best --export-onnx \
  --output-dir ./models/server
```

For even higher accuracy, replace the default GRU head with Conformer:

```bash
ww_trainer-train \
  --wake-word hey_jarvis \
  --metadata dataset.csv \
  --featurizer-type hubert \
  --featurizer "voidful/hubert-tiny-v2" \
  --arch conformer --d-model 64 --n-layers 4 \
  --loss-type bce,arcface,supcon \
  --loss-weight 0.5,0.3,0.2 \
  --epochs 30 --amp --device cuda \
  --save-best --output-dir ./models/best
```

**Deploy via distillation** (see [distillation.md](distillation.md)):
```python
from ww_trainer.distill import distill_hubert_to_cnn_lstm
distill_hubert_to_cnn_lstm("hubert.onnx", train_data, val_data, output_dir="student/")
```

---

## 3. Noisy Environment

**Target:** Kitchen, car, factory. Robust to background noise and reverb.

**Config:** PNCC + SNRAware + BCResNet(tau=3) + Focal loss.

PNCC is inherently noise-robust via power normalization (`feats.py:1100`). SNRAware adds per-frame SNR estimation (`feats.py:1630`).

```bash
ww_trainer-train \
  --wake-word hey_jarvis \
  --metadata dataset.csv \
  --featurizer-type pncc --n-pncc 13 \
  --arch bcresnet --tau 3 \
  --loss-type focal,supcon \
  --loss-weight 0.7,0.3 \
  --aug-prob 0.9 \
  --bg-noise-folder /data/musan/noise \
  --bg-speech-folder /data/musan/speech \
  --rir-folder /data/rirs \
  --snr-min 0 --snr-max 15 \
  --epochs 50 --batch-size 32 \
  --save-best --output-dir ./models/noisy
```

**Key:** Aggressive augmentation (`--aug-prob 0.9`) with low SNR range and RIR convolution. The PNCC extractor handles the rest.

**Alternative extractors for noise:** Gammatone (`feats.py:599`) or MFCC + VoiceActivityExtractor + SNRAwareExtractor stack.

---

## 4. Automatic Architecture Search

**Target:** Find the best extractor + head + loss combination without manual tuning.

**Config:** Genetic search with `full=True` over 4 extractors x 10 heads x 6 losses.

```python
from ww_trainer.sweep import run_genetic_search

results = run_genetic_search(
    metadata_csv="dataset.csv",
    population_size=20,
    generations=10,
    output_dir="arch_search",
    featurizer_type="mfcc",  # base type; overridden by full search
    epochs_per_trial=5,
    full=True,  # search extractors, heads, and losses
)
# Output: arch_search/genetic_results.json
```

Full search space (`sweep.py:131-147`):
- **Extractors:** mfcc, filterbank, sincnet, gammatone
- **Heads:** ffn, gru, cnn, bcresnet, tcresnet, dscnn, matchboxnet, res15, conformer, crnn
- **Losses:** bce, focal, label_smoothing_bce, supcon, arcface, ntxent
- **Features:** 13, 40, 64, 80 dimensions

After finding the best config, retrain with full epochs:

```bash
ww_trainer-train \
  --wake-word hey_jarvis \
  --metadata dataset.csv \
  --featurizer-type <best_type> --arch <best_arch> \
  --loss-type <best_loss> \
  --epochs 50 --save-best \
  --output-dir ./models/final
```

---

## 5. Deploy on Raspberry Pi (No PyTorch)

**Target:** RPi 3/4, ONNX Runtime only at inference time.

**Step 1 -- Train (on any machine):**
```bash
ww_trainer-train \
  --wake-word hey_jarvis \
  --metadata dataset.csv \
  --tier small \
  --epochs 50 --save-best --export-onnx \
  --output-dir ./models/rpi
```

**Step 2 -- Export extractor:**
```python
from ww_trainer.feats import MfccExtractor
ext = MfccExtractor(n_mfcc=40)
ext.export_to_onnx("models/rpi/mfcc.onnx", quantize=True)
```

**Step 3 -- Deploy (RPi, no torch installed):**
```python
# Only requires: pip install numpy onnxruntime
from ww_trainer.inference import OnnxWakeWordInferencer
import numpy as np

inf = OnnxWakeWordInferencer(
    "models/rpi/mfcc.onnx",
    "models/rpi/classifier_head.onnx",
)

# Streaming detection
cache = None
for chunk in microphone_stream():  # float32 arrays, 4000 samples
    prob, cache = inf.infer_streaming(chunk, cache)
    if prob > 0.5:
        print("Wake word detected!")
        cache = None
```

**For higher accuracy on RPi 4:** Use distilled HuBERT student instead of MFCC (see recipe 2 + [distillation.md](distillation.md)). Student ONNX runs in <10ms on RPi 4.

---

## Recipe Summary

| Recipe | Extractor | Head | Loss | Params (head) | Target |
|--------|-----------|------|------|---------------|--------|
| MCU smallest | MFCC(13)+Delta | BCResNet(tau=1) | Focal | ~6K | Cortex-M |
| Max accuracy | HuBERT | Conformer | BCE+ArcFace+SupCon | ~100K | GPU server |
| Noisy env | PNCC+SNRAware | BCResNet(tau=3) | Focal+SupCon | ~43K | RPi/SBC |
| Auto search | (searched) | (searched) | (searched) | varies | Any |
| RPi deploy | MFCC(40) | GRU(128) | BCE | ~200K | RPi 3/4 |
