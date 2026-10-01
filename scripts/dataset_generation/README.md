# Dataset Generation Examples

This folder contains standalone scripts for generating, augmenting, and benchmarking wake-word datasets.

---

## 1. Adversarial Generation

Generate phonetically similar "hard negatives" using GraphemeAug and LLMs.

```bash
# Generate 20 samples for "hey mycroft" using local LLM (Ollama)
python scripts/dataset_generation/01_adversarial_gen.py \
    --words "hey_mycroft" \
    --n-samples 20 \
    --llm-url "http://localhost:11434" \
    --llm-model "gemma3:4b"
```

## 2. TTS Synthesis & Voice Conversion

Synthesize wake-word samples using multiple engines and randomly clone voices.

```bash
# Generate 100 samples for "hey mycroft" using Edge TTS and Google
# Optional: use --vc-refs to point to a folder of target voices for cloning
python scripts/dataset_generation/02_tts_synth.py \
    --words "hey_mycroft" \
    --n-samples 100 \
    --out-dir ./data/synth
```

## 3. Stochastic Training Augmentation

Apply random background noise, reverb, and pitch/speed shifts to an existing dataset.

```bash
# Requires a metadata.csv in --input-dir
python scripts/dataset_generation/03_training_aug.py \
    --input-dir ./data/raw \
    --out-dir ./data/augmented \
    --bg-noise-dir ./data/noises/ambient \
    --rir-dir ./data/noises/rirs \
    --prob 0.9
```

## 4. Deterministic Benchmark Generation

Create structured test sets at fixed SNR levels for performance evaluation.

```bash
python scripts/dataset_generation/04_benchmark_gen.py \
    --clean-dir ./data/test/clean \
    --out-dir ./data/test/benchmarks \
    --noise-dir ./data/noises/ambient \
    --snrs "20,15,10,5,0"
```

## 5. OVOS Dataset Collection

Specific pipeline for collecting and synthesising datasets using OVOS plugins.

```bash
python scripts/dataset_generation/05_ovos_vc_gen.py \
    --words "hey_mycroft" \
    --out-dir ./data/ovos_dataset \
    --target-samples 500
```
