# Markov & HMM Feature Extractors

`ww-trainer` includes two classical statistical models adapted for high-performance wake-word feature extraction: `MarkovTransitionExtractor` and `HMMStateExtractor`.

These extractors bridge the gap between "bag of features" (like MFCC) and "temporal deep learning" (like GRU/Transformers) by capturing sequential patterns without adding many trainable neural parameters.

---

## 1. Markov Transition Extractor

The `MarkovTransitionExtractor` captures the local transition dynamics of quantized audio.

### How it works:
1.  **Quantization**: Continuous audio frames (MFCC, etc.) are mapped to discrete tokens via a Vector Quantization (VQ) codebook.
2.  **Context Mapping**: The last $N$ tokens (determined by `order`) are mapped to a unique **State ID**.
3.  **Transition Probability**: The extractor looks up the probability of each token following the current state from a trained transition matrix.
4.  **Enrichment**: These probabilities are appended to the base features.

### Training:
You must call `.fit(audio_list)` with wake-word samples.
```python
from ww_trainer.feats import MfccExtractor, MarkovTransitionExtractor

base = MfccExtractor()
markov = MarkovTransitionExtractor(base, n_codes=16, order=2)
markov.fit(wake_word_samples)
```

---

## 2. HMM State Extractor

The `HMMStateExtractor` uses a Hidden Markov Model to estimate the probability of the audio being in specific "latent states" (similar to phonemes).

### How it works:
It implements the **HMM Forward Algorithm**. For every frame, it computes the posterior probability distribution over all latent states.
- If your wake word has 4 distinct sounds, an 4-state HMM will output 4 probability channels that "light up" sequentially as the word is spoken.

### Training:
Uses the Baum-Welch algorithm (unsupervised) via the `markovonnx` library.
```python
from ww_trainer.feats import HMMStateExtractor

hmm = HMMStateExtractor(base, n_states=8, n_codes=32)
hmm.fit(wake_word_samples, n_iter=10)
```

---

## 3. Hybrid ONNX Pipelines

Both extractors support **Full Pipeline Export**. When you call `export_to_onnx()`, the resulting graph contains:
1.  The base featurizer (FFT, Mel-filterbanks, etc.)
2.  The VQ Codebook (for quantization)
3.  The Markov/HMM logic (Transition lookups or Forward algorithm)

The entire chain is optimized into a single symbolic graph that takes raw waveforms and outputs enriched features, ready for an ONNX classifier head.

### Performance Tip:
HMM and Markov extractors are extremely lightweight for inference compared to Transformers, making them ideal for "Secondary Validation" on microcontrollers.
