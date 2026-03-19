# Feature Extractors

All extractors inherit from `BaseExtractor` -- `ww_trainer/feats.py:57` and output `[B, T, F]` tensors.

---

## Standalone Extractors

| Extractor | Class | Line | Output Dim | ONNX Export | Learnable |
|-----------|-------|------|------------|-------------|-----------|
| MFCC | `MfccExtractor` | `feats.py:246` | `n_mfcc` (40) | Yes | No |
| FilterBank | `FilterbankExtractor` | `feats.py:332` | `n_mels` (80) | Yes | No |
| PLP | `PLPExtractor` | `feats.py:982` | `n_plp` (13) | Yes | No |
| PNCC | `PNCCExtractor` | `feats.py:1100` | `n_pncc` (13) | Yes | No |
| CQT | `CQTExtractor` | `feats.py:1217` | `n_bins * n_octaves` (84) | Yes | No |
| SincNet | `SincNetExtractor` | `feats.py:400` | `n_filters` (80) | Yes | Yes (freq bounds) |
| Gammatone | `GammatoneExtractor` | `feats.py:599` | `n_filters` (64) | Yes | No |
| LEAF | `LEAFExtractor` | `feats.py:815` | `n_filters` (40) | Yes | Yes (full frontend) |
| HuBERT | `HubertExtractor` | `feats.py:696` | `hidden_size` (768) | Via export | No (pretrained) |
| Wav2Vec2 | `Wav2Vec2Extractor` | `feats.py:730` | `hidden_size` (768) | Via export | No (pretrained) |
| Wav2Vec2Bert | `Wav2Vec2BertExtractor` | `feats.py:764` | `hidden_size` (1024) | Via export | No (pretrained) |
| ONNX | `OnnxFeatureExtractor` | `feats.py:105` | Model-dependent | N/A (already ONNX) | No |

## Wrapper Extractors

Wrappers decorate a base extractor, adding extra feature channels.

| Wrapper | Class | Line | Extra Dims | What It Adds |
|---------|-------|------|------------|--------------|
| Delta | `DeltaExtractor` | `feats.py:509` | `2 * base_dim` | First and second temporal derivatives |
| VoiceActivity | `VoiceActivityExtractor` | `feats.py:1321` | 4 | Log RMS energy, ZCR, spectral flatness, VAD probability |
| Pitch | `PitchExtractor` | `feats.py:1443` | 3 | Normalized F0, voicing probability, F0 delta |
| MultiResolution | `MultiResolutionExtractor` | `feats.py:1577` | `coarse_dim` | Concatenates fine + coarse extractor outputs |
| SNRAware | `SNRAwareExtractor` | `feats.py:1630` | 2 | Per-frame SNR estimate, noise floor estimate |
| Markov | `MarkovTransitionExtractor` | `feats.py:1757` | `n_codes` | Transition probabilities from trained Markov chain |
| HMM | `HMMStateExtractor` | `feats.py:1953` | `n_states` | Latent state posterior probabilities (Forward algorithm) |

Note: Unlike other wrappers, Markov and HMM extractors **do** support direct ONNX export of the full pipeline (Base + Wrapper).

---

## Detailed Descriptions

### MfccExtractor -- `feats.py:246`

Pure-PyTorch MFCC (Mel-Frequency Cepstral Coefficients). Computes STFT, mel filterbank, log, then DCT.

**Parameters:** `sr` (16000), `n_mfcc` (40), `n_mels` (40), `n_fft` (400), `hop_length` (160), `f_min` (0.0), `f_max` (sr/2).

**When to use:** Default choice for constrained devices. Well-understood, compact, ONNX-safe. Best when combined with Delta wrapper for temporal context.

**When NOT to use:** Noisy environments (PNCC or Gammatone better). High-accuracy requirements where transformer extractors dominate.

**Hardware fit:** MCU, RPi Zero, any device. Zero learnable parameters in extractor.

---

### FilterbankExtractor -- `feats.py:332`

Log-mel spectrogram without DCT. Same pipeline as MFCC minus the final DCT step.

**Parameters:** `sr` (16000), `n_mels` (80), `n_fft` (400), `hop_length` (160), `f_min` (0.0), `f_max` (sr/2).

**When to use:** When using 2D CNN heads (BCResNet, DSCNN) that benefit from the full mel spectrogram. Standard input for Whisper-style models.

**When NOT to use:** When feature dimension must be very small (use MFCC with fewer coefficients instead).

**Hardware fit:** MCU to laptop. Slightly larger features than MFCC but no extra compute.

---

### PLPExtractor -- `feats.py:982`

Perceptual Linear Prediction (Hermansky 1990). Models human auditory perception via Bark-scale warping, equal-loudness pre-emphasis, and cube-root compression.

**Parameters:** `sr` (16000), `n_plp` (13), `n_fft` (512), `hop_length` (160), `n_bark` (21), `lp_order` (12).

**When to use:** Speaker-independent wake words. More noise-robust than MFCC for many conditions due to perceptual modeling.

**When NOT to use:** When ONNX export size matters (slightly larger computation graph than MFCC). When you need high feature dimensions.

**Hardware fit:** MCU to RPi. Similar compute to MFCC.

---

### PNCCExtractor -- `feats.py:1100`

Power-Normalized Cepstral Coefficients (Kim & Stern 2016). Uses gammatone-like filterbank, medium-time power processing for asymmetric noise suppression, and 1/15-power nonlinearity instead of log.

**Parameters:** `sr` (16000), `n_pncc` (13), `n_fft` (512), `hop_length` (160), `n_filters` (40), `power` (1/15).

**When to use:** Noisy deployment environments (cars, kitchens, factories). Significantly outperforms MFCC in noise.

**When NOT to use:** Clean environments where MFCC is sufficient. The medium-time processing adds latency.

**Hardware fit:** RPi Zero and above. Slightly more compute than MFCC due to temporal smoothing.

---

### CQTExtractor -- `feats.py:1217`

Constant-Q Transform. Logarithmic frequency spacing gives better low-frequency resolution than FFT.

**Parameters:** `sr` (16000), `n_bins` (12), `n_octaves` (7), `f_min` (32.7), `hop_length` (160).

**When to use:** Tonal wake words, music-like patterns, or when low-frequency discrimination matters.

**When NOT to use:** Speech-only wake words where mel-scale is sufficient. Higher compute than mel filterbank.

**Hardware fit:** RPi and above. The FFT-based approximation uses a large `n_fft` internally.

---

### SincNetExtractor -- `feats.py:400`

Learnable sinc bandpass filters (Ravanelli & Bengio 2018). Frequency boundaries are learned during training. Hamming window, log1p energy.

**Parameters:** `sr` (16000), `n_filters` (80), `kernel_size` (251, must be odd), `stride` (160), `min_freq` (50.0), `min_band` (50.0).

**When to use:** When you want the extractor to adapt to your specific wake word's frequency profile. Good for domain-specific deployment.

**When NOT to use:** When you need a frozen, pre-trained extractor. Requires joint training with the head.

**Hardware fit:** RPi, small SBC. Learnable parameters are minimal (2 params per filter = 160 total for 80 filters).

---

### GammatoneExtractor -- `feats.py:599`

Gammatone filterbank on ERB frequency scale. Models the human auditory system's cochlear response.

**Parameters:** `sr` (16000), `n_filters` (64), `f_min` (50.0), `f_max` (sr/2), `frame_len` (400), `hop_length` (160), `order` (4).

**When to use:** Noisy environments. Better noise robustness than mel filterbank due to auditory modeling. Alternative to PNCC when you want simpler processing.

**When NOT to use:** When mel-scale features are sufficient. Larger conv kernels than MFCC.

**Hardware fit:** RPi and above. Conv1d with `frame_len`-sized kernels.

---

### LEAFExtractor -- `feats.py:815`

Learnable Audio Frontend (Zeghidour et al., ICLR 2021). Replaces fixed filterbanks with: Gabor convolution (learnable center freq + bandwidth), squared modulus, Gaussian low-pass pooling, PCEN normalization. All components are differentiable.

**Parameters:** `sr` (16000), `n_filters` (40), `window_len` (400), `hop_length` (160), `min_freq` (60.0), `max_freq` (sr/2), `pcen_alpha` (0.96), `pcen_delta` (2.0), `pcen_r` (0.5), `pcen_s` (0.025).

**When to use:** When you want maximum adaptability -- the entire frontend learns from data. Best with sufficient training data (1000+ samples).

**When NOT to use:** Small datasets (overfitting risk). When you need a frozen extractor for ONNX-only deployment without retraining.

**Hardware fit:** RPi and above. More parameters than SincNet but still compact.

---

### HubertExtractor -- `feats.py:696`

HuBERT self-supervised speech encoder. Requires `transformers`. Default model: `voidful/hubert-tiny-v2`.

**Parameters:** `model_name` (HuggingFace ID), `sample_rate` (16000).

**When to use:** Maximum accuracy. Best feature quality for wake word detection when compute is available.

**When NOT to use:** Edge devices, MCUs, anything without GPU or ample RAM. Use for training then distill to `CnnLstmExtractor`.

**Hardware fit:** Server/workstation for training. Export to ONNX + quantize for RPi4 inference.

---

### Wav2Vec2Extractor -- `feats.py:730`

Wav2Vec2 encoder. Default model: `patrickvonplaten/tiny-wav2vec2-no-tokenizer`.

**Parameters:** `model_name`, `sample_rate` (16000).

**When to use:** Similar to HuBERT. Alternative pretrained encoder when HuBERT is unavailable.

**When NOT to use:** Same constraints as HuBERT.

**Hardware fit:** Server/workstation. Export to ONNX for deployment.

---

### Wav2Vec2BertExtractor -- `feats.py:764`

`facebook/w2v-bert-2.0` encoder. 1024-dim features. The largest pretrained extractor available.

**Parameters:** `model_name`, `sample_rate` (16000).

**When to use:** When you need the highest quality features and have GPU resources. Best combined with knowledge distillation.

**When NOT to use:** Any constrained device. This is a multi-billion parameter model.

**Hardware fit:** Server with GPU only.

---

### OnnxFeatureExtractor -- `feats.py:105`

Loads any ONNX feature extractor model. Used for inference after exporting a PyTorch extractor, or for loading pre-exported models (e.g., Whisper via `from_whisper` classmethod at `feats.py:139`).

**Parameters:** `model_path` (ONNX file), `sample_rate` (16000).

**When to use:** Production inference. Load exported HuBERT/Wav2Vec2/SincNet/MFCC models.

**When NOT to use:** Training (use the native PyTorch extractors instead).

**Hardware fit:** Any device supported by ONNX Runtime (CPU, CUDA, TensorRT, etc.).

---

### DeltaExtractor -- `feats.py:509`

Appends first-order (delta) and second-order (delta-delta) temporal derivatives. Output dim = `3 * base_dim`. Uses +/-N frame regression formula (`feats.py:536`).

**Parameters:** `base_extractor`, `delta_width` (2).

**When to use:** Always consider with MFCC or PLP. Adds temporal context at zero model parameter cost. Standard practice in ASR.

**When NOT to use:** With transformer extractors (they already capture temporal context internally).

---

### VoiceActivityExtractor -- `feats.py:1321`

Appends 4 VAD features: log RMS energy, zero-crossing rate, spectral flatness, combined VAD probability (`feats.py:1349`).

**When to use:** When your dataset has lots of silence/noise and you want the classifier to explicitly focus on speech regions.

**When NOT to use:** Clean datasets with pre-trimmed audio.

---

### PitchExtractor -- `feats.py:1443`

Appends 3 pitch features via autocorrelation: normalized F0, voicing probability, F0 delta (`feats.py:1476`).

**When to use:** Multi-word wake phrases where prosody matters. Speaker-dependent wake words.

**When NOT to use:** Single short wake words where pitch is less discriminative.

---

### MultiResolutionExtractor -- `feats.py:1577`

Concatenates outputs from two extractors at different time resolutions. Coarse features are interpolated to match fine extractor's time axis.

**When to use:** When you want both fine temporal detail (short hop) and broad temporal patterns (long hop).

**When NOT to use:** When feature dimension budget is tight.

---

### SNRAwareExtractor -- `feats.py:1630`

Appends 2 SNR features: per-frame SNR estimate and noise floor estimate (`feats.py:1659`). Uses percentile-based noise floor tracking.

**When to use:** Noisy deployment. Lets the classifier weight clean frames more heavily.

**When NOT to use:** Clean environments.

---

### MarkovTransitionExtractor -- `feats.py:1757`

Classical sequential modeling using a Markov chain on top of quantized features. Captures the probability of acoustic patterns following one another.

**Parameters:** `n_codes` (64), `order` (2).

**When to use:** To add classical temporal context to a "bag of features" baseline. Extremely efficient for secondary validation. Best on top of MFCC or Filterbank.

**When NOT to use:** High-order models ($O>3$) with large vocabularies can result in massive transition matrices.

**Hardware fit:** Microcontrollers (ESP32) and above. Inference is a single integer lookup.

---

### HMMStateExtractor -- `feats.py:1953`

Hidden Markov Model state posterior extraction. Models the wake word as a sequence of latent acoustic units (like phonemes).

**Parameters:** `n_states` (8), `n_codes` (32).

**When to use:** Hybrid architectures. Provides "Sound Probability" channels that help deep learning heads focus on temporal progression. Excellent for reducing false positives.

**When NOT to use:** When you have a massive neural extractor (HuBERT) which already models temporal context deeply.

**Hardware fit:** Microcontrollers and above. Vectorized forward algorithm is very fast.

---

### SileroVadWrapper -- `feats.py:1321`

Neural VAD enrichment stream using the pre-trained `snakers4/silero-vad`. Appends a robust speech probability channel to every frame.

**Parameters:** `onnx_path` (optional).

**When to use:** Real-world noisy environments where heuristic energy-based VAD fails. Essential for production-grade wake-word models.

**When NOT to use:** When you cannot afford the extra compute of a secondary neural network (Silero VAD) at runtime.

**Hardware fit:** RPi Zero and above. 

---

## Recommended Combinations

| Scenario | Extractor Stack | Output Dim |
|----------|----------------|------------|
| MCU, minimal | `MfccExtractor(n_mfcc=13)` + `DeltaExtractor` | 39 |
| RPi Zero | `MfccExtractor(n_mfcc=40)` | 40 |
| RPi, noisy env | `PNCCExtractor` + `SNRAwareExtractor` | 15 |
| RPi, learnable | `SincNetExtractor` | 80 |
| RPi4, best accuracy | `OnnxFeatureExtractor` (exported HuBERT) | 768 |
| Server, training | `HubertExtractor` or `Wav2Vec2BertExtractor` | 768-1024 |
| Server, distillation | `CnnLstmExtractor` (student) mimicking HuBERT | 256 |
