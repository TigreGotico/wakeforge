# Feature Extractors

All extractors inherit from `BaseExtractor` — `ww_trainer/feats.py:77` and output `[B, T, F]` tensors.

---

## Standalone Extractors

| Extractor | Class | Line | Output Dim | ONNX Export | Learnable |
|-----------|-------|------|------------|-------------|-----------|
| MFCC | `MfccExtractor` | `feats.py:366` | `n_mfcc` (40) | Yes | No |
| FilterBank | `FilterbankExtractor` | `feats.py:460` | `n_mels` (80) | Yes | No |
| PLP | `PLPExtractor` | `feats.py:976` | `n_plp` (13) | Yes | No |
| PNCC | `PNCCExtractor` | `feats.py:1097` | `n_pncc` (13) | Yes | No |
| CQT | `CQTExtractor` | `feats.py:1217` | `n_bins * n_octaves` (84) | Yes | No |
| SincNet | `SincNetExtractor` | `feats.py:532` | `n_filters` (80) | Yes | Yes (freq bounds) |
| Gammatone | `GammatoneExtractor` | `feats.py:712` | `n_filters` (64) | Yes | No |
| LEAF | `LEAFExtractor` | `feats.py:809` | `n_filters` (40) | Yes | Yes (full frontend) |
| ONNX | `OnnxFeatureExtractor` | `feats.py:169` | Model-dependent | N/A (already ONNX) | No |
| WakeHuBERT family (pretrained) | `load_pretrained_featurizer` | `pretrained.py:138` | 128 or 256 | N/A (already ONNX) | No |
| Text (ONNX) | `OnnxTextExtractor` | `feats.py:2249` | Model-dependent | N/A (already ONNX) | No |

> **Removed in 0.2.0:** `HubertExtractor`, `Wav2Vec2Extractor`, and `Wav2Vec2BertExtractor` training wrappers are no longer in `feats.py`. Export those models once with the standalone scripts (`scripts/export_hubert.py`, `scripts/export_wav2vec2.py`, `scripts/export_w2vbert.py`), then load with `OnnxFeatureExtractor`.

## Wrapper Extractors

Wrappers decorate a base extractor, adding extra feature channels.

| Wrapper | Class | Line | Extra Dims | What It Adds |
|---------|-------|------|------------|--------------|
| Delta | `DeltaExtractor` | `feats.py:641` | `2 * base_dim` | First and second temporal derivatives |
| SileroVAD | `SileroVadWrapper` | `feats.py:1324` | 1 | Silero VAD speech probability per frame |
| VoiceActivity | `VoiceActivityExtractor` | `feats.py:1459` | 4 | Log RMS energy, ZCR, spectral flatness, VAD probability |
| Pitch | `PitchExtractor` | `feats.py:1583` | 3 | Normalized F0, voicing probability, F0 delta |
| MultiResolution | `MultiResolutionExtractor` | `feats.py:1720` | `coarse_dim` | Concatenates fine + coarse extractor outputs |
| SNRAware | `SNRAwareExtractor` | `feats.py:1776` | 2 | Per-frame SNR estimate, noise floor estimate |
| Markov | `MarkovTransitionExtractor` | `feats.py:1892` | `n_codes` | Transition probabilities from trained Markov chain |
| HMM | `HMMStateExtractor` | `feats.py:2091` | `n_states` | Latent state posterior probabilities (Forward algorithm) |

Note: Unlike other wrappers, Markov and HMM extractors **do** support direct ONNX export of the full pipeline (Base + Wrapper).

---

## Detailed Descriptions

### MfccExtractor — `feats.py:366`

Pure-PyTorch MFCC (Mel-Frequency Cepstral Coefficients — Davis & Mermelstein, IEEE TASSP 1980, <https://ieeexplore.ieee.org/document/1163420>). Computes STFT, mel filterbank, log, then DCT.

**Status:** `wakeforge`'s own implementation of a standard, decades-old signal-processing technique; no accuracy claim from a specific paper applies here.

**Parameters:** `sr` (16000), `n_mfcc` (40), `n_mels` (40), `n_fft` (400), `hop_length` (160), `f_min` (0.0), `f_max` (sr/2).

**When to use:** Default choice for constrained devices. Well-understood, compact, ONNX-safe. Best when combined with Delta wrapper for temporal context.

**When NOT to use:** High-accuracy requirements where transformer extractors dominate. Whether PNCC or Gammatone actually beats MFCC in noise for a given deployment is not measured by this project — see those extractors below for what is and is not backed by a citation.

**Hardware fit:** MCU, RPi Zero, any device. Zero learnable parameters in extractor.

**Data appetite:** not measured by this project.

---

### FilterbankExtractor — `feats.py:460`

Log-mel spectrogram (mel scale: Stevens, Volkmann & Newman, JASA 1937) without DCT. Same pipeline as MFCC minus the final DCT step.

**Status:** `wakeforge`'s own implementation of a standard technique; no accuracy claim from a specific paper applies here.

**Parameters:** `sr` (16000), `n_mels` (80), `n_fft` (400), `hop_length` (160), `f_min` (0.0), `f_max` (sr/2).

**When to use:** When using 2D CNN heads (BCResNet, DSCNN) that benefit from the full mel spectrogram. A common input choice for Whisper-style models.

**When NOT to use:** When feature dimension must be very small (use MFCC with fewer coefficients instead).

**Hardware fit:** MCU to laptop. Slightly larger features than MFCC but no extra compute.

**Data appetite:** not measured by this project.

---

### PLPExtractor — `feats.py:976`

Perceptual Linear Prediction (Hermansky, *Perceptual Linear Predictive (PLP) Analysis of Speech*, JASA 1990). Models human auditory perception via Bark-scale warping, equal-loudness pre-emphasis, and cube-root compression.

**Status:** `wakeforge`'s own implementation of the published technique; the paper's own accuracy results are for speech recognition, not keyword spotting, and are not reproduced here.

**Parameters:** `sr` (16000), `n_plp` (13), `n_fft` (512), `hop_length` (160), `n_bark` (21), `lp_order` (12).

**When to use:** Speaker-independent wake words — the perceptual modeling is motivated by, but not verified by this project to be, more resistant to noise than MFCC.

**When NOT to use:** When ONNX export size matters (slightly larger computation graph than MFCC). When you need high feature dimensions.

**Hardware fit:** MCU to RPi. Similar compute to MFCC.

**Data appetite:** not measured by this project.

---

### PNCCExtractor — `feats.py:1097`

Power-Normalized Cepstral Coefficients (Kim & Stern, IEEE/ACM TASLP 2016, <https://ieeexplore.ieee.org/document/7439789>). Uses gammatone-like filterbank, medium-time power processing for asymmetric noise suppression, and 1/15-power nonlinearity instead of log.

**Status:** `wakeforge`'s own implementation of the published technique. The paper reports a noise-resistance advantage over MFCC on the speech-recognition tasks it evaluates; `wakeforge` has not reproduced that comparison for wake-word detection.

**Parameters:** `sr` (16000), `n_pncc` (13), `n_fft` (512), `hop_length` (160), `n_filters` (40), `power` (1/15).

**When to use:** Noisy deployment environments (cars, kitchens, factories), per the source paper's motivation — not verified against MFCC by this project.

**When NOT to use:** Clean environments where MFCC is sufficient. The medium-time processing adds latency.

**Hardware fit:** RPi Zero and above. Slightly more compute than MFCC due to temporal smoothing.

**Data appetite:** not measured by this project.

---

### CQTExtractor — `feats.py:1217`

Constant-Q Transform (Brown, *Calculation of a Constant Q Spectral Transform*, JASA 1991). Logarithmic frequency spacing gives, by construction, higher low-frequency resolution than a linear-frequency FFT.

**Status:** `wakeforge`'s own implementation of a standard signal-processing transform; no wake-word accuracy claim from a paper applies here.

**Parameters:** `sr` (16000), `n_bins` (12), `n_octaves` (7), `f_min` (32.7), `hop_length` (160).

**When to use:** Tonal wake words, music-like patterns, or when low-frequency discrimination matters.

**When NOT to use:** Speech-only wake words where mel-scale is sufficient. Higher compute than mel filterbank.

**Hardware fit:** RPi and above. The FFT-based approximation uses a large `n_fft` internally.

**Data appetite:** not measured by this project.

---

### SincNetExtractor — `feats.py:532`

Learnable sinc bandpass filters (Ravanelli & Bengio, *Speaker Recognition from Raw Waveform with SincNet*, SLT 2018). ArXiv <https://arxiv.org/abs/1808.00158>. Frequency boundaries are learned during training. Hamming window, log1p energy.

**Status:** `wakeforge`'s own implementation of the published architecture; the paper's own results are for speaker recognition, not keyword spotting, and are not reproduced here.

**Parameters:** `sr` (16000), `n_filters` (80), `kernel_size` (251, must be odd), `stride` (160), `min_freq` (50.0), `min_band` (50.0).

**When to use:** When you want the extractor to adapt to your specific wake word's frequency profile. For domain-specific deployment.

**When NOT to use:** When you need a frozen, pre-trained extractor. Requires joint training with the head.

**Hardware fit:** RPi, small SBC. Learnable parameters are minimal (2 params per filter = 160 total for 80 filters).

**Data appetite:** not measured by this project. Being learnable, it needs training data to converge on useful filter boundaries, which is a real difference from the fixed extractors above — how much is not something this project has quantified.

---

### GammatoneExtractor — `feats.py:712`

Gammatone filterbank on ERB frequency scale (Patterson & Holdsworth, 1996 — an auditory model, not a keyword-spotting paper). Models the human auditory system's cochlear response.

**Status:** `wakeforge`'s own implementation of the auditory model; no wake-word accuracy claim from a paper applies here.

**Parameters:** `sr` (16000), `n_filters` (64), `f_min` (50.0), `f_max` (sr/2), `frame_len` (400), `hop_length` (160), `order` (4).

**When to use:** Noisy environments, per the auditory-modeling motivation — whether it actually beats mel filterbank on noise resistance is not measured by this project. An alternative to PNCC when you want simpler processing.

**When NOT to use:** When mel-scale features are sufficient. Larger conv kernels than MFCC.

**Hardware fit:** RPi and above. Conv1d with `frame_len`-sized kernels.

**Data appetite:** not measured by this project.

---

### LEAFExtractor — `feats.py:809`

Learnable Audio Frontend (Zeghidour et al., *LEAF: A Learnable Frontend for Audio Classification*, ICLR 2021). ArXiv <https://arxiv.org/abs/2101.08596>. Replaces fixed filterbanks with: Gabor convolution (learnable center freq + bandwidth), squared modulus, Gaussian low-pass pooling, PCEN normalization. All components are differentiable.

**Status:** `wakeforge`'s own implementation of the published architecture; the paper's own results are on general audio classification benchmarks, not keyword spotting, and are not reproduced here.

**Parameters:** `sr` (16000), `n_filters` (40), `window_len` (400), `hop_length` (160), `min_freq` (60.0), `max_freq` (sr/2), `pcen_alpha` (0.96), `pcen_delta` (2.0), `pcen_r` (0.5), `pcen_s` (0.025).

**When to use:** When you want maximum adaptability — the entire frontend learns from data.

**When NOT to use:** Small datasets (overfitting risk — the whole frontend is learned, so it needs more data than a fixed extractor to avoid it). When you need a frozen extractor for ONNX-only deployment without retraining.

**Hardware fit:** RPi and above. More parameters than SincNet but still compact.

**Data appetite:** not measured by this project. The "1000+ samples" figure previously on this page was not attributed to any measurement or source, so it has been removed rather than repeated; treat LEAF as the most data-hungry extractor in this file, since every component including the filterbank itself is learned, but get an actual number from a real run before relying on one.

---

### OnnxFeatureExtractor — `feats.py:169`

Loads any ONNX feature extractor model. Used for inference after exporting a PyTorch extractor, or for loading pre-exported models (e.g., Whisper via `from_whisper` classmethod at `feats.py:254`).

**Parameters:** `model_path` (ONNX file), `sample_rate` (16000).

**When to use:** Production inference. Load exported HuBERT/Wav2Vec2/SincNet/MFCC models.

**When NOT to use:** Training (use the native PyTorch extractors instead).

**Hardware fit:** Any device supported by ONNX Runtime (CPU, CUDA, TensorRT, etc.).

---

### Pretrained speech extractors (WakeHuBERT family) — `pretrained.py:133`

The pretrained extractors are small speech feature extractors distilled from
large self-supervised speech models with masked distillation: the student
hears noisy, partly masked audio and learns to predict the teacher's layers
on clean speech. Each turns 16 kHz audio into 128- or 256-dimensional
features at 50 frames per second. Each is published as ONNX, in float32 and
int8, in its own Hugging Face model repository, all listed in the
[ONNX feature extractors collection](https://huggingface.co/collections/TigreGotico/onnx-feature-extractors-690a3d2bced386cc3c338c77).

**Use `wakehubert` (an alias of `wakehubert-tiny`) as the default.** It is the
extractor behind the measured result below. The others are published for
comparison: other architectures, wider outputs, and other teachers.

The name is the repository name; add `-int8` for the int8 file (for example
`wakehubert-mel-tcn-wide-int8`). Selecting a name downloads the ONNX and its
`config.json` into the shared Hugging Face cache and wraps it in
`OnnxFeatureExtractor`. `feature_dim`, hop, frame rate, `streaming` and the
licence come from the config, so `--feature-dim` is not needed.

| Name | Params | Streaming | Licence | What it is |
|------|--------|-----------|---------|------------|
| `wakehubert-tiny` (`wakehubert`) | 0.64M | yes (2.5 s) | Apache-2.0 | HuBERT-base student, causal TCN, 128-d. **Default.** |
| `wakehubert-mel-tcn-deep` | 0.69M | yes (5 s) | Apache-2.0 | HuBERT-base student, 16-block causal TCN, 128-d |
| `wakehubert-mel-tcn-wide` | 2.01M | yes (4 s) | Apache-2.0 | HuBERT-base student, wide causal TCN, 256-d |
| `wakehubert-mel-gru` | 1.16M | no, recurrent | Apache-2.0 | HuBERT-base student, causal GRU, 128-d |
| `wakehubert-mel-bigru` | 0.96M | no, offline | Apache-2.0 | HuBERT-base student, bidirectional GRU, 128-d |
| `wakehubert-mel-attn` | 1.43M | float32 only (4 s) | Apache-2.0 | HuBERT-base student, windowed causal attention, 128-d |
| `wakehubert-mixconv` | 0.64M | yes (5 s) | Apache-2.0 | HuBERT-base student, causal mixed-kernel TCN, 128-d |
| `wakewav-mel-tcn` | 0.64M | yes (2.5 s) | CC-BY-SA-3.0 | WavLM-base+ student, causal TCN, 128-d |
| `wakewav-mel-tcn-wide` | 2.01M | yes (4 s) | CC-BY-SA-3.0 | WavLM-base+ student, wide causal TCN, 256-d |
| `wakewav-mel-gru` | 1.16M | no, recurrent | CC-BY-SA-3.0 | WavLM-base+ student, causal GRU, 128-d |
| `wakewav-mel-bigru` | 0.96M | no, offline | CC-BY-SA-3.0 | WavLM-base+ student, bidirectional GRU, 128-d |
| `wakexeus-mel-tcn` | 0.64M | yes (2.5 s) | CC-BY-NC-SA-4.0 | XEUS student, causal TCN, 128-d. Non-commercial. |
| `wakexeus-mel-tcn-wide` | 2.01M | yes (4 s) | CC-BY-NC-SA-4.0 | XEUS student, wide causal TCN, 256-d. Non-commercial. |
| `wakexeus-mel-gru` | 1.16M | no, recurrent | CC-BY-NC-SA-4.0 | XEUS student, causal GRU, 128-d. Non-commercial. |
| `wakexeus-mel-bigru` | 0.96M | no, offline | CC-BY-NC-SA-4.0 | XEUS student, bidirectional GRU, 128-d. Non-commercial. |

Selecting a `wakexeus-*` extractor logs a warning, because its CC-BY-NC-SA-4.0
licence forbids commercial use. Every extractor can be used for training and
for whole-window scoring with `OnnxWakeWordInferencer`.

#### Use a pretrained WakeHuBERT featurizer

Every training path accepts a registry name wherever it accepts a built-in
featurizer. A pretrained featurizer is frozen, so its features are computed
once per clip and reused (see *Feature store* below).

**Train.** Pick the featurizer by name and any head with `--arch`, or take a
tier. `wakehubert` is a GRU head on `wakehubert-tiny`; `wakehubert-bigru` is a
bidirectional GRU on the same featurizer, which scores whole windows.

```bash
ww_trainer-train --wake-word alexa --metadata train.csv --test-metadata test.csv \
    --tier wakehubert --epochs 30 --save-best
ww_trainer-train --wake-word alexa --metadata train.csv --test-metadata test.csv \
    --featurizer-type wakehubert-mel-tcn-wide --arch bigru --hidden-dim 128 --linear-dim 64
# --featurizer wakehubert works too; pin a revision with --featurizer-revision <commit>
```

```python
from ww_trainer.trainer import WakeWordTrainer
trainer = WakeWordTrainer(arch="gru", featurizer=None, featurizer_type="wakehubert",
                          losses_cfg=[{"name": "bce", "weight": 1.0}])
trainer.train(output_dir="out", train_data=train, test_data=test, epochs=30)
```

`--hidden-dim`, `--linear-dim` and `--gru-n-layers` size the GRU heads
(`gru`, `bigru`); `--linear-dim` is also the embedding size that the metric
losses (ArcFace, center, ProxyNCA, HALO) are sized to. Every head in
`HEAD_REGISTRY` trains on 128- and 256-dimensional features at 50 frames per
second; `phonmatch` also needs the keyword's phoneme IDs in each sample.

**Quickstart.** `--featurizer` replaces the tier's extractor and keeps its head:

```bash
ww_trainer-quickstart --wake-word "hey jarvis" --output-dir ./hey_jarvis --tier wakehubert
ww_trainer-quickstart --wake-word "hey jarvis" --output-dir ./hey_jarvis --tier small \
    --featurizer wakehubert-mel-tcn-wide
```

**Grid, random and genetic search.** Pass a name as `featurizer_type`, or put
names in the search space. The full search space (`full=True`) carries the
float32 graph of every repository licensed for commercial use as values of its
`featurizer_type` gene, and `bigru` in its `arch` gene. Candidates on the same
featurizer share one feature store in a process; `feature_cache_dir` shares it
with demes in other processes.

```python
from ww_trainer.sweep import run_genetic_search
run_genetic_search("data.csv", population_size=12, generations=5, full=True,
                   feature_cache_dir=".feature_cache")
run_genetic_search("data.csv", search_space={
    "featurizer_type": ["wakehubert", "wakehubert-mel-tcn-wide", "wakexeus-mel-tcn"],
    "arch": ["gru", "bigru"], "hidden_dim": [64, 128], "lr": [5e-4, 1e-3]})
```

Multi-stage training (`--training-stages`), `infinite_training_loop`, hard
negative mining, calibration (`--calibrate`), few-shot references and
checkpoint averaging take the featurizer from the model and need nothing else.

**Feature store.** Features of a pretrained (ONNX) featurizer are kept per clip in a
`FeatureStore` (`ww_trainer/feature_store.py`) and reused by every epoch,
mining pass, evaluation and calibration, and by every head trained on the same
featurizer file in the process. From the CLI they also persist under
`--feature-cache-dir`, so later runs and other processes reuse them;
`--no-feature-cache` featurizes every batch from audio. Waveform augmentation
still applies: with `--feature-cache-variants K`, an augmented draw picks one
of `K` augmented variants of the clip and computes it only the first time it
is drawn. With `K = 0` (the default) augmented draws are featurized on the fly
and not kept, which keeps augmentation continuous. Mixup blends the stored
features, and SpecAugment masks them: by default up to 100 ms of frames and a
tenth of the dimensions (5 frames and 13 dimensions for 128-d features at
50 fps). All live stores in a process share one memory budget
(`FeatureStore.max_bytes`, 4 GB). A store that needs room clears the stores
used least recently, so a store left from an earlier run in the same process
never blocks a later one; features that still do not fit are recomputed or
read from `--feature-cache-dir`. `FeatureStore.close()` frees one store and
`FeatureStore.close_all()` every one.

Built-in extractors without parameters (MFCC, filterbank, ...) use a store
only with `--feature-cache` (`cache_features=True` in Python), because they
are cheap and their training keeps augmenting audio: waveform Mixup and RPPL's
waveform-augmented view. An extractor with parameters (SincNet, LEAF) never
uses a store, even while frozen, and no store is used with
`--unfreeze-at-epoch`, since the extractor then trains again.

**Padding.** Batches are padded with zeros to their longest clip, and each
batch carries the real length of every row. The GRU heads, `ffn` and `ocsvm`
pool over real frames only (a bidirectional GRU also starts its backward pass
at each clip's last real frame), so a clip scores the same in any batch.

**Export and inference.** Exported heads (`best_*.onnx`, `final_model.onnx`,
and the streaming heads) record the featurizer in their ONNX metadata as
`pretrained_featurizer` (the registry name) and `featurizer_revision` (the
commit it was downloaded at). The featurizer ONNX is not copied beside them:
inference rebuilds it from the registry. When the model streams (a
unidirectional single-layer GRU head on a streaming featurizer), training also
writes `<stem>_streaming.onnx`, with a window set to the median positive clip
length in frames, and `<wake_word>_meta.json` records the featurizer, its
revision, the tier and the streaming heads.

```python
from ww_trainer.inference import OnnxWakeWordInferencer, OnnxStreamingWakeWord
model = OnnxWakeWordInferencer(None, "out/best_f1.onnx")     # featurizer from metadata
score = model.infer(audio)
sw = OnnxStreamingWakeWord.from_head("out/best_f1_streaming.onnx")
prob = sw.push(chunk)
```

```bash
ww_trainer-infer --model out/best_f1.onnx --audio clip.wav
```

**Streaming.** The streaming column gives the past audio that determines one
output frame; no frame depends on later audio. `OnnxStreamingWakeWord`
re-featurizes that much past audio with each chunk, so every streamed frame
equals its offline value. `OnnxStreamingWakeWord.from_extractor` and
`from_head` take the hop and context from the featurizer. They refuse
featurizers that cannot be streamed this way: the bidirectional models, the
recurrent models (whose frames depend on all earlier audio), and
`wakehubert-mel-attn-int8`, whose frames also depend on later audio; those
score whole windows with `OnnxWakeWordInferencer`. Feed chunks whose length is
a multiple of 320 samples. For `wakehubert`, one 0.1 s push costs a single
featurizer call over 2.6 s of audio, a few milliseconds on one CPU thread.

**Measured result.** With `wakehubert`, a GRU head trained on 900 synthetic "alexa" clips (TTS
with voice conversion and noise, babble, reverberation, speed and gain
augmentation) detected 95% (95% interval 92–97) of the real speakers in the
Picovoice wake-word benchmark at a threshold set for 0.5 false activations per
hour; 0.31 false activations per hour were measured on 6.5 h of held-out
streams. In babble, recall was 94 / 85 / 56% at 10 / 5 / 0 dB. These figures
come from a single run on one English wake word.

**When to use:** A speech-aware front end on CPU for wake words trained from
synthetic data, when MFCC-based models miss too many real speakers.

**Hardware fit:** Any device that runs ONNX Runtime; the int8 file suits
low-power CPUs.

---

### OnnxTextExtractor — `feats.py:2249`

Loads a pre-exported text encoder ONNX model and provides text-based feature embeddings as a `BaseExtractor`. Enables optional audio+text conditioning — the text features are concatenated to audio features, allowing the model to be conditioned on the target wake-word string at training time.

**Parameters:** `model_path` (ONNX file), `tokenizer_name` (HuggingFace tokenizer ID), `sample_rate` (16000).

**When to use:** Experimental audio+text conditioning; phonetic similarity losses; conditioning a shared model on multiple wake words via text prompts.

**When NOT to use:** Standard single-wake-word deployment (adds complexity without benefit). Not for MCUs.

**Hardware fit:** Any device supported by ONNX Runtime.

---

### DeltaExtractor — `feats.py:641`

Appends first-order (delta) and second-order (delta-delta) temporal derivatives. Output dim = `3 * base_dim`. Uses +/-N frame regression formula. A standard technique in ASR feature pipelines generally, not specific to one cited paper.

**Parameters:** `base_extractor`, `delta_width` (2).

**When to use:** Worth considering with MFCC or PLP. Adds temporal context at zero model parameter cost.

**When NOT to use:** With transformer extractors (they already capture temporal context internally).

---

### VoiceActivityExtractor — `feats.py:1459`

Appends 4 VAD features: log RMS energy, zero-crossing rate, spectral flatness, combined VAD probability.

**When to use:** When your dataset has lots of silence/noise and you want the classifier to explicitly focus on speech regions.

**When NOT to use:** Clean datasets with pre-trimmed audio.

---

### PitchExtractor — `feats.py:1583`

Appends 3 pitch features via autocorrelation: normalized F0, voicing probability, F0 delta.

**When to use:** Multi-word wake phrases where prosody matters. Speaker-dependent wake words.

**When NOT to use:** Single short wake words where pitch is less discriminative.

---

### MultiResolutionExtractor — `feats.py:1720`

Concatenates outputs from two extractors at different time resolutions. Coarse features are interpolated to match fine extractor's time axis.

**When to use:** When you want both fine temporal detail (short hop) and broad temporal patterns (long hop).

**When NOT to use:** When feature dimension budget is tight.

---

### SNRAwareExtractor — `feats.py:1776`

Appends 2 SNR features: per-frame SNR estimate and noise floor estimate (`feats.py:1659`). Uses percentile-based noise floor tracking.

**When to use:** Noisy deployment. Lets the classifier weight clean frames more heavily.

**When NOT to use:** Clean environments.

---

### MarkovTransitionExtractor — `feats.py:1892`

Classical sequential modeling using a Markov chain on top of quantized features. Captures the probability of acoustic patterns following one another. A generic technique, not from one specific cited paper.

**Parameters:** `n_codes` (64), `order` (2).

**When to use:** To add classical temporal context to a "bag of features" baseline. Cheap at inference (a lookup table) for a secondary check alongside a neural head. Best on top of MFCC or Filterbank.

**When NOT to use:** High-order models ($O>3$) with large vocabularies can result in massive transition matrices.

**Hardware fit:** Microcontrollers (ESP32) and above. Inference is a single integer lookup.

---

### HMMStateExtractor — `feats.py:2091`

Hidden Markov Model state posterior extraction. Models the wake word as a sequence of latent acoustic units (like phonemes).

**Parameters:** `n_states` (8), `n_codes` (32).

**When to use:** Hybrid architectures. Provides "Sound Probability" channels that help deep learning heads focus on temporal progression. Whether it reduces false positives in practice is not measured by this project.

**When NOT to use:** When you have a massive neural extractor (HuBERT) which already models temporal context deeply.

**Hardware fit:** Microcontrollers and above. Vectorized forward algorithm is very fast.

---

### SileroVadWrapper — `feats.py:1324`

Neural VAD enrichment stream using the pre-trained `snakers4/silero-vad` (GitHub <https://github.com/snakers4/silero-vad>). Appends a speech probability channel to every frame.

**Status:** the VAD model itself is Silero's pre-trained weights, used as-is, not retrained or verified by this project. `wakeforge`'s own code is the wrapper that feeds it into the feature pipeline.

**Parameters:** `onnx_path` (optional).

**When to use:** Real-world noisy environments where heuristic energy-based VAD is likely to fail.

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
| RPi4, largest-capacity extractor | `OnnxFeatureExtractor` (exported HuBERT) | 768 |
| Server, training | `OnnxFeatureExtractor` with exported HuBERT/Wav2Vec2-BERT | 768-1024 |
| Server, distillation | Export teacher ONNX → `OnnxFeatureExtractor` → train student | varies |

---


`wakeforge` includes two classical statistical models adapted for wake-word feature extraction: `MarkovTransitionExtractor` and `HMMStateExtractor`.

These extractors bridge the gap between "bag of features" (like MFCC) and "temporal deep learning" (like GRU/Transformers) by capturing sequential patterns without adding many trainable neural parameters. Neither is measured by this project against the deep-learning alternative it bridges toward — the design goal is stated, not a benchmarked result.

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

### Compute note:
HMM and Markov extractors run a lookup table or a small forward-algorithm pass, which is cheap next to a transformer's attention computation, so a microcontroller can afford them for a secondary check alongside a neural head. This is an architectural fact about the operations involved, not a measured latency or accuracy comparison.
