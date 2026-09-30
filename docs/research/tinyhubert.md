# TinyHuBERT: Distilling Self-Supervised Speech Models into a Streaming Wake-Word Feature Extractor

**Status:** the recipe is implemented in `scripts/research/tinyhubert.py`; the results in Section 6 are
single runs of a wake-word evaluation built on top of it, with their known limits stated alongside them.
**Requires:** a GPU for the teacher forward pass, `transformers`, local audio on disk.

---

## 1. Motivation

Self-supervised speech models such as HuBERT (Hsu et al., 2021) learn phonetic frame representations
from unlabelled audio. A wake-word head trained on such features separates a keyword from a similar-sounding
phrase far more easily than one trained on MFCC, but a model like HuBERT Base has about 95M parameters and
bidirectional attention: too large for an always-on device and unable to stream.

TinyHuBERT keeps the representation and drops the architecture. A small causal network is trained to predict
a frozen teacher's hidden states frame by frame; the trained network is then used as a frozen
`OnnxFeatureExtractor` under an ordinary wake-word head. The recipe is not tied to one teacher: any frozen
speech transformer that exposes intermediate hidden states can play the role, and the exported family is
named after it -- HuBERT gives WakeHuBERT, WavLM gives WakeWav, XEUS gives WakeXeus.

The design decisions, each of which the implementation encodes:

1. **Match the teacher's frame rate.** HuBERT and its relatives emit one frame every 20 ms: their
   convolutional front end has a total stride of 320 samples. Every student, whatever its own internal
   front end, is built to produce one output frame per 320 input samples, so every student frame has a
   teacher frame to learn from and no target is an interpolation of two.
2. **Strictly causal, with a fixed delay.** A teacher frame depends on the whole clip, past and future; a
   streaming student cannot see the future, so part of that target is unpredictable to it. The student
   stays causal and instead predicts teacher frame t at its own frame t + 1 + lag: by then it has heard the
   teacher frame's own window and `lag` further 20 ms frames of audio. Its features therefore describe the
   speech with a delay of (1 + lag) x 20 ms; the default lag of 4 gives 100 ms, which a wake-word decision
   tolerates.
3. **Distil middle layers, not only the last.** The last layer of a pretrained (not fine-tuned) teacher is
   shaped by its own pretraining objective; phonetic content, and keyword-spotting accuracy, peak in the
   middle layers. Following DistilHuBERT (Chang et al., 2022), a shared backbone feeds one linear head per
   distilled layer (default 4, 8 and 12).
4. **Standardise the targets.** A few teacher dimensions carry magnitudes far above the rest, and an
   unnormalised distance spends its gradient on them. Targets are standardised per layer and per dimension
   with statistics measured on the teacher before training.
5. **DistilHuBERT's loss.** Per layer: L1 distance plus `-log sigmoid(cosine similarity)`. The L1 term fixes
   the values, the cosine term the direction.
6. **Contrastive negatives from other clips only (optional).** An InfoNCE term that treats every other frame
   of the batch as a negative asks the student to separate neighbouring frames of one clip, which the teacher
   itself barely separates. `cross_clip_nce` excludes same-clip frames from the negatives. It is off by
   default (`--nce-weight 0`).
7. **Noise, reverberation and other talkers on the student's side only.** Wake words are spoken over
   television, music, kitchens and other people. The teacher always hears the clean clip; the student hears
   a distorted view of the same clip, so it learns to report the speech rather than the mixture (Section 3).
8. **Masking the student's own input (optional).** `--mask-prob` zeroes random spans of the student's
   normalised input, as wav2vec 2.0 masks its input for pretraining (Baevski et al., 2020); here it acts as a
   denoising regulariser on top of distillation rather than as the pretraining objective itself.

## 2. Students and teacher

**Teacher:** any Hugging Face speech transformer with `output_hidden_states=True`, frozen -- by default
`facebook/hubert-base-ls960`. `--teacher-spec` names further teachers (`model:layer,layer,...`) so several
teachers can be distilled from in one run, each run once per batch; a student picks its teacher with
`teacher=<part of the model name>`. The model family in the exported metadata follows the teacher: `hubert`
gives WakeHuBERT, `wavlm` WakeWav, `xeus` WakeXeus.

**`--student` picks the backbone:**

- **`wave-gru`** -- the original recipe: five causal residual convolutions directly on the waveform
  (channels 64, 128, 256, 256, 256; kernels 10, 8, 8, 4, 4; strides 5, 4, 4, 2, 2, a total stride of 320),
  each padded on the left only so frame t depends on nothing past sample 320(t + 1), followed by a
  unidirectional GRU and a LayerNorm. The exported backbone has 2.33M parameters at the defaults
  (`cnn_dim=64`, `gru_hidden=256`); the per-layer training heads add 0.6M and are not exported.
- **`mel-tcn`** -- a fixed causal log-mel front end (Hann-windowed DFT written as a convolution, so it
  exports as plain Conv/MatMul/Log and can be kept in float when the rest of the network is quantised),
  a strided stem down to the teacher's frame rate, then dilated causal depthwise-separable convolution
  blocks (a TCN) with a residual connection per block. `--mixconv` splits each depthwise convolution into
  groups of different kernel sizes at the same dilation (MixConv, Tan & Le 2019, as used in microWakeWord),
  mixing short and long receptive fields in one layer. `--target-blocks` predicts a chosen teacher layer
  (other than the last) from an interior block's output rather than from the exported features, so the
  backbone does not have to carry every teacher layer's information all the way to its final projection.
- **`mel-gru`** -- the `mel-tcn` front end and blocks followed by a causal unidirectional GRU.
- **`mel-bigru`** -- the same front end and blocks followed by a bidirectional GRU. It does not stream: every
  frame depends on the whole input, so a consumer recomputes the features over each window it classifies
  rather than carrying state forward.
- **`mel-attn`** -- the same front end and blocks followed by causal self-attention limited to the last
  `--attn-window` frames, so it streams with a fixed cache of that many frames.

Every backbone converges on the same interface: `features [batch, frames, feature_dim]`, exported, and one
linear head per distilled teacher layer, used only in training and discarded at export.

## 3. Training

**Data:** any directory of `.wav`, `.flac` or `.ogg` files, searched recursively, cropped at random to
`--crop-seconds` (default 2 s); `--build-shards` pre-cuts a directory into fixed-length int16 crop shards so
training reads memory instead of decoding audio on every step. The training set combines LibriSpeech (960 h),
Multilingual LibriSpeech (Pratap et al., 2020; the seven non-English languages) and a language-balanced
sample of Multilingual Spoken Words (Mazumder et al., 2021; 41 languages), pre-cut into 600k two-second
crops. Validation uses fixed, seeded crops of a held-out directory, so validation losses are comparable
across steps.

**Robustness on the student's side:** `--noise-dir` mixes each clip with noise (AudioSet- and MUSAN-style,
Snyder et al. 2015) at a random SNR (0-20 dB by default); `--rir-dir` convolves room impulse responses into
the student's speech; `--p-babble` replaces a share of the noise draws with one to three other talkers,
which `--babble-snr-min` keeps from ever being louder than the foreground voice (a 5 dB floor, independent of
`--snr-min`). `--nonspeech-dir` adds non-speech items (music, ambience, kitchens, traffic); a quarter of
training items by default are non-speech and are heard identically by teacher and student, so the objective
never asks the student to invent a clean/noisy distinction on audio that carries no speech.

**Objective:** the mean over distilled layers of `L1 - log sigmoid(cos)` on standardised targets, plus
`--nce-weight` times the cross-clip InfoNCE on the last distilled layer, plus `--mask-prob` span masking of
the student's own input (`mel-tcn` family only) during training. `--word-weight` adds a supervised word
objective beside distillation: a training-only linear classifier on the frame-averaged features predicts the
(language, word) class of a labelled clip from `--build-word-shards` (Multilingual Spoken Words), heard
through the same augmentation as speech and never by the teacher, in the spirit of openWakeWord's classifier
embedding; the classifier is discarded at export. Word accuracy is measured on every 50th clip of each class,
heard clean and never trained on, so every class is represented in it.

**Several students, one data stream:** `--student-spec name:key=value,...` trains several backbones --
any mix of architecture and timing options -- on one shared stream of batches, with each teacher run once
per batch rather than once per student; every student still writes its own `metrics.jsonl`, checkpoints and
export under `--out-dir/<name>`.

**Optimiser:** AdamW (lr 1e-3, weight decay 0.01), linear warm-up over 2,000 steps then cosine decay to zero
over `--steps`; gradients clipped at 5.

**Outputs:** `metrics.jsonl` (training loss, gradient norm, learning rate and throughput every 100 steps;
validation loss, L1 and cosine per layer every `--eval-every` steps; word-objective loss and accuracy where
enabled), `last.pt` and `best.pt` (resumable with `--resume`), `tinyhubert.onnx` from the best checkpoint,
and `tinyhubert.json` with the teacher, layers, lag, latency, frame rate, feature dimension and parameter
count.

## 4. Export

`tinyhubert.onnx` takes `waveform` `[batch, samples]` and returns `features` `[batch, frames, feature_dim]`,
the input layout `OnnxFeatureExtractor` feeds. The export is checked against the PyTorch model on a clip
whose length differs from the export's dummy input and refused if they disagree.

The exported graph processes a whole window; recurrent or attention state is not carried across calls as a
graph input or output, so a device re-runs the extractor over a sliding window rather than feeding it 20 ms
at a time. A stateful export is future work and changes no result below.

**`--int8`** writes a statically quantised `tinyhubert_int8.onnx` alongside the float one (QDQ, per-channel
weights, calibrated on held-out clips). The log-mel front end and its input batch norm stay in float -- a
fixed spectrogram gains nothing from int8 and its log compresses a dynamic range int8 cannot hold -- and so
do softmax, layer norm and the attention mask; onnxruntime leaves GRU ops in float regardless. The exported
metadata records the mean and worst-case per-frame cosine between the float and int8 features on clips never
used to calibrate.

## 5. Evaluation protocol

Wake-word heads are trained on synthetic clips only -- 900 clips of one wake word, generated with
text-to-speech voices passed through voice conversion -- with augmentation applied at head-training time
(noise, babble, reverberation, speed and gain changes); the checkpoint is chosen on a calibration set
disjoint from the evaluation set. Scoring uses the Picovoice wake-word benchmark's real recordings of the
same word (315 of which are decodable): recall at the threshold giving 0.5 false activations per hour over
6.5 hours of held-out streams built from LibriSpeech `test-clean`, three-talker babble from LibriSpeech
`test-other`, and held-out non-speech audio, plus recall with babble mixed into the positive stream at 10, 5
and 0 dB.

**Known limits of this protocol, stated rather than hidden:**

- The false-activation threshold is chosen on the same streams it is then reported on.
- 315 positive examples is a small evaluation set: differences under about five points between two rows are
  within its noise, and two head architectures trained on the same extractor have differed by up to ten
  points on it.
- Every number below is a single run, not an average over seeds.

## 6. Findings

Recall at 0.5 false activations per hour, quiet and with babble mixed in at 10/5/0 dB (single runs; see
Section 5 for the evaluation set's known limits).

| Extractor | Params | Quiet | Babble 10 dB | Babble 5 dB | Babble 0 dB |
|---|---|---|---|---|---|
| Frozen HuBERT Base, layer 8 (reference, not streaming) | 94M | 98.7 | 96 | 92 | 68 |
| mel-TCN, masked distillation (mask_prob 0.065), 30k steps | 0.64M | 96.5 | 95 | 89 | 63 |
| mel-TCN, same run without masking | 0.64M | 93 | 88 | 80 | 53 |
| Bidirectional GRU (not streaming), 30k steps | 0.96M | 94 | 93 | 91 | 68 |
| Feature union of four HuBERT students | -- | 96 | 94 | 90 | 57 |
| Wide mel-TCN, 100k steps | 2.0M | 95 | 92 | 85 | 58 |

The masked mel-TCN student is the best streaming result: within one to five points of the 94M-parameter
frozen teacher in babble, at about 1/150 of its size. Masking the student's own input during distillation is
the single largest lever found among the students tried -- the identical architecture and step count without
it loses three to five points at every noise level, most sharply at 0 dB babble.

**Negative and null results:**

- Extending the delay from 100 ms to 260-520 ms does not improve agreement with the teacher (+0.01 cosine):
  causality is not the ceiling on how well a student can imitate this teacher, and agreement on clean speech
  plateaus near cosine 0.60 / 0.48 / 0.65 across the distilled layers even with ten times the training data.
- A supervised word-classification objective (Multilingual Spoken Words classes) trained beside distillation
  hurt detection: the same wide mel-TCN student went from 95/91/85 to 82/68/50 (quiet/10 dB/5 dB) at a word
  loss weight of 0.3. It is kept as an option rather than removed, since a smaller weight or a different
  schedule is untried.
- WavLM-base+ and XEUS teachers were no better than HuBERT as distillation targets on the same data.
- Windowed self-attention, MixConv and deeper-narrower students all landed level with the plain mel-TCN
  student -- no clear win from any of the three over the simplest TCN shape.
- int8 static quantisation keeps feature cosine to the float model at 0.996-0.997 for the 0.64M mel-TCN
  students and 0.988 for the 2.0M one.
- Teacher agreement is a poor predictor of wake-word quality: the student with the best clean-speech cosine
  to its teacher was not the best detector in the table above.

**Conclusion.** A 0.64M-parameter streaming extractor trained with masked distillation comes within one to
five points of a frozen 94M-parameter teacher in babble, at 1/150 of its parameter count. Open next steps:
evaluate with thresholds set on a calibration set separate from every reported stream and with more
false-activation hours behind each rate; train masked students with more capacity and more steps; and
compare against openWakeWord, microWakeWord and Precise under one shared protocol.

## 7. Reproducing

Pre-cut crop shards once, then train from them instead of decoding audio on every step:

```bash
python scripts/research/tinyhubert.py \
    --audio-dir data/LibriSpeech/train-clean-100 \
    --build-shards runs/shards --crop-seconds 2.0
```

Train the masked mel-TCN student and a non-streaming bidirectional-GRU student side by side, on one shared
data and teacher stream:

```bash
python scripts/research/tinyhubert.py \
    --audio-dir data/LibriSpeech/train-clean-100 --val-dir data/LibriSpeech/dev-clean \
    --shards runs/shards \
    --noise-dir data/noise --rir-dir data/rirs --nonspeech-dir data/nonspeech \
    --p-babble 0.3 --out-dir runs/tinyhubert --steps 60000 \
    --student-spec "mel-tcn-masked:student=mel-tcn,mask_prob=0.065" \
    --student-spec "bigru:student=mel-bigru"

# continue an interrupted run (each --student-spec name resumes from its own last.pt)
python scripts/research/tinyhubert.py ... --resume runs/tinyhubert/last.pt
```

Add int8 export and quantisation agreement to any run with `--int8`. Build word-objective shards and train a
student with the supervised word loss beside distillation:

```bash
python scripts/research/tinyhubert.py \
    --build-word-shards runs/word-shards --word-list mswc.tsv --word-root data/mswc

python scripts/research/tinyhubert.py \
    --audio-dir data/LibriSpeech/train-clean-100 --val-dir data/LibriSpeech/dev-clean \
    --word-shards runs/word-shards --word-weight 0.3 \
    --out-dir runs/tinyhubert-word --steps 60000
```

Load the exported extractor under a wake-word head:

```bash
ww-trainer train --wake-word hey_mycroft \
    --metadata train/metadata.csv --test-metadata test/metadata.csv \
    --featurizer runs/tinyhubert/mel-tcn-masked/tinyhubert.onnx --featurizer-type onnx --feature-dim 128 \
    --arch gru --save-best
```

## 8. What is and is not new

Every ingredient comes from prior work; the contribution is the combination and the experiment.

| Ingredient | Source |
|---|---|
| Self-supervised speech teachers | HuBERT (Hsu et al. 2021), WavLM (Chen et al. 2022), XEUS (Chen et al. 2024) |
| Layer-wise distillation, L1 + log-sigmoid-cosine loss | DistilHuBERT, Chang et al. 2022 |
| InfoNCE | van den Oord et al. 2018 |
| Streaming student of a bidirectional teacher, with a latency budget | standard in streaming ASR distillation |
| Clean teacher, distorted student input | Cross-Distortion Mapping, Huang et al. 2022 |
| Span masking of the student's own input | wav2vec 2.0, Baevski et al. 2020 |
| Mixed-kernel depthwise convolutions | MixConv, Tan & Le 2019 |
| Supervised keyword classifier beside a learned embedding | openWakeWord |

DistilHuBERT's own student is a 24M-parameter Transformer with the teacher's full context; the students here
are one to two orders of magnitude smaller, causal, and pay a fixed delay instead.

## 9. References

- Hsu et al. (2021). *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden
  Units.* [arXiv:2106.07447](https://arxiv.org/abs/2106.07447)
- Chen et al. (2022). *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing.*
  [arXiv:2110.13900](https://arxiv.org/abs/2110.13900)
- Chen et al. (2024). *Towards Robust Speech Representation Learning for Thousands of Languages* (XEUS).
  [arXiv:2407.00837](https://arxiv.org/abs/2407.00837)
- Chang, Yang, Lee (2022). *DistilHuBERT: Speech Representation Learning by Layer-wise Distillation of
  Hidden-unit BERT.* [arXiv:2110.01900](https://arxiv.org/abs/2110.01900)
- Huang et al. (2022). *Improving Generalizability of Distilled Self-supervised Speech Processing Models
  under Distorted Settings.* [arXiv:2210.07978](https://arxiv.org/abs/2210.07978)
- van den Oord et al. (2018). *Representation Learning with Contrastive Predictive Coding.*
  [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
- Baevski et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.*
  [arXiv:2006.11477](https://arxiv.org/abs/2006.11477)
- Tan, Le (2019). *MixConv: Mixed Depthwise Convolutional Kernels.*
  [arXiv:1907.09595](https://arxiv.org/abs/1907.09595)
- Pratap et al. (2020). *MLS: A Large-Scale Multilingual Dataset for Speech Research.*
  [arXiv:2012.03411](https://arxiv.org/abs/2012.03411)
- Snyder, Chen, Povey (2015). *MUSAN: A Music, Speech, and Noise Corpus.*
  [arXiv:1510.08484](https://arxiv.org/abs/1510.08484)
- Mazumder et al. (2021). *Multilingual Spoken Words Corpus.* NeurIPS Datasets and Benchmarks Track.
  [proceedings](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/fe131d7f5a6b38b23cc967316c13dae2-Abstract-round2.html)

See [references.md](references.md) for the full project bibliography.
