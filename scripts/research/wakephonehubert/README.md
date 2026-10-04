# WakePhoneHuBERT

WakePhoneHuBERT is the WakeHuBERT-tiny int8 trunk with two small heads on its taps: a voice-activity head (VAD, one sigmoid
output) and an IPA phoneme head (392 CTC posteriors over the wav2vec2-xlsr-53-espeak-cv-ft symbol table, 100 ms delayed).
Its `features` output is 521 numbers per 20 ms frame: 128 WakeHuBERT values, 1 VAD probability, 392 IPA posteriors. The
published WakeHuBERT graph is carried unchanged inside the ONNX file, so its 128 values are bit-identical to
`wakehubert_int8.onnx`.

This directory holds everything that produces the published model and its numbers, in the order a run needs it.

| Folder | Step |
|---|---|
| `pool/` | data pool and teacher labels |
| `train/` | VAD warm start and the four training rounds |
| `export/` | ONNX build, validation, validation report |
| `bench/` | downstream benchmark (phoneme recognition, ESC-50, language ID, speaker ID, VAD, size and speed) |
| `zeroshot/` | zero-shot and probe evaluation of the exported ONNX |
| `aligner/` | forced alignment and a trained aligner head |
| `wakeword/` | plugin-faithful window scoring of wake-word heads |

`wph_model.py` defines the trunk wrapper (`Trunk`, nine taps plus the 128-wide output and the stacked log-mel) and the head
(`TapHead`). Tests: `pytest test/test_wph_model.py test/test_wph_export.py test/test_wph_windowing.py`.

## Conventions

Every script is run from one working directory (the examples use `work/`, relative to where the command is typed) and takes
each input and output location as an option; the defaults below are the paths the shell scripts use. The shell scripts take the
working directory as their first argument and read the other locations from environment variables named in their first lines.
`PY` names the Python interpreter.

All GPU steps ran on one NVIDIA A10. CPU-only steps are marked. Durations are read from the logs of the recorded run; where the
logs hold none the step says so. Throughputs are the last value a stage logged, and a duration given as `n / rate` is computed
from it.

Software of the recorded run: torch 2.14 (CUDA 13), transformers 5.18, onnxruntime 1.30, phonemizer 3.4 with espeak-ng,
webrtcvad-wheels 2.0.14. Install the rest of the imports with

```bash
uv pip install --prerelease=allow -e .[dev]
uv pip install transformers phonemizer pyarrow pandas scipy safetensors fsspec webrtcvad-wheels
```

and install `espeak-ng` and `ffmpeg` with the system package manager. The EfficientAT checkout
(<https://github.com/fschmid56/EfficientAT>) goes in `work/EfficientAT`; it supplies the `mn10_as` AudioSet classifier that
scores voice evidence in the pool (step 2) and is a reference row in the benchmark.

Inputs fetched from outside the Hub datasets used below:

| Input | Where it goes |
|---|---|
| LibriSpeech train-clean-100, dev-clean, test-clean (OpenSLR 12) | `work/data/LibriSpeech/` |
| AudioSet `ontology.json` (<https://github.com/audioset/ontology>) | `work/data/audioset/ontology.json` |
| `fma_metadata.zip` (<https://os.unil.cloud.switch.ch/fma/fma_metadata.zip>) | `work/data/fma_metadata.zip` |
| a directory of room impulse response wavs | `work/data/rir/` |
| a directory of noise wavs (the recorded run read 2,500 files) | `work/data/noise/` |
| `wakehubert_int8.onnx` from `TigreGotico/wakehubert-tiny` at revision `3caa605725a932440442692f644b7217f54a29c3` | `work/wakehubert_int8.onnx` |

```bash
hf download TigreGotico/wakehubert-tiny wakehubert_int8.onnx --revision 3caa605725a932440442692f644b7217f54a29c3 --local-dir work
```

`wph_model.load_trunk` downloads the same revision's `model.safetensors` and `student.py` on first use.

## 1. Data

```bash
bash pool/run_pool.sh work
```

CPU only; the logs hold no timings. It writes, under `work/pool/`, for each source a `<source>_audio.raw` (int16, one
fixed-length 16 kHz segment per row), `<source>_meta.jsonl` (id, labels or word and language, licence) and `<source>.done`
(row count):

| Source | Built by | Rows of the recorded run |
|---|---|---|
| `fma` | `build_pool.py fma`: `fma_full` tracks whose licence allows commercial use and derivatives, read by HTTP range, 10 s segments | 83,915 |
| `audioset` | `build_pool.py audioset --shards 40`: 40 unbalanced-train parquet shards of `agkphysics/AudioSet`, 10 s clips with labels | 79,994 |
| `mswc` | `build_pool.py mswc`: `MLCommons/ml_spoken_words`, every language, 1 s words, at most 60,000 clips per language for `ar as br ca cnh cs cv cy` and 20,000 for the rest | 793,163 |
| `mswcx` | `build_pool.py mswc --name mswcx --shard-offset 0.5`: a second sample of the same corpus from other archives, at most 25,000 clips per language | 788,046 |

If a decoding worker dies near the end of the `fma` build, take the complete segments with `pool_teachers.py --limit N`
(the recorded run used 83,915). `mswc_fetch.py` pre-fetches a language-balanced slice of the MSWC repository with gaps between
files, for `build_pool.py --local-dir`; `trim_mswc_pool.py` cuts a part-built `mswc` back to its last complete language before
a resume at another per-language cap.

## 2. Teacher labels

`train/run_rounds.sh` runs the teacher stages for each source just before the round that first uses it. The commands are:

```bash
python pool/snapshot_pool.py work/pool mswc mswc_a --type mswc --count 260964
python pool/pool_teachers.py work/pool --source fma
python pool/pool_teachers.py work/pool --source mswc_a --type mswc
```

`snapshot_pool.py` names a fixed range of a source without copying audio; `mswc_a` is the first 260,964 clips (languages `ar`
to `el`), `mswc_b` the rest (532,199), `audioset_a` the first 48,000 clips and `audioset_b` the remaining 31,994. Each source
is labelled by `pool_teachers.py` in stages, resumable by chunk of 4,096 rows:

| Stage | Teacher | Output |
|---|---|---|
| `silero` | Silero VAD v5 on CPU | speech probability per 20 ms frame |
| `text` (`mswc` types) | espeak-ng through the wav2vec2-xlsr-53-espeak-cv-ft tokenizer | teacher-tokenizer ids of each word, the CTC targets |
| `ipa` | wav2vec2-xlsr-53-espeak-cv-ft on the GPU | top-8 posteriors per frame |
| `voice` | EfficientAT `mn10_as`, causal 5 s windows every 320 ms | speech-subtree score, which marks AudioSet and FMA frames that are certainly not speech |
| `forced` | derived | forced non-speech mask per frame |

`mswcx` takes only the `text` stage (`--stages text`), 6 s of CPU. Per-source cost in the recorded run, `silero` on CPU and
`ipa` and `voice` on the A10 (the CPU stage runs beside the GPU stages):

| Source | Rows | silero | ipa | voice |
|---|---|---|---|---|
| `fma` | 83,915 | 2.5 min | 34 min | 40 min |
| `mswc_a` | 260,964 | 0.7 min | 9.5 min | 13.9 min |
| `mswc_b` | 532,199 | 1.3 min | 9.3 min | 12.5 min |
| `audioset_a` | 48,000 | 1.2 min | 12.8 min | 11.4 min |
| `audioset_b` | 31,994 | 0.8 min | 8.5 min | 7.6 min |

## 3. Training: four rounds

```bash
bash train/run_rounds.sh work
```

The shell script holds the whole recipe: the VAD warm start, then four rounds of `train_pool.py`, each resuming the previous
one, with the snapshots and teacher stages of step 2 between them. Environment variables `LIBRI`, `NOISE`, `RIR`, `ONTOLOGY`,
`EFFICIENTAT`, `EXCLUDE_TRACKS` (a file of FMA track ids kept out of training because an evaluation set uses them) and
`AUDIOSET_EVAL_DIR` (evaluation files named by AudioSet video id, to log overlap) override the default locations.

`train_pool.py` trains the two heads on one frozen WakeHuBERT-tiny forward pass per batch, each head with its own optimiser:

| Head | Shape | Loss |
|---|---|---|
| `vad` | `TapHead(1, 32)` | binary cross-entropy to Silero's speech probability |
| `ipa` | `TapHead(392, 192, side=128)` | CTC to the word's teacher-tokenizer ids, plus KL to the teacher's top-8 posteriors at a 5-frame delay |

Batches cycle through three kinds: 1 s MSWC words with noise (VAD and IPA targets), 3 s crops of AudioSet and FMA segments
(non-speech frames pinned to VAD 0 and an empty CTC target when the whole crop is non-speech), and 4 s synthetic mixtures of
LibriSpeech or MSWC speech with reverberant babble, music or noise (VAD only). A split of every source is held out and every
epoch reports VAD agreement with Silero, phone error rate on held-out words, and the rate of phones emitted on non-speech.

| Step | Command | Sources | Epochs x steps | Duration |
|---|---|---|---|---|
| VAD warm start | `train_vad.py vad-data` (4,000 clips of 8 s), then `vad` | LibriSpeech speech over noise, Silero targets | 30 epochs | not logged |
| Round 1 | `train_pool.py ... --init-vad` | `fma`, `mswc_a` | 6 x 3,000 | 24.0 min |
| Round 2 | `--resume round1` | `fma`, `audioset_a`, `mswc_a`, `mswc_b` | 6 x 3,000 | 14.2 min |
| Round 3 | `--resume round2` | adds `audioset_b` | 4 x 3,000 | 9.6 min |
| Round 4 | `--resume round3 --final-ipa-sources mswcx` | IPA head only, `mswcx` words with CTC targets | 2 x 5,000 | 8.0 min |

A resumed round restarts the schedule at half the original peak learning rate. Round 4 trains only the IPA head with CTC (no
teacher KL), 85% from `mswcx` clips with the noise mixing of the 1 s word batches and 15% from one-second crops of non-speech pool segments with
empty targets. Training throughput was 14 to 26 steps per second. The outputs are `vad_head.pt` and `ipa_head.pt` in each run
directory, `checkpoint.pt` for resuming, and `report.json`. The model is built from `work/runs/round4-final-ipa`.

## 4. Export

```bash
bash export/run_export.sh work work/wakehubert_int8.onnx work/runs/round4-final-ipa
```

CPU only. `prep_clips.py` draws 20, 64 and 30 clips (validation, calibration and acceptance; LibriSpeech, AudioSet and FMA)
from the held-out tails of the pool; `build_wakephonehubert.py` attaches the heads to the taps of the published int8 graph and
writes `wakephonehubert_int8.onnx` (all outputs: `hubert`, `layers`, `logmel`, `mfcc`, `vad`, `ipa`, `features`),
`wakephonehubert_features_int8.onnx` (only `features`), `config.json` and `vocab.json` under `work/export/final/`. It tries
`int8-static`, `int8-weights` and `float16` for each head, smallest first, and keeps the first that stays within 0.01 of the
float32 head's probabilities and 99.5% of its decisions on the acceptance clips; the combined build is checked again. The
recorded build kept both heads in `float16` and produced a 3,482,985-byte `wakephonehubert_int8.onnx`. Build: about 30 s,
from the build files' times.

## 5. Validation

`run_export.sh` continues with `validate_wakephonehubert.py` (about 20 s) and `write_validation_md.py`. The validator checks, on
random inputs and on the real clips:

- `hubert` is bit-identical to the published int8 featurizer at default threads and at one thread, in the full file and in the
  features-only file;
- each head in the ONNX against the PyTorch head, fed the ONNX's own taps and fed the float trunk's taps (maximum probability
  difference and decision agreement);
- the ONNX checker with full checking on every file;
- latency of one 1.5 s window on one thread, as the wake-word plugin runs it every 80 ms.

It writes `validation_report.json`; `write_validation_md.py` turns it and `config.json` into `VALIDATION.md` beside the model.

## 6. Benchmarks

`bench/` follows the SUPERB protocol: a frozen featurizer, a learned softmax-weighted sum of its entries, a small head. The
featurizers are log-mel, WakeHuBERT-tiny, WakePhoneHuBERT (taps, output, VAD and IPA entries), HuBERT-base, wav2vec2-xlsr-53-espeak
and EfficientAT `mn10_as`. Every Python script except `report.py` takes `--bench-dir` (default `work/bench`), `--heads` (the round-4 run
directory) and `--efficientat`; `report.py` takes `--bench-dir`.

```bash
bash bench/fetch_small.sh work/bench/data
bash bench/fetch_lid.sh work/bench/data
bash bench/run_baselines.sh work/bench work/EfficientAT work/data/LibriSpeech
bash bench/run_wph.sh work/bench work/runs/round4-final-ipa
```

| Script | Task | Notes |
|---|---|---|
| `pr.py prep`, `pr.py train <feat>`, `pr.py direct` | phoneme recognition on LibriSpeech test-clean | linear CTC probe, eSpeak targets; `direct` scores the teacher's own CTC head |
| `pr_full.py prep39`, `pr_full.py run <feat> <espeak\|superb39>` | the same with up to 100 epochs, with eSpeak or ARPAbet-39 targets | needs `librispeech-lexicon.txt` (OpenSLR 11) in `data/` |
| `extract_pooled.py <task> --feats ...`, `train_pooled.py <task> <feat>` | ESC-50, VoxLingua107 (ten languages), VoxCeleb1 speaker identification | per-clip mean-pooled entries, then the weighted sum and a linear layer; speaker identification streams VoxCeleb1 from the Hub and keeps only pooled features |
| `vad_eval.py` | LibriParty voice activity | Silero, WebRTC modes 0 to 3, the WakePhoneHuBERT VAD head |
| `speed.py` | parameters, file size and one-thread CPU latency per 80 ms of audio | |
| `report.py` | collects every `results.json` into `RESULTS.md` | |

Logged durations of the recorded run: phoneme-recognition probes 1,082 s (log-mel), 1,239 s (WakeHuBERT-tiny) and 578 s
(WakePhoneHuBERT) for at most 5 epochs; the 100-epoch log-mel probe 4,016 s; the ESC-50 and language-ID heads 3 to 14 s each on
cached features. The logs hold no time for feature extraction.

## 7. Zero-shot evaluation

`zeroshot/` runs on the exported ONNX (`--export-dir work/export/final`) and on the benchmark's data and features
(`--bench-dir`); its own files go under `--zeroshot-dir` (default `work/zeroshot`).

```bash
python zeroshot/phones_zs.py
python zeroshot/sound_zs.py extract && python zeroshot/sound_zs.py score
python zeroshot/language_zs.py extract && python zeroshot/language_zs.py score
python zeroshot/vad_zs.py extract && python zeroshot/vad_zs.py zeroshot && python zeroshot/vad_zs.py train
python zeroshot/kws_traces.py --negative test_clean='work/data/LibriSpeech/test-clean/**/*.flac' \
    --negative nonspeech='work/data/nonspeech/**/*.wav' --every nonspeech=10 \
    --positive alexa='work/data/keywords/alexa/*.wav' --positive jarvis='work/data/keywords/jarvis/*.wav'
python zeroshot/kws_eval.py
python zeroshot/phones_probe_onnx.py
python zeroshot/build_assets.py --keyword-clip work/data/keywords/jarvis/example.wav
```

| Script | What it measures |
|---|---|
| `phones_zs.py` | greedy CTC decoding of `ipa` on LibriSpeech dev-clean and test-clean, with and without the delay padding, and with a blank penalty chosen on dev |
| `sound_zs.py` | ESC-50 with nearest-centroid and a linear probe on the 521 features, against EfficientAT's own AudioSet classifier mapped to the ESC-50 classes |
| `language_zs.py` | VoxLingua107 with nearest-centroid on the features and on phone bigrams, and a linear probe |
| `vad_zs.py` | LibriParty: the raw VAD output, the same smoothed with a 5-frame median, and small heads trained on the frozen features; `--libriparty-train` points at the train sessions |
| `kws_traces.py`, `kws_eval.py` | zero-shot keyword spotting: a CTC keyword scorer over `ipa` on 1.5 s windows, recall at one and at half a false activation per hour |
| `phones_probe_onnx.py` | the trained phoneme probe of the benchmark, re-run through the ONNX with numpy |
| `build_assets.py` | the small heads, centroids and example clips that a model card's code snippets read |

`kws_traces.py` takes each negative stream as `NAME=GLOB` and each positive as `WORD=GLOB` (single utterances, padded with 1 s
of silence); `sets.json` carries the names to `kws_eval.py`. `phones_probe_onnx.py` reads the `head-seed*.pt` files that
`pr.py train wakephonehubert` writes (`--probe-dir`). Logged durations: `phones_zs.py` 65 s; the keyword traces about 11 min on
CPU for 31 h of negative audio; LibriParty feature extraction 112 s; the four VAD heads 22 to 28 s each.

## 8. Aligner

Two forced aligners on the exported ONNX, scored against Montreal Forced Aligner alignments of LibriSpeech test-clean
(`gilkeyio/librispeech-alignments`): CTC forced alignment over `ipa`, and a trained bidirectional head over the taps and
features with a constrained Viterbi. Every script takes `--work-dir` (default `work/aligner`), `--export-dir` and `--librispeech`.

```bash
python aligner/fetch_alignments.py
curl -L -o work/aligner/data/cmudict.dict https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict
python aligner/run_ctc.py
python aligner/train_head.py
python aligner/run_head.py
python aligner/score.py
python aligner/speed.py
```

| Script | Output | Duration |
|---|---|---|
| `fetch_alignments.py` | train-clean-100 alignments, test-clean parquet (paced, one shard every 20 s) | about 9 min |
| `run_ctc.py` | `results/{uniform,wph_ctc,teacher_ctc}.jsonl` | 94 s for the ONNX on CPU, 297 s for the teacher on the A10 |
| `train_head.py` | `head/aligner_head.pt`, `split.json`, `train_log.json`; 30 h of train-clean-100, 14 epochs, features computed from the ONNX on CPU | 12.9 min on the A10 |
| `run_head.py` | exports the head to ONNX and safetensors, aligns test-clean with reference and CMUdict pronunciations | not logged |
| `score.py` | `results/metrics.json`: word and phone boundary errors | seconds |
| `speed.py` | CPU seconds per second of audio on one thread | not logged |

## 9. Wake-word scoring

`wakeword/` scores wake-word heads the way the runtime plugin does: the stream is pushed in 1,280-sample (80 ms) blocks into a
24,000-sample buffer that starts as zeros, and the whole buffer is featurized on its own after every block. Heads are trained and
exported with `scripts/research/head_bench.py` on the features file (`wakephonehubert_features_int8.onnx`).

```bash
python wakeword/window_feats.py work/export/final/wakephonehubert_features_int8.onnx work/windows \
    speech='<speech wavs>' nonspeech='<non-speech wavs>' <word>='<positive wavs>'
python wakeword/score_windows.py work/windows head.onnx <name> <word> --out work/traces
python wakeword/score_stream.py work/export/final/wakephonehubert_features_int8.onnx work/traces \
    --heads <name>=head.onnx:<word> --sets speech='<speech wavs>' nonspeech='<non-speech wavs>' <word>='<positive wavs>'
```

`window_feats.py` stores the windows' features once and `score_windows.py` scores a head over them; `score_stream.py` streams the
same windows straight into several heads and keeps only the traces, which is the cheaper route for many heads. Positive sets
are padded with 1 s of silence on each side; every 10th non-speech file is used. Each trace file holds the speech, non-speech
and held-out positive scores per block, ready for a false-activations-per-hour and recall computation. The test
`test/test_wph_windowing.py` checks that both routes cut the same windows.
