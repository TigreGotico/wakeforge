# What a Good Wake Word Looks Like

wakeforge is a research framework. It implements the wake-word literature so
a result can be reproduced or an architecture compared against another, not
because every included architecture is a good default. A training run that
finishes with a falling loss and a passing quickstart command proves the
pipeline works. It proves nothing about whether the resulting detector is
safe to ship. This page states the numbers that separate the two, names
where each number comes from, and says plainly which of them this project
has measured itself.

## The two numbers that matter

A wake-word detector is judged on two axes that trade against each other:

| Metric | Meaning |
|---|---|
| **FA/hour** (False Activations per Hour) | How often the detector fires on audio that was not the wake word. Measured with a sliding window over long negative audio. |
| **FRR** (False Reject Rate) — equivalently **TPR** (Recall) `= 1 − FRR` | The fraction of real wake-word utterances the detector misses. Measured with one pass per positive clip. |

These definitions match `ww-benchmarks`
(<https://github.com/OpenVoiceOS/ww-benchmarks>), the harness used
to measure them — see its `eval_fn.py` (FNR/recall) and `eval_fp.py`
(FA/hour). A threshold controls the trade-off between the two; moving it
does not make a bad detector good, it only chooses which failure mode you
see more of.

## What counts as good

The wake-word literature converges on **1 false activation per hour** as the
conventional operating point for judging a detector, and measures recall
(or false-reject rate) at that point rather than in isolation — for example
Hou et al. (2020), *Mining Effective Negative Training Samples for Keyword
Spotting* (<https://ieeexplore.ieee.org/document/9053009>), reports relative
false-reject reductions "at a false alarm rate of once per hour." That is a
convention from the published literature, not a number wakeforge measured.

### The measured baseline: two rejected models

wakeforge has one set of measured numbers. They come from the first two
models it trained, and both models failed validation. They are the state
before the dataset fix, not a quality claim. The rows are in draft pull
request [OpenVoiceOS/ww-benchmarks#9](https://github.com/OpenVoiceOS/ww-benchmarks/pull/9).

The models are `hey_mycroft` and `wake_up`, `tier=small`, 50 epochs. The
threshold is 0.5. The run was on a laptop CPU on 2026-09-10.

| Model | Audio | Result | Bar |
|---|---|---|---|
| `hey_mycroft` | synthetic held-out set (2226 clips) | FNR 0.9 % | 5 % |
| `hey_mycroft` | held-out edge-tts voices (32 clips) | FNR 12.5 % | 10 % |
| `hey_mycroft` | near-miss phrases (80 clips) | 70 % accepted | 5 % |
| `hey_mycroft` | LibriSpeech test-clean (0.87 h) | 5120 FA/hour | 1 |
| `hey_mycroft` | `wake_word_noise` (1.02 h) | 2632 FA/hour | 1 |
| `hey_mycroft` | `FMA_3secs` music (0.42 h) | 3588 FA/hour | 1 |
| `wake_up` | LibriSpeech test-clean | 3867 FA/hour | 1 |
| `wake_up` | `wake_word_noise` | 2397 FA/hour | 1 |
| `wake_up` | `FMA_3secs` music | 3365 FA/hour | 1 |

The commands, from the `ww-benchmarks` root, with a model directory that
holds `featurizer.onnx` and `model.onnx`:

```bash
python eval_fn.py --model <model dir> --model-type wakeforge --wake-word-id hey_mycroft --threshold 0.5 --audio-folder evalsets/synth/hey_mycroft --dataset-id synthetic-wakewords-hey_mycroft
python eval_fp.py --model <model dir> --model-type wakeforge --wake-word-id hey_mycroft --threshold 0.5 --audio-folder evalsets/librispeech_test_clean --dataset-id librispeech-test-clean
```

The other rows change only `--audio-folder`, `--dataset-id` and
`--wake-word-id`. `--model-type wakeforge` is the adapter in draft pull
request [OpenVoiceOS/ww-benchmarks#8](https://github.com/OpenVoiceOS/ww-benchmarks/pull/8).

Read the table this way. The model finds the wake word in clean synthetic
audio. It also fires thousands of times per hour on any speech, noise or
music, and it accepts most phrases that only sound similar. The cause was
the training data. The negatives were VAD-trimmed fragments shorter than one
second, and there were no near-miss negatives. A model that learns from these
negatives scores any sustained sound as the wake word. A low training loss
and a high F1 score did not show this problem. Only `ww-benchmarks` showed it.

No model that wakeforge trained has cleared these bars yet. The CLI prints an
F1 score when a run completes. That F1 is a training-set metric. It does not
measure FA/hour or real-world recall.

## Data budget

How much speech and how many negatives a tier needs before it is worth
measuring is likewise not something this project has validated end to end.
The per-architecture pages in `reference/classifiers.md`,
`reference/extractors.md`, and `reference/losses.md` state, where known,
whether a data-appetite figure is the source paper's own reported number or
this project's own measurement — look there for the tier or method you are
evaluating rather than assuming a single rule applies to all of them.

## Measuring your own model

1. Train a model (`docs/getting_started/quickstart.md` or
   `docs/guides/training.md`).
2. Export it to ONNX (already automatic from the quickstart).
3. Run it through `ww-benchmarks`' `eval_fn.py` against a positive dataset
   and `eval_fp.py` against a long negative recording, per that repo's
   README.
4. Compare the resulting FA/hour and recall against the operating point you
   need for your product — not against the training F1 score.

## Which quickstart presets are demonstrations, not deployable models

`docs/getting_started/quickstart.md` already says this at the point the
choice is made: the quickstart's default preset (`tier=small`, synthetic
TTS positives, no real far-field recordings) is for "a working detector
*today*," while "< 0.5 FA/hour or > 95 % recall" and any real far-field
audio need the full pipeline — a distinction the quickstart page itself
draws, not a new rule introduced here. The same applies to the `micro` tier
example in `docs/learning_path.md` Stage 1 (10 epochs, no augmentation
data): it produces a model that proves the pipeline runs, not one anybody
should point a real device at. Treat every quickstart/Stage-1 output as a
smoke test until it has been through the measurement in the section above.
