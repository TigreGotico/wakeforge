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

**wakeforge has not published a measured FA/hour or false-reject figure for
any model it trains.** `ww-benchmarks` is the tool that would produce one,
and its `results/` directory is empty today — no run has been committed.
Until a run is committed there, treat every quickstart output's F1 score
(the number the CLI prints on completion) as a training-set metric, not
evidence of FA/hour or real-world recall: F1 does not measure either.

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
