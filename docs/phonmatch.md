# PhonMatchNet — Phoneme-Guided Wake Word Detection

PhonMatchNet (INTERSPEECH 2023, [ncsoft/PhonMatchNet](https://github.com/ncsoft/PhonMatchNet))
conditions wake-word detection on the **phoneme sequence** of the target keyword instead of
learning its raw acoustic fingerprint. This improves generalisation across speakers, accents,
and recording conditions — particularly for under-represented speakers.

ww-trainer integrates PhonMatchNet as a native modality with two usage modes.

---

## Training paradigm

PhonMatchNet requires a **multi-keyword dataset**: each sample carries the IPA phoneme
sequence of its keyword as a third column:

```
(audio_path, label, keyword_ipa_ids)
```

The model learns general phoneme-audio alignment rather than memorising a single keyword's
acoustics.  At inference, pass any IPA sequence to score an arbitrary wake word — provided
the model saw phonemically similar keywords during training.

**Phoneme IDs are passed per-batch** during training (not baked in at init time).
The `collate_fn` (`ww_trainer/dataset.py:collate_fn`) pads the per-sample ID lists into
a `[B, max_seq]` int64 tensor, which the training loop threads through to the model.

Negatives should carry the IPA of the keyword they are negative *for* (same keyword in a
single-keyword setup; the relevant keyword in a multi-keyword setup).

---

## Phoneme vocabulary

ww-trainer uses **IPA (International Phonetic Alphabet)** — 159 symbols covering all major
world languages (`ww_trainer/phonmatch.py:IPA_VOCAB`). ARPAbet is English-only; IPA is the
correct choice for any multilingual or non-English wake word.

```python
from ww_trainer.phonmatch import IPA_VOCAB, ipa_to_ids, arpabet_to_ids

# IPA input (espeak-ng / phonemizer / gruut output)
ids = ipa_to_ids(["h", "eɪ", "m", "aɪ", "k", "ɹ", "ʌ", "f", "t"])

# ARPAbet compat bridge (g2p_en / CMUdict output — stress digits stripped)
ids = arpabet_to_ids(["HH", "EY1", "M", "AY1", "K", "R", "AH0", "F", "T"])
```

**Obtaining IPA phonemes** (ww-trainer never does G2P internally):

| Tool | Command | Notes |
|------|---------|-------|
| espeak-ng | `espeak-ng -q --ipa "hey mycroft"` | Best multilingual coverage |
| phonemizer | `phonemize "hey mycroft" --backend espeak --language en-us` | Python API |
| gruut | `python -m gruut en-us "hey mycroft"` | JSON output, easy to parse |
| g2p_en | `g2p_en.G2p()("hey mycroft")` | ARPAbet → use `arpabet_to_ids()` |

---

## Dataset format

```python
from ww_trainer.phonmatch import ipa_to_ids

keyword_ids = ipa_to_ids(["h", "eɪ", "m", "aɪ", "k", "ɹ", "ʌ", "f", "t"])

# 3-tuple format: (path, label, keyword_ipa_ids)
train_data = [
    ("positive/sample1.wav", "1", keyword_ids),
    ("positive/sample2.wav", "1", keyword_ids),
    ("negative/noise1.wav",  "0", keyword_ids),  # IPA of the keyword rejected
    # For multi-keyword training, each keyword has its own IPA:
    # ("kw2/pos1.wav", "1", kw2_ids),
    # ("kw2/neg1.wav", "0", kw2_ids),
]
```

The `collate_fn` handles 2-tuples (no phoneme conditioning) and 3-tuples
(phoneme conditioning) transparently — no code change needed for existing datasets.

---

## Mode A — Cross-attention head (`arch="phonmatch"`)

`PhonMatchHead` (`ww_trainer/phonmatch.py:PhonMatchHead`) implements the full PhonMatchNet
cross-attention architecture. Phoneme IDs are passed **per batch** at `forward()` time.
ONNX export produces two inputs: `input_features` (audio) and `phoneme_ids`.

```python
from ww_trainer.phonmatch import ipa_to_ids
from ww_trainer import WakeWordTrainer

keyword_ids = ipa_to_ids(["h", "eɪ", "m", "aɪ", "k", "ɹ", "ʌ", "f", "t"])

# Dataset: 3-tuple format
train_data = [("pos/s1.wav", "1", keyword_ids), ("neg/n1.wav", "0", keyword_ids)]

trainer = WakeWordTrainer(
    arch="phonmatch",
    featurizer="mfcc",
    featurizer_type="mfcc",
    wake_word="hey mycroft",
    # No keyword_token_ids here — IDs come from the dataset per batch
    hidden_dim=128,
    n_heads=2,
    gru_layers=2,
    dropout=0.1,
)
trainer.train(output_dir="experiments/hey_mycroft/phonmatch_a",
              train_data=train_data, ...)
```

**Export + inference** — IPA token IDs passed at inference time:

```python
trainer.model.feature_extractor.export_to_onnx("best_f1_featurizer.onnx")
trainer.model.classifier.export_to_onnx("best_f1.onnx")

from ww_trainer.inference import OnnxWakeWordInferencer
model = OnnxWakeWordInferencer("best_f1_featurizer.onnx", "best_f1.onnx")
score = model.infer(wav_array, text_token_ids=keyword_ids)
```

---

## Mode B — Text featurizer ONNX (`text_featurizer=`)

`PhonMatchTextEncoder` (`ww_trainer/phonmatch.py:PhonMatchTextEncoder`) is a learnable
phoneme encoder that exports as a standalone ONNX file. Its output (`[B, 1, D]`) is
appended as extra feature channels to the audio features before any standard head.

Per-batch keyword IDs from the dataset flow through the text encoder at training time.
No `precompute()` call is needed during training.

```python
from ww_trainer.phonmatch import PhonMatchTextEncoder, ipa_to_ids
from ww_trainer import WakeWordTrainer

# Step 1 — export text encoder once
enc = PhonMatchTextEncoder(emb_dim=128)
enc.export_to_onnx("phoneme_encoder.onnx")

# Step 2 — build dataset with keyword IDs per sample
keyword_ids = ipa_to_ids(["h", "eɪ", "m", "aɪ", "k", "ɹ", "ʌ", "f", "t"])
train_data = [("pos/s1.wav", "1", keyword_ids), ("neg/n1.wav", "0", keyword_ids)]

# Step 3 — train; per-batch IDs flow automatically from dataset → collate_fn → model
trainer = WakeWordTrainer(
    arch="gru",
    featurizer="mfcc",
    featurizer_type="mfcc",
    wake_word="hey mycroft",
    text_featurizer="phoneme_encoder.onnx",
    text_emb_dim=128,
    hidden_dim=128,
)
trainer.train(output_dir="experiments/hey_mycroft/phonmatch_b",
              train_data=train_data, ...)
```

**Export + inference** — text token IDs passed at inference:

```python
trainer.model.feature_extractor.export_to_onnx("best_f1_featurizer.onnx")
trainer.model.classifier.export_to_onnx("best_f1.onnx")

from ww_trainer.inference import OnnxWakeWordInferencer
model = OnnxWakeWordInferencer(
    "best_f1_featurizer.onnx", "best_f1.onnx",
    text_extractor_path="phoneme_encoder.onnx",
    text_emb_dim=128,
)
score = model.infer(wav_array, text_token_ids=keyword_ids)
```

**Fixed-keyword inference optimisation** — call `precompute()` once to cache the embedding:

```python
model.text_ext.precompute(keyword_ids)
score = model.infer(wav_array)  # no text_token_ids needed after precompute
```

---

## Combining with HALO loss

HALO loss (`name="halo"`) pairs especially well with PhonMatchNet because the phoneme-aligned
embeddings form geometrically cleaner clusters than raw acoustic embeddings:

```python
trainer = WakeWordTrainer(
    arch="phonmatch",
    featurizer="mfcc",
    featurizer_type="mfcc",
    wake_word="hey mycroft",
    hidden_dim=128,
    losses_cfg=[
        {"name": "bce",  "weight": 1.0},
        {"name": "halo", "weight": 0.5, "embed_dim": 128, "num_classes": 2},
    ],
)
```

---

## Mode comparison

| | Mode A (`arch="phonmatch"`) | Mode B (`text_featurizer=`) |
|---|---|---|
| **Head** | `PhonMatchHead` (cross-attention) | Any existing head |
| **Phoneme IDs at training** | Per-batch from dataset | Per-batch from dataset |
| **Phoneme IDs at inference** | `text_token_ids=` kwarg | `text_token_ids=` kwarg |
| **Zero-shot** | Yes (pass any IPA) | Yes (pass any IPA) |
| **ONNX inputs** | `input_features` + `phoneme_ids` | Separate text encoder ONNX |
| **Best for** | Maximum accuracy | Flexibility / head swapping |

Both modes require multi-keyword training data for real zero-shot coverage.

---

## API reference

| Symbol | Location | Description |
|--------|----------|-------------|
| `IPA_VOCAB` | `phonmatch.py:IPA_VOCAB` | 159-symbol IPA→int mapping |
| `ARPABET_TO_IPA` | `phonmatch.py:ARPABET_TO_IPA` | ARPAbet→IPA conversion table |
| `ipa_to_ids()` | `phonmatch.py:ipa_to_ids` | IPA strings → token IDs |
| `arpabet_to_ids()` | `phonmatch.py:arpabet_to_ids` | ARPAbet strings → IPA token IDs |
| `PhonMatchTextEncoder` | `phonmatch.py:PhonMatchTextEncoder` | Learnable phoneme encoder + ONNX export |
| `PhonMatchHead` | `phonmatch.py:PhonMatchHead` | Cross-attention classifier head |
| `OnnxTextExtractor` | `feats.py:OnnxTextExtractor` | Runtime text featurizer wrapper |
| `collate_fn` | `dataset.py:collate_fn` | Handles 2-tuple and 3-tuple batches |

See also: `examples/42_phonmatch_training.py` for a complete runnable training script.
