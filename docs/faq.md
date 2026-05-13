# ww-trainer — FAQ

A topic-ordered Q&A. If you are brand-new, read top to bottom. Otherwise jump
to the section that matches your task.

- [1. First steps](#1-first-steps)
- [2. Install](#2-install)
- [3. Quickstart](#3-quickstart)
- [4. Datasets](#4-datasets)
- [5. Training](#5-training)
- [6. Metrics & evaluation](#6-metrics--evaluation)
- [7. Inference & deployment](#7-inference--deployment)
- [8. ONNX export](#8-onnx-export)
- [9. Feature extractors](#9-feature-extractors)
- [10. Classifier heads](#10-classifier-heads)
- [11. Embedded / ESP32](#11-embedded--esp32)
- [12. Hyperparameter search](#12-hyperparameter-search)
- [13. Notebooks & cloud](#13-notebooks--cloud)
- [14. Internals — where things live](#14-internals--where-things-live)
- [15. Troubleshooting](#15-troubleshooting)

---

## 1. First steps

**Q: What is a wake word and what does this framework build?**

A wake word is a short phrase ("hey jarvis") that an always-on listener detects
on-device, before any cloud STT runs. ww-trainer builds the listener: takes a
phrase or a dataset and emits two ONNX files you can ship to any onnxruntime
target — laptop, phone, Raspberry Pi, ESP32, browser.

**Q: Do I need a GPU?**

No. CPU works for the `micro` / `small` / `filterbank_small` / `gammatone_small`
/ `sincnet_small` tiers. A mid-range GPU speeds up the larger heads (BC-ResNet,
MatchboxNet, KWT, Conformer) and any model >1 M params. SSL featurizers (HuBERT,
Wav2Vec2-BERT) are used as pre-exported ONNX and run on CPU.

**Q: How much data do I need?**

For a smoke test, **zero** real recordings — TTS synthesises positives. For
production: hundreds of real far-field positives from multiple speakers plus
tens of hours of real negative audio. Hard-negative mining and voice conversion
close the rest of the gap.

**Q: Why are there two ONNX files instead of one?**

The featurizer (MFCC / SincNet / …) and the classifier head are exported
separately so you can:
- swap heads without re-exporting the featurizer,
- cache features on disk and reuse them across many head experiments,
- run the featurizer on a DSP and the head on a CPU.

Use `OnnxWakeWordInferencer(featurizer_path, head_path)` —
`ww_trainer/inference.py:8`.

**Q: What's the difference between F1, EER, FAR, FRR, and FA/hour?**

| Metric | Meaning | Typical target |
|---|---|---|
| **F1** | Harmonic mean of precision and recall | > 0.90 |
| **EER** | Operating point where false-accept = false-reject | < 0.05 |
| **FAR / FRR** | False-accept and false-reject at a chosen threshold | depends on product |
| **FA/hour** | False fires per hour of continuous ambient listening | < 1 |
| **AUT** | Area under the DET curve — lower is better, 0 is perfect. `area_under_det` — `ww_trainer/metrics.py` | the LiveKit operating metric |

Wake-word users care most about FA/hour and recall. Threshold tuning trades
one for the other; only data and loss choices can shift the whole curve.

**Q: Where do I start if my model false-fires or misses?**

1. Add real negative audio (podcasts, music, ambient).
2. Turn on hard-negative mining or use `scripts/train/train_infinite.py`.
3. Try `RobustProtoDiversityLoss` (RPPL) — [`rppl_whitepaper.md`](rppl_whitepaper.md).
4. Tune the threshold *only after* fixing the data; thresholding cannot rescue
   a bad model.

---

## 2. Install

**Q: How do I install ww-trainer?**

```bash
uv venv .venv
uv pip install -e ".[dev]"
```

**Q: What Python version is required?**

Python 3.10 or newer. Tested on 3.13.

**Q: What CUDA version is required?**

Optional. CUDA 11.8+ is recommended for the GPU path; the code auto-selects
CUDA when available and falls back to CPU otherwise.

**Q: What optional extras exist?**

| Extra | For… |
|---|---|
| `dev`, `test` | pytest + `onnxscript` (required for `torch.onnx.export` on torch ≥ 2.x) |
| `transformers` | HuBERT / Wav2Vec2 / W2V-BERT extractors |
| `sweep` | Optuna Bayesian search |
| `mlflow` | Experiment tracking |
| `datagen` | TTS + VAD synthetic dataset generation |
| `vc`, `vc-onnx`, `vc-torch`, `vc-linacodec` | Voice-conversion backends |
| `ocsvm` | One-Class SVM head (needs scikit-learn at fit time) |
| `mic` | Live microphone testing (sounddevice) |
| `viz` | UMAP embedding plots |
| `markov` | Markov / HMM extractors |
| `torchcodec` | FFmpeg-backed `torchaudio.save` / `.load` |

Install several at once: `uv pip install -e ".[dev,sweep,mlflow]"`.

**Q: Why is `torchcodec` an optional extra?**

It needs FFmpeg shared libraries at runtime. Minimal CI containers, embedded
targets, and many cloud runners do not provide them. `dataset._load_audio`
and `dataset._save_audio` already fall back to `soundfile` when torchcodec
is missing.

---

## 3. Quickstart

**Q: What's the minimum command to get a model?**

```bash
ww_trainer-quickstart --wake-word "hey computer" --output-dir ./hey_computer
```

Synthesises ~1000 TTS positives, mines negatives, trains a `small` tier model,
and emits `best_f1_featurizer.onnx` + `best_f1.onnx` under
`./hey_computer/model/`. Full reference in [`quickstart.md`](quickstart.md).

**Q: Python API equivalent?**

```python
from ww_trainer.quickstart import train_from_wakeword

result = train_from_wakeword("hey jarvis", "./hey_jarvis",
                             tier="micro", epochs=2)
print(result.best_onnx_path, result.metrics)
```

`train_from_wakeword` — `ww_trainer/quickstart.py:231`. Accepts any
`QuickstartConfig` field as a keyword.

**Q: I already have a dataset — how do I skip TTS synthesis?**

Pass `--reuse-dataset` (CLI) or `reuse_dataset=True` (Python). The dataset
directory must contain `train/metadata.csv` and `test/metadata.csv`.
`_run_or_load_datagen` — `ww_trainer/quickstart.py:97`.

**Q: Which tier should I pick?**

| Tier | Target | Params |
|---|---|---|
| `esp32_nano` / `esp32_sweet` / `esp32_max` | ESP32 / Cortex-M | ≤ 1 KB / 10 KB / 50 KB |
| `micro` / `delta_micro` | Pi Zero, very tight CPU | ~50 K |
| `small`, `filterbank_small`, `gammatone_small`, `sincnet_small` | Pi 3/4, generic Linux | ~200 K |
| `medium`, `large`, BCResNet/KWT/Conformer | server / GPU | 1–10 M |
| `hubert_small`, `hubert_medium` | GPU server | tens of M |

Run `ww_trainer-tiers` to list them. See also [`hardware_guide.md`](hardware_guide.md).

---

## 4. Datasets

**Q: What format does `AudioDataset` expect?**

A list of `(path, label)` tuples:

```python
samples = [
    ("/data/hey_jarvis_001.wav", "1"),  # wake
    ("/data/background_001.wav", "0"),  # not-wake
]
```

Supported file types: `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`. Label convention
is the string `"1"` / `"0"`. Full contract in [`data_contract.md`](data_contract.md).

**Q: What sample rate?**

16 000 Hz. `AudioDataset` auto-resamples sources at other rates via torchaudio.

**Q: Can I train offline?**

Yes. Pass `--no-augmentation-data` and `--reuse-dataset` to the quickstart, or
point `WakeWordTrainer` at local `bg_noise_folder`, `music_folder`,
`rir_folder`. The maintainer's offline mirror lives at
`/run/media/miro/endeavouros/ww/`.

**Q: How do I use my own dataset instead of TTS synthesis?**

Set `CUSTOM_TRAIN_CSV=/abs/path/to/metadata.csv` (`path,label` per line, no
header). Optionally set `CUSTOM_TEST_CSV`; if absent the notebook splits 80/20
and writes the split to `OUTPUT_DIR/dataset_split/` idempotently.

**Q: How do I force a specific HuggingFace dataset for positives?**

`HF_DATASET=org/repo-name` (e.g. `OpenVoiceOS/hey-jarvis-dataset`). Overrides
`find_positive_dataset` auto-detection. Negatives and augmentation proceed
normally.

**Q: How do I supply my own not-wake-word audio?**

`NEGATIVES_DIR=/path/to/neg_audio/` — used directly, no HF download for
general negatives. Works in all dataset modes.

**Q: How do I add extra HF repos without replacing the built-ins?**

| Env var | Appends to |
|---|---|
| `EXTRA_NEGATIVES_HF=org/repo1,org/repo2` | `NEGATIVE_DATASETS["general"]` |
| `EXTRA_BG_NOISE_HF=…` | background-noise pool |
| `EXTRA_MUSIC_HF=…` | music pool |
| `EXTRA_RIR_HF=…` | room-impulse-response pool |

**Q: How do I point augmentation at local folders?**

`BG_NOISE_DIR`, `MUSIC_DIR`, `RIR_DIR` to local paths. The corresponding
`DatagenResult` fields are overridden and HF downloads for that category are
skipped. Mix-and-match with the `EXTRA_*_HF` envs is supported.

**Q: How do I validate a dataset before training?**

`AudioDataset(samples, validate=True)` — checks all files are readable audio.
Class-imbalance warnings (>10:1) are always on.

---

## 5. Training

**Q: How do I start a training run from raw flags?**

```bash
uv run ww_trainer-train --help
```

Key flags: `--wake-folder`, `--non-wake-folder`, `--arch {ffn,gru,cnn,bcresnet,…}`,
`--epochs`, `--batch-size`, `--loss`, `--lr`. Full reference:
[`training.md`](training.md).

**Q: Which loss functions are supported?**

The full catalogue is in [`losses.md`](losses.md). The ones you reach for first:

| Loss | When |
|---|---|
| `bce` | always-safe baseline |
| `focal` | class imbalance |
| `arcface` | tight embedding clusters for retrieval-style heads |
| `supcon` / `ntxent` | self-/supervised contrastive |
| `triplet` / `soft_triplet` / `pair` | siamese / metric learning |
| `rppl` | recommended for production — composite prototype loss with hard-neg diversity. See [`rppl_whitepaper.md`](rppl_whitepaper.md). |
| `size_aware` | wrap any base loss to add sparsity + param-count penalty (ESP32). |

**Q: How do I freeze layers / do transfer learning?**

`--freeze-extractor` freezes the whole feature extractor; `--freeze-layers N`
freezes the first N classifier params; `--unfreeze-at-epoch N` does progressive
unfreezing. `WakeWordTrainer._freeze` — `ww_trainer/trainer.py`.

**Q: What is feature caching?**

`--feature-cache-dir .feature_cache/` (default). Un-augmented waveforms are
cached as `.npy` keyed by `MD5(file_content + extractor_identity)`. Bypassed
when augmentation is active. Disable with `--no-feature-cache`.
`FeatureCache` — `ww_trainer/cache.py`.

**Q: Epoch-level data replacement?**

`--replacement-ratio 0.4` drops 40 % of the epoch and replaces from the full
pool, reducing overfit. `--balanced-replacement` keeps 50/50 wake/non-wake in
the replaced portion. Complements hard-negative mining.

**Q: How do I run multi-GPU training?**

`ww_trainer.ddp` + `torchrun --nproc_per_node=N`. Call `setup_ddp()`,
`wrap_model_ddp(model, local_rank)`, `create_distributed_loader(dataset, batch_size)`.

**Q: How do I enable MLflow tracking?**

`--mlflow-uri http://localhost:5000` (or any valid tracking URI). Metrics,
artifacts, and ONNX checkpoints are logged automatically.

**Q: How do I run the smoke tests?**

```bash
.venv/bin/python -m pytest test/smoketests/ -v
```

~100 tests covering every extractor × head × loss combo on dummy data,
~25 seconds.

---

## 6. Metrics & evaluation

**Q: How do I evaluate a trained model?**

`scripts/eval/eval_hey_mycroft.py` (and friends) emit ROC / PR / DET, FP and FN
lists, confidence histograms, and an FP/hour estimate against an ambient
corpus. See [`benchmarking.md`](benchmarking.md).

**Q: What does `--target-fp-per-hour` do?**

End-of-epoch controller: if the measured FP/hour on `--ambient-dir` exceeds the
target, `max_neg_weight` is doubled for the next epoch. Requires
`--neg-weight-schedule {linear|cosine}` and `--ambient-dir`. Doubling site in
`ww_trainer/loop.py` after `estimate_fp_per_hour`.

**Q: How do I calibrate probabilities?**

`ww_trainer.calibration.calibrate_model(model, val_data, output_dir)` after
training. Fits Platt scaling on validation logits and saves `calibration.json`.
Apply with `apply_platt_scaling(logits, params)`.

**Q: What is the composite fitness score?**

`compute_fitness_score()` — `ww_trainer/evaluation.py` — combines detection
quality and model size: `(1 - 0.8·FP_rate - 0.2·FN_rate) · size_penalty`.
FP is penalised 4× harder than FN. Enable best-fitness checkpointing with
`--fitness-checkpoint` and set `--fitness-param-budget` for the size penalty.

**Q: How do I average several checkpoints?**

```python
from ww_trainer.checkpoint import select_best_checkpoints, average_checkpoints

keep = select_best_checkpoints(metric_history)  # FPPH ≤ p10 ∧ recall ≥ p90 ∧ acc ≥ p90
average_checkpoints([h["path"] for h in keep], out_path="best_avg.pt")
```

`select_best_checkpoints` falls back to the highest-recall checkpoint if no
entry passes all three gates. `ww_trainer/checkpoint.py`.

---

## 7. Inference & deployment

**Q: How do I run inference without PyTorch installed?**

`OnnxWakeWordInferencer` needs only `numpy` + `onnxruntime`:

```python
from ww_trainer.inference import OnnxWakeWordInferencer
import numpy as np

inf = OnnxWakeWordInferencer(
    extractor_path="mfcc.onnx",
    head_path="classifier_head.onnx",
    sample_rate=16000,
)

audio = np.zeros(16000, dtype=np.float32)
prob = inf.infer(audio)            # single clip → float in [0, 1]
probs = inf.infer_batch(batch)     # [B, T] → [B]
```

**Q: How do I run streaming inference (ONNX, no PyTorch)?**

```python
cache = None
for chunk in audio_chunks:                       # 1-D float32 numpy
    prob, cache = inf.infer_streaming(chunk, cache)
    if prob > 0.5:
        print("detected")
```

The cache is a `[T_cached, F]` numpy array grown and windowed automatically.
`OnnxWakeWordInferencer.infer_streaming` — `ww_trainer/inference.py`.

**Q: How do I run streaming inference with PyTorch?**

```python
from ww_trainer.feats import MfccExtractor, SlidingFeatureCacheTensor
from ww_trainer.model import FfnClassifierHead, BaseWakeModel

extractor = MfccExtractor(sr=16000, n_mfcc=13)
head = FfnClassifierHead(input_size=13, hidden_dim=64)
model = BaseWakeModel(feature_extractor=extractor,
                     classifier=head, sample_rate=16000).eval()

cache = SlidingFeatureCacheTensor(feature_dim=13, window_size=50)
for chunk in audio_chunks:
    prob = model.forward_streaming(chunk, cache)
```

`SlidingFeatureCacheTensor` is updated in-place each call.
`forward_streaming` — `ww_trainer/model.py`.

**Q: Can I run wake-word models in the browser?**

Yes — `examples/wakeword_tester.html` is a single-page tester using
onnxruntime-web. Drop in your two ONNX files; no build step.

---

## 8. ONNX export

**Q: How do I export a trained model to ONNX?**

The trainer's `--export-onnx` flag (on by default) writes the head ONNX at
every "best" checkpoint. Export the featurizer separately after training:

```python
trainer.model.load_checkpoint("best_f1.pt")
trainer.model.feature_extractor.export_to_onnx("best_f1_featurizer.onnx")
trainer.model.classifier.export_to_onnx("best_f1.onnx")
```

`ClassifierHead.export_to_onnx` — `ww_trainer/model.py:62`. All extractor and
head exports accept an optional `metadata: dict` of key/value strings.

**Q: What metadata is embedded?**

| Key | Meaning |
|---|---|
| `wake_word` | target phrase |
| `arch` | head class name |
| `featurizer` | extractor class name |
| `epoch` | checkpoint epoch |
| `metric_*` | F1 / Precision / Recall / Loss at save time |

Inspect with `onnx.load("model.onnx").metadata_props`.

**Q: Can I quantize the export?**

Yes. `export_to_onnx(quantize=True)` writes both INT8 and INT16 variants
alongside the float32 export.

**Q: Are Markov and HMM extractors ONNX-compatible?**

Yes. `MarkovTransitionExtractor` uses a vectorised state lookup; `HMMStateExtractor`
also exports cleanly. The entire pipeline (base extractor + Markov/HMM wrapper)
becomes a single symbolic graph.

**Q: Why is there a tiny PyTorch ↔ ONNX numeric drift?**

`torch.stft` and the exported ONNX STFT differ in windowing and FFT
implementation, producing ~1e-4 absolute differences. Use `1e-3` tolerance in
parity tests. Rarely affects wake-word accuracy.

---

## 9. Feature extractors

**Q: What extractors are available?**

The full catalogue with parameters and citations is in
[`extractors.md`](extractors.md). At a glance:

| Family | Notes |
|---|---|
| `mfcc`, `filterbank`, `delta_mfcc` | Classical, ONNX-exportable, fast |
| `sincnet`, `gammatone`, `leaf` | Learnable / auditory frontends |
| `plp`, `pncc`, `cqt` | Classical perceptual variants |
| `markov`, `hmm` | Non-neural temporal models |
| `onnx` (`OnnxFeatureExtractor`) | Load any pre-exported ONNX as a black-box featurizer |
| Enrichment wrappers (`vad`, `pitch`, `snr_aware`, `multi_resolution`, `delta`) | Stack on top of any base extractor |

For HuBERT / Wav2Vec2 / W2V-BERT: export the SSL model to ONNX once via
`scripts/export_hubert.py`, `scripts/export_wav2vec2.py`, or
`scripts/export_w2vbert.py`, then load with `OnnxFeatureExtractor`. Same ONNX
in training and inference guarantees feature parity.

**Q: How do I select a featurizer?**

```python
trainer = WakeWordTrainer(arch="gru", featurizer_type="mfcc")
```

Or via `create_model`:

```python
model = WakeWordTrainer.create_model("gru",
                                     featurizer="",
                                     featurizer_type="mfcc")
```

**Q: How do I query the feature dimension?**

```python
from ww_trainer.feats import MfccExtractor, OnnxFeatureExtractor
MfccExtractor(n_mfcc=40).feature_dim                # 40
OnnxFeatureExtractor("model.onnx").feature_dim      # from ONNX shape
```

`WakeWordTrainer.create_model` auto-detects `feature_dim` when the kwarg is
omitted.

**Q: Can I use any pre-trained audio ONNX as a featurizer?**

Yes — that's the design of `OnnxFeatureExtractor`. Build a Markov / Silero VAD /
HuBERT featurizer once with a standalone script, then load it as a black box:

```bash
python scripts/train_markov_featurizer.py --wake-folder ./my_wakes --out markov.onnx
ww_trainer-train --featurizer markov.onnx --featurizer-type onnx --arch gru ...
```

Examples 38, 39, 40 in [`examples/`](../examples/) show the multi-ONNX
pipeline pattern.

---

## 10. Classifier heads

**Q: What heads are available?**

The full catalogue is in [`classifiers.md`](classifiers.md). Highlights:

| Head | Use it for |
|---|---|
| `ffn` | smallest, baseline, ESP32 |
| `gru` | best general temporal head; production default |
| `cnn`, `bcresnet`, `tcresnet`, `dscnn`, `res15`, `matchboxnet` | KWS production architectures |
| `kwt`, `conformer` | attention-based; GPU |
| `mixconv`, `efficientnet` | image-CNN-style on log-mel |
| `crnn`, `convattention` | conv + recurrent / attention hybrids |
| `ocsvm` | anomaly-style; few or no negatives |

**Q: What is `OCSVMHead` and when should I use it?**

`OCSVMHead` — `ww_trainer/model.py:271` — is a two-stage classifier: an FFN
backbone produces embeddings, then a One-Class SVM provides the decision
boundary. Use it when you have few or no negatives — the OCSVM encloses the
positive embedding manifold rather than discriminating against negatives.

```python
# After normal training:
OCSVMHead.fit_ocsvm(dataloader)         # model.py:392
```

Requires `pip install ww_trainer[ocsvm]` (scikit-learn at fit time only). Tier
preset: `ocsvm_small` — `ww_trainer/tiers.py:174`.

**Q: Which OCSVM kernels are supported?**

`kernel={"rbf", "linear", "poly", "sigmoid"}` — all four are pure PyTorch
(`OCSVMHead._kernel_vals` — `model.py:343`) and ONNX-exportable. `gamma="scale"`
(default) is resolved by sklearn from training embeddings and stored as a
buffer. `"poly"` accepts `degree` and `coef0`.

**Q: Does `OCSVMHead` export to ONNX?**

Yes — support vectors, dual coefficients, bias, gamma, degree, coef0 are all
`register_buffer` tensors. **Export after `fit_ocsvm`** — exporting before
bakes in zero buffers and produces constant-zero scores.

**Q: What is `ConvAttentionHead`?**

A small head (Conv1d × N → MultiheadAttention + residual → mean-pool → Linear)
ported from
[livekit/livekit-wakeword](https://github.com/livekit/livekit-wakeword) under
Apache-2.0. `ConvAttentionHead` — `ww_trainer/model.py:1357`. Variable-`T` and
ONNX-exportable.

---

## 11. Embedded / ESP32

**Q: What ESP32 tiers exist?**

Targeting ESP32 (520 KB RAM, 4 MB flash):

| Tier | Budget |
|---|---|
| `esp32_nano` | ≤ 1 024 params (sub-1 KB int8) |
| `esp32_sweet` | ≤ 10 240 params |
| `esp32_max` | ≤ 51 200 params |

All use MFCC + FFN. Defined in `ww_trainer/tiers.py`.

**Q: What's the smallest possible model?**

MFCC-13 + FFN-8 = 97 params = 0.1 KB int8. The nano tier default is
MFCC-13 + FFN-16 = 241 params (0.2 KB).

**Q: How do I export an FFN head to a C header for the ESP32?**

```python
from ww_trainer.export_c import export_to_c_header
export_to_c_header(model, "model.h")
```

Generates a self-contained `.h` with int8 weights and a `ww_model_infer()`
function. FFN heads only.

**Q: What is `SizeAwareLoss`?**

A wrapper (`ww_trainer/loss.py:SizeAwareLoss`) that adds L1 sparsity +
param-count penalties to any base loss:

```python
losses_cfg = [{"name": "size_aware", "param_budget": 1024}]
```

**Q: How does the micro genetic search work?**

`sweep.run_micro_search()` uses composite fitness =
`accuracy_weight·F1 + size_weight·(1 − params / budget)`. Models exceeding the
tier's param budget are hard-rejected. The search space is auto-constrained
per tier.

More: [`esp32.md`](esp32.md), [`hardware_guide.md`](hardware_guide.md).

---

## 12. Hyperparameter search

**Q: What search strategies are available?**

Optuna (Bayesian), Grid, Random, Genetic, Two-Stage Genetic — full reference
in [`sweep.md`](sweep.md) and [`search_strategies.md`](search_strategies.md).

**Q: What is two-stage genetic search?**

A coarse global search (stage 1) seeds a focused fine-tune round (stage 2)
with the top-K configs. Each history entry carries a `stage` field.
`run_two_stage_genetic_search` — `ww_trainer/sweep.py:662`.

**Q: When should I use `n_demes > 1`?**

On multi-core CPUs or multi-GPU. Each deme is an independent GA population
run in parallel, preventing premature convergence. Use 2–4 demes (one per core).
Each writes to `output_dir/deme_{id}/` for isolation.
`run_genetic_search` — `ww_trainer/sweep.py:543`.

**Q: What fitness function should I pick?**

| `fitness_fn` | Behaviour |
|---|---|
| `f1` (default) | raw F1 |
| `exp_f1` | steepens the gradient — helps when F1 plateaus above 0.9 |
| `double_exp_f1` | extreme pressure near the optimum |

The transform is applied **only for selection**; `best_score` in the result is
always raw F1. Unknown values raise `ValueError`. `_validate_ga_params` —
`ww_trainer/sweep.py:24`.

**Q: What fields does each GA history entry contain?**

`{"generation": int, "best": float, "avg": float, "elapsed_seconds": float}`.
Two-stage runs add `"stage"` (1 or 2). `sweep.py:774-778`.

**Q: What is infinite training?**

A goal-based loop that mines hard negatives + optionally synthesises VC
positives every epoch and stops when configured F1/EER targets hit.
`scripts/train/train_infinite.py`. Designed for very large NWW pools.

---

## 13. Notebooks & cloud

**Q: How do I run the genetic search notebook on Kaggle?**

1. Upload or link the repo to a Kaggle notebook.
2. Add secrets in **Add-ons → Secrets**: `WAKE_WORD`, `OUTPUT_DIR`
   (e.g. `/kaggle/working/ww_output`).
3. Open `notebooks/genetic_search.ipynb`; the Config cell reads everything via
   `os.environ.get(...)`.
4. Run all cells. Platform auto-detection installs dependencies.
5. Outputs (ONNX, evolution plot, benchmark PNGs) land in `OUTPUT_DIR` and
   the Kaggle output panel.

**Q: Which env vars control the notebook?**

| Var | Default | Purpose |
|---|---|---|
| `WAKE_WORD` | `hey jarvis` | target phrase |
| `OUTPUT_DIR` | `./ww_output` | output root |
| `POPULATION` / `GENERATIONS` | 12 / 5 | GA budget |
| `TIERS_TO_TRAIN` | `micro,small,filterbank_small` | architecture tiers |
| `FINAL_EPOCHS` | 30 | per-tier final epochs |
| `SEARCH_TWO_STAGE` | `false` | enable two-stage GA |
| `SEARCH_DEMES` | 1 | parallel islands |
| `SEARCH_FITNESS_FN` | `f1` | selection pressure |
| `CUSTOM_TRAIN_CSV` / `CUSTOM_TEST_CSV` | — | BYO dataset |
| `HF_DATASET` | — | override positive source |
| `NEGATIVES_DIR` | — | override negative source |
| `BG_NOISE_DIR` / `MUSIC_DIR` / `RIR_DIR` | — | local augmentation |
| `EXTRA_NEGATIVES_HF` / `EXTRA_BG_NOISE_HF` / `EXTRA_MUSIC_HF` / `EXTRA_RIR_HF` | — | append HF repos |

All have safe defaults — the notebook runs end-to-end with no env vars set.

**Q: How does dataset resume safety work?**

Cell 4 is fully resume-safe across all three dataset modes:
- **BYO CSV**: no datagen at all; the optional 80/20 split is written once to
  `OUTPUT_DIR/dataset_split/` and reused.
- **HF / Auto**: `reuse_dataset=True` is always passed — if
  `dataset/train/metadata.csv` and `dataset/test/metadata.csv` exist under
  `OUTPUT_DIR`, datagen is skipped. `_run_or_load_datagen` —
  `ww_trainer/quickstart.py:97`.

---

## 14. Internals — where things live

| Concern | Module |
|---|---|
| Epoch training loop | `training_loop()` — `ww_trainer/loop.py` (delegated from `WakeWordTrainer.train`) |
| Visualisation (PCA, t-SNE, UMAP, ROC/PR/DET, RPPL dashboard) | `ww_trainer/visualization.py` |
| Hard-negative mining | `mine_hard_negatives()` — `ww_trainer/mining.py` |
| Checkpoint save/load | `ww_trainer/checkpoint.py` |
| Readiness score | `compute_readiness()` — `ww_trainer/evaluation.py` |
| Intermediate checkpoints | `save_intermediate_checkpoint()` — `ww_trainer/checkpoint.py` |
| TTS plugin discovery | `_collect_tts_plugins()` — uses `ovos_plugin_manager.tts.find_tts_plugins()` (auto-discovers edge-tts, piper, phoonnx, …) |
| VAD | `OVOSVADFactory` with `ovos-vad-plugin-silero` by default |
| Feature cache | `FeatureCache` — `ww_trainer/cache.py` |
| Silero VAD wrapper | `SileroVadWrapper` — `ww_trainer/feats.py` (lazy `torch.hub.load()` on first `forward`) |
| Hardware tier presets | `ww_trainer/tiers.py` |
| Quickstart pipeline | `_run_or_load_datagen`, `_train_from_datagen_result` — `ww_trainer/quickstart.py:97,144` |

**Q: How is augmentation wired from datagen to training?**

`_train_from_datagen_result` reads `bg_noise_dir`, `music_dir`, `rir_dir` from
`DatagenResult` and passes them as `bg_noise_folder`, `music_folder`,
`rir_folder` kwargs to `WakeWordTrainer` — `ww_trainer/quickstart.py:144`.

**Q: How do I add a new TTS engine?**

Install any OVOS TTS plugin (`uv pip install ovos-tts-plugin-<name>`). OPM
entry-point discovery picks it up automatically — no code changes.

---

## 15. Troubleshooting

**Q: `ModuleNotFoundError: No module named 'chatterbox_onnx'`**

Optional voice-conversion dependency. Only needed when `vc_folder` is set on
`AudioDataset`. `pip install ww_trainer[vc-onnx]`.

**Q: `ModuleNotFoundError: No module named 'torchcodec'` / `ImportError: TorchCodec is required for save_with_torchcodec`**

torchaudio ≥ 2.9 defaults to torchcodec for `load`/`save`. ww-trainer's
`dataset._load_audio` and `dataset._save_audio` already fall back to
`soundfile`; this error means external code is calling `torchaudio.load`/`save`
directly. Either route it through the helpers, or `pip install ww_trainer[torchcodec]`
(needs system FFmpeg).

**Q: `ModuleNotFoundError: No module named 'onnxscript'`**

PyTorch ≥ 2.x ONNX export requires `onnxscript`. It is included in the `[dev]`
and `[test]` extras. `pip install ww_trainer[dev]` or `pip install onnxscript`.

**Q: `ValueError: Ambiguous feature shape` in GRU**

`GruClassifierHead` cannot auto-detect orientation when both spatial
dimensions equal `input_size`. Pass features as `[B, T, F]` explicitly.

**Q: Training loss is NaN**

Common causes:
1. learning rate too high,
2. audio files contain silence / very short clips,
3. the batch has only one class (no valid triplets for metric losses).

Enable `--debug` for per-step loss logging.

**Q: ONNX file is much bigger than the PyTorch checkpoint.**

Expected. ONNX serialises constants for every weight in float32; the `.pt`
file stores a compressed dict. Use `quantize=True` on `export_to_onnx` for INT8
/ INT16 variants — these match or beat the `.pt` size.
