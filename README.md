# Wake Word Trainer

A research-grade training suite for **wake-word detection** — the always-on
keyword spotter that wakes "Hey Siri", "OK Google", or your own custom phrase.
Train, evaluate, and ship lightweight on-device detectors that run anywhere
from an ESP32 to a GPU server. Every component exports to ONNX; production
inference requires only `onnxruntime` and `numpy` — no PyTorch at runtime.

## What is a wake word?

A short phrase ("hey jarvis", "computer", "alexa") that a device listens for
continuously. When detected, downstream STT/NLU runs. A useful detector must
run on tiny hardware (sub-100 KB, <10 % CPU, no internet), tolerate noise and
distance, almost never false-fire (< 1 FA / hour), and trigger reliably when
spoken (> 90 % recall at that operating point). ww-trainer is the toolchain
that builds such a detector from a single phrase — synthesise data, train,
evaluate, export, deploy.

## Who is this for?

| You are… | Start here |
|---|---|
| **Hobbyist** waking a Pi with your own phrase | [`docs/getting_started/quickstart.md`](docs/getting_started/quickstart.md) — ONNX in 5 minutes |
| **Embedded engineer** shipping to ESP32 / MCU | [`docs/guides/embedded.md`](docs/guides/embedded.md) |
| **Voice-assistant integrator** (OVOS, Rhasspy, …) | [`docs/guides/inference.md`](docs/guides/inference.md) |
| **ML researcher** comparing architectures / losses | [`docs/guides/search.md`](docs/guides/search.md), [`docs/reference/losses.md`](docs/reference/losses.md), [`docs/research/rppl.md`](docs/research/rppl.md) |
| **New to ML** entirely | [`notebooks/kaggle_quickstart.ipynb`](notebooks/kaggle_quickstart.ipynb) — runs free on Kaggle |

## Highlights

- **Single-string-to-ONNX** quickstart — `train_from_wakeword("hey jarvis", out)` produces a deployable model.
- **17 featurizers × 15 classifier heads × 17 losses** — a real research surface.
- **Genetic + Bayesian HP search** with island-model parallelism, adaptive mutation, two-stage refinement.
- **Synthetic datagen** — TTS + voice conversion to bootstrap a dataset from zero recordings.
- **Hard-negative mining** and **infinite training** for industrial-scale negative pools.
- **ONNX-first**: featurizer and head export cleanly; no CUDA-only kernels.
- **Hardware tiers** from `esp32_nano` (sub-1 KB int8) to `hubert_medium`.

### Honest trade-offs

- CPU training works for small tiers; a mid-range GPU is the best UX for larger ones.
- Synthetic data is great for smoke-testing — production still needs real far-field recordings.
- ONNX-export is mandatory; non-traceable components (custom CUDA kernels, dynamic control flow) are out of scope. See [`docs/internals/known_issues.md`](docs/internals/known_issues.md).
- SSL featurizers (HuBERT, Wav2Vec2-BERT) are used as pre-exported ONNX and held frozen during downstream training — guarantees train/inference parity but limits adaptation.

## Install

```bash
uv pip install -e ".[dev]"
```

Optional extras (`sweep`, `transformers`, `mlflow`, `datagen`, `vc`, `mic`,
`viz`, `markov`, `ocsvm`, `torchcodec`) — see
[`docs/faq.md`](docs/faq.md#2-install).

## 60-second quickstart

```bash
ww_trainer-quickstart --wake-word "hey jarvis" --output-dir ./hey_jarvis
```

Or in Python:

```python
from ww_trainer.quickstart import train_from_wakeword
result = train_from_wakeword("hey jarvis", "./hey_jarvis",
                             tier="small", epochs=50)
print(result.best_onnx_path, result.metrics)
```

Output: `best_f1_featurizer.onnx` + `best_f1.onnx` under
`./hey_jarvis/model/`. Load both with `OnnxWakeWordInferencer` —
[`docs/guides/inference.md`](docs/guides/inference.md).

## Documentation

Everything lives in [`docs/`](docs/index.md). Start with:

- [docs/learning_path.md](docs/learning_path.md) — **zero-to-hero curriculum** with literature anchors
- [docs/faq.md](docs/faq.md) — topic-ordered Q&A in 15 sections
- [docs/index.md](docs/index.md) — full documentation index
- [examples/README.md](examples/README.md) — 43 runnable examples

## Contributing

Issues and pull requests welcome on the `dev` branch. Tests live in `test/`;
run with `uv run pytest`.

## Citation

```bibtex
@software{ww_trainer,
  title  = {ww-trainer: a research suite for on-device wake-word detection},
  author = {TigreGotico contributors},
  year   = {2026},
  url    = {https://github.com/TigreGotico/ww-trainer},
  note   = {Funded by NGI0 Commons Fund / NLnet, grant 101135429}
}
```

## Credits

Funded by [NGI0 Commons Fund](https://nlnet.nl/project/OpenVoiceOS) / NLnet
under grant agreement No
[101135429](https://cordis.europa.eu/project/id/101135429).

![](./ngi.png)

## License

Apache 2.0
