# Wake Word Trainer

A research-grade training suite for **wake-word detection** — the always-on
keyword spotter that wakes "Hey Siri", "OK Google", or your own custom phrase.
Train, evaluate, and ship lightweight on-device detectors that run anywhere
from an ESP32 to a GPU server. Every component exports to ONNX; production
inference requires only `onnxruntime` and `numpy` — no PyTorch at runtime.

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
