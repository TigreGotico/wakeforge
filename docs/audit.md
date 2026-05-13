# Known Issues

Open issues and known limitations. Evidence-based, with `file:line` citations.

---

## STFT parity (Low)

**File:** `ww_trainer/feats.py:317`

`torch.stft` in MFCC and Filterbank extractors produces slightly different
results in the exported ONNX graph vs. native PyTorch — internal FFT windowing
and floating-point optimisations in ONNX Runtime differ by ~`1e-4`. Use
`tolerance=1e-3` in parity tests. Documented in [`faq.md`](faq.md).

---

## `pytest --cov` on Python 3.13 (Low)

**File:** `test/conftest.py:6`

`uv run pytest --cov=ww_trainer` raises
`ImportError: cannot load module more than once per process` from
`numpy._core`. Known CPython 3.13 + numpy + coverage interaction when
coverage instruments the numpy C extension at import time.

Workarounds:
- `uv run pytest test/` (no `--cov`)
- `uv run pytest test/ --cov=ww_trainer --no-cov-on-fail`
- Python 3.11 / 3.12 for coverage runs.

---

## ONNX-export constraints

Every featurizer and head must export cleanly under `torch.onnx` opset 18.
This rules out:

- Custom CUDA kernels not registered as ONNX ops (e.g. `mamba-ssm`).
- Non-traceable control flow in `forward()` (e.g. parallel scans, dynamic
  Python branches on tensor values).
- `torchaudio.compliance.kaldi` / `torchaudio.transforms` in `forward()` —
  use `torch.stft` + buffer-registered filterbanks instead.

Verification per extractor / head: `export_to_onnx("test.onnx")` followed by
`onnx.checker.check_model(...)` must pass.

---

## SSL featurizers are frozen at training time

HuBERT, Wav2Vec2, and Wav2Vec2-BERT are used as **pre-exported ONNX** featurizers
via `OnnxFeatureExtractor`. The framework does not fine-tune their weights
during downstream wake-word training. This guarantees train/inference parity
but limits SSL adaptation. To adapt the SSL model, fine-tune externally with
the source library (fairseq, transformers), re-export, and reload.

---

## CPU-only training cost

The smaller tiers (`micro`, `delta_micro`, `small`, `filterbank_small`,
`gammatone_small`, `sincnet_small`) train fine on CPU. The transformer heads
(`KWT`, `Conformer`) and large featurizers (HuBERT-base) are practical only
on GPU.

---

## Reporting new issues

GitHub Issues: <https://github.com/TigreGotico/ww-trainer/issues>. Include
your `pyproject.toml` extras, Python and torch versions, and a minimal
repro script.
