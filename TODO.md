# TODO

## Feature Extractors / Classifier Heads

### AudioSet-trained ONNX featurizer — `notebooks/nb10_audioset_featurizer.ipynb`
**Priority: High** | Status: Notebook created, model to be published

Train a compact ONNX-exportable CNN encoder on AudioSet as a general-audio featurizer,
analogous to BEATs but fully within the ww-trainer ONNX pipeline. Trained model to be
published to TigreGotico/onnx-feature-extractors on HuggingFace for downstream use
via `OnnxFeatureExtractor`.

- Notebook: `notebooks/nb10_audioset_featurizer.ipynb`
- Architecture: 4-block residual CNN → GRU → 256-dim embedding, ~3M params
- Loss: NT-Xent contrastive (AudioSet weak labels as positive pairs)
- Export: standard `export_to_onnx()` + INT8 quantized variant

---

### ~~EfficientNet-B0 head on log-mel — `EfficientNetHead(ClassifierHead)`~~ ✓ DONE
**Priority: Medium** | Effort: Low

Apply a 2D image CNN (EfficientNet-B0) to the log-mel spectrogram treated as a
single-channel image. Widely used in production KWS pipelines. Fills the gap between
the existing 1D `CnnClassifierHead` and the heavier transformer heads.

- Add `EfficientNetHead(ClassifierHead)` to `model.py`
- Input: log-mel reshaped to `[B, 1, n_mels, T]`
- Use `torchvision.models.efficientnet_b0(pretrained=False)` with adapted first conv + classifier

---

### Mamba / SSM-based head — DROPPED (not ONNX-exportable)
**Status: Will not implement until ONNX export is viable**

`mamba-ssm` uses custom CUDA kernels not registered as ONNX ops.
`mamba2-minimal` parallel scan uses control flow untraceable by torch.onnx.
All ww-trainer heads must export cleanly via `export_to_onnx()`.

Revisit if: (a) mamba-ssm adds official ONNX support, or (b) a pure-PyTorch
ONNX-safe SSM implementation (e.g. Hawk/Griffin-style linear RNN) becomes available.

---

### Fine-tunable HuBERT — `unfreeze_top_n` option on `HubertExtractor`
**Priority: Medium** | Effort: Medium

Current `HubertExtractor` freezes all HuBERT parameters. Partial fine-tuning
(unfreeze top N transformer layers, layer-wise LR decay or LoRA adapters) can
meaningfully improve F1/EER on small wake-word datasets without full fine-tuning cost.

- Add `unfreeze_top_n: int = 0` param to `HubertExtractor.__init__`
- Add optional LoRA adapter support via `peft` library
- Add `hubert_small_finetune` tier preset
- GPU-only; document clearly in tier guide
