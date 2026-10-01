# TODO

## Feature Extractors / Classifier Heads

### AudioSet-trained ONNX featurizer — `notebooks/nb10_audioset_featurizer.ipynb`
**Priority: High** | Status: Notebook created, model to be published

Train a compact ONNX-exportable CNN encoder on AudioSet as a general-audio
featurizer, analogous to BEATs but fully within the ww-trainer ONNX pipeline.
Trained model to be published to `TigreGotico/onnx-feature-extractors` on
HuggingFace for downstream use via `OnnxFeatureExtractor`.

- Notebook: `notebooks/nb10_audioset_featurizer.ipynb`
- Architecture: 4-block residual CNN → GRU → 256-dim embedding, ~3M params
- Loss: NT-Xent contrastive (AudioSet weak labels as positive pairs)
- Export: standard `export_to_onnx()` + INT8 quantized variant

---

### ~~EfficientNet-B0 head on log-mel — `EfficientNetHead(ClassifierHead)`~~ ✓ DONE
**Priority: Medium** | Effort: Low

Apply a 2D image CNN (EfficientNet-B0) to the log-mel spectrogram treated as a
single-channel image. Widely used in production KWS pipelines. Fills the gap
between the existing 1D `CnnClassifierHead` and the heavier transformer heads.


---

## Pretrained Featurizer Hub — make experimentation 1-liner

**Priority: High** | Status: Design phase

Today, switching featurizers means manually exporting an ONNX, copying the path,
and wiring up `OnnxFeatureExtractor`. Goal: reduce that to one identifier so
researchers can sweep across pretrained encoders without leaving Python.

### 1. `PretrainedFeaturizer.from_pretrained(name)` registry
A thin loader pointing at `TigreGotico/onnx-feature-extractors` on HuggingFace:

```python
feat = PretrainedFeaturizer.from_pretrained("distilhubert")
trainer = WakeWordTrainer(feature_extractor=feat, ...)
```

- Resolves `name` → repo path + revision pin
- Downloads featurizer ONNX (+ optional INT8 variant) to a local cache
- Returns an `OnnxFeatureExtractor` already wired with the correct
  `sample_rate`, `hop`, `frame_size`, `output_dim` from a sidecar manifest
- Hash-pinned for reproducibility

### 2. Featurizer manifest schema (`manifest.json` next to each ONNX)
```json
{
  "name": "distilhubert",
  "sample_rate": 16000,
  "frame_size_ms": 25, "hop_ms": 20,
  "output_dim": 768,
  "layer": 6,
  "license": "apache-2.0",
  "paper": "https://arxiv.org/abs/2110.01900",
  "rtf_cpu": 0.12, "size_mb": 86,
  "sha256": "..."
}
```

Enables `ww-trainer list-featurizers` to print a comparison table without
downloading anything.

### 3. Initial featurizer zoo to publish
| Name | Source | Notes |
| :--- | :--- | :--- |
| `mfcc` | built-in | baseline, 0 params |
| `sincnet` | built-in | learnable filterbank |
| `distilhubert` | `ntu-spml/distilhubert` | already used in `nb07_distill` |
| `hubert-base-l6` | `facebook/hubert-base-ls960`, layer 6 | best KWS layer per HuBERT paper |
| `wav2vec2-base-l8` | `facebook/wav2vec2-base` | |
| `w2v-bert-2.0` | `facebook/w2v-bert-2.0` | export script exists |
| `yamnet` | TF-Hub port | AudioSet baseline |
| `panns-cnn14` | `qiuqiangkong/panns` | AudioSet SOTA-class |
| `beats-iter3` | `microsoft/BEATs` | if ONNX-tractable |
| `openl3-music` / `openl3-env` | | self-supervised |
| `clap-audio` | `laion/CLAP` audio encoder | embedding-space transfer |
| `tinyhubert` | our own distillation | already in repo |
| `audioset-resnet` | our own (above) | when ready |

### 4. CLI helpers
```bash
ww-trainer featurizers list                    # print zoo + manifests
ww-trainer featurizers download distilhubert   # cache locally
ww-trainer featurizers benchmark --eval set.json  # EER/RTF/size table
```

### 5. Frozen-featurizer adapter API
For fast sweeps, allow attaching a tiny trainable Linear/MLP adapter on top of a
frozen ONNX featurizer (features computed once per epoch, cached on disk).

```python
trainer = WakeWordTrainer(
    feature_extractor=PretrainedFeaturizer.from_pretrained("hubert-base-l6"),
    freeze_features=True,
    cache_features=True,   # one-shot precompute, reuse across runs
    head=FFNHead(input_dim=768, hidden=64),
)
```

### 6. Featurizer comparison harness
A notebook `notebooks/nb11_featurizer_zoo.ipynb`:
- For each zoo entry: train identical `FFNHead` for N epochs on `hey_mycroft`
- Report EER, FAR@1FRR, RTF, ONNX size, t-SNE of positive/negative embeddings
- Publish results table to README → user picks a featurizer at a glance

### 7. Layer-selection helper
For transformer SSL models, expose an optional `layer=` kwarg so users can probe
which intermediate layer transfers best (well-known: HuBERT L6–L8 ≫ final).

### 8. Resampling wrapper
Auto-insert a resampling stage when the dataset and featurizer sample rates
disagree (e.g. 22.05 kHz speech corpus + 16 kHz HuBERT). ONNX-traceable.

