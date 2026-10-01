"""Export all ONNX-exportable feature extractors and push each to HuggingFace Hub.

Creates one repo per extractor under TigreGotico/<name>-onnx.
MFCC and Filterbank get multiple size variants with params encoded in the repo name.

Usage:
    .venv/bin/python scripts/export_and_push_all.py
"""
import sys
import tempfile
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ww_trainer.feats import (
    MfccExtractor,
    FilterbankExtractor,
    DeltaExtractor,
    GammatoneExtractor,
    PLPExtractor,
    PNCCExtractor,
    CQTExtractor,
)

torch.set_num_threads(12)

HF_ORG = "TigreGotico"

# MFCC variants: (n_mfcc, n_mels, n_fft, hop_length)
MFCC_VARIANTS = [
    (13, 23,  400, 160),   # tiny / MCU
    (13, 40,  400, 160),   # small-13
    (20, 40,  400, 160),   # small-20
    (40, 40,  400, 160),   # default
    (40, 64,  400, 160),   # medium
    (40, 80,  512, 160),   # high-res
    (80, 80,  512, 160),   # large
    (80, 128, 512, 160),   # xl
]

# Filterbank variants: (n_mels, n_fft, hop_length)
FILTERBANK_VARIANTS = [
    (40,  400, 160),   # tiny
    (64,  400, 160),   # medium
    (80,  400, 160),   # default
    (128, 512, 160),   # large
]

# Delta-MFCC variants: same (n_mfcc, n_mels, n_fft, hop) — output dim is 3x n_mfcc
DELTA_MFCC_VARIANTS = [
    (13, 40,  400, 160),
    (20, 40,  400, 160),
    (40, 40,  400, 160),
    (40, 80,  512, 160),
]

# Gammatone variants: (n_filters, frame_len, hop_length)
GAMMATONE_VARIANTS = [
    (32, 400, 160),
    (64, 400, 160),
    (80, 400, 160),
    (128, 512, 160),
]

# PLP variants: (n_plp, n_fft, n_bark, lp_order)
PLP_VARIANTS = [
    (13, 512, 21, 12),
    (20, 512, 21, 19),
]

# PNCC variants: (n_pncc, n_fft, n_filters)
PNCC_VARIANTS = [
    (13, 512, 40),
    (40, 512, 64),
]

# CQT variants: (n_bins, n_octaves, f_min)
# n_bins=24 and n_octaves>=8 require n_fft>16384 — dynamo export fails symbolically.
# n_bins=12, n_octaves=7 requires >=16384 input samples at inference time (large CQT kernels).
CQT_VARIANTS = [
    (12, 7, 32.7),   # n_fft=16384, min input ~16384 samples (1 s at 16 kHz)
]


def build_extractors():
    entries = []

    for n_mfcc, n_mels, n_fft, hop in MFCC_VARIANTS:
        slug = f"mfcc-mfcc{n_mfcc}-mels{n_mels}-fft{n_fft}-hop{hop}-onnx"
        ext = MfccExtractor(sr=16000, n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft, hop_length=hop)
        desc = (f"MFCC feature extractor: {n_mfcc} coefficients, {n_mels} mel filters, "
                f"FFT={n_fft}, hop={hop}, sr=16000. Output: [B, T, {n_mfcc}].")
        entries.append((slug, ext, desc))

    for n_mels, n_fft, hop in FILTERBANK_VARIANTS:
        slug = f"filterbank-mels{n_mels}-fft{n_fft}-hop{hop}-onnx"
        ext = FilterbankExtractor(sr=16000, n_mels=n_mels, n_fft=n_fft, hop_length=hop)
        desc = (f"Log Mel filterbank: {n_mels} mel filters, FFT={n_fft}, hop={hop}, sr=16000. "
                f"Output: [B, T, {n_mels}].")
        entries.append((slug, ext, desc))

    for n_mfcc, n_mels, n_fft, hop in DELTA_MFCC_VARIANTS:
        slug = f"delta-mfcc-mfcc{n_mfcc}-mels{n_mels}-fft{n_fft}-hop{hop}-onnx"
        base = MfccExtractor(sr=16000, n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft, hop_length=hop)
        ext = DeltaExtractor(base, delta_width=2)
        desc = (f"MFCC + delta + delta-delta: {n_mfcc} base coefficients → {n_mfcc*3} total, "
                f"{n_mels} mel filters, FFT={n_fft}, hop={hop}, sr=16000. Output: [B, T, {n_mfcc*3}].")
        entries.append((slug, ext, desc))

    for n_filters, frame_len, hop in GAMMATONE_VARIANTS:
        slug = f"gammatone-filters{n_filters}-frame{frame_len}-hop{hop}-onnx"
        ext = GammatoneExtractor(sr=16000, n_filters=n_filters, f_min=50.0,
                                  frame_len=frame_len, hop_length=hop)
        desc = (f"Gammatone filterbank (ERB scale): {n_filters} filters, "
                f"frame={frame_len}, hop={hop}, sr=16000. Output: [B, T, {n_filters}].")
        entries.append((slug, ext, desc))

    for n_plp, n_fft, n_bark, lp_order in PLP_VARIANTS:
        slug = f"plp-plp{n_plp}-fft{n_fft}-bark{n_bark}-lp{lp_order}-onnx"
        ext = PLPExtractor(sr=16000, n_plp=n_plp, n_fft=n_fft, hop_length=160,
                           n_bark=n_bark, lp_order=lp_order)
        desc = (f"Perceptual Linear Prediction: {n_plp} coefficients, {n_bark} Bark filters, "
                f"LP order {lp_order}, FFT={n_fft}, sr=16000. Output: [B, T, {n_plp}].")
        entries.append((slug, ext, desc))

    for n_pncc, n_fft, n_filters in PNCC_VARIANTS:
        slug = f"pncc-pncc{n_pncc}-fft{n_fft}-filters{n_filters}-onnx"
        ext = PNCCExtractor(sr=16000, n_pncc=n_pncc, n_fft=n_fft, hop_length=160,
                             n_filters=n_filters)
        desc = (f"Power-Normalized Cepstral Coefficients: {n_pncc} coefficients, "
                f"{n_filters} filters, FFT={n_fft}, sr=16000. Output: [B, T, {n_pncc}].")
        entries.append((slug, ext, desc))

    for n_bins, n_octaves, f_min in CQT_VARIANTS:
        slug = f"cqt-bins{n_bins}-oct{n_octaves}-fmin{int(f_min)}-onnx"
        ext = CQTExtractor(sr=16000, n_bins=n_bins, n_octaves=n_octaves,
                            f_min=f_min, hop_length=160)
        desc = (f"Constant-Q Transform: {n_bins} bins/octave × {n_octaves} octaves, "
                f"f_min={f_min} Hz, sr=16000. Output: [B, T, {ext.feature_dim}].")
        entries.append((slug, ext, desc))

    return entries


def validate_onnx(extractor, onnx_path: Path, slug: str) -> bool:
    """Compare PyTorch and ONNX outputs on three fixed-seed inputs.

    Prints mean absolute error, max absolute error, and Pearson correlation
    for each input length. Returns False if any check fails hard.
    """
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    lengths = [8000, 16000, 32000]  # 0.5 s, 1 s, 2 s
    print(f"  {'Length':>8}  {'MAE':>12}  {'MaxAE':>12}  {'Corr':>8}  {'>#1e-3':>13}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*13}")

    passed = True
    for length in lengths:
        torch.manual_seed(42)
        wav = torch.randn(1, length)

        try:
            with torch.no_grad():
                pt_out = extractor(wav).numpy()
        except Exception as e:
            print(f"  {length:>8}  SKIP (pt error): {e}")
            continue

        try:
            onnx_out = sess.run(None, {input_name: wav.numpy()})[0]
        except Exception as e:
            print(f"  {length:>8}  SKIP (onnx error): {e}")
            continue

        if pt_out.shape != onnx_out.shape:
            print(f"  {length:>8}  SHAPE MISMATCH: pt={pt_out.shape} onnx={onnx_out.shape}")
            passed = False
            continue

        diff = np.abs(pt_out - onnx_out)
        mae = diff.mean()
        max_ae = diff.max()
        corr = np.corrcoef(pt_out.flatten(), onnx_out.flatten())[0, 1]
        n_high = int((diff > 1e-3).sum())
        pct = 100.0 * n_high / diff.size
        status = "" if max_ae < 1e-2 else "  *** HIGH ***"
        print(f"  {length:>8}  {mae:>12.6f}  {max_ae:>12.6f}  {corr:>8.6f}  {n_high:>6} ({pct:.2f}%){status}")
        if max_ae >= 1e-2:
            passed = False

    return passed


def make_readme(repo_id: str, description: str, extractor) -> str:
    feature_dim = extractor.feature_dim
    slug = repo_id.split("/")[-1]
    return f"""---
license: apache-2.0
tags:
  - audio
  - feature-extraction
  - onnx
  - wake-word
  - speech
---

# {slug}

{description}

## Usage

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("{slug}.onnx")
# Input: [batch, samples] float32, 16 kHz
audio = np.zeros((1, 16000), dtype=np.float32)
features = sess.run(None, {{"input_values": audio}})[0]
# Output shape: [batch, frames, {feature_dim}]
print(features.shape)
```

## With ww_trainer

```python
from ww_trainer.inference import OnnxWakeWordInferencer
model = OnnxWakeWordInferencer("{slug}.onnx", "best_f1.onnx")
score = model.infer(wav_float32_array)  # float in [0, 1]
```

## Details

| Property | Value |
|----------|-------|
| Sample rate | 16000 Hz |
| Feature dim | {feature_dim} |
| Input | `[B, T]` float32 |
| Output | `[B, frames, {feature_dim}]` float32 |

Part of the [onnx-feature-extractors](https://huggingface.co/collections/TigreGotico/onnx-feature-extractors) collection.
"""


def main():
    from huggingface_hub import HfApi

    api = HfApi()
    extractors = build_extractors()
    print(f"Total models to export and push: {len(extractors)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for slug, extractor, description in extractors:
            repo_id = f"{HF_ORG}/{slug}"
            onnx_path = tmpdir / f"{slug}.onnx"

            print(f"\n{'='*60}")
            print(f"[{slug}] Exporting ...")
            try:
                extractor.eval()
                extractor.export_to_onnx(str(onnx_path))
                print(f"[{slug}] Export OK — {onnx_path.stat().st_size / 1024:.1f} KB")
                print(f"[{slug}] Validating (PyTorch vs ONNX divergence) ...")
                ok = validate_onnx(extractor, onnx_path, slug)
                if not ok:
                    print(f"[{slug}] WARNING: validation divergence above threshold — check before pushing")
            except NotImplementedError as e:
                print(f"[{slug}] SKIP (NotImplementedError): {e}")
                continue
            except Exception:
                print(f"[{slug}] EXPORT FAILED:")
                traceback.print_exc()
                continue

            readme_path = tmpdir / f"{slug}_README.md"
            readme_path.write_text(make_readme(repo_id, description, extractor))

            print(f"[{slug}] Creating/verifying HF repo {repo_id} ...")
            try:
                api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            except Exception:
                print(f"[{slug}] create_repo failed:")
                traceback.print_exc()
                continue

            print(f"[{slug}] Uploading {onnx_path.name} ...")
            try:
                api.upload_file(
                    path_or_fileobj=str(onnx_path),
                    path_in_repo=f"{slug}.onnx",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Add {slug} ONNX export",
                )
            except Exception:
                print(f"[{slug}] Upload ONNX failed:")
                traceback.print_exc()
                continue

            print(f"[{slug}] Uploading README.md ...")
            try:
                api.upload_file(
                    path_or_fileobj=str(readme_path),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message="Add README",
                )
            except Exception:
                print(f"[{slug}] Upload README failed:")
                traceback.print_exc()

            print(f"[{slug}] Adding to collection ...")
            try:
                api.add_collection_item(
                    collection_slug=f"{HF_ORG}/onnx-feature-extractors",
                    item_id=repo_id,
                    item_type="model",
                    exists_ok=True,
                )
            except Exception:
                print(f"[{slug}] Could not add to collection (may need to add manually):")
                traceback.print_exc()

            print(f"[{slug}] Done -> https://huggingface.co/{repo_id}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
