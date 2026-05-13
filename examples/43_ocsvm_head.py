#!/usr/bin/env python3
"""Example 43 — OCSVMHead: two-stage FFN + One-Class SVM classifier.

Demonstrates the full OCSVMHead workflow:
  1. Build MFCC extractor + OCSVMHead + BaseWakeModel
  2. Forward pass before OCSVM is fitted (proxy score from all-zero SV buffer)
  3. Fit the OCSVM on a synthetic positive-only dataloader (Stage 2)
  4. Forward pass after fitting (RBF kernel decision score)
  5. ONNX export + parity check via onnxruntime

Prerequisites
-------------
    pip install ww_trainer[ocsvm]   # scikit-learn only needed for fit_ocsvm()

Use the ``ocsvm_small`` tier for a real training run:

    from ww_trainer.quickstart import train_from_wakeword
    train_from_wakeword("hey jarvis", "./hey_jarvis", tier="ocsvm_small", epochs=30)
"""
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ww_trainer.feats import MfccExtractor
from ww_trainer.model import OCSVMHead, BaseWakeModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_positive_loader(n_samples: int = 32, input_size: int = 40) -> DataLoader:
    """Return a DataLoader yielding (feats [B,T,F], labels [B]) with all positives."""
    feats = torch.randn(n_samples, 50, input_size)
    labels = torch.ones(n_samples, dtype=torch.long)
    ds = TensorDataset(feats, labels)
    return DataLoader(ds, batch_size=8)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    device = "cpu"
    INPUT_SIZE = 40   # MFCC bins
    HIDDEN_DIM = 128
    EMBED_DIM = 64

    print("=== OCSVMHead demo ===\n")

    # --- 1. Build components ---
    extractor = MfccExtractor(sr=16000, n_mfcc=INPUT_SIZE)
    head = OCSVMHead(
        input_size=extractor.feature_dim,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        nu=0.05,      # upper bound on fraction of outliers
        kernel="rbf",
        gamma="scale",
        device=device,
    )
    model = BaseWakeModel(extractor, head, device=device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {total:,} trainable params (backbone only; SV buffer grows after fitting)")

    # --- 2. Forward pass — before OCSVM is fitted ---
    audio = [torch.randn(16000)]   # 1 second at 16 kHz
    with torch.no_grad():
        score_before = model(audio).item()
    print(f"Score before fitting:  {score_before:.6f}  (all-zero SV buffer → constant 0)")

    # Embedding shape
    with torch.no_grad():
        emb = model.embed(audio)
    print(f"Embedding shape:       {emb.shape}   (backbone output)")

    # --- 3. Stage 2: fit OCSVM on positive-only data ---
    print("\nFitting OCSVM on synthetic positives …")
    loader = _make_positive_loader(n_samples=64, input_size=extractor.feature_dim)
    head.fit_ocsvm(loader)

    sv_count = head._sv_vectors.shape[0]
    print(f"OCSVM fitted: {sv_count} support vectors, rho={head._rho.item():.4f}")

    # --- 4. Forward pass — after OCSVM is fitted ---
    with torch.no_grad():
        score_after = model(audio).item()
    print(f"Score after fitting:   {score_after:.6f}  (RBF decision score; >0 = inlier)")

    # --- 5. ONNX export + parity check ---
    print("\nExporting to ONNX …")
    with tempfile.TemporaryDirectory() as tmpdir:
        ext_path = str(Path(tmpdir) / "mfcc_extractor.onnx")
        head_path = str(Path(tmpdir) / "ocsvm_head.onnx")

        extractor.export_to_onnx(ext_path)
        head.export_to_onnx(head_path)

        ext_kb = Path(ext_path).stat().st_size / 1024
        head_kb = Path(head_path).stat().st_size / 1024
        print(f"ONNX sizes: extractor={ext_kb:.1f} KB, head={head_kb:.1f} KB")

        # Parity check via onnxruntime
        try:
            import onnxruntime as ort

            feats = torch.randn(2, 50, extractor.feature_dim)
            head.eval()
            with torch.no_grad():
                pt_scores = head(feats).numpy()

            sess = ort.InferenceSession(head_path, providers=["CPUExecutionProvider"])
            ort_scores = sess.run(None, {"input_features": feats.numpy()})[0]

            max_diff = float(np.abs(pt_scores - ort_scores).max())
            print(f"ONNX parity check: max |PyTorch − ONNX| = {max_diff:.2e}")
            assert max_diff < 1e-4, f"Parity check failed: {max_diff}"
            print("ONNX parity OK ✓")
        except ImportError:
            print("(onnxruntime not installed — skipping parity check)")

    # --- Summary ---
    print("\n--- Summary ---")
    print("Stage 1: train backbone with BCE/focal loss (standard ww-trainer loop)")
    print("Stage 2: fit_ocsvm() — fits OneClassSVM on positive embeddings")
    print("Export:  both stages collapse into a single head ONNX graph")
    print("Tier:    use tier='ocsvm_small' in train_from_wakeword() for a real run")


if __name__ == "__main__":
    main()
