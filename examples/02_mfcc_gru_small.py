#!/usr/bin/env python3
"""Small tier: MFCC extractor + GRU classifier head.

A step up from micro (~200K params), suitable for RPi and small SBCs.
The GRU head captures temporal patterns better than FFN for longer
wake words or noisy environments.
"""
import torch

from ww_trainer.feats import MfccExtractor
from ww_trainer.model import GruClassifierHead, BaseWakeModel


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = MfccExtractor(sr=16000, n_mfcc=40)
    head = GruClassifierHead(
        input_size=extractor.feature_dim,
        hidden_dim=128,
        linear_dim=128,
        gru_n_layers=1,
        bidirectional=False,
        device=device,
    )
    model = BaseWakeModel(extractor, head, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Small GRU model: {total_params:,} parameters")

    # --- Batch inference ---
    batch = torch.randn(4, 16000, device=device)  # 4 samples
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits)
    print(f"Batch probabilities: {probs.tolist()}")

    # --- Bidirectional variant ---
    head_bidir = GruClassifierHead(
        input_size=extractor.feature_dim,
        hidden_dim=128,
        bidirectional=True,
        gru_n_layers=2,
        device=device,
    )
    model_bidir = BaseWakeModel(extractor, head_bidir, device=device)
    bidir_params = sum(p.numel() for p in model_bidir.parameters())
    print(f"Bidirectional GRU: {bidir_params:,} parameters")


if __name__ == "__main__":
    main()
