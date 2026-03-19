#!/usr/bin/env python3
"""ArcFace loss for wake word detection.

ArcFace (Additive Angular Margin Loss) enforces inter-class separability
by adding an angular margin penalty in cosine space. Originally designed
for face verification, it produces highly discriminative embeddings for
binary wake word detection.

Setup: MfccExtractor (13 dims) + GRU head (64 hidden) + ArcFace loss.
The GRU head's embed() output feeds into ArcFace which learns 2 class
centers (wake / not-wake) in embedding space.
"""
import torch
import torch.nn.functional as F

from ww_trainer.feats import MfccExtractor
from ww_trainer.loss import ArcFaceLoss, LossManager
from ww_trainer.model import GruClassifierHead, BaseWakeModel


def main() -> None:
    # --- Model setup ---
    ext = MfccExtractor(n_mfcc=13)
    head = GruClassifierHead(input_size=13, hidden_dim=64, linear_dim=64, device="cpu")
    model = BaseWakeModel(ext, head, device="cpu")

    # --- ArcFace loss (standalone demo) ---
    embed_dim = 64  # must match head's embed() output (= linear_dim)
    arcface = ArcFaceLoss(embed_dim=embed_dim, margin=0.5, scale=30.0)

    # Synthetic data
    wavs = torch.randn(8, 16000)
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])

    model.train()
    embeddings = model.embed(wavs)  # [8, 64]
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding norms: {embeddings.norm(dim=1).tolist()}")

    # Compute ArcFace loss directly
    loss = arcface(embeddings, labels)
    print(f"ArcFace loss: {loss.item():.4f}")

    # --- Combined training with LossManager ---
    # BCE handles classification, ArcFace enforces angular separation
    losses_cfg = [
        {"name": "bce", "weight": 1.0},
        {"name": "arcface", "weight": 0.3, "embed_dim": embed_dim, "margin": 0.5, "scale": 30.0},
    ]
    lm = LossManager(losses_cfg, device="cpu")

    # Mini training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("\nMini training loop (BCE + ArcFace):")
    for step in range(5):
        optimizer.zero_grad()
        total_loss, breakdown = lm.compute_loss(model, wavs, labels.float())
        total_loss.backward()
        optimizer.step()
        print(f"  Step {step}: total={breakdown['total']:.4f} "
              f"bce={breakdown['bce']:.4f} arcface={breakdown['arcface']:.4f}")

    # --- Show embedding separation after training ---
    model.eval()
    with torch.no_grad():
        emb = model.embed(wavs)
        emb_norm = F.normalize(emb, p=2, dim=1)
        wake_center = emb_norm[labels == 1].mean(dim=0)
        notwake_center = emb_norm[labels == 0].mean(dim=0)
        cosine_sep = F.cosine_similarity(wake_center.unsqueeze(0),
                                         notwake_center.unsqueeze(0)).item()
        print(f"\nCosine similarity between class centers: {cosine_sep:.4f}")
        print(f"(Lower = better separation; ArcFace pushes this toward -1)")


if __name__ == "__main__":
    main()
