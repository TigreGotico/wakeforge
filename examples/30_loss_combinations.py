#!/usr/bin/env python3
"""LossManager: combining multiple weighted losses for training.

Shows how to configure LossManager with different loss combinations:
  1. BCE only (baseline)
  2. BCE + Focal (balanced + hard example focus)
  3. BCE + SupCon + ArcFace (classification + contrastive + angular margin)

Each loss config is a list of dicts with 'name', 'weight', and
loss-specific parameters (margin, embed_dim, temperature, etc.).
"""
import torch

from ww_trainer.loss import LossManager, FocalLoss, ArcFaceLoss, SupConLoss
from ww_trainer.feats import MfccExtractor
from ww_trainer.model import GruClassifierHead, BaseWakeModel


def main() -> None:
    # Build a simple model for demonstrating loss computation
    ext = MfccExtractor(n_mfcc=13)
    head = GruClassifierHead(input_size=13, hidden_dim=64, device="cpu")
    model = BaseWakeModel(ext, head, device="cpu")
    model.train()

    # Synthetic batch
    wavs = torch.randn(8, 16000)
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32)

    # --- Config 1: BCE only (baseline) ---
    cfg_bce = [{"name": "bce", "weight": 1.0}]
    print("Config 1: BCE only")
    print(f"  Config: {cfg_bce}")
    lm1 = LossManager(cfg_bce, device="cpu")
    loss1, details1 = lm1.compute_loss(model, wavs, labels)
    print(f"  Total loss: {loss1.item():.4f}")
    print(f"  Breakdown: {details1}\n")

    # --- Config 2: BCE + Focal (weighted) ---
    cfg_bce_focal = [
        {"name": "bce", "weight": 0.5},
        {"name": "focal", "weight": 0.5, "alpha": 0.25, "gamma": 2.0},
    ]
    print("Config 2: BCE (0.5) + Focal (0.5)")
    print(f"  Config: {cfg_bce_focal}")
    lm2 = LossManager(cfg_bce_focal, device="cpu")
    loss2, details2 = lm2.compute_loss(model, wavs, labels)
    print(f"  Total loss: {loss2.item():.4f}")
    print(f"  Breakdown: {details2}\n")

    # --- Config 3: BCE + SupCon + ArcFace (multi-objective) ---
    cfg_multi = [
        {"name": "bce", "weight": 1.0},
        {"name": "supcon", "weight": 0.3, "temperature": 0.07},
        {"name": "arcface", "weight": 0.2, "embed_dim": 64, "margin": 0.5, "scale": 30.0},
    ]
    print("Config 3: BCE (1.0) + SupCon (0.3) + ArcFace (0.2)")
    print(f"  Config: {cfg_multi}")
    lm3 = LossManager(cfg_multi, device="cpu")
    loss3, details3 = lm3.compute_loss(model, wavs, labels)
    print(f"  Total loss: {loss3.item():.4f}")
    print(f"  Breakdown: {details3}\n")

    # --- Show all available loss names ---
    all_losses = [
        "bce", "focal", "label_smoothing_bce",
        "triplet", "soft_triplet", "pair", "cn2pair",
        "contrastive", "angular", "lse",
        "arcface", "center", "ntxent", "supcon",
        "proxy_nca", "multi_similarity", "rppl",
    ]
    print(f"All supported losses: {all_losses}")

    # --- Losses requiring embed_dim ---
    print("\nLosses requiring 'embed_dim' in config: arcface, center, proxy_nca")
    print("Losses requiring 'temperature': ntxent, supcon")
    print("Losses requiring 'margin': triplet, pair, cn2pair, contrastive, angular, arcface")


if __name__ == "__main__":
    main()
