#!/usr/bin/env python3
"""Hardware tier presets — pre-configured architectures for target devices.

ww-trainer includes predefined tier configurations mapping to specific
hardware targets. Each tier specifies the extractor, head architecture,
and hyperparameters optimized for that class of device.
"""
import torch

from ww_trainer.tiers import HARDWARE_TIERS, TierConfig
from ww_trainer.feats import MfccExtractor, FilterbankExtractor, SincNetExtractor, GammatoneExtractor
from ww_trainer.model import FfnClassifierHead, GruClassifierHead, CnnClassifierHead, BaseWakeModel

# Map tier extractor_type to extractor constructors
_EXTRACTOR_MAP = {
    "mfcc": lambda cfg, dev: MfccExtractor(sr=16000, n_mfcc=cfg.n_mfcc),
    "filterbank": lambda cfg, dev: FilterbankExtractor(sr=16000, n_mels=cfg.n_mfcc),
    "sincnet": lambda cfg, dev: SincNetExtractor(sr=16000, n_filters=cfg.n_mfcc),
    "gammatone": lambda cfg, dev: GammatoneExtractor(sr=16000, n_filters=cfg.n_mfcc),
}

_HEAD_MAP = {
    "ffn": lambda dim, cfg, dev: FfnClassifierHead(input_size=dim, hidden_dim=cfg.hidden_dim, device=dev),
    "gru": lambda dim, cfg, dev: GruClassifierHead(
        input_size=dim, hidden_dim=cfg.hidden_dim,
        bidirectional=cfg.bidirectional, gru_n_layers=cfg.gru_n_layers, device=dev),
    "cnn": lambda dim, cfg, dev: CnnClassifierHead(input_size=dim, conv_dim=cfg.hidden_dim, device=dev),
}


def build_from_tier(tier_name: str, device: str = "cpu") -> BaseWakeModel:
    """Build a complete model from a tier preset."""
    cfg: TierConfig = HARDWARE_TIERS[tier_name]
    ext_factory = _EXTRACTOR_MAP.get(cfg.extractor_type)
    if ext_factory is None:
        raise ValueError(f"Tier '{tier_name}' requires extractor '{cfg.extractor_type}' "
                         "which needs a pre-exported ONNX file.")
    extractor = ext_factory(cfg, device)
    head = _HEAD_MAP[cfg.head_arch](extractor.feature_dim, cfg, device)
    return BaseWakeModel(extractor, head, device=device)


def main() -> None:
    device = "cpu"

    print(f"{'Tier':<20} {'Extractor':<14} {'Head':<8} {'Params':>10} {'Target'}")
    print("-" * 75)

    for name, cfg in HARDWARE_TIERS.items():
        if cfg.extractor_type in _EXTRACTOR_MAP:
            model = build_from_tier(name, device)
            params = sum(p.numel() for p in model.parameters())

            # Quick forward pass to verify
            audio = torch.randn(1, 16000)
            with torch.no_grad():
                prob = torch.sigmoid(model(audio)).item()

            print(f"{name:<20} {cfg.extractor_type:<14} {cfg.head_arch:<8} "
                  f"{params:>10,} {cfg.target_hardware}")
        else:
            print(f"{name:<20} {cfg.extractor_type:<14} {cfg.head_arch:<8} "
                  f"{'(needs ONNX)':>10} {cfg.target_hardware}")


if __name__ == "__main__":
    main()
