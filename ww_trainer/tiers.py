"""Hardware tier presets for ww-trainer.

Each tier defines a complete extractor+head configuration optimised for a
specific hardware target. Pass --tier <name> to the CLI to use a preset.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TierConfig:
    """Full configuration for one hardware tier."""
    name: str
    extractor_type: str          # "mfcc" | "onnx" | "filterbank" | "sincnet" | "gammatone" | "leaf" | etc.
    head_arch: str               # "ffn" | "gru" | "cnn"
    hidden_dim: int
    n_mfcc: int = 40             # only used when extractor_type == "mfcc"
    bidirectional: bool = False  # only used when head_arch == "gru"
    gru_n_layers: int = 1
    description: str = ""
    approx_params: str = ""
    target_hardware: str = ""
    max_params: Optional[int] = None       # hard param budget (None = unlimited)
    max_size_kb: Optional[float] = None    # max int8 model size in KB


HARDWARE_TIERS: dict[str, TierConfig] = {
    "micro": TierConfig(
        name="micro",
        extractor_type="mfcc",
        head_arch="ffn",
        hidden_dim=128,
        n_mfcc=40,
        description="MFCC + FFN — smallest possible model",
        approx_params="~50K",
        target_hardware="MCU, RPi Zero",
    ),
    "small": TierConfig(
        name="small",
        extractor_type="mfcc",
        head_arch="gru",
        hidden_dim=128,
        n_mfcc=40,
        description="MFCC + GRU — good accuracy on constrained devices",
        approx_params="~200K",
        target_hardware="RPi, small SBC",
    ),
    "ssl_small": TierConfig(
        name="ssl_small",
        extractor_type="onnx",
        head_arch="ffn",
        hidden_dim=128,
        description="Pre-exported SSL ONNX featurizer + FFN — use with OnnxFeatureExtractor",
        approx_params="depends on featurizer + ~200K head",
        target_hardware="x86 / GPU server",
    ),
    "ssl_medium": TierConfig(
        name="ssl_medium",
        extractor_type="onnx",
        head_arch="gru",
        hidden_dim=256,
        bidirectional=False,
        gru_n_layers=2,
        description="Pre-exported SSL ONNX featurizer + GRU — use with OnnxFeatureExtractor",
        approx_params="depends on featurizer + ~1M head",
        target_hardware="GPU server",
    ),
"sincnet_small": TierConfig(
        name="sincnet_small",
        extractor_type="sincnet",
        head_arch="gru",
        hidden_dim=128,
        description="Learnable SincNet filterbank + GRU — trainable front-end",
        approx_params="~300K",
        target_hardware="RPi, small SBC",
    ),
    "filterbank_small": TierConfig(
        name="filterbank_small",
        extractor_type="filterbank",
        head_arch="gru",
        hidden_dim=128,
        description="Log-mel filterbank + GRU — fast and compact",
        approx_params="~200K",
        target_hardware="RPi, small SBC",
    ),
    "delta_micro": TierConfig(
        name="delta_micro",
        extractor_type="delta_mfcc",
        head_arch="ffn",
        hidden_dim=128,
        n_mfcc=13,
        description="Delta-MFCC (13 coeff x 3) + FFN — better than plain MFCC, same size",
        approx_params="~55K",
        target_hardware="MCU, RPi Zero",
    ),
    "gammatone_small": TierConfig(
        name="gammatone_small",
        extractor_type="gammatone",
        head_arch="gru",
        hidden_dim=128,
        description="Gammatone filterbank + GRU — noise-robust front-end",
        approx_params="~200K",
        target_hardware="RPi, small SBC",
    ),
"efficientnet_small": TierConfig(
        name="efficientnet_small",
        extractor_type="filterbank",
        head_arch="efficientnet",
        hidden_dim=128,
        description="Log-mel FilterBank + EfficientNet-B0 — production 2D CNN baseline",
        approx_params="~4M",
        target_hardware="RPi 4 / x86 laptop",
    ),
# --- ESP32 tiers (520 KB RAM, 4 MB flash) ---
    "esp32_nano": TierConfig(
        name="esp32_nano",
        extractor_type="mfcc",
        head_arch="ffn",
        hidden_dim=16,
        n_mfcc=13,
        description="MFCC-13 + FFN-16 — sub-1KB int8, absolute minimum",
        approx_params="~241",
        target_hardware="ESP32 (sub-1KB)",
        max_params=1024,
        max_size_kb=1.0,
    ),
    "esp32_sweet": TierConfig(
        name="esp32_sweet",
        extractor_type="mfcc",
        head_arch="ffn",
        hidden_dim=64,
        n_mfcc=13,
        description="MFCC-13 + FFN-64 — sub-10KB int8, best accuracy/size tradeoff",
        approx_params="~1K",
        target_hardware="ESP32 (sub-10KB)",
        max_params=10240,
        max_size_kb=10.0,
    ),
    "esp32_max": TierConfig(
        name="esp32_max",
        extractor_type="mfcc",
        head_arch="ffn",
        hidden_dim=128,
        n_mfcc=13,
        description="MFCC-13 + FFN-128 — sub-50KB int8, maximum ESP32 budget",
        approx_params="~2K",
        target_hardware="ESP32 (sub-50KB)",
        max_params=51200,
        max_size_kb=50.0,
    ),
    # Aliases for backwards compatibility
    "medium": TierConfig(
        name="medium",
        extractor_type="onnx",
        head_arch="ffn",
        hidden_dim=256,
        description="Alias: SSL ONNX featurizer + FFN head (medium resource)",
        approx_params="depends on featurizer + ~500K head",
        target_hardware="x86 / GPU server",
    ),
    "large": TierConfig(
        name="large",
        extractor_type="hubert",
        head_arch="gru",
        hidden_dim=256,
        bidirectional=True,
        gru_n_layers=2,
        description="Alias: HuBERT featurizer + bidirectional GRU (large, GPU)",
        approx_params="depends on featurizer + ~2M head",
        target_hardware="GPU server",
    ),
}


def list_tiers() -> str:
    """Return a formatted table of all hardware tiers."""
    rows = ["Tier      Extractor    Head   Hidden  Params          Hardware",
            "--------- ------------ ------ ------- --------------- --------------------------"]
    for t in HARDWARE_TIERS.values():
        bidir = " (bidir)" if t.bidirectional else ""
        rows.append(
            f"{t.name:<9} {t.extractor_type:<12} {t.head_arch+bidir:<15} "
            f"{t.hidden_dim:<7} {t.approx_params:<15} {t.target_hardware}"
        )
    return "\n".join(rows)


def get_tier(name: str) -> TierConfig:
    """Return a TierConfig by name, raising ValueError for unknown tiers."""
    if name not in HARDWARE_TIERS:
        raise ValueError(
            f"Unknown tier {name!r}. Available: {list(HARDWARE_TIERS)}"
        )
    return HARDWARE_TIERS[name]
