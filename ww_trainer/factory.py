"""Model and feature extractor factory functions.

Provides ``create_model`` and the registries (``EXTRACTOR_REGISTRY``,
``HEAD_REGISTRY``) used to build wake-word models from string identifiers.

External packages can register new extractors/heads via:
1. ``register_extractor("name", MyExtractor)`` / ``register_head(...)``
2. Entry points in pyproject.toml::

    [project.entry-points."ww_trainer.extractors"]
    my_extractor = "my_package:MyExtractor"

    [project.entry-points."ww_trainer.heads"]
    my_head = "my_package:MyHead"
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type

logger = logging.getLogger(__name__)

from ww_trainer.feats import (
    OnnxFeatureExtractor,
    MfccExtractor,
    FilterbankExtractor,
    SincNetExtractor,
    HubertExtractor,
    Wav2Vec2Extractor,
    Wav2Vec2BertExtractor,
    DeltaExtractor,
    GammatoneExtractor,
    LEAFExtractor,
    PLPExtractor,
    PNCCExtractor,
    CQTExtractor,
    VoiceActivityExtractor,
    PitchExtractor,
    SNRAwareExtractor,
)
from ww_trainer.model import (
    FfnClassifierHead, GruClassifierHead, CnnClassifierHead, BCResNetHead,
    TCResNetHead, DSCNNHead, MatchboxNetHead, Res15Head,
    KWTHead, ConformerHead, CRNNHead,
    BaseWakeModel,
)

EXTRACTOR_REGISTRY: Dict[str, type] = {
    "onnx": OnnxFeatureExtractor,
    "mfcc": MfccExtractor,
    "filterbank": FilterbankExtractor,
    "sincnet": SincNetExtractor,
    "hubert": HubertExtractor,
    "wav2vec2": Wav2Vec2Extractor,
    "wav2vec2bert": Wav2Vec2BertExtractor,
    "delta_mfcc": None,           # special: MfccExtractor wrapped in DeltaExtractor
    "gammatone": GammatoneExtractor,
    "delta_filterbank": None,     # special: FilterbankExtractor wrapped in DeltaExtractor
    "leaf": LEAFExtractor,
    "plp": PLPExtractor,
    "pncc": PNCCExtractor,
    "cqt": CQTExtractor,
}

HEAD_REGISTRY: Dict[str, Tuple[Type, Set[str]]] = {
    "ffn": (FfnClassifierHead, {"hidden_dim", "dropout"}),
    "gru": (GruClassifierHead, {"hidden_dim", "dropout", "bidirectional", "gru_n_layers"}),
    "cnn": (CnnClassifierHead, {"conv_dim", "linear_dim", "kernel_size", "stride"}),
    "bcresnet": (BCResNetHead, {"tau"}),
    "tcresnet": (TCResNetHead, {"variant", "channels", "kernel_size"}),
    "dscnn": (DSCNNHead, {"size"}),
    "matchboxnet": (MatchboxNetHead, {"B", "R", "C", "kernel_sizes"}),
    "res15": (Res15Head, {"channels"}),
    "kwt": (KWTHead, {"patch_len", "d_model", "n_heads", "n_layers", "dim_ff", "dropout"}),
    "conformer": (ConformerHead, {"d_model", "n_heads", "n_layers", "conv_kernel", "dim_ff", "dropout"}),
    "crnn": (CRNNHead, {"conv_channels", "gru_hidden", "gru_layers", "dropout"}),
}


def register_extractor(name: str, cls: type) -> None:
    """Register a custom feature extractor class.

    Args:
        name: String key for CLI/config use.
        cls: BaseExtractor subclass.

    Example::

        from ww_trainer.factory import register_extractor
        from my_package import MyExtractor
        register_extractor("my_ext", MyExtractor)
    """
    EXTRACTOR_REGISTRY[name] = cls
    logger.info("Registered extractor: %s → %s", name, cls.__name__)


def register_head(name: str, cls: type, valid_kwargs: Optional[set] = None) -> None:
    """Register a custom classifier head class.

    Args:
        name: String key for CLI/config use.
        cls: ClassifierHead subclass.
        valid_kwargs: Set of keyword argument names the head accepts
                      (used for filtering kwargs in create_model).

    Example::

        from ww_trainer.factory import register_head
        from my_package import MyHead
        register_head("my_head", MyHead, {"hidden_dim", "dropout"})
    """
    HEAD_REGISTRY[name] = (cls, valid_kwargs or set())
    logger.info("Registered head: %s → %s", name, cls.__name__)


def load_entry_point_plugins() -> None:
    """Discover and register plugins via setuptools entry points.

    Scans ``ww_trainer.extractors`` and ``ww_trainer.heads`` entry point
    groups.  Each entry point should resolve to a class.

    Called automatically on first ``create_model()`` invocation.
    """
    try:
        from importlib.metadata import entry_points as _ep_fn
    except ImportError:
        return

    def _get_eps(group: str) -> list:
        """Compat wrapper for entry_points across Python versions."""
        try:
            return list(_ep_fn(group=group))
        except TypeError:
            # Python < 3.12: entry_points() returns dict-like
            return list(_ep_fn().get(group, []))

    for ep in _get_eps("ww_trainer.extractors"):
        try:
            cls = ep.load()
            register_extractor(ep.name, cls)
        except Exception as exc:
            logger.warning("Failed to load extractor plugin %s: %s", ep.name, exc)

    for ep in _get_eps("ww_trainer.heads"):
        try:
            cls = ep.load()
            register_head(ep.name, cls)
        except Exception as exc:
            logger.warning("Failed to load head plugin %s: %s", ep.name, exc)


_plugins_loaded = False


def create_model(arch_name: str, featurizer: str, feature_dim: int = None,
                 featurizer_type: str = "onnx", sample_rate: int = 16000,
                 device: str = "auto",
                 shared_extractor=None, **kwargs: Any) -> BaseWakeModel:
    """Build a ``BaseWakeModel`` from architecture and featurizer names.

    Args:
        arch_name: Classifier head key (must exist in ``HEAD_REGISTRY``).
        featurizer: Path or identifier for the feature extractor.
        feature_dim: Override for extractor output dimension.
        featurizer_type: Key into ``EXTRACTOR_REGISTRY``.
        sample_rate: Audio sample rate.
        device: Target device string.
        shared_extractor: Pre-built extractor instance (bypasses registry).
        **kwargs: Forwarded to extractor and head constructors.

    Returns:
        Fully assembled ``BaseWakeModel``.
    """
    global _plugins_loaded
    if not _plugins_loaded:
        load_entry_point_plugins()
        _plugins_loaded = True

    # --- Build extractor ---
    if shared_extractor is not None:
        extractor = shared_extractor
    else:
        n_feat = kwargs.get("n_mfcc", kwargs.get("n_mels", kwargs.get("n_filters", 40)))
        _EXTRACTOR_BUILDERS = {
            "onnx": lambda: OnnxFeatureExtractor(featurizer, sample_rate, device),
            "mfcc": lambda: MfccExtractor(sr=sample_rate, n_mfcc=n_feat),
            "filterbank": lambda: FilterbankExtractor(sr=sample_rate, n_mels=n_feat),
            "sincnet": lambda: SincNetExtractor(sr=sample_rate, n_filters=n_feat),
            "gammatone": lambda: GammatoneExtractor(sr=sample_rate, n_filters=n_feat),
            "leaf": lambda: LEAFExtractor(sr=sample_rate, n_filters=n_feat),
            "plp": lambda: PLPExtractor(sr=sample_rate, n_plp=kwargs.get("n_plp", 13)),
            "pncc": lambda: PNCCExtractor(sr=sample_rate, n_pncc=kwargs.get("n_pncc", 13)),
            "cqt": lambda: CQTExtractor(sr=sample_rate),
            "hubert": lambda: HubertExtractor(featurizer, sample_rate, device),
            "wav2vec2": lambda: Wav2Vec2Extractor(featurizer, sample_rate, device),
            "wav2vec2bert": lambda: Wav2Vec2BertExtractor(
                featurizer or "facebook/w2v-bert-2.0", sample_rate, device),
            "delta_mfcc": lambda: DeltaExtractor(MfccExtractor(sr=sample_rate, n_mfcc=n_feat)),
            "delta_filterbank": lambda: DeltaExtractor(
                FilterbankExtractor(sr=sample_rate, n_mels=n_feat)),
        }
        builder = _EXTRACTOR_BUILDERS.get(featurizer_type)
        if builder is None:
            raise ValueError(
                f"Unknown featurizer_type: {featurizer_type!r}. "
                f"Choose from: {', '.join(sorted(_EXTRACTOR_BUILDERS))}"
            )
        extractor = builder()

        # Optional enrichment wrappers
        if kwargs.get("use_vad", False):
            extractor = VoiceActivityExtractor(extractor)
        if kwargs.get("use_pitch", False):
            extractor = PitchExtractor(extractor)
        if kwargs.get("use_snr", False):
            extractor = SNRAwareExtractor(extractor)

    if feature_dim is None:
        feature_dim = extractor.feature_dim

    # --- Build classifier head ---
    entry = HEAD_REGISTRY.get(arch_name)
    if entry is None:
        raise ValueError(
            f"Unknown classifier architecture: {arch_name!r}. "
            f"Choose from: {', '.join(sorted(HEAD_REGISTRY))}"
        )
    cls, valid_args = entry
    inst_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
    clf = cls(device=device, sample_rate=sample_rate, input_size=feature_dim, **inst_kwargs)

    return BaseWakeModel(
        feature_extractor=extractor,
        classifier=clf,
        sample_rate=sample_rate,
        device=device
    )
