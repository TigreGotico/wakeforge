"""Pretrained ONNX featurizers resolved by name from the Hugging Face Hub.

Each pretrained featurizer is an ONNX feature extractor published in its own
Hub model repository under ``TigreGotico`` (the "ONNX feature extractors"
collection), with the same layout: ``wakehubert.onnx`` (float32),
``wakehubert_int8.onnx`` and a ``config.json`` that states ``feature_dim``,
``output.hop_samples``, ``output.frame_rate_hz``, ``streaming`` and
``license``. Selecting one by name downloads it into the shared Hugging Face
cache and wraps it in :class:`~ww_trainer.feats.OnnxFeatureExtractor`::

    from ww_trainer.pretrained import load_pretrained_featurizer
    feat = load_pretrained_featurizer("wakehubert")   # alias of wakehubert-tiny
    feat(wav_tensor).shape                           # [1, samples // 320, 128]

Names are the repository names (``wakehubert-mel-tcn-wide``, ...), each with
an ``-int8`` variant. From the CLI, ``--featurizer-type <name>`` (or
``--featurizer <name>``) does the same; ``--tier wakehubert`` selects the
default one.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from huggingface_hub import hf_hub_download

from ww_trainer.feats import OnnxFeatureExtractor

logger = logging.getLogger(__name__)

HF_COLLECTION = ("https://huggingface.co/collections/TigreGotico/"
                 "onnx-feature-extractors-690a3d2bced386cc3c338c77")


@dataclass(frozen=True)
class PretrainedFeaturizer:
    """A named ONNX featurizer published in a Hub model repository.

    Args:
        repo_id: Hub model repository holding the ONNX files and ``config.json``.
        variant: Key into the repository config's ``files`` mapping
            (``"float32"`` or ``"int8"``).
        license: Licence as published; the downloaded ``config.json`` is the
            authority and is what :func:`load_pretrained_featurizer` checks.
        description: One line on what the extractor is.
        context_samples: Past audio that determines one output frame, measured
            on the published ONNX, or ``None`` when frames cannot be recomputed
            exactly from a bounded window (bidirectional or recurrent models,
            or an int8 graph whose frames depend on later audio). A
            ``receptive_field_samples`` entry in the config takes precedence.
        revision: Default revision (branch, tag or commit) to download.
    """
    repo_id: str
    variant: str = "float32"
    license: str = ""
    description: str = ""
    context_samples: Optional[int] = None
    revision: Optional[str] = None


_APACHE, _CC_BY_SA, _CC_BY_NC_SA = "apache-2.0", "cc-by-sa-3.0", "cc-by-nc-sa-4.0"

# Streaming context in 320-sample frames, (float32, int8), measured on each
# published ONNX: the smallest of 125, 150, 175, 200, 250, 300, 400 frames for
# which features streamed in 0.1 s chunks equal the offline features (within
# 1e-5 of their range) over 40 s of test audio. None: no bounded context
# reproduces them (recurrent or bidirectional models, and the int8 attention
# graph, whose frames also depend on later audio).
_REPOS = {
    "wakehubert-tiny": (_APACHE, "HuBERT-base student, causal TCN (default)", 125, 125),
    "wakehubert-mel-tcn-deep": (_APACHE, "HuBERT-base student, 16-block causal TCN", 250, 250),
    "wakehubert-mel-tcn-wide": (_APACHE, "HuBERT-base student, wide causal TCN, 256-d", 200, 200),
    "wakehubert-mel-gru": (_APACHE, "HuBERT-base student, causal GRU", None, None),
    "wakehubert-mel-bigru": (_APACHE, "HuBERT-base student, bidirectional GRU (offline)", None, None),
    "wakehubert-mel-attn": (_APACHE, "HuBERT-base student, windowed causal attention", 200, None),
    "wakehubert-mixconv": (_APACHE, "HuBERT-base student, causal mixed-kernel TCN", 250, 250),
    "wakewav-mel-tcn": (_CC_BY_SA, "WavLM-base+ student, causal TCN", 125, 125),
    "wakewav-mel-tcn-wide": (_CC_BY_SA, "WavLM-base+ student, wide causal TCN, 256-d", 200, 200),
    "wakewav-mel-gru": (_CC_BY_SA, "WavLM-base+ student, causal GRU", None, None),
    "wakewav-mel-bigru": (_CC_BY_SA, "WavLM-base+ student, bidirectional GRU (offline)", None, None),
    "wakexeus-mel-tcn": (_CC_BY_NC_SA, "XEUS student, causal TCN", 125, 125),
    "wakexeus-mel-tcn-wide": (_CC_BY_NC_SA, "XEUS student, wide causal TCN, 256-d", 200, 200),
    "wakexeus-mel-gru": (_CC_BY_NC_SA, "XEUS student, causal GRU", None, None),
    "wakexeus-mel-bigru": (_CC_BY_NC_SA, "XEUS student, bidirectional GRU (offline)", None, None),
}


def _entry(repo: str, variant: str, frames: Optional[int]) -> PretrainedFeaturizer:
    licence, description, _, _ = _REPOS[repo]
    return PretrainedFeaturizer(f"TigreGotico/{repo}", variant, licence, description,
                                None if frames is None else frames * 320)


PRETRAINED_FEATURIZERS: Dict[str, PretrainedFeaturizer] = {}
for _repo, (_, _, _fp32, _int8) in _REPOS.items():
    PRETRAINED_FEATURIZERS[_repo] = _entry(_repo, "float32", _fp32)
    PRETRAINED_FEATURIZERS[f"{_repo}-int8"] = _entry(_repo, "int8", _int8)
PRETRAINED_FEATURIZERS["wakehubert"] = PRETRAINED_FEATURIZERS["wakehubert-tiny"]
PRETRAINED_FEATURIZERS["wakehubert-int8"] = PRETRAINED_FEATURIZERS["wakehubert-tiny-int8"]


def is_non_commercial(licence: str) -> bool:
    """True for Creative Commons NonCommercial licences (``*-nc-*``)."""
    return "-nc" in licence.lower()


def resolve_pretrained(name: str, revision: Optional[str] = None) -> tuple[str, dict]:
    """Download a pretrained featurizer and return ``(onnx_path, config)``.

    Args:
        name: Key in :data:`PRETRAINED_FEATURIZERS`.
        revision: Revision to pin; defaults to the entry's own revision.

    Returns:
        Local path of the ONNX file and the parsed ``config.json``. The config
        carries the commit it was downloaded at under ``"_revision"`` when the
        file sits in a Hugging Face cache snapshot, else the requested revision.
    """
    if name not in PRETRAINED_FEATURIZERS:
        raise ValueError(
            f"Unknown pretrained featurizer {name!r}. "
            f"Available: {', '.join(sorted(PRETRAINED_FEATURIZERS))}"
        )
    entry = PRETRAINED_FEATURIZERS[name]
    rev = revision or entry.revision
    config_path = Path(hf_hub_download(entry.repo_id, "config.json", revision=rev))
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    snapshot = config_path.parent
    config["_revision"] = snapshot.name if snapshot.parent.name == "snapshots" else (rev or "")
    onnx_path = hf_hub_download(entry.repo_id, config["files"][entry.variant], revision=rev)
    return onnx_path, config


def load_pretrained_featurizer(name: str, revision: Optional[str] = None,
                               sample_rate: int = 16000,
                               device: str = "auto") -> OnnxFeatureExtractor:
    """Build an :class:`~ww_trainer.feats.OnnxFeatureExtractor` for a named featurizer.

    ``feature_dim``, hop, frame rate, streaming and licence come from the
    repository config. The extractor carries ``hop_samples``, ``frame_rate_hz``,
    ``streaming`` and ``context_samples``, which
    :meth:`~ww_trainer.inference.OnnxStreamingWakeWord.from_extractor` needs.
    A warning is logged when the licence forbids commercial use.
    """
    onnx_path, config = resolve_pretrained(name, revision)
    entry = PRETRAINED_FEATURIZERS[name]
    licence = config.get("license", entry.license)
    if is_non_commercial(licence):
        logger.warning("Pretrained featurizer %r is licensed %s: non-commercial use only.",
                       name, licence)
    output = config.get("output", {})
    hop = int(output.get("hop_samples", 160))
    context = entry.context_samples
    if context is not None and "receptive_field_samples" in config:
        context = int(config["receptive_field_samples"])
    return OnnxFeatureExtractor(
        onnx_path, sample_rate, device,
        feature_dim=int(config["feature_dim"]),
        hop_samples=hop,
        frame_rate_hz=float(output.get("frame_rate_hz", sample_rate / hop)),
        context_samples=context or 0,
        streaming=bool(config.get("streaming", False)) and context is not None,
        license=licence,
        pretrained_name=name,
        revision=config["_revision"],
    )


def featurizer_metadata(extractor) -> Dict[str, str]:
    """ONNX metadata that identifies a pretrained featurizer, or ``{}`` for any other.

    Exported heads carry ``pretrained_featurizer`` (registry name) and ``featurizer_revision`` so
    :class:`~ww_trainer.inference.OnnxWakeWordInferencer` and
    :class:`~ww_trainer.inference.OnnxStreamingWakeWord` rebuild the featurizer
    from the registry, plus ``feature_dim``, ``hop_samples`` and ``frame_rate_hz``.
    """
    if not isinstance(extractor, OnnxFeatureExtractor) or not extractor.pretrained_name:
        return {}
    return {
        "pretrained_featurizer": extractor.pretrained_name,
        "featurizer_revision": extractor.revision,
        "feature_dim": str(extractor.feature_dim),
        "hop_samples": str(extractor.hop_samples),
        "frame_rate_hz": str(extractor.frame_rate_hz),
    }
