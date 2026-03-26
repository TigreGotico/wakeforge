"""Hard-negative mining logic extracted from WakeWordTrainer."""
from __future__ import annotations

import logging
import math
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ww_trainer.dataset import AudioDataset, collate_fn

logger = logging.getLogger(__name__)


def mine_hard_negatives(
        model,
        nonwakes: List[Tuple[str, str]],
        device,
        hardness_cache: Optional[Dict[str, float]] = None,
        neg_threshold: float = 0.5,
        dataset_fraction: float = 0.33,
        cache_decay: float = 0.7,          # was 0.9 — decay faster so stale scores don't linger
        max_cache_size: int = 10000,
        use_embedding_mining: bool = True,
        wake_cache: Optional[List[Tuple[str, str]]] = None,
        feature_cache=None,
        rppl_proto: Optional[torch.Tensor] = None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, float]]:
    """Mine hard negatives from nonwake samples based on model confidence and embedding similarity.

    Hard negatives are samples where the model's confidence exceeds neg_threshold —
    i.e. genuine false positives the model is currently confused about.

    The hardness score is the raw model probability (above threshold = hard).
    Embedding similarity to the wake prototype is used as a secondary additive
    signal, not a multiplicative gate, so confidence-hard samples are never
    discarded just because they sound dissimilar.

    Args:
        model: The wake word model with .forward() and optionally .embed().
        nonwakes: list of (path, label="0") pairs.
        device: torch device.
        hardness_cache: rolling dict of {path: hardness_score}.
        neg_threshold: confidence above which a sample is considered a false positive.
        dataset_fraction: fraction of nonwake pool to scan each epoch.
        cache_decay: EMA decay for cached scores (lower = faster forgetting of stale scores).
        max_cache_size: max entries kept in the cache.
        use_embedding_mining: blend embedding similarity into hardness.
        wake_cache: wake samples used to build the prototype embedding (used only
            when rppl_proto is not provided).
        feature_cache: SharedWaveformCache instance.
        rppl_proto: pre-computed EMA wake prototype from RobustProtoDiversityLoss
            (shape ``(D,)`` or ``(1, D)``).  When provided, skips the batch mean
            computation and uses this stabilised prototype directly — this is the
            RPPL→mining feedback loop that makes confusable-negative targeting
            more accurate as training progresses.

    Returns:
        (hard_negatives, easy_negatives, updated_hardness_cache)
    """
    if hardness_cache is None:
        hardness_cache = {}

    if dataset_fraction <= 0.0:
        logger.info("[HardNegMining] Disabled — using full nonwake dataset")
        return [], nonwakes, hardness_cache

    if len(nonwakes) == 0:
        return [], [], hardness_cache

    has_embed = use_embedding_mining and hasattr(model, "embed")

    # Build wake prototype once before scanning negatives.
    # Preference order:
    #   1. rppl_proto — EMA prototype from RobustProtoDiversityLoss (most stable)
    #   2. batch mean over wake_cache (fallback when RPPL is not in use)
    wake_proto = None
    if has_embed:
        if rppl_proto is not None:
            # Normalise and reshape to (1, D) for cosine_similarity broadcast
            p = rppl_proto.detach().to(device).float()
            if p.dim() == 1:
                p = p.unsqueeze(0)
            norm = p.norm(dim=1, keepdim=True).clamp(min=1e-8)
            wake_proto = p / norm
            logger.debug("[Mining] Using RPPL EMA prototype for embedding scoring")
        elif wake_cache:
            wakes_subset = random.sample(wake_cache, min(len(wake_cache), 500))
            wake_loader = DataLoader(
                AudioDataset(wakes_subset, aug_prob=0, feature_cache=feature_cache),
                batch_size=128, shuffle=False,
                collate_fn=lambda b: collate_fn(b, device),
            )
            wake_embeds = []
            model.eval()
            with torch.no_grad():
                for wavs, _, _ in wake_loader:
                    wake_embeds.append(model.embed(wavs))
            if wake_embeds:
                wake_proto = torch.cat(wake_embeds, dim=0).mean(0, keepdim=True)

    sample_size = max(1, int(len(nonwakes) * dataset_fraction))
    subset = random.sample(nonwakes, min(sample_size, len(nonwakes)))

    loader = DataLoader(
        AudioDataset(subset, aug_prob=0.0, feature_cache=feature_cache),
        batch_size=128, shuffle=False,
        collate_fn=lambda b: collate_fn(b, device),
    )

    new_conf: Dict[str, float] = {}   # raw confidence scores from this pass
    emb_sims: Dict[str, float] = {}   # cosine similarity to wake prototype
    model.eval()
    with torch.no_grad():
        for wavs, _, paths in tqdm(loader, desc="Mining negatives", leave=False):
            logits = model(wavs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            for path, p in zip(paths, probs):
                new_conf[path] = float(p)

            # Collect embedding similarities in the same forward pass
            if has_embed and wake_proto is not None:
                emb = model.embed(wavs)
                sim = torch.nn.functional.cosine_similarity(emb, wake_proto)
                for path, s in zip(paths, sim.cpu().tolist()):
                    emb_sims[path] = float(s)

    # Update cache: EMA blend of old score and new confidence
    for path, conf in new_conf.items():
        old = hardness_cache.get(path, 0.0)
        hardness_cache[path] = cache_decay * old + (1 - cache_decay) * conf

    # Additive embedding boost: reward samples near the wake prototype.
    # Uses addition (not multiplication) so confidence-hard samples with low
    # embedding similarity are never penalised.
    if emb_sims:
        emb_weight = 0.2   # small additive bonus — confidence is still primary signal
        for path, s in emb_sims.items():
            hardness_cache[path] = hardness_cache.get(path, 0.0) + emb_weight * max(0.0, s)

    # Evict lowest-scoring entries when cache is full
    if len(hardness_cache) > max_cache_size:
        sorted_items = sorted(hardness_cache.items(), key=lambda kv: kv[1], reverse=True)
        hardness_cache = dict(sorted_items[:max_cache_size])

    sorted_cache = sorted(hardness_cache.items(), key=lambda kv: kv[1], reverse=True)

    # Hard = actually above neg_threshold in the current pass.
    # Easy = everything else in the cache (below threshold but still scored).
    above_threshold = {p for p, c in new_conf.items() if c >= neg_threshold}
    hard_negatives = [(p, "0") for p, _ in sorted_cache if p in above_threshold]
    easy_negatives = [(p, "0") for p, _ in sorted_cache if p not in above_threshold]

    logger.info(
        "Mined %d hard (conf≥%.2f) and %d easy negatives (scanned=%d/%d)",
        len(hard_negatives), neg_threshold, len(easy_negatives), sample_size, len(nonwakes),
    )
    return hard_negatives, easy_negatives, hardness_cache


def save_mining_cache(cache: dict, path: str) -> None:
    """Persist the hard-negative cache to disk."""
    import torch
    torch.save(cache, path)


def load_mining_cache(path: str) -> dict:
    """Load a persisted hard-negative cache from disk."""
    import torch
    import os
    if not os.path.exists(path):
        return {}
    return torch.load(path, map_location="cpu")
