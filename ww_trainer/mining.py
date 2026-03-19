"""Hard-negative mining logic extracted from WakeWordTrainer."""
from __future__ import annotations

import logging
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
        dataset_fraction: float = 0.2,
        cache_decay: float = 0.9,
        max_cache_size: int = 10000,
        use_embedding_mining: bool = True,
        embed_top_k: int = 1000,
        wake_cache: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, float]]:
    """Mine hard negatives from nonwake samples based on model confidence and embedding similarity.

    Args:
        model: The wake word model with .forward() and optionally .embed().
        nonwakes: list of (path, label="0") pairs.
        device: torch device.
        hardness_cache: rolling dict of {path: hardness_score}; pass in and receive updated copy.
        neg_threshold: probability below which samples are confidently nonwake.
        dataset_fraction: fraction of dataset to evaluate for mining.
        cache_decay: exponential decay factor for hardness cache.
        max_cache_size: max number of cached samples.
        use_embedding_mining: whether to refine with embedding similarity.
        embed_top_k: number of top embedding-similar samples to keep.
        wake_cache: optional list of (path, "1") wake samples for embedding similarity.

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

    sample_size = max(1, int(len(nonwakes) * dataset_fraction))
    subset = random.sample(nonwakes, min(sample_size, len(nonwakes)))

    loader = DataLoader(
        AudioDataset(subset, aug_prob=0.0),
        batch_size=128,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, device)
    )

    new_scores: Dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for wavs, _, paths in tqdm(loader, desc="Mining negatives", leave=False):
            logits = model(wavs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            for path, p in zip(paths, probs):
                old = hardness_cache.get(path, 0.0)
                hardness = max(0.0, p - neg_threshold)
                new_scores[path] = cache_decay * old + (1 - cache_decay) * hardness

    hardness_cache.update(new_scores)
    if len(hardness_cache) > max_cache_size:
        sorted_items = sorted(hardness_cache.items(), key=lambda kv: kv[1], reverse=True)
        hardness_cache = dict(sorted_items[:max_cache_size])

    sorted_cache = sorted(hardness_cache.items(), key=lambda kv: kv[1], reverse=True)
    cutoff = int(len(sorted_cache) * dataset_fraction)
    hard_paths = [p for p, _ in sorted_cache[:cutoff]]
    easy_paths = [p for p, _ in sorted_cache[cutoff:]]

    hard_negatives = [(p, "0") for p in hard_paths]
    easy_negatives = [(p, "0") for p in easy_paths]

    if use_embedding_mining and hasattr(model, "embed"):
        wakes_subset = []
        if wake_cache:
            try:
                wakes_subset = random.sample(wake_cache, min(len(wake_cache), 200))
            except Exception:
                wakes_subset = []

        if wakes_subset:
            wake_loader = DataLoader(AudioDataset(wakes_subset, aug_prob=0), batch_size=128, shuffle=True,
                                     collate_fn=lambda b: collate_fn(b, device))
            wake_embeds = []
            with torch.no_grad():
                for wavs, _, _ in wake_loader:
                    wake_embeds.append(model.embed(wavs))
            wake_proto = torch.cat(wake_embeds, dim=0).mean(0, keepdim=True)

            emb_loader = DataLoader(AudioDataset(subset, aug_prob=0), batch_size=128, shuffle=True,
                                    collate_fn=lambda b: collate_fn(b, device))
            emb_sims: Dict[str, float] = {}
            with torch.no_grad():
                for wavs, _, paths in emb_loader:
                    emb = model.embed(wavs)
                    sim = torch.nn.functional.cosine_similarity(emb, wake_proto)
                    for path, s in zip(paths, sim):
                        emb_sims[path] = float(s.cpu())

            for path, s in emb_sims.items():
                hardness_cache[path] = 0.5 * hardness_cache.get(path, 0.0) + 0.5 * s

            top_embed = sorted(hardness_cache.items(), key=lambda kv: kv[1], reverse=True)[:embed_top_k]
            hard_negatives = [(p, "0") for p, _ in top_embed]

    logger.info("Mined %d hard and %d easy negatives (subset=%d)", len(hard_negatives), len(easy_negatives), sample_size)
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
