import random
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F


# -------------------- Utility metric functions --------------------

def compute_metric_statistics(labels: torch.Tensor, embeds: torch.Tensor, margin: float = 1.0) -> Tuple[float, float, float]:
    """
    Calculates key scalar statistics for embedding monitoring.

    Returns: (mean_ap_dist, mean_an_dist, violation_fraction)
    """
    with torch.no_grad():
        device = embeds.device
        dist = pairwise_distance(embeds)
        B = len(labels)

        # 1. Collect all valid Anchor-Positive (AP) and Anchor-Negative (AN) distances
        ap_dists = []
        an_dists = []

        anchor_mask = labels == 1  # Only use wake word samples as anchors

        for i in torch.arange(B, device=device)[anchor_mask].tolist():
            # Positive mask: same label, not self
            pos_mask = (labels == labels[i]) & (torch.arange(B, device=device) != i)
            # Negative mask: different label
            neg_mask = (labels != labels[i])

            if pos_mask.any() and neg_mask.any():
                # Store all AP distances for this anchor
                ap_dists.append(dist[i][pos_mask])
                # Store all AN distances for this anchor
                an_dists.append(dist[i][neg_mask])

        if not ap_dists:
            return 0.0, 0.0, 0.0

        # Concatenate all distances
        all_ap = torch.cat(ap_dists)
        all_an = torch.cat(an_dists)

        # 2. Calculate Mean Distances
        mean_ap_dist = all_ap.mean().item()
        mean_an_dist = all_an.mean().item()

        # 3. Calculate Violation Fraction (uses existing efficient function)
        _, _, violation_frac = triplet_violation_fraction(labels, embeds, margin)

        return mean_ap_dist, mean_an_dist, violation_frac


def sample_triplets(labels: torch.Tensor, embeds: torch.Tensor, margin: float,
                    mining_type: str = "semihard", max_triplets: int = 100) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], float]:
    """
    Samples triplets (anchor, positive, negative) for metric learning.

    Args:
        labels: Ground truth labels (B,).
        embeds: Embeddings (B, D).
        margin: Triplet margin.
        mining_type: Strategy to use: 'semihard' (default), 'hard', or 'random'.
        max_triplets: Maximum number of triplets to sample.

    Returns:
        (anchor_indices, positive_indices, negative_indices, violation_fraction)
    """
    device = embeds.device
    dist = pairwise_distance(embeds)  # (B, B)
    B = len(labels)

    triplets: List[Tuple[int, int, int]] = []
    violations = 0
    total_valid = 0

    # Determine which indices to iterate over as anchors
    # Only wake word samples (label 1) can serve as anchors for A/P/N triplets
    anchor_mask = labels == 1
    anchor_indices = torch.arange(B, device=device)[anchor_mask].tolist()

    if not anchor_indices:
        return None, None, None, 0.0

    # Ensure anchors are randomly processed
    random.shuffle(anchor_indices)

    for i in anchor_indices:
        pos_mask = (labels == labels[i]) & (torch.arange(B, device=device) != i)
        neg_mask = (labels != labels[i])

        # Must have at least one positive and one negative
        if not pos_mask.any() or not neg_mask.any():
            continue

        pos_dists = dist[i][pos_mask]
        neg_dists = dist[i][neg_mask]
        total_valid += len(pos_dists) * len(neg_dists)

        # --- 1. Hard Positive Selection (Always use the hardest) ---
        hardest_pos_idx = pos_dists.argmax()
        hardest_pos = pos_dists[hardest_pos_idx]
        p_indices = torch.arange(B, device=device)[pos_mask]
        p_idx = p_indices[hardest_pos_idx]

        # --- 2. Negative Selection Strategy ---
        n_idx = torch.tensor(-1, device=device)  # Initialize for assignment
        neg_indices = torch.arange(B, device=device)[neg_mask]

        if mining_type == "random":
            # Simple random negative selection
            rand_idx = random.choice(range(len(neg_dists)))
            n_idx = neg_indices[rand_idx]

        elif mining_type == "hard":
            # Hardest negative (closest to anchor)
            hardest_neg_idx = neg_dists.argmin()
            n_idx = neg_indices[hardest_neg_idx]

        elif mining_type == "semihard":
            # Semi-hard negative (Further than hardest positive, but violates margin)
            # Find negatives D(a,n) such that D(a,p_hardest) < D(a,n) < D(a,p_hardest) + margin

            # Step A: Negatives further than hardest positive
            semi_hard_mask = neg_dists > hardest_pos

            if semi_hard_mask.any():
                # Step B: Select the closest among the semi-hard group
                semi_hard_neg_idx = torch.argmin(neg_dists[semi_hard_mask])
                neg_indices_semi = neg_indices[semi_hard_mask]
                n_idx = neg_indices_semi[semi_hard_neg_idx]
            else:
                # Fallback to Hard Negative mining if no semi-hard are found
                hardest_neg_idx = neg_dists.argmin()
                n_idx = neg_indices[hardest_neg_idx]

        else:
            raise ValueError(f"Unknown mining_type: {mining_type}")

        # --- 3. Triplet Collection (Only collect if violation exists) ---
        if dist[i, p_idx] + margin > dist[i, n_idx]:
            violations += 1
            triplets.append((i, p_idx.item(), n_idx.item()))

        if len(triplets) >= max_triplets:
            break

    violation_frac = violations / max(1, total_valid)
    if not triplets:
        return None, None, None, violation_frac

    a, p, n = zip(*triplets)
    return torch.tensor(a, device=device), torch.tensor(p, device=device), torch.tensor(n,
                                                                                        device=device), violation_frac

def pairwise_distance(embeddings: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)
    distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)
    distances = torch.clamp(distances, min=0.0)
    return torch.sqrt(distances + eps)


def pairwise_cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise cosine similarity matrix.
    Assumes embeddings are already L2-normalized.
    """
    # If not normalized, uncomment this:
    # embeds_normalized = F.normalize(embeddings, p=2, dim=1)

    # Cosine similarity is just the dot product of normalized vectors
    return torch.matmul(embeddings, embeddings.t())


def triplet_violation_fraction(labels: torch.Tensor, embeds: torch.Tensor, margin: float) -> Tuple[int, int, float]:
    pdist = pairwise_distance(embeds)
    dist_ap = pdist.unsqueeze(2)
    dist_an = pdist.unsqueeze(1)
    violations = (dist_ap - dist_an + margin) > 0
    labels = labels.unsqueeze(1)
    ap_mask = torch.eq(labels, labels.t()).unsqueeze(2) & (
        ~torch.eye(labels.size(0), dtype=torch.bool, device=embeds.device)).unsqueeze(2)
    an_mask = (~torch.eq(labels, labels.t())).unsqueeze(1)
    valid = ap_mask & an_mask
    violating = (valid & violations).sum().item()
    total_valid = valid.sum().item()
    frac = violating / total_valid if total_valid > 0 else 0.0
    return violating, total_valid, frac


def get_hard_pair_distances(labels: torch.Tensor, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    device = embeddings.device
    labels = labels.long().unsqueeze(1).to(device)
    pairwise_dist = pairwise_distance(embeddings)

    mask_anchor_positive = torch.eq(labels, labels.t())
    mask_anchor_negative = ~mask_anchor_positive
    mask_not_self = ~torch.eye(labels.size(0), dtype=torch.bool, device=device)

    mask_positives = mask_anchor_positive & mask_not_self

    positives_dist = pairwise_dist.clone()
    positives_dist[~mask_positives] = -1e9
    dist_ap_hard, _ = positives_dist.max(dim=1)

    negatives_dist = pairwise_dist.clone()
    negatives_dist[~mask_anchor_negative] = 1e9
    dist_an_hard, _ = negatives_dist.min(dim=1)

    dist_ap_hard = torch.clamp(dist_ap_hard, min=0.0)
    dist_an_hard = torch.clamp(dist_an_hard, min=0.0)
    return dist_ap_hard, dist_an_hard


def sample_semihard_triplets(labels: torch.Tensor, embeds: torch.Tensor, margin: float = 1.0, max_triplets: int = 1024):
    """Efficient semi-hard triplet mining within a batch.
    Selects at most one triplet per anchor for speed.
    Returns indices (a_idx, p_idx, n_idx)."""
    device = embeds.device
    labels = labels.to(device)
    embeds = F.normalize(embeds, p=2, dim=1)  # normalize embeddings
    dist = pairwise_distance(embeds)  # reuse your existing function

    triplets = []
    violations = 0
    total_valid = 0

    for i in range(len(labels)):
        pos_mask = (labels == labels[i]) & (torch.arange(len(labels), device=device) != i)
        neg_mask = labels != labels[i]

        if not pos_mask.any() or not neg_mask.any():
            continue

        pos_dists = dist[i][pos_mask]
        neg_dists = dist[i][neg_mask]

        total_valid += len(pos_dists) * len(neg_dists)

        # Choose hardest positive and semi-hard negative
        hardest_pos_idx = pos_dists.argmax()
        hardest_pos = pos_dists[hardest_pos_idx]

        semi_hard_neg_mask = neg_dists > hardest_pos
        if semi_hard_neg_mask.any():
            semi_hard_neg_idx = torch.argmin(neg_dists[semi_hard_neg_mask])
            neg_indices = torch.arange(len(labels), device=device)[neg_mask][semi_hard_neg_mask]
            n_idx = neg_indices[semi_hard_neg_idx]
        else:
            # fallback: use closest negative
            n_idx = torch.arange(len(labels), device=device)[neg_mask][neg_dists.argmin()]

        # positive index in original label space
        p_indices = torch.arange(len(labels), device=device)[pos_mask]
        p_idx = p_indices[hardest_pos_idx]

        if dist[i, p_idx] + margin > dist[i, n_idx]:
            violations += 1
            triplets.append((i, p_idx.item(), n_idx.item()))

        if len(triplets) >= max_triplets:
            break

    violation_frac = violations / max(1, total_valid)
    if len(triplets) == 0:
        return None, None, None, violation_frac

    a_idx, p_idx, n_idx = zip(*triplets)
    return torch.tensor(a_idx, device=device), torch.tensor(p_idx, device=device), torch.tensor(n_idx,
                                                                                                device=device), violation_frac
