#!/usr/bin/env python

from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ww_trainer.dataset import AudioDataset
from ww_trainer.utils import get_hard_pair_distances, sample_triplets, pairwise_distance, sample_semihard_triplets, \
    pairwise_cosine_similarity, timed


class CN2Plus1PairLoss(nn.Module):
    """(C_N,2 + 1)-pair loss (López-Espejo et al., TASLP 2021).

    This loss encourages the distance between positive pairs (anchor-positive) to be
    smaller than a function of distances involving negative samples, where
    $C_{N,2}$ refers to the combination of negative-negative pairs.
    """

    def __init__(self, d_max: float = 2.0, margin: float = 0.0) -> None:
        """
        Initialize the CN2Plus1PairLoss.

        Args:
            d_max: Maximum distance allowed for negative samples (used in regularization term K).
            margin: The margin used in the final softplus calculation.
        """
        super().__init__()
        self.d_max = d_max
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        """
        Compute the CN2+1-Pair Loss.

        Args:
            anchor: Anchor embeddings (B, D).
            positive: Positive embeddings corresponding to anchors (B, D).
            negatives: Negative embeddings (N_neg, D) or (1, N_neg, D).

        Returns:
            The computed loss value.
        """
        # negatives: (N_neg, D)
        if negatives.dim() == 2:
            pass  # already correct shape
        elif negatives.dim() == 3:
            negatives = negatives.squeeze(0)  # in case it's (1, N_neg, D)

        N_minus_1 = negatives.size(0)
        B, D = anchor.shape

        if N_minus_1 < 1:
            return torch.tensor(0.0, device=anchor.device, dtype=anchor.dtype)

        # normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negatives = F.normalize(negatives, p=2, dim=1)

        # distances
        D_ap = torch.sum((anchor - positive) ** 2, dim=1)  # (B,)

        # D_an: Anchor-Negative distances (B, N_neg)
        # expand anchor to (B, 1, D) and negatives to (1, N_neg, D) for broadcasting
        D_an = torch.sum((anchor.unsqueeze(1) - negatives.unsqueeze(0)) ** 2, dim=2)

        # negative-negative distances
        D_nn = torch.cdist(negatives, negatives, p=2) ** 2  # (N_neg, N_neg)
        mask = torch.triu(torch.ones_like(D_nn), diagonal=1).bool()

        D_nn_sum = D_nn[mask].mean() if mask.any() else torch.tensor(0.0, device=anchor.device)

        lam = 1.0 / N_minus_1
        K = (N_minus_1 - 1) * self.d_max / 2.0 + self.margin

        # compute loss
        # sum_term is (B, 1) or broadcastable
        sum_term = D_an.sum(1, keepdim=True) + D_nn_sum # D_nn_sum broadcasts across batch dimension
        arg = D_ap.unsqueeze(1) - lam * sum_term + K
        loss = F.softplus(arg).mean()
        return loss


class SoftTripletLoss(nn.Module):
    """
    Soft Triplet Loss: log(1 + exp(D(a, p) - D(a, n)))

    This uses a soft margin equivalent to softplus, ensuring a gradient flow
    even when the hard margin constraint is met.
    """
    def __init__(self) -> None:
        """
        Initialize the SoftTripletLoss.
        """
        super().__init__()

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Compute the Soft Triplet Loss.

        Args:
            anchor: Anchor embeddings (B, D).
            positive: Positive embeddings (B, D).
            negative: Negative embeddings (B, D).

        Returns:
            The computed loss value.
        """
        # Normalize embeddings for distance calculation
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # L2-squared distance (used for efficiency/consistency)
        d_ap = torch.sum((anchor - positive) ** 2, dim=1) # (B,)
        d_an = torch.sum((anchor - negative) ** 2, dim=1) # (B,)

        # Softplus Loss: log(1 + exp(d_ap - d_an))
        arg = d_ap - d_an
        loss = F.softplus(arg).mean()
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss implementation.
    Pulls positive pairs together and pushes negative pairs apart with a margin.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds: Batch of embeddings (B, D).
            labels: Batch of binary labels (B,).
        """
        pdist = pairwise_distance(embeds)  # (B, B)
        B = embeds.size(0)

        # Mask for positive pairs (same label, non-diagonal)
        labels_mat = labels.unsqueeze(0) == labels.unsqueeze(1)

        pos_mask = labels_mat & (~torch.eye(B, dtype=torch.bool, device=embeds.device))

        # Mask for negative pairs (different label)
        neg_mask = ~labels_mat

        # Positive loss term (pull similar closer)
        pos_dist_sq = pdist[pos_mask] ** 2

        # Negative loss term (push dissimilar beyond margin)
        neg_dist_sq = pdist[neg_mask] ** 2

        # max(0, margin - D_neg)^2
        neg_loss_term = torch.clamp(self.margin - neg_dist_sq.sqrt(), min=0.0) ** 2

        pos_loss = pos_dist_sq.mean() if pos_dist_sq.numel() > 0 else torch.tensor(0.0, device=embeds.device)
        neg_loss = neg_loss_term.mean() if neg_loss_term.numel() > 0 else torch.tensor(0.0, device=embeds.device)

        # Loss is the sum of both terms
        loss = pos_loss + neg_loss
        return loss


class LiftedStructureLoss(nn.Module):
    """
    Lifted Structured Embedding (LSE) Loss.
    Uses all positive and negative pairs in the batch, combining them via log-sum-exp.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds: Batch of embeddings (B, D).
            labels: Batch of binary labels (B,).
        """
        pdist = pairwise_distance(embeds)  # (B, B)
        B = embeds.size(0)

        # 1. Masks
        labels_mat = labels.unsqueeze(0) == labels.unsqueeze(1)
        # Positive pairs (same label, non-diagonal)
        pos_mask = labels_mat & (~torch.eye(B, dtype=torch.bool, device=embeds.device))

        # Handle case where no positive pairs exist
        D_pos = pdist[pos_mask]
        if D_pos.numel() == 0:
            return torch.tensor(0.0, device=embeds.device)

        # 2. Negative terms calculation (Log-Sum-Exp over all negatives)

        # Distance matrix for all negative pairs: D_ik where l_i != l_k
        D_neg = pdist.clone()
        # Set D_neg for positive pairs (l_i == l_k) to inf so exp(M - inf) is 0
        D_neg[pos_mask] = float('inf')

        # Exponential term: exp(Margin - D_neg)
        exp_neg = torch.exp(self.margin - D_neg)

        # Since we only consider k where l_k != l_i, we can sum along dim=1 (for anchor i)
        sum_exp_neg = torch.sum(exp_neg, dim=1)  # (B,)

        log_term = torch.log(torch.clamp(sum_exp_neg, min=1e-6))

        pos_i, pos_j = torch.nonzero(pos_mask, as_tuple=True)
        D_ij = pdist[pos_i, pos_j]
        L_ij = D_ij + log_term[pos_i] + log_term[pos_j] - self.margin
        loss = torch.clamp(L_ij, min=0.0).mean()
        return loss


class AngularLoss(nn.Module):
    """
    Margin-based Angular Loss using Cosine Similarity.
    Enforces that cos(A,P) > cos(A,N) + margin.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds: Batch of embeddings (B, D). Assumes embeddings are L2-normalized.
            labels: Batch of binary labels (B,).
        """
        # Ensure embeddings are normalized for cosine similarity to work as dot product
        norm_embeds = F.normalize(embeds, p=2, dim=1)

        # Cosine similarity matrix (B, B)
        cos_sim = pairwise_cosine_similarity(norm_embeds)

        B = embeds.size(0)
        labels_mat = labels.unsqueeze(0) == labels.unsqueeze(1)

        # 1. Identify Anchor-Positive and Anchor-Negative triplets

        # We need a triplet mining approach to use this efficiently.
        # For simplicity and leveraging existing utilities, we will use a structure
        # similar to TripletLoss, but apply the angular condition.
        # We will mine semi-hard triplets based on Euclidean distance,
        # and then apply the angular loss to those triplets.

        # Use Euclidean distance for mining (standard practice)
        margin_for_mining = 0.5  # A fixed small margin for mining
        a_idx, p_idx, n_idx, _ = sample_semihard_triplets(labels, embeds, margin=margin_for_mining)

        if a_idx is None or len(a_idx) == 0:
            return torch.tensor(0.0, device=embeds.device, requires_grad=True)

        # Get the similarities for the mined triplets
        # cos_AP: cosine(A, P)
        cos_ap = cos_sim[a_idx, p_idx]
        # cos_AN: cosine(A, N)
        cos_an = cos_sim[a_idx, n_idx]

        # Loss is defined as: max(0, cos(A,P) - cos(A,N) + margin)
        # We want the positive similarity (cos_AP) to be greater than the negative similarity (cos_AN) by the margin.

        # MarginRankingLoss: max(0, -target*(x1-x2)+margin)
        # Let x1 = cos_AN, x2 = cos_AP, target = 1
        # Loss = max(0, -1*(cos_AN - cos_AP) + margin)
        # Loss = max(0, cos_AP - cos_AN + margin)

        # Since we use a custom loss, we compute the hinge loss directly:
        loss = torch.clamp(cos_ap - cos_an + self.margin, min=0.0)

        return loss.mean()


class RobustProtoDiversityLoss(nn.Module):
    """
    Robust Prototype and Diversity Loss (RPPL).

    A composite loss function involving BCE, prototype contrastive loss,
    intra-class center loss, negative diversity loss, and optionally consistency loss.
    """
    def __init__(self, tau: float = 0.1, K_neg_proto: int = 0, alpha: float = 1.0, beta: float = 1.0, gamma: float = 0.5, delta: float = 0.1, eta: float = 0.5) -> None:
        """
        Initialize the RPPL loss components and weights.

        Args:
            tau: Temperature parameter for the prototype contrastive loss.
            K_neg_proto: Number of negative prototypes to generate (0 or 1 means use a single mean negative prototype).
            alpha: Weight for the BCE loss component.
            beta: Weight for the Prototype Loss component.
            gamma: Weight for the Negative Diversity Loss component.
            delta: Weight for the Positive Center Loss component.
            eta: Weight for the Consistency Loss component (requires aug_embeds).
        """
        super().__init__()
        self.tau = tau
        self.K_neg_proto = K_neg_proto
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, embeds: torch.Tensor, aug_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the Robust Prototype and Diversity Loss.

        Args:
            logits: Model output logits (B, 1).
            labels: Ground truth labels (B,).
            embeds: Model embeddings (B, D).
            aug_embeds: Embeddings of augmented versions of the input, for consistency loss (B, D).

        Returns:
            The computed total RPPL loss value.
        """
        device = embeds.device
        labels = labels.view(-1)

        # 1. Binary Cross-Entropy (BCE) Loss
        bce = F.binary_cross_entropy_with_logits(logits.view(-1), labels.float().to(device))

        # Normalize embeddings for distance/similarity calculation
        z = F.normalize(embeds, p=2, dim=1)

        pos_mask = labels == 1
        neg_mask = labels == 0
        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()

        proto_loss = center_loss = div_loss = cons_loss = torch.tensor(0.0, device=device)

        if n_pos >= 1 and n_neg >= 1:
            # Positive Prototype (Mean of all wake embeddings)
            p_w = z[pos_mask].mean(0, keepdim=True)

            # Negative Prototypes
            if self.K_neg_proto <= 1:
                # Single negative prototype (Mean of all non-wake embeddings)
                neg_protos = z[neg_mask].mean(0, keepdim=True)
            else:
                # Multiple negative prototypes (K-means or random sampling/grouping mean)
                negs = z[neg_mask]
                k = min(self.K_neg_proto, max(1, n_neg))
                perm = torch.randperm(n_neg, device=device)
                # Create k random groups and compute mean for each group
                groups = torch.stack([negs[perm[i::k]].mean(0) for i in range(k)], dim=0)
                neg_protos = groups

            all_protos = torch.cat([p_w, neg_protos], dim=0)

            # 2. Prototype Contrastive Loss
            sims = torch.matmul(z, all_protos.t()) / self.tau
            wake_logit = sims[:, 0]
            notwake_logits = sims[:, 1:]

            # Use logsumexp over all negative prototypes
            notwake_lse = torch.logsumexp(notwake_logits, dim=1)
            two_way = torch.stack([wake_logit, notwake_lse], dim=1)
            labels_two = (labels > 0).long()
            proto_loss = F.cross_entropy(two_way, labels_two.to(device))

            if n_pos >= 1:
                # 3. Positive Center Loss (Encourages positive samples to be close to their prototype)
                center_loss = ((z[pos_mask] - p_w) ** 2).sum(dim=1).mean()

            if n_neg > 1:
                # 4. Negative Diversity Loss (Encourages negative samples to be far from each other)
                pd = torch.cdist(z[neg_mask], z[neg_mask], p=2) ** 2
                mask = torch.triu(torch.ones_like(pd), diagonal=1).bool()
                if mask.any():
                    div_loss = -pd[mask].mean() # Maximize distance -> negative mean distance

        # 5. Consistency Loss (Encourages original and augmented embeddings to be similar)
        if aug_embeds is not None:
            z_aug = F.normalize(aug_embeds, p=2, dim=1)
            cons_loss = ((z - z_aug) ** 2).sum(dim=1).mean()

        # Weighted sum of all components
        loss = (self.alpha * bce +
                self.beta * proto_loss +
                self.gamma * div_loss +
                self.delta * center_loss +
                self.eta * cons_loss)

        return loss


class LossManager:
    """Manages multiple weighted loss functions for training."""

    def __init__(self, loss_configs: List[Dict[str, Any]], mining_type: str = "semihard", device: str = "cpu") -> None:
        """
        Initialize the LossManager.

        Args:
            loss_configs: A list of dictionaries, each describing a loss:
                          `{"name": str, "weight": float, ...}`.
            device: The torch device ('cpu' or 'cuda') where losses should reside.

        Raises:
            ValueError: If an unknown loss name is encountered.
        """
        self.device = torch.device(device)
        self.losses: List[Dict[str, Any]] = []
        self.mining_type = mining_type

        for cfg in loss_configs:
            name = cfg["name"].lower()
            weight = cfg.get("weight", 1.0)
            crit: nn.Module

            if name == "bce":
                crit = nn.BCEWithLogitsLoss().to(self.device)
            elif name == "triplet":
                crit = nn.TripletMarginLoss(margin=cfg.get("margin", 1.0), p=2).to(self.device)
            elif name == "soft_triplet":
                crit = SoftTripletLoss().to(self.device)
            elif name == "pair":
                crit = nn.MarginRankingLoss(margin=cfg.get("margin", 1.0)).to(self.device)
            elif name == "cn2pair":
                crit = CN2Plus1PairLoss(
                    d_max=cfg.get("d_max", 2.0),
                    margin=cfg.get("margin", 0.0)
                ).to(self.device)
            elif name == "lse":
                crit = LiftedStructureLoss(margin=cfg.get("margin", 1.0)).to(self.device)
            elif name == "contrastive":
                crit = ContrastiveLoss(margin=cfg.get("margin", 1.0)).to(self.device)
            elif name == "angular":
                crit = AngularLoss(margin=cfg.get("margin", 0.5)).to(self.device)
            elif name == "rppl":
                crit = RobustProtoDiversityLoss(
                    tau=cfg.get("tau", 0.1),
                    K_neg_proto=cfg.get("K_neg_proto", 0),
                    alpha=cfg.get("alpha", 1.0),
                    beta=cfg.get("beta", 1.0),
                    gamma=cfg.get("gamma", 0.5),
                    delta=cfg.get("delta", 0.1),
                    eta=cfg.get("eta", 0.5)
                ).to(self.device)
            else:
                raise ValueError(f"Unknown loss: {name}")

            self.losses.append({"name": name, "weight": weight, "criterion": crit, "margin": cfg.get("margin", 1.0)})

    @timed
    def compute_loss(self, model: nn.Module, wavs: torch.Tensor, labels: torch.Tensor, dataset_ref: Optional[AudioDataset] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and per-loss values based on the configured criteria.

        Args:
            model: The wake word model (must have `embed` method for metric losses).
            wavs: Input audio waveforms (list of tensors or padded tensor).
            labels: Ground truth labels (B,).
            dataset_ref: A reference to the dataset or data loader for optional
                         features like augmentation (e.g., for RPPL consistency loss).

        Returns:
            A tuple containing:
            1. The total weighted loss (torch.Tensor).
            2. A dictionary of per-loss values {"loss_name": value} (Dict[str, float]).
        """
        results: Dict[str, float] = {}
        total = torch.tensor(0.0, device=self.device)

        logits = model(wavs)
        embeds = model.embed(wavs)
        # Embedding normalization is performed inside the metric loss functions
        labels_float = labels.to(self.device).float().view(-1, 1)

        for loss_entry in self.losses:
            name = loss_entry["name"]
            weight = loss_entry["weight"]
            crit = loss_entry["criterion"]
            loss_val = torch.tensor(0.0, device=self.device, requires_grad=True)  # Zero loss must require grad

            if name == "bce":
                # Binary Cross-Entropy Loss
                loss_val = crit(logits.view_as(labels_float), labels_float)

            elif name in ["triplet", "soft_triplet"]:
                # Triplet Margin Loss or Soft Triplet Loss
                margin = loss_entry.get("margin", 1.0)
                a, p, n, _ = sample_triplets(
                    labels.view(-1).long(),
                    embeds,
                    margin=margin,
                    mining_type=self.mining_type
                )

                if a is not None and len(a) > 0:
                    loss_val = crit(embeds[a], embeds[p], embeds[n])

            elif name == "pair":
                # Margin Ranking Loss (Pair Loss)
                d_ap, d_an = get_hard_pair_distances(labels.view(-1).long(), embeds)
                if d_ap is not None and d_an is not None:
                    target = torch.ones_like(d_ap, device=self.device)
                    # Loss = max(0, -target*(x1-x2)+margin) where x1=D_an, x2=D_ap, target=1
                    # Goal: D_an - D_ap + margin > 0
                    loss_val = crit(d_an, d_ap, target)

            elif name == "cn2pair":
                # CN2Plus1-Pair Loss
                pos_mask = (labels.view(-1) == 1)
                neg_mask = (labels.view(-1) == 0)
                pos = embeds[pos_mask]
                neg = embeds[neg_mask]

                # Requires at least two positives and one negative
                if len(pos) >= 2 and len(neg) >= 1:
                    # Select unique anchors/positives from available positive samples
                    idx = torch.randperm(len(pos), device=self.device)
                    # Simplified random positive pair selection: (0, 1), (1, 2), ..., (N-2, N-1)
                    anchor, positive = pos[idx[:-1]], pos[idx[1:]]
                    loss_val = crit(anchor, positive, neg)

            elif name == "rppl":

                # Robust Prototype and Diversity Loss
                aug_embeds = None
                # Check if the dataset/dataloader supports fetching augmented waveforms
                if dataset_ref is not None and hasattr(dataset_ref, "get_augmented"):
                    aug_embeds_list = []
                    with torch.no_grad():
                        for w in wavs:
                            aug_w = dataset_ref.get_augmented(w).to(self.device)
                            # ensure batch dim
                            emb = model.embed(aug_w.unsqueeze(0)).squeeze(0)  # (D,)
                            aug_embeds_list.append(emb)
                    aug_embeds = torch.stack(aug_embeds_list, dim=0)
                    aug_embeds = F.normalize(aug_embeds, p=2, dim=1)

                loss_val = crit(logits, labels.view(-1), embeds, aug_embeds)

            total += weight * loss_val
            results[name] = float(loss_val.item())

        results["total"] = float(total.item())
        return total, results