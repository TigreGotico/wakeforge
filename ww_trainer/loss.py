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


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., ICCV 2017).

    Down-weights easy examples and focuses training on hard ones.
    Critical for imbalanced wake word datasets where negatives dominate.

    ``FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)``

    Args:
        alpha: Balancing factor for positive class (default 0.25).
        gamma: Focusing parameter — higher values focus more on hard examples.
        reduction: ``"mean"`` or ``"sum"``.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw logits ``[B]`` or ``[B, 1]``.
            labels: Binary labels ``[B]`` (float).

        Returns:
            Scalar loss.
        """
        logits = logits.view(-1)
        labels = labels.view(-1).float()
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

        p_t = p * labels + (1 - p) * (1 - labels)
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * ce
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class LabelSmoothingBCE(nn.Module):
    """Binary Cross-Entropy with label smoothing (Szegedy et al. 2016).

    Replaces hard labels {0, 1} with soft labels {smoothing, 1 - smoothing}.
    Prevents overconfident predictions and improves generalization.

    Args:
        smoothing: Label smoothing factor (default 0.1).
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed BCE.

        Args:
            logits: Raw logits ``[B]`` or ``[B, 1]``.
            labels: Binary labels ``[B]`` (float).

        Returns:
            Scalar loss.
        """
        logits = logits.view(-1)
        labels = labels.view(-1).float()
        smooth = labels * (1.0 - self.smoothing) + (1.0 - labels) * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, smooth)


class ArcFaceLoss(nn.Module):
    """Additive Angular Margin Loss / ArcFace (Deng et al., CVPR 2019).

    Adds an angular margin penalty in cosine space to enforce inter-class
    separability.  Gold standard for face/speaker verification, increasingly
    used for KWS.

    For binary wake word detection, uses 2 class centers (wake / not-wake).

    Args:
        embed_dim: Embedding dimension.
        margin: Angular margin in radians (default 0.5 ≈ 28.6°).
        scale: Cosine scaling factor (default 30.0).
    """

    def __init__(self, embed_dim: int, margin: float = 0.5,
                 scale: float = 30.0) -> None:
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(2, embed_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ArcFace loss.

        Args:
            embeds: Normalized embeddings ``[B, D]``.
            labels: Binary labels ``[B]`` (long).

        Returns:
            Scalar loss.
        """
        embeds = F.normalize(embeds, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        cosine = torch.matmul(embeds, w.t())  # [B, 2]

        # Add angular margin to the target class
        labels = labels.view(-1).long()
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        target_theta = theta[torch.arange(len(labels)), labels] + self.margin
        target_cos = torch.cos(target_theta.clamp(0, torch.pi))

        logits = cosine.clone()
        logits[torch.arange(len(labels)), labels] = target_cos
        logits = logits * self.scale

        return F.cross_entropy(logits, labels)


class CenterLoss(nn.Module):
    """Center Loss (Wen et al., ECCV 2016).

    Learns a center for each class and penalizes distance from embeddings
    to their corresponding class center.  Reduces intra-class variation.

    Args:
        embed_dim: Embedding dimension.
        num_classes: Number of classes (default 2 for wake word).
    """

    def __init__(self, embed_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, embed_dim))

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute center loss.

        Args:
            embeds: Embeddings ``[B, D]``.
            labels: Binary labels ``[B]`` (long).

        Returns:
            Scalar loss.
        """
        labels = labels.view(-1).long()
        centers_batch = self.centers[labels]  # [B, D]
        return ((embeds - centers_batch) ** 2).sum(dim=1).mean() / 2.0


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy / SimCLR loss (Chen et al. 2020).

    Contrastive loss that treats each sample's positive pair against all
    other samples as negatives.  For supervised use, positive pairs are
    samples sharing the same label.

    Args:
        temperature: Softmax temperature (default 0.07).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss.

        Args:
            embeds: Embeddings ``[B, D]``.
            labels: Binary labels ``[B]``.

        Returns:
            Scalar loss.
        """
        embeds = F.normalize(embeds, p=2, dim=1)
        B = embeds.size(0)
        sim = torch.matmul(embeds, embeds.t()) / self.temperature  # [B, B]

        # Mask: same label = positive, different = negative
        labels = labels.view(-1)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
        pos_mask.fill_diagonal_(False)

        # For numerical stability
        sim.fill_diagonal_(-1e9)

        # For each anchor, compute log-softmax over all others
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        # Mean of log-prob over positive pairs
        n_pos = pos_mask.sum(dim=1).clamp(min=1)
        loss = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos
        return loss.mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Extension of SimCLR to the supervised setting.  All samples of the
    same class form positive pairs; all others are negatives.  More
    stable and effective than triplet loss.

    Args:
        temperature: Softmax temperature (default 0.07).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute supervised contrastive loss.

        Args:
            embeds: Embeddings ``[B, D]``.
            labels: Binary labels ``[B]``.

        Returns:
            Scalar loss.
        """
        embeds = F.normalize(embeds, p=2, dim=1)
        B = embeds.size(0)
        labels = labels.view(-1)

        sim = torch.matmul(embeds, embeds.t()) / self.temperature
        # Mask out self-similarity
        self_mask = ~torch.eye(B, dtype=torch.bool, device=embeds.device)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & self_mask

        # Log-softmax over non-self entries
        sim_masked = sim.masked_fill(~self_mask, -1e9)
        log_prob = sim_masked - torch.logsumexp(sim_masked, dim=1, keepdim=True)

        n_pos = pos_mask.sum(dim=1).clamp(min=1).float()
        loss = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos

        # Only compute for anchors that have at least one positive
        valid = pos_mask.sum(dim=1) > 0
        if valid.any():
            return loss[valid].mean()
        return torch.tensor(0.0, device=embeds.device, requires_grad=True)


class ProxyNCALoss(nn.Module):
    """Proxy-NCA Loss (Movshovitz-Attias et al., ICCV 2017).

    Uses learnable proxies (one per class) instead of mining pairs/triplets.
    Each sample is compared to all class proxies via softmax.  Converges
    faster than triplet loss with less hyperparameter sensitivity.

    Args:
        embed_dim: Embedding dimension.
        num_classes: Number of classes (default 2).
        scale: Distance scaling factor.
    """

    def __init__(self, embed_dim: int, num_classes: int = 2,
                 scale: float = 8.0) -> None:
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_normal_(self.proxies)
        self.scale = scale

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute Proxy-NCA loss.

        Args:
            embeds: Embeddings ``[B, D]``.
            labels: Binary labels ``[B]`` (long).

        Returns:
            Scalar loss.
        """
        embeds = F.normalize(embeds, p=2, dim=1)
        proxies = F.normalize(self.proxies, p=2, dim=1)
        labels = labels.view(-1).long()

        # Negative squared L2 distance (similarity)
        dist = torch.cdist(embeds, proxies, p=2) ** 2  # [B, num_classes]
        logits = -self.scale * dist

        return F.cross_entropy(logits, labels)


class MultiSimilarityLoss(nn.Module):
    """Multi-Similarity Loss (Wang et al., CVPR 2019).

    Mines informative pairs using three similarities: self-similarity,
    relative similarity (positive), and negative similarity.  More
    effective pair mining than triplet or contrastive approaches.

    Args:
        alpha: Positive pair weighting (default 2.0).
        beta: Negative pair weighting (default 50.0).
        base: Margin base (default 0.5).
    """

    def __init__(self, alpha: float = 2.0, beta: float = 50.0,
                 base: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute multi-similarity loss.

        Args:
            embeds: Embeddings ``[B, D]``.
            labels: Binary labels ``[B]``.

        Returns:
            Scalar loss.
        """
        embeds = F.normalize(embeds, p=2, dim=1)
        B = embeds.size(0)
        labels = labels.view(-1)

        sim = torch.matmul(embeds, embeds.t())  # [B, B]
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
        pos_mask.fill_diagonal_(False)
        neg_mask = ~(labels.unsqueeze(0) == labels.unsqueeze(1))

        loss = torch.tensor(0.0, device=embeds.device, requires_grad=True)
        n_valid = 0

        for i in range(B):
            pos_sim = sim[i][pos_mask[i]]
            neg_sim = sim[i][neg_mask[i]]

            if pos_sim.numel() == 0 or neg_sim.numel() == 0:
                continue

            # Positive pair mining: harder than hardest negative + margin
            pos_thresh = neg_sim.max() + self.base
            hard_pos = pos_sim[pos_sim < pos_thresh]

            # Negative pair mining: harder than easiest positive - margin
            neg_thresh = pos_sim.min() - self.base
            hard_neg = neg_sim[neg_sim > neg_thresh]

            if hard_pos.numel() == 0 or hard_neg.numel() == 0:
                continue

            pos_term = (1.0 / self.alpha) * torch.logsumexp(
                -self.alpha * (hard_pos - self.base), dim=0
            )
            neg_term = (1.0 / self.beta) * torch.logsumexp(
                self.beta * (hard_neg - self.base), dim=0
            )

            loss = loss + pos_term + neg_term
            n_valid += 1

        if n_valid > 0:
            loss = loss / n_valid
        return loss


class SizeAwareLoss(nn.Module):
    """Wraps a base loss with L1 sparsity and parameter-count penalties.

    Encourages smaller models during training by adding:
    - L1 norm of all parameters (sparsity pressure)
    - Soft penalty proportional to param_count / param_budget

    Args:
        base_loss: The underlying loss module.
        l1_weight: Coefficient for L1 sparsity penalty.
        size_weight: Coefficient for the param-count penalty.
        param_budget: Target parameter count. Penalty grows as params exceed this.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        l1_weight: float = 1e-5,
        size_weight: float = 0.1,
        param_budget: int = 1024,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.l1_weight = l1_weight
        self.size_weight = size_weight
        self.param_budget = param_budget

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        model: nn.Module,
    ) -> torch.Tensor:
        """Compute base loss + sparsity penalty + size penalty.

        Args:
            logits: Model output logits ``(B,)`` or ``(B, 1)``.
            labels: Ground truth labels ``(B,)``.
            model: The model whose parameters are penalized.

        Returns:
            Scalar loss tensor.
        """
        base = self.base_loss(logits.view(-1), labels.view(-1).float())

        # L1 sparsity: sum of absolute values of all parameters
        l1 = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)

        # Size penalty: ratio of actual params to budget, clamped to [0, inf)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size_ratio = max(0.0, n_params / self.param_budget - 1.0)

        return base + self.l1_weight * l1 + self.size_weight * size_ratio


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
            elif name == "focal":
                crit = FocalLoss(
                    alpha=cfg.get("alpha", 0.25),
                    gamma=cfg.get("gamma", 2.0),
                ).to(self.device)
            elif name == "label_smoothing_bce":
                crit = LabelSmoothingBCE(
                    smoothing=cfg.get("smoothing", 0.1),
                ).to(self.device)
            elif name == "arcface":
                crit = ArcFaceLoss(
                    embed_dim=cfg.get("embed_dim", 128),
                    margin=cfg.get("margin", 0.5),
                    scale=cfg.get("scale", 30.0),
                ).to(self.device)
            elif name == "center":
                crit = CenterLoss(
                    embed_dim=cfg.get("embed_dim", 128),
                    num_classes=cfg.get("num_classes", 2),
                ).to(self.device)
            elif name == "ntxent":
                crit = NTXentLoss(
                    temperature=cfg.get("temperature", 0.07),
                ).to(self.device)
            elif name == "supcon":
                crit = SupConLoss(
                    temperature=cfg.get("temperature", 0.07),
                ).to(self.device)
            elif name == "proxy_nca":
                crit = ProxyNCALoss(
                    embed_dim=cfg.get("embed_dim", 128),
                    num_classes=cfg.get("num_classes", 2),
                    scale=cfg.get("scale", 8.0),
                ).to(self.device)
            elif name == "multi_similarity":
                crit = MultiSimilarityLoss(
                    alpha=cfg.get("alpha", 2.0),
                    beta=cfg.get("beta", 50.0),
                    base=cfg.get("base", 0.5),
                ).to(self.device)
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
            elif name == "size_aware":
                base = nn.BCEWithLogitsLoss().to(self.device)
                crit = SizeAwareLoss(
                    base_loss=base,
                    l1_weight=cfg.get("l1_weight", 1e-5),
                    size_weight=cfg.get("size_weight", 0.1),
                    param_budget=cfg.get("param_budget", 1024),
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

            elif name in ("focal", "label_smoothing_bce"):
                loss_val = crit(logits.view(-1), labels.to(self.device).float().view(-1))

            elif name in ("arcface", "center", "proxy_nca"):
                loss_val = crit(embeds, labels.to(self.device).view(-1))

            elif name in ("ntxent", "supcon", "multi_similarity"):
                # Need at least 1 pos and 1 neg in batch
                labs = labels.to(self.device).view(-1)
                if (labs == 1).any() and (labs == 0).any():
                    loss_val = crit(embeds, labs)

            elif name in ("lse", "contrastive", "angular"):
                labs = labels.to(self.device).view(-1)
                if (labs == 1).any() and (labs == 0).any():
                    loss_val = crit(embeds, labs)

            elif name == "size_aware":
                # SizeAwareLoss needs the model reference
                loss_val = crit(logits.view(-1), labels.to(self.device).float().view(-1), model)

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