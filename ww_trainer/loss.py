#!/usr/bin/env python

import inspect
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ww_trainer.dataset import AudioDataset
from ww_trainer.utils import get_hard_pair_distances, sample_triplets, pairwise_distance, sample_semihard_triplets, \
    pairwise_cosine_similarity, timed


def _kmeans_prototypes(x: torch.Tensor, k: int, n_iter: int = 5) -> torch.Tensor:
    """Cosine k-means on L2-normalised rows of ``x``; returns ``(k, D)`` centroids.

    Centroids are L2-normalised at every step. Empty clusters are reseeded from
    the row furthest from any current centroid. Detached from autograd graph —
    centroids are used as fixed targets, not parameters.
    """
    n, _d = x.shape
    if k <= 1 or n <= k:
        return x.detach().mean(0, keepdim=True) if k <= 1 else x.detach().clone()
    xd = x.detach()
    perm = torch.randperm(n, device=x.device)[:k]
    centroids = F.normalize(xd[perm], dim=1)
    for _ in range(n_iter):
        sims = xd @ centroids.t()  # (n, k)
        assign = sims.argmax(dim=1)
        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            mask = assign == j
            if mask.any():
                new_centroids[j] = xd[mask].mean(0)
            else:
                # Reseed empty cluster from the worst-fit point
                worst = sims.max(dim=1).values.argmin()
                new_centroids[j] = xd[worst]
        centroids = F.normalize(new_centroids, dim=1)
    return centroids


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
    Contrastive Loss (Hadsell et al., CVPR 2006).
    Pulls wake/wake pairs together and pushes wake/not-wake pairs apart with a margin.
    Not-wake/not-wake pairs are left alone: that class is open (speech, music, noise) and
    has no reason to be compact.
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

        labels = labels.view(-1).to(embeds.device)
        labels_mat = labels.unsqueeze(0) == labels.unsqueeze(1)
        wake = labels == 1
        pos_mask = labels_mat & wake.unsqueeze(1) & (~torch.eye(B, dtype=torch.bool, device=embeds.device))

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
    Lifted Structured Embedding (LSE) Loss (Song et al., CVPR 2016).

    For every positive pair (i, j), with N_i the clips of the other class:

        J_ij = log( sum_{k in N_i} exp(margin - D_ik) + sum_{l in N_j} exp(margin - D_jl) ) + D_ij
        L    = 1 / (2 |P|) * sum_{(i, j) in P} max(0, J_ij)^2

    Positive pairs are wake/wake pairs only: the not-wake class is open (speech, music, noise)
    and is not pulled together.
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
        labels = labels.view(-1).to(embeds.device)

        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        not_self = ~torch.eye(B, dtype=torch.bool, device=embeds.device)
        pos_mask = same & not_self & (labels == 1).unsqueeze(1) & (labels == 1).unsqueeze(0)
        neg_mask = ~same
        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=embeds.device, requires_grad=True)

        # log-sum-exp over each anchor's negatives, as terms to combine per pair
        neg_logits = (self.margin - pdist).masked_fill(~neg_mask, float("-inf"))  # (B, B)
        pos_i, pos_j = torch.nonzero(torch.triu(pos_mask, diagonal=1), as_tuple=True)
        both = torch.cat([neg_logits[pos_i], neg_logits[pos_j]], dim=1)  # (|P|/2, 2B)
        J = torch.logsumexp(both, dim=1) + pdist[pos_i, pos_j]
        return 0.5 * torch.clamp(J, min=0.0).pow(2).mean()


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

        if a_idx is not None:
            # the not-wake class is open (speech, music, noise): only wake clips anchor a triplet,
            # as in sample_triplets, so negatives are never pulled towards each other
            wake = labels.to(a_idx.device)[a_idx] == 1
            a_idx, p_idx, n_idx = a_idx[wake], p_idx[wake], n_idx[wake]
        if a_idx is None or len(a_idx) == 0:
            return torch.tensor(0.0, device=embeds.device, requires_grad=True)

        cos_ap = cos_sim[a_idx, p_idx]
        cos_an = cos_sim[a_idx, n_idx]
        # hinge on cos(A,N) - cos(A,P) + margin: zero once the positive is closer by the margin
        loss = torch.clamp(cos_an - cos_ap + self.margin, min=0.0)

        return loss.mean()


class RobustProtoDiversityLoss(nn.Module):
    """
    Robust Prototype and Diversity Loss (RPPL).

    A composite loss function for binary fixed-keyword wake-word detection:
      - BCE: standard binary cross-entropy
      - Proto-softmax: prototype-based contrastive loss (Snell et al. 2017)
        with EMA-stabilised wake prototype (stable across small batches)
      - Center loss: pull wake embeddings toward their prototype (Wen et al. 2016)
      - Hard-negative diversity: spread confusable negatives apart
        (only those with cos-sim to wake prototype > hard_div_threshold)
      - Proto-ranked consistency: under acoustic augmentation each sample
        should remain correctly classified against the shared prototypes

    New contributions vs. baseline:
      - EMA prototype: eliminates per-batch noise on 2–5 wake samples
      - Hard-div: targets gradient budget at confusable negatives
      - Proto-consistency: semantically stronger than L2 invariance
      - Warmup: geometric terms ramp in over warmup_epochs to avoid
        training against random early prototypes
    """

    def __init__(
        self,
        tau: float = 0.1,
        K_neg_proto: int = 0,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5,
        delta: float = 0.1,
        eta: float = 0.5,
        warmup_epochs: int = 5,
        proto_ema_alpha: float = 0.05,
        hard_div_threshold: float = 0.1,
        consistency_mode: str = "proto",
        div_margin: float = 1.0,
    ) -> None:
        """
        Args:
            tau: Temperature for prototype contrastive loss.
            K_neg_proto: Number of negative prototypes (0/1 = single mean).
            alpha: BCE weight.
            beta: Prototype loss weight.
            gamma: Diversity loss weight.
            delta: Center loss weight.
            eta: Consistency loss weight.
            warmup_epochs: Epochs over which geometric terms (proto/div/center) ramp to full weight.
            proto_ema_alpha: EMA decay for wake prototype (0.05 = slow decay, stable prototype).
            hard_div_threshold: Diversity only applied to negatives with cos-sim to wake proto above this.
            consistency_mode: "proto" (proto-ranked CE) or "l2" (L2 distance, legacy).
            div_margin: Squared-L2 margin for hinge diversity (no gradient once pairs are further apart).
        """
        super().__init__()
        if not 0.0 < proto_ema_alpha <= 1.0:
            raise ValueError(f"proto_ema_alpha must be in (0, 1], got {proto_ema_alpha}")
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if consistency_mode not in ("proto", "l2"):
            raise ValueError(f"consistency_mode must be 'proto' or 'l2', got {consistency_mode!r}")
        self.tau = tau
        self.K_neg_proto = K_neg_proto
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta
        self.warmup_epochs = warmup_epochs
        self.proto_ema_alpha = proto_ema_alpha
        self.hard_div_threshold = hard_div_threshold
        self.consistency_mode = consistency_mode
        self.div_margin = div_margin

        # EMA wake prototype — shape inferred on first forward pass
        self.register_buffer("proto_w_ema", torch.zeros(1))
        self.register_buffer("_proto_initialised", torch.tensor(False))
        self._epoch: int = 0

        # Sub-loss values from last forward pass (for external logging)
        self.last_bce: float = 0.0
        self.last_proto: float = 0.0
        self.last_div: float = 0.0
        self.last_center: float = 0.0
        self.last_cons: float = 0.0
        self.last_geo_scale: float = 0.0
        self.last_proto_ema_norm: float = 0.0

    def set_epoch(self, epoch: int) -> None:
        """Called by LossManager at the start of each epoch to drive warmup scheduling."""
        self._epoch = epoch

    @property
    def wake_prototype(self) -> torch.Tensor:
        """EMA wake prototype vector (shape ``(D,)``), or None if not yet initialised."""
        if not self._proto_initialised.item():
            return None
        return F.normalize(self.proto_w_ema.detach(), dim=0)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeds: torch.Tensor,
        aug_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = embeds.device
        labels = labels.view(-1)

        # 1. BCE
        bce = F.binary_cross_entropy_with_logits(logits.view(-1), labels.float().to(device))

        # Normalise embeddings
        z = F.normalize(embeds, p=2, dim=1)

        pos_mask = labels == 1
        neg_mask = labels == 0
        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())

        proto_loss = center_loss = div_loss = cons_loss = torch.tensor(0.0, device=device)
        p_w_stable: Optional[torch.Tensor] = None
        neg_protos: Optional[torch.Tensor] = None

        if n_pos >= 1 and n_neg >= 1:
            # Batch wake mean — used only to UPDATE the EMA, not as the loss target
            p_w_batch = z[pos_mask].mean(0)

            # EMA prototype update — TRAIN MODE ONLY (eval batches must not drift the target)
            if self.training:
                if not self._proto_initialised.item():
                    self.proto_w_ema = p_w_batch.detach().clone()
                    self._proto_initialised = torch.tensor(True, device=device)
                else:
                    self.proto_w_ema = (
                        (1.0 - self.proto_ema_alpha) * self.proto_w_ema
                        + self.proto_ema_alpha * p_w_batch.detach()
                    )
            elif not self._proto_initialised.item():
                # Eval-time cold start: use batch mean directly (no EMA write)
                p_w_stable_init = p_w_batch.detach()
                self.proto_w_ema = p_w_stable_init.clone()
                self._proto_initialised = torch.tensor(True, device=device)

            # Stable (EMA) wake prototype — used in all geometric terms
            p_w_stable = F.normalize(self.proto_w_ema, dim=0).unsqueeze(0)  # (1, D)

            # Negative prototypes
            if self.K_neg_proto <= 1:
                neg_protos = z[neg_mask].mean(0, keepdim=True)
            else:
                neg_protos = _kmeans_prototypes(
                    z[neg_mask], k=min(self.K_neg_proto, max(1, n_neg)), n_iter=5,
                )

            all_protos = torch.cat([p_w_stable, neg_protos], dim=0)

            # 2. Prototype contrastive loss (using stable EMA prototype)
            sims = torch.matmul(z, all_protos.t()) / self.tau
            wake_logit = sims[:, 0]
            notwake_lse = torch.logsumexp(sims[:, 1:], dim=1)
            two_way = torch.stack([wake_logit, notwake_lse], dim=1)
            labels_two = (labels > 0).long()
            proto_loss = F.cross_entropy(two_way, labels_two.to(device))

            # 3. Center loss (pull positives toward stable prototype)
            center_loss = ((z[pos_mask] - p_w_stable) ** 2).sum(dim=1).mean()

            # 4. Hard-negative diversity loss — hinge form, bounded gradient.
            # Only applied when at least 2 confusable negatives exist; no
            # fallback to all-negatives (which would defeat the targeting).
            if n_neg > 1:
                neg_embs = z[neg_mask]
                cos_to_wake = torch.matmul(neg_embs, p_w_stable.t()).squeeze(1)
                hard_mask = cos_to_wake > self.hard_div_threshold
                if int(hard_mask.sum().item()) >= 2:
                    target_embs = neg_embs[hard_mask]
                    pd = torch.cdist(target_embs, target_embs, p=2) ** 2
                    n_h = target_embs.shape[0]
                    upper = torch.triu(
                        torch.ones(n_h, n_h, device=device), diagonal=1,
                    ).bool()
                    # Hinge: pairs already > margin apart contribute 0 (zero gradient)
                    pair_d = pd[upper]
                    div_loss = torch.clamp(self.div_margin - pair_d, min=0.0).mean()

        # 5. Consistency loss
        if aug_embeds is not None:
            z_aug = F.normalize(aug_embeds, p=2, dim=1)
            if self.consistency_mode == "proto" and p_w_stable is not None and neg_protos is not None and n_pos >= 1 and n_neg >= 1:
                # Proto-ranked consistency: augmented samples must still be
                # classifiable against shared prototypes (stronger than L2)
                aug_wake_sim = torch.matmul(z_aug, p_w_stable.t()).squeeze(1) / self.tau
                aug_nwk_lse = torch.logsumexp(
                    torch.matmul(z_aug, neg_protos.t()) / self.tau, dim=1
                )
                aug_two_way = torch.stack([aug_wake_sim, aug_nwk_lse], dim=1)
                labels_two = (labels > 0).long()
                cons_loss = F.cross_entropy(aug_two_way, labels_two.to(device))
            else:
                # L2 fallback (legacy / ablation)
                cons_loss = ((z - z_aug) ** 2).sum(dim=1).mean()

        # Warmup: geometric terms ramp linearly from 0 → full over warmup_epochs
        geo_scale = min(1.0, self._epoch / max(1, self.warmup_epochs))

        loss = (
            self.alpha * bce
            + geo_scale * self.beta  * proto_loss
            + geo_scale * self.gamma * div_loss
            + geo_scale * self.delta * center_loss
            + self.eta * cons_loss
        )

        # Cache sub-loss values for external logging
        self.last_bce    = float(bce.item())
        self.last_proto  = float(proto_loss.item())
        self.last_div    = float(div_loss.item())
        self.last_center = float(center_loss.item())
        self.last_cons   = float(cons_loss.item())
        self.last_geo_scale = float(geo_scale)
        self.last_proto_ema_norm = (
            float(self.proto_w_ema.norm().item())
            if bool(self._proto_initialised.item())
            else 0.0
        )

        return loss


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., ICCV 2017).

    Down-weights easy examples and focuses training on hard ones.
    Critical for imbalanced wake word datasets where negatives dominate.

    ``FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)``

    Args:
        alpha: Balancing factor for positive class (default 0.90).
        gamma: Focusing parameter — higher values focus more on hard examples.
        reduction: ``"mean"`` or ``"sum"``.
    """

    def __init__(self, alpha: float = 0.90, gamma: float = 2.0,
                 reduction: str = "mean") -> None:
        # alpha=0.90 up-weights the positive (wake) class — correct for imbalanced KWS.
        # The original default of 0.25 suppressed positives, inverting focal loss's intent.
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

    def __init__(self, smoothing: float = 0.03) -> None:
        # 0.03 instead of 0.1 — binary classification needs light smoothing only.
        # 0.1 over-softens the positive target, preventing confident wake detections.
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

    @property
    def wake_prototype(self) -> torch.Tensor:
        """Normalised wake-class center vector (shape ``(D,)``)."""
        return F.normalize(self.weight[1].detach(), dim=0)

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

        # cos(theta + m) for the target class. Where theta + m would pass pi the curve turns back
        # up, so, as in the reference implementation, it continues as cos(theta) - m * sin(m):
        # the most misclassified clips keep a gradient instead of being clamped flat at pi
        labels = labels.view(-1).long()
        cos_t = cosine[torch.arange(len(labels)), labels]
        sin_t = torch.sqrt((1.0 - cos_t.pow(2)).clamp(min=1e-7))
        m = torch.tensor(self.margin)
        phi = cos_t * torch.cos(m) - sin_t * torch.sin(m)
        target_cos = torch.where(cos_t > torch.cos(torch.pi - m), phi, cos_t - self.margin * torch.sin(m))

        logits = cosine.clone()
        logits[torch.arange(len(labels)), labels] = target_cos
        logits = logits * self.scale

        return F.cross_entropy(logits, labels)


class CenterLoss(nn.Module):
    """Center Loss (Wen et al., ECCV 2016).

    Learns a center for each keyword class and penalizes the distance from each keyword
    embedding to its class center.  Reduces intra-class variation.  Class 0 (not-wake) is
    open (speech, music, noise), so its clips are not pulled to a center.

    Args:
        embed_dim: Embedding dimension.
        num_classes: Number of classes (default 2 for wake word).
    """

    def __init__(self, embed_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, embed_dim))

    @property
    def wake_prototype(self) -> torch.Tensor:
        """Normalised wake-class center vector (shape ``(D,)``)."""
        return F.normalize(self.centers[1].detach(), dim=0)

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute center loss.

        Args:
            embeds: Embeddings ``[B, D]``.
            labels: Binary labels ``[B]`` (long).

        Returns:
            Scalar loss.
        """
        labels = labels.view(-1).long()
        # class 0 (not-wake) is open, so only keyword clips are pulled to their center
        kw = labels != 0
        if not kw.any():
            return embeds.sum() * 0.0
        return ((embeds[kw] - self.centers[labels[kw]]) ** 2).sum(dim=1).mean() / 2.0


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

        # Mean of log-prob over positive pairs, for wake anchors only: the not-wake class is open
        # (speech, music, noise) and its members are not pulled towards each other
        n_pos = pos_mask.sum(dim=1).clamp(min=1)
        loss = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos
        anchors = (labels == 1) & (pos_mask.sum(dim=1) > 0)
        if anchors.any():
            return loss[anchors].mean()
        return torch.tensor(0.0, device=embeds.device, requires_grad=True)


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

        # Only wake anchors with at least one positive: the not-wake class is open (speech,
        # music, noise) and its members are not pulled towards each other
        valid = (labels == 1) & (pos_mask.sum(dim=1) > 0)
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

    @property
    def wake_prototype(self) -> torch.Tensor:
        """Normalised wake-class proxy vector (shape ``(D,)``)."""
        return F.normalize(self.proxies[1].detach(), dim=0)

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

    For each wake anchor i, pairs are mined with margin ``epsilon`` (a positive is kept if it is
    less similar than the most similar negative plus epsilon; a negative if it is more similar
    than the least similar positive minus epsilon), then weighted by

        1/alpha * log(1 + sum_P exp(-alpha * (S_ik - base)))
      + 1/beta  * log(1 + sum_N exp( beta  * (S_ik - base)))

    The ``1 +`` makes each term a soft hinge that goes to zero once the pairs are on the right
    side of ``base``. Only wake clips anchor: the not-wake class is open and is not pulled together.

    Args:
        alpha: Positive pair weighting (default 2.0).
        beta: Negative pair weighting (default 50.0).
        base: Similarity threshold lambda (default 0.5).
        epsilon: Mining margin (default 0.1).
    """

    def __init__(self, alpha: float = 2.0, beta: float = 50.0,
                 base: float = 0.5, epsilon: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.epsilon = epsilon

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute multi-similarity loss.

        Args:
            embeds: Embeddings ``[B, D]``.
            labels: Binary labels ``[B]``.

        Returns:
            Scalar loss.
        """
        embeds = F.normalize(embeds, p=2, dim=1)
        labels = labels.view(-1).to(embeds.device)
        sim = torch.matmul(embeds, embeds.t())  # [B, B]
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = same & ~torch.eye(len(labels), dtype=torch.bool, device=embeds.device)
        neg_mask = ~same

        def soft_hinge(x):  # log(1 + sum exp(x))
            return torch.logsumexp(torch.cat([x.new_zeros(1), x]), dim=0)

        terms = []
        for i in torch.nonzero(labels == 1).flatten().tolist():
            pos_sim, neg_sim = sim[i][pos_mask[i]], sim[i][neg_mask[i]]
            if pos_sim.numel() == 0 or neg_sim.numel() == 0:
                continue
            hard_pos = pos_sim[pos_sim - self.epsilon < neg_sim.max()]
            hard_neg = neg_sim[neg_sim + self.epsilon > pos_sim.min()]
            if hard_pos.numel() == 0 and hard_neg.numel() == 0:
                continue
            terms.append(soft_hinge(-self.alpha * (hard_pos - self.base)) / self.alpha
                         + soft_hinge(self.beta * (hard_neg - self.base)) / self.beta)
        if not terms:
            return torch.tensor(0.0, device=embeds.device, requires_grad=True)
        return torch.stack(terms).mean()


class HALOLoss(nn.Module):
    """Hyperbolic Anchor Loss Optimization (HALO).

    Distance-based cross-entropy loss that replaces dot-product similarity with
    squared Euclidean distance between embeddings and learnable class centroids.
    Adds an origin "abstain" sink (K+1 class) and a geometric radial regularizer.

    Compared to BCE, HALO yields better-calibrated probabilities and improved
    out-of-distribution detection — useful for rejecting non-wake-word audio.

    Reference: https://github.com/4rtemi5/halo — pisoni.ai/posts/halo/

    Args:
        emb_dims: Embedding dimension (must match ``model.embed()`` output).
        num_classes: Number of classes (2 for binary wake-word detection).
        learn_gamma: Whether temperature ``gamma`` is a learnable parameter.
        distill: Use teacher-free self-distillation for soft targets.
        label_smoothing: Soft-target spread (0 = hard labels).
        reduction: ``"mean"`` or ``"none"``.
    """

    def __init__(
        self,
        emb_dims: int,
        num_classes: int = 2,
        learn_gamma: bool = True,
        distill: bool = True,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.D = emb_dims
        self.K = num_classes
        self.distill = distill
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        # Learnable class centroids [K, D]
        self.centroids = nn.Parameter(torch.randn(num_classes, emb_dims))

        # Learnable temperature (initialized analytically)
        r_sq_init = 2.0
        r_sq_target = 1.0 - (2.0 / emb_dims)
        init_gamma = 20.0 / max(r_sq_init - r_sq_target, 1e-6)
        raw_gamma = torch.log(torch.expm1(torch.tensor(init_gamma)))
        if learn_gamma:
            self.gamma_raw = nn.Parameter(raw_gamma)
        else:
            self.register_buffer("gamma_raw", raw_gamma)

        # Fixed abstain-class bias (computed once from label_smoothing & K)
        r_sq_target_val = float(r_sq_target)
        if label_smoothing > 0:
            max_prob = 1.0 - label_smoothing + label_smoothing / num_classes
            min_prob = label_smoothing / num_classes
        else:
            max_prob, min_prob = 0.99, 0.01 / num_classes
        margin_ce = float(torch.log(torch.tensor(max_prob / min_prob)))
        t_ideal = init_gamma * (1.0 - r_sq_target_val)
        self.register_buffer("abstain_bias", torch.tensor(t_ideal - margin_ce))

    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute HALO loss.

        Args:
            embeddings: ``[B, D]`` — output of ``model.embed()``, not normalized.
            targets: ``[B]`` long tensor of class indices.

        Returns:
            Scalar loss (if ``reduction="mean"``) or ``[B]`` tensor.
        """
        targets = targets.view(-1).long()
        B, D = embeddings.shape
        gamma = F.softplus(self.gamma_raw)
        c = self.centroids  # [K, D]

        # Shifted logits: 2*(x·c)/D - ||c||²/D  (softmax-shift trick)
        x_sq = embeddings.pow(2).mean(dim=-1, keepdim=True)       # [B, 1]
        y_sq = c.pow(2).mean(dim=-1, keepdim=True)                # [K, 1]
        dot = (embeddings @ c.T) / D                               # [B, K]
        logits_k = gamma * (2.0 * dot - y_sq.T)                   # [B, K]

        # Abstain class logit (origin sink, no parameters)
        abstain = self.abstain_bias.expand(B, 1)                   # [B, 1]
        logits_kp1 = torch.cat([logits_k, abstain], dim=1)        # [B, K+1]

        # Soft target distillation
        if self.distill and self.label_smoothing > 0:
            logits_k_true = torch.clamp(logits_k - gamma * x_sq, max=0.0)
            is_correct = torch.zeros(B, self.K, device=embeddings.device, dtype=torch.bool)
            is_correct[torch.arange(B), targets] = True
            margin = logits_k_true / max(self.label_smoothing, 1e-8)
            target_logits = torch.where(is_correct, torch.zeros_like(margin), margin)
            target_probs_k = F.softmax(target_logits, dim=-1)
            # Append near-zero weight for abstain class
            abstain_prob = torch.zeros(B, 1, device=embeddings.device)
            target_probs = torch.cat([target_probs_k, abstain_prob], dim=1)
            target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            loss_ce = F.cross_entropy(logits_kp1, target_probs, reduction=self.reduction)
        else:
            loss_ce = F.cross_entropy(logits_kp1, targets, reduction=self.reduction)

        # Geometric radial regularizer ("soap bubble" term)
        c_true = c[targets]  # [B, D]
        r_sq = (embeddings - c_true).pow(2).mean(dim=-1)  # [B]
        r_sq = r_sq.clamp(min=1e-8)
        volume_coeff = 0.5 - 1.0 / D
        radial_nll = -(volume_coeff * torch.log(r_sq) - 0.5 * r_sq)

        if self.reduction == "mean":
            reg = radial_nll.mean()
        else:
            reg = radial_nll

        return loss_ce + reg


class OCSoftmaxLoss(nn.Module):
    """One-class softmax (Zhang, Jiang, Duan, IEEE SPL 2021).

    Built for a target class facing an open, unbounded other class: here the wake word
    against all speech, music and noise. A single learned direction ``w`` represents the wake
    word. Wake embeddings are pushed to cosine above ``m_wake`` with it and every other clip to
    cosine below ``m_other``; clips of the other class are never pulled towards each other or
    towards a center of their own.

        L = (1/N) * sum( softplus( alpha * margin ) )   over all N clips in the batch,
            margin = m_wake - cos for a wake clip, cos - m_other for every other clip

    Args:
        embed_dim: Embedding dimension.
        m_wake: Cosine a wake clip must exceed (default 0.9).
        m_other: Cosine every other clip must stay below (default 0.2).
        alpha: Scale (default 20.0).
    """

    def __init__(self, embed_dim: int, m_wake: float = 0.9, m_other: float = 0.2, alpha: float = 20.0) -> None:
        super().__init__()
        if not -1.0 <= m_other < m_wake <= 1.0:
            raise ValueError(f"need -1 <= m_other < m_wake <= 1, got {m_other}, {m_wake}")
        self.center = nn.Parameter(torch.randn(embed_dim))
        self.m_wake, self.m_other, self.alpha = m_wake, m_other, alpha

    @property
    def wake_prototype(self) -> torch.Tensor:
        """Normalised wake direction (shape ``(D,)``)."""
        return F.normalize(self.center.detach(), dim=0)

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos = F.normalize(embeds, dim=1) @ F.normalize(self.center, dim=0)
        wake = labels.view(-1).to(embeds.device) == 1
        margin = torch.where(wake, self.m_wake - cos, cos - self.m_other)
        return F.softplus(self.alpha * margin).mean()


class SizeAwareLoss(nn.Module):
    """Wraps a base loss with an L1 sparsity penalty and a parameter-count term.

    - L1 norm of all parameters: sparsity pressure, the only term with a gradient.
    - ``size_weight * max(0, param_count / param_budget - 1)``: a constant for a given
      architecture. It shifts the reported loss of an over-budget model but cannot shrink one;
      choosing the architecture does that.

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


def compute_neg_weight_schedule(
    step: int,
    total_steps: int,
    max_neg_weight: float,
    schedule: str = "linear",
) -> float:
    """Compute negative class weight at a given training step.

    Inspired by openWakeWord's technique of ramping negative weight
    from 1x to 1000x over training to suppress false positives.

    Args:
        step: Current training step (0-indexed).
        total_steps: Total number of training steps.
        max_neg_weight: Maximum negative weight at end of schedule.
        schedule: Schedule type — ``"linear"`` or ``"cosine"``.

    Returns:
        Negative weight value in ``[1.0, max_neg_weight]``.

    Raises:
        ValueError: If schedule is not ``"linear"`` or ``"cosine"``.
    """
    if total_steps <= 1:
        return max_neg_weight
    progress = min(step / max(1, total_steps - 1), 1.0)
    if schedule == "linear":
        return 1.0 + (max_neg_weight - 1.0) * progress
    elif schedule == "cosine":
        import math
        # Cosine schedule: slow start, fast middle, slow end
        return 1.0 + (max_neg_weight - 1.0) * 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unknown neg weight schedule: {schedule!r}. Use 'linear' or 'cosine'.")


class LossManager:
    """Manages multiple weighted loss functions for training."""

    def __init__(self, loss_configs: List[Dict[str, Any]], mining_type: str = "semihard",
                 device: str = "cpu",
                 neg_weight_schedule: Optional[str] = None,
                 max_neg_weight: float = 100.0) -> None:
        """
        Initialize the LossManager.

        Args:
            loss_configs: A list of dictionaries, each describing a loss:
                          `{"name": str, "weight": float, ...}`.
            mining_type: Triplet mining strategy.
            device: The torch device ('cpu' or 'cuda') where losses should reside.
            neg_weight_schedule: Dynamic negative weight schedule — ``"linear"``,
                ``"cosine"``, or ``None`` (disabled). When active, each not-wake clip's
                BCE term is multiplied by the current negative weight.
            max_neg_weight: Maximum negative class weight (default 100.0).

        Raises:
            ValueError: If an unknown loss name is encountered.
        """
        self.device = torch.device(device)
        self.losses: List[Dict[str, Any]] = []
        self.mining_type = mining_type
        self.neg_weight_schedule = neg_weight_schedule
        self.max_neg_weight = max_neg_weight
        self._current_neg_weight: float = 1.0
        self.spec_augment = None  # Set externally via set_spec_augment()

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
                    alpha=cfg.get("alpha", 0.90),
                    gamma=cfg.get("gamma", 2.0),
                ).to(self.device)
            elif name == "label_smoothing_bce":
                crit = LabelSmoothingBCE(
                    smoothing=cfg.get("smoothing", 0.03),
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
                    epsilon=cfg.get("epsilon", 0.1),
                ).to(self.device)
            elif name == "rppl":
                crit = RobustProtoDiversityLoss(
                    tau=cfg.get("tau", 0.1),
                    K_neg_proto=cfg.get("K_neg_proto", 0),
                    alpha=cfg.get("alpha", 1.0),
                    beta=cfg.get("beta", 1.0),
                    gamma=cfg.get("gamma", 0.5),
                    delta=cfg.get("delta", 0.1),
                    eta=cfg.get("eta", 0.5),
                    warmup_epochs=cfg.get("warmup_epochs", 5),
                    proto_ema_alpha=cfg.get("proto_ema_alpha", 0.05),
                    hard_div_threshold=cfg.get("hard_div_threshold", 0.1),
                    consistency_mode=cfg.get("consistency_mode", "proto"),
                    div_margin=cfg.get("div_margin", 1.0),
                ).to(self.device)
            elif name == "halo":
                crit = HALOLoss(
                    emb_dims=cfg.get("embed_dim", 128),
                    num_classes=cfg.get("num_classes", 2),
                    learn_gamma=cfg.get("learn_gamma", True),
                    distill=cfg.get("distill", True),
                    label_smoothing=cfg.get("label_smoothing", 0.1),
                    reduction=cfg.get("reduction", "mean"),
                ).to(self.device)
            elif name == "oc_softmax":
                crit = OCSoftmaxLoss(
                    embed_dim=cfg.get("embed_dim", 128),
                    m_wake=cfg.get("m_wake", 0.9),
                    m_other=cfg.get("m_other", 0.2),
                    alpha=cfg.get("alpha", 20.0),
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

    def parameters(self) -> List[nn.Parameter]:
        """Learnable parameters of the criteria: class centers, proxies, learned temperatures.

        They belong in the optimizer beside the model's; without them the centers of ArcFace,
        center, Proxy-NCA, HALO and OC-softmax never move from their random initial values.
        """
        return [p for e in self.losses for p in e["criterion"].parameters() if p.requires_grad]

    def update_neg_weight(self, step: int, total_steps: int) -> float:
        """Update the dynamic negative class weight for the current step.

        Call this once per batch when ``neg_weight_schedule`` is active.

        Args:
            step: Current global training step.
            total_steps: Total training steps across all epochs.

        Returns:
            The current negative weight value.
        """
        if self.neg_weight_schedule is not None:
            self._current_neg_weight = compute_neg_weight_schedule(
                step, total_steps, self.max_neg_weight, self.neg_weight_schedule,
            )
        return self._current_neg_weight

    def set_spec_augment(self, spec_augment: "SpectrogramAugment") -> None:
        """Attach a SpectrogramAugment instance for feature-level augmentation.

        When set, augmentation is applied to features inside ``compute_loss``
        during training.

        Args:
            spec_augment: A :class:`ww_trainer.augment.SpectrogramAugment` instance.
        """
        self.spec_augment = spec_augment

    def adjust_max_neg_weight(self, factor: float) -> None:
        """Multiply ``max_neg_weight`` by a factor (FPR-adaptive adjustment).

        Args:
            factor: Multiplier for the maximum negative weight.
        """
        self.max_neg_weight *= factor

    def step_epoch(self, epoch: int) -> None:
        """Notify epoch-aware loss criteria of the current epoch.

        Currently drives warmup scheduling for RPPL.
        """
        for entry in self.losses:
            crit = entry.get("criterion")
            if crit is not None and hasattr(crit, "set_epoch"):
                crit.set_epoch(epoch)

    def get_wake_prototype(self) -> Optional[torch.Tensor]:
        """Return the best available wake-class prototype vector for mining.

        Checks all active loss criteria in priority order:

        1. ``RobustProtoDiversityLoss`` — EMA prototype (most stable; accumulates
           over the full training run)
        2. ``OCSoftmaxLoss`` — learned wake direction
        3. ``ArcFaceLoss`` — learned angular class center for the wake class
        4. ``CenterLoss`` — learned Euclidean class center for the wake class
        5. ``ProxyNCALoss`` — learned proxy for the wake class

        Returns:
            Normalised ``(D,)`` tensor, or ``None`` if no suitable loss is active.
        """
        # Priority ordering: prefer the most stable representation first
        _PRIORITY = ("rppl", "oc_softmax", "arcface", "center", "proxy_nca")
        by_name = {
            entry["name"]: entry.get("criterion")
            for entry in self.losses
            if entry.get("criterion") is not None
        }
        for name in _PRIORITY:
            crit = by_name.get(name)
            if crit is None:
                continue
            proto = getattr(crit, "wake_prototype", None)
            if proto is not None:
                return proto
        return None

    @timed
    def compute_loss(self, model: nn.Module, wavs: torch.Tensor, labels: torch.Tensor,
                     dataset_ref: Optional[AudioDataset] = None,
                     text_token_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss and per-loss values based on the configured criteria.

        Args:
            model: The wake word model (must have ``embed`` for metric losses).
            wavs: Input audio waveforms (list of tensors or padded tensor).
            labels: Ground truth labels ``(B,)``.
            dataset_ref: Optional dataset reference for augmentation-based losses.
            text_token_ids: Optional ``[B, seq_len]`` int64 keyword phoneme IDs.
                When provided (multi-keyword training), passed to
                ``model.forward`` and ``model.embed`` so the text extractor
                runs per-batch.  ``None`` falls back to the precomputed cache.

        Returns:
            ``(total_loss, {"loss_name": float, ...})``
        """
        results: Dict[str, float] = {}
        total = torch.tensor(0.0, device=self.device)

        # Extract features once, optionally apply spectrogram augmentation
        if self.spec_augment is not None and model.training:
            from ww_trainer.feats import ensure_wav_list
            wavs_list = ensure_wav_list(wavs)
            feats = model.feature_extractor(wavs_list)
            if hasattr(model, '_apply_text_conditioning'):
                feats = model._apply_text_conditioning(feats, wavs_list, text_token_ids)
            feats = self.spec_augment(feats)
            if "phoneme_ids" in inspect.signature(model.classifier.forward).parameters:
                logits = model.classifier.forward(feats, phoneme_ids=text_token_ids)
                embeds = model.classifier.embed(feats, phoneme_ids=text_token_ids)
            else:
                logits = model.classifier.forward(feats)
                embeds = model.classifier.embed(feats)
        else:
            if text_token_ids is not None:
                logits = model(wavs, text_token_ids=text_token_ids)
                embeds = model.embed(wavs, text_token_ids=text_token_ids)
            else:
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
                # Binary Cross-Entropy Loss — with optional dynamic neg weight
                if self.neg_weight_schedule is not None and self._current_neg_weight > 1.0:
                    # the schedule weights the not-wake clips (false activations);
                    # pos_weight would weight the wake clips instead
                    w = torch.where(labels_float > 0.5, 1.0, self._current_neg_weight)
                    loss_val = F.binary_cross_entropy_with_logits(
                        logits.view_as(labels_float), labels_float, weight=w,
                    )
                else:
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
                # wake anchors only: a not-wake anchor's "hardest positive" is the farthest
                # other not-wake clip, and pulling it in would compact the open class
                wake = labels.view(-1).to(self.device) == 1
                d_ap, d_an = d_ap[wake], d_an[wake]
                if int(wake.sum()) >= 2 and not bool(wake.all()):
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

            elif name in ("arcface", "center", "proxy_nca", "oc_softmax"):
                loss_val = crit(embeds, labels.to(self.device).view(-1))

            elif name in ("ntxent", "supcon", "multi_similarity"):
                # Need at least 1 pos and 1 neg in batch
                labs = labels.to(self.device).view(-1)
                if (labs == 1).any() and (labs == 0).any():
                    loss_val = crit(embeds, labs)
                else:
                    import logging as _logging
                    _logging.getLogger(__name__).debug(
                        "Skipping %s — batch has no %s samples; no gradient this step.",
                        name, "positive" if not (labs == 1).any() else "negative",
                    )

            elif name in ("lse", "contrastive", "angular"):
                labs = labels.to(self.device).view(-1)
                if (labs == 1).any() and (labs == 0).any():
                    loss_val = crit(embeds, labs)
                else:
                    import logging as _logging
                    _logging.getLogger(__name__).debug(
                        "Skipping %s — batch has no %s samples; no gradient this step.",
                        name, "positive" if not (labs == 1).any() else "negative",
                    )

            elif name == "halo":
                loss_val = crit(embeds, labels.to(self.device).view(-1))

            elif name == "size_aware":
                # SizeAwareLoss needs the model reference
                loss_val = crit(logits.view(-1), labels.to(self.device).float().view(-1), model)

            elif name == "rppl":
                # Robust Prototype and Diversity Loss.
                # Augmented embeddings MUST flow gradients — the consistency loss
                # is what makes the model invariant to augmentation. Computing
                # them under no_grad would silently zero the consistency gradient.
                aug_embeds = None
                if dataset_ref is not None and hasattr(dataset_ref, "get_augmented"):
                    aug_embeds_list = []
                    for i, w in enumerate(wavs):
                        aug_w = dataset_ref.get_augmented(w).to(self.device)
                        kw = text_token_ids[i:i+1] if text_token_ids is not None else None
                        emb = model.embed(aug_w.unsqueeze(0), text_token_ids=kw).squeeze(0)
                        aug_embeds_list.append(emb)
                    aug_embeds = torch.stack(aug_embeds_list, dim=0)

                loss_val = crit(logits, labels.view(-1), embeds, aug_embeds)
                # Log RPPL sub-components
                results["rppl_bce"]    = crit.last_bce
                results["rppl_proto"]  = crit.last_proto
                results["rppl_div"]    = crit.last_div
                results["rppl_center"] = crit.last_center
                results["rppl_cons"]   = crit.last_cons
                results["rppl_geo_scale"]      = crit.last_geo_scale
                results["rppl_proto_ema_norm"] = crit.last_proto_ema_norm

            total += weight * loss_val
            results[name] = float(loss_val.item())

        results["total"] = float(total.item())
        return total, results