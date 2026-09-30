"""Losses against their published definitions, the one-class treatment of not-wake clips, and
the learned parameters of a loss reaching the optimizer."""
import math

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F

import ww_trainer.infinite_loop as infinite_loop
import ww_trainer.loop as loop
from ww_trainer.loss import (AngularLoss, ArcFaceLoss, CenterLoss, ContrastiveLoss, FocalLoss, LabelSmoothingBCE,
                             LiftedStructureLoss, LossManager, MultiSimilarityLoss, NTXentLoss, SupConLoss)
from ww_trainer.utils import pairwise_distance


def _unit(*v):
    return F.normalize(torch.tensor(v, dtype=torch.float32), dim=-1)


def _wake_and_spread_negatives(spread):
    """Two wake clips in the plane of axes 0 and 3, three not-wake clips in the plane of axes 1 and 2.

    Every wake/not-wake cosine is 0 whatever ``spread`` is, so only the not-wake clips'
    positions relative to each other change with it.
    """
    wake = [_unit(1, 0, 0, 0.05), _unit(1, 0, 0, -0.05)]
    neg = [_unit(0, math.cos(a), math.sin(a), 0) for a in (0.0, spread, 2 * spread)]
    return torch.stack(wake + neg), torch.tensor([1, 1, 0, 0, 0])


@pytest.mark.parametrize("loss", [ContrastiveLoss(margin=0.5), SupConLoss(), NTXentLoss(),
                                  MultiSimilarityLoss(), LiftedStructureLoss(margin=0.5)],
                         ids=lambda l: type(l).__name__)
def test_not_wake_clips_are_not_pulled_towards_each_other(loss):
    tight, labels = _wake_and_spread_negatives(0.05)
    wide, _ = _wake_and_spread_negatives(1.2)
    assert torch.allclose(loss(tight, labels), loss(wide, labels), atol=1e-5)


def test_center_loss_pulls_only_keyword_clips():
    c = CenterLoss(embed_dim=4)
    e, labels = _wake_and_spread_negatives(0.05)
    moved = e.clone()
    moved[2:] *= 7.0
    assert torch.allclose(c(e, labels), c(moved, labels))


def test_angular_loss_is_zero_when_positives_are_closer_and_pulls_them_in_otherwise():
    good = torch.stack([_unit(1, 0, 0), _unit(1, 0.1, 0), _unit(0, 1, 0), _unit(0, 0, 1)])
    labels = torch.tensor([1, 1, 0, 0])
    assert AngularLoss(margin=0.2)(good, labels).item() == 0.0
    bad = torch.stack([_unit(1, 0, 0), _unit(0, 0, 1), _unit(1, 0.2, 0), _unit(0, 1, 0)]).requires_grad_()
    loss = AngularLoss(margin=0.2)(bad, labels)
    assert loss.item() > 0
    loss.backward()
    with torch.no_grad():
        step = F.normalize(bad - 0.1 * bad.grad, dim=1)
    gap = lambda e: (F.cosine_similarity(e[0], e[1], dim=0) - F.cosine_similarity(e[0], e[2], dim=0)).item()
    assert gap(step) > gap(bad.detach())


def _lifted_reference(e, labels, margin):
    D = pairwise_distance(e)
    terms = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j] == 1:
                s = sum(math.exp(margin - D[i, k]) for k in range(n) if labels[k] != labels[i])
                s += sum(math.exp(margin - D[j, l]) for l in range(n) if labels[l] != labels[j])
                terms.append(max(0.0, math.log(s) + D[i, j].item()) ** 2)
    return 0.5 * sum(terms) / len(terms)


def test_lifted_structure_matches_the_paper():
    g = torch.Generator().manual_seed(0)
    e = torch.randn(7, 5, generator=g)
    labels = torch.tensor([1, 1, 1, 0, 0, 0, 0])
    assert math.isclose(LiftedStructureLoss(margin=1.0)(e, labels).item(), _lifted_reference(e, labels, 1.0),
                        rel_tol=1e-5)


def _ms_reference(e, labels, a=2.0, b=50.0, lam=0.5, eps=0.1):
    S = F.normalize(e, dim=1) @ F.normalize(e, dim=1).T
    terms = []
    for i in range(len(labels)):
        if labels[i] != 1:
            continue
        pos = [S[i, k].item() for k in range(len(labels)) if k != i and labels[k] == labels[i]]
        neg = [S[i, k].item() for k in range(len(labels)) if labels[k] != labels[i]]
        hp = [s for s in pos if s - eps < max(neg)]
        hn = [s for s in neg if s + eps > min(pos)]
        if not hp and not hn:
            continue
        terms.append(math.log(1 + sum(math.exp(-a * (s - lam)) for s in hp)) / a
                     + math.log(1 + sum(math.exp(b * (s - lam)) for s in hn)) / b)
    return sum(terms) / len(terms)


def test_multi_similarity_matches_the_paper_and_is_never_negative():
    g = torch.Generator().manual_seed(1)
    for _ in range(5):
        e = torch.randn(8, 6, generator=g)
        labels = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0])
        got = MultiSimilarityLoss()(e, labels).item()
        assert got >= 0.0
        assert math.isclose(got, _ms_reference(e, labels), rel_tol=1e-5)
    # one tight wake pair far from every negative: the positive term alone was negative before
    e = torch.stack([_unit(1, 0, 0), _unit(1, 0.01, 0), _unit(-1, 0, 0), _unit(-1, 0.2, 0.1)])
    assert MultiSimilarityLoss()(e, torch.tensor([1, 1, 0, 0])).item() >= 0.0


def test_arcface_keeps_a_gradient_for_the_most_misclassified_clips():
    torch.manual_seed(0)
    arc = ArcFaceLoss(embed_dim=2, margin=0.5)
    with torch.no_grad():
        arc.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    # a wake clip at cosine -0.95 to the wake center: theta = 2.83 > pi - margin
    e = torch.tensor([[0.95, math.sqrt(1 - 0.95 ** 2)]], requires_grad=True)
    arc(e, torch.tensor([1])).backward()
    assert arc.weight.grad[1].abs().sum() > 0


def test_the_negative_weight_schedule_weights_the_not_wake_clips():
    lm = LossManager([{"name": "bce"}], neg_weight_schedule="linear", max_neg_weight=10.0)
    lm.update_neg_weight(99, 100)
    logits, labels = torch.tensor([3.0, 3.0]), torch.tensor([1.0, 0.0])  # the not-wake clip is a false alarm
    ce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

    class Fixed(torch.nn.Module):
        training = False

        def forward(self, wavs):
            return logits.view(-1, 1)

        def embed(self, wavs):
            return torch.zeros(2, 4)

    total, _ = lm.compute_loss(Fixed(), torch.zeros(2, 10), labels)
    assert math.isclose(total.item(), ((ce[0] + 10.0 * ce[1]) / 2).item(), rel_tol=1e-5)


def test_the_manager_builds_focal_and_label_smoothing_with_the_class_defaults():
    lm = LossManager([{"name": "focal"}, {"name": "label_smoothing_bce"}])
    assert lm.losses[0]["criterion"].alpha == FocalLoss().alpha
    assert lm.losses[1]["criterion"].smoothing == LabelSmoothingBCE().smoothing


def test_oc_softmax_is_low_when_wake_clears_its_margin_and_others_stay_below_theirs():
    from ww_trainer.loss import OCSoftmaxLoss

    oc = OCSoftmaxLoss(embed_dim=3)
    with torch.no_grad():
        oc.center.copy_(torch.tensor([1.0, 0.0, 0.0]))
    good = torch.stack([_unit(1, 0.1, 0), _unit(0, 1, 0), _unit(0, 0, 1), _unit(-1, 0.3, 0)])
    bad = torch.stack([_unit(0, 1, 0), _unit(1, 0.1, 0), _unit(0, 0, 1), _unit(-1, 0.3, 0)])
    labels = torch.tensor([1, 0, 0, 0])
    assert oc(good, labels).item() < 0.05 < oc(bad, labels).item()
    assert LossManager([{"name": "oc_softmax", "embed_dim": 3}]).get_wake_prototype().shape == (3,)
    with pytest.raises(ValueError):
        OCSoftmaxLoss(embed_dim=3, m_wake=0.2, m_other=0.9)


@pytest.fixture
def tiny_dataset(tmp_path):
    data = []
    for i in range(8):
        t = np.linspace(0, 0.5, 8000)
        p = tmp_path / f"a{i}.wav"
        sf.write(str(p), np.sin(2 * np.pi * (440 + i * 100) * t).astype(np.float32), 16000)
        data.append((str(p), "1" if i < 4 else "0"))
    return data


@pytest.mark.parametrize("name, attr", [("arcface", "weight"), ("center", "centers"), ("proxy_nca", "proxies"),
                                        ("oc_softmax", "center")])
def test_a_losss_learned_parameters_train_with_the_model(tiny_dataset, tmp_path, monkeypatch, name, attr):
    from ww_trainer.trainer import WakeWordTrainer

    made = []

    class Spy(LossManager):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append((self, getattr(self.losses[1]["criterion"], attr).detach().clone()))

    monkeypatch.setattr(loop, "LossManager", Spy)
    trainer = WakeWordTrainer(arch="ffn", featurizer="", feature_dim=None, featurizer_type="mfcc", device="cpu",
                              losses_cfg=[{"name": "bce", "weight": 1.0}, {"name": name, "weight": 1.0}],
                              n_mfcc=13, hidden_dim=32)
    emb = trainer.model.embed(torch.zeros(1, 8000)).shape[-1]
    trainer.losses_cfg[1]["embed_dim"] = emb
    torch.manual_seed(0)
    trainer.train(train_data=tiny_dataset, test_data=tiny_dataset, epochs=2, batch_size=4, lr=1e-2,
                  output_dir=str(tmp_path / "out"), save_best="f1", tsne_every=0, pca_every=0, umap_every=0)
    manager, initial = made[0]
    after = getattr(manager.losses[1]["criterion"], attr).detach()
    assert not torch.equal(after, initial), f"{name}.{attr} never moved: it is not in the optimizer"


def test_resuming_a_checkpoint_saved_with_fewer_parameter_groups_restarts_the_optimizer(tmp_path, caplog):
    from ww_trainer.checkpoint import load_checkpoint

    model = torch.nn.Linear(2, 1)
    torch.save(model.state_dict(), tmp_path / "m.pt")
    old = torch.optim.Adam(model.parameters())
    torch.save({"epoch": 3, "metrics": {"f1": 0.5}, "optimizer_state": old.state_dict()}, tmp_path / "m.ts")
    new = torch.optim.Adam([{"params": model.parameters()}, {"params": [torch.nn.Parameter(torch.zeros(3))]}])
    assert load_checkpoint(model, tmp_path / "m.pt", "cpu", new) == (3, {"f1": 0.5})
    assert "parameter groups" in caplog.text


def test_resuming_with_a_same_size_group_of_different_shape_restarts_the_optimizer(tmp_path, caplog):
    """--loss-type arcface,center resumed from an arcface,center checkpoint saved with a
    different loss set: both sides have 2 groups, but the second group's parameter count
    (1 center vs 2) differs, which torch's own ``load_state_dict`` raises on."""
    from ww_trainer.checkpoint import load_checkpoint

    model = torch.nn.Linear(2, 1)
    torch.save(model.state_dict(), tmp_path / "m.pt")
    old = torch.optim.Adam([{"params": model.parameters()}, {"params": [torch.nn.Parameter(torch.zeros(3))]}])
    torch.save({"epoch": 5, "metrics": {"f1": 0.6}, "optimizer_state": old.state_dict()}, tmp_path / "m.ts")
    new = torch.optim.Adam([
        {"params": model.parameters()},
        {"params": [torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(3))]},
    ])
    assert load_checkpoint(model, tmp_path / "m.pt", "cpu", new) == (5, {"f1": 0.6})
    assert "parameter groups" in caplog.text


def test_pair_loss_fires_only_with_mixed_labels_and_moves_embeddings():
    class Fixed(torch.nn.Module):
        training = False

        def __init__(self, embeds):
            super().__init__()
            self._embeds = embeds

        def forward(self, wavs):
            return torch.zeros(len(wavs), 1)

        def embed(self, wavs):
            return self._embeds

    # two wake clips close together, two not-wake clips close together and far from wake:
    # a margin wider than the wake/not-wake gap forces the hinge to fire.
    mixed = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]], requires_grad=True)
    lm = LossManager([{"name": "pair", "margin": 3.0}])
    loss, _ = lm.compute_loss(Fixed(mixed), torch.zeros(4, 10), torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert loss.item() > 0
    loss.backward()
    assert mixed.grad is not None and mixed.grad.abs().sum().item() > 0

    single = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, -0.1], [0.7, 0.2]])
    loss2, _ = LossManager([{"name": "pair", "margin": 3.0}]).compute_loss(
        Fixed(single), torch.zeros(4, 10), torch.tensor([1.0, 1.0, 1.0, 1.0]),
    )
    assert loss2.item() == 0.0


def test_arcface_fallback_matches_the_exact_reference_value_past_pi_minus_margin():
    torch.manual_seed(0)
    margin, scale = 0.5, 30.0
    arc = ArcFaceLoss(embed_dim=2, margin=margin, scale=scale)
    with torch.no_grad():
        arc.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    # cosine to the wake center (class 1) is -0.95: theta = acos(-0.95) > pi - margin
    e = torch.tensor([[0.95, math.sqrt(1 - 0.95 ** 2)]])
    labels = torch.tensor([1])

    cos_t = -0.95
    assert cos_t < math.cos(math.pi - margin)  # confirms the fallback branch is the one under test
    sin_t = math.sqrt(1.0 - cos_t ** 2)
    target_cos = cos_t * math.cos(margin) - sin_t * math.sin(margin)  # cos(theta + m), for reference
    fallback_cos = cos_t - margin * math.sin(margin)  # the reference implementation's fallback
    logits = torch.tensor([0.95, fallback_cos]) * scale
    expected = F.cross_entropy(logits.unsqueeze(0), labels).item()

    got = arc(e, labels).item()
    assert math.isclose(got, expected, rel_tol=1e-5)
    # a plain cos(theta+m) (no fallback) would give a different value on this clip
    assert not math.isclose(got, F.cross_entropy((torch.tensor([0.95, target_cos]) * scale).unsqueeze(0),
                                                  labels).item(), rel_tol=1e-3)


def test_halo_through_the_manager_runs_a_step_and_its_parameters_change():
    from ww_trainer.loss import HALOLoss

    class Fixed(torch.nn.Module):
        training = True

        def __init__(self, embed_dim):
            super().__init__()
            self.linear = torch.nn.Linear(4, embed_dim)

        def forward(self, wavs):
            return self.linear(wavs)[:, :1]

        def embed(self, wavs):
            return self.linear(wavs)

    torch.manual_seed(0)
    model = Fixed(embed_dim=3)
    lm = LossManager([{"name": "bce", "weight": 1.0}, {"name": "halo", "weight": 1.0, "embed_dim": 3}])
    halo = lm.losses[1]["criterion"]
    assert isinstance(halo, HALOLoss)
    initial = halo.centroids.detach().clone()

    optimizer = torch.optim.SGD(list(model.parameters()) + lm.parameters(), lr=0.5)
    wavs = torch.randn(6, 4)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])  # float, as the dataset produces them

    loss, _ = lm.compute_loss(model, wavs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.equal(halo.centroids.detach(), initial)


def test_infinite_training_loop_moves_a_losss_learned_parameter(tmp_path, monkeypatch):
    from ww_trainer.trainer import WakeWordTrainer

    def _wavs(n, prefix, label):
        out = []
        for i in range(n):
            t = np.linspace(0, 0.5, 8000)
            p = tmp_path / f"{prefix}{i}.wav"
            sf.write(str(p), np.sin(2 * np.pi * (440 + i * 100) * t).astype(np.float32), 16000)
            out.append((str(p), label))
        return out

    wakes = _wavs(3, "w", "1")
    nww_pool = _wavs(6, "n", "0")
    test_data = wakes[:2] + nww_pool[:2]

    made = []

    class Spy(LossManager):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append((self, self.losses[1]["criterion"].centers.detach().clone()))

    monkeypatch.setattr(infinite_loop, "LossManager", Spy)
    trainer = WakeWordTrainer(arch="ffn", featurizer="", feature_dim=None, featurizer_type="mfcc", device="cpu",
                              losses_cfg=[{"name": "bce", "weight": 1.0}, {"name": "center", "weight": 1.0}],
                              n_mfcc=13, hidden_dim=32)
    emb = trainer.model.embed(torch.zeros(1, 8000)).shape[-1]
    trainer.losses_cfg[1]["embed_dim"] = emb

    torch.manual_seed(0)
    infinite_loop.infinite_training_loop(
        trainer, str(tmp_path / "out"), wakes, nww_pool, test_data,
        goal=infinite_loop.StoppingGoal(max_epochs=1, min_epochs=1, target_f1=None, target_eer=None),
        scan_size=10, batch_size=4, spec_augment=False, use_mixup=False,
        vc_per_epoch=0, eval_every=1, readiness_every=1, resume_cache=False,
    )

    manager, initial = made[0]
    after = manager.losses[1]["criterion"].centers.detach()
    assert not torch.equal(after, initial), "center loss never moved: infinite_training_loop's optimizer or device is wrong"
