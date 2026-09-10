"""Tests for ESP32 tiers, SizeAwareLoss, and micro search utilities."""
import pytest
import torch
import torch.nn as nn

from ww_trainer.tiers import get_tier, HARDWARE_TIERS, TierConfig
from ww_trainer.loss import SizeAwareLoss
from ww_trainer.sweep import _build_micro_search_space, _estimate_ffn_params


# ---- ESP32 tiers ----

@pytest.mark.parametrize("name", ["esp32_nano", "esp32_sweet", "esp32_max"])
def test_esp32_tier_exists(name: str) -> None:
    tier = get_tier(name)
    assert isinstance(tier, TierConfig)
    assert tier.name == name
    assert tier.max_params is not None
    assert tier.max_size_kb is not None


def test_esp32_nano_fields() -> None:
    tier = get_tier("esp32_nano")
    assert tier.extractor_type == "mfcc"
    assert tier.head_arch == "ffn"
    assert tier.hidden_dim == 16
    assert tier.n_mfcc == 13
    assert tier.max_params == 1024
    assert tier.max_size_kb == 1.0


def test_esp32_sweet_fields() -> None:
    tier = get_tier("esp32_sweet")
    assert tier.hidden_dim == 64
    assert tier.max_params == 10240


def test_esp32_max_fields() -> None:
    tier = get_tier("esp32_max")
    assert tier.hidden_dim == 128
    assert tier.max_params == 51200


def test_esp32_tiers_ordered_by_budget() -> None:
    nano = get_tier("esp32_nano")
    sweet = get_tier("esp32_sweet")
    maxt = get_tier("esp32_max")
    assert nano.max_params < sweet.max_params < maxt.max_params


# ---- ESP32 tier model instantiation ----

@pytest.mark.parametrize("name", ["esp32_nano", "esp32_sweet", "esp32_max"])
def test_esp32_tier_model_instantiation(name: str) -> None:
    """ESP32 tiers can be instantiated and produce output."""
    from ww_trainer.feats import MfccExtractor
    from ww_trainer.model import FfnClassifierHead, BaseWakeModel

    tier = get_tier(name)
    extractor = MfccExtractor(n_mfcc=tier.n_mfcc)
    head = FfnClassifierHead(input_size=tier.n_mfcc, hidden_dim=tier.hidden_dim, device="cpu")
    model = BaseWakeModel(feature_extractor=extractor, classifier=head, device="cpu")

    dummy = torch.randn(1, 16000)
    logits = model(dummy)
    assert logits.shape == (1,)


def test_esp32_nano_within_budget() -> None:
    """Nano tier default config fits within param budget."""
    from ww_trainer.feats import MfccExtractor
    from ww_trainer.model import FfnClassifierHead

    tier = get_tier("esp32_nano")
    head = FfnClassifierHead(input_size=tier.n_mfcc, hidden_dim=tier.hidden_dim, device="cpu")
    n_params = sum(p.numel() for p in head.parameters())
    assert n_params <= tier.max_params


# ---- SizeAwareLoss ----

def test_size_aware_loss_basic() -> None:
    """SizeAwareLoss produces a finite, backprop-able scalar."""
    base = nn.BCEWithLogitsLoss()
    loss_fn = SizeAwareLoss(base_loss=base, l1_weight=1e-5, size_weight=0.1, param_budget=1024)

    model = nn.Linear(13, 1)
    logits = model(torch.randn(4, 13)).squeeze(-1)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss = loss_fn(logits, labels, model)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    loss.backward()


def test_size_aware_loss_increases_with_larger_model() -> None:
    """Size penalty should be larger for a model that exceeds the budget."""
    base = nn.BCEWithLogitsLoss()
    logits = torch.zeros(4)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

    small_model = nn.Linear(4, 1)  # 5 params
    large_model = nn.Linear(64, 1)  # 65 params

    loss_small = SizeAwareLoss(base, l1_weight=0, size_weight=1.0, param_budget=10)
    loss_large = SizeAwareLoss(base, l1_weight=0, size_weight=1.0, param_budget=10)

    # Use model's own output for fair comparison
    with torch.no_grad():
        s_logits = small_model(torch.randn(4, 4)).squeeze(-1)
        l_logits = large_model(torch.randn(4, 64)).squeeze(-1)

    val_s = loss_small(s_logits, labels, small_model)
    val_l = loss_large(l_logits, labels, large_model)
    # Large model should have higher size penalty
    assert val_l > val_s


def _train_all_ones_linear(l1_weight: float, seed: int) -> float:
    torch.manual_seed(seed)
    model = nn.Linear(8, 1, bias=False)
    nn.init.ones_(model.weight)
    base = nn.BCEWithLogitsLoss()
    loss_fn = SizeAwareLoss(base, l1_weight=l1_weight, size_weight=0, param_budget=100)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(50):
        logits = model(torch.randn(4, 8)).squeeze(-1)
        labels = torch.zeros(4)
        loss = loss_fn(logits, labels, model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.weight.abs().mean().item()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_size_aware_loss_l1_encourages_sparsity(seed: int) -> None:
    """The L1 term must pull weights closer to zero than the same run without it."""
    without_l1 = _train_all_ones_linear(0.0, seed)
    with_l1 = _train_all_ones_linear(0.1, seed)
    assert with_l1 < without_l1
    assert with_l1 < 1.0


# ---- Micro search space ----

@pytest.mark.parametrize("tier_name", ["esp32_nano", "esp32_sweet", "esp32_max"])
def test_micro_search_space_valid(tier_name: str) -> None:
    space = _build_micro_search_space(tier_name)
    assert "arch" in space
    assert space["arch"] == ["ffn"]  # only FFN for ESP32
    assert "hidden_dim" in space
    assert "n_features" in space


def test_micro_search_space_rejects_non_esp32() -> None:
    with pytest.raises(ValueError, match="no param budget"):
        _build_micro_search_space("micro")


@pytest.mark.parametrize("tier_name", ["esp32_nano", "esp32_sweet", "esp32_max"])
def test_micro_search_space_all_configs_within_budget(tier_name: str) -> None:
    """Every combination in the search space should fit within the tier budget."""
    tier = get_tier(tier_name)
    space = _build_micro_search_space(tier_name)
    for n_feat in space["n_features"]:
        for hdim in space["hidden_dim"]:
            params = _estimate_ffn_params(n_feat, hdim)
            assert params <= tier.max_params, (
                f"{tier_name}: n_feat={n_feat}, hdim={hdim} -> {params} params > {tier.max_params}"
            )


# ---- Param estimation ----

def test_estimate_ffn_params_known_values() -> None:
    # Linear(13, 16) + Linear(16, 1) = 13*16+16 + 16*1+1 = 241
    assert _estimate_ffn_params(13, 16) == 241


def test_estimate_ffn_params_matches_actual() -> None:
    """Estimation should match actual PyTorch parameter count."""
    from ww_trainer.model import FfnClassifierHead
    for n_mfcc, hdim in [(13, 16), (13, 64), (20, 32)]:
        head = FfnClassifierHead(input_size=n_mfcc, hidden_dim=hdim, dropout=0, device="cpu")
        actual = sum(p.numel() for p in head.parameters())
        estimated = _estimate_ffn_params(n_mfcc, hdim)
        assert estimated == actual, f"n_mfcc={n_mfcc}, hdim={hdim}: est={estimated}, actual={actual}"
