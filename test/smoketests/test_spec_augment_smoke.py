"""Smoke tests for SpectrogramAugment frequency+time masking."""
import torch

from ww_trainer.augment import SpectrogramAugment


class TestSpectrogramAugment:
    """Tests for SpectrogramAugment on feature tensors."""

    def test_output_shape_batched(self) -> None:
        aug = SpectrogramAugment(n_time_masks=2, max_time_width=5,
                                 n_freq_masks=2, max_freq_width=3)
        feats = torch.randn(4, 50, 40)
        result = aug(feats)
        assert result.shape == (4, 50, 40)

    def test_output_shape_unbatched(self) -> None:
        aug = SpectrogramAugment()
        feats = torch.randn(50, 40)
        result = aug(feats)
        assert result.shape == (50, 40)

    def test_masking_zeroes_values(self) -> None:
        aug = SpectrogramAugment(n_time_masks=1, max_time_width=50,
                                 n_freq_masks=1, max_freq_width=40)
        feats = torch.ones(1, 50, 40)
        result = aug(feats)
        # At least some values should be zeroed
        assert (result == 0.0).any()

    def test_no_corruption_beyond_masking(self) -> None:
        aug = SpectrogramAugment(n_time_masks=0, max_time_width=1,
                                 n_freq_masks=0, max_freq_width=1)
        feats = torch.randn(2, 50, 40)
        result = aug(feats)
        # With 0 masks, output should equal input
        assert torch.allclose(result, feats)

    def test_does_not_modify_input(self) -> None:
        aug = SpectrogramAugment()
        feats = torch.randn(2, 50, 40)
        original = feats.clone()
        _ = aug(feats)
        assert torch.allclose(feats, original)

    def test_small_features(self) -> None:
        aug = SpectrogramAugment(max_time_width=100, max_freq_width=100)
        feats = torch.randn(1, 3, 2)
        result = aug(feats)
        assert result.shape == (1, 3, 2)
