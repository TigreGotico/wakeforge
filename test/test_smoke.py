"""Smoke tests — verify all public symbols can be imported without error."""
import pytest
import torch


def test_import_feats():
    from ww_trainer.feats import (
        BaseExtractor,
        OnnxFeatureExtractor,
        SlidingFeatureCacheTensor,
        WavInput,
        ensure_wav_list,
    )
    assert BaseExtractor is not None
    assert OnnxFeatureExtractor is not None
    assert SlidingFeatureCacheTensor is not None


def test_import_model():
    from ww_trainer.model import (
        BaseWakeModel,
        ClassifierHead,
        CnnClassifierHead,
        FfnClassifierHead,
        GruClassifierHead,
    )
    assert BaseWakeModel is not None
    assert ClassifierHead is not None


def test_import_loss():
    from ww_trainer.loss import (
        AngularLoss,
        CN2Plus1PairLoss,
        ContrastiveLoss,
        LiftedStructureLoss,
        LossManager,
        RobustProtoDiversityLoss,
        SoftTripletLoss,
    )
    assert LossManager is not None


def test_import_dataset():
    from ww_trainer.dataset import AudioDataset, collate_fn
    assert AudioDataset is not None
    assert collate_fn is not None


def test_import_utils():
    from ww_trainer.utils import (
        compute_metric_statistics,
        get_hard_pair_distances,
        pairwise_cosine_similarity,
        pairwise_distance,
        sample_semihard_triplets,
        sample_triplets,
        triplet_violation_fraction,
    )
    assert pairwise_distance is not None


def test_ensure_wav_list_1d():
    from ww_trainer.feats import ensure_wav_list
    wav = torch.zeros(16000)
    result = ensure_wav_list(wav)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].shape == (16000,)


def test_ensure_wav_list_2d():
    from ww_trainer.feats import ensure_wav_list
    wav = torch.zeros(3, 16000)
    result = ensure_wav_list(wav)
    assert len(result) == 3


def test_sliding_feature_cache():
    from ww_trainer.feats import SlidingFeatureCacheTensor
    cache = SlidingFeatureCacheTensor(feature_dim=768, window_size=50)
    feats = torch.randn(10, 768)
    out = cache(feats)
    assert out.shape == (10, 768)

    feats2 = torch.randn(45, 768)
    out2 = cache(feats2)
    assert out2.shape[0] == 50  # window full
    assert out2.shape[1] == 768
