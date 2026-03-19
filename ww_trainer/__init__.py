"""ww-trainer: Full training and research suite for wake words.

All exported to ONNX — from MFCC on microcontrollers to HuBERT on servers.
"""
from ww_trainer.version import VERSION_STR as __version__

# --- Feature Extractors ---
from ww_trainer.feats import (
    BaseExtractor,
    OnnxFeatureExtractor,
    MfccExtractor,
    FilterbankExtractor,
    SincNetExtractor,
    GammatoneExtractor,
    DeltaExtractor,
    LEAFExtractor,
    PLPExtractor,
    PNCCExtractor,
    CQTExtractor,
    VoiceActivityExtractor,
    PitchExtractor,
    MultiResolutionExtractor,
    SNRAwareExtractor,
    SlidingFeatureCacheTensor,
)

# --- Classifier Heads ---
from ww_trainer.model import (
    ClassifierHead,
    BaseWakeModel,
    FfnClassifierHead,
    CnnClassifierHead,
    GruClassifierHead,
    BCResNetHead,
    TCResNetHead,
    DSCNNHead,
    MatchboxNetHead,
    Res15Head,
    KWTHead,
    ConformerHead,
    CRNNHead,
    AttentionPooling,
)

# --- Losses ---
from ww_trainer.loss import (
    LossManager,
    FocalLoss,
    LabelSmoothingBCE,
    ArcFaceLoss,
    CenterLoss,
    NTXentLoss,
    SupConLoss,
    ProxyNCALoss,
    MultiSimilarityLoss,
    ContrastiveLoss,
    LiftedStructureLoss,
    AngularLoss,
    SoftTripletLoss,
    CN2Plus1PairLoss,
    RobustProtoDiversityLoss,
)

# --- Inference ---
from ww_trainer.inference import OnnxWakeWordInferencer

# --- Dataset ---
from ww_trainer.dataset import AudioDataset, collate_fn

# --- Tiers ---
from ww_trainer.tiers import HARDWARE_TIERS, TierConfig

# --- Distillation ---
from ww_trainer.distill import CnnLstmExtractor, KnowledgeDistillationTrainer

# --- Metrics ---
from ww_trainer.metrics import (
    DetectionReport,
    compute_eer,
    compute_far_frr,
    find_optimal_threshold,
    det_curve,
    classification_report,
)

# --- Factory ---
from ww_trainer.factory import create_model, EXTRACTOR_REGISTRY, HEAD_REGISTRY

# --- Evaluation ---
from ww_trainer.evaluation import evaluate_model, evaluate_detection, log_metrics_csv

# --- Augmentation ---
from ww_trainer.augment import (
    AugmentationPipeline,
    AudioTransform,
    MixBackground,
    ApplyReverb,
    PitchShift,
    SpeedPerturb,
    GaussianNoise,
    VolumePerturb,
    TimeShift,
    SpecAugment,
    Normalize,
)
