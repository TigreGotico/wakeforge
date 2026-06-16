"""ww-trainer: Full training and research suite for wake words.

All exported to ONNX — from MFCC on microcontrollers to HuBERT on servers.
"""
from ww_trainer.version import __version__

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
    MarkovTransitionExtractor,
    HMMStateExtractor,
    OnnxTextExtractor,
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
    ConvAttentionHead,
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
from ww_trainer.inference import OnnxWakeWordInferencer, OnnxStreamingWakeWord

# --- Dataset ---
from ww_trainer.dataset import AudioDataset, collate_fn

# --- Tiers ---
from ww_trainer.tiers import HARDWARE_TIERS, TierConfig

# --- PhonMatchNet text encoder + cross-attention head ---
from ww_trainer.phonmatch import (
    IPA_VOCAB,
    ARPABET_TO_IPA,
    PHONEME_VOCAB,   # alias for IPA_VOCAB — backward compat
    VOCAB_SIZE,
    ipa_to_ids,
    arpabet_to_ids,
    phonemes_to_ids,  # alias for arpabet_to_ids — backward compat
    PhonMatchTextEncoder,
    PhonMatchHead,
)

# --- Distillation ---
from ww_trainer.distill import CnnLstmExtractor, KnowledgeDistillationTrainer

# --- Metrics ---
from ww_trainer.metrics import (
    DetectionReport,
    compute_eer,
    compute_far_frr,
    find_optimal_threshold,
    det_curve,
    area_under_det,
    classification_report,
)

# --- Checkpoint helpers ---
from ww_trainer.checkpoint import (
    save_checkpoint,
    average_checkpoints,
    select_best_checkpoints,
)

# --- Factory ---
from ww_trainer.factory import (
    create_model, EXTRACTOR_REGISTRY, HEAD_REGISTRY,
    register_extractor, register_head,
)

# --- Evaluation ---
from ww_trainer.evaluation import evaluate_model, evaluate_detection, log_metrics_csv

# --- Quantization ---
from ww_trainer.quantize import quantize_onnx, quantize_model_pair, QuantizationReport

# --- Reproducibility ---
from ww_trainer.reproducibility import set_seed, get_seed_info

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
