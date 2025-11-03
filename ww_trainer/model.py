import abc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.quantization import quantize_dynamic, QuantType

from ww_trainer.feats import EndToEndCnnGruExtractor, WavInput, MfccExtractor, OnnxFeatureExtractor, ensure_wav_list, BaseExtractor


class ClassifierHead(torch.nn.Module):
    def __init__(self, input_size: int, sample_rate: int = 16000, device="auto") -> None:
        super().__init__()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = sample_rate
        self.device = torch.device(device)
        self.input_size = input_size

    @abc.abstractmethod
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo=False):
        import onnx

        dummy_features = torch.zeros(1, 200, self.input_size, device=self.device)

        dynamic_axes = {
            "input_features": {1: "T_features"} # Time dimension (T) is dynamic
        }

        torch.onnx.export(
            self,
            (dummy_features,),
            out,
            input_names=["input_features"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=18,
            do_constant_folding=True,
            dynamo=dynamo,
            verbose=False,

            training=torch.onnx.TrainingMode.EVAL
        )
        onnx_model = onnx.load(out)
        onnx.checker.check_model(onnx_model)
        print(f"✅ Exported ONNX model to {out}")

        if quantize:
            out_int8 = str(Path(out).with_stem(Path(out).stem + "_int8"))
            quantize_dynamic(out, out_int8,
                             op_types_to_quantize=["MatMul", "Gemm"],
                             weight_type=QuantType.QInt8)
            print(f"✅ Quantized ONNX model saved to {out_int8}")


class BaseWakeModel(nn.Module):
    """Abstract base for wake models providing device handling and utility helpers.

    Subclasses must implement `preprocess`, `forward`, and `embed`.
    """

    def __init__(self,
                 feature_extractor: BaseExtractor,
                 classifier: ClassifierHead,
                 sample_rate: int = 16000,
                 device: str = "auto") -> None:
        super().__init__()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device: torch.device = torch.device(device)
        self.sample_rate = sample_rate
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.to(self.device)

    # --- abstract methods ---
    def forward(self, wavs: WavInput) -> torch.Tensor:
        wavs = ensure_wav_list(wavs)
        feats = self.feature_extractor(wavs)
        return self.classifier.forward(feats)

    def embed(self, wavs: WavInput) -> torch.Tensor:
        wavs = ensure_wav_list(wavs)
        feats = self.feature_extractor(wavs)
        return self.classifier.embed(feats)

    # --- convenience ---
    def load_checkpoint(self, ckpt_path: str) -> None:
        state = torch.load(ckpt_path, map_location=self.device)
        try:
            self.load_state_dict(state, strict=False)
        except Exception:
            if isinstance(state, dict) and "model" in state:
                self.load_state_dict(state["model"], strict=False)
            else:
                raise

    def save_checkpoint(self, ckpt_path: str):
        torch.save(self.state_dict(), ckpt_path)

    def infer(self, audio: np.ndarray) -> float:
        """Single-waveform inference returning sigmoid(logit)."""
        if audio.ndim != 1:
            raise ValueError("audio must be 1-D numpy array")
        wav_tensor = torch.tensor(audio, dtype=torch.float32)
        logits = self.forward([wav_tensor])
        prob = torch.sigmoid(logits).item() if isinstance(logits, torch.Tensor) else float(logits)
        return float(prob)

    def export_to_onnx(self, out: str,
                       simplify: bool = False,
                       quantize: bool = False):
        # featurizer already exported previously, only head missing
        self.classifier.export_to_onnx(out, simplify, quantize)


# ---------------------- classifier heads ----------------------


class FfnClassifierHead(ClassifierHead):
    """HuBERT-based wake-word detector (feedforward head)."""

    def __init__(self, sample_rate: int = 16000,
                 hidden_dim: int = 128,
                 dropout=0.2,
                 device: str = "auto",
                 input_size=None) -> None:
        super().__init__(sample_rate=sample_rate, device=device,  input_size=input_size)
        self.sequential = nn.Sequential(nn.Linear(self.input_size, hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, 1))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        pooled = feats.mean(dim=1)
        return self.sequential(pooled).squeeze(-1)

    def embed(self, feats: WavInput) -> torch.Tensor:
        pooled = feats.mean(dim=1)
        return F.relu(self.sequential[0](pooled))


class CnnClassifierHead(ClassifierHead):

    def __init__(self, sample_rate: int = 16000,
                 conv_dim: int = 256,
                 linear_dim: int = 128,
                 kernel_size: int = 3,
                 stride: int = 1,
                 device: str = "auto",
                 input_size=None) -> None:
        super().__init__(sample_rate=sample_rate, device=device, input_size=input_size)
        self.conv = nn.Sequential(
            nn.Conv1d(self.input_size, conv_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc1 = nn.Linear(conv_dim, linear_dim)
        self.fc2 = nn.Linear(linear_dim, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(feats.transpose(1, 2)).squeeze(-1)
        h = F.relu(self.fc1(conv_out))
        return self.fc2(h).squeeze(-1)

    def embed(self, feats: WavInput) -> torch.Tensor:
        conv_out = self.conv(feats.transpose(1, 2)).squeeze(-1)
        return F.relu(self.fc1(conv_out))


class GruClassifierHead(ClassifierHead):

    def __init__(self,
                 hidden_dim: int = 128,
                 linear_dim: int = 128,
                 dropout=0.0,
                 bidirectional=False,
                 gru_n_layers=1,
                 sample_rate: int = 16000,
                 device: str = "auto",
                 input_size=None) -> None:
        super().__init__(sample_rate=sample_rate, device=device, input_size=input_size)
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=hidden_dim,
                          num_layers=gru_n_layers,
                          dropout=dropout if gru_n_layers > 1 else 0.0,
                          bidirectional=bidirectional,
                          batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * (2 if bidirectional else 1), linear_dim)
        self.fc2 = nn.Linear(linear_dim, 1)

    def _ensure_correct_shape(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Auto-detect if input is [B, F, T] instead of [B, T, F]
        and transpose if necessary. Handles edge cases safely.
        """
        if feats.ndim != 3:
            raise ValueError(f"Expected 3D tensor [B, T, F], got shape {feats.shape}")

        B, D1, D2 = feats.shape
        # Heuristic: whichever matches input_size is feature dim
        if D1 == self.input_size and D2 != self.input_size:
            return feats.transpose(1, 2)
        return feats  # already correct

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats = self._ensure_correct_shape(feats)
        out, _ = self.gru(feats)
        pooled = out.mean(dim=1)
        h = F.relu(self.fc1(pooled))
        return self.fc2(h).squeeze(-1)

    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        feats = self._ensure_correct_shape(feats)
        out, _ = self.gru(feats)
        pooled = out.mean(dim=1)
        return F.relu(self.fc1(pooled))


# ---------------------- MFCC-based models ----------------------

class MfccCnnWakeModel(BaseWakeModel):
    """Simple CNN on log-mel / mfcc spectrograms."""

    def __init__(
            self,
            conv_dim: int = 256,
            linear_dim: int = 128,
            kernel_size: int = 3,
            stride: int = 1,
            n_mels: int = 64,
            n_mfcc: int = 40,
            n_fft: int = 400,
            hop_length: int = 160,
            sample_rate: int = 16000,
            device: str = "auto"
    ) -> None:
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        clf = CnnClassifierHead(sample_rate=sample_rate, device=device,
                                input_size=self.n_mfcc,
                                conv_dim=conv_dim, linear_dim=linear_dim,
                                kernel_size=kernel_size, stride=stride)
        super().__init__(sample_rate=sample_rate, device=device,
                         classifier=clf,
                         feature_extractor=MfccExtractor(
                             n_mels=n_mels,
                             n_mfcc=n_mfcc,
                             n_fft=n_fft,
                             hop_length=hop_length,
                             sample_rate=sample_rate))


class MfccGruWakeModel(BaseWakeModel):
    """Mel/MFCC + GRU + classifier with ONNX-safe forward/export."""

    def __init__(
            self,
            hidden_dim: int = 128,
            linear_dim: int = 128,
            dropout=0.0,
            bidirectional=False,
            gru_n_layers=1,
            n_mels: int = 64,
            n_mfcc: int = 40,
            n_fft: int = 400,
            hop_length: int = 160,
            sample_rate: int = 16000,
            device: str = "auto",
    ) -> None:
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.hidden_dim = hidden_dim
        self.num_layers = gru_n_layers
        self.bidirectional = bidirectional
        self.gru_dropout = dropout if gru_n_layers > 1 else 0.0

        clf = GruClassifierHead(hidden_dim=hidden_dim, linear_dim=linear_dim, dropout=dropout,
                                input_size=self.n_mfcc,
                                bidirectional=bidirectional, gru_n_layers=gru_n_layers,
                                sample_rate=sample_rate, device=device)
        super().__init__(sample_rate=sample_rate, device=device,
                         classifier=clf,
                         feature_extractor=MfccExtractor(
                             n_mels=n_mels,
                             n_mfcc=n_mfcc,
                             n_fft=n_fft,
                             hop_length=hop_length,
                             sample_rate=sample_rate))

# ---------------------- from scratch models ----------------------

class RawCnnGruWakeModel(BaseWakeModel):
    """
    End-to-end model using the high-performance CNN-GRU backbone and a simple FFN classifier head.
    """

    def __init__(self, sample_rate: int = 16000,
                 feature_dim: int = 256,  # The output dimension for the embeddings
                 hidden_dim: int = 128,
                 dropout: float = 0.2,
                 device: str = "auto") -> None:
        # 1. CNN-GRU Extractor (The new, powerful backbone)
        extractor = EndToEndCnnGruExtractor(
            output_feature_dim=feature_dim,
            gru_hidden_dim=feature_dim // 2,
            device=device
        )

        # 2. Classifier Head
        clf = FfnClassifierHead(
            sample_rate=sample_rate,
            device=device,
            input_size=feature_dim,  # Must match extractor's output_feature_dim
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        super().__init__(
            sample_rate=sample_rate,
            device=device,
            feature_extractor=extractor,
            classifier=clf
        )
        self.to(self.device)