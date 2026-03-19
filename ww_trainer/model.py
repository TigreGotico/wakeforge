import abc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.quantization import quantize_dynamic, QuantType

from ww_trainer.feats import  WavInput,  ensure_wav_list, BaseExtractor


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

    def forward_streaming(self, audio_chunk: torch.Tensor,
                          cache: "SlidingFeatureCacheTensor") -> float:
        """Process one audio chunk with a rolling feature cache.

        Args:
            audio_chunk: 1-D float32 tensor (one chunk of audio at self.sample_rate).
            cache: SlidingFeatureCacheTensor instance — updated in-place.

        Returns:
            Sigmoid probability as a Python float.
        """
        from ww_trainer.feats import SlidingFeatureCacheTensor  # noqa: F401 (type only)
        feats = self.feature_extractor([audio_chunk])   # [1, T_new, F]
        cached = cache(feats.squeeze(0))                # [T_window, F]
        logit = self.classifier.forward(cached.unsqueeze(0))  # [1]
        return torch.sigmoid(logit).item()

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
                       quantize: bool = False,
                       export_featurizer=False):
        # usually featurizer was already exported previously, only head missing
        self.classifier.export_to_onnx(out, simplify, quantize)
        if export_featurizer:
            out = out.replace(".onnx", "") + "_featurizer.onnx"
            self.feature_extractor.export_to_onnx(out, simplify, quantize)


# ---------------------- classifier heads ----------------------


class FfnClassifierHead(ClassifierHead):
    """HuBERT-based wake-word detector (feedforward head)."""

    def __init__(self, sample_rate: int = 16000,
                 hidden_dim: int = 128,
                 dropout=0.2,
                 device: str = "auto",
                 input_size=None) -> None:
        self.hidden_dim = hidden_dim
        self.dropout = dropout
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
        feats = feats.transpose(1, 2)  # [B, T, F] -> [B, F, T] for Conv1d
        conv_out = self.conv(feats).squeeze(-1)
        h = F.relu(self.fc1(conv_out))
        return self.fc2(h).squeeze(-1)

    def embed(self, feats: WavInput) -> torch.Tensor:
        feats = feats.transpose(1, 2)  # [B, T, F] -> [B, F, T] for Conv1d
        conv_out = self.conv(feats).squeeze(-1)
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
        if D1 == self.input_size and D2 == self.input_size:
            raise ValueError(
                f"Ambiguous feature shape {feats.shape}; cannot determine time vs feature dim "
                f"when both equal input_size={self.input_size}. Pass [B, T, F] explicitly."
            )
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


# ---------------------- BC-ResNet classifier head ----------------------


class _ConvBNReLU(nn.Module):
    """Conv2d block with optional SubSpectralNorm/BatchNorm and activation.

    Ported from Qualcomm AI Research's BC-ResNet (Interspeech 2021).
    """

    def __init__(
        self,
        in_plane: int,
        out_plane: int,
        idx: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        groups: int = 1,
        use_dilation: bool = False,
        activation: bool = True,
        swish: bool = False,
        bn: bool = True,
        ssn: bool = False,
        ssn_groups: int = 5,
    ) -> None:
        super().__init__()
        from ww_trainer.subspectralnorm import SubSpectralNorm

        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        dilation = (idx + 1, 1) if use_dilation else 1
        if use_dilation:
            if isinstance(kernel_size, int):
                padding = (dilation[0] * (kernel_size // 2), kernel_size // 2)
            else:
                padding = (dilation[0] * (kernel_size[0] // 2), kernel_size[1] // 2)

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_plane, out_plane, kernel_size,
                stride=stride, padding=padding,
                dilation=dilation, groups=groups, bias=False,
            )
        ]

        if bn:
            if ssn:
                layers.append(SubSpectralNorm(out_plane, spec_groups=ssn_groups))
            else:
                layers.append(nn.BatchNorm2d(out_plane))

        if activation:
            layers.append(nn.SiLU(inplace=True) if swish else nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.block(x)


class _BCResBlock(nn.Module):
    """Broadcasted Residual block with 2D and 1D pathways.

    The 2D pathway processes frequency-temporal features while the 1D pathway
    operates on temporally-pooled features, broadcasting back to add 2D detail.
    """

    def __init__(self, in_plane: int, out_plane: int, idx: int,
                 stride: tuple[int, int] = (1, 1),
                 ssn_groups: int = 5) -> None:
        super().__init__()
        self.transition_block = in_plane != out_plane

        # --- 2D pathway (f2) ---
        f2_layers: list[nn.Module] = []
        if self.transition_block:
            f2_layers.append(
                _ConvBNReLU(in_plane, out_plane, idx, kernel_size=1)
            )
        f2_layers.append(
            _ConvBNReLU(
                out_plane, out_plane, idx,
                kernel_size=(3, 1), stride=stride,
                groups=out_plane, activation=False, ssn=True,
                ssn_groups=ssn_groups,
            )
        )
        self.f2 = nn.Sequential(*f2_layers)

        # --- 1D pathway (f1) ---
        self.f1 = nn.Sequential(
            _ConvBNReLU(
                out_plane, out_plane, idx,
                kernel_size=(1, 3), groups=out_plane,
                use_dilation=True, swish=True,
            ),
            _ConvBNReLU(out_plane, out_plane, idx, kernel_size=1),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with broadcasted residual connection."""
        shortcut = x
        x = self.f2(x)
        aux_2d = x
        x = x.mean(dim=2, keepdim=True)  # [B, C, 1, W] — pool frequency
        x = self.f1(x)               # 1D pathway
        x = x + aux_2d               # broadcast back to 2D
        if not self.transition_block:
            x = x + shortcut
        return F.relu(x, inplace=True)


def _bc_block_stage(
    num_layers: int,
    last_channel: int,
    cur_channel: int,
    idx: int,
    use_stride: bool,
    ssn_groups: int = 5,
) -> nn.ModuleList:
    """Build a stage of BCResBlocks."""
    channels = [last_channel] + [cur_channel] * num_layers
    stage = nn.ModuleList()
    for i in range(num_layers):
        stride = (2, 1) if use_stride and i == 0 else (1, 1)
        stage.append(_BCResBlock(channels[i], channels[i + 1], idx, stride,
                                 ssn_groups=ssn_groups))
    return stage


def _largest_divisor(n: int, max_val: int = 16) -> int:
    """Return the largest divisor of *n* that is ≤ *max_val*, minimum 1."""
    for d in range(min(n, max_val), 0, -1):
        if n % d == 0:
            return d
    return 1


class BCResNetHead(ClassifierHead):
    """BC-ResNet classifier head for wake word detection on mel-spectrogram features.

    Ported from Qualcomm AI Research's BC-ResNet (Kim et al., Interspeech 2021).
    Expects input features of shape ``[B, T, F]`` (e.g. from MfccExtractor or
    FilterbankExtractor).  Internally reshapes to ``[B, 1, F, T]`` for 2D
    convolutions, then outputs a single binary logit per sample.

    The ``tau`` parameter controls model width:

    ======  ===========  ==============
    tau     base_c       Approx params
    ======  ===========  ==============
    1       8            ~6 K
    1.5     12           ~12 K
    2       16           ~20 K
    3       24           ~43 K
    6       48           ~160 K
    8       64           ~280 K
    ======  ===========  ==============

    Args:
        input_size: Feature dimension F (number of mel bins, e.g. 40).
        tau: Width multiplier (1, 1.5, 2, 3, 6, 8). Default 8.
        sample_rate: Audio sample rate (metadata only).
        device: Device placement.
    """

    def __init__(
        self,
        input_size: int = 40,
        tau: float = 8,
        sample_rate: int = 16000,
        device: str = "auto",
    ) -> None:
        super().__init__(input_size=input_size, sample_rate=sample_rate, device=device)
        self.tau = tau
        base_c = int(tau * 8)

        self.n = [2, 2, 4, 4]  # layers per stage
        c = [
            base_c * 2,           # head output channels
            base_c,               # stage 0
            int(base_c * 1.5),    # stage 1
            base_c * 2,           # stage 2
            int(base_c * 2.5),    # stage 3
            base_c * 4,           # classifier hidden
        ]
        self.s = [1, 2]  # stages that use stride

        # --- Head ---
        self.cnn_head = nn.Sequential(
            nn.Conv2d(1, c[0], kernel_size=5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
        )

        # --- Body: 4 stages of BCResBlocks ---
        # Track frequency dimension through strides to compute valid SSN groups.
        # Head: stride (2,1) on freq with kernel=5, padding=2
        freq = (input_size + 2 * 2 - 5) // 2 + 1  # after head conv

        self.bc_blocks = nn.ModuleList()
        for i, num_layers in enumerate(self.n):
            use_stride = i in self.s
            last_c = c[0] if i == 0 else c[i]
            # SSN is applied AFTER the strided conv, so compute post-stride freq.
            # Conv kernel=(3,1), padding=(1,0), stride=(2,1) if use_stride.
            if use_stride:
                freq_after_stride = (freq + 2 * 1 - 3) // 2 + 1
            else:
                freq_after_stride = freq  # stride=1 with padding=1 preserves
            ssn_g = _largest_divisor(freq_after_stride, max_val=16)
            self.bc_blocks.append(
                _bc_block_stage(num_layers, last_c, c[i + 1], i, use_stride,
                                ssn_groups=ssn_g)
            )
            if use_stride:
                freq = freq_after_stride

        # --- Classifier (binary: 1 logit) ---
        # Adaptive freq kernel: original uses 5, but clamp to final freq dim
        cls_freq_k = min(5, freq)
        self._classifier = nn.Sequential(
            nn.Conv2d(c[-2], c[-2], kernel_size=(cls_freq_k, 5),
                      groups=c[-2], padding=(0, 2), bias=False),
            nn.Conv2d(c[-2], c[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(c[-1]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c[-1], 1, kernel_size=1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Classify mel-spectrogram features.

        Args:
            feats: ``[B, T, F]`` mel-spectrogram features.

        Returns:
            ``[B]`` raw logits (apply sigmoid for probability).
        """
        # [B, T, F] → [B, 1, F, T]  (channel=1, freq=F, time=T)
        x = feats.transpose(1, 2).unsqueeze(1)
        x = self.cnn_head(x)
        for i, num_layers in enumerate(self.n):
            for j in range(num_layers):
                x = self.bc_blocks[i][j](x)
        x = self._classifier(x)
        return x.squeeze(-1).squeeze(-1).squeeze(-1)

    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from penultimate layer.

        Args:
            feats: ``[B, T, F]`` mel-spectrogram features.

        Returns:
            ``[B, C]`` embedding vectors.
        """
        x = feats.transpose(1, 2).unsqueeze(1)
        x = self.cnn_head(x)
        for i, num_layers in enumerate(self.n):
            for j in range(num_layers):
                x = self.bc_blocks[i][j](x)
        # Run through classifier up to (but not including) final conv
        for layer in list(self._classifier.children())[:-1]:
            x = layer(x)
        return x.flatten(1)
