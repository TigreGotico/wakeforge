import abc
import inspect
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.quantization import quantize_dynamic, QuantType

from ww_trainer.feats import WavInput, ensure_wav_list, BaseExtractor
from ww_trainer.utils import embed_onnx_metadata

logger = logging.getLogger(__name__)


def _accepts_phoneme_ids(fn) -> bool:
    """Return True if *fn* has a ``phoneme_ids`` parameter (duck-typing check)."""
    try:
        return "phoneme_ids" in inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False


class ClassifierHead(torch.nn.Module):
    def __init__(self, input_size: int, sample_rate: int = 16000, device="auto") -> None:
        super().__init__()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = sample_rate
        self.device = torch.device(device)
        self.input_size = input_size

    def _apply(self, fn: "Callable") -> "ClassifierHead":
        """Override to keep ``self.device`` in sync with ``.to()``/``.cuda()``/``.cpu()``."""
        result = super()._apply(fn)
        for p in self.parameters():
            self.device = p.device
            break
        return result

    @abc.abstractmethod
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo=False, metadata: dict = None):
        import onnx

        dummy_features = torch.zeros(1, 200, self.input_size, device=self.device)

        dynamic_axes = {
            "input_features": {0: "batch_size", 1: "T_features"},
            "logits": {0: "batch_size"},
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
        logger.info("Exported ONNX model to %s", out)

        if metadata:
            embed_onnx_metadata(out, metadata)

        if quantize:
            out_int8 = str(Path(out).with_stem(Path(out).stem + "_int8"))
            quantize_dynamic(out, out_int8,
                             op_types_to_quantize=["MatMul", "Gemm"],
                             weight_type=QuantType.QInt8)
            if metadata:
                embed_onnx_metadata(out_int8, metadata)
            logger.info("Quantized ONNX model saved to %s", out_int8)


class BaseWakeModel(nn.Module):
    """Abstract base for wake models providing device handling and utility helpers.

    Subclasses must implement `preprocess`, `forward`, and `embed`.
    """

    def __init__(self,
                 feature_extractor: BaseExtractor,
                 classifier: ClassifierHead,
                 sample_rate: int = 16000,
                 device: str = "auto",
                 text_extractor: Optional["OnnxTextExtractor"] = None,
                 keyword: Optional[str] = None) -> None:
        super().__init__()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device: torch.device = torch.device(device)
        self.sample_rate = sample_rate
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.text_extractor = text_extractor  # Optional[OnnxTextExtractor]
        self.keyword = keyword                # stored for metadata / export only
        # Cache inspect result — classifier is fixed after init
        self._classifier_takes_phoneme_ids: bool = _accepts_phoneme_ids(classifier.forward)
        self.to(self.device)

    def _apply_text_conditioning(
        self,
        feats: torch.Tensor,
        wavs: list,
        text_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Append text embedding channels to audio features if text_extractor is set.

        Args:
            feats: Audio features ``[B, T, F]``.
            wavs: Original wav list — used only to determine ``B`` when
                ``token_ids`` is ``None`` (inference with cached embedding).
            text_token_ids: Optional ``[B, seq_len]`` int64 keyword phoneme IDs.
                When provided, the ONNX encoder runs per-batch (training / zero-shot).
                When ``None``, falls back to the cached embedding from
                :meth:`~ww_trainer.feats.OnnxTextExtractor.precompute`.

        Returns:
            ``[B, T, F]`` when no text extractor, ``[B, T, F+D]`` otherwise.
        """
        if self.text_extractor is None:
            return feats
        text_emb = self.text_extractor(wavs, token_ids=text_token_ids)  # [B, 1, D]
        text_emb = text_emb.to(feats.device)                             # align devices
        text_emb = text_emb.expand(-1, feats.shape[1], -1)              # [B, T, D]
        return torch.cat([feats, text_emb], dim=-1)                      # [B, T, F+D]

    # --- forward / embed ---
    def forward(self, wavs: WavInput,
                text_token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the full model: extract features, optionally condition on text, classify.

        Args:
            wavs: Audio input — list of 1-D tensors or a padded ``[B, T]`` tensor.
            text_token_ids: Optional ``[B, seq_len]`` int64 phoneme IDs.
                Pass per-batch keyword IDs during multi-keyword training; omit
                for single-keyword inference (uses precomputed cache).

        Returns:
            ``[B]`` raw logits (pre-sigmoid).
        """
        wavs = ensure_wav_list(wavs)
        feats = self.feature_extractor(wavs)
        feats = self._apply_text_conditioning(feats, wavs, text_token_ids)
        # PhonMatchHead takes phoneme_ids directly; all other heads ignore the kwarg
        if self._classifier_takes_phoneme_ids:
            return self.classifier.forward(feats, phoneme_ids=text_token_ids)
        return self.classifier.forward(feats)

    def embed(self, wavs: WavInput,
              text_token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return embeddings for metric losses.

        Args:
            wavs: Audio input.
            text_token_ids: Optional per-batch keyword phoneme IDs.

        Returns:
            ``[B, D]`` embeddings.
        """
        wavs = ensure_wav_list(wavs)
        feats = self.feature_extractor(wavs)
        feats = self._apply_text_conditioning(feats, wavs, text_token_ids)
        if self._classifier_takes_phoneme_ids:
            return self.classifier.embed(feats, phoneme_ids=text_token_ids)
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
                          cache: "SlidingFeatureCacheTensor",
                          smoother: "Optional[PredictionSmoother]" = None) -> float:
        """Process one audio chunk with a rolling feature cache.

        Args:
            audio_chunk: 1-D float32 tensor (one chunk of audio at self.sample_rate).
            cache: SlidingFeatureCacheTensor instance — updated in-place.
            smoother: Optional ``PredictionSmoother`` for temporal smoothing.

        Returns:
            Sigmoid probability as a Python float (smoothed if smoother provided).
        """
        from ww_trainer.feats import SlidingFeatureCacheTensor  # noqa: F401 (type only)
        feats = self.feature_extractor([audio_chunk])   # [1, T_new, F]
        cached = cache(feats.squeeze(0))                # [T_window, F]
        logit = self.classifier.forward(cached.unsqueeze(0))  # [1]
        prob = torch.sigmoid(logit).item()
        if smoother is not None:
            prob = smoother.update(prob)
        return prob

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
                       export_featurizer: bool = False,
                       metadata: dict = None) -> None:
        # usually featurizer was already exported previously, only head missing
        self.classifier.export_to_onnx(out, quantize=quantize, dynamo=simplify, metadata=metadata)
        if export_featurizer:
            f_out = out.replace(".onnx", "") + "_featurizer.onnx"
            self.feature_extractor.export_to_onnx(f_out, quantize=quantize, metadata=metadata)


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


class OCSVMHead(ClassifierHead):
    """Two-stage wake-word head: FFN backbone + One-Class SVM decision boundary.

    Stage 1 — normal training loop with BCE/focal loss; backbone learns discriminative embeddings.
    Stage 2 — call :meth:`fit_ocsvm` after training to fit an OCSVM on positive embeddings.
    The OCSVM decision function is stored as torch buffers so :meth:`export_to_onnx` works
    without any sklearn dependency at inference time.

    Args:
        input_size: Feature dimension from the upstream extractor.
        sample_rate: Audio sample rate (Hz).
        device: Torch device string or ``"auto"``.
        hidden_dim: Intermediate FFN width.
        embed_dim: Output embedding dimension (input to OCSVM).
        dropout: Dropout probability in the backbone.
        nu: OCSVM ``nu`` parameter (upper bound on fraction of outliers).
        kernel: OCSVM kernel — ``"rbf"`` (default), ``"linear"``, ``"poly"``,
            or ``"sigmoid"``. All four are implemented in pure PyTorch and are
            ONNX-exportable. Kernel parameters are stored as buffers after
            :meth:`fit_ocsvm` so no sklearn is needed at inference time.
        gamma: Kernel coefficient passed to sklearn. ``"scale"`` (default) lets
            sklearn compute ``1 / (n_features × X.var())`` from the training
            embeddings; that exact value is stored as a buffer after fitting.
        degree: Degree for the ``"poly"`` kernel (ignored for other kernels).
        coef0: Independent term for ``"poly"`` and ``"sigmoid"`` kernels.
    """

    _SUPPORTED_KERNELS = frozenset({"rbf", "linear", "poly", "sigmoid"})

    def __init__(
        self,
        input_size: int,
        sample_rate: int = 16000,
        device: str = "auto",
        hidden_dim: int = 128,
        embed_dim: int = 64,
        dropout: float = 0.1,
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: "Union[str, float]" = "scale",
        degree: int = 3,
        coef0: float = 0.0,
    ) -> None:
        super().__init__(input_size=input_size, sample_rate=sample_rate, device=device)
        if kernel not in self._SUPPORTED_KERNELS:
            raise ValueError(
                f"kernel={kernel!r} is not supported. Choose from: "
                + ", ".join(sorted(self._SUPPORTED_KERNELS))
            )
        self.embed_dim = embed_dim
        self._nu = nu
        self._kernel = kernel
        self._gamma = gamma
        self._degree_init = degree
        self._coef0_init = coef0
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        # Support-vector buffers — populated by fit_ocsvm(); zeros = unfitted sentinel.
        # All-zero SVs produce constant-zero decision scores before fitting.
        self.register_buffer("_sv_vectors", torch.zeros(1, embed_dim))
        self.register_buffer("_sv_weights", torch.zeros(1))
        self.register_buffer("_rho", torch.zeros(1))
        # Kernel parameter buffers — exact values stored after fitting.
        # Sentinels are reasonable defaults but produce zero scores with zero SVs.
        self.register_buffer("_gamma_val", torch.tensor(1.0 / embed_dim))
        self.register_buffer("_degree_val", torch.tensor(float(degree)))
        self.register_buffer("_coef0_val", torch.tensor(coef0))

    def _kernel_vals(self, emb: torch.Tensor) -> torch.Tensor:
        """Compute kernel matrix K(emb, support_vectors) — pure torch, ONNX-safe.

        Args:
            emb: ``[B, embed_dim]`` embeddings.

        Returns:
            ``[B, N]`` kernel evaluations against the N support vectors.
        """
        dot = emb @ self._sv_vectors.T  # [B, N]
        if self._kernel == "linear":
            return dot
        if self._kernel == "poly":
            return (self._gamma_val * dot + self._coef0_val) ** self._degree_val
        if self._kernel == "sigmoid":
            return torch.tanh(self._gamma_val * dot + self._coef0_val)
        # rbf (default)
        sq_x = (emb ** 2).sum(dim=1, keepdim=True)   # [B, 1]
        sq_sv = (self._sv_vectors ** 2).sum(dim=1)    # [N]
        sq_dist = sq_x + sq_sv - 2.0 * dot            # [B, N]
        return torch.exp(-self._gamma_val * sq_dist)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Compute wake-word score.

        Before fitting, all SV buffers are zero, producing a constant-zero score.
        After :meth:`fit_ocsvm`, returns the OCSVM decision value.

        Args:
            feats: ``[B, T, F]`` feature frames.

        Returns:
            ``[B]`` scores (positive = inlier / wake-word after fitting).
        """
        emb = self.backbone(feats.mean(dim=1))           # [B, embed_dim]
        k = self._kernel_vals(emb)                        # [B, N]
        return (k * self._sv_weights).sum(dim=1) - self._rho.squeeze()

    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        """Return backbone embeddings.

        Args:
            feats: ``[B, T, F]`` feature frames.

        Returns:
            ``[B, embed_dim]`` embeddings.
        """
        return self.backbone(feats.mean(dim=1))

    def fit_ocsvm(self, dataloader: "torch.utils.data.DataLoader") -> None:
        """Fit the OCSVM on positive-class embeddings from *dataloader*.

        Requires ``scikit-learn``. After fitting, all kernel parameters (support
        vectors, dual coefficients, bias, gamma, degree, coef0) are stored as
        torch buffers so subsequent ONNX export needs no sklearn.

        **Always call :meth:`export_to_onnx` after fitting** — an ONNX graph
        exported before fitting bakes in the all-zero sentinel buffers and will
        produce constant-zero scores regardless of the input.

        Args:
            dataloader: Yields ``(feats, labels, ...)`` batches where ``labels == 1``
                        marks positive (wake-word) samples.

        Raises:
            ImportError: If ``scikit-learn`` is not installed.
            ValueError: If no positive-class samples are found in *dataloader*.
        """
        try:
            from sklearn.svm import OneClassSVM
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for fit_ocsvm(). "
                "Install with: pip install ww_trainer[ocsvm]"
            ) from exc

        self.eval()
        embeds = []
        with torch.no_grad():
            for batch in dataloader:
                feats, labels = batch[0], batch[1]
                pos = feats[labels == 1]
                if pos.shape[0]:
                    embeds.append(self.embed(pos.to(self.device)).cpu())

        if not embeds:
            raise ValueError("No positive samples found in dataloader — cannot fit OCSVM.")

        X = torch.cat(embeds, dim=0).numpy()
        svm = OneClassSVM(
            nu=self._nu, kernel=self._kernel, gamma=self._gamma,
            degree=int(self._degree_val.item()), coef0=float(self._coef0_val.item()),
        )
        svm.fit(X)

        self._sv_vectors = torch.tensor(svm.support_vectors_, dtype=torch.float32).to(self.device)
        self._sv_weights = torch.tensor(svm.dual_coef_[0], dtype=torch.float32).to(self.device)
        self._rho = torch.tensor([svm.offset_[0]], dtype=torch.float32).to(self.device)
        # svm._gamma holds the actual float (resolves "scale"/"auto" from training data)
        self._gamma_val = torch.tensor(float(svm._gamma), dtype=torch.float32).to(self.device)
        self._degree_val = torch.tensor(float(svm.degree), dtype=torch.float32).to(self.device)
        self._coef0_val = torch.tensor(float(svm.coef0), dtype=torch.float32).to(self.device)
        logger.info(
            "OCSVMHead fitted: %d support vectors, kernel=%s, gamma=%.6f.",
            self._sv_vectors.shape[0], self._kernel, svm._gamma,
        )


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


# ---------------------- Attention pooling ----------------------


class AttentionPooling(nn.Module):
    """Multi-head self-attention pooling over the time dimension.

    Replaces naive mean-pooling with a learned weighted sum.  Can be
    inserted into any classifier head that pools ``[B, T, F]`` → ``[B, F]``.

    Args:
        dim: Feature dimension (F).
        n_heads: Number of attention heads.
        dropout: Attention dropout rate.
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool ``[B, T, F]`` → ``[B, F]`` via attention."""
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x, need_weights=False)
        return out.squeeze(1)


# ---------------------- TC-ResNet head ----------------------


class _TCResBlock(nn.Module):
    """Temporal convolution residual block (Choi et al. 2019)."""

    def __init__(self, in_c: int, out_c: int, kernel_size: int = 9,
                 dilation: int = 1) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.shortcut = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class TCResNetHead(ClassifierHead):
    """TC-ResNet classifier head (Choi et al., Interspeech 2019).

    Purely 1D temporal convolutions with residual connections.
    Operates on ``[B, T, F]`` features transposed to ``[B, F, T]``.

    ``variant=8`` → 3 blocks (TC-ResNet8), ``variant=14`` → 6 blocks (TC-ResNet14).

    Args:
        input_size: Feature dimension F.
        variant: 8 or 14 (number of weight layers).
        channels: Channel width for residual blocks.
        kernel_size: Temporal kernel size.
        sample_rate: Metadata only.
        device: Device placement.
    """

    def __init__(self, input_size: int = 40, variant: int = 8,
                 channels: int = 64, kernel_size: int = 9,
                 sample_rate: int = 16000, device: str = "auto") -> None:
        super().__init__(input_size=input_size, sample_rate=sample_rate, device=device)
        assert variant in (8, 14), "variant must be 8 or 14"
        n_blocks = 3 if variant == 8 else 6
        layers: list[nn.Module] = [_TCResBlock(input_size, channels, kernel_size)]
        for _ in range(n_blocks - 1):
            layers.append(_TCResBlock(channels, channels, kernel_size))
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats.transpose(1, 2)  # [B, F, T]
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats.transpose(1, 2)
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)


# ---------------------- DS-CNN head ----------------------


class _DSConvBlock(nn.Module):
    """Depthwise-separable 2D convolution block (Zhang et al. 2017)."""

    def __init__(self, in_c: int, out_c: int, kernel: tuple[int, int] = (3, 3),
                 stride: tuple[int, int] = (1, 1)) -> None:
        super().__init__()
        pad = (kernel[0] // 2, kernel[1] // 2)
        self.dw = nn.Conv2d(in_c, in_c, kernel, stride=stride, padding=pad, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.dw(x)))
        return F.relu(self.bn2(self.pw(x)))


class DSCNNHead(ClassifierHead):
    """DS-CNN classifier head (Zhang et al., "Hello Edge", 2017).

    Depthwise-separable 2D CNN for keyword spotting on spectrograms.
    The ARM/Google benchmark standard for microcontroller KWS.
    Expects ``[B, T, F]`` mel features.

    Sizes: ``"S"`` ~20K params, ``"M"`` ~80K, ``"L"`` ~250K.

    Args:
        input_size: Feature dimension F (mel bins).
        size: ``"S"``, ``"M"``, or ``"L"``.
        sample_rate: Metadata only.
        device: Device placement.
    """

    _CONFIGS = {
        "S": {"channels": [64, 64, 64, 64], "first_c": 64},
        "M": {"channels": [64, 64, 64, 64, 64], "first_c": 64},
        "L": {"channels": [128, 128, 128, 128, 128, 128], "first_c": 128},
    }

    def __init__(self, input_size: int = 40, size: str = "M",
                 sample_rate: int = 16000, device: str = "auto") -> None:
        super().__init__(input_size=input_size, sample_rate=sample_rate, device=device)
        cfg = self._CONFIGS[size]
        first_c = cfg["first_c"]
        channels = cfg["channels"]

        self.first_conv = nn.Sequential(
            nn.Conv2d(1, first_c, (3, 3), stride=(2, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(first_c),
            nn.ReLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        in_c = first_c
        for out_c in channels:
            blocks.append(_DSConvBlock(in_c, out_c))
            in_c = out_c
        self.ds_blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_c, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats.transpose(1, 2).unsqueeze(1)  # [B, 1, F, T]
        x = self.first_conv(x)
        x = self.ds_blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x).squeeze(-1)

    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats.transpose(1, 2).unsqueeze(1)
        x = self.first_conv(x)
        x = self.ds_blocks(x)
        return self.pool(x).flatten(1)


# ---------------------- MatchboxNet head ----------------------


class _TCSConvBlock(nn.Module):
    """Time-channel separable convolution block (Majumdar & Ginsburg 2020).

    1D depthwise (temporal) → pointwise → BatchNorm → ReLU, with residual.
    """

    def __init__(self, channels: int, kernel_size: int = 11,
                 n_sub: int = 1, dilation: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(n_sub):
            pad = (kernel_size - 1) * dilation // 2
            layers.extend([
                nn.Conv1d(channels, channels, kernel_size, padding=pad,
                          dilation=dilation, groups=channels, bias=False),
                nn.Conv1d(channels, channels, 1, bias=False),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
            ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x


class MatchboxNetHead(ClassifierHead):
    """MatchboxNet classifier head (Majumdar & Ginsburg, NVIDIA 2020).

    1D time-channel separable convolutions with residual connections.
    Architecture: ``MatchboxNet-B×R×C`` where B=blocks, R=sub-blocks, C=channels.

    Predefined sizes:

    ======  ====  ====  =======  ===========
    size     B     R     C       ~Params
    ======  ====  ====  =======  ===========
    3×1×64   3    1     64       ~25K
    3×2×64   3    2     64       ~45K
    6×2×64   6    2     64       ~90K
    ======  ====  ====  =======  ===========

    Args:
        input_size: Feature dimension F.
        B: Number of blocks.
        R: Sub-blocks per block.
        C: Channel width.
        kernel_sizes: Kernel size per block (cycles if shorter than B).
        sample_rate: Metadata only.
        device: Device placement.
    """

    def __init__(self, input_size: int = 40, B: int = 3, R: int = 2,
                 C: int = 64, kernel_sizes: tuple[int, ...] = (11, 13, 15),
                 sample_rate: int = 16000, device: str = "auto") -> None:
        super().__init__(input_size=input_size, sample_rate=sample_rate, device=device)
        self.prologue = nn.Sequential(
            nn.Conv1d(input_size, C, 11, padding=5, bias=False),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True),
        )
        blocks: list[nn.Module] = []
        for i in range(B):
            ks = kernel_sizes[i % len(kernel_sizes)]
            blocks.append(_TCSConvBlock(C, kernel_size=ks, n_sub=R))
        self.blocks = nn.Sequential(*blocks)
        self.epilogue = nn.Sequential(
            nn.Conv1d(C, C * 2, 1, bias=False),
            nn.BatchNorm1d(C * 2),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(C * 2, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats.transpose(1, 2)  # [B, F, T]
        x = self.prologue(x)
        x = self.blocks(x)
        x = self.epilogue(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats.transpose(1, 2)
        x = self.prologue(x)
        x = self.blocks(x)
        x = self.epilogue(x)
        return self.pool(x).squeeze(-1)


# ---------------------- Res15 head ----------------------


class _Res15Block(nn.Module):
    """Dilated residual block (Tang & Lin 2018)."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class Res15Head(ClassifierHead):
    """Res15 classifier head (Tang & Lin, Interspeech 2018).

    1D residual network with exponentially increasing dilation.
    6 residual blocks with dilations [1, 2, 4, 8, 16, 32].

    Args:
        input_size: Feature dimension F.
        channels: Hidden channel width.
        sample_rate: Metadata only.
        device: Device placement.
    """

    def __init__(self, input_size: int = 40, channels: int = 45,
                 sample_rate: int = 16000, device: str = "auto") -> None:
        super().__init__(input_size=input_size, sample_rate=sample_rate, device=device)
        self.stem = nn.Sequential(
            nn.Conv1d(input_size, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[
            _Res15Block(channels, dilation=2 ** i) for i in range(6)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats.transpose(1, 2)  # [B, F, T]
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats.transpose(1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)


# ---------------------- KWT (Keyword Transformer) head ----------------------


class KWTHead(ClassifierHead):
    """Keyword Transformer head (Berg et al. 2021).

    Vision Transformer adapted for spectrograms: patches along time,
    positional embedding, transformer encoder, CLS token for classification.

    Args:
        input_size: Feature dimension F (mel bins).
        patch_len: Number of time frames per patch.
        d_model: Transformer embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer encoder layers.
        dim_ff: Feed-forward hidden dimension.
        dropout: Dropout rate.
        sample_rate: Metadata only.
        device: Device placement.
    """

    def __init__(self, input_size: int = 40, patch_len: int = 5,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 4,
                 dim_ff: int = 128, dropout: float = 0.1,
                 sample_rate: int = 16000, device: str = "auto") -> None:
        super().__init__(input_size=input_size, sample_rate=sample_rate, device=device)
        self.patch_len = patch_len
        patch_dim = input_size * patch_len
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # max 200 patches + 1 CLS
        self.pos_embed = nn.Parameter(torch.randn(1, 201, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)

    def _patchify(self, feats: torch.Tensor) -> torch.Tensor:
        """``[B, T, F]`` → ``[B, N_patches, patch_dim]``."""
        B, T, F = feats.shape
        # Truncate to multiple of patch_len
        n_patches = T // self.patch_len
        feats = feats[:, :n_patches * self.patch_len, :]
        return feats.reshape(B, n_patches, F * self.patch_len)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self._patchify(feats)                   # [B, N, patch_dim]
        x = self.patch_proj(x)                       # [B, N, d_model]
        B, N, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)              # [B, N+1, d_model]
        x = x + self.pos_embed[:, :N + 1, :]
        x = self.encoder(x)
        x = self.norm(x[:, 0])                      # CLS token
        return self.fc(x).squeeze(-1)

    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        x = self._patchify(feats)
        x = self.patch_proj(x)
        B, N, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :N + 1, :]
        x = self.encoder(x)
        return self.norm(x[:, 0])


# ---------------------- Conformer head ----------------------


class _ConformerBlock(nn.Module):
    """Single Conformer block: FFN → MHSA → Conv → FFN (Gulati et al. 2020)."""

    def __init__(self, d_model: int, n_heads: int, conv_kernel: int = 31,
                 dim_ff: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        # Half-step FFN
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, dim_ff),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        # Self-attention
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        # Convolution module
        self.norm_conv = nn.LayerNorm(d_model)
        pad = (conv_kernel - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, conv_kernel, padding=pad, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout),
        )
        # Half-step FFN
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, dim_ff),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        xn = self.norm_attn(x)
        x = x + self.attn(xn, xn, xn, need_weights=False)[0]
        xn = self.norm_conv(x).transpose(1, 2)
        x = x + self.conv(xn).transpose(1, 2)
        x = x + 0.5 * self.ffn2(x)
        return self.norm_out(x)


class ConformerHead(ClassifierHead):
    """Conformer classifier head (Gulati et al. 2020).

    Convolution-augmented transformer combining self-attention with
    depthwise convolutions. Dominates ASR; increasingly used for KWS.

    Args:
        input_size: Feature dimension F.
        d_model: Conformer hidden dimension.
        n_heads: Attention heads.
        n_layers: Number of Conformer blocks.
        conv_kernel: Convolution kernel size in each block.
        dim_ff: Feed-forward hidden dimension.
        dropout: Dropout rate.
        sample_rate: Metadata only.
        device: Device placement.
    """

    def __init__(self, input_size: int = 40, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 4, conv_kernel: int = 15,
                 dim_ff: int = 128, dropout: float = 0.1,
                 sample_rate: int = 16000, device: str = "auto") -> None:
        super().__init__(input_size=input_size, sample_rate=sample_rate, device=device)
        self.proj = nn.Linear(input_size, d_model)
        self.blocks = nn.Sequential(*[
            _ConformerBlock(d_model, n_heads, conv_kernel, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        self.pool = AttentionPooling(d_model, n_heads=n_heads)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.proj(feats)       # [B, T, d_model]
        x = self.blocks(x)
        x = self.pool(x)           # [B, d_model]
        return self.fc(x).squeeze(-1)

    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.proj(feats)
        x = self.blocks(x)
        return self.pool(x)


# ---------------------- CRNN head ----------------------


class _MixConvGroup(nn.Module):
    """Single channel group with depthwise + pointwise convolutions."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.block(x)


class _MixConvBlock(nn.Module):
    """Mixed depthwise convolution block with variable kernel sizes per channel group.

    Splits input channels into groups, applies different temporal kernels per group,
    concatenates, then adds a residual connection.
    """

    def __init__(self, channels: int, kernel_groups: list[list[int]]) -> None:
        super().__init__()
        n_groups = len(kernel_groups)
        self.group_size = channels // n_groups
        self.n_groups = n_groups
        # Handle remainder channels: add to last group
        self._last_group_extra = channels - self.group_size * n_groups

        self.groups = nn.ModuleList()
        for i, kernels in enumerate(kernel_groups):
            g_channels = self.group_size + (self._last_group_extra if i == n_groups - 1 else 0)
            # For each group, use the first kernel in the list
            k = kernels[0] if kernels else 3
            self.groups.append(_MixConvGroup(g_channels, k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with mixed kernel groups + residual."""
        residual = x
        splits = []
        pos = 0
        for i, group in enumerate(self.groups):
            g_channels = self.group_size + (self._last_group_extra if i == self.n_groups - 1 else 0)
            splits.append(group(x[:, pos:pos + g_channels, :]))
            pos += g_channels
        out = torch.cat(splits, dim=1)
        return F.relu(out + residual, inplace=True)


class MixConvHead(ClassifierHead):
    """MixConv classifier head with mixed depthwise convolutions.

    Inspired by micro-wake-word's MixedNet which uses variable kernel sizes
    per channel group for efficient multi-scale temporal feature extraction.

    Each block splits channels into N groups and applies different temporal
    kernels per group (e.g., [3], [5,7], [9,13]), enabling multi-scale
    pattern capture without depth overhead.

    Args:
        input_size: Feature dimension F.
        n_blocks: Number of MixConv blocks (default 3).
        filters: Channel width (default 64).
        kernel_groups: Kernel sizes per channel group
            (default ``[[3], [5, 7], [9, 13]]``).
        sample_rate: Metadata only.
        device: Device placement.
    """

    def __init__(
        self,
        input_size: int = 40,
        n_blocks: int = 3,
        filters: int = 64,
        kernel_groups: Optional[list[list[int]]] = None,
        sample_rate: int = 16000,
        device: str = "auto",
    ) -> None:
        super().__init__(input_size=input_size, sample_rate=sample_rate, device=device)
        if kernel_groups is None:
            kernel_groups = [[3], [5, 7], [9, 13]]

        self.stem = nn.Sequential(
            nn.Conv1d(input_size, filters, 3, padding=1, bias=False),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[
            _MixConvBlock(filters, kernel_groups) for _ in range(n_blocks)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(filters, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Classify features.

        Args:
            feats: ``[B, T, F]`` feature tensor.

        Returns:
            ``[B]`` raw logits.
        """
        x = feats.transpose(1, 2)  # [B, F, T]
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from penultimate layer.

        Args:
            feats: ``[B, T, F]`` feature tensor.

        Returns:
            ``[B, filters]`` embedding vectors.
        """
        x = feats.transpose(1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)


class CRNNHead(ClassifierHead):
    """CRNN (CNN + RNN) classifier head.

    2D CNN frontend extracts local patterns from spectrograms, then
    a GRU processes the temporal sequence. Common in production KWS.

    Args:
        input_size: Feature dimension F (mel bins).
        conv_channels: CNN channel width.
        gru_hidden: GRU hidden size.
        gru_layers: Number of GRU layers.
        dropout: Dropout rate.
        sample_rate: Metadata only.
        device: Device placement.
    """

    def __init__(self, input_size: int = 40, conv_channels: int = 32,
                 gru_hidden: int = 64, gru_layers: int = 1,
                 dropout: float = 0.1,
                 sample_rate: int = 16000, device: str = "auto") -> None:
        super().__init__(input_size=input_size, sample_rate=sample_rate, device=device)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, conv_channels, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(conv_channels, conv_channels * 2, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(conv_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        rnn_input = conv_channels * 2 * (input_size // 4)
        self.gru = nn.GRU(rnn_input, gru_hidden, num_layers=gru_layers,
                          batch_first=True, dropout=dropout if gru_layers > 1 else 0.0)
        self.fc = nn.Linear(gru_hidden, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats.transpose(1, 2).unsqueeze(1)  # [B, 1, F, T]
        x = self.cnn(x)                         # [B, C, F', T]
        B, C, Fp, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fp)  # [B, T, C*F']
        out, _ = self.gru(x)
        return self.fc(out.mean(dim=1)).squeeze(-1)

    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats.transpose(1, 2).unsqueeze(1)
        x = self.cnn(x)
        B, C, Fp, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fp)
        out, _ = self.gru(x)
        return out.mean(dim=1)


class EfficientNetHead(ClassifierHead):
    """EfficientNet-B0 classifier head applied to 2-D log-mel spectrograms.

    Treats the feature sequence ``[B, T, F]`` as a single-channel image
    ``[B, 1, F, T]`` and applies EfficientNet-B0 with an adapted classifier.
    No pre-trained weights are used — trained from scratch on the wake-word task.

    Requires ``torchvision``::

        pip install torchvision

    Args:
        input_size: Number of mel bins (frequency axis), e.g. 40 or 80.
        dropout: Dropout before the final linear layer.
        sample_rate: Metadata only.
        device: Device placement.
    """

    def __init__(
        self,
        input_size: int = 40,
        dropout: float = 0.2,
        sample_rate: int = 16000,
        device: str = "auto",
    ) -> None:
        super().__init__(input_size=input_size, sample_rate=sample_rate, device=device)
        try:
            from torchvision.models import efficientnet_b0
        except ImportError:
            raise ImportError("Install torchvision for EfficientNetHead: pip install torchvision")
        import torch.nn as nn
        backbone = efficientnet_b0(weights=None)
        # Replace first conv to accept 1-channel input instead of 3
        backbone.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        # Remove the original classifier; we add our own binary head
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 1),
        )
        self.net = backbone

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, T, F]
        x = feats.transpose(1, 2).unsqueeze(1)  # [B, 1, F, T]
        return self.net(x).squeeze(-1)           # [B]

    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats.transpose(1, 2).unsqueeze(1)
        # Pool features before the classifier head
        x = self.net.features(x)
        x = self.net.avgpool(x)
        return x.flatten(1)




# ---------------------- ConvAttention head ----------------------


class ConvAttentionHead(ClassifierHead):
    """1D-convolution + self-attention classifier head.

    Adapted from ``livekit/livekit-wakeword`` (Apache-2.0). Their docs report
    "60x lower AUT and 100x fewer false positives per hour than openWakeWord"
    using this head over the frozen Google-speech + openWakeWord embedding
    front-end. The original implementation assumes a fixed ``(T=16, F=96)``
    embedding shape and applies ``LayerNorm([layer_dim, n_timesteps])``; we
    relax that to ``LayerNorm(layer_dim)`` over the channel axis so the head
    works with any time length and exports to ONNX with a dynamic ``T_features``
    axis (matches the rest of ``ww-trainer``).

    Pipeline: ``Conv1d(F→D, k=3) → (Conv1d(D→D, k=3))^n_blocks →
    MultiheadAttention(D, n_heads) + residual + LayerNorm → mean-pool(T) →
    Linear(D, 1)``.

    Args:
        input_size: Feature dimension ``F`` (e.g. mel bins or embedding dim).
        layer_dim: Internal channel width ``D``.
        n_blocks: Number of additional Conv1d blocks after the projection.
        n_heads: Self-attention heads. Auto-clipped to a divisor of ``layer_dim``.
        dropout: Dropout after attention.
        sample_rate: Metadata only.
        device: Device placement.
    """

    def __init__(self, input_size: int = 96, layer_dim: int = 32,
                 n_blocks: int = 1, n_heads: int = 4, dropout: float = 0.0,
                 sample_rate: int = 16000, device: str = "auto") -> None:
        super().__init__(input_size=input_size, sample_rate=sample_rate, device=device)
        self.layer_dim = layer_dim
        # Conv stack: project F → D, then n_blocks of D → D.
        conv: list[nn.Module] = [
            nn.Conv1d(input_size, layer_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_blocks):
            conv += [
                nn.Conv1d(layer_dim, layer_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*conv)
        self.conv_norm = nn.LayerNorm(layer_dim)
        heads = min(n_heads, layer_dim)
        while heads > 1 and layer_dim % heads != 0:
            heads -= 1
        self.n_heads = heads
        self.attention = nn.MultiheadAttention(
            embed_dim=layer_dim, num_heads=heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(layer_dim)
        self.fc = nn.Linear(layer_dim, 1)

    def _trunk(self, feats: torch.Tensor) -> torch.Tensor:
        """``[B, T, F]`` → ``[B, D]`` (mean-pooled, post-attention)."""
        # Conv1d expects [B, F, T].
        x = feats.transpose(1, 2)
        x = self.conv(x)
        # Back to [B, T, D] for LayerNorm/Attention.
        x = x.transpose(1, 2)
        x = self.conv_norm(x)
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = self.attn_norm(x + attn_out)
        return x.mean(dim=1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(self._trunk(feats)).squeeze(-1)

    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        return self._trunk(feats)
