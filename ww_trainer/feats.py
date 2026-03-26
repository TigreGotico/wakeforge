import abc
import logging
import math
import numpy as np
from pathlib import Path
from typing import List, Union, TypeAlias
from ww_trainer.utils import timed, embed_onnx_metadata
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnxruntime.quantization import quantize_dynamic, QuantType

logger = logging.getLogger(__name__)

WavInput: TypeAlias = Union[torch.Tensor, List[torch.Tensor]]


def ensure_wav_list(wavs: WavInput) -> List[torch.Tensor]:
    """Normalize `wavs` to a list of 1-D torch.Tensor waveforms."""
    if isinstance(wavs, torch.Tensor):
        if wavs.dim() == 1:
            return [wavs]
        if wavs.dim() == 2:
            return [wavs[i] for i in range(wavs.shape[0])]
        raise ValueError("Unsupported tensor shape for wavs; expected [T] or [B, T].")
    if isinstance(wavs, list):
        return wavs
    raise TypeError("wavs must be a torch.Tensor or a list of torch.Tensor")


class SlidingFeatureCacheTensor(torch.nn.Module):
    def __init__(self, feature_dim=768, window_size=50):
        super().__init__()
        self.window_size = window_size
        self.feature_dim = feature_dim
        # Tensor to hold recent features
        self.register_buffer("feature_cache", torch.zeros(window_size, feature_dim))
        # Stored as a buffer so state_dict save/load preserves the valid-frame count.
        self.register_buffer("_current_len", torch.tensor(0, dtype=torch.long))

    @property
    def current_len(self) -> int:
        return int(self._current_len.item())

    @current_len.setter
    def current_len(self, val: int) -> None:
        self._current_len.fill_(val)

    def forward(self, new_feats):
        """
        new_feats: [T_new, feature_dim]
        """
        T_new = new_feats.shape[0]
        # Shift old cache to make room for new features
        if self.current_len + T_new > self.window_size:
            shift = self.current_len + T_new - self.window_size
            new_len = self.current_len - shift
            if new_len > 0:
                self.feature_cache[:new_len] = self.feature_cache[shift:self.current_len].clone()
            self.current_len = max(new_len, 0)

        # Append new features
        self.feature_cache[self.current_len:self.current_len + T_new] = new_feats
        self.current_len += T_new

        # Return cached features
        return self.feature_cache[:self.current_len]


class BaseExtractor(torch.nn.Module):

    def __init__(self, sample_rate: int = 16000, device="auto") -> None:
        super().__init__()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = sample_rate
        self.device = torch.device(device)
        # Guarantees at least one buffer so _apply() can always sync self.device
        # even for pure-function extractors that register no parameters or buffers.
        self.register_buffer("_device_anchor", torch.empty(0))

    def _apply(self, fn: "Callable") -> "BaseExtractor":
        """Override to keep ``self.device`` in sync with ``.to()``/``.cuda()``/``.cpu()``."""
        result = super()._apply(fn)
        # After _apply, detect device from first parameter or buffer
        for p in self.parameters():
            self.device = p.device
            break
        else:
            for b in self.buffers():
                self.device = b.device
                break
        return result

    @property
    @abc.abstractmethod
    def feature_dim(self) -> int:
        ...

    @abc.abstractmethod
    def forward(self, wavs: WavInput, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, wavs: WavInput, **kwargs) -> torch.Tensor:
        return self.forward(wavs, **kwargs)

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        import onnx
        dummy_wav = torch.zeros(1, self.sample_rate, device=self.device)

        if dynamo:
            # Dynamo exporter: dynamic_axes not supported — use dynamic_shapes.
            # Reset dynamo cache so stale compiled graphs from training don't
            # interfere with the export trace.
            import torch._dynamo
            torch._dynamo.reset()
            batch_dim = torch.export.Dim("batch_size")
            time_dim  = torch.export.Dim("time")
            torch.onnx.export(self, (dummy_wav,), out,
                              input_names=["input_values"], output_names=["features"],
                              dynamic_shapes={"wavs": {0: batch_dim, 1: time_dim}},
                              opset_version=18,
                              dynamo=True)
        else:
            dynamic_axes = {"input_values": {0: "batch_size", 1: "time"},
                            "features": {0: "batch_size", 1: "time"}}
            torch.onnx.export(self, dummy_wav, out,
                              input_names=["input_values"], output_names=["features"],
                              dynamic_axes=dynamic_axes,
                              opset_version=18,
                              do_constant_folding=True,
                              dynamo=False,
                              verbose=False,
                              external_data=False,
                              training=torch.onnx.TrainingMode.EVAL)
        onnx_model = onnx.load(out)
        onnx.checker.check_model(onnx_model)
        logger.info("Exported ONNX model to %s", out)

        if metadata:
            embed_onnx_metadata(out, metadata)

        if quantize:
            out_int8 = str(Path(out).with_stem(Path(out).stem + "_int16"))
            quantize_dynamic(out, out_int8, op_types_to_quantize=["MatMul", "Gemm"], weight_type=QuantType.QInt16)
            logger.info("Quantized ONNX model saved to %s", out_int8)
            out_int8 = str(Path(out).with_stem(Path(out).stem + "_int8"))
            quantize_dynamic(out, out_int8, op_types_to_quantize=["MatMul", "Gemm"], weight_type=QuantType.QInt8)
            logger.info("Quantized ONNX model saved to %s", out_int8)


class OnnxFeatureExtractor(BaseExtractor):
    def __init__(self, model_path, sample_rate: int = 16000, device="auto"):
        super().__init__(sample_rate, device)

        # 1. Determine Execution Providers based on device
        if self.device.type == "cuda":
            # Prefer CUDA EP, fall back to CPU EP
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            # Use only CPU EP
            providers = ["CPUExecutionProvider"]

        # 2. Initialize ONNX runtime session with selected providers
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.input_info = self.sess.get_inputs()[0]
        self.output_info = self.sess.get_outputs()[0]

        self.input_name = self.input_info.name
        self.output_name = self.output_info.name

    @property
    def feature_dim(self) -> int:
        shape = self.output_info.shape
        if shape is not None and len(shape) >= 1 and shape[-1] is not None:
            try:
                return int(shape[-1])
            except (ValueError, TypeError):
                pass  # named/dynamic dim — fall through to dummy inference
        # Dynamic axis: run a dummy inference
        import numpy as np
        dummy = np.zeros((1, self.sample_rate), dtype=np.float32)
        out = self.sess.run([self.output_name], {self.input_name: dummy})[0]
        return int(out.shape[-1])

    @classmethod
    def from_whisper(cls, model_size: str = "tiny", cache_dir: str = None,
                     device: str = "auto") -> "OnnxFeatureExtractor":
        """Load a Whisper encoder as an OnnxFeatureExtractor.

        Downloads and caches the Whisper ONNX encoder from HuggingFace
        optimum-whisper or whisper-onnx. Requires `optimum` or the onnx
        file to be pre-exported.

        Args:
            model_size: 'tiny', 'base', 'small', 'medium', 'large'
            cache_dir: Directory to cache the ONNX file. Defaults to ~/.cache/ww_trainer/whisper/
            device: 'cpu', 'cuda', or 'auto'

        Returns:
            OnnxFeatureExtractor loaded with the Whisper encoder.

        Note:
            Export Whisper to ONNX first:
                from optimum.exporters.onnx import main_export
                main_export("openai/whisper-tiny", output="whisper_tiny_onnx/", task="feature-extraction")
            Then load:
                OnnxFeatureExtractor.from_whisper("tiny", cache_dir="whisper_tiny_onnx/")
        """
        import os
        from pathlib import Path

        cache_dir = cache_dir or str(Path.home() / ".cache" / "ww_trainer" / "whisper" / model_size)

        # Try common file names
        candidates = [
            os.path.join(cache_dir, "encoder_model.onnx"),
            os.path.join(cache_dir, "model.onnx"),
            os.path.join(cache_dir, f"whisper_{model_size}.onnx"),
            os.path.join(cache_dir, f"whisper-{model_size}.onnx"),
        ]

        onnx_path = None
        for c in candidates:
            if os.path.exists(c):
                onnx_path = c
                break

        if onnx_path is None:
            raise FileNotFoundError(
                f"Whisper ONNX encoder not found in {cache_dir}. "
                f"Export it first:\n"
                f"  from optimum.exporters.onnx import main_export\n"
                f"  main_export('openai/whisper-{model_size}', output='{cache_dir}/', "
                f"task='feature-extraction')\n"
                f"Looked for: {candidates}"
            )

        return cls(onnx_path, sample_rate=16000, device=device)

    def forward(self, wavs: WavInput) -> torch.Tensor:
        wav_list = ensure_wav_list(wavs)

        # 1. Process each waveform individually with the ONNX session
        feats_list = []
        for w in wav_list:
            # The ONNX model expects a batched input, so unsqueeze the 1D waveform
            wav_np = w.unsqueeze(0).cpu().numpy()  # [1, T] -> NumPy

            # 2. Run ONNX inference for a single item (batch size 1)
            feats_np = self.sess.run([self.output_name], {self.input_name: wav_np})[0]
            feats_tensor = torch.from_numpy(feats_np).to(torch.float32).to(self.device).squeeze(
                0)  # [1, T_feats, 768] -> [T_feats, 768]
            feats_list.append(feats_tensor)

        # Ensure all features have the same time dimension T_feats before stacking
        max_feat_len = max(f.shape[0] for f in feats_list)
        padded_feats = [F.pad(f, (0, 0, 0, max_feat_len - f.shape[0])) if f.shape[0] < max_feat_len else f for f in
                        feats_list]

        return torch.stack(padded_feats, dim=0)  # returns [B, T_feats, 768]

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        raise NotImplementedError(
            "OnnxFeatureExtractor wraps an onnxruntime session and cannot be re-exported "
            "via PyTorch. The model is already an ONNX file — copy it directly."
        )


def _build_mel_filterbank(n_mels: int, n_fft: int, sample_rate: int,
                          f_min: float, f_max: float) -> torch.Tensor:
    """Build a triangular mel filterbank matrix (pure PyTorch, ONNX-safe).

    Returns:
        Tensor of shape [n_mels, n_fft // 2 + 1].
    """
    def hz_to_mel(f):
        return 2595.0 * torch.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    m_min = hz_to_mel(torch.tensor(f_min))
    m_max = hz_to_mel(torch.tensor(f_max))
    m_points = torch.linspace(m_min, m_max, n_mels + 2)
    f_points = mel_to_hz(m_points)
    bins = torch.floor((n_fft + 1) * f_points / sample_rate).long()

    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f_m_minus, f_m):
            fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return fb


class MfccExtractor(BaseExtractor):
    """Pure-PyTorch MFCC extractor, ONNX-exportable.
    Output: [B, T, n_mfcc] — compatible with all ClassifierHead inputs.
    Reference export: https://huggingface.co/TigreGotico/mfcc-onnx
    """

    def __init__(self, sr: int = 16000, n_mfcc: int = 40, n_mels: int = 40,
                 n_fft: int = 400, hop_length: int = 160,
                 f_min: float = 0.0, f_max: float = None):
        super().__init__(sample_rate=sr)
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max or sr / 2
        mel_fb = _build_mel_filterbank(n_mels, n_fft, sr, self.f_min, self.f_max)
        self.register_buffer("mel_fb", mel_fb)
        dct_mat = self._dct_matrix(n_mfcc, n_mels)
        self.register_buffer("dct_mat", dct_mat)

    @property
    def feature_dim(self) -> int:
        return self.n_mfcc

    def _mel_filterbank(self):
        def hz_to_mel(f):
            return 2595.0 * torch.log10(1.0 + f / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10 ** (m / 2595.0) - 1.0)

        m_min = hz_to_mel(torch.tensor(self.f_min))
        m_max = hz_to_mel(torch.tensor(self.f_max))
        m_points = torch.linspace(m_min, m_max, self.n_mels + 2)
        f_points = mel_to_hz(m_points)
        bins = torch.floor((self.n_fft + 1) * f_points / self.sample_rate).long()

        fb = torch.zeros(self.n_mels, self.n_fft // 2 + 1)
        for m in range(1, self.n_mels + 1):
            f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
            for k in range(f_m_minus, f_m):
                fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
        return fb

    def _dct_matrix(self, n_mfcc, n_mels):
        n = torch.arange(n_mels).float()
        k = torch.arange(n_mfcc).float().unsqueeze(1)
        dct = torch.cos(math.pi / n_mels * (n + 0.5) * k)
        dct[0] *= 1 / math.sqrt(2.0)
        return dct * math.sqrt(2.0 / n_mels)

    def forward(self, wavs):
        # If input is a list, pad and batch them
        if isinstance(wavs, list):
            wavs = [w.to(torch.float32) for w in wavs]
            max_len = max(w.shape[-1] for w in wavs)
            batch = torch.stack([F.pad(w, (0, max_len - w.shape[-1])) for w in wavs])
        else:
            batch = wavs.unsqueeze(0) if wavs.ndim == 1 else wavs

        # Ensure batch is at least n_fft long for stft reflect padding
        if batch.shape[-1] < self.n_fft:
            batch = F.pad(batch, (0, self.n_fft - batch.shape[-1]))

        device = batch.device
        window = torch.hann_window(self.n_fft, device=device)

        # Compute STFT
        stft = torch.view_as_real(torch.stft(
            batch,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        ))
        real = stft[..., 0]
        imag = stft[..., 1]
        power = (real.pow(2) + imag.pow(2))

        mel_spec = torch.matmul(self.mel_fb.to(device), power)
        log_mel = torch.log(mel_spec + 1e-10)
        mfcc = torch.matmul(self.dct_mat.to(device), log_mel)

        # Transpose to [B, T, n_mfcc]
        return mfcc.transpose(1, 2)

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        # torch.stft + view_as_real fails with the TorchScript ONNX exporter;
        # the dynamo exporter decomposes STFT into ONNX-safe real ops correctly.
        super().export_to_onnx(out, quantize=quantize, dynamo=True, metadata=metadata)


class FilterbankExtractor(BaseExtractor):
    """Log Mel filterbank (log-mel spectrogram), ONNX-exportable.

    Output: [B, T, n_mels] — same interface as MfccExtractor but without DCT.
    Faster than MFCC; used in Whisper-style models.

    Args:
        sr: Sample rate (default 16000).
        n_mels: Number of mel filterbank channels (default 80).
        n_fft: FFT size (default 400).
        hop_length: Hop length in samples (default 160).
        f_min: Minimum frequency in Hz (default 0.0).
        f_max: Maximum frequency in Hz (default sr/2).
    """

    def __init__(self, sr: int = 16000, n_mels: int = 80,
                 n_fft: int = 400, hop_length: int = 160,
                 f_min: float = 0.0, f_max: float = None):
        super().__init__(sample_rate=sr)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max or sr / 2
        mel_fb = _build_mel_filterbank(n_mels, n_fft, sr, self.f_min, self.f_max)
        self.register_buffer("mel_fb", mel_fb)

    @property
    def feature_dim(self) -> int:
        return self.n_mels

    def forward(self, wavs):
        """Compute log-mel filterbank features.

        Args:
            wavs: [B, T] tensor or list of 1-D tensors.

        Returns:
            Tensor of shape [B, T_frames, n_mels].
        """
        if isinstance(wavs, list):
            wavs = [w.to(torch.float32) for w in wavs]
            max_len = max(w.shape[-1] for w in wavs)
            batch = torch.stack([F.pad(w, (0, max_len - w.shape[-1])) for w in wavs])
        else:
            batch = wavs.unsqueeze(0) if wavs.ndim == 1 else wavs
        batch = batch.to(torch.float32)

        device = batch.device
        window = torch.hann_window(self.n_fft, device=device)

        stft = torch.view_as_real(torch.stft(
            batch,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        ))
        real = stft[..., 0]
        imag = stft[..., 1]
        power = real.pow(2) + imag.pow(2)

        mel_spec = torch.matmul(self.mel_fb.to(device), power)
        log_mel = torch.log(mel_spec + 1e-10)

        return log_mel.transpose(1, 2)  # [B, T_frames, n_mels]

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        # Same STFT issue as MfccExtractor — force dynamo exporter.
        super().export_to_onnx(out, quantize=quantize, dynamo=True, metadata=metadata)


class SincNetExtractor(BaseExtractor):
    """Learnable SincNet filterbank, ONNX-exportable.

    Applies a bank of learnable bandpass sinc filters to raw waveform,
    then computes frame energy via log1p + transpose.
    Output: [B, T, n_filters]

    This is a simplified SincNet — learnable frequency boundaries, fixed
    Hamming window. Suitable for micro/small tier models.

    Args:
        sr: Sample rate (default 16000).
        n_filters: Number of sinc filters (default 80).
        kernel_size: Filter kernel length in samples (default 251, must be odd).
        stride: Stride for the convolution (default 160 — same as hop_length).
        min_freq: Minimum filter frequency Hz (default 50.0).
        min_band: Minimum filter bandwidth Hz (default 50.0).
    """

    def __init__(self, sr: int = 16000, n_filters: int = 80,
                 kernel_size: int = 251, stride: int = 160,
                 min_freq: float = 50.0, min_band: float = 50.0):
        super().__init__(sample_rate=sr)
        if kernel_size % 2 == 0:
            kernel_size += 1  # must be odd

        self.kernel_size = kernel_size
        self.stride = stride
        self.min_freq = min_freq
        self.min_band = min_band
        self._feature_dim = n_filters

        # Learnable parameters: low frequency and bandwidth for each filter
        low_hz = torch.linspace(min_freq, sr / 2 - min_band - min_freq, n_filters)
        band_hz = torch.full((n_filters,), (sr / 2 - min_freq) / n_filters)
        self.low_hz_ = torch.nn.Parameter(low_hz)
        self.band_hz_ = torch.nn.Parameter(band_hz)

        # Pre-computed time indices and window (registered buffers for ONNX export)
        n_lin = torch.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size)
        self.register_buffer('_n', n_lin)
        window = 0.54 - 0.46 * torch.cos(
            2 * math.pi * torch.arange(kernel_size).float() / (kernel_size - 1)
        )
        self.register_buffer('_window', window)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @staticmethod
    def _sinc(x: torch.Tensor) -> torch.Tensor:
        """Normalised sinc: sin(pi*x)/(pi*x), with sinc(0)=1. ONNX-safe."""
        # Avoid division by zero using clamp — pure tensor ops, no python control flow
        x_safe = x.clone()
        zero_mask = x.abs() < 1e-8
        x_safe = torch.where(zero_mask, torch.ones_like(x), x)
        y = torch.sin(math.pi * x_safe) / (math.pi * x_safe)
        return torch.where(zero_mask, torch.ones_like(y), y)

    def forward(self, wavs):
        """Compute SincNet filterbank features.

        Args:
            wavs: [B, T] tensor or list of 1-D tensors.

        Returns:
            Tensor of shape [B, T_frames, n_filters].
        """
        if isinstance(wavs, list):
            wavs = [w.to(torch.float32) for w in wavs]
            max_len = max(w.shape[-1] for w in wavs)
            batch = torch.stack([F.pad(w, (0, max_len - w.shape[-1])) for w in wavs])
        else:
            batch = wavs.unsqueeze(0) if wavs.ndim == 1 else wavs
        batch = batch.to(torch.float32)

        # Build sinc filters from learned parameters (ensure same device as input)
        low_hz = self.min_freq + torch.abs(self.low_hz_.to(batch.device))   # [n_filters]
        high_hz = low_hz + self.min_band + torch.abs(self.band_hz_.to(batch.device))  # [n_filters]

        low_hz = torch.clamp(low_hz, float(self.min_freq),
                             float(self.sample_rate / 2 - self.min_band))
        high_hz = torch.clamp(high_hz, min=low_hz + self.min_band).clamp(
            max=float(self.sample_rate / 2))

        n = self._n.to(batch.device)  # [kernel_size]
        f_low = low_hz.unsqueeze(1) / self.sample_rate   # [n_filters, 1]
        f_high = high_hz.unsqueeze(1) / self.sample_rate  # [n_filters, 1]

        # Use manual sinc (sin(pi*x)/(pi*x)) for ONNX compatibility
        low_pass1 = 2 * f_low * self._sinc(2 * f_low * n)   # [n_filters, kernel_size]
        low_pass2 = 2 * f_high * self._sinc(2 * f_high * n)  # [n_filters, kernel_size]

        band_pass = low_pass2 - low_pass1   # [n_filters, kernel_size]
        band_pass = band_pass * self._window.to(batch.device)
        band_pass = band_pass / (band_pass.abs().sum(dim=1, keepdim=True) + 1e-8)

        filters = band_pass.unsqueeze(1)  # [n_filters, 1, kernel_size]

        # Apply filters: batch [B, T] -> [B, 1, T]
        out = F.conv1d(batch.unsqueeze(1), filters, stride=self.stride,
                       padding=self.kernel_size // 2)  # [B, n_filters, T_frames]

        out = torch.log1p(out.abs())

        return out.transpose(1, 2)  # [B, T_frames, n_filters]


class DeltaExtractor(BaseExtractor):
    """Wraps any BaseExtractor and appends delta + delta-delta features.

    Given a base extractor producing [B, T, F], this returns [B, T, 3*F]:
    - First F dims: original features
    - Next F dims: delta (first-order temporal derivative)
    - Last F dims: delta-delta (second-order temporal derivative)

    This is a pure-PyTorch operation and is fully ONNX-exportable.
    The deltas are computed using the standard ±N-frame regression formula.

    Args:
        base_extractor: Any BaseExtractor instance.
        delta_width: Number of frames on each side for delta computation (default 2).
    """

    def __init__(self, base_extractor: BaseExtractor, delta_width: int = 2):
        super().__init__(sample_rate=base_extractor.sample_rate)
        self.base = base_extractor
        self.delta_width = delta_width
        self._feature_dim = base_extractor.feature_dim * 3

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @staticmethod
    def _compute_delta(feats: torch.Tensor, width: int) -> torch.Tensor:
        """Compute delta features using ±width frame regression.

        Args:
            feats: [B, T, F]
            width: number of frames on each side

        Returns:
            delta: [B, T, F]
        """
        # Pad with edge replication so output length == input length
        # feats: [B, T, F] → pad T dim: (width, width)
        padded = F.pad(feats.transpose(1, 2), (width, width), mode='replicate')  # [B, F, T+2w]
        padded = padded.transpose(1, 2)  # [B, T+2w, F]

        denom = 2 * sum(i * i for i in range(1, width + 1))
        delta = torch.zeros_like(feats)
        for i in range(1, width + 1):
            delta += i * (padded[:, width + i:width + i + feats.shape[1]] -
                         padded[:, width - i:width - i + feats.shape[1]])
        return delta / denom

    def forward(self, wavs: WavInput) -> torch.Tensor:
        """Extract base features then append delta and delta-delta.

        Returns:
            Tensor of shape [B, T, 3*F]
        """
        feats = self.base(wavs)                                          # [B, T, F]
        delta = self._compute_delta(feats, self.delta_width)             # [B, T, F]
        delta2 = self._compute_delta(delta, self.delta_width)            # [B, T, F]
        return torch.cat([feats, delta, delta2], dim=-1)                 # [B, T, 3*F]

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        """Export the full DeltaExtractor (base + delta computation) to ONNX.

        Forces dynamo=True so that any STFT-based base extractor (MFCC, Filterbank, …)
        exports correctly — the TorchScript exporter cannot handle complex torch.stft output.
        """
        # Always use dynamo: base extractor may use torch.stft (MFCC/Filterbank/PLP/PNCC/CQT)
        # which the TorchScript ONNX exporter cannot handle.
        super().export_to_onnx(out, quantize=quantize, dynamo=True, metadata=metadata)


class GammatoneExtractor(BaseExtractor):
    """Gammatone filterbank, ONNX-exportable.

    Models the human auditory system using gammatone-shaped bandpass filters
    on an ERB (Equivalent Rectangular Bandwidth) frequency scale.
    Better than mel filterbank in noisy environments.

    Output: [B, T, n_filters]

    All filter coefficients are computed at init and stored as buffers —
    no Python loops in forward(), fully ONNX-exportable.

    Args:
        sr: Sample rate (default 16000).
        n_filters: Number of gammatone filters (default 64).
        f_min: Minimum centre frequency in Hz (default 50.0).
        f_max: Maximum centre frequency in Hz (default sr/2).
        frame_len: Analysis window in samples (default 400).
        hop_length: Hop length in samples (default 160).
        order: Gammatone filter order (default 4).
    """

    def __init__(self, sr: int = 16000, n_filters: int = 64,
                 f_min: float = 50.0, f_max: float = None,
                 frame_len: int = 400, hop_length: int = 160,
                 order: int = 4):
        super().__init__(sample_rate=sr)
        self.n_filters = n_filters
        self.frame_len = frame_len
        self.hop_length = hop_length
        self.order = order
        f_max = f_max or sr / 2

        # Compute ERB centre frequencies
        centre_freqs = self._erb_space(f_min, f_max, n_filters)

        # Build gammatone impulse responses as conv filters
        # Each filter: h(t) = t^(order-1) * exp(-2π*b*t) * cos(2π*f_c*t)
        # where b = 1.019 * ERB(f_c) (equivalent rectangular bandwidth)
        t = torch.arange(frame_len, dtype=torch.float32) / sr

        filters = []
        for fc in centre_freqs:
            erb = 24.7 * (4.37 * fc / 1000 + 1)  # ERB formula
            b = 1.019 * erb
            envelope = t.pow(order - 1) * torch.exp(-2 * math.pi * b * t)
            carrier = torch.cos(2 * math.pi * fc * t)
            h = envelope * carrier
            # Normalize
            h = h / (h.abs().max() + 1e-8)
            filters.append(h)

        # Stack: [n_filters, frame_len]
        filter_bank = torch.stack(filters, dim=0)
        self.register_buffer('filter_bank', filter_bank)
        self._feature_dim = n_filters

    @staticmethod
    def _erb_space(f_min: float, f_max: float, n: int) -> list:
        """Compute n centre frequencies uniformly spaced on the ERB scale."""
        # ERB scale: E(f) = 21.4 * log10(0.00437*f + 1)
        e_min = 21.4 * math.log10(0.00437 * f_min + 1)
        e_max = 21.4 * math.log10(0.00437 * f_max + 1)
        erb_points = [e_min + i * (e_max - e_min) / (n - 1) for i in range(n)]
        # Invert: f = (10^(E/21.4) - 1) / 0.00437
        return [(10 ** (e / 21.4) - 1) / 0.00437 for e in erb_points]

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, wavs: WavInput) -> torch.Tensor:
        """Apply gammatone filterbank and compute log energy per frame.

        Returns:
            Tensor of shape [B, T_frames, n_filters]
        """
        if isinstance(wavs, list):
            wavs = [w.to(torch.float32) for w in wavs]
            max_len = max(w.shape[-1] for w in wavs)
            batch = torch.stack([F.pad(w, (0, max_len - w.shape[-1])) for w in wavs])
        else:
            batch = wavs.unsqueeze(0) if wavs.ndim == 1 else wavs
        batch = batch.to(torch.float32)

        # Apply all gammatone filters via Conv1d: [B, 1, T] → [B, n_filters, T]
        fb = self.filter_bank.to(batch.device).unsqueeze(1)  # [n_filters, 1, frame_len]
        out = F.conv1d(batch.unsqueeze(1), fb,
                       stride=self.hop_length,
                       padding=self.frame_len // 2)  # [B, n_filters, T_frames]

        # Log energy
        out = torch.log1p(out.pow(2))

        return out.transpose(1, 2)  # [B, T_frames, n_filters]


class HubertExtractor(BaseExtractor):
    """HuBERT encoder for training. Requires `transformers` library.
    Export with export_to_onnx() -> use OnnxFeatureExtractor for inference.
    """
    _REQUIRES_TRANSFORMERS = True

    def __init__(self, model_name: str = "voidful/hubert-tiny-v2",
                 sample_rate: int = 16000, device: str = "auto") -> None:
        super().__init__(sample_rate=sample_rate, device=device)
        try:
            from transformers import HubertModel
        except ImportError:
            raise ImportError(
                "Install transformers for HubertExtractor: pip install transformers"
            )
        self.hubert = HubertModel.from_pretrained(model_name).eval().to(self.device)
        self._feature_dim = self.hubert.config.hidden_size

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, wavs: WavInput) -> torch.Tensor:
        wav_list = ensure_wav_list(wavs)
        max_len = max(w.shape[0] for w in wav_list)
        padded = [F.pad(w, (0, max_len - w.shape[0])) if w.shape[0] < max_len else w for w in wav_list]
        wav_tensor = torch.stack(padded, dim=0).to(torch.float32).to(self.device)
        abs_max = wav_tensor.abs().amax(dim=1, keepdim=True).clamp(min=1e-9)
        wav_tensor = wav_tensor / abs_max
        with torch.no_grad():
            feats = self.hubert(wav_tensor).last_hidden_state
        return feats  # [B, T, C]

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        raise NotImplementedError(
            "HubertExtractor uses a HuggingFace transformers model that is too large for "
            "CPU-only deployment. Export it separately with optimum, then load the ONNX "
            "file via OnnxFeatureExtractor."
        )


class Wav2Vec2Extractor(BaseExtractor):
    """Wav2Vec2 encoder for training. Requires `transformers` library.
    Export with export_to_onnx() -> use OnnxFeatureExtractor for inference.
    """
    _REQUIRES_TRANSFORMERS = True

    def __init__(self, model_name: str = "patrickvonplaten/tiny-wav2vec2-no-tokenizer",
                 sample_rate: int = 16000, device: str = "auto") -> None:
        super().__init__(sample_rate=sample_rate, device=device)
        try:
            from transformers import Wav2Vec2Model
        except ImportError:
            raise ImportError(
                "Install transformers for Wav2Vec2Extractor: pip install transformers"
            )
        self.model = Wav2Vec2Model.from_pretrained(model_name).eval().to(self.device)
        self._feature_dim = self.model.config.hidden_size

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        if isinstance(wavs, list):
            wavs = torch.stack(wavs, dim=0)
        if wavs.ndim == 1:
            wavs = wavs.unsqueeze(0)
        wavs = wavs.to(self.device)
        wavs = (wavs - wavs.mean(dim=-1, keepdim=True)) / (wavs.std(dim=-1, keepdim=True) + 1e-5)
        with torch.no_grad():
            outputs = self.model(wavs)
        return outputs.last_hidden_state  # [B, T', hidden]

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        raise NotImplementedError(
            "Wav2Vec2Extractor uses a HuggingFace transformers model. Export it separately "
            "with optimum, then load the ONNX file via OnnxFeatureExtractor."
        )


class Wav2Vec2BertExtractor(BaseExtractor):
    """facebook/w2v-bert-2.0 encoder for training. Requires `transformers`.

    Produces 1024-dimensional features. Export with export_to_onnx() then
    load with OnnxFeatureExtractor for inference.

    _REQUIRES_TRANSFORMERS = True

    Args:
        model_name: HuggingFace model identifier (default: facebook/w2v-bert-2.0).
        sample_rate: Audio sample rate (default 16000).
        device: Torch device string or "auto".
    """
    _REQUIRES_TRANSFORMERS = True

    def __init__(self, model_name: str = "facebook/w2v-bert-2.0",
                 sample_rate: int = 16000, device: str = "auto") -> None:
        super().__init__(sample_rate=sample_rate, device=device)
        try:
            from transformers import Wav2Vec2BertModel
        except ImportError:
            raise ImportError(
                "Install transformers for Wav2Vec2BertExtractor: pip install transformers"
            )
        self.model = Wav2Vec2BertModel.from_pretrained(model_name).eval().to(self.device)
        self._feature_dim = self.model.config.hidden_size

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, wavs: WavInput) -> torch.Tensor:
        """Extract Wav2Vec2Bert features.

        Args:
            wavs: [B, T] tensor or list of 1-D tensors.

        Returns:
            Tensor of shape [B, T_frames, hidden_size].
        """
        if isinstance(wavs, list):
            wavs = torch.stack(wavs, dim=0)
        if wavs.ndim == 1:
            wavs = wavs.unsqueeze(0)
        wavs = wavs.to(torch.float32).to(self.device)
        wavs = (wavs - wavs.mean(dim=-1, keepdim=True)) / (wavs.std(dim=-1, keepdim=True) + 1e-5)
        with torch.no_grad():
            outputs = self.model(wavs)
        return outputs.last_hidden_state  # [B, T', hidden_size]

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        raise NotImplementedError(
            "Wav2Vec2BertExtractor uses a HuggingFace transformers model. Export it separately "
            "with optimum, then load the ONNX file via OnnxFeatureExtractor."
        )


class TorchAudioHubertExtractor(BaseExtractor):
    """HuBERT extractor using ``torchaudio.pipelines`` — no ``transformers`` dependency.

    Uses the bundled torchaudio HuBERT model which is downloaded from
    torch hub on first use. Produces the same ``[B, T, hidden]`` output
    as :class:`HubertExtractor` but only requires ``torchaudio``.

    Args:
        bundle_name: Name of the torchaudio pipeline bundle.
            Options: ``"HUBERT_BASE"``, ``"HUBERT_LARGE"``,
            ``"HUBERT_XLARGE"``, ``"HUBERT_ASR_LARGE"``, ``"HUBERT_ASR_XLARGE"``.
        sample_rate: Audio sample rate (must match bundle's expected rate).
        device: Torch device string or ``"auto"``.
    """

    _BUNDLE_MAP = {
        "HUBERT_BASE": "torchaudio.pipelines.HUBERT_BASE",
        "HUBERT_LARGE": "torchaudio.pipelines.HUBERT_LARGE",
        "HUBERT_XLARGE": "torchaudio.pipelines.HUBERT_XLARGE",
        "HUBERT_ASR_LARGE": "torchaudio.pipelines.HUBERT_ASR_LARGE",
        "HUBERT_ASR_XLARGE": "torchaudio.pipelines.HUBERT_ASR_XLARGE",
    }

    def __init__(
        self,
        bundle_name: str = "HUBERT_BASE",
        sample_rate: int = 16000,
        device: str = "auto",
    ) -> None:
        super().__init__(sample_rate=sample_rate, device=device)
        import torchaudio

        if bundle_name not in self._BUNDLE_MAP:
            raise ValueError(
                f"Unknown bundle {bundle_name!r}. "
                f"Available: {list(self._BUNDLE_MAP)}"
            )
        bundle = getattr(torchaudio.pipelines, bundle_name)
        self.model = bundle.get_model().eval().to(self.device)
        expected_sr = bundle.sample_rate
        if sample_rate != expected_sr:
            logger.warning(
                "TorchAudioHubertExtractor: bundle expects %d Hz but got %d Hz. "
                "Audio will NOT be resampled automatically.",
                expected_sr, sample_rate,
            )
        # Feature dim from a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, sample_rate, device=self.device)
            out, _ = self.model.extract_features(dummy)
            self._feature_dim = out[-1].shape[-1]

    @property
    def feature_dim(self) -> int:
        """Output feature dimension."""
        return self._feature_dim

    def forward(self, wavs: WavInput, **kwargs) -> torch.Tensor:
        """Extract HuBERT features from audio.

        Args:
            wavs: Audio input.

        Returns:
            ``[B, T, hidden]`` feature tensor from the last transformer layer.
        """
        wav_list = ensure_wav_list(wavs)
        max_len = max(w.shape[0] for w in wav_list)
        padded = [
            F.pad(w, (0, max_len - w.shape[0])) if w.shape[0] < max_len else w
            for w in wav_list
        ]
        wav_tensor = torch.stack(padded, dim=0).to(torch.float32).to(self.device)
        abs_max = wav_tensor.abs().amax(dim=1, keepdim=True).clamp(min=1e-9)
        wav_tensor = wav_tensor / abs_max
        with torch.no_grad():
            features, _ = self.model.extract_features(wav_tensor)
        return features[-1]  # last layer: [B, T, hidden]

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        raise NotImplementedError(
            "TorchAudioHubertExtractor wraps a torchaudio JIT model with complex internal "
            "control flow that cannot be traced by the PyTorch ONNX exporter. "
            "Export the base extractor separately."
        )


class LEAFExtractor(BaseExtractor):
    """LEAF: Learnable Audio Frontend (Zeghidour et al., ICLR 2021, Google).

    Replaces fixed mel filterbanks with fully learnable components:
    - Gabor convolution layer (learnable center freq + bandwidth)
    - Squared modulus (energy)
    - Gaussian low-pass pooling (learnable smoothing)
    - Per-Channel Energy Normalization (PCEN)

    All components are differentiable — the entire frontend is trained
    end-to-end with the classifier.

    Args:
        sr: Sample rate.
        n_filters: Number of filters (output feature dimension).
        window_len: Window length in samples (default 400 = 25ms at 16kHz).
        hop_length: Hop length in samples (default 160 = 10ms at 16kHz).
        min_freq: Minimum center frequency in Hz.
        max_freq: Maximum center frequency in Hz (default sr/2).
        pcen_alpha: PCEN alpha (smoothing).
        pcen_delta: PCEN delta (bias).
        pcen_r: PCEN r (exponent).
        pcen_s: PCEN s (gain normalization strength).
    """

    def __init__(
        self,
        sr: int = 16000,
        n_filters: int = 40,
        window_len: int = 400,
        hop_length: int = 160,
        min_freq: float = 60.0,
        max_freq: float = None,
        pcen_alpha: float = 0.96,
        pcen_delta: float = 2.0,
        pcen_r: float = 0.5,
        pcen_s: float = 0.025,
    ) -> None:
        super().__init__(sample_rate=sr)
        self._n_filters = n_filters
        self.hop_length = hop_length
        max_freq = max_freq or sr / 2.0

        # --- Gabor filterbank (learnable sinc-like filters) ---
        # Initialize center frequencies linearly in mel scale
        low_mel = 2595.0 * math.log10(1.0 + min_freq / 700.0)
        high_mel = 2595.0 * math.log10(1.0 + max_freq / 700.0)
        mel_points = torch.linspace(low_mel, high_mel, n_filters)
        center_hz = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        # Learnable parameters: center frequency and bandwidth
        self.center_hz = torch.nn.Parameter(center_hz)
        # Initialize bandwidth as ERB
        erb = 24.7 * (4.37 * center_hz / 1000.0 + 1.0)
        self.bandwidth_hz = torch.nn.Parameter(erb)

        self.window_len = window_len
        # Time axis for Gabor kernel
        t = torch.arange(-(window_len // 2), (window_len + 1) // 2, dtype=torch.float32) / sr
        self.register_buffer("t", t)

        # --- Gaussian low-pass pooling ---
        self.lowpass_sigma = torch.nn.Parameter(torch.full((n_filters,), 2.0))

        # --- PCEN parameters (learnable) ---
        self.pcen_alpha = torch.nn.Parameter(torch.full((1, n_filters, 1), pcen_alpha))
        self.pcen_delta = torch.nn.Parameter(torch.full((1, n_filters, 1), pcen_delta))
        self.pcen_r = torch.nn.Parameter(torch.full((1, n_filters, 1), pcen_r))
        self.pcen_s = torch.nn.Parameter(torch.full((1, n_filters, 1), pcen_s))

    @property
    def feature_dim(self) -> int:
        return self._n_filters

    def _gabor_filters(self) -> torch.Tensor:
        """Compute Gabor filters from learnable parameters. Returns [n_filters, 1, window_len]."""
        center = self.center_hz.clamp(min=1.0)
        bw = self.bandwidth_hz.clamp(min=1.0)
        # Gaussian envelope
        sigma = 1.0 / (2.0 * math.pi * bw.unsqueeze(1))
        gaussian = torch.exp(-0.5 * (self.t.unsqueeze(0) / sigma) ** 2)
        # Cosine carrier
        carrier = torch.cos(2.0 * math.pi * center.unsqueeze(1) * self.t.unsqueeze(0))
        filters = gaussian * carrier
        # Normalize
        filters = filters / (filters.norm(dim=1, keepdim=True) + 1e-8)
        return filters.unsqueeze(1)  # [n_filters, 1, window_len]

    def _gaussian_lowpass(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-channel Gaussian low-pass pooling along time."""
        sigma = self.lowpass_sigma.clamp(min=0.5)
        kernel_size = int(sigma.max().item() * 6) | 1  # odd
        kernel_size = max(kernel_size, 3)
        half = kernel_size // 2
        t = torch.arange(-half, half + 1, device=x.device, dtype=torch.float32)
        # Per-filter Gaussian kernel [n_filters, 1, kernel_size]
        kernels = torch.exp(-0.5 * (t.unsqueeze(0) / sigma.unsqueeze(1)) ** 2)
        kernels = kernels / (kernels.sum(dim=1, keepdim=True) + 1e-8)
        kernels = kernels.unsqueeze(1)  # [n_filters, 1, kernel_size]
        # Depthwise conv
        return F.conv1d(x, kernels, padding=half, groups=x.size(1))

    def _pcen(self, x: torch.Tensor) -> torch.Tensor:
        """Per-Channel Energy Normalization."""
        # Compute smoothed energy via EMA approximation (simple IIR)
        alpha = torch.sigmoid(self.pcen_alpha)
        # Use a simple low-pass as approximation of the IIR smoother
        smooth = self._ema_smooth(x, alpha)
        # PCEN formula
        return (x / (smooth + self.pcen_delta).pow(self.pcen_r) + 1e-6).pow(self.pcen_s) - 1.0

    @staticmethod
    def _ema_smooth(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Exponential moving average along time (dim=2) for PCEN."""
        C = x.size(1)
        kernel_size = min(x.size(2), 21)
        # alpha: [1, C, 1] → [C]
        a = alpha.view(C)
        t_k = torch.arange(kernel_size, device=x.device, dtype=torch.float32).flip(0)
        # weights: [C, K]
        weights = (1 - a).unsqueeze(1) * a.unsqueeze(1).pow(t_k.unsqueeze(0))
        weights = weights.unsqueeze(1)  # [C, 1, K]
        pad = kernel_size - 1
        return F.conv1d(F.pad(x, (pad, 0)), weights, groups=C)

    def forward(self, wavs: WavInput, **kwargs) -> torch.Tensor:
        """Extract LEAF features from raw audio.

        Args:
            wavs: ``[B, T]`` tensor or list of 1-D tensors.

        Returns:
            ``[B, T_frames, n_filters]`` learnable filterbank features.
        """
        wav_list = ensure_wav_list(wavs)
        max_len = max(w.shape[-1] for w in wav_list)
        _dev = self.center_hz.device
        batch = torch.stack([
            F.pad(w, (0, max_len - w.shape[-1])) for w in wav_list
        ]).to(_dev)

        if batch.ndim == 1:
            batch = batch.unsqueeze(0)
        # [B, T] → [B, 1, T]
        x = batch.unsqueeze(1)

        # Step 1: Gabor convolution
        filters = self._gabor_filters()  # [n_filters, 1, window_len]
        x = F.conv1d(x, filters, stride=self.hop_length,
                      padding=self.window_len // 2)  # [B, n_filters, T']

        # Step 2: Squared modulus
        x = x ** 2

        # Step 3: Gaussian low-pass pooling
        x = self._gaussian_lowpass(x)

        # Step 4: PCEN
        x = self._pcen(x)

        # [B, n_filters, T'] → [B, T', n_filters]
        return x.transpose(1, 2)


# ---------------------- PLP Extractor ----------------------


class PLPExtractor(BaseExtractor):
    """Perceptual Linear Prediction feature extractor (Hermansky 1990).

    Models human auditory perception using:
    1. Bark-scale spectral warping
    2. Equal-loudness pre-emphasis
    3. Intensity-loudness power law (cube root compression)
    4. Autoregressive (LP) modeling
    5. Cepstral conversion

    More noise-robust than MFCC for many conditions.

    Args:
        sr: Sample rate.
        n_plp: Number of PLP coefficients to output.
        n_fft: FFT size.
        hop_length: Hop length in samples.
        n_bark: Number of Bark-scale filters.
        lp_order: Linear prediction order.
    """

    def __init__(
        self,
        sr: int = 16000,
        n_plp: int = 13,
        n_fft: int = 512,
        hop_length: int = 160,
        n_bark: int = 21,
        lp_order: int = 12,
    ) -> None:
        super().__init__(sample_rate=sr)
        self._n_plp = n_plp
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bark = n_bark
        self.lp_order = lp_order

        # Precompute Bark filterbank
        n_bins = n_fft // 2 + 1
        freqs = torch.linspace(0, sr / 2, n_bins)
        barks = 6.0 * torch.arcsinh(freqs / 600.0)
        bark_min = barks[1]
        bark_max = barks[-2]
        bark_centers = torch.linspace(bark_min, bark_max, n_bark)
        bark_width = 1.3  # Bark bandwidth

        # Triangular Bark filterbank [n_bark, n_bins]
        fb = torch.zeros(n_bark, n_bins)
        for i in range(n_bark):
            low = bark_centers[i] - bark_width
            high = bark_centers[i] + bark_width
            for j in range(n_bins):
                if low <= barks[j] <= bark_centers[i]:
                    fb[i, j] = (barks[j] - low) / (bark_centers[i] - low + 1e-8)
                elif bark_centers[i] < barks[j] <= high:
                    fb[i, j] = (high - barks[j]) / (high - bark_centers[i] + 1e-8)
        self.register_buffer("bark_fb", fb)

        # Equal-loudness weighting (simplified ITU-R 468)
        f_hz = bark_centers * 600.0  # rough bark → Hz
        # Approximate equal-loudness curve (inverted threshold of hearing)
        el = (f_hz ** 2 + 56.8e6) * f_hz ** 4 / (
            (f_hz ** 2 + 6.3e6) ** 2 * (f_hz ** 2 + 3.8e8) + 1e-10
        )
        el = el / (el.max() + 1e-10)
        self.register_buffer("equal_loudness", el.unsqueeze(0).unsqueeze(-1))

        # Hanning window
        self.register_buffer("window", torch.hann_window(n_fft))

    @property
    def feature_dim(self) -> int:
        return self._n_plp

    def forward(self, wavs: WavInput, **kwargs) -> torch.Tensor:
        """Extract PLP features.

        Args:
            wavs: ``[B, T]`` tensor or list of 1-D tensors.

        Returns:
            ``[B, T_frames, n_plp]`` PLP coefficients.
        """
        wav_list = ensure_wav_list(wavs)
        max_len = max(w.shape[-1] for w in wav_list)
        batch = torch.stack([
            F.pad(w, (0, max_len - w.shape[-1])) for w in wav_list
        ]).to(self.bark_fb.device)

        # Power spectrum
        spec = torch.view_as_real(torch.stft(
            batch, self.n_fft, self.hop_length, window=self.window,
            return_complex=True, onesided=True,
        ))
        power = spec[..., 0] ** 2 + spec[..., 1] ** 2  # [B, n_bins, T']

        # Bark-scale warping
        bark_spec = torch.matmul(self.bark_fb, power)  # [B, n_bark, T']

        # Equal-loudness pre-emphasis
        bark_spec = bark_spec * self.equal_loudness

        # Intensity-loudness power law (cube root)
        bark_spec = bark_spec.clamp(min=1e-10).pow(1.0 / 3.0)

        # Simplified cepstral conversion via DCT-like transform
        # (full LP analysis would require Levinson-Durbin, approximated here)
        n = torch.arange(self._n_plp, device=bark_spec.device, dtype=torch.float32)
        k = torch.arange(self.n_bark, device=bark_spec.device, dtype=torch.float32)
        dct_mat = torch.cos(math.pi * n.unsqueeze(1) * (2 * k.unsqueeze(0) + 1) / (2 * self.n_bark))
        plp = torch.matmul(dct_mat, bark_spec)  # [B, n_plp, T']

        return plp.transpose(1, 2)  # [B, T', n_plp]

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        super().export_to_onnx(out, quantize=quantize, dynamo=True, metadata=metadata)


# ---------------------- PNCC Extractor ----------------------


class PNCCExtractor(BaseExtractor):
    """Power-Normalized Cepstral Coefficients (Kim & Stern 2016).

    Designed for noise robustness using:
    1. Gammatone-like filterbank
    2. Medium-time power processing (asymmetric noise suppression)
    3. Power-law nonlinearity (1/15 power instead of log)
    4. DCT to produce cepstral coefficients

    Significantly outperforms MFCC in noisy conditions.

    Args:
        sr: Sample rate.
        n_pncc: Number of PNCC coefficients.
        n_fft: FFT size.
        hop_length: Hop length in samples.
        n_filters: Number of gammatone-like filters.
        power: Power-law exponent (default 1/15).
    """

    def __init__(
        self,
        sr: int = 16000,
        n_pncc: int = 13,
        n_fft: int = 512,
        hop_length: int = 160,
        n_filters: int = 40,
        power: float = 1.0 / 15.0,
    ) -> None:
        super().__init__(sample_rate=sr)
        self._n_pncc = n_pncc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self._power = power

        # Gammatone-like filterbank on ERB scale
        n_bins = n_fft // 2 + 1
        freqs = torch.linspace(0, sr / 2, n_bins)
        erb_low = 9.265 * math.log(1 + 20.0 / (24.7 * 9.265))
        erb_high = 9.265 * math.log(1 + (sr / 2) / (24.7 * 9.265))
        erb_points = torch.linspace(erb_low, erb_high, n_filters + 2)
        center_freqs = 24.7 * 9.265 * (torch.exp(erb_points / 9.265) - 1)

        fb = torch.zeros(n_filters, n_bins)
        for i in range(n_filters):
            low = center_freqs[i]
            mid = center_freqs[i + 1]
            high = center_freqs[i + 2]
            for j in range(n_bins):
                if low <= freqs[j] <= mid:
                    fb[i, j] = (freqs[j] - low) / (mid - low + 1e-8)
                elif mid < freqs[j] <= high:
                    fb[i, j] = (high - freqs[j]) / (high - mid + 1e-8)
        self.register_buffer("filterbank", fb)
        self.register_buffer("window", torch.hann_window(n_fft))

        # DCT matrix [n_pncc, n_filters]
        n = torch.arange(n_pncc, dtype=torch.float32)
        k = torch.arange(n_filters, dtype=torch.float32)
        dct_mat = torch.cos(math.pi * n.unsqueeze(1) * (2 * k.unsqueeze(0) + 1) / (2 * n_filters))
        self.register_buffer("dct_mat", dct_mat)

    @property
    def feature_dim(self) -> int:
        return self._n_pncc

    def forward(self, wavs: WavInput, **kwargs) -> torch.Tensor:
        """Extract PNCC features.

        Args:
            wavs: ``[B, T]`` tensor or list of 1-D tensors.

        Returns:
            ``[B, T_frames, n_pncc]`` PNCC coefficients.
        """
        wav_list = ensure_wav_list(wavs)
        max_len = max(w.shape[-1] for w in wav_list)
        batch = torch.stack([
            F.pad(w, (0, max_len - w.shape[-1])) for w in wav_list
        ]).to(self.filterbank.device)

        # Power spectrum
        spec = torch.view_as_real(torch.stft(
            batch, self.n_fft, self.hop_length, window=self.window,
            return_complex=True, onesided=True,
        ))
        power = spec[..., 0] ** 2 + spec[..., 1] ** 2  # [B, n_bins, T']

        # Gammatone filterbank
        filtered = torch.matmul(self.filterbank, power)  # [B, n_filters, T']

        # Medium-time power processing (asymmetric temporal smoothing)
        # Forward pass: slow adaptation to rising energy (noise floor tracking)
        # Simplified as temporal mean with asymmetric weighting
        kernel_size = min(filtered.size(2), 11)
        if kernel_size > 1:
            avg_kernel = torch.ones(1, 1, kernel_size, device=filtered.device) / kernel_size
            padded = F.pad(filtered, (kernel_size - 1, 0))
            noise_floor = F.conv1d(
                padded.view(-1, 1, padded.size(2)), avg_kernel
            ).view(filtered.shape)
            # Asymmetric suppression
            filtered = torch.max(filtered - noise_floor, torch.zeros_like(filtered))
            filtered = filtered + 1e-10

        # Power-law nonlinearity (1/15 power instead of log)
        filtered = filtered.clamp(min=1e-10).pow(self._power)

        # DCT → cepstral coefficients
        pncc = torch.matmul(self.dct_mat, filtered)  # [B, n_pncc, T']

        return pncc.transpose(1, 2)  # [B, T', n_pncc]

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        super().export_to_onnx(out, quantize=quantize, dynamo=True, metadata=metadata)


# ---------------------- CQT Extractor ----------------------


class CQTExtractor(BaseExtractor):
    """Constant-Q Transform feature extractor.

    Better low-frequency resolution than FFT (logarithmic frequency spacing).
    Each frequency bin has a constant Q factor (center_freq / bandwidth).
    Good for tonal wake words and music-like patterns.

    Args:
        sr: Sample rate.
        n_bins: Number of CQT bins per octave.
        n_octaves: Number of octaves.
        f_min: Minimum frequency in Hz.
        hop_length: Hop length in samples.
    """

    def __init__(
        self,
        sr: int = 16000,
        n_bins: int = 12,
        n_octaves: int = 7,
        f_min: float = 32.7,
        hop_length: int = 160,
    ) -> None:
        super().__init__(sample_rate=sr)
        self._total_bins = n_bins * n_octaves
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.n_octaves = n_octaves

        # Compute CQT center frequencies (geometric spacing)
        total = self._total_bins
        freqs = f_min * (2.0 ** (torch.arange(total, dtype=torch.float32) / n_bins))
        # Filter out frequencies above Nyquist
        valid = freqs < sr / 2
        freqs = freqs[valid]
        self._total_bins = len(freqs)

        # Q factor
        q = 1.0 / (2.0 ** (1.0 / n_bins) - 1.0)

        # Precompute CQT kernels as complex sinusoids windowed by Hanning
        max_len = int(q * sr / freqs[0].item()) + 1
        # For efficiency, use FFT-based CQT approximation
        # Store center frequencies and compute per-frame via Goertzel-like approach
        self.register_buffer("center_freqs", freqs)
        self.register_buffer("q_val", torch.tensor(q))

        # Use STFT with large FFT + frequency-domain resampling
        self.n_fft = max(512, 2 ** int(math.ceil(math.log2(max_len))))
        self.register_buffer("window", torch.hann_window(self.n_fft))

        # CQT filter mapping: for each CQT bin, which FFT bins to weight
        fft_freqs = torch.linspace(0, sr / 2, self.n_fft // 2 + 1)
        fb = torch.zeros(self._total_bins, self.n_fft // 2 + 1)
        for i, cf in enumerate(freqs):
            bw = cf / q
            low = cf - bw / 2
            high = cf + bw / 2
            mask = (fft_freqs >= low) & (fft_freqs <= high)
            if mask.any():
                weights = torch.zeros_like(fft_freqs)
                weights[mask] = 1.0 - ((fft_freqs[mask] - cf) / (bw / 2 + 1e-8)).abs()
                weights = weights.clamp(min=0)
                fb[i] = weights / (weights.sum() + 1e-8)
        self.register_buffer("cqt_fb", fb)

    @property
    def feature_dim(self) -> int:
        return self._total_bins

    def forward(self, wavs: WavInput, **kwargs) -> torch.Tensor:
        """Extract CQT features.

        Args:
            wavs: ``[B, T]`` tensor or list of 1-D tensors.

        Returns:
            ``[B, T_frames, n_cqt_bins]`` log-magnitude CQT features.
        """
        wav_list = ensure_wav_list(wavs)
        max_len = max(w.shape[-1] for w in wav_list)
        batch = torch.stack([
            F.pad(w, (0, max_len - w.shape[-1])) for w in wav_list
        ]).to(self.cqt_fb.device)

        # STFT
        spec = torch.view_as_real(torch.stft(
            batch, self.n_fft, self.hop_length, window=self.window,
            return_complex=True, onesided=True,
        ))
        magnitude = (spec[..., 0] ** 2 + spec[..., 1] ** 2).sqrt()  # [B, n_bins, T']

        # Apply CQT filterbank
        cqt = torch.matmul(self.cqt_fb, magnitude)  # [B, n_cqt_bins, T']

        # Log-magnitude
        cqt = (cqt.clamp(min=1e-10)).log()

        return cqt.transpose(1, 2)  # [B, T', n_cqt_bins]

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        super().export_to_onnx(out, quantize=quantize, dynamo=True, metadata=metadata)


# ---------------------- Feature Enrichment Wrappers ----------------------


class SileroVadWrapper(BaseExtractor):
    """Wrapper that appends Silero VAD probabilities to any base extractor.

    Uses the highly-optimized snakers4/silero-vad (v4+) model to compute
    robust speech probabilities per frame and concatenates them to the 
    base features. This provides a state-of-the-art neural VAD signal 
    to the downstream classifier without needing to train it from scratch.

    Silero VAD operates on 512-sample chunks at 16kHz. This wrapper 
    unfolds the audio into chunks, runs batch inference, and interpolates 
    the resulting probabilities to match the time dimension of the base features.

    If *onnx_path* is provided, it uses ONNX Runtime for inference, which 
    is recommended for production environments and standard blackbox pipelines.
    Otherwise, it lazy-loads the PyTorch Hub model.

    Note: Direct ONNX export of the full combined pipeline (Base + Silero) 
    is not supported due to external model dependencies.

    Args:
        base_extractor: Any BaseExtractor to wrap.
        onnx_path: Optional path to a pre-trained Silero VAD ONNX file.
        force_reload: Force redownload of the Silero model from torch.hub.
    """

    def __init__(
        self,
        base_extractor: BaseExtractor,
        onnx_path: str | None = None,
        force_reload: bool = False
    ) -> None:
        super().__init__(sample_rate=base_extractor.sample_rate)
        if base_extractor.sample_rate != 16000:
            raise ValueError("Silero VAD requires a 16000 Hz sample rate.")
            
        self.base = base_extractor
        self.onnx_path = onnx_path
        self.chunk_size = 512
        
        self._force_reload = force_reload
        if onnx_path:
            import onnxruntime as ort
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if torch.cuda.is_available()
                else ["CPUExecutionProvider"]
            )
            self.ort_sess = ort.InferenceSession(onnx_path, providers=providers)
            self._input_name = self.ort_sess.get_inputs()[0].name
            self._output_name = self.ort_sess.get_outputs()[0].name
            self.vad_model = None
        else:
            # Lazy: loaded on first forward() to avoid network access at import/init
            self.vad_model = None
            self.ort_sess = None

    @property
    def feature_dim(self) -> int:
        return self.base.feature_dim + 1

    def forward(self, wavs: WavInput, **kwargs) -> torch.Tensor:
        """Extract base features and append Silero VAD probabilities.

        Args:
            wavs: Audio input.

        Returns:
            ``[B, T, base_dim + 1]`` enriched features.
        """
        # Lazy-load PyTorch Hub model on first forward() call
        if self.vad_model is None and self.ort_sess is None:
            import torch.hub
            self.vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=self._force_reload,
                trust_repo=True,
            )
            self.vad_model.eval()

        # 1. Base features [B, T_base, F]
        base_feats = self.base(wavs, **kwargs)

        # Format wavs to [B, L]
        wav_list = ensure_wav_list(wavs)
        max_len = max(w.shape[-1] for w in wav_list)
        # Pad to multiple of chunk_size for easy unfolding
        pad_len = (self.chunk_size - (max_len % self.chunk_size)) % self.chunk_size
        batch = torch.stack([
            F.pad(w, (0, max_len - w.shape[-1] + pad_len)) for w in wav_list
        ]).to(base_feats.device)
        
        B, L = batch.shape
        T_chunks = L // self.chunk_size
        
        # 2. Unfold into chunks [B * T_chunks, chunk_size]
        chunks = batch.view(B * T_chunks, self.chunk_size)
        
        # 3. Silero VAD inference
        if self.ort_sess:
            # ONNX Runtime path
            # Convert to numpy for ORT
            chunks_np = chunks.detach().cpu().numpy().astype(np.float32)
            vad_out = self.ort_sess.run([self._output_name], {self._input_name: chunks_np})[0]
            vad_probs = torch.from_numpy(vad_out).to(base_feats.device)
        else:
            # PyTorch Hub path
            with torch.no_grad():
                self.vad_model.to(chunks.device)
                vad_probs = self.vad_model(chunks, self.sample_rate) # [B * T_chunks, 1]
            
        # Reshape to [B, 1, T_chunks] for interpolation
        vad_probs = vad_probs.view(B, T_chunks, 1).transpose(1, 2)
        
        # 4. Interpolate to match base feature length [B, 1, T_base]
        T_base = base_feats.shape[1]
        vad_aligned = F.interpolate(
            vad_probs, 
            size=T_base, 
            mode="linear", 
            align_corners=False
        ).transpose(1, 2) # [B, T_base, 1]
        
        # 5. Concatenate
        return torch.cat([base_feats, vad_aligned], dim=-1)

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        """Export is not supported for external JIT wrappers."""
        raise NotImplementedError(
            "SileroVadWrapper cannot be exported to ONNX via standard PyTorch tracing "
            "because it wraps an external PyTorch Hub JIT model with complex internal control flow. "
            "Export the base extractor separately, and use the official Silero ONNX model for the VAD stream."
        )


class VoiceActivityExtractor(BaseExtractor):
    """Wrapper that appends voice activity features to any base extractor.

    Adds per-frame energy-based VAD signals as extra feature channels:
    - RMS energy (log-scaled)
    - Zero-crossing rate (speech vs noise indicator)
    - Spectral flatness (tonal vs noise-like)
    - Voice activity probability (combined soft decision)

    These signals let the classifier explicitly focus on speech regions
    without learning this from scratch.

    Args:
        base_extractor: Any BaseExtractor to wrap.
        frame_len: Frame length in samples for VAD computation.
        hop_length: Hop length in samples (should match base extractor).
    """

    def __init__(
        self,
        base_extractor: BaseExtractor,
        frame_len: int = 400,
        hop_length: int = 160,
    ) -> None:
        super().__init__(sample_rate=base_extractor.sample_rate)
        self.base = base_extractor
        self.frame_len = frame_len
        self.hop_length = hop_length
        self._n_vad_feats = 4  # energy, zcr, spectral_flatness, vad_prob

    @property
    def feature_dim(self) -> int:
        return self.base.feature_dim + self._n_vad_feats

    def _compute_vad_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """Compute frame-level VAD features from raw audio.

        Args:
            wavs: ``[B, T]`` waveform tensor.

        Returns:
            ``[B, T_frames, 4]`` VAD features.
        """
        B, T = wavs.shape
        device = wavs.device

        # Frame the signal [B, n_frames, frame_len]
        n_frames = max(1, (T - self.frame_len) // self.hop_length + 1)
        indices = torch.arange(self.frame_len, device=device).unsqueeze(0) + \
                  torch.arange(n_frames, device=device).unsqueeze(1) * self.hop_length
        indices = indices.clamp(max=T - 1)
        frames = wavs[:, indices]  # [B, n_frames, frame_len]

        # 1. Log RMS energy
        rms = (frames ** 2).mean(dim=-1).clamp(min=1e-10).sqrt()
        log_energy = rms.log()
        # Normalize to [0, 1] range per utterance
        e_min = log_energy.min(dim=-1, keepdim=True).values
        e_max = log_energy.max(dim=-1, keepdim=True).values
        log_energy = (log_energy - e_min) / (e_max - e_min + 1e-8)

        # 2. Zero-crossing rate
        signs = torch.sign(frames)
        zcr = (signs[:, :, 1:] != signs[:, :, :-1]).float().mean(dim=-1)

        # 3. Spectral flatness (geometric mean / arithmetic mean of spectrum)
        windowed = frames * torch.hann_window(self.frame_len, device=device)
        spec = torch.fft.rfft(windowed, dim=-1).abs().clamp(min=1e-10)
        log_spec = spec.log()
        geo_mean = log_spec.mean(dim=-1).exp()
        arith_mean = spec.mean(dim=-1)
        spectral_flatness = (geo_mean / (arith_mean + 1e-10)).clamp(0, 1)

        # 4. Voice activity probability (soft combination)
        # High energy + low ZCR + low flatness → likely speech
        vad_prob = torch.sigmoid(
            3.0 * log_energy - 2.0 * zcr - 2.0 * spectral_flatness
        )

        return torch.stack([log_energy, zcr, spectral_flatness, vad_prob], dim=-1)

    def forward(self, wavs: WavInput, **kwargs) -> torch.Tensor:
        """Extract base features concatenated with VAD features.

        Args:
            wavs: ``[B, T]`` tensor or list of 1-D tensors.

        Returns:
            ``[B, T_frames, base_dim + 4]`` enriched features.
        """
        # Get base features
        base_feats = self.base(wavs, **kwargs)  # [B, T', F]

        # Prepare batch for VAD computation
        wav_list = ensure_wav_list(wavs)
        max_len = max(w.shape[-1] for w in wav_list)
        batch = torch.stack([
            F.pad(w, (0, max_len - w.shape[-1])) for w in wav_list
        ]).to(base_feats.device)

        # Compute VAD features
        vad_feats = self._compute_vad_features(batch)  # [B, T_vad, 4]

        # Align time dimensions (interpolate if different)
        T_base = base_feats.size(1)
        T_vad = vad_feats.size(1)
        if T_vad != T_base:
            vad_feats = F.interpolate(
                vad_feats.transpose(1, 2), size=T_base, mode="linear",
                align_corners=False
            ).transpose(1, 2)

        return torch.cat([base_feats, vad_feats], dim=-1)

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        """Export is not supported natively due to dynamic shape constraints."""
        raise NotImplementedError(
            "VoiceActivityExtractor cannot be directly exported to ONNX because "
            "it relies on `torch.arange` framing bound to dynamic sequence lengths and "
            "untraceable Python list comprehensions (`ensure_wav_list`). "
            "Export the base extractor separately."
        )


class PitchExtractor(BaseExtractor):
    """Wrapper that appends pitch (F0) features to any base extractor.

    Adds per-frame:
    - Fundamental frequency (F0) in Hz, normalized
    - Voicing probability (0 = unvoiced, 1 = voiced)
    - F0 delta (rate of pitch change)

    Pitch information helps distinguish speakers, tonal patterns,
    and speech prosody — useful for multi-word wake phrases.

    Args:
        base_extractor: Any BaseExtractor to wrap.
        f0_min: Minimum F0 in Hz (default 50).
        f0_max: Maximum F0 in Hz (default 600).
        frame_len: Frame length in samples.
        hop_length: Hop length in samples.
    """

    def __init__(
        self,
        base_extractor: BaseExtractor,
        f0_min: float = 50.0,
        f0_max: float = 600.0,
        frame_len: int = 400,
        hop_length: int = 160,
    ) -> None:
        super().__init__(sample_rate=base_extractor.sample_rate)
        self.base = base_extractor
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.frame_len = frame_len
        self.hop_length = hop_length
        self._n_pitch_feats = 3  # f0_norm, voicing_prob, f0_delta

    @property
    def feature_dim(self) -> int:
        return self.base.feature_dim + self._n_pitch_feats

    def _compute_pitch_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """Compute per-frame pitch features via autocorrelation.

        Args:
            wavs: ``[B, T]`` waveform.

        Returns:
            ``[B, T_frames, 3]`` (normalized F0, voicing probability, F0 delta).
        """
        B, T = wavs.shape
        device = wavs.device
        sr = self.sample_rate

        # Frame the signal
        n_frames = max(1, (T - self.frame_len) // self.hop_length + 1)
        indices = torch.arange(self.frame_len, device=device).unsqueeze(0) + \
                  torch.arange(n_frames, device=device).unsqueeze(1) * self.hop_length
        indices = indices.clamp(max=T - 1)
        frames = wavs[:, indices]  # [B, n_frames, frame_len]

        # Window
        win = torch.hann_window(self.frame_len, device=device)
        frames = frames * win

        # Autocorrelation via FFT
        n_fft = 2 ** int(math.ceil(math.log2(self.frame_len * 2)))
        spec = torch.fft.rfft(frames, n=n_fft, dim=-1)
        acf = torch.fft.irfft(spec * spec.conj(), n=n_fft, dim=-1)
        acf = acf[..., :self.frame_len]

        # Normalize
        acf = acf / (acf[..., :1].clamp(min=1e-10))

        # Search for peak in valid lag range
        lag_min = max(1, int(sr / self.f0_max))
        lag_max = min(self.frame_len - 1, int(sr / self.f0_min))

        if lag_max <= lag_min:
            zeros = torch.zeros(B, n_frames, 3, device=device)
            return zeros

        acf_search = acf[..., lag_min:lag_max + 1]  # [B, n_frames, lag_range]
        peak_vals, peak_indices = acf_search.max(dim=-1)

        # F0 from lag
        lags = peak_indices + lag_min
        f0 = sr / lags.float().clamp(min=1)

        # Normalize F0 to [0, 1]
        f0_norm = (f0 - self.f0_min) / (self.f0_max - self.f0_min + 1e-8)
        f0_norm = f0_norm.clamp(0, 1)

        # Voicing probability from autocorrelation peak
        voicing_prob = peak_vals.clamp(0, 1)

        # F0 delta (finite difference)
        f0_delta = torch.zeros_like(f0_norm)
        if n_frames > 1:
            f0_delta[:, 1:] = f0_norm[:, 1:] - f0_norm[:, :-1]

        return torch.stack([f0_norm, voicing_prob, f0_delta], dim=-1)

    def forward(self, wavs: WavInput, **kwargs) -> torch.Tensor:
        """Extract base features + pitch features.

        Returns:
            ``[B, T_frames, base_dim + 3]`` enriched features.
        """
        base_feats = self.base(wavs, **kwargs)

        wav_list = ensure_wav_list(wavs)
        max_len = max(w.shape[-1] for w in wav_list)
        batch = torch.stack([
            F.pad(w, (0, max_len - w.shape[-1])) for w in wav_list
        ]).to(base_feats.device)

        pitch_feats = self._compute_pitch_features(batch)

        T_base = base_feats.size(1)
        T_pitch = pitch_feats.size(1)
        if T_pitch != T_base:
            pitch_feats = F.interpolate(
                pitch_feats.transpose(1, 2), size=T_base, mode="linear",
                align_corners=False
            ).transpose(1, 2)

        return torch.cat([base_feats, pitch_feats], dim=-1)

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        """Export is not supported natively due to dynamic shape constraints and FFTs."""
        raise NotImplementedError(
            "PitchExtractor cannot be directly exported to ONNX because "
            "it relies on dynamically computed FFT lag ranges and `torch.arange` "
            "framing which bind the time dimension as a constant during tracing. "
            "Export the base extractor separately."
        )


class MultiResolutionExtractor(BaseExtractor):
    """Wrapper combining features from two extractors at different time resolutions.

    Runs two extractors (e.g. same type with different hop lengths) and
    concatenates their outputs along the feature dimension.  The coarse
    extractor captures broad temporal patterns while the fine extractor
    captures details.

    Args:
        fine_extractor: Extractor with shorter hop length (more frames).
        coarse_extractor: Extractor with longer hop length (fewer frames).
    """

    def __init__(
        self,
        fine_extractor: BaseExtractor,
        coarse_extractor: BaseExtractor,
    ) -> None:
        super().__init__(sample_rate=fine_extractor.sample_rate)
        self.fine = fine_extractor
        self.coarse = coarse_extractor

    @property
    def feature_dim(self) -> int:
        return self.fine.feature_dim + self.coarse.feature_dim

    def forward(self, wavs: WavInput, **kwargs) -> torch.Tensor:
        """Extract multi-resolution features.

        Returns:
            ``[B, T_frames, fine_dim + coarse_dim]`` concatenated features.
            Time dimension matches the fine extractor; coarse is interpolated.
        """
        fine_feats = self.fine(wavs, **kwargs)     # [B, T_fine, F1]
        coarse_feats = self.coarse(wavs, **kwargs)  # [B, T_coarse, F2]

        T_fine = fine_feats.size(1)
        T_coarse = coarse_feats.size(1)
        if T_coarse != T_fine:
            coarse_feats = F.interpolate(
                coarse_feats.transpose(1, 2), size=T_fine, mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        return torch.cat([fine_feats, coarse_feats], dim=-1)

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        """Export is not supported natively due to dynamic size constraints in F.interpolate."""
        raise NotImplementedError(
            "MultiResolutionExtractor cannot be directly exported to ONNX because "
            "it relies on `F.interpolate` with a dynamic sequence length `size`, "
            "which PyTorch's ONNX tracer evaluates as a static integer, breaking dynamic axes. "
            "Export each extractor separately."
        )


class SNRAwareExtractor(BaseExtractor):
    """Wrapper that appends per-frame SNR estimate to any base extractor.

    Estimates signal-to-noise ratio per frame using a simple noise floor
    tracker.  The SNR channel lets the classifier weight clean frames
    more heavily than noisy ones.

    Adds 2 features:
    - Estimated per-frame SNR (dB, normalized)
    - Noise floor estimate (log energy, normalized)

    Args:
        base_extractor: Any BaseExtractor to wrap.
        frame_len: Frame length in samples.
        hop_length: Hop length in samples.
        noise_percentile: Percentile for noise floor estimation (0-100).
    """

    def __init__(
        self,
        base_extractor: BaseExtractor,
        frame_len: int = 400,
        hop_length: int = 160,
        noise_percentile: float = 10.0,
    ) -> None:
        super().__init__(sample_rate=base_extractor.sample_rate)
        self.base = base_extractor
        self.frame_len = frame_len
        self.hop_length = hop_length
        self.noise_percentile = noise_percentile

    @property
    def feature_dim(self) -> int:
        return self.base.feature_dim + 2

    def _compute_snr_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """Estimate per-frame SNR.

        Args:
            wavs: ``[B, T]`` waveform.

        Returns:
            ``[B, T_frames, 2]`` (normalized SNR, normalized noise floor).
        """
        B, T = wavs.shape
        device = wavs.device

        # Frame the signal
        n_frames = max(1, (T - self.frame_len) // self.hop_length + 1)
        indices = torch.arange(self.frame_len, device=device).unsqueeze(0) + \
                  torch.arange(n_frames, device=device).unsqueeze(1) * self.hop_length
        indices = indices.clamp(max=T - 1)
        frames = wavs[:, indices]  # [B, n_frames, frame_len]

        # Frame energy (log)
        frame_energy = (frames ** 2).mean(dim=-1).clamp(min=1e-10).log()  # [B, n_frames]

        # Noise floor: rolling minimum (percentile of energy)
        # Use quantile as noise floor estimate
        k = max(1, int(self.noise_percentile / 100.0 * n_frames))
        sorted_energy, _ = frame_energy.sort(dim=-1)
        noise_floor = sorted_energy[:, min(k, n_frames - 1)].unsqueeze(-1)  # [B, 1]
        noise_floor = noise_floor.expand_as(frame_energy)

        # SNR in dB
        snr_db = frame_energy - noise_floor  # log-domain subtraction = dB difference

        # Normalize to [0, 1]
        snr_max = snr_db.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        snr_norm = (snr_db / snr_max).clamp(0, 1)

        e_min = frame_energy.min(dim=-1, keepdim=True).values
        e_max = frame_energy.max(dim=-1, keepdim=True).values
        noise_norm = (noise_floor - e_min) / (e_max - e_min + 1e-8)

        return torch.stack([snr_norm, noise_norm], dim=-1)

    def forward(self, wavs: WavInput, **kwargs) -> torch.Tensor:
        """Extract base features + SNR features.

        Returns:
            ``[B, T_frames, base_dim + 2]`` enriched features.
        """
        base_feats = self.base(wavs, **kwargs)

        wav_list = ensure_wav_list(wavs)
        max_len = max(w.shape[-1] for w in wav_list)
        batch = torch.stack([
            F.pad(w, (0, max_len - w.shape[-1])) for w in wav_list
        ]).to(base_feats.device)

        snr_feats = self._compute_snr_features(batch)

        T_base = base_feats.size(1)
        T_snr = snr_feats.size(1)
        if T_snr != T_base:
            snr_feats = F.interpolate(
                snr_feats.transpose(1, 2), size=T_base, mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        return torch.cat([base_feats, snr_feats], dim=-1)

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        """Export is not supported natively due to dynamic shape and slicing constraints."""
        raise NotImplementedError(
            "SNRAwareExtractor cannot be directly exported to ONNX because "
            "it uses `torch.arange` framing and conditional slicing based on a dynamically "
            "computed percentile that eagerly evaluate during standard tracing. "
            "Export the base separately."
        )


# ---------------------- Markov-Based Extractors ----------------------


class MarkovTransitionExtractor(BaseExtractor):
    """Feature extractor using Markov chain transition probabilities.

    Quantizes audio frames into discrete tokens via k-means, then
    uses a trained Markov chain's transition probabilities as features.
    Captures temporal dynamics without any neural network parameters.

    Workflow:
    1. Base extractor produces ``[B, T, F]`` features
    2. Each frame is quantized to nearest codebook entry (VQ)
    3. For each frame, the Markov transition probability vector is looked up
    4. Transition probs are appended as extra feature channels

    The Markov chain must be pre-trained on wake word audio via ``fit()``.
    Without training, outputs uniform probabilities (still valid features).

    Args:
        base_extractor: Any BaseExtractor for initial features.
        n_codes: Number of VQ codebook entries (discrete tokens).
        order: Markov chain order (context length).
    """

    def __init__(
        self,
        base_extractor: BaseExtractor,
        n_codes: int = 64,
        order: int = 2,
    ) -> None:
        super().__init__(sample_rate=base_extractor.sample_rate)
        self.base = base_extractor
        self.n_codes = n_codes
        self.order = order

        # Codebook for vector quantization [n_codes, F]
        self.register_buffer("_codebook", torch.zeros(n_codes, base_extractor.feature_dim))

        # Transition matrix [n_codes^order, n_codes] — uniform initially
        n_states = n_codes ** order
        self.register_buffer("_transition_matrix", torch.ones(n_states, n_codes) / n_codes)

        self._fitted = False

    @property
    def feature_dim(self) -> int:
        return self.base.feature_dim + self.n_codes

    def _quantize(self, feats: torch.Tensor) -> torch.Tensor:
        """Assign each frame to nearest codebook entry.

        Args:
            feats: ``[B, T, F]`` continuous features.

        Returns:
            ``[B, T]`` integer token IDs.
        """
        if not self._fitted:
            # No codebook yet — random assignment
            return torch.randint(0, self.n_codes, (feats.shape[0], feats.shape[1]),
                                 device=feats.device)
        # Ensure codebook is on same device as input features
        cb = self._codebook.to(feats.device)
        # [B, T, F] vs [K, F] → [B, T, K] distances
        dists = torch.cdist(feats, cb.unsqueeze(0).expand(feats.shape[0], -1, -1))
        return dists.argmin(dim=-1)  # [B, T]

    def _context_to_state(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert token sequences into state indices for all time steps.

        Initial frames (t < order) use clamped context (repeating the first token).

        Args:
            token_ids: ``[B, T]`` token sequences.

        Returns:
            ``[B, T]`` state indices.
        """
        B, T = token_ids.shape
        states = torch.zeros(B, T, dtype=torch.long, device=token_ids.device)
        for i in range(self.order):
            # Target index for each t is t - order + 1 + i
            t_indices = torch.arange(T, device=token_ids.device)
            fetch_indices = (t_indices - self.order + 1 + i).clamp(min=0)
            states = states * self.n_codes + token_ids[:, fetch_indices]
        return states

    def fit(self, audio_list: list[torch.Tensor]) -> None:
        """Train the codebook and Markov chain from audio samples.

        Ensures boundary frames are correctly accounted for in transitions.

        Args:
            audio_list: List of 1-D waveform tensors (wake word samples).
        """
        # Step 1: Extract features from all samples
        all_feats = []
        for wav in audio_list:
            with torch.no_grad():
                feats = self.base(wav.unsqueeze(0))  # [1, T, F]
            all_feats.append(feats.squeeze(0))
        all_feats_cat = torch.cat(all_feats, dim=0)  # [N, F]

        # Step 2: K-means for codebook
        self._codebook.data = self._simple_kmeans(all_feats_cat, self.n_codes)
        self._fitted = True

        # Step 3: Quantize all features
        all_tokens = []
        for feats in all_feats:
            tokens = self._quantize(feats.unsqueeze(0)).squeeze(0)
            all_tokens.append(tokens)

        # Step 4: Count transitions → build matrix
        n_states = self.n_codes ** self.order
        counts = torch.zeros(n_states, self.n_codes, device=self._codebook.device)
        for tokens in all_tokens:
            # Vectorized state extraction for this utterance
            states = self._context_to_state(tokens.unsqueeze(0)).squeeze(0)  # [T]
            
            # Transition is P(token[t+1] | state[t])
            # Valid for t = 0 to T-2
            if len(tokens) > 1:
                curr_states = states[:-1]
                next_tokens = tokens[1:]
                # Vectorized count accumulation
                counts.index_put_((curr_states, next_tokens), torch.tensor(1.0, device=counts.device), accumulate=True)

        # Normalize with smoothing
        self._transition_matrix.data = (counts + 1e-5) / (counts.sum(dim=1, keepdim=True) + 1e-5 * self.n_codes)

    @staticmethod
    def _simple_kmeans(data: torch.Tensor, k: int, n_iter: int = 20) -> torch.Tensor:
        """Simple k-means clustering.

        Args:
            data: ``[N, F]`` feature vectors.
            k: Number of clusters.
            n_iter: Number of iterations.

        Returns:
            ``[K, F]`` codebook.
        """
        # Initialize with random samples
        indices = torch.randperm(len(data))[:k]
        centroids = data[indices].clone()

        for _ in range(n_iter):
            dists = torch.cdist(data, centroids)
            assignments = dists.argmin(dim=1)
            for j in range(k):
                mask = assignments == j
                if mask.any():
                    centroids[j] = data[mask].mean(dim=0)

        return centroids

    def forward(self, wavs: WavInput, **kwargs) -> torch.Tensor:
        """Extract base features + Markov transition probabilities.

        Args:
            wavs: Audio input.

        Returns:
            ``[B, T, base_dim + n_codes]`` enriched features.
        """
        base_feats = self.base(wavs, **kwargs)  # [B, T, F]
        token_ids = self._quantize(base_feats)  # [B, T]

        # Vectorized state computation
        states = self._context_to_state(token_ids)

        # Look up transition probabilities for each frame
        trans_feats = self._transition_matrix[states]  # [B, T, n_codes]

        return torch.cat([base_feats, trans_feats], dim=-1)

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        """Export the Markov extractor as a standalone ONNX graph."""
        # Export full pipeline (Base + Markov)
        self.eval()
        dummy_wav = torch.zeros(1, self.sample_rate, device=self.device)
        
        torch.onnx.export(
            self,
            (dummy_wav,),
            out,
            input_names=["input_values"],
            output_names=["features"],
            dynamic_axes={
                "input_values": {0: "batch_size", 1: "time"},
                "features": {0: "batch_size", 1: "time"}
            },
            opset_version=18
        )
        
        if metadata:
            embed_onnx_metadata(out, metadata)
        logger.info("Exported MarkovTransitionExtractor to %s", out)


class HMMStateExtractor(BaseExtractor):
    """Feature extractor using Hidden Markov Model state posteriors.

    Trains an HMM on quantized audio frame sequences, then extracts
    per-frame state posterior probabilities as features. Models
    temporal phoneme-like patterns without neural networks.

    The HMM forward algorithm provides ``P(state | observations_so_far)``
    for each frame — a rich temporal feature capturing sequential structure.

    Args:
        base_extractor: Any BaseExtractor for initial features.
        n_states: Number of HMM hidden states.
        n_codes: Number of VQ observation tokens.
    """

    def __init__(
        self,
        base_extractor: BaseExtractor,
        n_states: int = 8,
        n_codes: int = 32,
    ) -> None:
        super().__init__(sample_rate=base_extractor.sample_rate)
        self.base = base_extractor
        self.n_states = n_states
        self.n_codes = n_codes

        self.register_buffer("_codebook", torch.zeros(n_codes, base_extractor.feature_dim))

        # HMM parameters (uniform initialization)
        self.register_buffer("_pi", torch.ones(n_states) / n_states)
        self.register_buffer("_A", torch.ones(n_states, n_states) / n_states)
        self.register_buffer("_B", torch.ones(n_states, n_codes) / n_codes)
        self._fitted = False

    @property
    def feature_dim(self) -> int:
        return self.base.feature_dim + self.n_states

    def fit(self, audio_list: list[torch.Tensor], n_iter: int = 10) -> None:
        """Train codebook + HMM via Baum-Welch (unsupervised).

        Args:
            audio_list: List of 1-D waveform tensors.
            n_iter: EM iterations.
        """
        try:
            from markovonnx import HiddenMarkovModel, Vocabulary
        except ImportError:
            raise ImportError("markovonnx is required for HMMStateExtractor.fit(). "
                              "Install with: pip install markovonnx")

        # Extract and quantize features
        all_feats = []
        for wav in audio_list:
            with torch.no_grad():
                feats = self.base(wav.unsqueeze(0)).squeeze(0)
            all_feats.append(feats)

        all_feats_cat = torch.cat(all_feats, dim=0)
        self._codebook.data = MarkovTransitionExtractor._simple_kmeans(all_feats_cat, self.n_codes)

        # Quantize to token sequences
        obs_sequences = []
        for feats in all_feats:
            dists = torch.cdist(feats.unsqueeze(0), self._codebook.unsqueeze(0))
            tokens = dists.squeeze(0).argmin(dim=-1).tolist()
            obs_sequences.append([str(t) for t in tokens])

        # Train HMM
        obs_vocab = Vocabulary()
        obs_vocab.build_from_sequences(obs_sequences)
        state_vocab = Vocabulary()
        state_vocab.build_from_sequences([[str(i) for i in range(self.n_states)]])

        hmm = HiddenMarkovModel(
            n_states=self.n_states,
            obs_vocab=obs_vocab,
            state_vocab=state_vocab,
        )
        hmm.fit_unsupervised(obs_sequences, n_iter=n_iter)

        # Extract parameters as tensors
        self._pi.data = torch.tensor(hmm.pi, dtype=torch.float32)
        self._A.data = torch.tensor(hmm.A, dtype=torch.float32)
        
        # Slice emission matrix: skip column 0 (markovonnx.Vocabulary.UNK)
        # and ensure it matches n_codes exactly.
        B_mat = hmm.B[:, 1:1+self.n_codes]
        self._B.data = torch.tensor(B_mat, dtype=torch.float32)
        self._fitted = True

    def forward(self, wavs: WavInput, **kwargs) -> torch.Tensor:
        """Extract base features + HMM state posteriors.

        Returns:
            ``[B, T, base_dim + n_states]`` enriched features.
        """
        base_feats = self.base(wavs, **kwargs)
        B_size, T, F = base_feats.shape

        # Quantize
        if self._fitted:
            # Ensure codebook is on same device as input features
            cb = self._codebook.to(base_feats.device)
            dists = torch.cdist(base_feats, cb.unsqueeze(0).expand(B_size, -1, -1))
            token_ids = dists.argmin(dim=-1)
        else:
            token_ids = torch.randint(0, self.n_codes, (B_size, T), device=base_feats.device)

        # Forward algorithm
        device = base_feats.device
        pi = self._pi.to(device)
        A = self._A.to(device)
        B = self._B.to(device)

        hmm_feats = torch.zeros(B_size, T, self.n_states, device=device)

        # Initial
        alpha = pi * B[:, token_ids[:, 0].clamp(0, self.n_codes - 1)].T
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-10)
        hmm_feats[:, 0] = alpha

        # Forward pass (time loop remains, but batch is vectorized)
        for t in range(1, T):
            obs = token_ids[:, t].clamp(0, self.n_codes - 1)
            # (alpha @ A) * B[:, obs]
            # alpha: [B, S], A: [S, S] -> [B, S]
            # B[:, obs]: [S, B] -> [B, S]
            alpha = torch.matmul(alpha, A) * B[:, obs].T
            alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-10)
            hmm_feats[:, t] = alpha

        return torch.cat([base_feats, hmm_feats], dim=-1)

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False, metadata: dict = None) -> None:
        """Export the HMM extractor as a standalone ONNX graph."""
        # Export full pipeline (Base + HMM)
        self.eval()
        dummy_wav = torch.zeros(1, self.sample_rate, device=self.device)
        
        torch.onnx.export(
            self,
            (dummy_wav,),
            out,
            input_names=["input_values"],
            output_names=["features"],
            dynamic_axes={
                "input_values": {0: "batch_size", 1: "time"},
                "features": {0: "batch_size", 1: "time"}
            },
            opset_version=18
        )
        if metadata:
            embed_onnx_metadata(out, metadata)
        logger.info("Exported HMMStateExtractor to %s", out)
