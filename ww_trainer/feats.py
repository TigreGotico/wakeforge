import abc
import math
from pathlib import Path
from typing import List, Union, TypeAlias
from ww_trainer.utils import timed
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnxruntime.quantization import quantize_dynamic, QuantType

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
        self.current_len = 0  # number of valid frames in the buffer

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

    @property
    def feature_dim(self) -> int:
        raise NotImplementedError("Subclasses must implement feature_dim")

    @abc.abstractmethod
    def forward(self, wavs: WavInput, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, wavs: WavInput, **kwargs) -> torch.Tensor:
        return self.forward(wavs, **kwargs)

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo=False):
        import onnx
        dummy_wav = torch.zeros(1, self.sample_rate, device=self.device)
        dynamic_axes = {"input_values": {0: "batch_size", 1: "time"}, "features": {0: "batch_size", 1: "time"}
                        # optional, if output has same batch/time dims
                        }

        torch.onnx.export(self, dummy_wav, out, input_names=["input_values"], output_names=["features"],
                          dynamic_axes=dynamic_axes,
                          opset_version=18,
                          do_constant_folding=True,
                          dynamo=dynamo,
                          verbose=False,
                          external_data=False,
                          training=torch.onnx.TrainingMode.EVAL)
        onnx_model = onnx.load(out)
        onnx.checker.check_model(onnx_model)
        print(f"✅ Exported ONNX model to {out}")

        if quantize:
            out_int8 = str(Path(out).with_stem(Path(out).stem + "_int16"))
            quantize_dynamic(out, out_int8, op_types_to_quantize=["MatMul", "Gemm"], weight_type=QuantType.QInt16)
            print(f"✅ Quantized ONNX model saved to {out_int8}")
            out_int8 = str(Path(out).with_stem(Path(out).stem + "_int8"))
            quantize_dynamic(out, out_int8, op_types_to_quantize=["MatMul", "Gemm"], weight_type=QuantType.QInt8)
            print(f"✅ Quantized ONNX model saved to {out_int8}")


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

        device = batch.device
        window = torch.hann_window(self.n_fft, device=device)

        # Compute STFT manually (return_complex=False for ONNX)
        stft = torch.stft(
            batch,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=False
        )
        real = stft[..., 0]
        imag = stft[..., 1]
        power = (real.pow(2) + imag.pow(2))

        mel_spec = torch.matmul(self.mel_fb.to(device), power)
        log_mel = torch.log(mel_spec + 1e-10)
        mfcc = torch.matmul(self.dct_mat.to(device), log_mel)

        # Transpose to [B, T, n_mfcc]
        return mfcc.transpose(1, 2)


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

        stft = torch.stft(
            batch,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=False,
        )
        real = stft[..., 0]
        imag = stft[..., 1]
        power = real.pow(2) + imag.pow(2)

        mel_spec = torch.matmul(self.mel_fb.to(device), power)
        log_mel = torch.log(mel_spec + 1e-10)

        return log_mel.transpose(1, 2)  # [B, T_frames, n_mels]


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

    def export_to_onnx(self, out: str, quantize: bool = False, dynamo: bool = False):
        """Export the full DeltaExtractor (base + delta computation) to ONNX."""
        import onnx
        dummy_wav = torch.zeros(1, self.sample_rate, device=self.device)
        dynamic_axes = {
            "input_values": {0: "batch_size", 1: "time"},
            "features": {0: "batch_size", 1: "time"}
        }
        torch.onnx.export(
            self, dummy_wav, out,
            input_names=["input_values"],
            output_names=["features"],
            dynamic_axes=dynamic_axes,
            opset_version=18,
            do_constant_folding=True,
            dynamo=dynamo,
            verbose=False,
            external_data=False,
            training=torch.onnx.TrainingMode.EVAL,
        )
        onnx_model = onnx.load(out)
        onnx.checker.check_model(onnx_model)
        print(f"Exported DeltaExtractor ONNX to {out}")
        if quantize:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            from pathlib import Path
            out_int8 = str(Path(out).with_stem(Path(out).stem + "_int8"))
            quantize_dynamic(out, out_int8, weight_type=QuantType.QInt8)


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
