import abc
import math
from pathlib import Path
from typing import List, Union, TypeAlias

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
            self.feature_cache[:-shift] = self.feature_cache[shift:self.current_len]
            self.current_len -= shift

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
                          dynamic_axes=dynamic_axes, opset_version=18, do_constant_folding=True, dynamo=dynamo,
                          verbose=False,

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


class MfccExtractor(BaseExtractor):  # onnx exportable
    def __init__(self, sample_rate=16000, n_mfcc=13, n_mels=40, n_fft=400, hop_length=160, f_min=0.0, f_max=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2

        mel_fb = self._mel_filterbank()
        self.register_buffer("mel_fb", mel_fb)
        dct_mat = self._dct_matrix(n_mfcc, n_mels)
        self.register_buffer("dct_mat", dct_mat)

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

    def _dct_matrix_ortho(self, n_mfcc, n_mels):
        # exact match to torchaudio/librosa "ortho" DCT-II
        n = torch.arange(n_mels).float()
        k = torch.arange(n_mfcc).float().unsqueeze(1)
        dct = torch.cos(math.pi / n_mels * (n + 0.5) * k)
        dct *= math.sqrt(2.0 / n_mels)
        dct[0] /= math.sqrt(2.0)
        return dct

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

        return mfcc


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
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

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


class CnnBlock(torch.nn.Module):
    """A CNN block with residual connection, Batch Norm, and GELU activation."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.GELU()

        # 1x1 convolution for residual connection if dimensions change
        if in_channels != out_channels or stride != 1:
            self.residual = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm1d(out_channels))
        else:
            self.residual = torch.nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class EndToEndCnnGruExtractor(BaseExtractor):
    """
    High-Performance Extractor: CNN stack (10ms frame rate)
    followed by a Bidirectional GRU for temporal modeling, outputting context-aware embeddings.
    """

    def __init__(self, output_feature_dim: int = 256,  # Final embedding dimension
                 cnn_base_dim: int = 64, gru_hidden_dim: int = 128, gru_num_layers: int = 2,
                 device: str = "auto") -> None:
        super().__init__()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.output_feature_dim = output_feature_dim
        self.gru_hidden_dim = gru_hidden_dim

        # --- 1. CNN Downsampling Stack (Total Stride = 160 for 10ms frame rate) ---
        self.cnn_stack = torch.nn.Sequential(  # Layer 1: Aggressive initial downsampling (16kHz -> 1.6kHz, Stride 10)
            CnnBlock(1, cnn_base_dim, kernel_size=10, stride=10, padding=0),
            # Layer 2: Downsampling (1.6kHz -> 400Hz, Stride 4)
            CnnBlock(cnn_base_dim, cnn_base_dim * 2, kernel_size=4, stride=4, padding=0),
            # Layer 3: Feature Refinement (400Hz -> 200Hz, Stride 2)
            CnnBlock(cnn_base_dim * 2, cnn_base_dim * 4, kernel_size=3, stride=2, padding=1),
            # Layer 4: Final Refinement (200Hz -> 100Hz, Stride 2)
            CnnBlock(cnn_base_dim * 4, cnn_base_dim * 4, kernel_size=3, stride=2, padding=1), )

        cnn_out_channels = cnn_base_dim * 4

        # --- 2. Temporal Modeling (GRU) ---
        self.gru = torch.nn.GRU(input_size=cnn_out_channels, hidden_size=gru_hidden_dim, num_layers=gru_num_layers,
                                bidirectional=True,  # Bidirectional GRU captures context from both directions
                                batch_first=True)

        # Output of Bidirectional GRU is hidden_dim * 2
        gru_out_channels = gru_hidden_dim * 2

        # --- 3. Final Projection ---
        self.projection = torch.nn.Linear(gru_out_channels, output_feature_dim)

        self.to(self.device)

    def forward(self, wavs: WavInput) -> torch.Tensor:
        """Output: sequence of context-aware embeddings [B, T_feats, F_out]"""
        wavs = ensure_wav_list(wavs)
        batch = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True).to(self.device)

        # 1. CNN Stack (Input: [B, 1, T] -> Output: [B, C, T_feats])
        cnn_input = batch.unsqueeze(1)
        cnn_output = self.cnn_stack(cnn_input)

        # Transpose to GRU input format [B, T_feats, C]
        gru_input = cnn_output.transpose(1, 2)

        # 2. Bidirectional GRU (Output: [B, T_feats, hidden_dim * 2])
        gru_output, _ = self.gru(gru_input)

        # 3. Final Projection (Output: [B, T_feats, F_out])
        final_feats = self.projection(gru_output)

        return final_feats


if __name__ == "__main__":
    import torch
    import torchaudio
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt

    # ---- import your ONNX-safe MFCC ----

    SAMPLE_RATE = 16000
    N_MFCC = 13
    N_MELS = 40
    N_FFT = 400
    HOP_LENGTH = 160

    torch.manual_seed(0)
    waveform = torch.randn(1, SAMPLE_RATE)

    # --- 1️⃣ torchaudio reference ---
    mfcc_ta = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_fft": N_FFT,
            "n_mels": N_MELS,
            "hop_length": HOP_LENGTH,
            "mel_scale": "htk",
        },
    )(waveform)
    mfcc_ta = mfcc_ta.squeeze().numpy()

    # --- 2️⃣ your ONNX-friendly version ---
    model = MfccExtractor(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    model.eval()
    with torch.no_grad():
        mfcc_onnx = model(waveform).squeeze().numpy()

    # --- 3️⃣ librosa reference ---
    y = waveform.squeeze().numpy()
    mfcc_librosa = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        htk=True,
    )


    # librosa returns (n_mfcc, frames) like torchaudio

    # --- numeric comparisons ---
    def compare(name, a, b):
        diff = np.abs(a - b)
        mean_diff = diff.mean()
        max_diff = diff.max()
        corr = np.corrcoef(a.flatten(), b.flatten())[0, 1]
        print(f"{name} → mean={mean_diff:.6f}, max={max_diff:.6f}, corr={corr:.6f}")


    print("🔍 Numeric comparisons")
    compare("ONNX vs torchaudio", mfcc_onnx, mfcc_ta)
    compare("ONNX vs librosa", mfcc_onnx, mfcc_librosa)
    compare("torchaudio vs librosa", mfcc_ta, mfcc_librosa)

    # --- visualization ---
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(mfcc_ta, origin="lower", aspect="auto")
    plt.title("torchaudio MFCC")

    plt.subplot(1, 3, 2)
    plt.imshow(mfcc_onnx, origin="lower", aspect="auto")
    plt.title("ONNX-safe MFCC")

    plt.subplot(1, 3, 3)
    plt.imshow(mfcc_librosa, origin="lower", aspect="auto")
    plt.title("librosa MFCC")

    plt.tight_layout()
    plt.show()

    plt.show()
