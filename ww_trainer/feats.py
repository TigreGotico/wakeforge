import abc
from pathlib import Path
from typing import List, Union, TypeAlias

import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchaudio
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
            dynamic_axes=dynamic_axes, opset_version=18, do_constant_folding=True, dynamo=dynamo, verbose=False,

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


class MfccExtractor(BaseExtractor):
    def __init__(self, feature_type: str = "mfcc", n_mels: int = 64, n_mfcc: int = 40, n_fft: int = 400,
                 hop_length: int = 160, sample_rate: int = 16000, device: str = "auto"):
        super().__init__(sample_rate, device)
        self.feature_type = feature_type.lower()
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

        if self.feature_type == "mel":
            self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                                            n_mels=n_mels).to(self.device)
            self.db = torchaudio.transforms.AmplitudeToDB(stype="power").to(self.device)
        elif self.feature_type == "mfcc":
            self.mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc,
                                                   melkwargs={"n_fft": n_fft, "hop_length": hop_length,
                                                              "n_mels": n_mels}).to(self.device)
        else:
            raise ValueError("feature_type must be 'mel' or 'mfcc'")

    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        # wavs: [B, T]
        # wavs = wavs.to(self.device)
        specs = []
        for w in wavs:
            if self.feature_type == "mel":
                m_db = self.db(self.mel(w))
                specs.append(m_db.unsqueeze(0))  # [1, F, T]
            else:
                mf = self.mfcc(w)
                specs.append(mf.unsqueeze(0))  # [1, F, T]
        return torch.stack(specs).unsqueeze(1)  # [B, 1, F, T]

    @staticmethod
    def spec_to_time_first(specs: torch.Tensor) -> torch.Tensor:
        """Convert [B, 1, F, T] -> [B, T, F] used by RNNs (squeeze channel)."""
        return specs.squeeze(1).transpose(1, 2)  # [B, T, F]


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
        self.cnn_stack = torch.nn.Sequential(# Layer 1: Aggressive initial downsampling (16kHz -> 1.6kHz, Stride 10)
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
