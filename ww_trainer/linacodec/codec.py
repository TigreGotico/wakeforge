"""LinaCodec — vendored from https://github.com/ysharma3501/LinaCodec
Patched to accept an explicit *device* argument instead of hardcoding .cuda().
"""
import torch
from .vocoder.vocos import Vocos  # noqa: E402 — vendored copy
from huggingface_hub import snapshot_download
from .model import LinaCodecModel
from .util import load_audio, load_vocoder, vocode


class LinaCodec:
    def __init__(self, model_path=None, device: str = "auto"):
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        if model_path is None:
            model_path = snapshot_download("YatharthS/LinaCodec")

        model = LinaCodecModel.from_pretrained(
            config_path=f"{model_path}/config.yaml",
            weights_path=f"{model_path}/model.safetensors",
        ).eval().to(self._device)

        model.load_distilled_wavlm(f"{model_path}/wavlm_encoder.pth",
                                   device=self._device)
        model.distilled_layers = [6, 9]

        vocos = Vocos.from_hparams(f"{model_path}/vocoder/config.yaml").to(self._device)
        vocos.load_state_dict(torch.load(
            f"{model_path}/vocoder/pytorch_model.bin",
            map_location=self._device,
        ))

        self.model = model
        self.vocos = vocos

    @torch.no_grad()
    def encode(self, audio_path):
        """Encode audio → (content_token_indices, global_embedding)."""
        audio = load_audio(audio_path, sample_rate=self.model.config.sample_rate)
        audio = audio.to(self._device)
        features = self.model.encode(audio)
        return features.content_token_indices, features.global_embedding

    @torch.no_grad()
    def decode(self, content_tokens, global_embedding):
        """Decode tokens + embedding → 48 kHz waveform."""
        # autocast only makes sense on CUDA; skip on CPU to avoid errors
        if self._device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                mel = self.model.decode(
                    content_token_indices=content_tokens,
                    global_embedding=global_embedding,
                )
        else:
            mel = self.model.decode(
                content_token_indices=content_tokens,
                global_embedding=global_embedding,
            )
        waveform = vocode(self.vocos, mel.unsqueeze(0))
        return waveform

    def convert_voice(self, source_file, reference_file):
        """Voice conversion: content of *source_file* in the timbre of *reference_file*."""
        speech_tokens, _ = self.encode(source_file)
        _, ref_global_embedding = self.encode(reference_file)
        return self.decode(speech_tokens, ref_global_embedding)
