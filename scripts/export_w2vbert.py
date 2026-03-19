import torch
from transformers import Wav2Vec2BertModel, AutoFeatureExtractor, Wav2Vec2Model

import torch.nn.functional as F
from ww_trainer.feats import BaseExtractor


import torch
from transformers import AutoFeatureExtractor, Wav2Vec2Model
from ww_trainer.feats import BaseExtractor, ensure_wav_list, WavInput


class Wav2Vec2FeatureExtractor(BaseExtractor):
    """
    Wrapper for facebook/wav2vec2-base.
    Takes raw waveform [B, T] and returns hidden states [B, T', 768].
    """

    def __init__(self, model_name="patrickvonplaten/tiny-wav2vec2-no-tokenizer", sample_rate: int = 16000, device="auto"):
        super().__init__(sample_rate=sample_rate, device=device)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def forward(self, wavs: torch.Tensor):
        # Ensure proper shape [B, T]
        if wavs.ndim == 1:
            wavs = wavs.unsqueeze(0)
        wavs = wavs.to(self.device)

        # The processor is not ONNX-safe, so avoid it during export
        # Instead, just normalize manually (like the processor does)
        wavs = (wavs - wavs.mean(dim=-1, keepdim=True)) / (wavs.std(dim=-1, keepdim=True) + 1e-5)

        # Forward through pretrained model
        with torch.no_grad():
            outputs = self.model(wavs)
        return outputs.last_hidden_state  # [B, T', 768]


class W2VBertFeatureExtractor(BaseExtractor):
    """
    Wrapper for facebook/w2v-bert-2.0.
    Takes raw waveform [B, T] and returns hidden states [B, T', 1024].
    """

    def __init__(self, model_name="facebook/wav2vec2-base", sample_rate: int = 16000, device="auto"):
        super().__init__(sample_rate=sample_rate, device=device)
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2BertModel.from_pretrained(model_name)
        self.model.eval()
        self.to(self.device)

    def forward(self, wavs: torch.Tensor):
        # ensure correct shape and type
        if wavs.ndim == 1:
            wavs = wavs.unsqueeze(0)
        inputs = self.processor(wavs, sampling_rate=self.sample_rate, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state  # [B, T, C]



from transformers import HubertModel
class HuBERTExtractor(BaseExtractor):
    """ loads HuBERT encoder and provides preprocess => last_hidden_state ([B, T, 768])."""

    def __init__(self, hubert_model="voidful/hubert-tiny-v2", sample_rate: int = 16000, device="auto") -> None:
        super().__init__(sample_rate, device)
        self.hubert = HubertModel.from_pretrained(hubert_model).eval().to(self.device)

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



if __name__ == "__main__":

    # ybelkada/hubert-tiny-random
    HuBERTExtractor(device="cpu").export_to_onnx("hubert-tiny-v2.onnx", quantize=True, dynamo=False)
