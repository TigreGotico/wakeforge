from ww_trainer.feats import BaseExtractor


class MfccExtractor(BaseExtractor):  # onnx exportable
    """not meant to be used directly, has been exported to onnx previously
    https://huggingface.co/TigreGotico/mfcc-onnx
    """
    def __init__(self, sr=16000, mfcc=40, mels=40, fft=400, hop=160, f_min=0.0, f_max=None):
        super().__init__()
        sample_rate = sr
        self.sample_rate = sample_rate
        self.n_mfcc = mfcc
        self.n_mels = mels
        self.n_fft = fft
        self.hop_length = hop
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2

        mel_fb = self._mel_filterbank()
        self.register_buffer("mel_fb", mel_fb)
        dct_mat = self._dct_matrix(mfcc,mels)
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


if __name__ == "__main__":
    import torch
    import torchaudio
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt

    print(OnnxFeatureExtractor("/home/miro/PycharmProjects/wakeHuBert/wakehubert/distillhubert_int8.onnx", device="cpu").feature_dim)
    combos = [
        dict(sr=16000, mfcc=13, mels=40, fft=400, hop=160),
        dict(sr=16000, mfcc=20, mels=64, fft=400, hop=160),
        dict(sr=16000, mfcc=30, mels=64, fft=400, hop=160),
        dict(sr=16000, mfcc=40, mels=40, fft=400, hop=160),
        dict(sr=16000, mfcc=40, mels=64, fft=400, hop=160),
        dict(sr=16000, mfcc=40, mels=80, fft=512, hop=160),
    ]
    for params in combos:
        slug = "_".join([f"{k}{v}" for k, v in params.items()])
        MfccExtractor(**params).export_to_onnx(f"mfcc_{slug}.onnx", quantize=True)
        model = OnnxFeatureExtractor(f"mfcc_{slug}.onnx", device="cpu")


    # ---- import your ONNX-safe MFCC ----

    SAMPLE_RATE = 16000
    N_MFCC = 40
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

    # --- 2️⃣ ONNX version ---
    model = OnnxFeatureExtractor("/mfcc_sr16000_mfcc40_mels40_fft400_hop160_int8.onnx", device="cpu")
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
