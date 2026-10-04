"""Every featurizer turns 16 kHz mono float32 audio into named *entries*.

Frame featurizers return, per entry, a tensor [B, T, d] at about 50 Hz plus the valid frame count of each row.
The downstream mixes the entries with a learned softmax-weighted sum (entries whose width differs from the
featurizer's width go through a learned linear projection first), SUPERB style.

    logmel           mel2 (128)                                     the trunk's own normalised log-mel, frames 2t,2t+1 stacked
    wakehubert       tap0..tap8 (256), out (128)                    frozen WakeHuBERT-tiny trunk
    wakephonehubert  wakehubert entries + vad (1), ipa (V)         trunk + the trained heads of --heads
    hubert-base      hs0..hs12 (768)                                facebook/hubert-base-ls960, one utterance per call (group norm)
    wav2vec2-espeak  hs0..hs24 (1024)                               facebook/wav2vec2-xlsr-53-espeak-cv-ft, attention mask
    efficientat-mn10 emb (960), clip level only                     EfficientAT mn10_as, 32 kHz, left-padded to at least 5 s
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SR = 16000
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pool"))
HEADS_DIR = Path("work/runs/round4-final-ipa")
EFFICIENTAT_DIR = Path("work/EfficientAT")

HUBERT = ("facebook/hubert-base-ls960", None)
W2V = ("facebook/wav2vec2-xlsr-53-espeak-cv-ft", "3e836924cfd3b858a0bbdbc1f7ef412105d00446")
FRAME_FEATS = ("logmel", "wakehubert", "wakephonehubert", "hubert-base", "wav2vec2-espeak")
CLIP_FEATS = FRAME_FEATS + ("efficientat-mn10",)
MASKED_SAMPLES = 20 * SR
DISPLAY = {"logmel": "log-mel", "wakehubert": "WakeHuBERT-tiny", "wakephonehubert": "WakePhoneHuBERT", "hubert-base": "HuBERT-base (teacher)",
           "wav2vec2-espeak": "wav2vec2-xlsr-53-espeak (teacher)", "efficientat-mn10": "EfficientAT mn10_as (teacher)"}


def add_args(ap):
    """The options every benchmark script shares: its working directory, the trained heads, the EfficientAT checkout."""
    ap.add_argument("--bench-dir", default="work/bench", help="data/, feats/ and results/ of the benchmark")
    ap.add_argument("--heads", default=str(HEADS_DIR), help="train_pool.py run dir with vad_head.pt and ipa_head.pt (wakephonehubert rows)")
    ap.add_argument("--efficientat", default=str(EFFICIENTAT_DIR), help="EfficientAT checkout (efficientat-mn10 rows)")


def configure(a):
    """Apply add_args' options; returns the benchmark directory."""
    global HEADS_DIR, EFFICIENTAT_DIR
    HEADS_DIR, EFFICIENTAT_DIR = Path(a.heads), Path(a.efficientat)
    return Path(a.bench_dir)


def heads_dir():
    return HEADS_DIR if all((HEADS_DIR / f).exists() for f in ("vad_head.pt", "ipa_head.pt")) else None


def _taphead(sd):
    from wph_model import TapHead
    side = 128 if any(k.startswith("side.") for k in sd) else 0
    h = TapHead(sd["o.weight"].shape[0], hidden=sd["c1.weight"].shape[0], side=side)
    h.load_state_dict(sd)
    return h


class Featurizer:
    """name -> entries. `dims` maps entry name to width; `width` is the weighted-sum width."""

    def __init__(self, name, device="cuda"):
        self.name, self.dev = name, device
        if name in ("logmel", "wakehubert", "wakephonehubert"):
            from wph_model import Trunk, load_trunk
            self.trunk = Trunk(load_trunk()).to(device).eval()
            if name == "logmel":
                self.dims = {"mel2": 128}
                self.width = 128
            else:
                self.dims = {f"tap{i}": 256 for i in range(9)} | {"out": 128}
                self.width = 256
            if name == "wakephonehubert":
                d = heads_dir()
                if d is None:
                    raise SystemExit(f"wakephonehubert: {HEADS_DIR} holds no vad_head.pt and ipa_head.pt")
                self.heads_from = str(d)
                self.vad = _taphead(torch.load(d / "vad_head.pt", map_location="cpu")).to(device).eval()
                ipa = torch.load(d / "ipa_head.pt", map_location="cpu", weights_only=False)
                self.ipa = _taphead(ipa["head"]).to(device).eval()
                self.dims |= {"vad": 1, "ipa": self.ipa.o.weight.shape[0]}
        elif name in ("hubert-base", "wav2vec2-espeak"):
            from transformers import AutoModel, Wav2Vec2ForCTC
            rid, rev = HUBERT if name == "hubert-base" else W2V
            if name == "wav2vec2-espeak":
                self.ctc = Wav2Vec2ForCTC.from_pretrained(rid, revision=rev, attn_implementation="sdpa").to(device).eval()
                self.model = self.ctc.wav2vec2
            else:
                self.model = AutoModel.from_pretrained(rid, attn_implementation="sdpa").to(device).eval()
            n = self.model.config.num_hidden_layers + 1
            w = self.model.config.hidden_size
            self.dims = {f"hs{i}": w for i in range(n)}
            self.width = w
            self.masked = self.model.config.feat_extract_norm == "layer"
        elif name == "efficientat-mn10":
            cwd = os.getcwd()
            sys.path.insert(0, str(EFFICIENTAT_DIR.resolve()))
            os.chdir(EFFICIENTAT_DIR)
            from models.mn.model import get_model
            from models.preprocess import AugmentMelSTFT
            self.model = get_model(width_mult=1.0, pretrained_name="mn10_as").to(device).eval()
            self.mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320).to(device).eval()
            os.chdir(cwd)
            self.dims = {"emb": 960}
            self.width = 960
        else:
            raise ValueError(name)
        for m in self.modules():
            for p in m.parameters():
                p.requires_grad_(False)

    def modules(self):
        return [v for v in vars(self).values() if isinstance(v, torch.nn.Module)]

    @property
    def frame_level(self):
        return self.name != "efficientat-mn10"

    @torch.no_grad()
    def frames(self, wavs):
        """wavs: list of 1-D float32 numpy arrays. Returns ({entry: [B, T, d] float32 on device}, lengths [B])."""
        if self.name in ("logmel", "wakehubert", "wakephonehubert"):
            n = max(len(w) for w in wavs)
            x = torch.zeros(len(wavs), n, device=self.dev)
            for i, w in enumerate(wavs):
                x[i, :len(w)] = torch.as_tensor(w)
            taps, out, mel2 = self.trunk(x)
            lens = torch.tensor([min(len(w) // 320, out.shape[-1]) for w in wavs])
            if self.name == "logmel":
                return {"mel2": mel2.transpose(1, 2)}, lens
            e = {f"tap{i}": taps[:, i].transpose(1, 2) for i in range(9)} | {"out": out.transpose(1, 2)}
            if self.name == "wakephonehubert":
                e["vad"] = torch.sigmoid(self.vad(taps, out, mel2))
                e["ipa"] = torch.softmax(self.ipa(taps, out, mel2), -1)
            return e, lens
        if self.name in ("hubert-base", "wav2vec2-espeak"):
            outs, lens = [], []
            if self.masked:
                groups, cur, tot = [], [], 0
                for i in sorted(range(len(wavs)), key=lambda i: len(wavs[i])):
                    if cur and (len(cur) + 1) * len(wavs[i]) > MASKED_SAMPLES:
                        groups.append(cur); cur = []
                    cur.append(i)
                groups.append(cur)
            else:
                groups = [[i] for i in range(len(wavs))]
            per = [None] * len(wavs)
            for g in groups:
                n = max(len(wavs[i]) for i in g)
                x = torch.zeros(len(g), n, device=self.dev)
                am = torch.zeros(len(g), n, dtype=torch.long, device=self.dev)
                for j, i in enumerate(g):
                    w = torch.as_tensor(wavs[i]).to(self.dev)
                    w = (w - w.mean()) / torch.sqrt(w.var() + 1e-7)
                    x[j, :len(w)] = w
                    am[j, :len(w)] = 1
                o = self.model(x, attention_mask=am if self.masked else None, output_hidden_states=True)
                hs = torch.stack(o.hidden_states, 1)
                fl = self.model._get_feat_extract_output_lengths(am.sum(1)).cpu()
                for j, i in enumerate(g):
                    per[i] = (hs[j], int(fl[j]))
            T = max(p[0].shape[1] for p in per)
            hs = torch.stack([F.pad(p[0], (0, 0, 0, T - p[0].shape[1])) for p in per])
            lens = torch.tensor([p[1] for p in per])
            return {f"hs{i}": hs[:, i] for i in range(hs.shape[1])}, lens
        raise ValueError(f"{self.name} is clip-level only")

    @torch.no_grad()
    def pooled(self, wavs):
        """Per-row mean over valid frames of every entry: {entry: [B, d] float32 numpy}."""
        if self.name == "efficientat-mn10":
            import torchaudio
            res = []
            for w in wavs:
                x = torch.as_tensor(w).to(self.dev)[None]
                if x.shape[1] < 5 * SR:
                    x = F.pad(x, (5 * SR - x.shape[1], 0))
                x = torchaudio.functional.resample(x, SR, 32000)
                logits, emb = self.model(self.mel(x).unsqueeze(1))
                res.append(emb.float()[0].cpu().numpy())
            return {"emb": np.stack(res)}
        e, lens = self.frames(wavs)
        T = next(iter(e.values())).shape[1]
        m = (torch.arange(T)[None] < lens[:, None]).float().to(self.dev)[..., None]
        return {k: ((v * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy() for k, v in e.items()}


class WeightedSum(torch.nn.Module):
    """SUPERB featurizer mix: per-entry standardisation (fixed, from training statistics), a learned linear for entries
    narrower or wider than `width`, then softmax weights. Works on [B, T, d] or pooled [B, d] entries alike."""

    def __init__(self, dims, width, stats):
        super().__init__()
        self.names = list(dims)
        self.w = torch.nn.Parameter(torch.zeros(len(self.names)))
        self.proj = torch.nn.ModuleDict({k.replace(".", "_"): torch.nn.Linear(d, width) for k, d in dims.items() if d != width})
        for k in self.names:
            mu, sd = stats[k]
            self.register_buffer(f"mu_{k}", torch.as_tensor(mu, dtype=torch.float32))
            self.register_buffer(f"sd_{k}", torch.as_tensor(sd, dtype=torch.float32).clamp(min=1e-5))

    def forward(self, e):
        w = torch.softmax(self.w, 0)
        acc = 0
        for i, k in enumerate(self.names):
            x = (e[k] - getattr(self, f"mu_{k}")) / getattr(self, f"sd_{k}")
            if k in self.proj:
                x = self.proj[k](x)
            acc = acc + w[i] * x
        return acc

    def weights(self):
        return dict(zip(self.names, [round(float(v), 4) for v in torch.softmax(self.w.detach(), 0)]))


def param_count(f):
    seen = {id(p): p.numel() for m in f.modules() for p in m.parameters()}
    return sum(seen.values())
