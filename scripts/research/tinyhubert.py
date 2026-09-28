#!/usr/bin/env python3
"""TinyHuBERT: distil HuBERT into a small causal CNN+GRU feature extractor for wake-word models.

The student reads raw 16 kHz audio and emits one frame every 20 ms (a total stride of 320
samples, HuBERT's own frame rate), strictly causally: frame t depends only on samples before
the end of its own 20 ms block. A shared backbone (causal CNN, unidirectional GRU, LayerNorm)
feeds one linear head per distilled teacher layer, as in DistilHuBERT; the heads exist only for
training, and the exported extractor is the backbone.

HuBERT's frames are contextual in both directions, which a causal student cannot reproduce
exactly. The student therefore predicts each teacher frame ``--lag-frames`` frames late: the
student frame paired with teacher frame t has heard the teacher frame's whole convolutional
window plus ``lag`` further frames of audio. A wake-word decision is not harmed by 100 ms of
latency, and the lag makes the target learnable instead of partly unpredictable.

Targets are standardised per layer and per dimension with statistics measured on the teacher
before training, since a few HuBERT dimensions carry very large magnitudes and would otherwise
dominate any distance. The loss per layer is DistilHuBERT's: L1 distance plus
``-log sigmoid(cosine)``. An optional contrastive term draws its negatives from other clips
only, because neighbouring frames of one clip are near duplicates.

With ``--noise-dir`` the student hears the clip mixed with noise at a random SNR while the
teacher hears it clean, so the extractor learns to report the speech and not the noise.

    tinyhubert.py --audio-dir LibriSpeech/train-clean-100 --val-dir LibriSpeech/dev-clean \\
        --noise-dir noise/ --out-dir runs/tinyhubert --steps 60000

Outputs in ``--out-dir``: ``metrics.jsonl`` (training and validation losses, per-layer cosine
on held-out audio), ``last.pt`` and ``best.pt`` checkpoints (resumable with ``--resume``),
``tinyhubert.onnx`` (input ``waveform`` [batch, samples], output ``features`` [batch, frames,
dim]; loads as ``ww_trainer.feats.OnnxFeatureExtractor``) and ``tinyhubert.json`` describing it.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SR = 16000
HOP = 320
AUDIO_EXT = {".wav", ".flac", ".ogg"}


# ------------------------------------------------------------------ student
class CausalConv1d(nn.Module):
    """A strided convolution whose output frame i sees inputs up to the end of block i only.

    Left padding of ``kernel - stride`` makes frame i cover inputs ``[i*s - (k - s), (i + 1)*s)``,
    its own block and the past, and the output length ``floor(n / stride)``: a trailing partial
    block is not emitted until it is complete.
    """

    def __init__(self, in_ch, out_ch, kernel, stride=1, bias=False):
        super().__init__()
        if kernel < stride:
            raise ValueError(f"kernel {kernel} < stride {stride} would skip samples")
        self.pad = kernel - stride
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class CausalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel, stride)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, 3, 1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.res = CausalConv1d(in_ch, out_ch, stride, stride) if (in_ch != out_ch or stride != 1) else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        return self.act(self.bn2(self.conv2(y)) + self.res(x))


STRIDES = (5, 4, 4, 2, 2)
KERNELS = (10, 8, 8, 4, 4)
assert math.prod(STRIDES) == HOP


class WakeHuBERTStudent(nn.Module):
    """Causal CNN + GRU backbone with one training head per distilled teacher layer.

    ``forward(wav)`` takes ``[B, N]`` audio and returns the backbone features ``[B, floor(N/320), gru_hidden]``;
    ``heads(features)`` returns the per-layer predictions used by the distillation loss.
    """

    def __init__(self, cnn_dim=64, gru_hidden=256, gru_layers=1, n_targets=3, target_dim=768):
        super().__init__()
        chans = (cnn_dim, cnn_dim * 2, cnn_dim * 4, cnn_dim * 4, cnn_dim * 4)
        blocks, c_in = [], 1
        for c, k, s in zip(chans, KERNELS, STRIDES):
            blocks.append(CausalBlock(c_in, c, k, s))
            c_in = c
        self.cnn = nn.Sequential(*blocks)
        self.gru = nn.GRU(c_in, gru_hidden, num_layers=gru_layers, batch_first=True)
        self.ln = nn.LayerNorm(gru_hidden)
        self.proj = nn.ModuleList(nn.Linear(gru_hidden, target_dim) for _ in range(n_targets))
        self.feature_dim = gru_hidden

    def forward(self, wav):
        x = self.cnn(wav.unsqueeze(1)).transpose(1, 2)
        x, _ = self.gru(x)
        return self.ln(x)

    def heads(self, feats):
        return [p(feats) for p in self.proj]


class Extractor(nn.Module):
    """The exported part: the backbone alone."""

    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, waveform):
        return self.student(waveform)


# ------------------------------------------------------------------ teacher
class Teacher:
    """Frozen HuBERT returning the chosen hidden layers as ``[L, B, T, D]`` in float32.

    ``dtype`` float16 runs the forward pass under autocast; on hubert-base-ls960 its hidden states
    match float32 to a cosine of 0.999999 at a quarter of the time, so the targets are the same.
    """

    def __init__(self, name, layers, device, dtype=torch.float32):
        from transformers import AutoConfig, HubertModel

        cfg = AutoConfig.from_pretrained(name)
        if max(layers) > cfg.num_hidden_layers:
            raise ValueError(f"{name} has {cfg.num_hidden_layers} layers; asked for {layers}")
        self.model = HubertModel.from_pretrained(name).to(device).eval().requires_grad_(False)
        self.layers = layers
        self.dim = cfg.hidden_size
        self.dtype = dtype
        self.device_type = torch.device(device).type
        # hubert-base-ls960's feature extractor does not normalise; a checkpoint that does is
        # normalised here the same way (zero mean, unit variance per clip)
        from transformers import AutoFeatureExtractor
        self.normalize = bool(getattr(AutoFeatureExtractor.from_pretrained(name), "do_normalize", False))

    @torch.no_grad()
    def __call__(self, wav):
        if self.normalize:
            wav = (wav - wav.mean(1, keepdim=True)) / (wav.std(1, keepdim=True) + 1e-7)
        with torch.autocast(self.device_type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            hs = self.model(wav, output_hidden_states=True).hidden_states
        return torch.stack([hs[i] for i in self.layers]).float()


def pair_frames(student_T, teacher_T, lag):
    """Teacher frame t is paired with student frame t + 1 + lag.

    HuBERT frame t is computed from samples [320t, 320t + 400); student frame t + 1 has heard up
    to 320(t + 2) > 320t + 400, so with lag 0 the student has heard the teacher frame's whole
    window, and each further lag frame adds 20 ms of context. Returns the number of pairs.
    """
    return max(0, min(teacher_T, student_T - 1 - lag))


# ------------------------------------------------------------------ loss
class TargetNorm(nn.Module):
    """Per-layer, per-dimension standardisation of teacher targets, measured once."""

    def __init__(self, n_layers, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_layers, 1, 1, dim))
        self.register_buffer("std", torch.ones(n_layers, 1, 1, dim))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, batches):
        s = sq = n = 0
        for t in batches:  # [L, B, T, D]
            flat = t.flatten(1, 2).double()
            s = s + flat.sum(1)
            sq = sq + (flat ** 2).sum(1)
            n += flat.shape[1]
        mean = s / n
        std = (sq / n - mean ** 2).clamp_min(1e-8).sqrt()
        self.mean.copy_(mean[:, None, None].float())
        self.std.copy_(std[:, None, None].float())
        self.fitted.fill_(True)

    def forward(self, t):
        return (t - self.mean) / self.std


def distil_loss(preds, targets):
    """DistilHuBERT's per-layer objective: L1 plus -log sigmoid(cosine), averaged over layers.

    preds, targets: ``[L, B, T, D]`` (targets standardised). Returns (loss, per-layer l1, per-layer cosine).
    """
    l1 = (preds - targets).abs().mean(dim=(1, 2, 3))
    cos = F.cosine_similarity(preds, targets, dim=-1)  # [L, B, T]
    loss = (l1 - F.logsigmoid(cos).mean(dim=(1, 2))).mean()
    return loss, l1.detach(), cos.mean(dim=(1, 2)).detach()


def cross_clip_nce(preds, targets, temperature=0.1, max_frames=1024):
    """InfoNCE over frames whose negatives come from other clips only.

    preds, targets: ``[B, T, D]`` for one layer. A frame's positive is the teacher frame at the
    same position; teacher frames of the same clip are excluded from its negatives, because
    neighbouring frames are near duplicates and the teacher itself does not separate them.
    """
    B, T, D = preds.shape
    if B < 2:
        return preds.new_zeros(())
    clip = torch.arange(B, device=preds.device).repeat_interleave(T)
    p, t = preds.reshape(B * T, D), targets.reshape(B * T, D)
    if B * T > max_frames:
        idx = torch.randperm(B * T, device=preds.device)[:max_frames]
        p, t, clip = p[idx], t[idx], clip[idx]
    logits = F.normalize(p, dim=-1) @ F.normalize(t, dim=-1).T / temperature
    same = clip[:, None] == clip[None, :]
    logits = logits.masked_fill(same & ~torch.eye(len(clip), dtype=torch.bool, device=p.device), float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(clip), device=p.device))


# ------------------------------------------------------------------ data
def list_audio(root):
    files = sorted(str(p) for p in Path(root).rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if not files:
        raise SystemExit(f"no audio ({', '.join(sorted(AUDIO_EXT))}) under {root}")
    return files


def load_mono(path, sr=SR):
    wav, file_sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(1))
    if file_sr != sr:
        import torchaudio
        wav = torchaudio.functional.resample(wav, file_sr, sr)
    return wav


def crop(wav, n, rng):
    if len(wav) >= n:
        s = rng.randint(0, len(wav) - n)
        return wav[s:s + n]
    return F.pad(wav, (0, n - len(wav)))


def mix_at_snr(clean, noise, snr_db):
    pc = clean.pow(2).mean()
    pn = noise.pow(2).mean().clamp_min(1e-10)
    if pc <= 1e-10:
        return clean.clone()
    return clean + noise * torch.sqrt(pc / (pn * 10 ** (snr_db / 10)))


class Clips(Dataset):
    """Random crops for training, or fixed crops (seeded per file) for validation.

    Returns ``(clean, student_input)``; the student input is the clean crop mixed with a noise
    crop at an SNR drawn from ``snr_range`` with probability ``p_noise``, else the clean crop.
    """

    def __init__(self, files, seconds, noise_files=(), p_noise=0.0, snr_range=(0.0, 20.0), fixed=False, seed=0):
        self.files, self.n = files, int(seconds * SR)
        self.noise, self.p_noise, self.snr = list(noise_files), p_noise, snr_range
        self.fixed, self.seed = fixed, seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i) if self.fixed else random
        clean = crop(load_mono(self.files[i]), self.n, rng)
        noisy = clean
        if self.noise and rng.random() < self.p_noise:
            noise = crop(load_mono(rng.choice(self.noise)), self.n, rng)
            noisy = mix_at_snr(clean, noise, rng.uniform(*self.snr))
        peak = noisy.abs().max()
        if peak > 1.0:
            noisy = noisy / peak
        return clean, noisy


def forever(loader):
    while True:
        yield from loader


# ------------------------------------------------------------------ training
def step_pairs(student, teacher, norm, clean, noisy, lag, amp=False):
    targets = norm(teacher(clean))  # [L, B, Tt, D]
    with torch.autocast(clean.device.type, dtype=torch.bfloat16, enabled=amp):
        feats = student(noisy)  # [B, Ts, H]
        n = pair_frames(feats.shape[1], targets.shape[2], lag)
        if n == 0:
            raise ValueError(f"no frame pairs: crop too short for lag {lag}")
        preds = torch.stack(student.heads(feats[:, 1 + lag:1 + lag + n]))
    return preds.float(), targets[:, :, :n]


@torch.no_grad()
def evaluate(student, teacher, norm, loader, lag, device):
    student.eval()
    tot, l1s, coss, k = 0.0, 0, 0, 0
    for clean, noisy in loader:
        preds, targets = step_pairs(student, teacher, norm, clean.to(device), noisy.to(device), lag)
        loss, l1, cos = distil_loss(preds, targets)
        tot, l1s, coss, k = tot + loss.item(), l1s + l1, coss + cos, k + 1
    student.train()
    return tot / k, (l1s / k).tolist(), (coss / k).tolist()


def export(student, out_dir, meta, device):
    import onnxruntime as ort

    path = out_dir / "tinyhubert.onnx"
    ext = Extractor(student).eval().cpu()
    dummy = torch.randn(1, SR)
    torch.onnx.export(ext, dummy, str(path), input_names=["waveform"], output_names=["features"],
                      dynamic_axes={"waveform": {0: "batch", 1: "samples"}, "features": {0: "batch", 1: "frames"}},
                      opset_version=17, dynamo=False)
    probe = torch.randn(2, int(1.37 * SR))
    with torch.no_grad():
        ref = ext(probe).numpy()
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"waveform": probe.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    if got.shape != ref.shape or diff > 1e-3:
        raise SystemExit(f"ONNX export disagrees with torch: shape {got.shape} vs {ref.shape}, max diff {diff}")
    meta = {**meta, "onnx_max_abs_diff_vs_torch": diff,
            "params_exported": sum(p.numel() for n, p in student.named_parameters() if not n.startswith("proj."))}
    (out_dir / "tinyhubert.json").write_text(json.dumps(meta, indent=1) + "\n")
    student.to(device)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--audio-dir", required=True, help="training audio, searched recursively")
    ap.add_argument("--val-dir", required=True, help="held-out audio for validation")
    ap.add_argument("--noise-dir", help="noise audio mixed into the student's input (teacher hears it clean)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--teacher", default="facebook/hubert-base-ls960")
    ap.add_argument("--layers", default="4,8,12", help="teacher hidden layers to distil (DistilHuBERT's)")
    ap.add_argument("--lag-frames", type=int, default=4, help="extra 20 ms frames of audio the student hears before predicting a teacher frame")
    ap.add_argument("--cnn-dim", type=int, default=64)
    ap.add_argument("--gru-hidden", type=int, default=256)
    ap.add_argument("--gru-layers", type=int, default=1)
    ap.add_argument("--crop-seconds", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--nce-weight", type=float, default=0.0, help="cross-clip InfoNCE on the last distilled layer")
    ap.add_argument("--p-noise", type=float, default=0.5)
    ap.add_argument("--snr-min", type=float, default=0.0)
    ap.add_argument("--snr-max", type=float, default=20.0)
    ap.add_argument("--norm-batches", type=int, default=50, help="teacher batches used to fit the target statistics")
    ap.add_argument("--val-clips", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--teacher-dtype", choices=("float32", "float16"), default=None,
                    help="default float16 on a GPU, float32 on CPU")
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast for the student's forward pass")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", help="checkpoint to continue from (last.pt)")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    layers = [int(x) for x in a.layers.split(",")]

    a.teacher_dtype = a.teacher_dtype or ("float16" if device.type == "cuda" else "float32")
    teacher = Teacher(a.teacher, layers, device, getattr(torch, a.teacher_dtype))
    student = WakeHuBERTStudent(a.cnn_dim, a.gru_hidden, a.gru_layers, len(layers), teacher.dim).to(device)
    norm = TargetNorm(len(layers), teacher.dim).to(device)

    noise = list_audio(a.noise_dir) if a.noise_dir else []
    train_ds = Clips(list_audio(a.audio_dir), a.crop_seconds, noise, a.p_noise, (a.snr_min, a.snr_max))
    val_files = list_audio(a.val_dir)
    val_files = random.Random(a.seed).sample(val_files, min(a.val_clips, len(val_files)))
    # validation hears noise too (fixed per clip), so it measures the objective the student trains on
    val_ds = Clips(val_files, a.crop_seconds, noise, a.p_noise, (a.snr_min, a.snr_max), fixed=True, seed=a.seed)
    train = forever(DataLoader(train_ds, a.batch_size, shuffle=True, num_workers=a.workers, drop_last=True,
                               persistent_workers=a.workers > 0))
    val = DataLoader(val_ds, a.batch_size, num_workers=a.workers)

    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    step, best = 0, float("inf")
    if a.resume:
        ck = torch.load(a.resume, map_location=device)
        student.load_state_dict(ck["student"]); norm.load_state_dict(ck["norm"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck["best"]
    if not bool(norm.fitted):
        norm.fit(teacher(next(train)[0].to(device)) for _ in range(a.norm_batches))

    meta = {"teacher": a.teacher, "layers": layers, "lag_frames": a.lag_frames, "sample_rate": SR, "hop": HOP,
            "frame_rate_hz": SR / HOP, "latency_ms": (1 + a.lag_frames) * HOP / SR * 1000, "feature_dim": a.gru_hidden,
            "cnn_dim": a.cnn_dim, "gru_layers": a.gru_layers, "noise": bool(noise), "args": vars(a)}
    log = open(out / "metrics.jsonl", "a")
    params = sum(p.numel() for n, p in student.named_parameters() if not n.startswith("proj."))
    print(f"student backbone {params:,} params; {len(train_ds.files)} training files, {len(val_files)} validation, "
          f"{len(noise)} noise; teacher layers {layers}, lag {a.lag_frames} frames", flush=True)

    t0, run = time.time(), []
    student.train()
    while step < a.steps:
        clean, noisy = next(train)
        preds, targets = step_pairs(student, teacher, norm, clean.to(device), noisy.to(device), a.lag_frames, a.amp)
        loss, l1, cos = distil_loss(preds, targets)
        nce = cross_clip_nce(preds[-1], targets[-1]) if a.nce_weight > 0 else preds.new_zeros(())
        total = loss + a.nce_weight * nce
        opt.zero_grad(set_to_none=True)
        total.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        opt.step(); sched.step(); step += 1
        if not torch.isfinite(total):
            raise SystemExit(f"non-finite loss at step {step}")
        run.append((total.item(), nce.item(), gnorm.item()))
        if step % 100 == 0:
            m = np.mean(run, axis=0); run = []
            rec = {"step": step, "train_loss": m[0], "train_nce": m[1], "grad_norm": m[2], "lr": sched.get_last_lr()[0],
                   "train_cos": cos.tolist(), "steps_per_s": 100 / (time.time() - t0)}
            log.write(json.dumps(rec) + "\n"); log.flush(); t0 = time.time()
            print(json.dumps(rec), flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            v, vl1, vcos = evaluate(student, teacher, norm, val, a.lag_frames, device)
            rec = {"step": step, "val_loss": v, "val_l1": vl1, "val_cos": vcos}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
            ck = {"student": student.state_dict(), "norm": norm.state_dict(), "opt": opt.state_dict(),
                  "sched": sched.state_dict(), "step": step, "best": min(best, v), "meta": meta}
            torch.save(ck, out / "last.pt")
            if v < best:
                best = v
                torch.save(ck, out / "best.pt")
    student.load_state_dict(torch.load(out / "best.pt", map_location=device)["student"])
    path = export(student, out, {**meta, "best_val_loss": best}, device)
    print(f"exported {path}", flush=True)


if __name__ == "__main__":
    main()
