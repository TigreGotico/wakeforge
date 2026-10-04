"""Train the VAD head on the frozen WakeHuBERT-tiny trunk against Silero VAD v5, as the warm start of train_pool.py.

    train_vad.py vad-data <out_dir> --speech DIR --noise DIR [--clips 4000] [--seconds 8]
    train_vad.py vad <data_dir> <out_dir> [--epochs 30]

vad-data renders --clips clips of --seconds each from LibriSpeech flac files under --speech and wav files in --noise (speech,
speech over noise at 0-20 dB, noise, near-silence), with Silero's per-frame probabilities as targets. vad trains
TapHead(1, 32) with BCE on them, every tenth clip held out, and writes vad_head.pt and report.json.
"""
import argparse
import glob
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wph_model import TapHead, Trunk, load_trunk  # noqa: E402

SR, HOP, CHUNK, CONTEXT = 16000, 320, 512, 64
SILERO = ("onnx-community/silero-vad", "onnx/model.onnx", "e71cae966052b992a7eca6b17738916ce0eca4ec")


def silero_probs(sess, w):
    state = np.zeros((2, 1, 128), np.float32); ctx = np.zeros(CONTEXT, np.float32); out = []
    for i in range(0, len(w) - CHUNK + 1, CHUNK):
        p, state = sess.run(None, {"input": np.concatenate([ctx, w[i:i + CHUNK]])[None], "state": state, "sr": np.array(SR, np.int64)})
        out.append(float(p[0, 0])); ctx = w[i + CHUNK - CONTEXT:i + CHUNK]
    return np.array(out, np.float32)


def frame_targets(probs, frames):
    idx = np.minimum(((np.arange(frames) + 1) * HOP) // CHUNK, len(probs)) - 1
    return np.where(idx >= 0, probs[np.maximum(idx, 0)], 0.0).astype(np.float32)


def _read(rng, p, n):
    import soundfile as sf
    w, _ = sf.read(p, dtype="float32", always_2d=True); w = w.mean(1)
    if len(w) < n:
        return np.tile(w, n // len(w) + 1)[:n]
    s = rng.randrange(len(w) - n + 1); return w[s:s + n]


def _vad_clip(args):
    seed, speech, noise, n = args
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    global _SESS
    if "_SESS" not in globals():
        _SESS = ort.InferenceSession(hf_hub_download(SILERO[0], SILERO[1], revision=SILERO[2]), providers=["CPUExecutionProvider"])
    import soundfile as sf
    rng = random.Random(seed)

    def talk():
        out = []
        while sum(map(len, out)) < n:
            w, _ = sf.read(rng.choice(speech), dtype="float32", always_2d=True)
            out += [w.mean(1), np.zeros(rng.randrange(0, SR), np.float32)]
        return np.concatenate(out)[:n]

    kind = rng.randrange(4)
    if kind == 0:
        w = talk()
    elif kind == 1:
        s, v = talk(), _read(rng, rng.choice(noise), n)
        snr = rng.uniform(0, 20); ps, pv = np.mean(s ** 2), max(np.mean(v ** 2), 1e-10)
        w = s + v * np.sqrt(ps / (pv * 10 ** (snr / 10)))
    elif kind == 2:
        w = _read(rng, rng.choice(noise), n)
    else:
        w = np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 10 ** rng.uniform(-4, -2.5)
    w = (w * 10 ** rng.uniform(-1, 0) / max(np.abs(w).max(), 1e-6)).astype(np.float32)
    return kind, (w * 32767).astype(np.int16), frame_targets(silero_probs(_SESS, w), n // HOP)


def vad_data(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    speech = sorted(glob.glob(f"{a.speech}/**/*.flac", recursive=True)); noise = sorted(glob.glob(f"{a.noise}/*.wav"))
    assert speech and noise, (len(speech), len(noise))
    n = int(a.seconds * SR)
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(_vad_clip, [(i, speech, noise, n) for i in range(a.clips)], chunksize=8))
    np.save(out / "kind.npy", np.array([r[0] for r in res])); np.save(out / "audio.npy", np.stack([r[1] for r in res]))
    np.save(out / "target.npy", np.stack([r[2] for r in res]))
    print("VAD_DATA_DONE", len(res), flush=True)


def taps_of(trunk, wav16):
    with torch.no_grad():
        return trunk(wav16.float().cuda() / 32767)


def vad(a):
    d, out = Path(a.data_dir), Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    A, Y, K = np.load(d / "audio.npy", mmap_mode="r"), np.load(d / "target.npy"), np.load(d / "kind.npy")
    hold = np.arange(len(A)) % 10 == 0
    tr, va = np.flatnonzero(~hold), np.flatnonzero(hold)
    trunk = Trunk(load_trunk()).cuda().eval()
    torch.manual_seed(0)
    head = TapHead(1, hidden=32).cuda(); opt = torch.optim.Adam(head.parameters(), 1e-3)
    for ep in range(a.epochs):
        rng = np.random.default_rng(ep); rng.shuffle(tr)
        for b in range(0, len(tr), 32):
            j = np.sort(tr[b:b + 32])
            taps, o, m = taps_of(trunk, torch.from_numpy(np.asarray(A[j])))
            z = head(taps, o, m)[..., 0]
            y = torch.from_numpy(Y[j][:, :z.shape[1]]).cuda()
            loss = F.binary_cross_entropy_with_logits(z, y)
            opt.zero_grad(); loss.backward(); opt.step()
        pv = []
        with torch.no_grad():
            for b in range(0, len(va), 32):
                j = va[b:b + 32]; taps, o, m = taps_of(trunk, torch.from_numpy(np.asarray(A[j])))
                pv.append(torch.sigmoid(head(taps, o, m)[..., 0]).cpu().numpy())
        pv = np.concatenate(pv); yv = Y[va][:, :pv.shape[1]]
        print(json.dumps({"epoch": ep + 1, "agree@0.5": round(float(((pv >= .5) == (yv >= .5)).mean()), 4),
                          "mae": round(float(np.abs(pv - yv).mean()), 4)}), flush=True)
    rep = {"params": sum(p.numel() for p in head.parameters()), "mix_weights": torch.softmax(head.mix, 0).tolist()}
    for k, name in enumerate(["speech", "speech+noise", "noise", "near-silence"]):
        m = K[va] == k
        rep[name] = {"silero_mean": round(float(yv[m].mean()), 4), "head_mean": round(float(pv[m].mean()), 4),
                     "agree@0.5": round(float(((pv[m] >= .5) == (yv[m] >= .5)).mean()), 4)}
    rep["overall_agree@0.5"] = round(float(((pv >= .5) == (yv >= .5)).mean()), 4)
    (out / "report.json").write_text(json.dumps(rep, indent=1)); torch.save(head.state_dict(), out / "vad_head.pt")
    print(json.dumps(rep, indent=1)); print("VAD_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("vad-data"); p.add_argument("out_dir"); p.add_argument("--speech", required=True); p.add_argument("--noise", required=True)
    p.add_argument("--clips", type=int, default=4000); p.add_argument("--seconds", type=float, default=8.0)
    p = sub.add_parser("vad"); p.add_argument("data_dir"); p.add_argument("out_dir"); p.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    {"vad-data": vad_data, "vad": vad}[a.cmd](a)


if __name__ == "__main__":
    main()
