"""vad_eval.py [--systems silero,webrtc] [--systems wakephonehubert] [--bench-dir DIR] [--heads RUN_DIR]

Plain VAD frame F1 on LibriParty (SpeechBrain) for Silero v5, WebRTC VAD and the WakePhoneHuBERT vad head.

Reference: union of every speaker's utterance spans in the session JSON, on a 10 ms grid. Each system's decisions are held
over its own frame (Silero 32 ms, WebRTC 30 ms, WakePhoneHuBERT 20 ms) and read on that grid. Speech-class precision, recall
and F1 pooled over all frames of the 50 eval sessions. Score-producing systems are reported at 0.5 and at the threshold that
maximises F1 on the 50 dev sessions.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import featlib  # noqa: E402

B = Path("work/bench")
LP = B / "data/libriparty/LibriParty/dataset"
GRID = 160


def sessions(split):
    for d in sorted((LP / split).iterdir(), key=lambda p: int(p.name.split("_")[1])):
        meta = json.loads((d / f"{d.name}.json").read_text())
        wav, sr = sf.read(d / f"{d.name}_mixture.wav", dtype="float32")
        assert sr == 16000
        n = len(wav) // GRID
        ref = np.zeros(n, bool)
        for k, segs in meta.items():
            if k.isdigit():
                for s in segs:
                    ref[int(round(s["start"] * 100)):int(round(s["stop"] * 100))] = True
        yield d.name, wav, ref


def on_grid(scores, hop, n):
    idx = np.minimum((np.arange(n) * GRID) // hop, len(scores) - 1)
    return scores[idx]


def prf(ref, hyp):
    tp = int((ref & hyp).sum()); fp = int((~ref & hyp).sum()); fn = int((ref & ~hyp).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4)}


def score_silero(wav):
    from pool_teachers import CHUNK, silero_batch, silero_session
    global _S
    if "_S" not in globals():
        _S = silero_session()
    p = silero_batch(_S, wav[None, :len(wav) // CHUNK * CHUNK])[0]
    return p, CHUNK


def score_webrtc(wav, mode):
    import webrtcvad
    v = webrtcvad.Vad(mode)
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    fr = 480
    return np.array([v.is_speech(pcm[i:i + fr].tobytes(), 16000) for i in range(0, len(pcm) - fr + 1, fr)], float), fr


def score_wph(wav):
    import torch
    from featlib import Featurizer
    global _F
    if "_F" not in globals():
        _F = Featurizer("wakephonehubert")
    e, _ = _F.frames([wav])
    return e["vad"][0, :, 0].cpu().numpy(), 320


def main():
    global B, LP
    ap = argparse.ArgumentParser()
    featlib.add_args(ap)
    ap.add_argument("--systems", default="silero,webrtc")
    ap.add_argument("--json-name", default="results.json")
    ap.add_argument("--raw-prefix", default="raw")
    a = ap.parse_args()
    B = featlib.configure(a)
    LP = B / "data/libriparty/LibriParty/dataset"
    systems = []
    for s in a.systems.split(","):
        systems += [f"webrtc-{m}" for m in range(4)] if s == "webrtc" else [s]
    out = {}
    od = B / "results/vad"
    od.mkdir(parents=True, exist_ok=True)
    for name in systems:
        t0 = time.time()
        res = {}
        raw = {}
        for split in ("dev", "eval"):
            refs, scs, names, native = [], [], [], []
            for sess, wav, ref in sessions(split):
                if name == "silero":
                    sc, hop = score_silero(wav)
                elif name.startswith("webrtc"):
                    sc, hop = score_webrtc(wav, int(name.split("-")[1]))
                else:
                    sc, hop = score_wph(wav)
                refs.append(ref); scs.append(on_grid(sc, hop, len(ref))); names.append(sess); native.append(np.asarray(sc, np.float32))
            res[split] = (np.concatenate(refs), np.concatenate(scs))
            raw[f"{split}_sessions"] = np.asarray(names)
            raw[f"{split}_offsets"] = np.cumsum([0] + [len(x) for x in refs]).astype(np.int64)
            raw[f"{split}_labels"] = res[split][0]
            raw[f"{split}_score_10ms"] = res[split][1].astype(np.float32)
            raw[f"{split}_native_hop_samples"] = np.asarray(hop)
            raw[f"{split}_native_offsets"] = np.cumsum([0] + [len(x) for x in native]).astype(np.int64)
            raw[f"{split}_native_score"] = np.concatenate(native)
        np.savez_compressed(od / f"{a.raw_prefix}-{name}.npz", **raw)
        r = {"eval_at_0.5": prf(res["eval"][0], res["eval"][1] >= 0.5), "dev_at_0.5": prf(res["dev"][0], res["dev"][1] >= 0.5),
             "eval_speech_fraction": round(float(res["eval"][0].mean()), 4), "seconds": round(time.time() - t0)}
        if not name.startswith("webrtc"):
            ths = np.round(np.arange(0.05, 0.96, 0.05), 2)
            f1s = [prf(res["dev"][0], res["dev"][1] >= t)["f1"] for t in ths]
            t = float(ths[int(np.argmax(f1s))])
            r["dev_tuned_threshold"] = t
            r["eval_at_dev_threshold"] = prf(res["eval"][0], res["eval"][1] >= t)
        out[name] = r
        print(json.dumps({name: r}), flush=True)
    f = od / a.json_name
    allr = json.loads(f.read_text()) if f.exists() else {}
    allr.update(out)
    f.write_text(json.dumps(allr, indent=1))


if __name__ == "__main__":
    main()
