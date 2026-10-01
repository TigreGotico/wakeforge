"""Synthesize a wake word, or phrases that sound like it, across every speaker of a multi-speaker Piper voice.

Each clip draws a speaker, a phrasing of the text, and Piper's length, noise and phoneme-duration noise scales
at random, so a few thousand clips cover hundreds of voices at several speaking rates. Clips are written as
16 kHz mono FLAC, one directory per phrase list, with a manifest row per clip recording everything drawn.

    06_piper_multispeaker.py --voice en_US-libritts_r-medium.onnx --phrases phrases.json --word hey_mycroft \
        --kind positive --n 10000 --out out/ --shard 0/4
"""
import argparse, csv, json, random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import soxr

_Session = ort.InferenceSession


def _sized_session(path, sess_options=None, **kw):
    # onnxruntime ignores OMP_NUM_THREADS and starts a thread per core; inside a one-core quota those threads
    # only contend, so a generator job runs on --threads threads and never spin-waits
    so = sess_options or ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = THREADS
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return _Session(path, sess_options=so, **kw)


ort.InferenceSession = _sized_session
THREADS = 1
from piper import PiperVoice, SynthesisConfig  # noqa: E402  (after the session patch)

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--phrases", required=True, help="JSON: {word: {positive: [...], negative: [...]}}")
    ap.add_argument("--word", required=True)
    ap.add_argument("--kind", choices=("positive", "negative"), required=True)
    ap.add_argument("--n", type=int, required=True, help="clips over all shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1, help="onnxruntime threads for this job")
    a = ap.parse_args()
    global THREADS
    THREADS = a.threads

    i0, n0 = (int(x) for x in a.shard.split("/"))
    texts = json.load(open(a.phrases))[a.word][a.kind]
    voice = PiperVoice.load(a.voice)
    speakers = voice.config.num_speakers
    out = Path(a.out) / a.word / a.kind
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / f"manifest.{i0}of{n0}.csv"
    done = {r[0] for r in csv.reader(open(manifest))} if manifest.exists() else set()
    with open(manifest, "a", newline="") as fh:
        w = csv.writer(fh)
        for k in range(i0, a.n, n0):
            name = f"{k:06d}.flac"
            if name in done:
                continue
            rng = random.Random(f"{a.seed}-{a.word}-{a.kind}-{k}")
            text, spk = rng.choice(texts), rng.randrange(speakers)
            ls, ns, nw = rng.uniform(0.75, 1.35), rng.uniform(0.4, 1.0), rng.uniform(0.5, 1.1)
            cfg = SynthesisConfig(speaker_id=spk, length_scale=ls, noise_scale=ns, noise_w_scale=nw)
            audio = np.concatenate([c.audio_float_array for c in voice.synthesize(text, syn_config=cfg)])
            audio = soxr.resample(audio, voice.config.sample_rate, SR).astype(np.float32)
            if not len(audio) or not np.isfinite(audio).all() or np.abs(audio).max() < 1e-3:
                continue
            sf.write(out / name, audio, SR, format="FLAC", subtype="PCM_16")
            w.writerow([name, text, spk, f"{ls:.3f}", f"{ns:.3f}", f"{nw:.3f}", f"{len(audio) / SR:.3f}"])
            fh.flush()


if __name__ == "__main__":
    main()
