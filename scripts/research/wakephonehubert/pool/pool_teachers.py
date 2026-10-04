"""pool_teachers.py <pool_dir> --source <name> --type <mswc|audioset|fma> [--limit N] [--stages silero,text,ipa,voice,forced]

<name> is any source in pool_dir: a build_pool.py source (<name>_audio.raw) or a snapshot_pool.py slice of one
(<name>.slice.json naming the base raw file, start segment and count). --type sets the segment length and the rules
(1 s words with CTC targets for mswc, 10 s segments for audioset and fma) and is recorded in the progress file.

Teacher outputs for the segments of one build_pool.py source, written beside it and resumable by chunk:
  <src>_silero.npy     [n, frames] fp16     Silero VAD v5 per 20 ms student frame
  <src>_tk_idx.npy     [n, T, 8] int16      wav2vec2-xlsr-53-espeak-cv-ft top-8 per frame (clip-normalised input)
  <src>_tk_p.npy       [n, T, 8] fp16       their renormalised probabilities
  <src>_voice.npy      [n, W] fp16          voice evidence: EfficientAT mn10_as scores over causal 5 s windows (zero-padded on
                                            the left) ending every 16 student frames, maximum over the AudioSet Speech
                                            subtree (/m/09x0r)
  <src>_forced.npy     [n, frames] bool     forced non-speech: (fma, or audioset with no Human-voice label) and voice < 0.1
                                            in the window ending at or after the frame; always False for mswc
  <src>_ipa_ids.jsonl  (mswc) {"voice", "ids"} per clip: the teacher tokenizer's ids for the word through the language's
                       espeak-ng voice (pad, <s>, </s>, <unk>/word delimiter dropped); empty when the language has no voice,
                       espeak switches language on the word, or it has more than 24 ids
Progress lives in <src>_progress.json. n is the source's .done count, or --limit on a source still being built.
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SR, HOP, CHUNK, CONTEXT = 16000, 320, 512, 64
SEG = {"mswc": SR, "audioset": 10 * SR, "fma": 10 * SR}
SILERO = ("onnx-community/silero-vad", "onnx/model.onnx", "e71cae966052b992a7eca6b17738916ce0eca4ec")
W2V = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
VOICE_ROOT, SPEECH_ROOT = "/m/09l8g", "/m/09x0r"
WINDOW_HOP, WINDOW_S, SR_T = 16, 5, 32000
FORCED_VOICE_MAX = 0.1


def frame_targets(probs, frames):
    """Silero chunk probabilities to one value per 20 ms student frame (no torch import: the CPU workers must not load it)."""
    idx = np.minimum(((np.arange(frames) + 1) * HOP) // CHUNK, len(probs)) - 1
    return np.where(idx >= 0, probs[np.maximum(idx, 0)], 0.0).astype(np.float32)


def silero_batch(sess, w):
    """Silero probabilities over a batch [b, samples]: one session call per 32 ms chunk for all rows."""
    b = len(w)
    state = np.zeros((2, b, 128), np.float32); ctx = np.zeros((b, CONTEXT), np.float32); out = []
    for i in range(0, w.shape[1] - CHUNK + 1, CHUNK):
        x = w[:, i:i + CHUNK]
        p, state = sess.run(None, {"input": np.concatenate([ctx, x], 1), "state": state, "sr": np.array(SR, np.int64)})
        out.append(p[:, 0]); ctx = x[:, -CONTEXT:]
    return np.stack(out, 1).astype(np.float32)


def silero_session():
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    return ort.InferenceSession(hf_hub_download(SILERO[0], SILERO[1], revision=SILERO[2]), so, providers=["CPUExecutionProvider"])


def source_audio(d, name, seg):
    """(raw path, first segment, segments available, complete) for a source or a snapshot slice of one."""
    d = Path(d)
    sl = d / f"{name}.slice.json"
    if sl.exists():
        j = json.loads(sl.read_text())
        assert j["seg"] == seg, (sl, j["seg"], seg)
        return j["raw"], j["start"], j["count"], True
    raw = d / f"{name}_audio.raw"
    done = d / f"{name}.done"
    if done.exists():
        return str(raw), 0, int(done.read_text().strip()), True
    with open(d / f"{name}_meta.jsonl", "rb") as f:
        lines = sum(b.count(b"\n") for b in iter(lambda: f.read(1 << 24), b""))
    return str(raw), 0, min(raw.stat().st_size // (2 * seg), lines), False


def open_audio(raw, start, n, seg):
    return np.memmap(raw, np.int16, "r", offset=2 * seg * start, shape=(n, seg))


def _silero_task(args):
    raw, start, seg, n, lo, hi = args
    global _S
    if "_S" not in globals():
        _S = silero_session()
    A = open_audio(raw, start, n, seg)
    probs = silero_batch(_S, np.asarray(A[lo:hi], np.float32) / 32767)
    return lo, np.stack([frame_targets(p, seg // HOP) for p in probs]).astype(np.float16)


VOICE_ALIASES = {"en": "en-us", "fr": "fr-fr", "zh": "cmn-latn-pinyin"}
DROP_IDS = (0, 1, 2, 3)
MAX_IDS = 24


def espeak_voices():
    import subprocess
    lines = subprocess.run(["espeak-ng", "--voices"], capture_output=True, text=True).stdout.splitlines()[1:]
    return {l.split()[1] for l in lines if len(l.split()) > 1}


def voice_for(code, voices):
    """MSWC language code -> espeak-ng voice, or None: region suffix stripped, en -> en-us, fr -> fr-fr,
    zh -> cmn-latn-pinyin (plain cmn spells tone digits out as English numbers)."""
    base = code.split("-")[0]
    v = VOICE_ALIASES.get(base, base)
    return v if v in voices else None


def _ids_task(args):
    """Teacher-tokenizer ids per word for one voice, or the reason there are none: "switch" when espeak switches
    language on the word (it then reads it with another voice, typically as spelled-out English letters), "unk" when
    a phone espeak produced is not in the teacher's table (dropping it would leave a wrong target)."""
    voice, words = args
    global _TOK
    if "_TOK" not in globals():
        from transformers import Wav2Vec2PhonemeCTCTokenizer
        _TOK = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(W2V)
    from phonemizer.backend import EspeakBackend
    flagged = EspeakBackend(voice, language_switch="keep-flags").phonemize([w.lower() for w in words], strip=True)
    out = {}
    _TOK.init_backend(voice); _TOK.phonemizer_lang = voice
    delim = _TOK.word_delimiter_token
    for w, f in zip(words, flagged):
        if "(" in f:
            out[w] = "switch"
            continue
        toks = [t for t in _TOK.phonemize(w.strip().lower()).split(" ") if t.strip() and t != delim]
        ids = _TOK.convert_tokens_to_ids(toks)
        bad = [t for t, i in zip(toks, ids) if i == _TOK.unk_token_id]
        out[w] = ("unk:" + " ".join(bad)) if bad else [i for i in ids if i not in DROP_IDS]
    return out


def usable(x):
    return isinstance(x, list) and 0 < len(x) <= MAX_IDS


def subtree(ontology, root):
    by = {o["id"]: o for o in ontology}
    out, stack = set(), [root]
    while stack:
        k = stack.pop()
        if k in out or k not in by:
            continue
        out.add(k); stack += by[k].get("child_ids", [])
    return out


class Pool:
    def __init__(self, d, src, typ, limit, chunk):
        self.d, self.src, self.type, self.seg = Path(d).resolve(), src, typ, SEG[typ]
        self.frames = self.seg // HOP
        self.raw, self.start, n, complete = source_audio(d, src, self.seg)
        if not complete and not limit:
            raise SystemExit(f"{src} is still being built (no {src}.done); snapshot it with snapshot_pool.py or pass --limit")
        self.n = min(n, limit) if limit else n
        self.audio = open_audio(self.raw, self.start, self.n, self.seg)
        self.prog_path = self.d / f"{src}_progress.json"
        self.prog = json.loads(self.prog_path.read_text()) if self.prog_path.exists() else {"n": self.n, "chunk": chunk, "type": typ}
        if self.prog.get("type", src) != typ:
            raise SystemExit(f"{self.prog_path} records type {self.prog.get('type', src)}, this run has --type {typ}")
        self.prog["type"] = typ
        if (self.prog["n"], self.prog["chunk"]) != (self.n, chunk):
            raise SystemExit(f"{self.prog_path} was written for n={self.prog['n']} chunk={self.prog['chunk']}, this run has n={self.n} "
                             f"chunk={chunk}; move the {src}_* teacher files aside first")
        self.lock = threading.Lock()

    def path(self, name):
        return self.d / f"{self.src}_{name}"

    def done(self, stage):
        return self.prog.get(stage, 0)

    def mark(self, stage, value):
        with self.lock:
            self.prog[stage] = value
            tmp = self.prog_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.prog)); tmp.replace(self.prog_path)

    def out(self, name, dtype, shape, stage):
        p = self.path(f"{name}.npy")
        if p.exists() and self.done(stage):
            a = np.load(p, mmap_mode="r+")
            assert a.shape == shape and a.dtype == np.dtype(dtype), (p, a.shape, shape, a.dtype, dtype)
            return a
        return np.lib.format.open_memmap(p, "w+", dtype, shape)


def chunks(n, size):
    return [(lo, min(lo + size, n)) for lo in range(0, n, size)]


def stage_silero(P, ex, chunk, task):
    cs = chunks(P.n, chunk)
    start = P.done("silero")
    if start >= len(cs):
        print("silero already complete", flush=True); return
    out = P.out("silero", np.float16, (P.n, P.frames), "silero")
    jobs = [(c, lo2, min(lo2 + task, hi)) for c, (lo, hi) in enumerate(cs) if c >= start for lo2 in range(lo, hi, task)]
    last = {}
    for c, lo, hi in jobs:
        last[c] = lo
    t0, done = time.time(), 0
    for (c, lo, hi), (lo_r, y) in zip(jobs, ex.map(_silero_task, [(P.raw, P.start, P.seg, P.n, lo, hi) for _, lo, hi in jobs])):
        out[lo_r:lo_r + len(y)] = y; done += len(y)
        if lo == last[c]:
            out.flush(); P.mark("silero", c + 1)
            print(json.dumps({"stage": "silero", "chunk": c + 1, "of": len(cs), "seg_per_s": round(done / (time.time() - t0), 1)}), flush=True)


def stage_text(P, ex):
    if P.type != "mswc" or P.done("ids"):
        return
    from transformers import Wav2Vec2PhonemeCTCTokenizer
    tok = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(W2V)
    voices = espeak_voices()
    meta = []
    with open(P.path("meta.jsonl"), encoding="utf-8") as f:
        for line in f:
            if len(meta) >= P.n:
                break
            meta.append(json.loads(line))
    by = {}
    for m in meta:
        by.setdefault(m["lang"], set()).add(m["word"])
    vmap = {lang: voice_for(lang, voices) for lang in by}
    jobs = [(lang, sorted(ws)[i:i + 400]) for lang, ws in sorted(by.items()) if vmap[lang] for i in range(0, len(ws), 400)]
    ids, t0 = {}, time.time()
    for (lang, _), res in zip(jobs, ex.map(_ids_task, [(vmap[l], w) for l, w in jobs])):
        for w, x in res.items():
            ids[(lang, w)] = x
    for lang in sorted(by):
        ws = sorted(by[lang]); got = [ids.get((lang, w)) for w in ws]
        switched, unk = sum(x == "switch" for x in got), sum(isinstance(x, str) and x.startswith("unk") for x in got)
        good = [(w, x) for w, x in zip(ws, got) if usable(x)]
        samples = [(w, " ".join(tok.convert_ids_to_tokens(x))) for w, x in good[:3]]
        ratio = float(np.median([len(x) / max(1, len(w)) for w, x in good])) if good else None
        flag = vmap[lang] is not None and (switched / len(ws) > 0.2 or unk / len(ws) > 0.2 or (ratio or 0) > 2.0)
        print(json.dumps({"stage": "ids", "lang": lang, "voice": vmap[lang], "words": len(ws), "empty_share": round(1 - len(good) / len(ws), 3),
                          "language_switch_share": round(switched / len(ws), 3), "unknown_phone_share": round(unk / len(ws), 3),
                          "ids_per_char": None if ratio is None else round(ratio, 2), "samples": samples, "FLAG_check_samples": flag},
                         ensure_ascii=False), flush=True)
    kept = 0
    tmp = P.path("ipa_ids.jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for m in meta:
            x = ids.get((m["lang"], m["word"]))
            x = x if usable(x) else []
            kept += bool(x)
            f.write(json.dumps({"voice": vmap[m["lang"]], "ids": x}) + "\n")
    tmp.replace(P.path("ipa_ids.jsonl")); P.mark("ids", 1)
    print(json.dumps({"stage": "ids", "clips": len(meta), "with_targets": kept, "words": len(ids), "s": round(time.time() - t0, 1)}), flush=True)


def stage_ipa(P, chunk, batch_seconds, topk):
    import torch
    from transformers import Wav2Vec2ForCTC
    cs = chunks(P.n, chunk)
    start = P.done("ipa")
    if start >= len(cs):
        print("ipa already complete", flush=True); return
    model = Wav2Vec2ForCTC.from_pretrained(W2V).cuda().eval().half()
    T = int(model._get_feat_extract_output_lengths(torch.tensor(P.seg)))
    oi = P.out("tk_idx", np.int16, (P.n, T, topk), "ipa"); op = P.out("tk_p", np.float16, (P.n, T, topk), "ipa")
    bs = max(1, int(batch_seconds * SR // P.seg))
    t0, done = time.time(), 0
    for c in range(start, len(cs)):
        lo, hi = cs[c]
        for i in range(lo, hi, bs):
            x = torch.from_numpy(np.asarray(P.audio[i:min(i + bs, hi)], np.float32) / 32767).cuda()
            x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-7)
            with torch.no_grad():
                pr = torch.softmax(model(x.half()).logits.float(), -1)
            top = pr.topk(topk, -1)
            oi[i:i + len(x)] = top.indices.cpu().numpy().astype(np.int16)
            op[i:i + len(x)] = (top.values / top.values.sum(-1, keepdim=True)).cpu().numpy().astype(np.float16)
            done += len(x)
        oi.flush(); op.flush(); P.mark("ipa", c + 1)
        print(json.dumps({"stage": "ipa", "chunk": c + 1, "of": len(cs), "seg_per_s": round(done / (time.time() - t0), 1)}), flush=True)


def efficientat_model(efficientat_dir):
    import sys
    import csv
    efficientat_dir = str(Path(efficientat_dir).resolve())
    sys.path.insert(0, efficientat_dir)
    cwd = os.getcwd(); os.chdir(efficientat_dir)
    from models.mn.model import get_model
    from models.preprocess import AugmentMelSTFT
    model = get_model(width_mult=1.0, pretrained_name="mn10_as").cuda().eval()
    mel = AugmentMelSTFT(n_mels=128, sr=SR_T, win_length=800, hopsize=320).cuda().eval()
    ids = [r[1] for r in list(csv.reader(open("metadata/class_labels_indices.csv")))[1:]]
    os.chdir(cwd)
    return model, mel, ids


def window_ends(frames):
    return np.arange(WINDOW_HOP, frames + 1, WINDOW_HOP)


def stage_voice(P, chunk, efficientat_dir, ontology_path, batch_windows):
    import torch
    import torchaudio
    cs = chunks(P.n, chunk)
    start = P.done("voice")
    if start >= len(cs):
        print("voice already complete", flush=True); return
    model, mel, ids = efficientat_model(efficientat_dir)
    speech = subtree(json.loads(Path(ontology_path).read_text()), SPEECH_ROOT)
    sp_idx = torch.tensor([i for i, c in enumerate(ids) if c in speech]).cuda()
    ends = window_ends(P.frames); W = len(ends)
    win = WINDOW_S * SR_T
    starts = [int(e * HOP * 2) for e in ends]
    ov = None
    bs = max(1, batch_windows // W)
    t0, done = time.time(), 0
    for c in range(start, len(cs)):
        lo, hi = cs[c]
        for i in range(lo, hi, bs):
            w = torch.from_numpy(np.asarray(P.audio[i:min(i + bs, hi)], np.float32) / 32767).cuda()
            w = torch.nn.functional.pad(torchaudio.functional.resample(w, SR, SR_T), (win, 0))
            segs = torch.stack([w[:, s:s + win] for s in starts], 1).reshape(-1, win)
            with torch.no_grad():
                logits, _ = model(mel(segs).unsqueeze(1))
            p = torch.sigmoid(logits.float()).reshape(len(w), W, -1)
            if ov is None:
                ov = P.out("voice", np.float16, (P.n, W), "voice")
            ov[i:i + len(w)] = p[..., sp_idx].amax(-1).cpu().numpy().astype(np.float16)
            done += len(w)
        ov.flush(); P.mark("voice", c + 1)
        print(json.dumps({"stage": "voice", "chunk": c + 1, "of": len(cs), "seg_per_s": round(done / (time.time() - t0), 1)}), flush=True)


def stage_forced(P, chunk, ontology_path):
    if P.done("voice") < len(chunks(P.n, chunk)):
        print("forced needs the voice stage complete; skipped", flush=True); return
    if P.type == "mswc":
        f = np.zeros((P.n, P.frames), bool)
    else:
        v = np.load(P.path("voice.npy"), mmap_mode="r")
        win = np.minimum(np.arange(P.frames) // WINDOW_HOP, v.shape[1] - 1)
        f = np.asarray(v, np.float32)[:, win] < FORCED_VOICE_MAX
        if P.type == "audioset":
            voice = subtree(json.loads(Path(ontology_path).read_text()), VOICE_ROOT)
            ok = np.zeros(P.n, bool)
            with open(P.path("meta.jsonl")) as fh:
                for i, line in enumerate(fh):
                    if i >= P.n:
                        break
                    ok[i] = not (set(json.loads(line)["labels"]) & voice)
            f &= ok[:, None]
    np.save(P.path("forced.npy"), f); P.mark("forced", 1)
    print(json.dumps({"stage": "forced", "frames_forced": round(float(f.mean()), 4), "segments_all_forced": int(f.all(1).sum()),
                      "segments_any_forced": int(f.any(1).sum()), "n": P.n}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool_dir"); ap.add_argument("--source", required=True)
    ap.add_argument("--type", choices=list(SEG), help="defaults to --source when that is a type name")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stages", default="silero,text,ipa,voice,forced")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--silero-task", type=int, default=64, help="segments per CPU task")
    ap.add_argument("--ipa-batch-seconds", type=float, default=320)
    ap.add_argument("--voice-batch-windows", type=int, default=384)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--ontology", default="work/data/audioset/ontology.json", help="AudioSet ontology.json (voice and speech subtrees)")
    ap.add_argument("--efficientat", default="work/EfficientAT", help="checkout of the EfficientAT repository (mn10_as weights)")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    typ = a.type or (a.source if a.source in SEG else None)
    if typ is None:
        raise SystemExit("--type is required for a source not named mswc, audioset or fma")
    P = Pool(a.pool_dir, a.source, typ, a.limit, a.chunk)
    stages = set(a.stages.split(","))
    print(json.dumps({"source": a.source, "n": P.n, "seg": P.seg, "progress": P.prog}), flush=True)
    import multiprocessing as mp
    ex = ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"))
    errors = []

    def cpu():
        try:
            if "silero" in stages:
                stage_silero(P, ex, a.chunk, a.silero_task)
            if "text" in stages:
                stage_text(P, ex)
        except BaseException as e:
            errors.append(e); raise

    if stages & {"text", "ipa"}:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer
    th = threading.Thread(target=cpu); th.start()
    if "ipa" in stages:
        stage_ipa(P, a.chunk, a.ipa_batch_seconds, a.topk)
    if "voice" in stages:
        stage_voice(P, a.chunk, a.efficientat, a.ontology, a.voice_batch_windows)
    th.join(); ex.shutdown()
    if errors:
        raise SystemExit(f"CPU stage failed: {errors[0]!r}")
    if "forced" in stages:
        stage_forced(P, a.chunk, a.ontology)
    print("POOL_TEACHERS_DONE", a.source, json.dumps(P.prog), flush=True)


if __name__ == "__main__":
    main()
