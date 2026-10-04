"""Shared helpers for the WakePhoneHuBERT zero-shot benchmark: the exported ONNX model, chunked long-audio inference,
greedy CTC decoding over the ipa output, and the keyword-spotting CTC scorer.

Every script of this directory takes --export-dir, --zeroshot-dir, --bench-dir and --efficientat (add_args) and applies them
with configure before it reads any path."""
import json
from pathlib import Path

import numpy as np

SR = 16000
HOP = 320
EXPORT = Path("work/export/final")
ZS = Path("work/zeroshot")
BENCH = Path("work/bench")
EFFICIENTAT = Path("work/EfficientAT")
LAYOUT = {"hubert": (0, 128), "vad": (128, 129), "ipa": (129, 521)}
DELAY = 5
CTX = 150 * HOP


def add_args(ap):
    ap.add_argument("--export-dir", default=str(EXPORT), help="build_wakephonehubert.py output directory: the ONNX files and vocab.json")
    ap.add_argument("--zeroshot-dir", default=str(ZS), help="where this benchmark writes its data, features and results")
    ap.add_argument("--bench-dir", default=str(BENCH), help="the downstream benchmark's data, feats and results (bench/)")
    ap.add_argument("--efficientat", default=str(EFFICIENTAT), help="EfficientAT checkout, for the sound-event reference")


def configure(a):
    global EXPORT, ZS, BENCH, EFFICIENTAT
    EXPORT, ZS, BENCH, EFFICIENTAT = Path(a.export_dir), Path(a.zeroshot_dir), Path(a.bench_dir), Path(a.efficientat)


def session(name="wakephonehubert_int8.onnx", threads=1):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(EXPORT / name), so, providers=["CPUExecutionProvider"])


def run(sess, wav, outputs=("features",), chunk_s=60):
    """Whole-file outputs of a causal model; long audio is cut into chunks that start on a frame boundary and carry
    CTX samples of real left context (more than the 41,920-sample receptive field), whose frames are dropped."""
    inp = sess.get_inputs()[0].name
    want = [n for n in outputs]
    step = chunk_s * SR
    total = len(wav) // HOP
    parts = {n: [] for n in want}
    for s in range(0, max(len(wav), 1), step):
        a = max(0, s - CTX)
        x = wav[a:s + step][None].astype(np.float32)
        if x.shape[1] < 2 * HOP:
            break
        o = dict(zip(want, sess.run(want, {inp: x})))
        skip = (s - a) // HOP
        for n in want:
            parts[n].append(o[n][0, skip:])
    return {n: np.concatenate(v)[:total] for n, v in parts.items()}


def vocab():
    return json.loads((EXPORT / "vocab.json").read_text())["symbols"]


def greedy(ids):
    """CTC greedy: collapse repeats, drop ids 0-3 (blank, <s>, </s>, <unk>)."""
    out, prev = [], -1
    for t in ids:
        t = int(t)
        if t != prev and t > 3:
            out.append(t)
        prev = t
    return out


def edit(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
            prev, d[j] = d[j], cur
    return d[len(b)]


def keyword_ids(word, lang="en-us"):
    """eSpeak phonemisation through the teacher tokenizer, as the phone targets of the benchmark are made."""
    from transformers import Wav2Vec2PhonemeCTCTokenizer
    t = Wav2Vec2PhonemeCTCTokenizer.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    if t.phonemizer_lang != lang or not hasattr(t, "backend"):
        t.init_backend(lang)
        t.phonemizer_lang = lang
    toks = [p for p in t.phonemize(word.lower()).split(" ") if p.strip()]
    drop = {t.pad_token_id, t.unk_token_id, t.bos_token_id, t.eos_token_id, t.convert_tokens_to_ids(t.word_delimiter_token)}
    return [i for i in t.convert_tokens_to_ids(toks) if i not in drop]


def kws_scores(ipa, kw):
    """Keyword score of each window. ipa: [N, T, 392] posteriors; kw: phone ids.

    Every frame is scored relative to its own best token, r_t(c) = log p_t(c) - max_c' log p_t(c'), so a free phone
    loop (the filler model) costs 0 on every frame and the keyword's path cost is a log likelihood ratio against it.
    The keyword is a CTC path over blank, k1, blank, k2, ..., kL, blank that may start and end on any frame.
    Returns (best_path, forward), each [N], the best end-frame score divided by L: best_path takes the max over
    alignments, forward the log-sum over them."""
    lp = np.log(np.clip(ipa.astype(np.float32), 1e-10, None))
    r = lp - lp.max(-1, keepdims=True)
    L = len(kw)
    lab = [0]
    for k in kw:
        lab += [k, 0]
    S = len(lab)
    lab = np.asarray(lab)
    skip = np.zeros(S, bool)
    for s in range(2, S):
        skip[s] = lab[s] != 0 and lab[s] != lab[s - 2]
    e = r[:, :, lab]
    N, T, _ = e.shape
    neg = np.float32(-1e30)
    res = []
    for mode in ("max", "sum"):
        a = np.full((N, S), neg, np.float32)
        best = np.full(N, neg, np.float32)
        for t in range(T):
            p1 = np.concatenate([np.full((N, 1), neg, np.float32), a[:, :-1]], 1)
            p2 = np.concatenate([np.full((N, 2), neg, np.float32), a[:, :-2]], 1)
            p2 = np.where(skip[None], p2, neg)
            start = np.full((N, S), neg, np.float32)
            start[:, :2] = 0.0
            stack = np.stack([a, p1, p2, start])
            if mode == "max":
                prev = stack.max(0)
            else:
                m = stack.max(0)
                prev = m + np.log(np.exp(stack - m).sum(0))
            a = prev + e[:, t]
            best = np.maximum(best, np.maximum(a[:, -1], a[:, -2]))
        res.append(best / L)
    return res[0], res[1]
