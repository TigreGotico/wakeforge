"""Shared pieces: data loading, IPA phonemisation, CTC word timing, boundary metrics.

Every script takes --work-dir (data/, head/ and results/ of the aligner), --export-dir (the build with
wakephonehubert_int8.onnx and vocab.json) and --librispeech (add_args), and applies them with configure."""
import io, json, pathlib
import numpy as np
import soundfile as sf

ROOT = pathlib.Path("work/aligner")
EXPORT = pathlib.Path("work/export/final")
LIBRI = pathlib.Path("work/data/LibriSpeech")
WPH = EXPORT / "wakephonehubert_int8.onnx"
TEST_PARQUET = ROOT / "data" / "hf" / "data" / "test_clean-00000-of-00001.parquet"
TEACHER = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
HOP = 0.02
DELAY_FRAMES = 5

# ARPAbet without stress, plus silence at index 0
ARPA = ["sil", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G",
        "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH",
        "UH", "UW", "V", "W", "Y", "Z", "ZH", "spn"]
ARPA_ID = {p: i for i, p in enumerate(ARPA)}


def add_args(ap):
    ap.add_argument("--work-dir", default=str(ROOT), help="data/, head/ and results/ of the aligner")
    ap.add_argument("--export-dir", default=str(EXPORT), help="build_wakephonehubert.py output directory")
    ap.add_argument("--librispeech", default=str(LIBRI), help="LibriSpeech root holding train-clean-100")


def configure(a):
    global ROOT, EXPORT, LIBRI, WPH, TEST_PARQUET
    ROOT, EXPORT, LIBRI = pathlib.Path(a.work_dir), pathlib.Path(a.export_dir), pathlib.Path(a.librispeech)
    WPH = EXPORT / "wakephonehubert_int8.onnx"
    TEST_PARQUET = ROOT / "data" / "hf" / "data" / "test_clean-00000-of-00001.parquet"


def strip_stress(p):
    return p.rstrip("012")


def load_test(limit=None):
    import pyarrow.parquet as pq
    tb = pq.read_table(TEST_PARQUET, columns=["id", "audio", "transcript", "words", "phonemes"])
    out = []
    for r in tb.to_pylist()[:limit]:
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        assert sr == 16000
        text = r["transcript"].split()
        if len(text) != len(r["words"]):
            text = [w["word"] for w in r["words"]]
        out.append(dict(id=r["id"], audio=x, words=r["words"], phonemes=r["phonemes"], text=text))
    return out


def word_phones(words, phonemes):
    """Group the MFA phones under their words (phones lie inside word spans)."""
    groups, j = [], 0
    for w in words:
        g = []
        while j < len(phonemes) and phonemes[j]["start"] < w["end"] - 1e-6:
            if phonemes[j]["start"] >= w["start"] - 1e-6:
                g.append(phonemes[j])
            j += 1
        groups.append(g)
    return groups


class IPA:
    """eSpeak NG en-us phonemisation, one word at a time, to the xlsr-53-espeak-cv-ft ids
    (the table in vocab.json); symbols outside the table are dropped."""

    def __init__(self):
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator
        self.backend = EspeakBackend("en-us", language_switch="remove-flags")
        self.sep = Separator(phone=" ", word="", syllable="")
        syms = json.load(open(EXPORT / "vocab.json"))["symbols"]
        self.vocab = {p: i for i, p in enumerate(syms) if i > 3}

    def __call__(self, words):
        out = self.backend.phonemize(words, separator=self.sep, strip=True)
        return [[self.vocab[p] for p in o.split() if p in self.vocab] for o in out]


def ctc_word_spans(logp, word_ids, shift_frames):
    """Forced-align a (T, V) log-prob tensor; return [(start_s, end_s)] per word.
    Words with no tokens get a zero-length span at the previous word's end."""
    import torch
    import torchaudio.functional as AF
    flat = [i for w in word_ids for i in w]
    labels, scores = AF.forced_align(logp[None].float(), torch.tensor([flat], dtype=torch.int32), blank=0)
    spans = AF.merge_tokens(labels[0], scores[0].exp())
    assert len(spans) == len(flat)
    out, k, last = [], 0, 0.0
    for w in word_ids:
        if not w:
            out.append((last, last))
            continue
        s = (spans[k].start - shift_frames) * HOP
        e = (spans[k + len(w) - 1].end - shift_frames) * HOP
        s, e = max(s, 0.0), max(e, 0.0)
        out.append((s, e))
        last = e
        k += len(w)
    return out


def boundary_errors(rows):
    """rows: dicts with ref_start/ref_end/pred_start/pred_end lists. Returns abs errors (s) for starts and ends."""
    st = np.concatenate([np.abs(np.array(r["pred_start"]) - np.array(r["ref_start"])) for r in rows])
    en = np.concatenate([np.abs(np.array(r["pred_end"]) - np.array(r["ref_end"])) for r in rows])
    return st, en


def summarise(err):
    return dict(n=int(err.size), within_20ms=float(np.mean(err <= 0.020 + 1e-9)),
                within_50ms=float(np.mean(err <= 0.050 + 1e-9)),
                mean_ms=float(err.mean() * 1000), median_ms=float(np.median(err) * 1000))


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]
