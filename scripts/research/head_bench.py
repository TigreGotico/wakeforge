"""Wake-word head bench on frozen featurizers: features computed once, many heads trained at once.

    head_bench.py cache <featurizer.onnx> <cache_dir> [--metadata kws/train.csv --val kws/val.csv] [--feature-store DIR]
    bench.py fit <cache_dir> <out_dir> [--heads gru,gru-h256,kwt:d_model=32,...] [--loss bce] [--epochs 20] [--lr-schedule cosine --warmup-epochs 2]
    bench.py export <out_dir> <head_name> [--allow-fixed-frames] [--calibrate --cache-dir DIR --target-fa-per-hour 1.0]
                                                         (ONNX head that kws_eval.py can score)

``cache`` takes the featurizer as an ONNX path or as a pretrained name (``wakehubert-tiny``, ``wakewav-mel-tcn-int8``, ...);
a path that exists wins. The cache's meta.json records the resolved file and its sha1.

``fit --heads`` takes the GRU specs (``gru``, ``gru-h256``, ``bigru-l2``) and any ``ww_trainer.factory.HEAD_REGISTRY`` name,
optionally with keyword overrides, ``kwt:d_model=32,n_layers=2``. ``ocsvm`` and ``phonmatch`` cannot give one logit per
window from features alone and are refused. ``--loss`` is ``bce``, ``focal`` or ``label_smoothing_bce``, or a weighted sum such as
``bce+0.5*focal``. The logged validation loss is BCE whichever loss trains, and the kept checkpoint is the one with the highest
calibration recall, then the lowest validation BCE. Training-loss values do not compare across losses: focal (alpha 0.9)
weights positives about nine to one and its value is a fifth of BCE's on random logits and a seventeenth at zero logits with one positive in six, so a ``0.5*focal`` term is small beside ``bce``.

``export`` refuses a head whose graph only runs at the cache's frame count (kwt, conformer) unless ``--allow-fixed-frames``
is given; the ONNX file then carries ``fixed_frames`` in its metadata.

``export --calibrate`` puts an affine map on the head's logit (a Mul and an Add before the output), so the exported head
still outputs one logit and the plugin's sigmoid needs no change. The map is read
on ``calib_stream.npy`` (``cache --calib-stream-hours`` or ``calib-stream``): hours of continuous dev-clean and dev-other
speech, babble of both and AudioSet noise featurized as one stream, never audio the head trains on (a source that
contains a training file or the augmentation talker pool is refused). The head is scored every 80 ms over its last
window and detections are counted as the plugin counts them, with a 2 s quiet period after each. The logit giving
``--target-fa-per-hour`` detections per hour maps to probability 0.5; a rate with fewer than 20 detections in the audio
is extended from the measured tail. The scale is chosen so the median calibration positive maps to probability 0.9,
which keeps positives off the sigmoid's saturated end.

The operating point is then chosen on the same data: thresholds from 0.05 to 0.99 on the calibrated probability, each
scoring recall on the calibration positives and the debounced false activations on the stream. ``--positives-per-hour``
(default 4) says how many real wake words there are per hour of audio, which weights true positives against false
activations in the precision. The F2-optimal threshold becomes the ONNX ``default_threshold`` and the metadata records
``roc_auc`` (calibration positives against every stream block), ``f2_threshold``, ``f2``, ``f2_recall`` and
``positives_per_hour``. A threshold outside [0.5, 0.7] is reported as a calibration that is not good enough.

``cache`` runs the featurizer once over every 1.5 s training and validation window (the same windows
ww_trainer-train reads) and stores float16 features; every training positive is also stored as
--pos-aug augmented copies and a quarter of the negatives once (``augmenter``), since a head trained on
clean synthetic clips alone does not survive noise. ``fit`` trains every listed head on the same batches:
each epoch uses every positive and five times as many negatives, half the highest-scoring negatives
of the previous epoch (hard negatives) and half drawn at random; BCE; the kept checkpoint is the one with
the highest calibration recall. Heads share the batch order, so they differ only in architecture.

``--feature-store DIR`` keeps every window's features keyed by the featurizer file and the window's exact
samples, so a later ``cache`` run featurizes only windows it has not seen: the negatives several words share,
unchanged clips, and augmented copies, which are seeded by their source's content and copy index rather than
by their position in the metadata. ``fit --lr-schedule cosine`` anneals the rate to lr/100 over the epochs
after ``--warmup-epochs`` of linear warmup (the peak is the epoch after them); the rate of every epoch is logged in fit.jsonl.

Checkpoints are chosen on a calibration set that no evaluation touches, because the synthetic validation
set is separable within a few epochs and its loss then keeps falling as the head over-fits: the
validation positives mixed with LibriSpeech dev-other babble (three talkers) at 10 and 5 dB and with
AudioSet noise at 5 dB, against one hour of dev-other speech and dev-other babble cut into windows at a
0.5 s hop. Calibration recall is the share of those positives above the highest-scoring negative window.
"""
import argparse, ast, csv, hashlib, json, math, os, platform, random, re, subprocess, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

SR, N = 16000, 24000


PATH_MAP = []  # (old prefix, new prefix) pairs from --path-map, for metadata written on another machine
CALIB_SPEECH = "data/LibriSpeech/dev-other"


def read_meta(path):
    rows = [r for r in csv.reader(open(path)) if r]
    files = []
    for r in rows:
        f = r[0]
        for old, new in PATH_MAP:
            if f.startswith(old):
                f = new + f[len(old):]
        files.append(f)
    return files, np.array([int(r[1]) for r in rows], np.int8)


def load_window(p):
    import soundfile as sf
    w, sr = sf.read(p, dtype="float32", always_2d=True)
    w = w.mean(1)
    assert sr == SR, (p, sr)
    out = np.zeros(N, np.float32); out[:min(N, len(w))] = w[:N]
    return out


def calibration_windows(val_files, val_labels, seconds=3600, seed=0):
    """(noisy positives [P, N], negative windows [M, N]) from sources the evaluation never uses."""
    import glob, soundfile as sf
    rng = random.Random(seed)
    other = sorted(glob.glob(f"{CALIB_SPEECH}/**/*.flac", recursive=True))
    noise = sorted(glob.glob("data/noise-audioset/*.wav"))

    def clip(p, n):
        w, sr = sf.read(p, dtype="float32", always_2d=True); w = w.mean(1)
        if len(w) < n:
            w = np.tile(w, n // len(w) + 1)
        s = rng.randint(0, len(w) - n); return w[s:s + n]

    def babble(n):
        return sum(clip(rng.choice(other), n) for _ in range(3)) / 3

    def mix(x, v, snr):
        pc, pn = np.mean(x ** 2), max(np.mean(v ** 2), 1e-10)
        y = x + v * np.sqrt(pc / (pn * 10 ** (snr / 10)))
        return (y / max(1.0, np.abs(y).max())).astype(np.float32)

    pos = [load_window(f) for f, l in zip(val_files, val_labels) if l == 1]
    noisy = [mix(x, babble(N), 10) for x in pos] + [mix(x, babble(N), 5) for x in pos] + \
            [mix(x, clip(rng.choice(noise), N), 5) for x in pos]
    neg, total = [], 0
    for p in rng.sample(other, len(other)):
        w, _ = sf.read(p, dtype="float32")
        neg += [w[i:i + N] for i in range(0, len(w) - N + 1, SR // 2)]; total += len(w)
        if total >= seconds * SR / 2:
            break
    for _ in range(int(seconds / 2 / 5)):
        b = babble(5 * SR); neg += [b[i:i + N] for i in range(0, len(b) - N + 1, SR // 2)]
    return np.stack(noisy), np.stack(neg).astype(np.float32)


TALK = "data/LibriSpeech/train-clean-100"
DEVICE = bool(int(__import__("os").environ.get("WF_DEVICE_AUG", "0")))


def augmenter(seed=0, pool=400):
    """Head-time augmentation of a 1.5 s window from training-only pools (never the evaluation's): gain,
    speed, room reverberation, and AudioSet noise or LibriSpeech train-clean-100 babble at 0-20 dB. The
    pools are ``pool`` crops read once into memory.

    ``aug(x, key)`` draws every random choice from a generator seeded by ``key`` (and ``seed``), so the same
    key gives the same augmented window in every run: ``cache`` keys a copy by its source file's content hash
    and the copy's index, which is what lets a feature store reuse augmented windows across runs."""
    import glob, soundfile as sf
    rng = random.Random(seed)

    def crops(files, n, k):
        out = []
        for p in rng.sample(files, min(k, len(files))):
            try:
                w, sr = sf.read(p, dtype="float32", always_2d=True)
            except Exception:
                continue
            w = w.mean(1)
            if len(w) < n:
                w = np.tile(w, n // len(w) + 1)
            s = rng.randint(0, len(w) - n); out.append(w[s:s + n])
        return out

    noise = crops(sorted(glob.glob("data/noise-audioset/*.wav")), N, pool)
    talk = crops(sorted(glob.glob(f"{TALK}/**/*.flac", recursive=True)), N, pool)
    rirs = [r / (np.abs(r).max() + 1e-9) for r in crops(sorted(glob.glob("data/rir/**/*.wav", recursive=True)), 8000, 100)]

    def aug(x, key):
        rng = random.Random(int.from_bytes(hashlib.sha1(f"{seed}:{key}".encode()).digest()[:8], "big"))
        y = x
        if rng.random() < 0.5:  # move the word within the window: real audio has it anywhere
            k = rng.randint(-SR * 2 // 5, SR * 2 // 5)
            y = np.roll(y, k)
            if k > 0: y[:k] = 0
            elif k < 0: y[k:] = 0
        if rng.random() < 0.5:  # speed by linear interpolation, kept at N samples
            f = rng.uniform(0.9, 1.1)
            y = np.interp(np.arange(N) * f, np.arange(N), y, right=0.0).astype(np.float32)
        if rirs and rng.random() < (0.6 if DEVICE else 0.3):
            h = rng.choice(rirs); L = N + len(h) - 1
            r = np.fft.irfft(np.fft.rfft(y, L) * np.fft.rfft(h, L), L)[:N]
            y = r * np.sqrt(np.mean(y ** 2) / (np.mean(r ** 2) + 1e-12))
        if rng.random() < 0.8:
            v = sum(rng.choice(talk) for _ in range(rng.randint(1, 3))) if rng.random() < 0.4 else rng.choice(noise)
            snr = rng.uniform(5 if rng.random() < 0.4 else 0, 20)
            pc, pn = np.mean(y ** 2), max(np.mean(v ** 2), 1e-10)
            y = y + v * np.sqrt(pc / (pn * 10 ** (snr / 10)))
        if DEVICE:
            return device(y, rng)
        y = y * 10 ** (rng.uniform(-6, 6) / 20)
        return (y / max(1.0, np.abs(y).max())).astype(np.float32)

    def device(y, rng):
        # what a real device adds: speech after the word, the microphone's band and colour, level-dependent
        # gain and compression, clipping when the input is too hot, and the microphone's own noise floor
        if talk and rng.random() < 0.3:
            v = rng.choice(talk); cut = rng.randint(N // 2, N - 1)
            y = y.copy(); y[cut:] += v[: N - cut] * np.sqrt(np.mean(y ** 2) / (np.mean(v ** 2) + 1e-10)) * rng.uniform(0.3, 1.0)
        F = np.fft.rfft(y); f = np.fft.rfftfreq(N, 1 / 16000)
        if rng.random() < 0.6:
            lo, hi = rng.uniform(60, 300), rng.uniform(3000, 7800)
            F = F / (1 + (lo / np.maximum(f, 1)) ** 4) / (1 + (f / hi) ** 8)
        if rng.random() < 0.5:
            bands = 10 ** (np.array([rng.uniform(-6, 6) for _ in range(8)]) / 20)
            F = F * np.interp(f, np.linspace(0, 8000, 8), bands)
        y = np.fft.irfft(F, N).astype(np.float32)
        y = y / (np.abs(y).max() + 1e-9) * 10 ** (rng.uniform(-30, 0) / 20)
        if rng.random() < 0.3:
            k = rng.uniform(0.4, 0.9); y = np.sign(y) * np.abs(y) ** k
        if rng.random() < 0.3:
            y = y * 10 ** (rng.uniform(6, 24) / 20)
            y = np.tanh(y) if rng.random() < 0.5 else np.clip(y, -1, 1)
        if rng.random() < 0.5:
            n = np.random.default_rng(rng.randint(0, 2 ** 31)).standard_normal(N).astype(np.float32)
            if rng.random() < 0.5:
                n = np.cumsum(n); n -= np.convolve(n, np.ones(64) / 64, "same")
            y = y + n / (np.std(n) + 1e-9) * np.sqrt(np.mean(y ** 2)) * 10 ** (-rng.uniform(25, 50) / 20)
        return np.clip(y, -1, 1).astype(np.float32)
    return aug


def file_sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


CPUINFO = Path("/proc/cpuinfo")


def device_name(provider):
    """Model name of the card or processor an execution provider runs on."""
    if provider == "CUDAExecutionProvider":
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                              text=True, check=True).stdout.splitlines()[0].strip()
    if CPUINFO.exists():
        for line in CPUINFO.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


class FeatureStore:
    """Featurizer outputs keyed by content: sha1 of the featurizer file's sha1, the onnxruntime version, the
    execution provider and the device model the session runs on, and the exact float32 window.

    A window is featurized once per featurizer file and runtime in any run that shares the store (a CUDA and a
    CPU session differ slightly before the float16 cast, so neither serves the other); the values are the
    float16 arrays ``cache`` writes, one ``.npy`` per window under a directory named by the key's first two
    hex characters."""

    def __init__(self, root, featurizer, provider, device):
        import onnxruntime
        self.root = Path(root)
        self.model = f"{file_sha1(featurizer)}:{onnxruntime.__version__}:{provider}:{device}"

    def key(self, wav):
        return hashlib.sha1(self.model.encode() + np.ascontiguousarray(wav, np.float32).tobytes()).hexdigest()

    def path(self, key):
        return self.root / key[:2] / f"{key}.npy"

    def get(self, key):
        p = self.path(key)
        return np.load(p) if p.exists() else None

    def put(self, key, feats):
        p = self.path(key); p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp.npy")
        np.save(tmp, feats.astype(np.float16)); tmp.replace(p)


CALIB_SECONDS = 3600


def resolve_featurizer(arg, revision=None):
    """(onnx path, pretrained name or None, revision or None): a path that exists wins over a pretrained name."""
    if os.path.exists(arg):
        return arg, None, None
    from ww_trainer.pretrained import resolve_pretrained
    path, config = resolve_pretrained(arg, revision)
    return path, arg, config["_revision"]


def cache(a):
    import onnxruntime as ort
    out = Path(a.cache_dir); out.mkdir(parents=True, exist_ok=True)
    onnx_path, fname, frev = resolve_featurizer(a.featurizer, a.featurizer_revision)
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sess = ort.InferenceSession(onnx_path, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    provider = sess.get_providers()[0]
    store = FeatureStore(a.feature_store, onnx_path, provider, device_name(provider)) if a.feature_store else None

    def featurize(wins):
        f = sess.run(None, {name: wins})[0]
        if f.shape[0] != len(wins):
            f = np.concatenate([sess.run(None, {name: w[None]})[0] for w in wins])
        return f.astype(np.float16)

    vf, vl = read_meta(a.val)
    cpos, cneg = calibration_windows(vf, vl, seconds=CALIB_SECONDS)
    splits = [("train", *read_meta(a.metadata), None), ("val", vf, vl, None),
              ("calib", [None] * (len(cpos) + len(cneg)), np.r_[np.ones(len(cpos)), np.zeros(len(cneg))].astype(np.int8),
               np.concatenate([cpos, cneg]))]
    if a.pos_aug:
        # every training positive also as --pos-aug augmented copies, and a quarter of the negatives once;
        # a copy is keyed by its source's content and how many copies of that content came before it
        aug = augmenter()
        tf, tl = read_meta(a.metadata)
        extra = [(f, 1) for f, l in zip(tf, tl) if l == 1 for _ in range(a.pos_aug)]
        extra += [(f, 0) for f, l in zip(tf, tl) if l == 0][::4]
        digests, seen, keyed = {}, {}, []
        for f, _ in extra:
            if f not in digests:
                digests[f] = file_sha1(f)
            h = digests[f]
            keyed.append(("aug", f, f"{h}:{seen.get(h, 0)}")); seen[h] = seen.get(h, 0) + 1
        splits[0] = ("train", list(tf) + keyed, np.r_[tl, [l for _, l in extra]].astype(np.int8), None)
    for split, files, labels, audio in splits:
        feats, hits, misses = None, 0, 0
        for i in range(0, len(files), 64):
            wins = audio[i:i + 64] if audio is not None else np.stack(
                [aug(load_window(f[1]), f[2]) if isinstance(f, tuple) else load_window(f) for f in files[i:i + 64]])
            if store is None:
                f = featurize(wins)
            else:
                keys = [store.key(w) for w in wins]
                got = [store.get(k) for k in keys]
                todo = [j for j, g in enumerate(got) if g is None]
                if todo:
                    for j, v in zip(todo, featurize(wins[todo])):
                        store.put(keys[j], v); got[j] = v
                hits += len(wins) - len(todo); misses += len(todo)
                f = np.stack(got)
            if feats is None:
                feats = np.lib.format.open_memmap(out / f"{split}.npy", "w+", np.float16, (len(files), *f.shape[1:]))
            feats[i:i + len(wins)] = f
        feats.flush(); np.save(out / f"{split}_labels.npy", labels)
        print(split, feats.shape, int(labels.sum()), "positives", flush=True)
        if store is not None:
            print(f"{split} feature store: {hits} hits, {misses} featurized", flush=True)
    (out / "meta.json").write_text(json.dumps({"featurizer": onnx_path, "featurizer_name": fname, "featurizer_revision": frev,
                                                  "featurizer_sha1": file_sha1(onnx_path), "metadata": a.metadata, "val": a.val}) + "\n")
    if a.calib_stream_hours:
        calib_stream(a)



def clip_subset(labels, copies, n, seed):
    """Row indices of ``n`` training positives chosen by clip, each with its ``copies`` augmented rows.

    ``cache`` writes the metadata rows first, positives leading, then every positive's augmented copies as one
    contiguous block in the same order, then a quarter of the negatives. A subset taken by row would keep an
    augmented copy of a clip whose original was dropped, which is a different clip count than the one reported.
    """
    labels = np.asarray(labels)
    n_orig = int(np.argmin(labels == 1)) if labels[0] == 1 else 0
    start = n_orig + int(np.argmax(labels[n_orig:] == 1))
    if labels.sum() != n_orig * (1 + copies) or not labels[start:start + n_orig * copies].all():
        raise SystemExit(f"cache layout does not match {copies} copies of {n_orig} leading positives; "
                         "pass the --pos-aug the cache was built with")
    if n > n_orig:
        raise SystemExit(f"--max-pos {n} exceeds the {n_orig} positive clips in the cache")
    keep = np.random.default_rng(seed).permutation(n_orig)[:n]
    rows = np.concatenate([keep] + [start + keep * copies + t for t in range(copies)])
    return np.sort(rows)


class GRUHead(nn.Module):
    """ww_trainer's GruClassifierHead: GRU, mean over time, ReLU linear, one logit."""

    def __init__(self, dim, hidden=128, linear=128, layers=1, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(dim, hidden, num_layers=layers, batch_first=True, bidirectional=bidirectional)
        self.fc1 = nn.Linear(hidden * (2 if bidirectional else 1), linear)
        self.fc2 = nn.Linear(linear, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc2(F.relu(self.fc1(out.mean(1)))).squeeze(-1)


def make_head(spec, dim):
    """gru | gru-h256 | gru-l2 | bigru, joined by '-' options."""
    kw = {}
    for part in spec.split("-")[1:]:
        if part.startswith("h"): kw["hidden"] = int(part[1:])
        elif part.startswith("l"): kw["layers"] = int(part[1:])
    return GRUHead(dim, bidirectional=spec.startswith("bigru"), **kw)


LEGACY_GRU = re.compile(r"(bi)?gru(-[hl]\d+)*$")
REFUSED_HEADS = {"ocsvm": "a one-class kernel model fitted on embeddings, with no gradient path from a logit",
                 "phonmatch": "its forward pass needs the keyword's phoneme ids besides the features"}
LOSSES = {"bce", "focal", "label_smoothing_bce"}


def split_specs(text):
    """Head specs of a --heads value: commas separate heads, except inside brackets and between the
    ``key=value`` overrides of one ``name:`` spec."""
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch == "," and depth == 0:
            parts.append(cur); cur = ""
            continue
        depth += (ch in "[(") - (ch in "])")
        cur += ch
    parts.append(cur)
    specs = []
    for p in parts:
        if specs and "=" in p and ":" not in p.split("=")[0] and ":" in specs[-1]:
            specs[-1] += "," + p
        else:
            specs.append(p)
    return specs


def head_names():
    from ww_trainer.factory import HEAD_REGISTRY
    return sorted(set(HEAD_REGISTRY) - set(REFUSED_HEADS))


def build_head(spec, dim, frames):
    """A GRU spec builds the bench's own GRUHead; ``name`` or ``name:key=value,...`` builds the factory head of
    that name at feature size ``dim``. The head is run once on a [2, frames, dim] input, so one that cannot
    produce one logit per window of the cache's length fails here."""
    if LEGACY_GRU.match(spec):
        return make_head(spec, dim)
    from ww_trainer.factory import HEAD_REGISTRY
    name, _, opts = spec.partition(":")
    if name in REFUSED_HEADS:
        raise SystemExit(f"head {name!r} is not supported: {REFUSED_HEADS[name]}. Supported: gru-style specs and {', '.join(head_names())}")
    if name not in HEAD_REGISTRY:
        raise SystemExit(f"unknown head {spec!r}. Supported: gru, gru-h256, bigru-l2 style specs and {', '.join(head_names())}")
    cls, valid = HEAD_REGISTRY[name]
    kw = {}
    for item in split_specs_items(opts):
        k, eq, v = item.partition("=")
        if not eq or k not in valid:
            raise SystemExit(f"head {name!r}: bad override {item!r}; keys are {', '.join(sorted(valid))}")
        try:
            kw[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            kw[k] = v
    head = cls(device="cpu", sample_rate=SR, input_size=dim, **kw)
    head.eval()
    with torch.no_grad():
        out = head(torch.zeros(2, frames, dim))
    if tuple(out.shape) != (2,):
        raise SystemExit(f"head {spec!r} gives output of shape {tuple(out.shape)} on [2, {frames}, {dim}], not one logit per window")
    return head


def split_specs_items(opts):
    return [i for i in split_specs(opts) if i] if opts else []


def stem(spec):
    return spec.replace(":", ".").replace(",", "+")


def make_loss(text):
    """``bce``, ``focal``, ``label_smoothing_bce`` or a sum of them, each optionally ``weight*name``."""
    parts = []
    for term in text.split("+"):
        w, _, name = term.rpartition("*")
        if name not in LOSSES:
            raise SystemExit(f"unsupported loss {name!r}; supported: {', '.join(sorted(LOSSES))}, summed with '+' and weighted as 'weight*name'. "
                             "Metric losses need a head embedding and pairing that this bench does not provide")
        parts.append((name, float(w) if w else 1.0))
    if parts == [("bce", 1.0)]:
        return F.binary_cross_entropy_with_logits
    from ww_trainer.loss import LossManager
    crits = [(w, LossManager([{"name": n}]).losses[0]["criterion"]) for n, w in parts]
    return lambda logits, y: sum(w * c(logits, y) for w, c in crits)


def lr_scheduler(opt, schedule, epochs, warmup):
    """None for a constant rate; for ``cosine``, ``warmup`` epochs of linear ramp below the full rate, the peak
    at the epoch after them, then cosine annealing to lr/100 on the last epoch. A run whose peak epoch is its
    last (``warmup`` = ``epochs`` - 1) never anneals. Stepped once per epoch, after it."""
    if schedule == "constant":
        if warmup:
            raise SystemExit("--warmup-epochs applies to --lr-schedule cosine")
        return None
    if not 0 <= warmup < epochs:
        raise SystemExit(f"--warmup-epochs {warmup} must be below --epochs {epochs}")
    from torch.optim.lr_scheduler import LambdaLR
    floor, span = 0.01, epochs - 1 - warmup

    def multiplier(e):
        if e < warmup:
            return (e + 1) / (warmup + 1)
        if not span:
            return 1.0
        return floor + (1 - floor) * (1 + math.cos(math.pi * (e - warmup) / span)) / 2

    return LambdaLR(opt, multiplier)


def fit(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = Path(a.cache_dir)
    X = torch.from_numpy(np.load(c / "train.npy", mmap_mode="r")[:]).to(dev)
    y = torch.from_numpy(np.load(c / "train_labels.npy").astype(np.float32)).to(dev)
    Xv = torch.from_numpy(np.load(c / "val.npy")[:]).to(dev)
    yv = torch.from_numpy(np.load(c / "val_labels.npy").astype(np.float32)).to(dev)
    Xc = torch.from_numpy(np.load(c / "calib.npy")[:]).to(dev)
    yc = torch.from_numpy(np.load(c / "calib_labels.npy").astype(np.float32)).to(dev)
    pos, neg = torch.nonzero(y == 1).squeeze(1), torch.nonzero(y == 0).squeeze(1)
    if a.max_pos:
        pos = torch.from_numpy(clip_subset(y.cpu().numpy(), a.pos_aug, a.max_pos, a.subset_seed)).to(dev)
        print(f"training on {a.max_pos} positive clips, {len(pos)} rows with their augmented copies", flush=True)
    g = torch.Generator(device="cpu").manual_seed(a.seed); torch.manual_seed(a.seed)
    specs = split_specs(a.heads)
    loss_fn = make_loss(a.loss)
    heads = {s: build_head(s, X.shape[-1], X.shape[1]).to(dev) for s in specs}
    batch_stats = {s for s in specs if not LEGACY_GRU.match(s)}
    opts = {s: torch.optim.Adam(h.parameters(), lr=a.lr) for s, h in heads.items()}
    scheds = [lr_scheduler(o, a.lr_schedule, a.epochs, a.warmup_epochs) for o in opts.values()]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    best = {s: (-1.0, 0.0) for s in specs}
    # five negatives per positive, half of them hard; capped by the negatives there are
    n_neg = min(5 * len(pos), len(neg)); n_hard = n_neg // 2
    hard = neg[torch.randperm(len(neg), generator=g)[:n_hard].to(dev)]
    log = open(out / "fit.jsonl", "a")
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        rand = neg[torch.randperm(len(neg), generator=g)[: n_neg - len(hard)].to(dev)]
        idx = torch.cat([pos, hard, rand]); idx = idx[torch.randperm(len(idx), generator=g).to(dev)]
        for h in heads.values(): h.train()
        for i in range(0, len(idx), a.batch_size):
            b = idx[i:i + a.batch_size]
            xb, yb = X[b].float(), y[b]
            for s, h in heads.items():
                if len(b) < 2 and s in batch_stats:
                    continue
                loss = loss_fn(h(xb), yb)
                opts[s].zero_grad(set_to_none=True); loss.backward(); opts[s].step()
        rec = {"epoch": ep, "seconds": round(time.time() - t0, 1), "lr": opts[specs[0]].param_groups[0]["lr"]}
        with torch.no_grad():
            for h in heads.values(): h.eval()
            # hard negatives for the next epoch: the negatives the first head scores highest
            first = heads[specs[0]]
            sc = torch.cat([first(X[neg[i:i + 1024]].float()) for i in range(0, len(neg), 1024)])
            hard = neg[torch.topk(sc, n_hard).indices]
            for s, h in heads.items():
                lv = torch.cat([h(Xv[i:i + 1024].float()) for i in range(0, len(Xv), 1024)])
                vl = F.binary_cross_entropy_with_logits(lv, yv).item()
                p = torch.sigmoid(lv)
                lc = torch.cat([h(Xc[i:i + 1024].float()) for i in range(0, len(Xc), 1024)])
                cal = (lc[yc == 1] > lc[yc == 0].max()).float().mean().item()
                rec[s] = {"val_loss": round(vl, 4), "val_recall@0.5": round(((p >= 0.5) & (yv == 1)).sum().item() / max(1, (yv == 1).sum().item()), 4),
                          "val_fp@0.5": int(((p >= 0.5) & (yv == 0)).sum().item()), "calib_recall": round(cal, 4)}
                if (cal, -vl) > best[s]:
                    best[s] = (cal, -vl); torch.save({"spec": s, "dim": X.shape[-1], "frames": X.shape[1], "loss": a.loss, "state": h.state_dict(), "epoch": ep, "calib_recall": cal}, out / f"{stem(s)}.pt")
        log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
        for sch in scheds:
            if sch is not None:
                sch.step()


BLOCK_SECONDS = 0.08
DEBOUNCE_BLOCKS = 25
STREAM_SEGMENT = 20 * SR
CALIB_POS_PROB = 0.9
CALIB_SCALE_RANGE = (1e-3, 1e3)
MIN_EVENTS = 20


OPENSLR = "https://www.openslr.org/resources/12/"


def refuse_overlap(sources, metadata, talk):
    """Calibration speech must be audio no head trains on: neither the augmentation talker pool nor a directory
    holding a file of the training metadata."""
    train = {Path(f).resolve() for f in read_meta(metadata)[0]} if metadata else set()
    pool = Path(talk).resolve()
    for src in (Path(d).resolve() for d in sources):
        if src == pool or pool in src.parents or src in pool.parents:
            raise SystemExit(f"calibration source {src} overlaps the augmentation pool {pool}")
        hit = next((f for f in sorted(train) if src in f.parents), None)
        if hit:
            raise SystemExit(f"calibration source {src} holds the training file {hit}")


def calibration_stream_audio(hours, sources, seed=1):
    """20 s segments cycling through continuous speech from ``sources``, three-talker babble of it and AudioSet
    noise: ``hours`` of audio in all."""
    import glob, soundfile as sf
    rng = random.Random(seed)
    speech = []
    for d in sources:
        found = sorted(glob.glob(f"{d}/**/*.flac", recursive=True))
        if not found:
            raise SystemExit(f"no .flac under {d} (LibriSpeech splits are at {OPENSLR}<split>.tar.gz)")
        speech += found
    noise = sorted(glob.glob("data/noise-audioset/*.wav"))
    n = STREAM_SEGMENT

    def clip(p):
        w, _ = sf.read(p, dtype="float32", always_2d=True); w = w.mean(1)
        if len(w) < n:
            w = np.tile(w, n // len(w) + 1)
        s = rng.randint(0, len(w) - n); return w[s:s + n]

    def talk():
        out = []
        while sum(map(len, out)) < n:
            w, _ = sf.read(rng.choice(speech), dtype="float32", always_2d=True); out.append(w.mean(1))
        return np.concatenate(out)[:n]

    makers = [talk, lambda: sum(clip(rng.choice(speech)) for _ in range(3)) / 3, lambda: clip(rng.choice(noise))]
    for i in range(math.ceil(hours * 3600 * SR / n)):
        yield makers[i % 3]().astype(np.float32)


def calib_stream(a):
    """Featurize ``--calib-stream-hours`` of continuous negative audio into ``calib_stream.npy`` ([frames, dim],
    float16, one stream), the audio a head's detections are counted on."""
    import onnxruntime as ort
    out = Path(a.cache_dir); out.mkdir(parents=True, exist_ok=True)
    refuse_overlap(a.calib_stream_speech, a.metadata, a.aug_talk)
    onnx_path, _, _ = resolve_featurizer(a.featurizer, a.featurizer_revision)
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    sess = ort.InferenceSession(onnx_path, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    segments, batch = [], []

    def flush():
        wins = np.stack(batch)
        f = sess.run(None, {name: wins})[0]
        if f.shape[0] != len(wins):
            f = np.concatenate([sess.run(None, {name: w[None]})[0] for w in wins])
        segments.extend(f.astype(np.float16)); batch.clear()

    for w in calibration_stream_audio(a.calib_stream_hours, a.calib_stream_speech):
        batch.append(w)
        if len(batch) == 4:
            flush()
    if batch:
        flush()
    stream = np.concatenate(segments)
    np.save(out / "calib_stream.npy", stream)
    print("calib_stream", stream.shape, flush=True)


def debounced_events(z, thr):
    """Detections the plugin fires on per-block logits ``z``: the first block at or above ``thr``, then
    ``DEBOUNCE_BLOCKS`` blocks that are not scored."""
    above = np.flatnonzero(np.asarray(z) >= thr)
    n, j = 0, 0
    while j < len(above):
        n += 1
        j = int(np.searchsorted(above, above[j] + DEBOUNCE_BLOCKS + 1))
    return n


def threshold_for_events(z, events):
    lo, hi = float(np.min(z)), float(np.max(z)) + 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if debounced_events(z, mid) > events:
            lo = mid
        else:
            hi = mid
    return hi


def stream_threshold(z, hours, target_fa_per_hour):
    """Logit at which the plugin's detections on the stream come to ``target_fa_per_hour``. A target below
    ``MIN_EVENTS`` detections in the audio is too thin a tail to read directly: the tail is taken as exponential
    through the thresholds giving ``MIN_EVENTS`` and three times as many detections, and extended to the target."""
    want = target_fa_per_hour * hours
    if want >= MIN_EVENTS:
        return threshold_for_events(z, round(want)), False
    hi, lo = threshold_for_events(z, MIN_EVENTS), threshold_for_events(z, 3 * MIN_EVENTS)
    if hi <= lo:
        raise SystemExit(f"calibration stream of {hours:.2f} h has too few detections to read a tail from")
    return hi + math.log(MIN_EVENTS / want) * (hi - lo) / math.log(3), True


F2_PROBABILITIES = np.arange(5, 100) / 100
F2_RANGE = (0.5, 0.7)


def roc_auc(pos, neg):
    """Probability that a positive scores above a negative, a tie counting half."""
    neg = np.sort(np.asarray(neg, np.float64))
    pos = np.asarray(pos, np.float64)
    below = np.searchsorted(neg, pos, "left") + np.searchsorted(neg, pos, "right")
    return float(below.sum() / (2 * len(pos) * len(neg)))


def f2_operating_point(pos, stream, hours, positives_per_hour):
    """The threshold in ``F2_PROBABILITIES`` with the best F2 on calibrated logits: recall is the share of ``pos``
    at or above it, and precision weighs the recall of ``positives_per_hour * hours`` wake words against the
    debounced false activations on ``stream``. Among equal F2 the highest threshold wins."""
    pos = np.asarray(pos, np.float64)
    wanted = positives_per_hour * hours
    best = None
    for prob in F2_PROBABILITIES:
        thr = math.log(prob / (1 - prob))
        recall = float((pos >= thr).mean())
        hits = recall * wanted
        precision = hits / (hits + debounced_events(stream, thr)) if hits else 0.0
        f2 = 5 * precision * recall / (4 * precision + recall) if hits else 0.0
        if best is None or f2 >= best[0]:
            best = (f2, float(prob), recall)
    return {"f2": best[0], "f2_threshold": best[1], "f2_recall": best[2]}


def calibration_verdict(f2_threshold):
    if F2_RANGE[0] <= f2_threshold <= F2_RANGE[1]:
        return f"calibration good: the F2-optimal threshold {f2_threshold:.2f} is within {F2_RANGE[0]} to {F2_RANGE[1]}"
    return f"warning: the F2-optimal threshold {f2_threshold:.2f} is outside {F2_RANGE[0]} to {F2_RANGE[1]}; the calibration is not good enough"


def fit_calibration(pos_logits, stream_logits, hours, target_fa_per_hour, positives_per_hour=4.0, pos_prob=CALIB_POS_PROB):
    """(a, b, info) with a > 0 for ``z' = a * z + b``: the stream's threshold for ``target_fa_per_hour`` maps to 0,
    probability 0.5, and the median positive maps to ``pos_prob``. ``info`` carries what the metadata records,
    including the F2-optimal threshold and the AUC on the calibrated scores."""
    z_thr, extrapolated = stream_threshold(stream_logits, hours, target_fa_per_hour)
    gap = float(np.median(pos_logits)) - z_thr
    if gap <= 0:
        raise SystemExit(f"median calibration positive ({np.median(pos_logits):.3f}) does not exceed the logit "
                         f"({z_thr:.3f}) giving {target_fa_per_hour} detections per hour; nothing to calibrate")
    a = float(np.clip(math.log(pos_prob / (1 - pos_prob)) / gap, *CALIB_SCALE_RANGE))
    shift = float(-a * z_thr)
    info = {"calib_hours": hours, "calib_events": debounced_events(stream_logits, z_thr), "calib_extrapolated": extrapolated,
            "roc_auc": roc_auc(pos_logits, stream_logits), "positives_per_hour": positives_per_hour,
            **f2_operating_point(a * np.asarray(pos_logits) + shift, a * np.asarray(stream_logits) + shift, hours, positives_per_hour)}
    return a, shift, info


def stream_logits(h, stream, window, block):
    """The head's logit at every ``block`` frames of ``stream``, over the last ``window`` frames."""
    x = torch.from_numpy(stream).unfold(0, window, block).permute(0, 2, 1)
    with torch.no_grad():
        return torch.cat([h(x[i:i + 1024].float()) for i in range(0, len(x), 1024)]).numpy().astype(np.float64)


class Affine(nn.Module):
    def __init__(self, head, scale, shift):
        super().__init__()
        self.head, self.scale, self.shift = head, scale, shift

    def forward(self, x):
        return self.scale * self.head(x) + self.shift


def calibration(h, a):
    c = Path(a.cache_dir)
    x = torch.from_numpy(np.load(c / "calib.npy")[:])
    y = np.load(c / "calib_labels.npy")
    with torch.no_grad():
        z = torch.cat([h(x[i:i + 1024].float()) for i in range(0, len(x), 1024)]).numpy()
    stream = np.load(c / "calib_stream.npy")
    window = x.shape[1]
    fps = window / (N / SR)
    zs = stream_logits(h, stream, window, round(BLOCK_SECONDS * fps))
    return fit_calibration(z[y == 1], zs, len(zs) * BLOCK_SECONDS / 3600, a.target_fa_per_hour, a.positives_per_hour)


def export(a):
    import io
    import onnxruntime as ort
    from ww_trainer.utils import embed_onnx_metadata
    ck = torch.load(Path(a.out_dir) / f"{stem(a.head)}.pt", map_location="cpu")
    frames = ck.get("frames", 75)
    h = build_head(ck["spec"], ck["dim"], frames); h.load_state_dict(ck["state"]); h.eval()
    path = Path(a.out_dir) / f"{stem(a.head)}.onnx"
    if a.calibrate:
        if not a.cache_dir:
            raise SystemExit("--calibrate needs --cache-dir")
        scale, shift, info = calibration(h, a)
        h = Affine(h, scale, shift)
    gen = torch.Generator().manual_seed(0)

    def write(dynamic):
        buf = io.BytesIO()
        axes = {"features": {0: "batch", 1: "frames"} if dynamic else {0: "batch"}, "logit": {0: "batch"}}
        torch.onnx.export(h, torch.randn(1, frames, ck["dim"]), buf, input_names=["features"], output_names=["logit"],
                          dynamic_axes=axes, opset_version=17, dynamo=False)
        return buf.getvalue()

    def diff(model, t):
        x = torch.randn(3, t, ck["dim"], generator=gen)
        got = ort.InferenceSession(model, providers=["CPUExecutionProvider"]).run(None, {"features": x.numpy()})[0]
        with torch.no_grad():
            return float(np.abs(got - h(x).numpy()).max())

    model = write(True)
    try:
        dynamic = diff(model, frames + 25) < 1e-4
    except Exception:
        dynamic = False
    if not dynamic:
        if not a.allow_fixed_frames:
            raise SystemExit(f"head {a.head!r} exports with a fixed {frames}-frame input: onnxruntime fails at any other length. "
                             "Pass --allow-fixed-frames to export it anyway")
        model = write(False)
    err = diff(model, frames)
    if not err < 1e-4:
        raise SystemExit(f"{path}: onnxruntime differs from torch by {err} on random input")
    path.write_bytes(model)
    meta = {} if dynamic else {"fixed_frames": str(frames)}
    if a.calibrate:
        meta.update({"calibrated": "1", "calib_a": repr(scale), "calib_b": repr(shift), "default_threshold": repr(info["f2_threshold"]),
                     "target_fa_per_hour": repr(a.target_fa_per_hour), **{k: repr(v) for k, v in info.items()}})
    if meta:
        embed_onnx_metadata(str(path), meta)
    print(path, f"max abs diff {err:.2e}", "frames dynamic" if dynamic else f"fixed_frames {frames}")
    if a.calibrate:
        print(f"roc_auc {info['roc_auc']:.4f} f2 {info['f2']:.4f} at {info['f2_threshold']:.2f} (recall {info['f2_recall']:.3f}); "
              + calibration_verdict(info["f2_threshold"]))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cache"); p.add_argument("featurizer", help="ONNX file, or a pretrained name such as wakehubert-tiny")
    p.add_argument("--featurizer-revision", default=None, help="Hub revision of a pretrained featurizer"); p.add_argument("cache_dir")
    p.add_argument("--metadata", default="kws/train.csv"); p.add_argument("--val", default="kws/val.csv")
    p.add_argument("--pos-aug", type=int, default=8, help="augmented copies of every training positive (0: none)")
    p.add_argument("--path-map", action="append", default=[], help="old=new prefix rewrite for metadata paths")
    p.add_argument("--aug-talk", default="data/LibriSpeech/train-clean-100", help="speech whose talkers become head-time babble")
    p.add_argument("--calib-speech", default="data/LibriSpeech/dev-other", help="LibriSpeech split used for calibration speech and babble")
    p.add_argument("--feature-store", default=None, help="directory of featurizer outputs keyed by window content, reused across runs")
    p.add_argument("--calib-stream-hours", type=float, default=10.0, help="hours of continuous negative audio featurized for export --calibrate (0: none)")
    p.add_argument("--calib-stream-speech", action="append", default=[], help="LibriSpeech split for the calibration stream, repeatable (default dev-clean and dev-other)")
    p = sub.add_parser("calib-stream"); p.add_argument("featurizer", help="ONNX file, or a pretrained name"); p.add_argument("cache_dir")
    p.add_argument("--featurizer-revision", default=None)
    p.add_argument("--calib-stream-hours", type=float, default=10.0)
    p.add_argument("--calib-stream-speech", action="append", default=[])
    p.add_argument("--metadata", default=None, help="training metadata whose files the calibration speech must not contain")
    p.add_argument("--aug-talk", default="data/LibriSpeech/train-clean-100")
    p = sub.add_parser("fit"); p.add_argument("cache_dir"); p.add_argument("out_dir")
    p.add_argument("--heads", default="gru", help="gru-style specs and HEAD_REGISTRY names, name:key=value,... for overrides")
    p.add_argument("--loss", default="bce", help="bce, focal, label_smoothing_bce, or a sum such as bce+0.5*focal")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--lr", type=float, default=5e-4); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-pos", type=int, default=0, help="train on this many positive clips, chosen by clip (0: all)")
    p.add_argument("--pos-aug", type=int, default=0, help="augmented copies per positive the cache was built with (needed by --max-pos)")
    p.add_argument("--subset-seed", type=int, default=0, help="which clips --max-pos keeps")
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                   help="constant: --lr every epoch; cosine: cosine annealing to lr/100 after the warmup")
    p.add_argument("--warmup-epochs", type=int, default=0, help="epochs of linear warmup before the cosine schedule")
    p = sub.add_parser("export"); p.add_argument("out_dir"); p.add_argument("head")
    p.add_argument("--allow-fixed-frames", action="store_true", help="export a head whose graph only runs at the cache's frame count, recording it as fixed_frames metadata")
    p.add_argument("--calibrate", action="store_true", help="put an affine logit map fitted on the cache's calibration positives and negative stream before the output")
    p.add_argument("--cache-dir", default=None, help="cache whose calib.npy and calib_stream.npy --calibrate reads")
    p.add_argument("--target-fa-per-hour", type=float, default=1.0, help="detections per hour on the calibration stream that map to probability 0.5, ")
    p.add_argument("--positives-per-hour", type=float, default=4.0, help="real wake words per hour of audio, which weights recall against false activations in the F2 choice")
    a = ap.parse_args()
    if a.cmd in ("cache", "calib-stream") and not a.calib_stream_speech:
        a.calib_stream_speech = ["data/LibriSpeech/dev-clean", "data/LibriSpeech/dev-other"]
    global CALIB_SPEECH, TALK
    if a.cmd == "cache":
        PATH_MAP.extend(tuple(m.split("=", 1)) for m in a.path_map)
        CALIB_SPEECH, TALK = a.calib_speech, a.aug_talk
    {"cache": cache, "calib-stream": calib_stream, "fit": fit, "export": export}[a.cmd](a)


if __name__ == "__main__":
    main()
