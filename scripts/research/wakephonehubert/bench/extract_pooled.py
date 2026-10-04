"""extract_pooled.py <task> --feats logmel,wakehubert,...

Mean pooling followed by a linear layer commutes with the weighted sum, the per-entry standardisation and the entry
projections (all affine), so storing each entry's mean over valid frames and training the weighted sum + linear on those
vectors is the same model as SUPERB's frame-level weighted sum, mean pooling and linear head, at a fraction of the cost.

task esc50  data/esc50/ESC-50-master.zip, read in place (44.1 kHz wav, resampled to 16 kHz), folds 1-5
task lid    data/voxlingua107/train/<lang>-<shard>.tar (train; 10% of videos per language held out as dev),
            data/voxlingua107/dev.tar restricted to the same ten languages (test)
task sid    VoxCeleb1 streamed from the Hub zip (ProgramComputer/voxceleb vox1/vox1_dev_wav.zip + vox1_test_wav.zip; stored or deflated members),
            never stored; official iden split (2 dev, 3 test) with the training side cut to 8 utterances per speaker. Resumable at file granularity.
Output: feats/<task>/<feat>/<entry>.npy float16 [N, d], feats/<task>/meta.json (keys, labels, split), done mask per feat.
"""
import argparse
import csv
import hashlib
import io
import shutil
import json
import struct
import sys
import tarfile
import time
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import featlib  # noqa: E402
from featlib import SR, Featurizer  # noqa: E402

B = Path("work/bench")
D = B / "data"
LID_LANGS = ["sv", "nl", "hy", "fr", "fa", "ar", "da", "lv", "fi", "et"]
VOX = "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/"
BATCH_SECONDS = 60


def to16k(x, sr):
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g)
    return np.ascontiguousarray(x, dtype=np.float32)


ESC_ZIP = D / "esc50/ESC-50-master.zip"


def set_dirs(b):
    global B, D, ESC_ZIP
    B = Path(b)
    D = B / "data"
    ESC_ZIP = D / "esc50/ESC-50-master.zip"


def meta_esc50():
    import zipfile
    with zipfile.ZipFile(ESC_ZIP) as z:
        rows = list(csv.DictReader(io.TextIOWrapper(z.open("ESC-50-master/meta/esc50.csv"))))
    return [{"key": r["filename"], "label": int(r["target"]), "fold": int(r["fold"]), "member": "ESC-50-master/audio/" + r["filename"]} for r in rows]


def video_dev(key):
    vid = key.split("__U__")[0]
    return int(hashlib.md5(vid.encode()).hexdigest(), 16) % 10 == 0


def meta_lid():
    rows = []
    for lang in LID_LANGS:
        for t in sorted((D / "voxlingua107/train").glob(f"{lang}-*.tar")):
            with tarfile.open(t) as tf:
                for m in tf.getmembers():
                    if m.name.endswith(".wav"):
                        k = m.name[:-4]
                        rows.append({"key": k, "label": LID_LANGS.index(lang), "split": "dev" if video_dev(k) else "train", "tar": str(t), "member": m.name})
    with tarfile.open(D / "voxlingua107/dev.tar") as tf:
        for m in tf.getmembers():
            lang = m.name.split("-")[0]
            if m.name.endswith(".wav") and lang in LID_LANGS:
                rows.append({"key": m.name[:-4], "label": LID_LANGS.index(lang), "split": "test", "tar": str(D / "voxlingua107/dev.tar"), "member": m.name})
    return rows


SID_TRAIN_PER_SPEAKER = 8


def meta_sid():
    """Subset of the official identification split: all 1,251 speakers; training side reduced to a fixed random sample of
    8 utterances per speaker (numpy default_rng(0) over each speaker's sorted training keys); the official dev and test
    utterances kept whole."""
    rows, spk, train = [], {}, {}
    for line in open(D / "voxceleb1/iden_split.txt"):
        s, path = line.split()
        sp = path.split("/")[0]
        spk.setdefault(sp, len(spk))
        if s == "1":
            train.setdefault(sp, []).append(path)
        else:
            rows.append({"key": path, "label": spk[sp], "split": {"2": "dev", "3": "test"}[s]})
    rng = np.random.default_rng(0)
    for sp in sorted(train):
        keys = sorted(train[sp])
        for k in sorted(rng.choice(keys, min(SID_TRAIN_PER_SPEAKER, len(keys)), replace=False)):
            rows.append({"key": str(k), "label": spk[sp], "split": "train"})
    return rows


def audio_files(task, rows, done):
    if task == "esc50":
        import zipfile
        with zipfile.ZipFile(ESC_ZIP) as z:
            for i, r in enumerate(rows):
                if not done[i]:
                    x, sr = sf.read(io.BytesIO(z.read(r["member"])), dtype="float32")
                    yield i, to16k(x, sr)
    elif task == "lid":
        by_tar = {}
        for i, r in enumerate(rows):
            if not done[i]:
                by_tar.setdefault(r["tar"], []).append(i)
        for t, idx in by_tar.items():
            with tarfile.open(t) as tf:
                for i in idx:
                    x, sr = sf.read(io.BytesIO(tf.extractfile(rows[i]["member"]).read()), dtype="float32")
                    yield i, to16k(x, sr)
    elif task == "sid":
        index = {r["key"]: i for i, r in enumerate(rows)}
        yield from vox_stream(index, done)


class HTTPStream:
    def __init__(self, url, start):
        import requests
        self.r = requests.get(url, headers={"Range": f"bytes={start}-"}, stream=True, timeout=120)
        self.r.raise_for_status()
        self.raw, self.pos = self.r.raw, start

    def read(self, n):
        out = bytearray()
        while len(out) < n:
            b = self.raw.read(n - len(out))
            if not b:
                break
            out += b
        self.pos += len(out)
        return bytes(out)


def vox_stream(index, done):
    state_f = B / "feats/sid/stream_state.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    for z in ("vox1_dev_wav.zip", "vox1_test_wav.zip"):
        if state.get(z) == "done":
            continue
        start = state.get(z, 0)
        for attempt in range(20):
            try:
                s = HTTPStream(VOX + z, start)
                while True:
                    hdr = s.read(30)
                    if len(hdr) < 30 or hdr[:4] != b"PK\x03\x04":
                        break
                    _, _, flag, method, _, _, _, csize, usize, nlen, xlen = struct.unpack("<IHHHHHIIIHH", hdr)
                    name = s.read(nlen).decode()
                    extra = s.read(xlen)
                    if csize == 0xFFFFFFFF:
                        p = 0
                        while p < len(extra):
                            hid, hl = struct.unpack("<HH", extra[p:p + 4])
                            if hid == 1:
                                usize, csize = struct.unpack("<QQ", extra[p + 4:p + 20])
                            p += 4 + hl
                    assert method in (0, 8) and not flag & 8, (name, method, flag)
                    data = s.read(csize)
                    start = s.pos
                    key = name[4:] if name.startswith("wav/") else name
                    if name.endswith(".wav") and key in index and not done[index[key]]:
                        if method == 8:
                            data = zlib.decompress(data, -15)
                        x, sr = sf.read(io.BytesIO(data), dtype="float32")
                        yield index[key], to16k(x, sr), (z, start)
                break
            except Exception as e:
                print(json.dumps({"stream_error": repr(e)[:300], "zip": z, "offset": start, "attempt": attempt}), flush=True)
                time.sleep(30)
        else:
            raise SystemExit(f"{z}: stream failed 20 times at {start}")
        state[z] = "done"
        state_f.write_text(json.dumps(state))


def main():
    ap = argparse.ArgumentParser()
    featlib.add_args(ap)
    ap.add_argument("task", choices=["esc50", "lid", "sid"]); ap.add_argument("--feats", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    set_dirs(featlib.configure(a))
    torch.set_num_threads(a.threads)
    torch.cuda.set_per_process_memory_fraction(6.5 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    out = B / "feats" / a.task
    out.mkdir(parents=True, exist_ok=True)
    mf = out / "meta.json"
    if mf.exists():
        rows = json.loads(mf.read_text())["rows"]
    else:
        rows = {"esc50": meta_esc50, "lid": meta_lid, "sid": meta_sid}[a.task]()
        mf.write_text(json.dumps({"rows": rows}))
    N = len(rows)
    feats = [Featurizer(f) for f in a.feats.split(",")]
    stores, dones = {}, {}
    for f in feats:
        fd = out / f.name
        fd.mkdir(exist_ok=True)
        stores[f.name] = {k: np.lib.format.open_memmap(fd / f"{k}.npy", mode="r+" if (fd / f"{k}.npy").exists() else "w+", dtype=np.float16, shape=(N, d))
                          for k, d in f.dims.items()}
        dp = fd / "done.npy"
        dones[f.name] = np.load(dp) if dp.exists() else np.zeros(N, bool)
        (fd / "info.json").write_text(json.dumps({"dims": f.dims, "width": f.width, "heads_from": getattr(f, "heads_from", None)}))
    done_all = np.logical_and.reduce([dones[f.name] for f in feats])
    print(json.dumps({"task": a.task, "rows": N, "todo": int((~done_all).sum())}), flush=True)
    t0, n, secs, last = time.time(), 0, 0.0, None

    def flush(batch):
        wavs = [w for _, w in batch]
        idx = [i for i, _ in batch]
        for f in feats:
            todo = [j for j, i in enumerate(idx) if not dones[f.name][i]]
            if not todo:
                continue
            p = f.pooled([wavs[j] for j in todo])
            for k, v in p.items():
                stores[f.name][k][[idx[j] for j in todo]] = v.astype(np.float16)
            dones[f.name][[idx[j] for j in todo]] = True
            torch.cuda.empty_cache()

    def save(pos=None):
        free = shutil.disk_usage("/").free / 2**30
        if free < 22:
            for f in feats:
                for m in stores[f.name].values():
                    m.flush()
                np.save(out / f.name / "done.npy", dones[f.name])
            raise SystemExit(f"STOP: {free:.1f} GB free on /, under the 22 GB floor; progress saved, rerun to resume")
        for f in feats:
            for m in stores[f.name].values():
                m.flush()
            np.save(out / f.name / "done.npy", dones[f.name])
        if pos is not None:
            sf_ = out / "stream_state.json"
            st = json.loads(sf_.read_text()) if sf_.exists() else {}
            st[pos[0]] = pos[1]
            sf_.write_text(json.dumps(st))

    batch, bsec = [], 0.0
    for item in audio_files(a.task, rows, done_all):
        i, w = item[0], item[1]
        pos = item[2] if len(item) > 2 else None
        batch.append((i, w)); bsec += len(w) / SR
        if bsec >= BATCH_SECONDS or len(batch) >= 32:
            flush(batch); n += len(batch); secs += bsec
            batch, bsec = [], 0.0
            if n % 256 < 32:
                save(pos)
            if n % 1024 < 32:
                el = time.time() - t0
                print(json.dumps({"done": n, "audio_h": round(secs / 3600, 2), "x_realtime": round(secs / max(el, 1), 1), "elapsed_s": round(el),
                                  "gpu_mb": torch.cuda.max_memory_allocated() // 2**20}), flush=True)
    if batch:
        flush(batch); n += len(batch)
    save()
    print(json.dumps({"EXTRACT_DONE": a.task, "rows": n, "missing": {f.name: int((~dones[f.name]).sum()) for f in feats}}), flush=True)


if __name__ == "__main__":
    main()
