"""build_pool.py <out_dir> <source> [options]

Builds the distillation pool: AudioSet unbalanced slice, MSWC and FMA tracks with a commercial-use licence.

Sources, each streamed without staging archives on disk and written as fixed-length 16 kHz int16 segments:
  audioset  --shards N      unbalanced-train parquet shards of agkphysics/AudioSet (10 s clips, labels kept), or the local
                            parquet files of --parquet-glob; --name sets the source name (default audioset)
  mswc      --per-lang N    MLCommons/ml_spoken_words opus, every language (or --langs), N random valid clips each (1 s) of
                            --split (train by default), streamed or read from a local copy (--local-dir)
  fma       --hours H       fma_full tracks whose licence allows commercial use and derivatives (CC BY, CC BY-SA,
                            CC0, public domain, Free Art), read by HTTP range from the zip, cut into 10 s segments
Outputs per source: <source>_audio.raw (int16, [n, seg]) + <source>_meta.jsonl (one row per segment: source id,
labels or word/lang, licence) + <source>.done with the count.
"""
import argparse
import io
import json
import random
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SR = 16000


def to16k(b, fmt_hint=None):
    import soundfile as sf
    try:
        w, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
        w = w.mean(1)
    except Exception:
        r = subprocess.run(["ffmpeg", "-v", "quiet", "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"], input=b, capture_output=True)
        if r.returncode or not r.stdout:
            return None
        return np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768
    if sr != SR:
        import torch, torchaudio
        torch.set_num_threads(1)
        w = torchaudio.functional.resample(torch.from_numpy(w), sr, SR).numpy()
    return w


def fix(w, n):
    w = np.pad(w, (0, max(0, n - len(w))))[:n]
    return (np.clip(w, -1, 1) * 32767).astype(np.int16)


def _as_one(b):
    w = to16k(b)
    return None if w is None else fix(w, 10 * SR)


def audioset(a, out):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, HfApi
    import glob
    if a.parquet_glob:
        pick = sorted(glob.glob(a.parquet_glob))
    else:
        files = sorted(f for f in HfApi().list_repo_files("agkphysics/AudioSet", repo_type="dataset") if f.startswith("data/unbal_train/"))
        rng = random.Random(0); pick = sorted(rng.sample(files, min(a.shards, len(files))))
    fs = HfFileSystem()
    name = a.name if a.name != "mswc" else "audioset"
    raw = open(out / f"{name}_audio.raw", "ab"); meta = open(out / f"{name}_meta.jsonl", "a"); n = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for f in pick:
            with (open(f, "rb") if a.parquet_glob else fs.open(f"datasets/agkphysics/AudioSet/{f}", "rb", block_size=64 << 20)) as fh:
                pf = pq.ParquetFile(fh)
                for g in range(pf.num_row_groups):
                    rows = pf.read_row_group(g, columns=["video_id", "audio", "labels"]).to_pylist()
                    for r, w in zip(rows, ex.map(_as_one, [r["audio"]["bytes"] for r in rows], chunksize=4)):
                        if w is None:
                            continue
                        raw.write(w.tobytes()); n += 1
                        meta.write(json.dumps({"id": r["video_id"], "labels": r["labels"], "licence": "AudioSet (YouTube; CC BY 4.0 labels)"}) + "\n")
            print("audioset", f, n, flush=True)
    raw.close(); meta.close(); (out / f"{name}.done").write_text(str(n))


def _mswc_member(args):
    b, = args
    w = to16k(b)
    return None if w is None else fix(w, SR)


def _open(fs, base, local_dir, f):
    """A repo file: from --local-dir when given (files fetched beforehand, e.g. by mswc_fetch.py), else streamed."""
    return open(Path(local_dir) / f, "rb") if local_dir else fs.open(base + f, "rb", block_size=64 << 20)


def mswc(a, out):
    import csv
    from huggingface_hub import HfFileSystem, HfApi
    files = HfApi().list_repo_files("MLCommons/ml_spoken_words", repo_type="dataset")
    langs = sorted({f.split("/")[2] for f in files if f.startswith("data/opus/")})
    fs = HfFileSystem(); base = "datasets/MLCommons/ml_spoken_words/"
    have = {}
    if (out / f"{a.name}_meta.jsonl").exists():
        for line in open(out / f"{a.name}_meta.jsonl", encoding="utf-8"):
            l = json.loads(line)["lang"]; have[l] = have.get(l, 0) + 1
    n = sum(have.values())
    assert (out / f"{a.name}_audio.raw").stat().st_size == n * SR * 2 if n else True, "mswc raw and meta disagree"
    raw = open(out / f"{a.name}_audio.raw", "ab"); meta = open(out / f"{a.name}_meta.jsonl", "a")
    with ProcessPoolExecutor(12) as ex:
        for lang in langs:
            if a.langs and lang not in a.langs.split(","):
                continue
            if lang in have:
                print("mswc have", lang, have[lang], flush=True); continue
            with _open(fs, base, a.local_dir, f"data/splits/{lang}/splits.tar.gz") as fh, tarfile.open(fileobj=fh, mode="r:gz") as t:
                rows = []
                for m in t.getmembers():
                    if m.name.endswith(f"{a.split}.csv"):
                        rows = [r for r in csv.DictReader(io.TextIOWrapper(t.extractfile(m), encoding="utf-8")) if str(r.get("VALID", "True")).lower() == "true"]
            want = {r["LINK"].replace("/", "_", 1): r["WORD"] for r in rows}
            shards = sorted((f for f in files if f.startswith(f"data/opus/{lang}/{a.split}/audio/") and f.endswith(".tar.gz")
                             and (not a.local_dir or (Path(a.local_dir) / f).exists())), key=lambda f: int(f.rsplit("/", 1)[1].split(".")[0]))
            k = min(a.shards_per_lang, len(shards))
            shards = sorted({shards[min(len(shards) - 1, int((i + a.shard_offset) * len(shards) / k))] for i in range(k)})
            got = 0
            for s in shards:
                if got >= a.per_lang:
                    break
                with _open(fs, base, a.local_dir, s) as fh, tarfile.open(fileobj=fh, mode="r|gz") as t:
                    batch = []
                    for m in t:
                        name = m.name.rsplit("/", 1)[-1]
                        if name in want:
                            batch.append((name, t.extractfile(m).read()))
                    per_shard = (a.per_lang - got) // max(1, len([x for x in shards if x >= s]))
                    random.Random(f"{lang}{s}").shuffle(batch)
                    batch = batch[:max(per_shard, 0)]
                    for (name, _), w in zip(batch, ex.map(_mswc_member, [(b,) for _, b in batch], chunksize=32)):
                        if w is None:
                            continue
                        raw.write(w.tobytes()); n += 1; got += 1
                        meta.write(json.dumps({"id": name, "lang": lang, "word": want[name], "split": a.split, "licence": "MSWC (CC BY 4.0)"}, ensure_ascii=False) + "\n")
            print("mswc", lang, got, "from", len(shards), "shards; total", n, flush=True)
    raw.close(); meta.close(); (out / f"{a.name}.done").write_text(str(n))


OK_LICENCES = ("attribution", "cc0", "public domain", "free art")


def _fma_track(args):
    url, name, tid, lic, max_s = args
    import fsspec, zipfile
    try:
        with fsspec.open(url, "rb", block_size=4 << 20) as fh:
            z = zipfile.ZipFile(fh)
            b = z.read(name)
    except Exception:
        return tid, lic, []
    w = to16k(b)
    if w is None:
        return tid, lic, []
    w = w[:int(max_s * SR)]
    return tid, lic, [fix(w[i:i + 10 * SR], 10 * SR) for i in range(0, len(w) - 10 * SR + 1, 10 * SR)]


def fma(a, out):
    import zipfile, pandas as pd, fsspec
    meta_zip = zipfile.ZipFile(a.fma_metadata)
    t = pd.read_csv(io.BytesIO(meta_zip.read("fma_metadata/tracks.csv")), index_col=0, header=[0, 1], low_memory=False)
    lic = t[("track", "license")].fillna("").str.lower()
    ok = lic.apply(lambda s: any(k in s for k in OK_LICENCES) and "noncommercial" not in s and "non-commercial" not in s and "noderiv" not in s and "no deriv" not in s)
    tids = sorted(t.index[ok].tolist())
    url = "https://os.unil.cloud.switch.ch/fma/fma_full.zip"
    with fsspec.open(url, "rb", block_size=4 << 20) as fh:
        names = {int(Path(n).stem): n for n in zipfile.ZipFile(fh).namelist() if n.endswith(".mp3")}
    tids = [i for i in tids if i in names]
    random.Random(0).shuffle(tids)
    budget = int(a.hours * 3600 / 10)
    per_track_s = 180
    raw = open(out / "fma_audio.raw", "ab"); meta = open(out / "fma_meta.jsonl", "a"); n = 0
    with ProcessPoolExecutor(16) as ex:
        for tid, l, segs in ex.map(_fma_track, [(url, names[i], i, t.loc[i, ("track", "license")], per_track_s) for i in tids], chunksize=2):
            for k, s in enumerate(segs):
                raw.write(s.tobytes()); n += 1
                meta.write(json.dumps({"id": f"fma{tid}_{k}", "track": tid, "licence": l}) + "\n")
            if n >= budget:
                break
            if n % 2000 < len(segs):
                print("fma", n, flush=True)
    raw.close(); meta.close(); (out / "fma.done").write_text(str(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("source", choices=["audioset", "mswc", "fma"])
    ap.add_argument("--shards", type=int, default=110); ap.add_argument("--per-lang", type=int, default=60000); ap.add_argument("--shards-per-lang", type=int, default=3)
    ap.add_argument("--shard-offset", type=float, default=0.0, help="mswc: pick archives at (i + offset) * n / k")
    ap.add_argument("--name", default="mswc", help="mswc: output source name")
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"], help="mswc: MSWC split (its <split>.csv and audio shards)")
    ap.add_argument("--local-dir", default="", help="mswc: read the splits archive and audio shards from this local repo copy")
    ap.add_argument("--langs", default="", help="mswc: comma-separated languages (default every language)")
    ap.add_argument("--parquet-glob", default="", help="audioset: local parquet files (e.g. a bal_train copy) instead of Hub unbal_train shards")
    ap.add_argument("--workers", type=int, default=24, help="audioset: decoding processes")
    ap.add_argument("--hours", type=float, default=400); ap.add_argument("--fma-metadata", default="work/data/fma_metadata.zip", help="fma: the fma_metadata.zip archive (tracks.csv)")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    {"audioset": audioset, "mswc": mswc, "fma": fma}[a.source](a, out)
    print("POOL_DONE", a.source, flush=True)


if __name__ == "__main__":
    main()
