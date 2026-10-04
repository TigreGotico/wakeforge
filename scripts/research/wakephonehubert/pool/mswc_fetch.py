"""Fetch a language-balanced slice of MLCommons/ml_spoken_words (opus) into a local copy of the repository layout: per
language the splits archive, the dev and test audio shards and --train-shards train shards, paced with a gap between
files so the Hub does not shape the address. build_pool.py reads the copy with --local-dir.

    mswc_fetch.py <out_dir> [--langs a,b,...] [--gap 20]
"""
import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "MLCommons/ml_spoken_words"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--langs", default="")
    ap.add_argument("--gap", type=float, default=20.0)
    ap.add_argument("--train-shards", type=int, default=3)
    a = ap.parse_args()
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    langs = sorted({f.split("/")[2] for f in files if f.startswith("data/opus/")})
    if a.langs:
        langs = [l for l in langs if l in a.langs.split(",")]
    smallest_first = sorted(langs, key=lambda l: sum(1 for f in files if f.startswith(f"data/opus/{l}/train/audio/")))
    for lang in smallest_first:
        shards = sorted((f for f in files if f.startswith(f"data/opus/{lang}/train/audio/")), key=lambda f: int(f.rsplit("/", 1)[1].split(".")[0]))
        picks = sorted({shards[int(i * len(shards) / min(a.train_shards, len(shards)))] for i in range(min(a.train_shards, len(shards)))})
        devtest = [f for f in files if f.startswith(f"data/opus/{lang}/dev/audio/") or f.startswith(f"data/opus/{lang}/test/audio/")]
        wanted = [f"data/splits/{lang}/splits.tar.gz"] + sorted(devtest) + picks
        for f in wanted:
            if f not in files:
                print("MISSING", f, flush=True); continue
            if (Path(a.out_dir) / f).exists():
                continue
            t = time.time()
            p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=a.out_dir)
            size = Path(p).stat().st_size
            print(f"GOT {f} {size / 1e6:.1f} MB {size / 1e6 / max(time.time() - t, 1e-3):.2f} MB/s", flush=True)
            time.sleep(a.gap)
    print("FETCH_DONE", flush=True)


if __name__ == "__main__":
    main()
