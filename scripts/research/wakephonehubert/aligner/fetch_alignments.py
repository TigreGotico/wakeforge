"""fetch_alignments.py [--work-dir work/aligner]

Fetch MFA word/phone alignments (gilkeyio/librispeech-alignments, CC-BY-4.0) into <work-dir>/data.
train_clean_100: alignment columns only (audio is local). test_clean: whole shard (audio included)."""
import argparse, json, time, pathlib
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

REPO = "gilkeyio/librispeech-alignments"
ap = argparse.ArgumentParser()
ap.add_argument("--work-dir", default="work/aligner")
out = pathlib.Path(ap.parse_args().work_dir) / "data"
out.mkdir(parents=True, exist_ok=True)
fs = HfFileSystem()


def dump(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


for split, n in (("train_clean_100", 14),):
    dst = out / f"{split}.alignments.jsonl"
    if dst.exists():
        continue
    rows = []
    for i in range(n):
        t = time.time()
        name = f"datasets/{REPO}/data/{split}-{i:05d}-of-{n:05d}.parquet"
        with fs.open(name, "rb", block_size=8 * 2**20) as f:
            tb = pq.ParquetFile(f).read(columns=["id", "transcript", "words", "phonemes"])
        rows += tb.to_pylist()
        print(split, i, tb.num_rows, f"{time.time()-t:.1f}s", flush=True)
        time.sleep(20)
    dump(rows, dst)

p = hf_hub_download(REPO, "data/test_clean-00000-of-00001.parquet", repo_type="dataset",
                    local_dir=str(out / "hf"))
print("test_clean parquet", p, flush=True)
