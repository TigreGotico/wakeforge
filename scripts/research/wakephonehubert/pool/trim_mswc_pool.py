"""trim_mswc_pool.py <pool_dir> <last_language_to_keep>

Cuts the mswc source back to its last complete language, so that build_pool.py can resume at a smaller per-language cap.
The dropped metadata is kept beside it as mswc_meta.jsonl.with-partial-next.
"""
import json
import os
import sys

pool, keep_through = sys.argv[1], sys.argv[2]
meta = os.path.join(pool, "mswc_meta.jsonl")
lines = open(meta, encoding="utf-8").readlines()
langs = [json.loads(l)["lang"] for l in lines]
keep = max(i for i, l in enumerate(langs) if l == keep_through) + 1
assert keep == len(langs) or langs[keep] != keep_through
os.rename(meta, meta + ".with-partial-next")
open(meta, "w", encoding="utf-8").writelines(lines[:keep])
os.truncate(os.path.join(pool, "mswc_audio.raw"), keep * 32000)
print("kept", keep, "next language dropped:", langs[keep] if keep < len(langs) else None, "raw bytes", os.path.getsize(os.path.join(pool, "mswc_audio.raw")))
