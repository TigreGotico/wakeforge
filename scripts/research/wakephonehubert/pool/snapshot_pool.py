"""snapshot_pool.py <pool_dir> <src> <new_name> --type <mswc|audioset|fma> [--start S] [--count N]

Writes <new_name>.slice.json (base raw path, start segment, count, segment length, type), <new_name>_meta.jsonl (the
matching meta lines) and <new_name>.done (the count). No audio is copied: pool_teachers.py and train_pool.py read the
base raw file at the slice's offset. Without --count the slice runs to the end of the source, which must then be done.
Refuses a range the raw file or the meta file does not fully hold, and an existing <new_name>.
"""
import argparse
import json
from pathlib import Path

from pool_teachers import SEG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool_dir"); ap.add_argument("src"); ap.add_argument("new_name")
    ap.add_argument("--type", required=True, choices=list(SEG))
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--count", type=int, default=0)
    a = ap.parse_args()
    d = Path(a.pool_dir); seg = SEG[a.type]
    raw, meta = d / f"{a.src}_audio.raw", d / f"{a.src}_meta.jsonl"
    for k in (".slice.json", "_meta.jsonl", ".done", "_audio.raw", "_progress.json"):
        if (d / f"{a.new_name}{k}").exists():
            raise SystemExit(f"{d / (a.new_name + k)} exists; pick another name")
    if (d / f"{a.src}.slice.json").exists():
        raise SystemExit(f"{a.src} is itself a slice; snapshot its base source instead")
    if a.count:
        count = a.count
    else:
        done = d / f"{a.src}.done"
        if not done.exists():
            raise SystemExit(f"{a.src} is still being built; pass --count")
        count = int(done.read_text().strip()) - a.start
    end = a.start + count
    have_raw = raw.stat().st_size // (2 * seg)
    if count <= 0 or end > have_raw:
        raise SystemExit(f"range {a.start}..{end} not in {raw.name}, which holds {have_raw} complete segments")
    lines = []
    with open(meta, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= end:
                break
            if i >= a.start:
                if not line.endswith("\n"):
                    break
                lines.append(line)
    if len(lines) != count:
        raise SystemExit(f"{meta.name} holds only {a.start + len(lines)} complete lines, range needs {end}")
    tmp = d / f"{a.new_name}_meta.jsonl.tmp"
    tmp.write_text("".join(lines), encoding="utf-8"); tmp.replace(d / f"{a.new_name}_meta.jsonl")
    (d / f"{a.new_name}.slice.json").write_text(json.dumps({"raw": str(raw.resolve()), "start": a.start, "count": count, "seg": seg,
                                                            "type": a.type, "base": a.src}))
    (d / f"{a.new_name}.done").write_text(str(count))
    first, last = json.loads(lines[0]), json.loads(lines[-1])
    print(json.dumps({"snapshot": a.new_name, "base": a.src, "start": a.start, "count": count, "first": first.get("id"), "last": last.get("id"),
                      "langs": sorted({json.loads(l).get("lang") for l in lines} - {None})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
