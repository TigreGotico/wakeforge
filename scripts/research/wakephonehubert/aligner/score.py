"""score.py [--work-dir ...]

Boundary metrics from the raw result files."""
import argparse, json
import numpy as np
import common
from common import *

ap = argparse.ArgumentParser()
common.add_args(ap)
common.configure(ap.parse_args())
ROOT = common.ROOT

out = {}
for name in ["uniform", "wph_ctc", "teacher_ctc", "head_ref", "head_cmudict"]:
    p = ROOT / "results" / f"{name}.jsonl"
    if not p.exists():
        continue
    rows = read_jsonl(p)
    ok = [r for r in rows if r.get("pred_start") is not None]
    st, en = boundary_errors(ok)
    res = dict(utts=len(rows), failed=len(rows) - len(ok), words=int(st.size),
               word_all=summarise(np.concatenate([st, en])), word_start=summarise(st), word_end=summarise(en))
    if "ref_phones" in ok[0]:
        ps, pe = [], []
        for r in ok:
            for h, g in zip(r["pred_phones"], r["ref_phones"]):
                ps.append(abs(h[1] - g[1])); pe.append(abs(h[2] - g[2]))
        ps, pe = np.array(ps), np.array(pe)
        res.update(phones=int(ps.size), phone_all=summarise(np.concatenate([ps, pe])),
                   phone_start=summarise(ps), phone_end=summarise(pe))
    out[name] = res
json.dump(out, open(ROOT / "results" / "metrics.json", "w"), indent=1)
for k, v in out.items():
    print(k, v["utts"], v["failed"], v["words"])
    for m in ["word_all", "word_start", "word_end", "phone_all", "phone_start", "phone_end"]:
        if m in v:
            s = v[m]
            print(f"  {m:12s} n={s['n']:6d} <=20ms {s['within_20ms']*100:5.1f}% <=50ms {s['within_50ms']*100:5.1f}% mean {s['mean_ms']:6.1f} median {s['median_ms']:5.1f}")
