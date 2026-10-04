"""pr_full.py prep39                       ARPAbet phone targets from librispeech-lexicon.txt -> data/pr_targets_superb39.json
pr_full.py run <feat> <espeak|superb39>   pr.py's probe unchanged, epochs 100, patience 10; results.full.json / results.superb39.json

Both take --bench-dir (default work/bench); prep39 also takes --librispeech and run --heads for the wakephonehubert rows.
pr.py is imported unchanged. Its base directory is pointed at full-base-<targets>/ (data/pr_targets.json there is the chosen
target file), so nothing under results/pr is overwritten; outputs are then copied into results/pr/<feat>/ under new names.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import featlib  # noqa: E402

BENCH = Path("work/bench")


def prep39(librispeech):
    import pr as P
    lex = {}
    for line in (BENCH / "data/librispeech-lexicon.txt").read_text().splitlines():
        w, *ph = line.split()
        lex.setdefault(w, [re.sub(r"\d", "", p) for p in ph])
    phones = sorted({p for v in lex.values() for p in v})
    assert len(phones) == 39, phones
    vocab = ["<blank>"] + phones
    pid = {p: i for i, p in enumerate(vocab)}
    out, stats = {}, {}
    for split in ("train", "dev", "test"):
        rows = []
        miss = set()
        dropped = 0
        for tr in sorted((Path(librispeech) / P.SPLITS[split]).glob("*/*/*.trans.txt")):
            for line in tr.read_text().splitlines():
                uid, text = line.split(" ", 1)
                words = text.split()
                bad = [w for w in words if w not in lex]
                if bad:
                    miss.update(bad)
                    dropped += 1
                    continue
                f = str(tr.parent / f"{uid}.flac")
                rows.append([f, P.sf.info(f).frames, [pid[p] for w in words for p in lex[w]]])
        stats[split] = {"utterances_total": len(rows) + dropped, "utterances_dropped": dropped, "missing_word_types": len(miss),
                        "missing_words": sorted(miss)}
        out[split] = rows
        print(split, stats[split]["utterances_total"], "utterances, dropped", dropped, "missing word types", len(miss), sorted(miss)[:20], flush=True)
    out["vocab"] = vocab
    out["dropped_ids"] = [0]
    out["lexicon"] = {"file": "librispeech-lexicon.txt (openslr resource 11)", "pronunciation": "first listed", "stress": "stripped"}
    out["drop_stats"] = stats
    (BENCH / "data/pr_targets_superb39.json").write_text(json.dumps(out))


def run(feat, targets, epochs, patience, workers):
    import torch
    base = BENCH / f"full-base-{targets}"
    (base / "data").mkdir(parents=True, exist_ok=True)
    link = base / "data/pr_targets.json"
    src = BENCH / ("data/pr_targets.json" if targets == "espeak" else "data/pr_targets_superb39.json")
    if not link.exists():
        link.symlink_to(src.resolve())
    import pr as P
    torch.cuda.set_per_process_memory_fraction(1.0 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    P.B = base
    name = "results.full.json" if targets == "espeak" else "results.superb39.json"
    a = argparse.Namespace(cmd="train", feat=feat, epochs=epochs, patience=patience, lr=1e-2, seeds=[0, 1], workers=workers, threads=2,
                           limit=0, json_name=name)
    P.train(a)
    tag = "full" if targets == "espeak" else "superb39"
    src, dst = base / "results/pr" / feat, BENCH / "results/pr" / feat
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / name, dst / name)
    shutil.copy(src / "raw.npz", dst / f"raw.{tag}.npz")
    for s in (0, 1):
        shutil.copy(src / f"head-seed{s}.pt", dst / f"head-seed{s}.{tag}.pt")
    print("COPIED", dst, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    featlib.add_args(ap)
    ap.add_argument("cmd", choices=["prep39", "run"]); ap.add_argument("feat", nargs="?"); ap.add_argument("targets", nargs="?", choices=["espeak", "superb39"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--librispeech", default="work/data/LibriSpeech", help="prep39: directory holding train-clean-100, dev-clean and test-clean")
    a = ap.parse_args()
    BENCH = featlib.configure(a)
    if a.cmd == "prep39":
        prep39(a.librispeech)
    else:
        run(a.feat, a.targets, 100, 10, a.workers)
