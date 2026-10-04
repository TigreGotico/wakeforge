"""pr.py prep                         phone targets for train-clean-100 / dev-clean / test-clean -> data/pr_targets.json
pr.py train <feat> [--epochs 5]       train a linear CTC head per seed (0 and 1), early stop on dev-clean PER, test on test-clean
pr.py direct                          wav2vec2-xlsr-53-espeak's own CTC head, greedy, no training, on dev-clean and test-clean

Targets: Wav2Vec2PhonemeCTCTokenizer (facebook/wav2vec2-xlsr-53-espeak-cv-ft), phonemizer_lang en-us, with pad, <s>, </s>,
<unk> and the word delimiter dropped. The CTC output is that tokenizer's full table (blank = pad id 0).
Features are computed on the fly on the GPU. Both seeds see the same batches in the same order (batch order from seed 0);
the seed sets the head's initialisation, so one featurizer forward feeds both heads.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import featlib  # noqa: E402
from featlib import W2V, Featurizer, WeightedSum, param_count  # noqa: E402

B = Path("work/bench")
SPLITS = {"train": "train-clean-100", "dev": "dev-clean", "test": "test-clean"}
BATCH_SECONDS = 160
_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        from transformers import Wav2Vec2PhonemeCTCTokenizer
        _TOK = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(W2V[0])
    return _TOK


def drop_ids():
    t = tok()
    return {t.pad_token_id, t.unk_token_id, t.bos_token_id, t.eos_token_id, t.convert_tokens_to_ids(t.word_delimiter_token)}


def _phon(texts):
    t, d = tok(), drop_ids()
    if t.phonemizer_lang != "en-us" or not hasattr(t, "backend"):
        t.init_backend("en-us"); t.phonemizer_lang = "en-us"
    return [[i for i in t.convert_tokens_to_ids([p for p in t.phonemize(x.lower()).split(" ") if p.strip()]) if i not in d] for x in texts]


def prep(a):
    out = {}
    for split, name in SPLITS.items():
        rows = []
        for tr in sorted((Path(a.librispeech) / name).glob("*/*/*.trans.txt")):
            for line in tr.read_text().splitlines():
                uid, text = line.split(" ", 1)
                f = tr.parent / f"{uid}.flac"
                rows.append([str(f), sf.info(str(f)).frames, text])
        chunks = [rows[i:i + 200] for i in range(0, len(rows), 200)]
        with Pool(4) as p:
            ids = [x for c in p.map(_phon, [[r[2] for r in c] for c in chunks]) for x in c]
        out[split] = [[r[0], r[1], i] for r, i in zip(rows, ids)]
        print(split, len(rows), "utterances", sum(r[1] for r in rows) / 16000 / 3600, "h", flush=True)
    t = tok()
    out["vocab"] = t.convert_ids_to_tokens(list(range(len(t))))
    out["dropped_ids"] = sorted(drop_ids())
    (B / "data/pr_targets.json").write_text(json.dumps(out))


def batches(rows, rng):
    idx = sorted(range(len(rows)), key=lambda i: rows[i][1])
    out, cur, tot = [], [], 0
    for i in idx:
        if cur and (len(cur) + 1) * rows[i][1] > BATCH_SECONDS * 16000:
            out.append(cur); cur = []
        cur.append(i)
    out.append(cur)
    if rng is not None:
        rng.shuffle(out)
    return out


class DS(torch.utils.data.Dataset):
    def __init__(self, rows, bl):
        self.rows, self.bl = rows, bl

    def __len__(self):
        return len(self.bl)

    def __getitem__(self, k):
        rows = [self.rows[i] for i in self.bl[k]]
        return [sf.read(r[0], dtype="float32")[0] for r in rows], [r[2] for r in rows]


def loader(rows, bl, workers):
    return torch.utils.data.DataLoader(DS(rows, bl), batch_size=None, shuffle=False, num_workers=workers, prefetch_factor=4,
                                       persistent_workers=False)


def edit(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    b = np.asarray(b)
    prev = np.arange(len(b) + 1)
    for i, x in enumerate(a, 1):
        cur = np.empty_like(prev)
        cur[0] = i
        sub = prev[:-1] + (b != x)
        dele = prev[1:] + 1
        cur[1:] = np.minimum(sub, dele)
        cur = np.minimum.accumulate(cur - np.arange(len(cur))) + np.arange(len(cur))
        prev = cur
    return int(prev[-1])


def greedy(lp, n):
    best = lp[:n].argmax(-1).tolist()
    out, prev = [], 0
    for t in best:
        if t != prev and t != 0:
            out.append(t)
        prev = t
    return out


def stats(feat, rows, n=60, seed=123):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows), n, replace=False)
    acc = {}
    for k in range(0, n, 6):
        wavs = [sf.read(rows[i][0], dtype="float32")[0] for i in pick[k:k + 6]]
        e, lens = feat.frames(wavs)
        T = next(iter(e.values())).shape[1]
        m = (torch.arange(T)[None] < lens[:, None]).to(feat.dev)
        for name, v in e.items():
            x = v[m].double()
            s = acc.setdefault(name, [0, 0, 0])
            s[0] += x.shape[0]; s[1] = s[1] + x.sum(0); s[2] = s[2] + (x * x).sum(0)
    out = {}
    for name, (c, s1, s2) in acc.items():
        mu = s1 / c
        out[name] = ((mu).float().cpu(), torch.sqrt((s2 / c - mu * mu).clamp(min=1e-10)).float().cpu())
    return out


RAW_BYTES_MAX = 2 * 2**30


def evaluate(feat, heads, rows, workers, raw=None):
    errs = [0] * len(heads); ref = 0
    if raw is not None:
        raw.update({"order": [i for b in batches(rows, None) for i in b], "lp": [[] for _ in heads], "hyp": [[] for _ in heads],
                    "edits": [[] for _ in heads], "frames": [], "lp_bytes": 0, "lp_ok": True})
    for h in heads:
        h.eval()
    with torch.no_grad():
        for wavs, tg in loader(rows, batches(rows, None), workers):
            e, lens = feat.frames(wavs)
            ref += sum(len(t) for t in tg)
            if raw is not None:
                raw["frames"] += [int(n) for n in lens]
            for j, (ws, lin) in enumerate(heads):
                lp = lin(ws(e)).log_softmax(-1).cpu()
                for i, t in enumerate(tg):
                    h = greedy(lp[i], int(lens[i]))
                    n = edit(h, t)
                    errs[j] += n
                    if raw is not None:
                        raw["hyp"][j].append(h); raw["edits"][j].append(n)
                        if raw["lp_ok"]:
                            x = lp[i, :int(lens[i])].half().numpy()
                            raw["lp"][j].append(x); raw["lp_bytes"] += x.nbytes
                            if raw["lp_bytes"] > RAW_BYTES_MAX:
                                raw["lp_ok"] = False; raw["lp"] = [[] for _ in heads]
    for h in heads:
        h.train()
    return [e / ref for e in errs]


def cat(lists):
    off = np.cumsum([0] + [len(x) for x in lists]).astype(np.int64)
    flat = np.fromiter((v for x in lists for v in x), np.int16, count=int(off[-1]))
    return flat, off


def save_raw(path, a, heads, rows, raw):
    order = raw["order"]
    d = {"utt_ids": np.asarray([Path(rows[i][0]).stem for i in order]), "frames": np.asarray(raw["frames"], np.int32)}
    d["ref"], d["ref_offsets"] = cat([rows[i][2] for i in order])
    d["ref_len"] = np.asarray([len(rows[i][2]) for i in order], np.int32)
    d["seeds"] = np.asarray(a.seeds)
    d["posteriors_saved"] = np.asarray(raw["lp_ok"])
    for j, s in enumerate(a.seeds):
        d[f"hyp_seed{s}"], d[f"hyp_offsets_seed{s}"] = cat(raw["hyp"][j])
        d[f"edits_seed{s}"] = np.asarray(raw["edits"][j], np.int32)
        if raw["lp_ok"]:
            d[f"logprob_f16_seed{s}"] = np.concatenate(raw["lp"][j])
    d["frame_offsets"] = np.cumsum([0] + [int(n) for n in raw["frames"]]).astype(np.int64)
    np.savez(path, **d)
    print("RAW_SAVED", path, "posteriors" if raw["lp_ok"] else "posteriors omitted (over 2 GB)", flush=True)


def train(a):
    T = json.loads((B / "data/pr_targets.json").read_text())
    V = len(T["vocab"])
    out = B / "results/pr" / a.feat
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    feat = Featurizer(a.feat)
    rows = T["train"][:a.limit] if a.limit else T["train"]
    dev = T["dev"][:a.limit // 10] if a.limit else T["dev"]
    st = stats(feat, rows)
    heads = []
    for s in a.seeds:
        torch.manual_seed(s)
        ws = WeightedSum(feat.dims, feat.width, st).cuda()
        heads.append(torch.nn.ModuleList([ws, torch.nn.Linear(feat.width, V)]).cuda())
    opts = [torch.optim.Adam(h.parameters(), lr=a.lr) for h in heads]
    rng = np.random.default_rng(0)
    hist = [[] for _ in heads]; best = [(9.0, -1, None)] * len(heads); stop = [False] * len(heads)
    t0 = time.time()
    for ep in range(a.epochs):
        bl = batches(rows, rng)
        losses = [[] for _ in heads]
        for step, (wavs, tg) in enumerate(loader(rows, bl, a.workers)):
            e, lens = feat.frames(wavs)
            tgt = torch.tensor([x for t in tg for x in t], dtype=torch.long)
            tl = torch.tensor([len(t) for t in tg])
            for j, (h, o) in enumerate(zip(heads, opts)):
                if stop[j]:
                    continue
                ws, lin = h
                lp = lin(ws(e)).log_softmax(-1).transpose(0, 1)
                loss = F.ctc_loss(lp, tgt, lens, tl, blank=0, zero_infinity=True)
                o.zero_grad(); loss.backward(); o.step()
                losses[j].append(float(loss))
            if step % 200 == 0:
                print(json.dumps({"epoch": ep, "step": step, "of": len(bl), "loss": [round(float(np.mean(l[-200:])), 4) if l else None for l in losses],
                                  "elapsed_s": round(time.time() - t0), "gpu_mb": torch.cuda.max_memory_allocated() // 2**20}), flush=True)
        pers = evaluate(feat, heads, dev, a.workers)
        for j, h in enumerate(heads):
            if stop[j]:
                continue
            hist[j].append({"epoch": ep, "train_loss": float(np.mean(losses[j])), "dev_per": pers[j], "weights": h[0].weights()})
            if pers[j] < best[j][0]:
                best[j] = (pers[j], ep, {k: v.detach().clone() for k, v in h.state_dict().items()})
            elif ep - best[j][1] >= a.patience:
                stop[j] = True
        print(json.dumps({"epoch_done": ep, "dev_per": pers, "weights": [h[0].weights() for h in heads], "elapsed_s": round(time.time() - t0)}), flush=True)
        if all(stop):
            break
    for j, h in enumerate(heads):
        h.load_state_dict(best[j][2])
    trows = T["test"][:a.limit // 10] if a.limit else T["test"]
    raw = {}
    test = evaluate(feat, heads, trows, a.workers, raw)
    for j, s in enumerate(a.seeds):
        torch.save(heads[j].state_dict(), out / f"head-seed{s}.pt")
    res = {"task": "PR", "featurizer": a.feat, "metric": "PER (test-clean)", "budget": {"epochs_max": a.epochs, "patience": a.patience,
           "lr": a.lr, "optimizer": "Adam", "batch_seconds": BATCH_SECONDS}, "featurizer_params": param_count(feat),
           "heads_from": getattr(feat, "heads_from", None), "limit": a.limit,
           "seeds": {str(s): {"test_per": test[j], "best_dev_per": best[j][0], "best_epoch": best[j][1], "weights": heads[j][0].weights(),
                              "history": hist[j]} for j, s in enumerate(a.seeds)},
           "train_seconds": round(time.time() - t0)}
    (out / (a.json_name if not a.limit else "results.smoke.json")).write_text(json.dumps(res, indent=1))
    save_raw(out / ("raw.npz" if not a.limit else "raw.smoke.npz"), a, heads, trows, raw)
    print("PR_DONE", json.dumps({s: res["seeds"][s]["test_per"] for s in res["seeds"]}), flush=True)


def direct(a):
    from transformers import Wav2Vec2ForCTC
    T = json.loads((B / "data/pr_targets.json").read_text())
    m = Wav2Vec2ForCTC.from_pretrained(W2V[0], revision=W2V[1], attn_implementation="sdpa").cuda().eval()
    d = set(T["dropped_ids"])
    res = {"task": "PR", "featurizer": "wav2vec2-espeak", "mode": "own CTC head, greedy, no training", "params": sum(p.numel() for p in m.parameters())}
    for split in ("dev", "test"):
        err = ref = 0
        for i, r in enumerate(T[split]):
            w = torch.from_numpy(sf.read(r[0], dtype="float32")[0]).cuda()
            w = (w - w.mean()) / torch.sqrt(w.var() + 1e-7)
            with torch.no_grad():
                ids = m(w[None]).logits[0].argmax(-1).tolist()
            hyp, prev = [], -1
            for t in ids:
                if t != prev and t not in d:
                    hyp.append(t)
                prev = t
            err += edit(hyp, r[2]); ref += len(r[2])
        res[f"{split}_per"] = err / ref
        print(split, err / ref, flush=True)
    out = B / "results/pr/wav2vec2-espeak-direct"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    featlib.add_args(ap)
    ap.add_argument("--librispeech", default="work/data/LibriSpeech", help="directory holding train-clean-100, dev-clean and test-clean")
    ap.add_argument("cmd", choices=["prep", "train", "direct"]); ap.add_argument("feat", nargs="?")
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-2); ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--json-name", default="results.json")
    a = ap.parse_args()
    B = featlib.configure(a)
    a.seeds = [int(s) for s in a.seeds.split(",")]
    {"prep": lambda: prep(a), "train": lambda: train(a), "direct": lambda: direct(a)}[a.cmd]()
