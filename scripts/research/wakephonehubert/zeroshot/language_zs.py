"""VoxLingua107, the benchmark's ten languages and splits (feats/lid/meta.json): zero-shot and trained rows.

  extract   ONNX per clip -> results/language/onnx.npz: mean-pooled features (521: hubert, vad, ipa), mean ipa posterior (392), greedy
            phone sequence (ragged)
  score     zero-shot: cosine nearest-class-centroid (training clips as exemplars, nothing learned) on mean-pooled
            features, and on the per-clip phone-bigram distribution of the greedy ipa decoding (square-root of the
            relative bigram frequencies, cosine); trained: linear probe on mean-pooled features, seeds 0 and 1,
            train_pooled.py's budget, early stop on the dev split
"""
import argparse
import io
import json
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import probe  # noqa: E402
import zslib  # noqa: E402

OD = zslib.ZS / "results/language"
LANGS = ["sv", "nl", "hy", "fr", "fa", "ar", "da", "lv", "fi", "et"]
V = 392


def to16k(x, sr):
    from math import gcd
    from scipy.signal import resample_poly
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 16000:
        g = gcd(16000, sr)
        x = resample_poly(x, 16000 // g, sr // g)
    return np.ascontiguousarray(x, dtype=np.float32)


def one_tar(task):
    tar, items = task
    s = zslib.session("wakephonehubert_features_int8.onnx", threads=1)
    out = []
    with tarfile.open(tar) as tf:
        for i, member in items:
            x, sr = sf.read(io.BytesIO(tf.extractfile(member).read()), dtype="float32")
            o = zslib.run(s, to16k(x, sr), ("features",))["features"][:, :521]
            ipa = o[:, 129:521]
            out.append((i, o.mean(0), ipa.mean(0), zslib.greedy(ipa.argmax(-1)), len(o)))
    return out


def extract():
    OD.mkdir(parents=True, exist_ok=True)
    rows = json.loads((zslib.BENCH / "feats/lid/meta.json").read_text())["rows"]
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["tar"], []).append((i, r["member"]))
    N = len(rows)
    F = np.zeros((N, 521), np.float32); P = np.zeros((N, V), np.float32); seqs = [None] * N; frames = np.zeros(N, np.int64)
    with Pool(8) as p:
        for res in p.imap_unordered(one_tar, list(by.items())):
            for i, f, pm, sq, n in res:
                F[i], P[i], seqs[i], frames[i] = f, pm, sq, n
            print(json.dumps({"tar_done": len(res)}), flush=True)
    np.savez_compressed(OD / "onnx.npz", keys=np.asarray([r["key"] for r in rows]), label=np.asarray([r["label"] for r in rows]),
                        split=np.asarray([r["split"] for r in rows]), features_mean=F, ipa_mean=P, frames=frames,
                        phones=np.concatenate([np.asarray(s, np.int16) for s in seqs]), phones_offsets=np.cumsum([0] + [len(s) for s in seqs]).astype(np.int64))
    print("EXTRACT_DONE", flush=True)


def bigrams(phones, off):
    """Square root of each clip's relative phone-bigram frequencies, as a dense [N, V*V] float32 array."""
    from scipy.sparse import csr_matrix
    r, c = [], []
    for i in range(len(off) - 1):
        s = phones[off[i]:off[i + 1]].astype(np.int64)
        if len(s) > 1:
            r.append(np.full(len(s) - 1, i)); c.append(s[:-1] * V + s[1:])
    r, c = np.concatenate(r), np.concatenate(c)
    M = csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(off) - 1, V * V))
    M.sum_duplicates()
    tot = np.asarray(M.sum(1)).ravel()
    M = csr_matrix(M.multiply(1 / np.maximum(tot, 1)[:, None]))
    M.data = np.sqrt(M.data)
    return M


def ncc_sparse(M, y, tr, te, C):
    cen = np.stack([np.asarray(M[tr][y[tr] == c].mean(0)).ravel() for c in range(C)])
    cen /= np.linalg.norm(cen, axis=1, keepdims=True) + 1e-12
    s = np.asarray(M[te] @ cen.T)
    s /= np.sqrt(np.asarray(M[te].multiply(M[te]).sum(1))) + 1e-12
    return s, float((s.argmax(1) == y[te]).mean())


def score():
    import torch
    torch.set_num_threads(4)
    O = np.load(OD / "onnx.npz")
    y, sp = O["label"], O["split"]
    tr, dv, te = (np.where(sp == s)[0] for s in ("train", "dev", "test"))
    res = {"task": "language", "set": "VoxLingua107, ten languages", "sizes": {"train": len(tr), "dev": len(dv), "test": len(te)},
           "phones_per_clip_median": float(np.median(np.diff(O["phones_offsets"])))}
    raw = {"keys": O["keys"], "label": y, "split": sp}
    s, a = probe.ncc(O["features_mean"], y, tr, te, 10); res["zeroshot_ncc_features_mean"] = a; raw["ncc_features_scores_test"] = s
    s, a = probe.ncc(O["ipa_mean"], y, tr, te, 10); res["zeroshot_ncc_ipa_mean"] = a; raw["ncc_ipa_mean_scores_test"] = s
    M = bigrams(O["phones"], O["phones_offsets"])
    s, a = ncc_sparse(M, y, tr, te, 10); res["zeroshot_ncc_phone_bigrams"] = a; raw["ncc_bigram_scores_test"] = s
    for seed in (0, 1):
        lg, a, d, ep = probe.linear_probe(O["features_mean"], y, tr, dv, te, 10, seed)
        res[f"probe_features_mean_seed{seed}"] = {"test_acc": a, "best_dev_acc": d, "best_epoch": ep}
        raw[f"probe_features_logits_test_seed{seed}"] = lg
    raw["test_index"] = te
    np.savez_compressed(OD / "raw.npz", **raw)
    (OD / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def main():
    global OD
    ap = argparse.ArgumentParser()
    zslib.add_args(ap)
    ap.add_argument("cmd", choices=["extract", "score"])
    a = ap.parse_args()
    zslib.configure(a)
    OD = zslib.ZS / "results/language"
    {"extract": extract, "score": score}[a.cmd]()


if __name__ == "__main__":
    main()
