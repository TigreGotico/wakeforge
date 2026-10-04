"""build_assets.py [--keyword-clip WAV] [--export-dir ...] [--zeroshot-dir ...] [--bench-dir ...]

Files the model-card snippets read, under <zeroshot-dir>/card_assets/: the model, example clips, and the small
heads and centroids the benchmark produced (ESC-50 and language-ID ones are the fold-1 / seed-0 models of the tables).
--keyword-clip is copied to clips/keyword.wav."""
import argparse
import io
import json
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import zslib  # noqa: E402

D = 521


def fit_linear(X, y, tr, dv, C, seed=0):
    """probe.linear_probe's training, returning the standardisation and the layer."""
    import torch.nn.functional as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-5
    Z = torch.as_tensor((X - mu) / sd, dtype=torch.float32)
    Y = torch.as_tensor(y)
    lin = torch.nn.Linear(Z.shape[1], C)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    best = (-1.0, -1, None)
    for ep in range(50):
        perm = rng.permutation(tr)
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            loss = F.cross_entropy(lin(Z[b]), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            da = float((lin(Z[dv]).argmax(-1) == Y[dv]).float().mean())
        if da > best[0]:
            best = (da, ep, {k: v.clone() for k, v in lin.state_dict().items()})
        elif ep - best[1] >= 5:
            break
    lin.load_state_dict(best[2])
    return mu, sd, lin.weight.detach().numpy(), lin.bias.detach().numpy()


def centroids(X, y, tr, C):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-5
    Z = (X - mu) / sd
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    cen = np.stack([Z[tr][y[tr] == c].mean(0) for c in range(C)])
    return mu, sd, cen / np.linalg.norm(cen, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    zslib.add_args(ap)
    ap.add_argument("--keyword-clip", default="", help="a wav of a spoken keyword for the card's keyword-spotting snippet")
    a = ap.parse_args()
    zslib.configure(a)
    A = zslib.ZS / "card_assets"
    R = zslib.ZS / "results"
    (A / "heads").mkdir(parents=True, exist_ok=True)
    (A / "clips").mkdir(exist_ok=True)
    for f in ("wakephonehubert_int8.onnx", "wakephonehubert_features_int8.onnx", "vocab.json", "config.json"):
        shutil.copy(zslib.EXPORT / f, A / f)
    ck = torch.load(R / "vad/head-conv2-seed0.pt", map_location="cpu")
    thr = json.loads((R / "vad/trained.json").read_text())["conv2_seed0"]["dev_tuned_threshold"]
    np.savez(A / "heads/vad_conv2.npz", mu=ck["mu"].numpy(), sd=ck["sd"].numpy(), threshold=np.float32(thr),
             **{k.replace(".", "_"): v.numpy() for k, v in ck["state"].items()})
    O = np.load(R / "sound/onnx.npz")
    X, y, fold = O["features_mean"][:, :D], O["label"], O["fold"]
    cats = [None] * 50
    for c, l in zip(O["category"], y):
        cats[l] = str(c)
    tr = np.where((fold != 1) & (fold != 2))[0]; dv = np.where(fold == 2)[0]; te = np.where(fold == 1)[0]
    mu, sd, cen = centroids(X, y, tr, 50)
    np.savez(A / "heads/esc50_centroids.npz", mu=mu, sd=sd, centroids=cen, classes=np.asarray(cats))
    mu, sd, W, b = fit_linear(X, y, tr, dv, 50)
    acc = float(((((X[te] - mu) / sd) @ W.T + b).argmax(1) == y[te]).mean())
    np.savez(A / "heads/esc50_linear.npz", mu=mu, sd=sd, W=W, b=b, classes=np.asarray(cats))
    print(json.dumps({"esc50_linear_fold1_test_acc": acc}))
    L = np.load(R / "language/onnx.npz")
    X, y, sp = L["features_mean"][:, :D], L["label"], L["split"]
    tr, dv, te = (np.where(sp == s)[0] for s in ("train", "dev", "test"))
    langs = ["sv", "nl", "hy", "fr", "fa", "ar", "da", "lv", "fi", "et"]
    mu, sd, cen = centroids(X, y, tr, 10)
    np.savez(A / "heads/lid10_centroids.npz", mu=mu, sd=sd, centroids=cen, classes=np.asarray(langs))
    mu, sd, W, b = fit_linear(X, y, tr, dv, 10)
    acc = float(((((X[te] - mu) / sd) @ W.T + b).argmax(1) == y[te]).mean())
    np.savez(A / "heads/lid10_linear.npz", mu=mu, sd=sd, W=W, b=b, classes=np.asarray(langs))
    print(json.dumps({"lid_linear_test_acc": acc}))
    z = zipfile.ZipFile(zslib.BENCH / "data/esc50/ESC-50-master.zip")
    i0 = int(np.where(fold == 1)[0][0])
    fn = str(O["files"][i0])
    x, sr = sf.read(io.BytesIO(z.read("ESC-50-master/audio/" + fn)), dtype="float32")
    sf.write(A / "clips/esc50_fold1.wav", x, sr)
    print(json.dumps({"esc50_clip": fn, "category": str(O["category"][i0])}))
    rows = json.loads((zslib.BENCH / "feats/lid/meta.json").read_text())["rows"]
    r = next(r for r in rows if r["split"] == "test" and r["label"] == 3)
    with tarfile.open(r["tar"]) as tf:
        x, sr = sf.read(io.BytesIO(tf.extractfile(r["member"]).read()), dtype="float32")
    sf.write(A / "clips/voxlingua_fr.wav", x, sr)
    print(json.dumps({"lid_clip": r["key"], "lang": langs[r["label"]]}))
    T = json.loads((zslib.BENCH / "data/pr_targets.json").read_text())["test"]
    shutil.copy(T[0][0], A / "clips/librispeech_test_clean.flac")
    print(json.dumps({"phones_clip": T[0][0]}))
    w, sr = sf.read(zslib.BENCH / "data/libriparty/LibriParty/dataset/eval/session_0/session_0_mixture.wav", dtype="float32")
    sf.write(A / "clips/libriparty_eval_session0_60s.wav", w[:60 * sr], sr)
    if a.keyword_clip:
        shutil.copy(a.keyword_clip, A / "clips/keyword.wav")


if __name__ == "__main__":
    main()
