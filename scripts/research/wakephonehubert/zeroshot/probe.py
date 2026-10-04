"""Training-free nearest-class-centroid and a linear probe on pooled vectors, with train_pooled.py's budget."""
import numpy as np


def standardise(X, tr):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-5
    return (X - mu) / sd


def ncc(X, y, tr, te, C):
    """Cosine nearest-centroid after standardising with training statistics. Returns (scores [len(te), C], accuracy)."""
    Z = standardise(X.astype(np.float64), tr)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    cen = np.stack([Z[tr][y[tr] == c].mean(0) for c in range(C)])
    cen /= np.linalg.norm(cen, axis=1, keepdims=True) + 1e-12
    s = Z[te] @ cen.T
    return s, float((s.argmax(1) == y[te]).mean())


def linear_probe(X, y, tr, dv, te, C, seed, lr=1e-3, batch=256, epochs=50, patience=5):
    """Standardisation (training statistics) + one linear layer, Adam, early stop on dev accuracy (patience 5); the
    best-dev state is tested. Returns (test logits, test accuracy, best dev accuracy, best epoch)."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Z = torch.as_tensor(standardise(X.astype(np.float32), tr), dtype=torch.float32)
    Y = torch.as_tensor(y)
    lin = torch.nn.Linear(Z.shape[1], C)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)

    def acc(idx):
        with torch.no_grad():
            return float((lin(Z[idx]).argmax(-1) == Y[idx]).float().mean())

    best = (-1.0, -1, None)
    for ep in range(epochs):
        perm = rng.permutation(tr)
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            loss = F.cross_entropy(lin(Z[b]), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        da = acc(dv)
        if da > best[0]:
            best = (da, ep, {k: v.clone() for k, v in lin.state_dict().items()})
        elif ep - best[1] >= patience:
            break
    lin.load_state_dict(best[2])
    with torch.no_grad():
        lg = lin(Z[te]).numpy()
    return lg, float((lg.argmax(1) == y[te]).mean()), best[0], best[1]
