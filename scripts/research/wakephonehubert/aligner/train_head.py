"""train_head.py [--hours 30] [--epochs 14] [--dev 400] [--out head] [--work-dir ...] [--export-dir ...] [--librispeech ...]

Train the approach-B aligner head on frozen WakePhoneHuBERT outputs, computed on the fly
(no feature cache), against frame labels from the MFA phone alignments of train-clean-100."""
import argparse, json, random, time
import numpy as np, soundfile as sf, torch, onnxruntime as ort
from torch.utils.data import Dataset, DataLoader
import common
from common import *
from head import AlignHead, viterbi_align

ap = argparse.ArgumentParser()
common.add_args(ap)
ap.add_argument("--hours", type=float, default=30)
ap.add_argument("--epochs", type=int, default=14)
ap.add_argument("--dev", type=int, default=400, help="development utterances")
ap.add_argument("--out", default="head", help="output directory under --work-dir")
args = ap.parse_args()
common.configure(args)
ROOT, WPH, LIBRI = common.ROOT, common.WPH, common.LIBRI
SEED, DEV_SPK, BS = 0, 12, 16
HOURS, EPOCHS = args.hours, args.epochs
out = ROOT / args.out
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
out.mkdir(parents=True, exist_ok=True)
TRAIN_WPH = WPH

rows = read_jsonl(ROOT / "data" / "train_clean_100.alignments.jsonl")
rows = [r for r in rows if r["words"] and r["phonemes"]]
spk = sorted({r["id"].split("-")[0] for r in rows})
dev_spk = set(random.Random(SEED).sample(spk, DEV_SPK))


def path_of(uid):
    s, c, _ = uid.split("-")
    return LIBRI / "train-clean-100" / s / c / f"{uid}.flac"


pool = [r for r in rows if r["id"].split("-")[0] not in dev_spk]
random.Random(SEED).shuffle(pool)
train, h = [], 0.0
for r in pool:
    if h >= HOURS * 3600:
        break
    d = sf.info(str(path_of(r["id"]))).duration
    r["dur"] = d; h += d; train.append(r)
dev = [r for r in rows if r["id"].split("-")[0] in dev_spk]
random.Random(SEED).shuffle(dev)
dev = dev[:args.dev]
split = dict(seed=SEED, train_hours=h / 3600, train_utts=len(train), train_speakers=len({r["id"].split("-")[0] for r in train}),
             dev_speakers=sorted(dev_spk), dev_utts=len(dev), train_ids=[r["id"] for r in train], dev_ids=[r["id"] for r in dev])
json.dump(split, open(out / "split.json", "w"))
print(f"train {len(train)} utts {h/3600:.2f} h, {split['train_speakers']} speakers; dev {len(dev)} utts from {DEV_SPK} held-out speakers", flush=True)


def frame_labels(phonemes, T):
    y = np.zeros(T, np.int64)
    c = np.arange(T) * HOP + HOP / 2
    for p in phonemes:
        y[(c >= p["start"]) & (c < p["end"])] = ARPA_ID[strip_stress(p["phoneme"])]
    return y


class DS(Dataset):
    def __init__(self, items):
        self.items, self.sess = items, None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.sess is None:
            so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
            self.sess = ort.InferenceSession(str(TRAIN_WPH), so, providers=["CPUExecutionProvider"])
        r = self.items[i]
        x, _ = sf.read(str(path_of(r["id"])), dtype="float32")
        L, F = self.sess.run(["layers", "features"], {"waveform": x[None]})
        T = L.shape[1]
        return torch.from_numpy(L[0]).half(), torch.from_numpy(F[0, :, :521]).half(), torch.from_numpy(frame_labels(r["phonemes"], T)), i


def collate(b):
    T = max(x[0].shape[0] for x in b)
    L = torch.zeros(len(b), T, 9, 256, dtype=torch.half); F = torch.zeros(len(b), T, 521, dtype=torch.half)
    Y = torch.full((len(b), T), -100, dtype=torch.long)
    for k, (l, f, y, _) in enumerate(b):
        L[k, :len(y)] = l; F[k, :len(y)] = f; Y[k, :len(y)] = y
    return L, F, Y, [x[3] for x in b]


def boundary_eval(model, loader, items):
    model.eval(); errs, correct, total = [], 0, 0
    with torch.no_grad():
        for L, F, Y, idx in loader:
            lp = model(L.cuda().float(), F.cuda().float()).cpu()
            m = Y >= 0
            correct += (lp.argmax(-1)[m] == Y[m]).sum().item(); total += m.sum().item()
            for k, i in enumerate(idx):
                r = items[i]
                T = int(m[k].sum())
                ph = word_phones(r["words"], r["phonemes"])
                seq = [[ARPA_ID[strip_stress(p["phoneme"])] for p in g] for g in ph]
                al = viterbi_align(lp[k, :T].numpy(), seq)
                ref = [p for g in ph for p in g]; hyp = [p for w in al for p in w]
                errs += [abs(a[1] - b["start"]) for a, b in zip(hyp, ref)] + [abs(a[2] - b["end"]) for a, b in zip(hyp, ref)]
    model.train(); e = np.array(errs)
    return correct / total, float(np.mean(e <= 0.02 + 1e-9)), float(np.mean(e) * 1000)


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.2)
    g = torch.Generator(); g.manual_seed(SEED)
    tl = DataLoader(DS(train), batch_size=BS, shuffle=True, collate_fn=collate, num_workers=6, generator=g,
                    persistent_workers=True, prefetch_factor=4)
    dl = DataLoader(DS(dev), batch_size=BS, collate_fn=collate, num_workers=4)
    model = AlignHead().cuda()
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    steps = EPOCHS * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.05)
    best, log = -1, []
    for ep in range(EPOCHS):
        t0, tot, n = time.time(), 0.0, 0
        for L, F, Y, _ in tl:
            lp = model(L.cuda().float(), F.cuda().float())
            loss = torch.nn.functional.nll_loss(lp.reshape(-1, lp.shape[-1]), Y.cuda().reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item(); n += 1
        acc, w20, mae = boundary_eval(model, dl, dev)
        rec = dict(epoch=ep, train_loss=tot / n, dev_frame_acc=acc, dev_phone_bnd_within20=w20, dev_phone_bnd_mae_ms=mae,
                   sec=time.time() - t0, mix=torch.softmax(model.mix, 0).tolist())
        log.append(rec); print(json.dumps(rec), flush=True)
        if w20 > best:
            best = w20; torch.save(model.state_dict(), out / "aligner_head.pt")
    json.dump(log, open(out / "train_log.json", "w"), indent=1)
