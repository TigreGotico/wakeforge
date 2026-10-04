"""ESC-50, five folds: zero-shot and trained rows for WakePhoneHuBERT's `features`, and EfficientAT mn10_as as an
outside sound-event model for reference.

  extract   ONNX `features` per clip, mean-pooled -> results/sound/onnx.npz; EfficientAT mn10_as clip logits
            (AudioSet, 527 classes) -> results/sound/reference-efficientat.npz
  score     zero-shot: cosine nearest-class-centroid on mean-pooled features, centroids from the three training folds
            of each test fold (train_pooled.py's split: fold k tested, fold k+1 dev, the other three train);
            trained: linear probe on the same vectors, seeds 0 and 1, train_pooled.py's budget;
            reference: EfficientAT's AudioSet classifier mapped to the ESC-50 classes (score of a class = max logit
            over its AudioSet classes in the table below; nothing trained)
"""
import argparse
import csv
import io
import json
import os
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import probe  # noqa: E402
import zslib  # noqa: E402

ZIP = zslib.BENCH / "data/esc50/ESC-50-master.zip"
OD = zslib.ZS / "results/sound"
D = 521

ESC_TO_AUDIOSET = {
    "dog": ["Dog", "Bark", "Bow-wow", "Yip", "Howl", "Growling", "Whimper (dog)"],
    "rooster": ["Chicken, rooster", "Crowing, cock-a-doodle-doo"],
    "pig": ["Pig", "Oink"],
    "cow": ["Cattle, bovinae", "Moo"],
    "frog": ["Frog", "Croak"],
    "cat": ["Cat", "Meow", "Purr", "Caterwaul"],
    "hen": ["Fowl", "Cluck"],
    "insects": ["Insect", "Fly, housefly", "Bee, wasp, etc.", "Buzz", "Mosquito"],
    "sheep": ["Sheep", "Bleat"],
    "crow": ["Crow", "Caw"],
    "rain": ["Rain", "Raindrop", "Rain on surface"],
    "sea_waves": ["Ocean", "Waves, surf"],
    "crackling_fire": ["Fire", "Crackle"],
    "crickets": ["Cricket"],
    "chirping_birds": ["Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet"],
    "water_drops": ["Drip"],
    "wind": ["Wind", "Rustling leaves", "Wind noise (microphone)"],
    "pouring_water": ["Pour", "Fill (with liquid)", "Trickle, dribble", "Water tap, faucet"],
    "toilet_flush": ["Toilet flush"],
    "thunderstorm": ["Thunderstorm", "Thunder"],
    "crying_baby": ["Baby cry, infant cry"],
    "sneezing": ["Sneeze"],
    "clapping": ["Clapping", "Applause"],
    "breathing": ["Breathing", "Gasp", "Pant", "Wheeze"],
    "coughing": ["Cough", "Throat clearing"],
    "footsteps": ["Walk, footsteps"],
    "laughing": ["Laughter", "Giggle", "Chuckle, chortle", "Belly laugh", "Snicker", "Baby laughter"],
    "brushing_teeth": ["Toothbrush", "Electric toothbrush"],
    "snoring": ["Snoring"],
    "drinking_sipping": ["Liquid", "Gurgling", "Gargling"],
    "door_wood_knock": ["Knock"],
    "mouse_click": ["Clicking"],
    "keyboard_typing": ["Typing", "Computer keyboard", "Typewriter"],
    "door_wood_creaks": ["Creak", "Squeak"],
    "can_opening": ["Burst, pop"],
    "washing_machine": ["Whir", "Mechanical fan"],
    "vacuum_cleaner": ["Vacuum cleaner"],
    "clock_alarm": ["Alarm clock", "Alarm"],
    "clock_tick": ["Tick-tock", "Tick", "Clock"],
    "glass_breaking": ["Shatter", "Glass"],
    "helicopter": ["Helicopter"],
    "chainsaw": ["Chainsaw"],
    "siren": ["Siren", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)", "Civil defense siren"],
    "car_horn": ["Vehicle horn, car horn, honking", "Toot"],
    "engine": ["Engine", "Idling", "Engine starting", "Medium engine (mid frequency)", "Heavy engine (low frequency)", "Light engine (high frequency)"],
    "train": ["Train", "Rail transport", "Railroad car, train wagon", "Train wheels squealing"],
    "church_bells": ["Church bell", "Change ringing (campanology)"],
    "airplane": ["Aircraft", "Fixed-wing aircraft, airplane", "Aircraft engine", "Jet engine", "Propeller, airscrew"],
    "fireworks": ["Fireworks", "Firecracker"],
    "hand_saw": ["Sawing"],
}


def rows():
    with zipfile.ZipFile(ZIP) as z:
        r = list(csv.DictReader(io.TextIOWrapper(z.open("ESC-50-master/meta/esc50.csv"))))
    return r


def to16k(x, sr):
    from math import gcd
    from scipy.signal import resample_poly
    if x.ndim > 1:
        x = x.mean(1)
    g = gcd(16000, sr)
    return np.ascontiguousarray(resample_poly(x, 16000 // g, sr // g), dtype=np.float32)


_S = {}


def onnx_one(fn):
    if "s" not in _S:
        _S["s"] = zslib.session("wakephonehubert_features_int8.onnx", threads=1)
        _S["z"] = zipfile.ZipFile(ZIP)
    x, sr = sf.read(io.BytesIO(_S["z"].read("ESC-50-master/audio/" + fn)), dtype="float32")
    return zslib.run(_S["s"], to16k(x, sr), ("features",))["features"].mean(0)


def mapping_matrix():
    names = [r[2] for r in list(csv.reader(open(zslib.EFFICIENTAT / "metadata/class_labels_indices.csv")))[1:]]
    idx = {n: i for i, n in enumerate(names)}
    missing = [n for v in ESC_TO_AUDIOSET.values() for n in v if n not in idx]
    assert not missing, missing
    return names, idx


def eat():
    cwd = os.getcwd()
    eat_dir = zslib.EFFICIENTAT.resolve()
    sys.path.insert(0, str(eat_dir))
    os.chdir(eat_dir)
    from models.mn.model import get_model
    from models.preprocess import AugmentMelSTFT
    m = get_model(width_mult=1.0, pretrained_name="mn10_as").eval()
    mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320).eval()
    os.chdir(cwd)
    return m, mel


def extract():
    OD.mkdir(parents=True, exist_ok=True)
    R = rows()
    with Pool(6) as p:
        F = np.stack(p.map(onnx_one, [r["filename"] for r in R], chunksize=8)).astype(np.float32)
    np.savez(OD / "onnx.npz", files=np.asarray([r["filename"] for r in R]), label=np.asarray([int(r["target"]) for r in R]),
             fold=np.asarray([int(r["fold"]) for r in R]), category=np.asarray([r["category"] for r in R]), features_mean=F)
    if (OD / "reference-efficientat.npz").exists():
        return
    import torch
    import torch.nn.functional as Fn
    import torchaudio
    torch.set_num_threads(4)
    m, mel = eat()
    z = zipfile.ZipFile(ZIP)
    lg = []
    with torch.no_grad():
        for r in R:
            x, sr = sf.read(io.BytesIO(z.read("ESC-50-master/audio/" + r["filename"])), dtype="float32")
            w = torch.as_tensor(to16k(x, sr))[None]
            if w.shape[1] < 5 * 16000:
                w = Fn.pad(w, (5 * 16000 - w.shape[1], 0))
            lg.append(m(mel(torchaudio.functional.resample(w, 16000, 32000)).unsqueeze(1))[0][0].numpy())
    np.savez(OD / "reference-efficientat.npz", logits=np.stack(lg))


def score():
    import torch
    torch.set_num_threads(4)
    O = np.load(OD / "onnx.npz")
    y, fold, cat = O["label"], O["fold"], O["category"]
    X = O["features_mean"][:, :D]
    cats = [None] * 50
    for c, l in zip(cat, y):
        cats[l] = str(c)
    names, idx = mapping_matrix()
    groups = [[idx[n] for n in ESC_TO_AUDIOSET[c]] for c in cats]
    ref_logits = np.load(OD / "reference-efficientat.npz")["logits"]
    s_ref = np.stack([ref_logits[:, g].max(1) for g in groups], 1)

    def per_fold(s):
        return [float((s[fold == k].argmax(1) == y[fold == k]).mean()) for k in range(1, 6)]
    res = {"task": "sound", "set": "ESC-50, 2,000 clips, 5 folds of 400", "features_dims": D,
           "reference_efficientat_mapped": {"folds": per_fold(s_ref), "mean": float(np.mean(per_fold(s_ref)))}}
    raw = {"files": O["files"], "label": y, "fold": fold, "categories": np.asarray(cats), "reference_efficientat_esc_scores": s_ref}
    accs, sc = [], np.zeros((len(y), 50), np.float32)
    for k in range(1, 6):
        dk = k % 5 + 1
        tr = np.where((fold != k) & (fold != dk))[0]; te = np.where(fold == k)[0]
        s, a = probe.ncc(X, y, tr, te, 50)
        sc[te] = s; accs.append(a)
    res["zeroshot_ncc_features_mean"] = {"folds": accs, "mean": float(np.mean(accs))}
    raw["ncc_scores"] = sc
    for seed in (0, 1):
        accs, lg = [], np.zeros((len(y), 50), np.float32)
        for k in range(1, 6):
            dk = k % 5 + 1
            tr = np.where((fold != k) & (fold != dk))[0]; dv = np.where(fold == dk)[0]; te = np.where(fold == k)[0]
            l, a, _, _ = probe.linear_probe(X, y, tr, dv, te, 50, seed)
            lg[te] = l; accs.append(a)
        res[f"probe_features_mean_seed{seed}"] = {"folds": accs, "mean": float(np.mean(accs))}
        raw[f"probe_logits_seed{seed}"] = lg
    np.savez_compressed(OD / "raw.npz", **raw)
    (OD / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v["mean"] if isinstance(v, dict) and "mean" in v else v) for k, v in res.items()}), flush=True)


def main():
    global OD, ZIP
    ap = argparse.ArgumentParser()
    zslib.add_args(ap)
    ap.add_argument("cmd", choices=["extract", "score"])
    a = ap.parse_args()
    zslib.configure(a)
    OD = zslib.ZS / "results/sound"
    ZIP = zslib.BENCH / "data/esc50/ESC-50-master.zip"
    {"extract": extract, "score": score}[a.cmd]()


if __name__ == "__main__":
    main()
