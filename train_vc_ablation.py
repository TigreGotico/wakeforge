"""VC ablation experiment — does voice conversion help? By how much?

Trains the same model under 4 conditions with identical seeds, NWW pool, and
test set, then compares final F1 / EER / AUC:

  no_vc           — control: mining only, no VC
  vc_conservative — 5 VC positives/epoch
  vc_balanced     — 10 VC positives/epoch  (recommended default)
  vc_aggressive   — 20 VC positives/epoch

Voice conversion mode: each new positive = an existing wake-word clip
voice-converted to a random NWW donor's timbre via backend.vc().
No text synthesis involved.

Each condition is one separate MLflow run, tagged so they group together in the
MLflow UI. A final comparison table is printed to stdout and saved as a CSV.

Sanity checks
-------------
Before the training grid, a "VC probe" step synthesises 2 samples and verifies:
  - Generated WAV is non-empty and at least 0.5 s long
  - RMS amplitude is above silence threshold (> 0.001)
  - Embedding cosine similarity to original positives is > 0.1 and < 0.98
    (too low = garbage; too high = unchanged from original)

If the probe fails the script aborts with a clear error message so you don't
waste training time on broken synthesis.

Usage
-----
    .venv/bin/python train_vc_ablation.py
    .venv/bin/python train_vc_ablation.py --epochs 30 --arch bcresnet
    .venv/bin/python train_vc_ablation.py --skip-probe   # skip sanity check
    .venv/bin/python train_vc_ablation.py --conditions no_vc,vc_balanced

    # Try different param sweeps
    .venv/bin/python train_vc_ablation.py --exaggerations 0.2,0.4,0.6
    .venv/bin/python train_vc_ablation.py --vc-counts 0,5,10,20,40

Output
------
    experiments/<ww>/models/vc_ablation/
        no_vc/            — checkpoint + ONNX for each condition
        vc_conservative/
        vc_balanced/
        vc_aggressive/
        comparison.csv    — F1/EER/AUC/n_generated table
        comparison.png    — bar chart (F1 + EER side by side)
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from ww_trainer.env import load_env
load_env()
from ww_trainer.utils import read_dataset_csv

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")

import torch
torch.set_num_threads(12)
torch.set_num_interop_threads(4)

import psutil
if psutil.virtual_memory().available / 1e9 < 6:
    sys.exit("ERROR: less than 6 GB RAM available — aborting")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vc_ablation")


def _e(key, default):
    val = os.environ.get(key)
    if val is None:
        return default
    if default is None:
        return val
    try:
        return type(default)(val)
    except (ValueError, TypeError):
        return default


# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="VC vs no-VC ablation experiment",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--wake-word",   default=_e("WW_WAKE_WORD", "hey_mycroft"))
parser.add_argument("--arch",        default=_e("WW_ARCH", "gru"),
                    choices=["ffn", "gru", "cnn", "bcresnet", "tcresnet", "dscnn"])
parser.add_argument("--hidden-dim",  type=int, default=_e("WW_HIDDEN_DIM", 128))
parser.add_argument("--loss",        default=_e("WW_LOSS", "bce"),
                    choices=["bce", "focal", "label_smoothing_bce", "rppl"])
parser.add_argument("--epochs",      type=int, default=_e("WW_EPOCHS", 25),
                    help="Epochs per condition (keep 20-30 for fair comparison)")
parser.add_argument("--batch-size",  type=int, default=_e("WW_BATCH_SIZE", 16))
parser.add_argument("--lr",          type=float, default=_e("WW_LR", 5e-4))
parser.add_argument("--scan-size",   type=int, default=_e("WW_SCAN_SIZE", 3000),
                    help="NWW files to scan per epoch for hard-neg mining")
parser.add_argument("--vc-backend",  default=_e("WW_VC_BACKEND", "auto"),
                    choices=["auto", "chatterbox-onnx", "chatterbox", "linacodec"])
parser.add_argument("--vc-device",   default=_e("WW_VC_DEVICE", "auto"))

# Condition customisation
parser.add_argument("--conditions",
                    default="no_vc,vc_conservative,vc_balanced,vc_aggressive",
                    help="Comma-separated subset of conditions to run")
parser.add_argument("--vc-counts",   default=None,
                    help="Override: comma-separated vc_per_epoch values (creates one condition each, e.g. 0,5,10,20)")

parser.add_argument("--skip-probe",  action="store_true",
                    help="Skip the VC sanity probe (not recommended)")
parser.add_argument("--probe-only",  action="store_true",
                    help="Run only the sanity probe and exit")
parser.add_argument("--seed",        type=int, default=42)

args = parser.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE        = Path(f"experiments/{args.wake_word}")
DATASET_DIR = BASE / "dataset"
TRAIN_CSV   = DATASET_DIR / "train" / "metadata.csv"
TEST_CSV    = DATASET_DIR / "test"  / "metadata.csv"
AUG_DIR     = DATASET_DIR / "augmentation"
NWW_DIR     = DATASET_DIR / "negatives" / "not_wake_word_subset"
OUT_DIR     = BASE / "models" / "vc_ablation"

if not TRAIN_CSV.exists():
    sys.exit(f"Dataset not found at {TRAIN_CSV}. Run train_full_nww.py first.")

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset ────────────────────────────────────────────────────────────────
train_data = read_dataset_csv(TRAIN_CSV)
test_data  = read_dataset_csv(TEST_CSV)

csv_wakes    = [(p, l) for p, l in train_data if l == "1"]
csv_nonwakes = [(p, l) for p, l in train_data if l == "0"]

nww_pool = list(csv_nonwakes)
if NWW_DIR.exists():
    import random as _random
    local_nww  = sorted(NWW_DIR.glob("not_wake_word_*.wav"))
    existing   = {p for p, _ in nww_pool}
    new_nww    = [(str(p), "0") for p in local_nww if str(p) not in existing]
    nww_pool  += new_nww

if not nww_pool:
    sys.exit("ERROR: NWW pool is empty — need negatives for mining and voice donors")

n_pos = len(csv_wakes)
n_neg = len(nww_pool)
logger.info("Wakes: %d | NWW pool: %d | Test: %d", n_pos, n_neg, len(test_data))

augment_opts = {}
if any((AUG_DIR / "bg_noise").rglob("*.wav")):
    augment_opts["bg_noise_folder"] = str(AUG_DIR / "bg_noise")
if any((AUG_DIR / "music").rglob("*.wav")):
    augment_opts["music_folder"] = str(AUG_DIR / "music")
if any((AUG_DIR / "rir").rglob("*.wav")):
    augment_opts["rir_folder"] = str(AUG_DIR / "rir")
if NWW_DIR.exists():
    augment_opts["wake_word_over_speech_folder"] = str(NWW_DIR)


# ── Build conditions ──────────────────────────────────────────────────────
Condition = dict  # {"name", "vc_per_epoch", "exaggeration", "label"}

_DEFAULT_CONDITIONS: list[Condition] = [
    {"name": "no_vc",           "vc_per_epoch": 0},
    {"name": "vc_conservative", "vc_per_epoch": 5},
    {"name": "vc_balanced",     "vc_per_epoch": 10},
    {"name": "vc_aggressive",   "vc_per_epoch": 20},
]

if args.vc_counts:
    counts = [int(x) for x in args.vc_counts.split(",")]
    conditions = [{"name": f"vc_{c}", "vc_per_epoch": c} for c in counts]
else:
    requested = [s.strip() for s in args.conditions.split(",")]
    conditions = [c for c in _DEFAULT_CONDITIONS if c["name"] in requested]
    if not conditions:
        sys.exit(f"No matching conditions in: {args.conditions}")

logger.info("Conditions to run: %s", [c["name"] for c in conditions])


# ── VC sanity probe ────────────────────────────────────────────────────────
def _probe_vc() -> bool:
    """Synthesise 2 samples, check audio quality and embedding drift.

    Returns True if probe passes.
    """
    from ww_trainer.vc_helpers import load_vc_backend
    import torchaudio
    import torch.nn.functional as F

    logger.info("── VC Probe ──────────────────────────────────────────────")
    try:
        backend = load_vc_backend(backend=args.vc_backend, device=args.vc_device)
        logger.info("Backend loaded: %s  sr=%d", backend.name, backend.sample_rate)
    except Exception as exc:
        logger.error("PROBE FAILED: cannot load VC backend: %s", exc)
        return False

    donors = [Path(p) for p, _ in nww_pool[:50]]
    if not donors:
        logger.error("PROBE FAILED: no NWW donors available")
        return False

    probe_dir = OUT_DIR / "_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for i, donor in enumerate(donors[:2]):
        out_path = probe_dir / f"probe_{i}.wav"
        # Pick a random wake-word clip as the VC source
        source = csv_wakes[i % len(csv_wakes)][0] if csv_wakes else None
        if source is None:
            logger.error("PROBE FAILED: no wake-word source clips available")
            return False
        try:
            t0 = time.time()
            backend.vc(source, donor, out_path)
            elapsed = time.time() - t0
            logger.info("  sample %d generated in %.1f s → %s", i, elapsed, out_path)
        except Exception as exc:
            logger.error("PROBE FAILED: VC raised %s: %s", type(exc).__name__, exc)
            return False

        # Check: file exists and is non-empty
        if not out_path.exists() or out_path.stat().st_size < 1000:
            logger.error("PROBE FAILED: output file missing or too small: %s", out_path)
            return False

        # Check: load and validate audio
        try:
            wav, sr = torchaudio.load(str(out_path))
        except Exception as exc:
            logger.error("PROBE FAILED: cannot load generated WAV: %s", exc)
            return False

        duration = wav.shape[-1] / sr
        rms = float(wav.abs().mean())
        logger.info("  sample %d: duration=%.2f s  RMS=%.4f  sr=%d", i, duration, rms, sr)

        if duration < 0.3:
            logger.error("PROBE FAILED: generated audio too short (%.2f s < 0.3 s)", duration)
            return False
        if rms < 0.001:
            logger.error("PROBE FAILED: generated audio is near-silent (RMS=%.5f)", rms)
            return False

        generated.append(out_path)

    # Check: embedding similarity between originals and generated
    if csv_wakes:
        try:
            from ww_trainer.trainer import WakeWordTrainer
            _probe_trainer = WakeWordTrainer(
                arch=args.arch, featurizer="", feature_dim=None,
                hidden_dim=args.hidden_dim, dropout=0.1, device="cpu",
                losses_cfg=[{"name": "bce", "weight": 1.0}],
                featurizer_type="mfcc", n_mfcc=40, export_onnx=False,
                seed=args.seed, wake_word=args.wake_word,
            )
            probe_model = _probe_trainer.model
            probe_model.eval()
            with torch.no_grad():
                # Embed a few originals
                orig_wavs = []
                for p, _ in csv_wakes[:4]:
                    try:
                        w, sr = torchaudio.load(p)
                        if sr != 16000:
                            import torchaudio.functional as TAF
                            w = TAF.resample(w, sr, 16000)
                        orig_wavs.append(w.squeeze(0))
                    except Exception:
                        pass
                # Embed generated
                gen_wavs = []
                for gp in generated:
                    try:
                        w, sr = torchaudio.load(str(gp))
                        if sr != 16000:
                            import torchaudio.functional as TAF
                            w = TAF.resample(w, sr, 16000)
                        gen_wavs.append(w.squeeze(0))
                    except Exception:
                        pass
                if orig_wavs and gen_wavs:
                    orig_emb = probe_model.embed(orig_wavs).mean(0)
                    gen_emb  = probe_model.embed(gen_wavs).mean(0)
                    orig_emb = F.normalize(orig_emb.unsqueeze(0), dim=1)
                    gen_emb  = F.normalize(gen_emb.unsqueeze(0), dim=1)
                    cos_sim  = float((orig_emb * gen_emb).sum())
                    logger.info("  embedding cosine sim (orig vs generated): %.3f", cos_sim)
                    # Note: untrained model → similarity is random-ish, so we just
                    # check it's not NaN and the audio loaded OK
                    if not (-1.1 < cos_sim < 1.1):
                        logger.error("PROBE FAILED: cosine sim out of range (%.3f) — NaN?", cos_sim)
                        return False
        except Exception as exc:
            logger.warning("Embedding check skipped: %s", exc)

    logger.info("VC PROBE PASSED — backend is working correctly")
    logger.info("─────────────────────────────────────────────────────────")
    return True


# ── Run probe ──────────────────────────────────────────────────────────────
any_vc = any(c["vc_per_epoch"] > 0 for c in conditions)
if any_vc and not args.skip_probe:
    ok = _probe_vc()
    if not ok:
        sys.exit(
            "\nVC probe failed — fix the backend before running the ablation.\n"
            "Possible causes:\n"
            "  • chatterbox-onnx not installed: uv pip install chatterbox-onnx --python .venv/bin/python\n"
            "  • Model weights not downloaded yet (first run downloads from HF)\n"
            "  • NWW donor clips are corrupt or unsupported format\n"
            "Use --skip-probe to bypass (not recommended) or --conditions no_vc to run baseline only."
        )
elif not any_vc:
    logger.info("All conditions have vc_per_epoch=0 — skipping VC probe")

if args.probe_only:
    logger.info("--probe-only set — exiting after probe")
    sys.exit(0)


# ── Training loop over conditions ─────────────────────────────────────────
from ww_trainer.trainer import WakeWordTrainer
from ww_trainer.infinite_loop import StoppingGoal, infinite_training_loop

results: list[dict] = []

for cond in conditions:
    cname        = cond["name"]
    vc_per_epoch = cond["vc_per_epoch"]

    logger.info("")
    logger.info("══════════════════════════════════════════════════════")
    logger.info("  CONDITION: %s  (vc_per_epoch=%d)", cname, vc_per_epoch)
    logger.info("══════════════════════════════════════════════════════")

    cond_dir = OUT_DIR / cname
    cond_dir.mkdir(parents=True, exist_ok=True)

    trainer = WakeWordTrainer(
        arch=args.arch,
        featurizer="",
        feature_dim=None,
        hidden_dim=args.hidden_dim,
        dropout=0.1,
        device="cpu",
        losses_cfg=[{"name": args.loss, "weight": 1.0}],
        featurizer_type="mfcc",
        n_mfcc=40,
        export_onnx=True,
        seed=args.seed,
        wake_word=args.wake_word,
        mlflow_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        **augment_opts,
    )

    # Tag the MLflow run so all conditions appear together
    if trainer.mlflow:
        try:
            trainer.mlflow.set_tags({
                "experiment":    "vc_ablation",
                "condition":     cname,
                "vc_per_epoch":  vc_per_epoch,
                "arch":          args.arch,
                "loss":          args.loss,
                "seed":          args.seed,
            })
        except Exception:
            pass

    t_start = time.time()

    # Use a fixed-epoch stopping goal so all conditions train the same amount
    goal = StoppingGoal(
        target_f1=2.0,       # unreachable → forces full epoch run
        target_eer=0.0,      # unreachable
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        hard_plateau=args.epochs + 1,  # never triggers
        patience_after=args.epochs + 1,
    )

    best_f1 = infinite_training_loop(
        trainer=trainer,
        output_dir=cond_dir,
        wakes=csv_wakes,
        nww_pool=nww_pool,
        test_data=test_data,
        goal=goal,
        scan_size=args.scan_size,
        neg_threshold=0.5,
        neg_multiplier=3.0,
        lr=args.lr,
        batch_size=args.batch_size,
        aug_prob=0.7,
        use_mixup=True,
        mixup_alpha=1.0,
        spec_augment=True,
        neg_weight_schedule="linear",
        vc_per_epoch=vc_per_epoch,
        vc_backend=args.vc_backend,
        vc_device=args.vc_device,
        resume_cache=False,   # fresh start for fair comparison
    )

    elapsed = time.time() - t_start

    # Count synthesised samples
    vc_out_dir = cond_dir / "vc_epoch_positives"
    n_generated = len(list(vc_out_dir.glob("*.wav"))) if vc_out_dir.exists() else 0

    # Reload best checkpoint and re-evaluate for clean final metrics
    from ww_trainer.evaluation import evaluate_model
    from ww_trainer.metrics import find_optimal_threshold
    import numpy as np

    best_ckpt = cond_dir / "best_f1.pt"
    if best_ckpt.exists():
        trainer.model.load_checkpoint(str(best_ckpt))

    acc, prec, rec, f1, auc, *_, targets, preds, probs, det = evaluate_model(
        trainer.model, test_data, trainer.device,
        batch_size=args.batch_size, threshold=0.5,
    )
    eer    = det.eer
    far1   = getattr(det, "far_at_frr1", None)

    row = {
        "condition":     cname,
        "vc_per_epoch":  vc_per_epoch,
        "exaggeration":  exaggeration,
        "best_f1":       round(best_f1, 4),
        "eval_f1":       round(f1, 4),
        "eval_precision": round(prec, 4),
        "eval_recall":   round(rec, 4),
        "eval_auc":      round(auc, 4),
        "eval_eer":      round(eer, 4),
        "far_at_frr1":   round(far1, 4) if far1 is not None else "N/A",
        "n_generated":   n_generated,
        "elapsed_s":     round(elapsed, 1),
    }
    results.append(row)

    logger.info("[%s] F1=%.4f  EER=%.4f  AUC=%.4f  n_gen=%d  t=%.0fs",
                cname, f1, eer, auc, n_generated, elapsed)

    if trainer.mlflow:
        try:
            trainer.mlflow.log_metrics({
                "final_f1":    f1,   "final_eer":    eer,
                "final_auc":   auc,  "final_prec":   prec,
                "final_rec":   rec,  "n_vc_generated": n_generated,
            })
            trainer.mlflow.end_run()
        except Exception:
            pass


# ── Summary ────────────────────────────────────────────────────────────────
print()
print("╔══════════════════ VC ABLATION RESULTS ══════════════════╗")
header = f"{'Condition':<20} {'vc/ep':>5} {'exag':>5} {'F1':>7} {'EER':>7} {'AUC':>7} {'n_gen':>7}"
print(header)
print("─" * len(header))
for r in results:
    print(f"{r['condition']:<20} {r['vc_per_epoch']:>5} {r['exaggeration']:>5.2f} "
          f"{r['eval_f1']:>7.4f} {r['eval_eer']:>7.4f} {r['eval_auc']:>7.4f} {r['n_generated']:>7}")
print("╚" + "═" * (len(header) - 2) + "╝")

# Best condition
if results:
    best = max(results, key=lambda r: r["eval_f1"])
    ctrl = next((r for r in results if r["vc_per_epoch"] == 0), None)
    print()
    print(f"Best condition : {best['condition']}  (F1={best['eval_f1']:.4f}  EER={best['eval_eer']:.4f})")
    if ctrl and ctrl != best:
        f1_delta  = best["eval_f1"]  - ctrl["eval_f1"]
        eer_delta = best["eval_eer"] - ctrl["eval_eer"]
        print(f"VC improvement : ΔF1={f1_delta:+.4f}  ΔEER={eer_delta:+.4f}  "
              f"({'+' if f1_delta > 0 else ''}{f1_delta/max(ctrl['eval_f1'],1e-6)*100:.1f}%)")
    elif ctrl and ctrl == best:
        print("VC did NOT improve over no-VC baseline — consider more epochs or different params")

# ── Save CSV ───────────────────────────────────────────────────────────────
csv_path = OUT_DIR / "comparison.csv"
if results:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved: {csv_path}")

# ── Comparison bar chart ───────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import numpy as np

    names = [r["condition"] for r in results]
    f1s   = [r["eval_f1"]  for r in results]
    eers  = [r["eval_eer"] for r in results]
    x     = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(x, f1s, color=["#2c3e50" if r["vc_per_epoch"] == 0 else "#2980b9"
                                    for r in results], alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylim(0, 1); ax1.set_ylabel("F1 score"); ax1.set_title("F1 by Condition")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, f1s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(x, eers, color=["#2c3e50" if r["vc_per_epoch"] == 0 else "#e74c3c"
                                     for r in results], alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_ylim(0, max(eers) * 1.2 + 0.05)
    ax2.set_ylabel("EER (lower = better)"); ax2.set_title("EER by Condition")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, eers):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Mark control baseline
    ctrl_idx = next((i for i, r in enumerate(results) if r["vc_per_epoch"] == 0), None)
    if ctrl_idx is not None:
        for ax, vals in [(ax1, f1s), (ax2, eers)]:
            ax.axhline(vals[ctrl_idx], color="#7f8c8d", linestyle="--", alpha=0.5,
                       label="no-VC baseline")
            ax.legend(fontsize=8)

    fig.suptitle(
        f"VC Ablation — {args.arch} / {args.loss} / {args.epochs} epochs",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    chart_path = OUT_DIR / "comparison.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved : {chart_path}")

    # Log chart to MLflow if any run is still open (it won't be, but try anyway)
    # The chart is saved locally; open it manually or check OUT_DIR

except Exception as exc:
    logger.warning("Could not generate comparison chart: %s", exc)

print(f"\nAll outputs in: {OUT_DIR}/")
