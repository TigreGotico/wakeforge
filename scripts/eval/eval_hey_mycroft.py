"""
Full evaluation of the trained "hey mycroft" model.
Produces: ROC/PR/DET plots, confidence histogram, learning curve,
FP/FN sample lists, detection report, and FP/hour estimate.
"""
import os, csv, logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import torch
torch.set_num_threads(4)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
EXP = Path("experiments/hey_mycroft")
MODEL_DIR = EXP / "model"
DATASET_DIR = EXP / "dataset"
EVAL_DIR = EXP / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH   = MODEL_DIR / "best_f1.onnx"
TEST_CSV    = DATASET_DIR / "test" / "metadata.csv"
METRICS_CSV = MODEL_DIR / "metrics_log.csv"
# Use downloaded negatives as ambient audio for FP/hour
AMBIENT_DIR = DATASET_DIR / "negatives"

# ── Load model (ONNX pipeline) ────────────────────────────────────────────────
# Two-file ONNX pipeline: featurizer (MFCC) + classifier head (FFN).
from ww_trainer.inference import OnnxWakeWordInferencer

FEAT_ONNX = MODEL_DIR / "best_f1_featurizer.onnx"
HEAD_ONNX = MODEL_DIR / "best_f1.onnx"

# Export featurizer ONNX if missing (checkpoint export only saves the head).
if not FEAT_ONNX.exists():
    logger.info("Featurizer ONNX missing — exporting from best_f1.pt ...")
    from ww_trainer.trainer import WakeWordTrainer
    _t = WakeWordTrainer(
        arch="ffn", featurizer="", feature_dim=None, featurizer_type="mfcc",
        wake_word="hey mycroft", device="cpu", hidden_dim=128, n_mfcc=40,
        losses_cfg=[{"name": "bce", "weight": 1.0}], export_onnx=False,
    )
    _t.model.load_checkpoint(str(MODEL_DIR / "best_f1.pt"))
    _t.model.eval()
    _t.model.feature_extractor.export_to_onnx(str(FEAT_ONNX))

model = OnnxWakeWordInferencer(str(FEAT_ONNX), str(HEAD_ONNX), device="cpu")
logger.info("Loaded ONNX pipeline: featurizer=%s  head=%s", FEAT_ONNX.name, HEAD_ONNX.name)

# ── Load test set ────────────────────────────────────────────────────────────
def read_csv(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(",", 1)
                if len(parts) == 2:
                    rows.append((parts[0], parts[1]))
    return rows

test_data = read_csv(TEST_CSV)
logger.info("Test set: %d samples", len(test_data))

# ── Run inference on test set ────────────────────────────────────────────────
import soundfile as sf

y_true, y_scores, paths_all = [], [], []
errors = []
for path, label in test_data:
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        prob = model.infer(wav)
        y_true.append(int(label))
        y_scores.append(float(prob))
        paths_all.append(path)
    except Exception as e:
        errors.append((path, str(e)))

y_true   = np.array(y_true)
y_scores = np.array(y_scores)
logger.info("Inference done. %d ok, %d errors", len(y_true), len(errors))

# ── Optimal threshold + DetectionReport ─────────────────────────────────────
from ww_trainer.metrics import (
    find_optimal_threshold, classification_report as detection_report,
    estimate_fp_per_hour
)

opt_thresh, best_f1_val = find_optimal_threshold(y_true, y_scores, criterion="f1")
report = detection_report(y_true, y_scores, threshold=opt_thresh)

print("\n══════════════════════════════════════════")
print("  Detection Report — hey mycroft (best_f1.onnx)")
print("══════════════════════════════════════════")
print(f"  Threshold (opt F1)  : {opt_thresh:.4f}")
print(f"  Accuracy            : {report.accuracy:.4f}")
print(f"  Precision           : {report.precision:.4f}")
print(f"  Recall              : {report.recall:.4f}")
print(f"  F1                  : {report.f1:.4f}")
print(f"  AUC (ROC)           : {report.auc:.4f}")
print(f"  EER                 : {report.eer:.4f}  (threshold={report.eer_threshold:.4f})")
print(f"  FAR @ threshold     : {report.far:.4f}")
print(f"  FRR @ threshold     : {report.frr:.4f}")
print(f"  Positives / Negatives: {report.n_positive} / {report.n_negative}")
print()
print("  FAR at fixed FRR targets:")
for frr_t, far_v in sorted(report.far_at_frr.items()):
    print(f"    FRR={frr_t*100:.0f}%  →  FAR={far_v*100:.3f}%")
print("══════════════════════════════════════════\n")

# ── FP / FN lists ────────────────────────────────────────────────────────────
preds = (y_scores >= opt_thresh).astype(int)
fp_list = [(p, s) for p, t, pr, s in zip(paths_all, y_true, preds, y_scores) if pr == 1 and t == 0]
fn_list = [(p, s) for p, t, pr, s in zip(paths_all, y_true, preds, y_scores) if pr == 0 and t == 1]

print(f"  False Positives : {len(fp_list)}")
for p, s in fp_list[:10]:
    print(f"    score={s:.3f}  {Path(p).name}")
if len(fp_list) > 10:
    print(f"    ... ({len(fp_list)-10} more)")

print(f"\n  False Negatives : {len(fn_list)}")
for p, s in fn_list[:10]:
    print(f"    score={s:.3f}  {Path(p).name}")
if len(fn_list) > 10:
    print(f"    ... ({len(fn_list)-10} more)")

# Save full FP/FN csvs
for kind, lst in [("false_positives", fp_list), ("false_negatives", fn_list)]:
    out = EVAL_DIR / f"{kind}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["score", "path"])
        for p, s in sorted(lst, key=lambda x: x[1], reverse=(kind == "false_positives")):
            w.writerow([f"{s:.4f}", p])
    logger.info("Saved %s → %s", kind, out)

# ── FP / hour on ambient audio ───────────────────────────────────────────────
ambient_paths = sorted(AMBIENT_DIR.rglob("*.wav"))[:200]   # cap at 200 files
if ambient_paths:
    logger.info("Estimating FP/hour on %d ambient files...", len(ambient_paths))
    fp_per_hour = estimate_fp_per_hour(
        infer_fn=model.infer,
        ambient_audio_paths=ambient_paths,
        threshold=opt_thresh,
        window_sec=1.5,
        stride_sec=0.5,
    )
    print(f"\n  FP/hour (ambient negatives, thresh={opt_thresh:.3f}): {fp_per_hour:.2f}")
    # Also at EER threshold
    fp_per_hour_eer = estimate_fp_per_hour(
        infer_fn=model.infer,
        ambient_audio_paths=ambient_paths,
        threshold=report.eer_threshold,
        window_sec=1.5,
        stride_sec=0.5,
    )
    print(f"  FP/hour (EER threshold={report.eer_threshold:.3f}):      {fp_per_hour_eer:.2f}")
else:
    logger.warning("No ambient audio found for FP/hour estimation")
    fp_per_hour = None

# ── Plot 1: Learning curves ───────────────────────────────────────────────────
epochs, losses, f1s, aucs, precs, recs = [], [], [], [], [], []
with open(METRICS_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        losses.append(float(row["loss"]))
        f1s.append(float(row["f1"]))
        aucs.append(float(row["auc"]))
        precs.append(float(row["precision"]))
        recs.append(float(row["recall"]))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("hey mycroft — Training Curves", fontsize=13, fontweight="bold")

axes[0].plot(epochs, losses, "tab:red", linewidth=2)
axes[0].set_title("BCE Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].grid(alpha=0.3)

axes[1].plot(epochs, f1s, "tab:blue", linewidth=2, label="F1")
axes[1].plot(epochs, precs, "tab:green", linewidth=1.5, linestyle="--", label="Precision")
axes[1].plot(epochs, recs, "tab:orange", linewidth=1.5, linestyle="--", label="Recall")
axes[1].set_title("F1 / Precision / Recall"); axes[1].set_xlabel("Epoch")
axes[1].set_ylim(0, 1.05); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(epochs, aucs, "tab:purple", linewidth=2)
axes[2].set_title("AUC-ROC"); axes[2].set_xlabel("Epoch")
axes[2].set_ylim(0.8, 1.005); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(EVAL_DIR / "learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Saved learning_curves.png")

# ── Plot 2: ROC curve ─────────────────────────────────────────────────────────
from sklearn.metrics import roc_curve, auc as sk_auc, precision_recall_curve
fpr_arr, tpr_arr, _ = roc_curve(y_true, y_scores)
roc_auc = sk_auc(fpr_arr, tpr_arr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_arr, tpr_arr, "tab:blue", linewidth=2, label=f"AUC = {roc_auc:.4f}")
ax.plot([0,1],[0,1],"k--", linewidth=1, alpha=0.4)
ax.scatter([report.far], [report.recall], color="red", zorder=5, label=f"Operating point (thresh={opt_thresh:.3f})")
ax.scatter([report.eer], [1-report.eer], color="orange", marker="*", s=120, zorder=5, label=f"EER = {report.eer:.4f}")
ax.set_xlabel("False Accept Rate (FAR)"); ax.set_ylabel("True Accept Rate (1-FRR)")
ax.set_title("ROC Curve — hey mycroft"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(EVAL_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Saved roc_curve.png")

# ── Plot 3: Precision-Recall curve ───────────────────────────────────────────
prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_scores)
pr_auc = sk_auc(rec_arr, prec_arr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec_arr, prec_arr, "tab:green", linewidth=2, label=f"AUC = {pr_auc:.4f}")
ax.scatter([report.recall], [report.precision], color="red", zorder=5, label=f"Operating point")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — hey mycroft"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(EVAL_DIR / "pr_curve.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Saved pr_curve.png")

# ── Plot 4: DET curve ─────────────────────────────────────────────────────────
from ww_trainer.metrics import det_curve
fars_d, frrs_d, _ = det_curve(y_true, y_scores, n_thresholds=500)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fars_d * 100, frrs_d * 100, "tab:red", linewidth=2)
ax.scatter([report.eer*100], [report.eer*100], color="orange", marker="*", s=150, zorder=5,
           label=f"EER = {report.eer*100:.2f}%")
ax.scatter([report.far*100], [report.frr*100], color="blue", zorder=5,
           label=f"Operating point")
ax.set_xlabel("FAR (%)"); ax.set_ylabel("FRR (%)")
ax.set_title("DET Curve — hey mycroft"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(EVAL_DIR / "det_curve.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Saved det_curve.png")

# ── Plot 5: Score distributions ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
pos_scores = y_scores[y_true == 1]
neg_scores = y_scores[y_true == 0]
bins = np.linspace(0, 1, 50)
ax.hist(neg_scores, bins=bins, alpha=0.6, color="tab:red",   label=f"Non-wake (n={len(neg_scores)})", density=True)
ax.hist(pos_scores, bins=bins, alpha=0.6, color="tab:green", label=f"Wake word (n={len(pos_scores)})", density=True)
ax.axvline(opt_thresh, color="black", linewidth=2, linestyle="--", label=f"Threshold = {opt_thresh:.3f}")
ax.axvline(report.eer_threshold, color="gray", linewidth=1.5, linestyle=":", label=f"EER threshold = {report.eer_threshold:.3f}")
ax.set_xlabel("Model Score"); ax.set_ylabel("Density")
ax.set_title("Score Distributions — hey mycroft"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(EVAL_DIR / "score_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Saved score_distributions.png")

# ── Plot 6: FAR at FRR operating points ──────────────────────────────────────
frr_targets = np.linspace(0.01, 0.30, 100)
far_at_frr_vals = []
for target in frr_targets:
    best = 1.0
    for far_v, frr_v in zip(fars_d, frrs_d):
        if frr_v <= target:
            best = min(best, far_v)
    far_at_frr_vals.append(best)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(frr_targets * 100, np.array(far_at_frr_vals) * 100, "tab:blue", linewidth=2)
ax.set_xlabel("FRR (%)"); ax.set_ylabel("FAR (%)")
ax.set_title("FAR vs FRR tradeoff — hey mycroft"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(EVAL_DIR / "far_vs_frr.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info("Saved far_vs_frr.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n  Plots saved to: {EVAL_DIR.resolve()}")
print(f"    learning_curves.png")
print(f"    roc_curve.png")
print(f"    pr_curve.png")
print(f"    det_curve.png")
print(f"    score_distributions.png")
print(f"    far_vs_frr.png")
print(f"    false_positives.csv  ({len(fp_list)} entries)")
print(f"    false_negatives.csv  ({len(fn_list)} entries)")
if fp_per_hour is not None:
    print(f"\n  FP/hour @ opt threshold : {fp_per_hour:.2f}")
    print(f"  FP/hour @ EER threshold : {fp_per_hour_eer:.2f}")
