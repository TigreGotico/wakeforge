"""
Live microphone listener — runs trained models in parallel and shows
a real-time confidence dashboard in the terminal.

Usage:
  python listen_all.py
  python listen_all.py --threshold 0.6
  python listen_all.py --models-dir experiments/hey_mycroft/ablation
  python listen_all.py --models-dir experiments/hey_mycroft/ablation --max-models 8
  python listen_all.py --window 1.5 --stride 0.5
"""
import argparse
import os
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Live multi-model wake-word dashboard")
parser.add_argument("--models-dir", default="experiments/hey_mycroft/models",
                    help="Directory to search recursively for model subdirs")
parser.add_argument("--max-models", type=int, default=5,
                    help="Max models to load; picks randomly if more are found (default: 5)")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--window", type=float, default=1.5,
                    help="Audio window length in seconds (default: 1.5)")
parser.add_argument("--stride", type=float, default=0.5,
                    help="Stride between windows in seconds (default: 0.5)")
parser.add_argument("--sr", type=int, default=16000)
parser.add_argument("--bar-width", type=int, default=30)
parser.add_argument("--seed", type=int, default=None,
                    help="Random seed for model selection (default: random)")
args = parser.parse_args()

# ── Discover models (recursive) ───────────────────────────────────────────────
models_dir = Path(args.models_dir)
if not models_dir.exists():
    sys.exit(f"ERROR: models dir not found: {models_dir}")

# Search recursively: a valid model dir has best_f1_featurizer.onnx + best_f1.onnx
candidates = []
skipped = []
for d in sorted(models_dir.rglob("*")):
    if not d.is_dir():
        continue
    feat = d / "best_f1_featurizer.onnx"
    head = d / "best_f1.onnx"
    if feat.exists() and head.exists():
        # Build a display name relative to models_dir, replacing path separators
        rel = d.relative_to(models_dir)
        display = str(rel).replace("/", "›")
        candidates.append((display, str(feat), str(head)))
    elif head.exists():
        skipped.append(f"{d.relative_to(models_dir)} (missing featurizer.onnx)")

if skipped:
    print(f"  [skip] {len(skipped)} dirs missing featurizer.onnx:", file=sys.stderr)
    for s in skipped[:5]:
        print(f"    {s}", file=sys.stderr)
    if len(skipped) > 5:
        print(f"    ... and {len(skipped)-5} more", file=sys.stderr)

if not candidates:
    sys.exit("ERROR: no complete model dirs found (need best_f1_featurizer.onnx + best_f1.onnx)")

# Random sampling when more candidates than --max-models
if len(candidates) > args.max_models:
    rng = random.Random(args.seed)
    selected = rng.sample(candidates, args.max_models)
    print(f"  Found {len(candidates)} models, randomly selected {args.max_models}:", file=sys.stderr)
    selected.sort(key=lambda x: x[0])
else:
    selected = candidates

ARCHS = selected
print(f"  Loaded {len(ARCHS)} models from {models_dir}:", file=sys.stderr)
for name, *_ in ARCHS:
    print(f"    {name}", file=sys.stderr)

# ── Load models ───────────────────────────────────────────────────────────────
from ww_trainer.inference import OnnxWakeWordInferencer

inferencer = {}
for name, feat_path, head_path in ARCHS:
    inferencer[name] = OnnxWakeWordInferencer(feat_path, head_path, device="cpu")

# ── Shared audio ring buffer ───────────────────────────────────────────────────
SR = args.sr
WINDOW_SAMPLES = int(args.window * SR)
STRIDE_SAMPLES = int(args.stride * SR)
THRESHOLD = args.threshold
BAR_W = args.bar_width

audio_buf = np.zeros(WINDOW_SAMPLES * 2, dtype=np.float32)
buf_lock  = threading.Lock()
chunk_event = threading.Event()

# ── Per-model state ───────────────────────────────────────────────────────────
scores    = {name: 0.0 for name, *_ in ARCHS}
detected  = {name: False for name, *_ in ARCHS}
det_times = {name: 0.0 for name, *_ in ARCHS}
score_lock = threading.Lock()

# ── Inference threads (one per model) ─────────────────────────────────────────
def inference_worker(name):
    model = inferencer[name]
    while True:
        chunk_event.wait()
        with buf_lock:
            window = audio_buf[-WINDOW_SAMPLES:].copy()
        score = float(model.infer(window))
        with score_lock:
            scores[name] = score
            detected[name] = score >= THRESHOLD
            if detected[name]:
                det_times[name] = time.time()

for name, *_ in ARCHS:
    t = threading.Thread(target=inference_worker, args=(name,), daemon=True)
    t.start()

# ── Audio callback ─────────────────────────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    global audio_buf
    chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
    with buf_lock:
        audio_buf = np.concatenate([audio_buf[len(chunk):], chunk])
    chunk_event.set()
    chunk_event.clear()

# ── Terminal dashboard ────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

def bar(score, width, is_detected):
    filled = int(score * width)
    color = RED if is_detected else (YELLOW if score > 0.3 else GREEN)
    return color + "█" * filled + DIM + "░" * (width - filled) + RESET

def render_dashboard():
    now = time.time()
    lines = ["\033[H"]
    lines.append(f"{BOLD}{CYAN}  Wake Word Dashboard  —  threshold={THRESHOLD:.2f}  —  {time.strftime('%H:%M:%S')}{RESET}")
    lines.append(f"  {DIM}{'Model':<28} {'Confidence':>{BAR_W+2}}  {'Score':>6}  {'Status'}{RESET}")
    lines.append(f"  {'─'*70}")

    with score_lock:
        snap_scores    = dict(scores)
        snap_detected  = dict(detected)
        snap_det_times = dict(det_times)

    for name, *_ in ARCHS:
        s   = snap_scores[name]
        d   = snap_detected[name]
        b   = bar(s, BAR_W, d)
        since = now - snap_det_times[name]
        if since < 1.5:
            status = f"{RED}{BOLD}◉ DETECTED{RESET}"
        elif d:
            status = f"{RED}● yes{RESET}"
        else:
            status = f"{DIM}○ no{RESET}"
        # Truncate long names for display
        disp = (name[:26] + "…") if len(name) > 27 else name
        lines.append(f"  {disp:<28} {b}  {s:>6.3f}  {status}")

    lines.append(f"  {'─'*70}")
    lines.append(f"  {DIM}Ctrl+C to stop  |  {len(ARCHS)} models active{RESET}  " + " " * 10)

    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()

# ── Main ──────────────────────────────────────────────────────────────────────
try:
    import sounddevice as sd
except ImportError:
    sys.exit("ERROR: sounddevice not installed.\nRun: pip install sounddevice")

sys.stdout.write("\033[2J\033[?25l")
sys.stdout.flush()
print(f"\033[H{BOLD}  Initializing microphone...{RESET}")
sys.stdout.flush()

try:
    with sd.InputStream(
        samplerate=SR,
        channels=1,
        dtype="float32",
        blocksize=STRIDE_SAMPLES,
        callback=audio_callback,
    ):
        while True:
            render_dashboard()
            time.sleep(0.1)

except KeyboardInterrupt:
    pass
finally:
    sys.stdout.write(f"\033[?25h\033[{len(ARCHS) + 6:d}B\n")
    sys.stdout.flush()
    print("  Stopped.")
