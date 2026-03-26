"""
Live microphone listener — runs all trained models in parallel and shows
a real-time confidence dashboard in the terminal.

Usage:
  python listen_all.py
  python listen_all.py --threshold 0.6
  python listen_all.py --models-dir experiments/hey_mycroft/models
  python listen_all.py --window 1.5 --stride 0.5
"""
import argparse
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Live multi-model wake-word dashboard")
parser.add_argument("--models-dir", default="experiments/hey_mycroft/models",
                    help="Directory containing model subdirs (default: experiments/hey_mycroft/models)")
parser.add_argument("--threshold", type=float, default=0.5,
                    help="Detection threshold (default: 0.5)")
parser.add_argument("--window", type=float, default=1.5,
                    help="Audio window length in seconds (default: 1.5)")
parser.add_argument("--stride", type=float, default=0.5,
                    help="Stride between windows in seconds (default: 0.5)")
parser.add_argument("--sr", type=int, default=16000,
                    help="Sample rate (default: 16000)")
parser.add_argument("--bar-width", type=int, default=30,
                    help="Width of confidence bar (default: 30)")
args = parser.parse_args()

# ── Discover models ───────────────────────────────────────────────────────────
models_dir = Path(args.models_dir)
if not models_dir.exists():
    sys.exit(f"ERROR: models dir not found: {models_dir}")

ARCHS = []
for d in sorted(models_dir.iterdir()):
    if not d.is_dir():
        continue
    feat = d / "best_f1_featurizer.onnx"
    head = d / "best_f1.onnx"
    if feat.exists() and head.exists():
        ARCHS.append((d.name, str(feat), str(head)))
    else:
        missing = []
        if not feat.exists(): missing.append("featurizer.onnx")
        if not head.exists(): missing.append("head.onnx")
        print(f"  [skip] {d.name}: missing {', '.join(missing)}", file=sys.stderr)

if not ARCHS:
    sys.exit("ERROR: no complete model dirs found (need best_f1_featurizer.onnx + best_f1.onnx)")

print(f"  Loaded {len(ARCHS)} models: {', '.join(a[0] for a in ARCHS)}")

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
latest_chunk = None
chunk_event = threading.Event()

# ── Per-model state ───────────────────────────────────────────────────────────
scores   = {name: 0.0 for name, *_ in ARCHS}
detected = {name: False for name, *_ in ARCHS}
det_times = {name: 0.0 for name, *_ in ARCHS}  # last detection timestamp
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
CLEAR_LINE = "\033[2K\r"

def bar(score, width, detected):
    filled = int(score * width)
    if detected:
        color = RED
    elif score > 0.3:
        color = YELLOW
    else:
        color = GREEN
    return color + "█" * filled + DIM + "░" * (width - filled) + RESET

def render_dashboard():
    now = time.time()
    lines = []
    lines.append(f"\033[H")  # move cursor to top
    lines.append(f"{BOLD}{CYAN}  Wake Word: hey mycroft  —  threshold={THRESHOLD:.2f}  —  {time.strftime('%H:%M:%S')}{RESET}")
    lines.append(f"  {DIM}{'Arch':<14} {'Confidence':>{BAR_W+2}}  {'Score':>6}  {'Status'}{RESET}")
    lines.append(f"  {'─'*60}")

    with score_lock:
        snap_scores   = dict(scores)
        snap_detected = dict(detected)
        snap_det_times = dict(det_times)

    for name, *_ in ARCHS:
        s = snap_scores[name]
        d = snap_detected[name]
        b = bar(s, BAR_W, d)
        # Flash "DETECTED" for 1.5s after last trigger
        since = now - snap_det_times[name]
        if since < 1.5:
            status = f"{RED}{BOLD}◉ DETECTED{RESET}"
        elif d:
            status = f"{RED}● yes{RESET}"
        else:
            status = f"{DIM}○ no{RESET}"
        lines.append(f"  {name:<14} {b}  {s:>6.3f}  {status}")

    lines.append(f"  {'─'*60}")
    lines.append(f"  {DIM}Ctrl+C to stop{RESET}  " + " " * 20)

    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()

# ── Main ──────────────────────────────────────────────────────────────────────
try:
    import sounddevice as sd
except ImportError:
    sys.exit("ERROR: sounddevice not installed.\nRun: pip install sounddevice")

# Clear screen and hide cursor
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
    # Show cursor, move to bottom
    sys.stdout.write("\033[?25h\033[{:d}B\n".format(len(ARCHS) + 6))
    sys.stdout.flush()
    print("  Stopped.")
