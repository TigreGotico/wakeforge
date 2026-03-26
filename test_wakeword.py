"""
CLI test script for exported wake word ONNX models.

Usage:
  # Test on an audio file
  python test_wakeword.py --dir experiments/hey_mycroft/models/small --audio sample.wav

  # Real-time microphone test
  python test_wakeword.py --dir experiments/hey_mycroft/models/small --mic

  # Override threshold
  python test_wakeword.py --dir experiments/hey_mycroft/models/small --audio sample.wav --threshold 0.7

  # Score every window in a file
  python test_wakeword.py --dir experiments/hey_mycroft/models/small --audio long_audio.wav --all-windows
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Test an exported wake word ONNX model",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument("--dir", required=True,
                    help="Model directory containing best_f1.onnx + best_f1_featurizer.onnx")
parser.add_argument("--audio", default=None,
                    help="Path to .wav file for file-based testing")
parser.add_argument("--mic", action="store_true",
                    help="Use microphone for real-time testing")
parser.add_argument("--threshold", type=float, default=None,
                    help="Detection threshold (default: 0.5)")
parser.add_argument("--window-sec", type=float, default=1.5,
                    help="Window length in seconds (default: 1.5)")
parser.add_argument("--stride-sec", type=float, default=0.5,
                    help="Stride between windows in seconds (default: 0.5)")
parser.add_argument("--sr", type=int, default=16000,
                    help="Sample rate (default: 16000)")
parser.add_argument("--all-windows", action="store_true",
                    help="Print score for every window (not just detections)")
args = parser.parse_args()

if not args.audio and not args.mic:
    parser.error("Specify --audio <file> or --mic")
if args.audio and args.mic:
    parser.error("--audio and --mic are mutually exclusive")

# ── Load model ────────────────────────────────────────────────────────────────
model_dir = Path(args.dir)
feat_onnx = model_dir / "best_f1_featurizer.onnx"
head_onnx = model_dir / "best_f1.onnx"

if not feat_onnx.exists():
    sys.exit(f"ERROR: featurizer ONNX not found at {feat_onnx}\n"
             f"Run train_full.py or eval_hey_mycroft.py to export it.")
if not head_onnx.exists():
    sys.exit(f"ERROR: head ONNX not found at {head_onnx}")

from ww_trainer.inference import OnnxWakeWordInferencer

model = OnnxWakeWordInferencer(str(feat_onnx), str(head_onnx), device="cpu")

threshold = args.threshold if args.threshold is not None else 0.5
sr = args.sr
window_samples = int(args.window_sec * sr)
stride_samples = int(args.stride_sec * sr)

print(f"\n  Model    : {model_dir.name}")
print(f"  Feat     : {feat_onnx.name}")
print(f"  Head     : {head_onnx.name}")
print(f"  Threshold: {threshold:.3f}")
print(f"  Window   : {args.window_sec}s  Stride: {args.stride_sec}s")
print()

# ── File mode ─────────────────────────────────────────────────────────────────
if args.audio:
    import soundfile as sf

    audio_path = Path(args.audio)
    if not audio_path.exists():
        sys.exit(f"ERROR: audio file not found: {audio_path}")

    wav, file_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    # Resample if needed
    if file_sr != sr:
        try:
            import resampy
            wav = resampy.resample(wav, file_sr, sr)
        except ImportError:
            print(f"  WARNING: file SR={file_sr} != {sr}; resampy not installed, using raw audio")

    duration = len(wav) / sr
    print(f"  File     : {audio_path.name}")
    print(f"  Duration : {duration:.2f}s  ({len(wav)} samples @ {sr} Hz)")
    print()

    # Single-window mode (file shorter than window)
    if len(wav) <= window_samples:
        # Pad or use as-is
        chunk = np.zeros(window_samples, dtype=np.float32)
        chunk[:len(wav)] = wav
        score = float(model.infer(chunk))
        detected = score >= threshold
        flag = "  *** DETECTED ***" if detected else ""
        print(f"  Score: {score:.4f}{flag}")
        sys.exit(0)

    # Sliding window
    detections = 0
    n_windows = 0
    t0 = time.time()

    print(f"  {'Time':>8}  {'Score':>7}  Detection")
    print("  " + "-" * 36)

    offset = 0
    while offset + window_samples <= len(wav):
        chunk = wav[offset:offset + window_samples]
        score = float(model.infer(chunk))
        t_sec = offset / sr
        detected = score >= threshold
        if detected:
            detections += 1
        if args.all_windows or detected:
            flag = " *** WAKE WORD ***" if detected else ""
            print(f"  {t_sec:>7.2f}s  {score:>7.4f}  {'YES' if detected else 'no'}{flag}")
        n_windows += 1
        offset += stride_samples

    elapsed = time.time() - t0
    print("  " + "-" * 36)
    print(f"\n  Windows scanned : {n_windows}")
    print(f"  Detections      : {detections}")
    print(f"  Inference time  : {elapsed*1000:.1f} ms total  ({elapsed/n_windows*1000:.2f} ms/window)")

# ── Microphone mode ───────────────────────────────────────────────────────────
elif args.mic:
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("ERROR: sounddevice not installed.\n"
                 "Install it with:  pip install sounddevice")

    print(f"  Listening... (Ctrl+C to stop)")
    print(f"  Speak the wake word near your microphone.")
    print()
    print(f"  {'Time':>8}  {'Score':>7}  Detection")
    print("  " + "-" * 36)

    buf = np.zeros(0, dtype=np.float32)
    start_time = time.time()

    def audio_callback(indata, frames, time_info, status):
        global buf
        if status:
            print(f"  [mic status: {status}]", file=sys.stderr)
        chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        buf = np.concatenate([buf, chunk])

    try:
        with sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocksize=stride_samples,
            callback=audio_callback,
        ):
            while True:
                time.sleep(args.stride_sec * 0.5)
                if len(buf) >= window_samples:
                    window = buf[-window_samples:].copy()
                    score = float(model.infer(window))
                    detected = score >= threshold
                    t_sec = time.time() - start_time
                    if detected or args.all_windows:
                        flag = " *** WAKE WORD ***" if detected else ""
                        print(f"  {t_sec:>7.2f}s  {score:>7.4f}  {'YES' if detected else 'no'}{flag}",
                              flush=True)
                    elif int(t_sec) % 5 == 0 and int(t_sec) > 0:
                        # Heartbeat every 5s so user knows it's running
                        print(f"  {t_sec:>7.2f}s  {score:>7.4f}  (listening...)", flush=True)
    except KeyboardInterrupt:
        print("\n\n  Stopped.")
