#!/usr/bin/env python3
"""Live microphone wake-word test — stateful streaming mode (O(1) per frame).

Uses ``OnnxStreamingWakeWord``: the GRU hidden state and a sliding window of GRU
outputs are carried as ONNX state, so each audio chunk costs O(chunk) regardless
of history — the always-on / MCU deployment pattern. ONNX only, no PyTorch.

Requires the streaming head ONNX (``<name>_streaming.onnx``); produce it with
``scripts/export_streaming.py``.

Contrast with ``mic_test.py`` (rolling-window), which re-featurizes the whole
window each hop — simpler and model-agnostic, but O(window) per step.

Usage:
    python scripts/eval/mic_stream.py --model-dir trained_models/gru/hey_mycroft \
        --window 100 --threshold 0.6
"""
import argparse
import glob
import json
import os
import queue

import numpy as np
import sounddevice as sd

from ww_trainer.inference import OnnxStreamingWakeWord, PredictionSmoother

SR = 16000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--name", default="best_f1")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--chunk", type=float, default=0.1, help="Audio chunk seconds")
    ap.add_argument("--patience", type=int, default=2)
    args = ap.parse_args()

    feat = f"{args.model_dir}/{args.name}_featurizer.onnx"
    head = f"{args.model_dir}/{args.name}_streaming.onnx"
    if not os.path.exists(head):
        raise SystemExit(f"{head} not found — run:\n"
                         f"  python scripts/export_streaming.py --model-dir {args.model_dir} "
                         f"--window {args.window}")

    hidden = 128
    metas = glob.glob(os.path.join(args.model_dir, "*_meta.json"))
    if metas:
        hidden = json.load(open(metas[0])).get("hidden_dim", 128)

    sw = OnnxStreamingWakeWord(feat, head, window=args.window, hidden_dim=hidden)
    smoother = PredictionSmoother(method="ema", threshold=args.threshold,
                                  patience=args.patience, ema_alpha=0.5,
                                  frame_rate_hz=1.0 / args.chunk)

    H = int(args.chunk * SR)
    q: queue.Queue = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy() if indata.ndim > 1 else indata.copy())

    print(f"=== LIVE (stateful streaming): say the wake word  (threshold={args.threshold}) ===")
    print("Ctrl+C to stop.\n")
    with sd.InputStream(channels=1, samplerate=SR, blocksize=H, dtype="float32", callback=cb):
        try:
            while True:
                block = q.get()
                prob = sw.push(block)
                smoothed = smoother.update(prob)
                bar = "#" * int(smoothed * 40)
                tag = "WAKE!" if smoother.is_triggered() else " -- "
                print(f"[{tag}] {smoothed:.3f} {bar}", flush=True)
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
