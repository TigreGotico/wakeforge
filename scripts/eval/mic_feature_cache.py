#!/usr/bin/env python3
"""Live microphone wake-word test — feature-cache streaming mode.

Uses ``OnnxWakeWordInferencer.infer_streaming``: each chunk is featurized once
(only the new audio), appended to a rolling **feature** cache, and the head runs
over the cached window. Cheaper than the rolling-window demo (which re-featurizes
the whole window each hop) at the cost of minor chunk-boundary effects in the
features. Works for any head.

Mic mode trio:
  - mic_test.py          rolling raw-audio window  (model-agnostic, robust)
  - mic_feature_cache.py feature-cache streaming   (this file; cheaper)
  - mic_stream.py        stateful O(1) streaming   (GRU heads; always-on / MCU)

Usage:
    python scripts/eval/mic_feature_cache.py --model-dir trained_models/.../hey_mycroft \
        --threshold 0.5
"""
import argparse
import queue

import numpy as np
import sounddevice as sd

from ww_trainer.inference import OnnxWakeWordInferencer, PredictionSmoother

SR = 16000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", help="Dir with <name>_featurizer.onnx + <name>.onnx")
    ap.add_argument("--name", default="best_f1")
    ap.add_argument("--extractor-model", help="Featurizer ONNX (overrides --model-dir)")
    ap.add_argument("--classifier-model", help="Classifier ONNX (overrides --model-dir)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--chunk", type=float, default=0.1, help="Audio chunk seconds")
    ap.add_argument("--patience", type=int, default=2)
    args = ap.parse_args()

    if args.extractor_model and args.classifier_model:
        ext, clf = args.extractor_model, args.classifier_model
    elif args.model_dir:
        ext = f"{args.model_dir}/{args.name}_featurizer.onnx"
        clf = f"{args.model_dir}/{args.name}.onnx"
    else:
        ap.error("provide --model-dir or both --extractor-model and --classifier-model")

    inf = OnnxWakeWordInferencer(extractor_path=ext, head_path=clf)
    smoother = PredictionSmoother(method="ema", threshold=args.threshold,
                                  patience=args.patience, ema_alpha=0.5,
                                  frame_rate_hz=1.0 / args.chunk)

    H = int(args.chunk * SR)
    q: queue.Queue = queue.Queue()
    cache = None

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy() if indata.ndim > 1 else indata.copy())

    print(f"=== LIVE (feature-cache streaming): say the wake word  (threshold={args.threshold}) ===")
    print("Ctrl+C to stop.\n")
    with sd.InputStream(channels=1, samplerate=SR, blocksize=H, dtype="float32", callback=cb):
        try:
            while True:
                block = q.get()
                prob, cache = inf.infer_streaming(block, cache)
                smoothed = smoother.update(prob)
                bar = "#" * int(smoothed * 40)
                tag = "WAKE!" if smoother.is_triggered() else " -- "
                print(f"[{tag}] {smoothed:.3f} {bar}", flush=True)
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
