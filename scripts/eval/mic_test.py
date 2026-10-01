#!/usr/bin/env python3
"""Live microphone wake-word test — rolling-window mode (robust, any model).

Maintains a rolling window of raw audio and re-runs the whole-clip ONNX path
(``OnnxWakeWordInferencer.infer``) on it every hop. This is the recommended
live demo: it re-featurizes the window each step, so it has no chunk-boundary
artifacts and works for any trained model (MFCC/SincNet/HuBERT, FFN/GRU/...).

For the always-on, O(1)-per-frame stateful path (GRU heads), see
``mic_stream.py``.

Usage:
    python scripts/eval/mic_test.py --model-dir trained_models/.../hey_mycroft \
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
    ap.add_argument("--window", type=float, default=1.5, help="Rolling window seconds")
    ap.add_argument("--hop", type=float, default=0.25, help="Inference hop seconds")
    ap.add_argument("--patience", type=int, default=2, help="Consecutive hops to fire")
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
                                  frame_rate_hz=1.0 / args.hop)

    W, H = int(args.window * SR), int(args.hop * SR)
    buf = np.zeros(W, dtype=np.float32)
    q: queue.Queue = queue.Queue()
    sd.default.samplerate = SR

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy() if indata.ndim > 1 else indata.copy())

    print(f"=== LIVE (rolling window): say the wake word  (threshold={args.threshold}) ===")
    print("Ctrl+C to stop.\n")
    with sd.InputStream(channels=1, samplerate=SR, blocksize=H, dtype="float32", callback=cb):
        try:
            while True:
                block = q.get()
                n = len(block)
                buf = np.roll(buf, -n)
                buf[-n:] = block
                prob = inf.infer(buf)
                smoothed = smoother.update(prob)
                bar = "#" * int(smoothed * 40)
                tag = "WAKE!" if smoother.is_triggered() else " -- "
                print(f"[{tag}] {smoothed:.3f} {bar}", flush=True)
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
