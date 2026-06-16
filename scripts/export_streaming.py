#!/usr/bin/env python3
"""Export a *stateful streaming* GRU head ONNX from a trained checkpoint.

The streaming head carries the GRU hidden state + a sliding window of GRU
outputs as ONNX state I/O, so live detection costs O(1) per frame instead of
re-running the GRU over the whole window each step (the always-on / MCU path).
It is bit-exact with the batch head (export aborts if the parity check fails).

Reads ``<wake_word>_meta.json`` in the model dir to rebuild the architecture.

Usage:
    python scripts/export_streaming.py --model-dir trained_models/gru/hey_mycroft \
        --window 100
"""
import argparse
import glob
import json
import os

from ww_trainer.factory import create_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--name", default="best_f1", help="Checkpoint stem (default best_f1)")
    ap.add_argument("--window", type=int, default=100,
                    help="GRU-output frames to pool; ~frames in one training clip "
                         "(16 kHz MFCC ≈ 100 frames/s).")
    args = ap.parse_args()

    metas = glob.glob(os.path.join(args.model_dir, "*_meta.json"))
    if not metas:
        raise SystemExit(f"No *_meta.json in {args.model_dir}")
    meta = json.load(open(metas[0]))
    if meta.get("bidirectional"):
        raise SystemExit("Streaming export needs a unidirectional GRU (bidirectional=false).")

    model = create_model(
        "gru", "", featurizer_type="mfcc", feature_dim=None,
        sample_rate=meta.get("sample_rate", 16000), device="cpu",
        n_mfcc=meta.get("n_mfcc", 40), hidden_dim=meta.get("hidden_dim", 128),
        bidirectional=False, gru_n_layers=meta.get("gru_n_layers", 1),
    )
    ckpt = os.path.join(args.model_dir, f"{args.name}.pt")
    model.load_checkpoint(ckpt)
    model.eval()

    out = os.path.join(args.model_dir, f"{args.name}_streaming.onnx")
    model.classifier.export_streaming_onnx(
        out, window=args.window,
        metadata={"wake_word": meta.get("wake_word", ""), "mode": "streaming",
                  "window": str(args.window)})
    print(f"Wrote {out}")
    print(f"Run live with:  python scripts/eval/mic_stream.py --model-dir {args.model_dir} "
          f"--window {args.window}")


if __name__ == "__main__":
    main()
