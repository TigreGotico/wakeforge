#!/usr/bin/env python3
"""All four inference modes on one model, side by side.

ww-trainer can run a trained model four ways. They trade off how much audio is
reconsidered each step against compute cost. This example builds one GRU model,
exports the ONNX it needs, and runs the same audio through every mode so you can
see the APIs and how the scores relate.

  1. Batch / whole-clip     — OnnxWakeWordInferencer.infer(audio)
  2. Rolling window         — infer() over a sliding raw-audio buffer
  3. Feature-cache stream   — OnnxWakeWordInferencer.infer_streaming(chunk, cache)
  4. Stateful stream O(1)   — OnnxStreamingWakeWord.push(chunk)   (GRU heads only)

Modes 1-3 work for any head; mode 4 needs a unidirectional GRU head and a
streaming-head ONNX exported with GruClassifierHead.export_streaming_onnx.

Runtime inference (modes 1-4) uses only numpy + onnxruntime — no PyTorch.
"""
import tempfile
from pathlib import Path

import numpy as np
import torch

from ww_trainer.feats import MfccExtractor
from ww_trainer.model import GruClassifierHead, BaseWakeModel
from ww_trainer.inference import OnnxWakeWordInferencer, OnnxStreamingWakeWord

SR = 16000
N_MFCC = 40
HIDDEN = 128
WINDOW = 100          # GRU-output frames ≈ one 1 s clip (16 kHz MFCC ≈ 100 fps)


def main() -> None:
    # ---- Build a model (untrained — scores are arbitrary; we compare APIs) ----
    extractor = MfccExtractor(sr=SR, n_mfcc=N_MFCC)
    head = GruClassifierHead(input_size=N_MFCC, hidden_dim=HIDDEN,
                             bidirectional=False, gru_n_layers=1, device="cpu")
    model = BaseWakeModel(extractor, head, device="cpu").eval()

    tmp = Path(tempfile.mkdtemp())
    feat_onnx = str(tmp / "feat.onnx")
    head_onnx = str(tmp / "head.onnx")
    stream_onnx = str(tmp / "head_streaming.onnx")
    extractor.export_to_onnx(feat_onnx)
    head.export_to_onnx(head_onnx)
    head.export_streaming_onnx(stream_onnx, window=WINDOW)   # bit-exact streaming head
    print(f"Exported ONNX to {tmp}/\n")

    # One second of audio (a real deployment feeds the mic; here: synthetic).
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(SR).astype(np.float32) * 0.1

    inf = OnnxWakeWordInferencer(extractor_path=feat_onnx, head_path=head_onnx)

    # ---- Mode 1: batch / whole-clip ----
    p_batch = inf.infer(audio)
    print(f"1. batch / whole-clip         : {p_batch:.4f}")

    # ---- Mode 2: rolling window (re-featurize a sliding raw-audio buffer) ----
    win_samples, hop = SR, SR // 4         # 1 s window, 0.25 s hop
    peak = 0.0
    for i in range(0, max(1, len(audio) - win_samples + 1), hop):
        peak = max(peak, inf.infer(audio[i:i + win_samples]))
    print(f"2. rolling window (peak)      : {peak:.4f}")

    # ---- Mode 3: feature-cache streaming (featurize only the new chunk) ----
    cache = None
    p_fc = 0.0
    for i in range(0, len(audio), 1600):   # 0.1 s chunks
        chunk = audio[i:i + 1600]
        if len(chunk) < 400:
            break
        p_fc, cache = inf.infer_streaming(chunk, cache)
    print(f"3. feature-cache streaming    : {p_fc:.4f}")

    # ---- Mode 4: stateful O(1) streaming (carry GRU state) ----
    sw = OnnxStreamingWakeWord(feat_onnx, stream_onnx, window=WINDOW, hidden_dim=HIDDEN)
    p_state = 0.0
    for i in range(0, len(audio), 1600):
        chunk = audio[i:i + 1600]
        if len(chunk) < 400:
            break
        p_state = sw.push(chunk)
    print(f"4. stateful O(1) streaming    : {p_state:.4f}")

    # ---- Parity: stateful over exactly one window == batch over that window ----
    feats = extractor([torch.from_numpy(audio[:win_samples])])      # [1, T, F]
    with torch.no_grad():
        ref = torch.sigmoid(head.forward(feats[:, :WINDOW, :])).item()
    sw.reset()
    import onnxruntime as ort
    sess = ort.InferenceSession(stream_onnx, providers=["CPUExecutionProvider"])
    h = np.zeros((1, 1, HIDDEN), np.float32)
    ow = np.zeros((1, WINDOW, HIDDEN), np.float32)
    logit = 0.0
    for t in range(WINDOW):
        frame = feats[:, t:t + 1, :].numpy().astype(np.float32)
        out, h, ow = sess.run(None, {"feat_frame": frame, "h_in": h, "out_window": ow})
        logit = float(np.asarray(out).ravel()[0])
    p_stream_win = 1.0 / (1.0 + np.exp(-logit))
    print(f"\nParity check (one window): batch={ref:.5f}  stateful-onnx={p_stream_win:.5f}"
          f"  match={abs(ref - p_stream_win) < 1e-3}")

    print("\nWhich to use:")
    print("  live mic, any model      -> mode 2  (scripts/eval/mic_test.py)")
    print("  live mic, cheaper        -> mode 3  (scripts/eval/mic_feature_cache.py)")
    print("  always-on / MCU, GRU     -> mode 4  (scripts/eval/mic_stream.py)")
    print("  offline scoring / eval   -> mode 1")


if __name__ == "__main__":
    main()
