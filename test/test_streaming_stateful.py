"""Stateful streaming GRU head: ONNX export parity with the batch head.

The streaming head carries the GRU hidden state + a sliding window of GRU
outputs as ONNX state I/O (O(1) per frame). It must reproduce the batch head's
score bit-for-bit when fed the same frames — that is the contract that lets the
always-on / MCU path match the offline model.
"""
import numpy as np
import onnxruntime as ort
import torch

from ww_trainer.model import GruClassifierHead


def test_streaming_head_matches_batch(tmp_path):
    torch.manual_seed(0)
    F_dim, H, WIN = 24, 32, 40
    head = GruClassifierHead(hidden_dim=H, linear_dim=H, input_size=F_dim,
                             bidirectional=False, gru_n_layers=1, device="cpu").eval()

    onnx_path = str(tmp_path / "streaming_head.onnx")
    # export() includes its own parity guard; this test also checks via onnxruntime.
    head.export_streaming_onnx(onnx_path, window=WIN)

    feats = torch.randn(1, WIN, F_dim)
    with torch.no_grad():
        ref = float(head.forward(feats))

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    h = np.zeros((1, 1, H), dtype=np.float32)
    ow = np.zeros((1, WIN, H), dtype=np.float32)
    logit = None
    for t in range(WIN):
        frame = feats[:, t:t + 1, :].numpy().astype(np.float32)
        logit, h, ow = sess.run(None, {"feat_frame": frame, "h_in": h, "out_window": ow})
    onnx_logit = float(np.asarray(logit).ravel()[0])

    assert abs(ref - onnx_logit) < 1e-3, f"batch={ref:.5f} stream={onnx_logit:.5f}"


def test_streaming_export_rejects_bidirectional(tmp_path):
    head = GruClassifierHead(hidden_dim=16, input_size=12, bidirectional=True,
                             device="cpu").eval()
    try:
        head.export_streaming_onnx(str(tmp_path / "x.onnx"), window=20)
        assert False, "expected ValueError for bidirectional GRU"
    except ValueError:
        pass


def test_streaming_inferencer_roundtrip(tmp_path):
    """OnnxStreamingWakeWord drives the exported head with carried state."""
    from ww_trainer.inference import OnnxStreamingWakeWord
    from ww_trainer.feats import MfccExtractor

    torch.manual_seed(0)
    n_mfcc = 24
    feat = MfccExtractor(n_mfcc=n_mfcc, sr=16000)
    feat_path = str(tmp_path / "feat.onnx")
    feat.export_to_onnx(feat_path)

    head = GruClassifierHead(hidden_dim=32, input_size=n_mfcc, bidirectional=False,
                             device="cpu").eval()
    head_path = str(tmp_path / "head_streaming.onnx")
    head.export_streaming_onnx(head_path, window=50)

    sw = OnnxStreamingWakeWord(feat_path, head_path, window=50, hidden_dim=32)
    sw.reset()
    prob = 0.0
    for _ in range(10):
        prob = sw.push(np.random.randn(1600).astype(np.float32))  # 0.1 s chunks
    assert 0.0 <= prob <= 1.0
