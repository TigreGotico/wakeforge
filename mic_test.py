#!/usr/bin/env python3
import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from colorama import Fore, Style
import onnxruntime as ort

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.5  # seconds per new block

class SplitOnnxFeaturePipeline:
    """
    Universal two-stage ONNX feature + classifier pipeline.

    ✅ Supports arbitrary extractors (MFCC, HuBERT, spectrograms)
    ✅ Auto-detects dtype, layout (T×F vs F×T)
    ✅ Auto-estimates overlap from model output
    """

    def __init__(self, extractor_path: str, classifier_path: str, sample_rate=16000):
        self.extractor_path = extractor_path
        self.classifier_path = classifier_path
        self.sample_rate = sample_rate

        # --- Load models ---
        self.extractor_sess = self._load_onnx_session(extractor_path, "Feature Extractor")
        self.classifier_sess = self._load_onnx_session(classifier_path, "Classifier Head")

        self.extractor_input_name = self.extractor_sess.get_inputs()[0].name
        self.extractor_output_name = self.extractor_sess.get_outputs()[0].name
        self.classifier_input_name = self.classifier_sess.get_inputs()[0].name
        self.classifier_output_name = self.classifier_sess.get_outputs()[0].name

        # Infer dtype
        input_type = self.extractor_sess.get_inputs()[0].type
        self.dtype = np.float16 if input_type == "tensor(float16)" else np.float32
        print(Fore.CYAN + f"Extractor dtype: {self.dtype}" + Style.RESET_ALL)

        # Get classifier expected feature dimension
        self.classifier_expected_dim = self._get_classifier_input_dim()

        # Auto-detect layout and overlap
        self.transpose_needed = None
        self.samples_per_feature, self.overlap_samples = self._estimate_stride_and_overlap()

        print(Fore.MAGENTA + f"Estimated stride: {self.samples_per_feature} samples "
                             f"({self.samples_per_feature / self.sample_rate * 1000:.1f} ms)" + Style.RESET_ALL)
        print(Fore.MAGENTA + f"Estimated overlap: {self.overlap_samples} samples "
                             f"({self.overlap_samples / self.sample_rate * 1000:.1f} ms)" + Style.RESET_ALL)

    def _load_onnx_session(self, path: str, name: str) -> ort.InferenceSession:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            print(f"Loading {name} from {path}")
            sess = ort.InferenceSession(path, providers=providers)
            if not sess.get_inputs():
                raise RuntimeError(f"{name} has no inputs!")
            return sess
        except Exception as e:
            print(Fore.RED + f"Error loading {name}: {e}" + Style.RESET_ALL)
            sys.exit(1)

    def _get_classifier_input_dim(self) -> int:
        in_shape = self.classifier_sess.get_inputs()[0].shape
        if in_shape is None or len(in_shape) < 3:
            print(Fore.YELLOW + "⚠️ Classifier input shape not fully specified; assuming last dim is feature size." + Style.RESET_ALL)
            return None
        return int(in_shape[-1])

    def _estimate_stride_and_overlap(self) -> tuple[int, int]:
        """
        Estimate stride (samples per feature frame) and overlap from model output.
        Uses a 1-second test tone to infer T_out / T_in ratio.
        """
        try:
            dummy_audio = np.random.randn(1, self.sample_rate).astype(self.dtype)
            feats = self.extractor_sess.run([self.extractor_output_name],
                                            {self.extractor_input_name: dummy_audio})[0]
            # Determine time dimension
            time_axis = 1 if feats.shape[1] > feats.shape[2] else 2
            num_frames = feats.shape[time_axis]
            samples_per_feature = self.sample_rate / num_frames
            # 25 ms window - stride difference = overlap
            window_ms = 25.0
            stride_ms = samples_per_feature / self.sample_rate * 1000
            overlap_ms = max(window_ms - stride_ms, 0)
            overlap_samples = int(overlap_ms * self.sample_rate / 1000)
            return int(samples_per_feature), overlap_samples
        except Exception as e:
            print(Fore.YELLOW + f"⚠️ Could not estimate stride automatically: {e}. Falling back to defaults." + Style.RESET_ALL)
            return 160, 400  # fallback (10ms stride, 25ms window)

    def _maybe_fix_layout(self, feats: np.ndarray) -> np.ndarray:
        """Automatically detect and fix feature orientation (T×F vs F×T)."""
        if self.transpose_needed is None:
            feat_dim = feats.shape[-1] if feats.ndim == 3 else feats.shape[0]
            if self.classifier_expected_dim and feat_dim != self.classifier_expected_dim:
                self.transpose_needed = True
                print(Fore.YELLOW + f"Auto-transpose enabled: extractor output last dim={feat_dim}, "
                                    f"classifier expects {self.classifier_expected_dim}" + Style.RESET_ALL)
            else:
                self.transpose_needed = False

        if self.transpose_needed:
            if feats.ndim == 3:
                feats = np.transpose(feats, (0, 2, 1))
            elif feats.ndim == 2:
                feats = feats.T
        if feats.ndim == 2:
            feats = np.expand_dims(feats, 0)
        return feats

    def infer(self, current_block: np.ndarray, cache: np.ndarray) -> tuple[float, np.ndarray]:
        full_audio = np.concatenate([cache, current_block]).astype(self.dtype)
        audio = np.expand_dims(full_audio, 0)
        feats = self.extractor_sess.run([self.extractor_output_name],
                                        {self.extractor_input_name: audio})[0]
        feats = self._maybe_fix_layout(feats)
        logits = self.classifier_sess.run([self.classifier_output_name],
                                          {self.classifier_input_name: feats.astype(np.float32)})[0]
        prob = 1.0 / (1.0 + np.exp(-logits))
        new_cache = current_block[-self.overlap_samples:].copy()
        return float(prob.squeeze()), new_cache



def setup_audio_queue(samplerate):
    q = queue.Queue()

    def callback(indata, frames, time_, status):
        if status:
            print(status, flush=True)
        q.put(indata[:, 0].copy() if indata.ndim > 1 else indata.copy())

    stream = sd.InputStream(
        channels=1,
        samplerate=samplerate,
        blocksize=int(samplerate * BLOCK_DURATION),
        dtype="float32",
        callback=callback,
    )
    return q, stream


def main(args):
    print(Fore.CYAN + "=== Universal Real-Time ONNX Wake Word Detector ===" + Style.RESET_ALL)
    print(f"Extractor: {args.extractor_model}")
    print(f"Classifier: {args.classifier_model}")
    print(f"Block: {BLOCK_DURATION}s | Overlap: {args.overlap} samples | Threshold: {args.threshold}")
    print("Press Ctrl+C to exit.\n")

    model = SplitOnnxFeaturePipeline(args.extractor_model, args.classifier_model)
    q, stream = setup_audio_queue(SAMPLE_RATE)
    stream.start()

    cache = np.zeros(args.overlap, dtype=np.float32)
    avg_score = 0.0
    alpha = 0.2

    try:
        while True:
            block = q.get()
            score, cache = model.infer(block, cache)
            avg_score = alpha * score + (1 - alpha) * avg_score
            bar = "█" * int(avg_score * 30)

            if avg_score > args.threshold:
                print(Fore.RED + f"[WAKE] {avg_score:.3f} {bar}" + Style.RESET_ALL)
            else:
                print(Fore.GREEN + f"[----] {avg_score:.3f} {bar}" + Style.RESET_ALL)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print(Fore.YELLOW + "\nStopping listener..." + Style.RESET_ALL)
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal ONNX-based real-time wake word detector")
    parser.add_argument("--extractor-model", required=True, help="Path to feature extractor ONNX (mfcc.onnx / hubert.onnx / etc.)")
    parser.add_argument("--classifier-model", required=True, help="Path to classifier ONNX model")
    parser.add_argument("--threshold", type=float, default=0.1, help="Wake-word detection threshold")
    parser.add_argument("--overlap", type=int, default=400, help="Audio overlap samples between blocks")
    args = parser.parse_args()
    main(args)
