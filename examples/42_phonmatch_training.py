"""Example 42 — PhonMatchNet training with IPA phoneme conditioning.

The correct training paradigm requires a *multi-keyword* dataset: each sample
carries the IPA phoneme sequence of its keyword as a third column:

    (audio_path, label, keyword_ipa_ids)

This lets the model learn general phoneme-audio alignment rather than
memorising a single keyword's acoustics.  At inference, pass any IPA sequence
to score an arbitrary wake word.

Two usage modes
---------------
Mode A — cross-attention head (arch="phonmatch")
    PhonMatchHead receives per-batch phoneme IDs and performs cross-modal
    attention internally.  ONNX export has two inputs: audio features and
    phoneme IDs.

Mode B — text featurizer ONNX (any head + OnnxTextExtractor)
    A PhonMatchTextEncoder is exported to ONNX once.  The text ONNX is loaded
    at training time; each batch feeds its per-sample keyword IDs through the
    text encoder, appending the resulting embedding to audio features before
    the classifier head.  At inference, pass token IDs at call time.

Dataset format
--------------
Each tuple in train_data / test_data must be one of:
    (path, label)               — existing format, no phoneme conditioning
    (path, label, keyword_ids)  — keyword_ids is a List[int] of IPA token IDs

Negatives should carry the IPA of the *keyword they are negative for*, or of a
random keyword from the training vocabulary — this teaches the model to reject
audio that does not match the phoneme query.

Prerequisites
-------------
- Audio data in ``experiments/hey_mycroft/dataset/`` (run datagen first).
- IPA phonemes for your wake words — use any G2P tool:
    * espeak-ng:  ``espeak-ng -q --ipa "hey mycroft"``
    * phonemizer: ``phonemize "hey mycroft" --backend espeak --language en-us``
    * gruut:      ``python -m gruut en-us "hey mycroft"``
  OR use the built-in ARPAbet→IPA bridge for g2p_en output.
"""
import glob
import os
import random
import torch

from ww_trainer.phonmatch import (
    ipa_to_ids,
    arpabet_to_ids,
    PhonMatchTextEncoder,
    PhonMatchHead,
)
from ww_trainer import WakeWordTrainer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WAKE_WORD = "hey mycroft"
OUTPUT_DIR = "experiments/hey_mycroft/phonmatch"
DATASET_DIR = "experiments/hey_mycroft/dataset"

# IPA phonemes for training keywords (en-US, espeak-ng output).
# In a real multi-keyword setup, this dict would contain many keywords.
KEYWORD_IPA = {
    "hey mycroft": ipa_to_ids(["h", "eɪ", "m", "aɪ", "k", "ɹ", "ʌ", "f", "t"]),
    # Add more keywords here for better zero-shot generalisation:
    # "hey computer":  ipa_to_ids([...]),
    # "ok google":     ipa_to_ids([...]),
}

# Default keyword for single-keyword training / inference demo
HEY_MYCROFT_IPA = KEYWORD_IPA["hey mycroft"]


# ---------------------------------------------------------------------------
# Dataset loading — 3-tuple format (path, label, keyword_ids)
# ---------------------------------------------------------------------------

def _load_data(dataset_dir: str, wake_word: str):
    """Return (train_data, test_data) as 3-tuple lists.

    Each positive sample carries the IPA of its keyword.
    Each negative sample carries the same IPA (teaching the model to reject
    audio that does not match this phoneme query).  With a multi-keyword
    dataset, negatives would carry the IPA of the keyword they are negative
    *for*.
    """
    pos_dir = os.path.join(dataset_dir, "positive")
    neg_dir = os.path.join(dataset_dir, "negative")

    pos_files = glob.glob(os.path.join(pos_dir, "**/*.wav"), recursive=True)
    neg_files = glob.glob(os.path.join(neg_dir, "**/*.wav"), recursive=True)

    if not pos_files:
        raise FileNotFoundError(
            f"No positive samples found in {pos_dir}. "
            "Run scripts/data/download_ww_data.py first."
        )

    keyword_ids = KEYWORD_IPA.get(wake_word, HEY_MYCROFT_IPA)

    random.seed(42)
    random.shuffle(pos_files)
    random.shuffle(neg_files)

    split = 0.9
    pos_train = [(p, "1", keyword_ids) for p in pos_files[: int(len(pos_files) * split)]]
    pos_test  = [(p, "1", keyword_ids) for p in pos_files[int(len(pos_files) * split):]]
    neg_train = [(p, "0", keyword_ids) for p in neg_files[: int(len(neg_files) * split)]]
    neg_test  = [(p, "0", keyword_ids) for p in neg_files[int(len(neg_files) * split):]]

    return pos_train + neg_train, pos_test + neg_test


# ---------------------------------------------------------------------------
# Mode A — PhonMatchHead (per-batch phoneme IDs, two ONNX inputs at export)
# ---------------------------------------------------------------------------

def train_mode_a():
    """PhonMatchHead receives phoneme IDs per batch — no fixed buffer.

    At inference, pass the target keyword's IPA token IDs alongside audio.
    ONNX export has two inputs: ``input_features`` and ``phoneme_ids``.
    """
    print("=== Mode A: PhonMatchHead (cross-attention, per-batch IDs) ===")
    train_data, test_data = _load_data(DATASET_DIR, WAKE_WORD)

    trainer = WakeWordTrainer(
        arch="phonmatch",
        featurizer="mfcc",
        featurizer_type="mfcc",
        wake_word=WAKE_WORD,
        # PhonMatchHead kwargs — no keyword_token_ids needed:
        hidden_dim=128,
        n_heads=2,
        gru_layers=2,
        dropout=0.1,
        losses_cfg=[
            {"name": "bce", "weight": 1.0},
            # Optional HALO for better OOD calibration:
            # {"name": "halo", "embed_dim": 128, "weight": 0.5},
        ],
        device="cpu",
    )
    torch.set_num_threads(12)

    best = trainer.train(
        output_dir=os.path.join(OUTPUT_DIR, "mode_a"),
        train_data=train_data,
        test_data=test_data,
        epochs=30,
        batch_size=16,
        lr=1e-3,
        export_onnx=True,
    )
    print(f"Mode A best fitness: {best:.4f}")
    return trainer


# ---------------------------------------------------------------------------
# Mode B — text featurizer ONNX + any head (GRU shown)
# ---------------------------------------------------------------------------

def train_mode_b():
    """Text featurizer ONNX path: PhonMatchTextEncoder + standard GRU head.

    Per-batch keyword IDs flow through the text encoder; the resulting
    embedding is appended to audio features before the GRU head.
    """
    print("=== Mode B: GRU head + PhonMatchTextEncoder ONNX ===")

    text_encoder_path = os.path.join(OUTPUT_DIR, "mode_b", "phoneme_encoder.onnx")
    os.makedirs(os.path.dirname(text_encoder_path), exist_ok=True)

    EMB_DIM = 128
    enc = PhonMatchTextEncoder(emb_dim=EMB_DIM)
    enc.export_to_onnx(
        text_encoder_path,
        seq_len=16,
        metadata={"wake_word": WAKE_WORD, "vocab": "IPA"},
    )
    print(f"Exported text encoder → {text_encoder_path}")

    train_data, test_data = _load_data(DATASET_DIR, WAKE_WORD)

    trainer = WakeWordTrainer(
        arch="gru",
        featurizer="mfcc",
        featurizer_type="mfcc",
        wake_word=WAKE_WORD,
        # Text modality — feature_dim auto-widened to mfcc_dim + EMB_DIM
        text_featurizer=text_encoder_path,
        text_emb_dim=EMB_DIM,
        hidden_dim=128,
        gru_n_layers=2,
        losses_cfg=[{"name": "bce", "weight": 1.0}],
        device="cpu",
    )
    torch.set_num_threads(12)

    # Per-batch keyword IDs flow from dataset through collate_fn → model.
    # No precompute() needed during training.

    best = trainer.train(
        output_dir=os.path.join(OUTPUT_DIR, "mode_b"),
        train_data=train_data,
        test_data=test_data,
        epochs=30,
        batch_size=16,
        lr=1e-3,
        export_onnx=True,
    )
    print(f"Mode B best fitness: {best:.4f}")
    return trainer


# ---------------------------------------------------------------------------
# Inference demo
# ---------------------------------------------------------------------------

def infer_demo(mode_a_trainer=None, mode_b_trainer=None, sample_wav: str = None):
    """Show ONNX inference for both modes.

    At inference, pass the target keyword's IPA token IDs.  The model was
    trained on per-batch IDs, so inference with any IPA sequence is valid —
    accuracy depends on phonemic coverage during training.
    """
    import numpy as np
    from ww_trainer.inference import OnnxWakeWordInferencer

    if sample_wav is None or not os.path.exists(sample_wav):
        print("\n(Skipping inference demo — supply sample_wav= path)")
        return

    wav = np.zeros(16000, dtype=np.float32)  # placeholder

    if mode_a_trainer is not None:
        model_dir = os.path.join(OUTPUT_DIR, "mode_a")
        inferencer = OnnxWakeWordInferencer(
            extractor_path=os.path.join(model_dir, "best_f1_featurizer.onnx"),
            head_path=os.path.join(model_dir, "best_f1.onnx"),
        )
        # Pass IPA token IDs at inference time
        score = inferencer.infer(wav, text_token_ids=HEY_MYCROFT_IPA)
        print(f"Mode A score: {score:.4f}")

    if mode_b_trainer is not None:
        model_dir = os.path.join(OUTPUT_DIR, "mode_b")
        inferencer = OnnxWakeWordInferencer(
            extractor_path=os.path.join(model_dir, "best_f1_featurizer.onnx"),
            head_path=os.path.join(model_dir, "best_f1.onnx"),
            text_extractor_path=os.path.join(model_dir, "phoneme_encoder.onnx"),
            text_emb_dim=128,
        )
        score = inferencer.infer(wav, text_token_ids=HEY_MYCROFT_IPA)
        print(f"Mode B score: {score:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PhonMatchNet training example")
    parser.add_argument("--mode", choices=["a", "b", "both"], default="a",
                        help="Training mode: A=cross-attn head, B=text ONNX featurizer")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    DATASET_DIR = args.dataset_dir
    OUTPUT_DIR = args.output_dir

    trainer_a = trainer_b = None
    if args.mode in ("a", "both"):
        trainer_a = train_mode_a()
    if args.mode in ("b", "both"):
        trainer_b = train_mode_b()

    infer_demo(trainer_a, trainer_b)
