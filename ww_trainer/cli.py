"""CLI entry point for wake-word model training.

Provides the ``train`` click command that wires up data loading, loss
configuration, and invokes ``WakeWordTrainer``.
"""
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import click
import torch

from ww_trainer.tiers import get_tier, list_tiers
from ww_trainer.trainer import WakeWordTrainer


@click.command(help="""
Train a wake-word detection model using the WakeWordTrainer.

This command handles full training — data loading, loss setup, adaptive sampling,
hard-negative mining, and evaluation — with optional MLflow tracking and ONNX export.
""")
# -------------------------- Hardware tier preset --------------------------
@click.option("--tier", default=None,
              type=click.Choice(["micro", "small", "medium", "large"]),
              help="Hardware tier preset. Overrides --arch and --featurizer-type if set. "
                   "Run with --list-tiers to see all options.")
@click.option("--list-tiers", "show_tiers", is_flag=True, default=False,
              help="Print hardware tier table and exit.")
# -------------------------- Dataset --------------------------
@click.option("--wake-word", required=True,
              help="Name of the wake word (used in model naming and metadata).")
@click.option("--metadata", required=True,
              help="Path to CSV file containing training samples as 'path,label'.")
@click.option("--test-metadata", default=None,
              help="Optional CSV with test data (same format). If not provided, dataset is split.")
@click.option("--split", default=0.8, type=float,
              help="Train/test split ratio if --test-metadata is not provided (default: 0.8).")
# -------------------------- Training --------------------------
@click.option("--epochs", default=50, type=int, help="Number of training epochs.")
@click.option("--batch-size", default=16, type=int, help="Mini-batch size.")
@click.option("--lr", default=5e-4, type=float, help="Initial learning rate.")
@click.option("--resume", default=None,  help="Resume training from an existing checkpoint (.pt).")
@click.option("--output-dir", default=None, help="Directory to store checkpoints, metrics, and visualizations.")
@click.option("--save-best", is_flag=True, help="If set, saves separate checkpoints for best precision/recall/F1/loss.")
# -------------------------- Architecture --------------------------
@click.option("--featurizer", type=str, help="path feature extractor .onnx model")
@click.option("--feature-dim", type=int,  help="Number of output features from onnx featurizer.")
@click.option("--arch", default="gru", help="Model architecture (e.g., gru, cnn, ffn).")
@click.option("--device", type=click.Choice(["cpu", "cuda", "auto"]), default="auto",
              help="'cuda', 'cpu', or 'auto' (auto-selects CUDA if available).")
@click.option("--sample-rate", type=int, default=16000, help="Audio sample rate used for training.")
@click.option("--export-onnx", is_flag=True,  help="If set, export checkpoints to ONNX format.")
@click.option("--export-c", "export_c_path", type=click.Path(), default=None,
              help="Export final FFN model as C header for ESP32 (e.g. model.h).")
@click.option("--calibrate", is_flag=True, help="Fit Platt scaling on validation set after training.")
# -------------------------- Enrichment --------------------------
@click.option("--use-vad", is_flag=True, help="Enable heuristic energy-based VAD enrichment.")
@click.option("--use-neural-vad", is_flag=True, help="Enable pre-trained neural VAD (Silero) enrichment.")
@click.option("--vad-onnx", "vad_onnx_path", type=click.Path(exists=True), 
              help="Path to pre-trained Silero VAD ONNX model (requirement for --use-neural-vad).")
@click.option("--use-pitch", is_flag=True, help="Enable pitch-tracking enrichment.")
@click.option("--use-snr", is_flag=True, help="Enable per-frame SNR estimation enrichment.")
# -------------------------- Loss Configuration --------------------------
@click.option("--loss-type", default="bce",
              help="Loss type(s): 'bce', 'triplet', 'pair', 'cn2pair', 'rppl', or comma-separated combination.")
@click.option("--loss-weight", default="1.0",
              help="Comma-separated weights for multiple losses, e.g. '0.5,0.5'.")
@click.option("--triplet-margin", default=1.0, type=float, help="Margin used for triplet-based losses.")
@click.option("--mining-type", "mining_type",
              type=click.Choice(['semihard', 'hard', 'random']), default="semihard",
              help="Triplet mining strategy (only used with triplet-style losses).")
# -------------------------- Hard-Negative Mining --------------------------
@click.option("--neg-threshold", default=0.5, type=float,
              help="Model confidence below which predictions are considered non-wake.")
@click.option("--mine-sample", default=0.2, type=float,
              help="Fraction of the non-wake dataset to sample for mining.")
@click.option("--patience", default=2, type=int,
              help="Number of epochs with no new hard negatives before early stopping.")
@click.option("--base-hard", default=0.5, type=float,
              help="Initial ratio of hard negatives per wake sample (early training).")
@click.option("--max-hard", default=5.0, type=float,
              help="Maximum ratio of hard negatives near the end of training.")
@click.option("--base-easy", default=1.5, type=float,
              help="Initial ratio of easy negatives to stabilize early learning.")
@click.option("--min-easy", default=0.2, type=float,
              help="Minimum ratio of easy negatives in late training.")
@click.option("--base-random", default=0.1, type=float,
              help="Baseline ratio of random negatives maintained for diversity.")
@click.option("--total-ratio", default=5.0, type=float,
              help="Overall target number of negatives per wake sample.")
@click.option("--blend-ratio", default=0.7, type=float,
              help="Blend factor between progress-based and LR-based adaptation (0-1).")
# -------------------------- Ambient Validation --------------------------
@click.option("--ambient-dir", default=None, type=click.Path(exists=True),
              help="Folder of long-form non-wake audio for FP/hour estimation.")
# -------------------------- Negative Weight Scheduling --------------------------
@click.option("--neg-weight-schedule", default=None,
              type=click.Choice(["linear", "cosine"]),
              help="Dynamic negative weight schedule (ramps from 1x to max over training).")
@click.option("--max-neg-weight", default=100.0, type=float,
              help="Maximum negative class weight for BCE/focal losses (default: 100).")
@click.option("--target-fpr", default=None, type=float,
              help="Target false positive rate. If exceeded, max_neg_weight doubles per epoch.")
# -------------------------- Augmentation --------------------------
@click.option('--aug-prob', default=0.8, type=float,
              help='Probability of applying any augmentation to each training sample.')
@click.option('--vc-prob', default=0.1, type=float,
              help='Probability of applying voice-cloning augmentation.')
@click.option('--bg-noise-folder', default=None,
              help='Folder with random noise samples for augmentation.')
@click.option('--mic-noise-folder', default=None,
              help='Folder with microphone or silence background clips.')
@click.option('--music-folder', default=None,
              help='Folder with music clips.')
@click.option('--bg-speech-folder', default=None,
              help='Folder with background speech clips.')
@click.option('--rir-folder', default=None,
              help='Folder containing Room Impulse Responses (RIRs) for reverberation simulation.')
@click.option('--vc-folder', default=None,
              help='Folder with multiple voices for random voice cloning.')
@click.option('--snr-min', default=0.0, type=float, help='Minimum SNR for noise mixing.')
@click.option('--snr-max', default=20.0, type=float, help='Maximum SNR for noise mixing.')
@click.option('--pitch-min', default=-1.0, type=float, help='Minimum pitch shift in semitones.')
@click.option('--pitch-max', default=1.0, type=float, help='Maximum pitch shift in semitones.')
@click.option('--speed-min', default=0.95, type=float, help='Minimum speed perturbation factor.')
@click.option('--speed-max', default=1.05, type=float, help='Maximum speed perturbation factor.')
# -------------------------- Performance --------------------------
@click.option("--amp", "use_amp", is_flag=True, default=False,
              help="Enable mixed-precision training (requires CUDA).")
@click.option("--accumulate-grad-batches", default=1, type=int,
              help="Accumulate gradients over N batches before optimizer step (default: 1).")
# -------------------------- Logging --------------------------
@click.option("--metrics-log", default="metrics_log.csv",
              help="Path to CSV file where per-epoch metrics will be appended.")
@click.option("--mlflow-uri", default=None,
              help="Optional MLflow tracking URI.")
# -------------------------- Visualization --------------------------
@click.option("--pca-every", default=1, type=int,
              help="Run PCA visualization every N epochs (0 disables).")
@click.option("--tsne-every", default=0, type=int,
              help="Run t-SNE embedding visualization every N epochs (0 disables).")
@click.option("--umap-every", default=0, type=int,
              help="Run UMAP embedding visualization every N epochs (0 disables).")
def train(**opts: dict) -> None:
    """Train a wake word model using WakeWordTrainer with optional ONNX export."""
    show_tiers = opts.pop("show_tiers", False)
    if show_tiers:
        click.echo(list_tiers())
        return

    tier = opts.pop("tier", None)

    metadata = opts.pop("metadata")
    test_metadata = opts.pop("test_metadata")
    ww_name = opts.get("wake_word")
    mlflow_uri = opts.pop("mlflow_uri")
    arch = opts.pop("arch")
    out_dir = opts.pop("output_dir") or f"trained_models/{arch}/{ww_name}"
    onnx_model = opts.pop("featurizer")
    feat_dim = opts.pop("feature_dim")
    use_amp = opts.pop("use_amp", False)
    accumulate_grad_batches = opts.pop("accumulate_grad_batches", 1)
    ambient_dir = opts.pop("ambient_dir", None)
    neg_weight_schedule = opts.pop("neg_weight_schedule", None)
    max_neg_weight = opts.pop("max_neg_weight", 100.0)
    target_fpr = opts.pop("target_fpr", None)

    if tier is not None:
        tc = get_tier(tier)
        arch = tc.head_arch
        featurizer_type = tc.extractor_type
        opts["hidden_dim"] = tc.hidden_dim
        opts["bidirectional"] = tc.bidirectional
        opts["gru_n_layers"] = tc.gru_n_layers
        if tc.extractor_type == "mfcc":
            opts["n_mfcc"] = tc.n_mfcc
            feat_dim = None
        click.secho(f"[Tier] Using preset '{tier}': {tc.description}", fg="cyan")

    if opts["device"] == "auto":
        opts["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    click.secho(f"Device: {opts['device']}", fg="green", bold=True)

    if opts["mine_sample"] == 0:
        click.secho("Hard-negative mining disabled", fg="yellow", bold=True)
    else:
        click.secho("Hard-negative mining enabled", fg="green", bold=True)

    loss_types = [x.strip().lower() for x in opts.pop("loss_type").split(",")]
    loss_weights = [float(x.strip()) for x in str(opts.pop("loss_weight")).split(",")]
    if len(loss_weights) == 1 and len(loss_types) > 1:
        loss_weights = [loss_weights[0]] * len(loss_types)
    if len(loss_weights) != len(loss_types):
        raise click.BadParameter("Number of loss weights must match number of loss types")

    losses_cfg: List[dict] = []
    for name, w in zip(loss_types, loss_weights):
        cfg: dict = {"name": name, "weight": w}
        if name in ("triplet", "pair", "cn2pair"):
            cfg["margin"] = opts.get("triplet_margin", 1.0)
        losses_cfg.append(cfg)

    with open(metadata, "r", encoding="utf-8") as f:
        entries: List[Tuple[str, str]] = [tuple(line.strip().split(",", 1))
                                          for line in f if line.strip()]
    random.shuffle(entries)

    if test_metadata:
        with open(test_metadata, "r", encoding="utf-8") as f:
            test_data = [tuple(line.strip().split(",", 1)) for line in f if line.strip()]
        train_data = entries
    else:
        split_idx = int(len(entries) * opts["split"])
        train_data, test_data = entries[:split_idx], entries[split_idx:]

    train_data = [f for f in train_data if os.path.isfile(f[0])]
    test_data = [f for f in test_data if os.path.isfile(f[0])]

    click.secho(f"Training {arch} on {len(train_data)} samples", fg="blue", bold=True)

    export_c_path = opts.pop("export_c_path", None)
    calibrate = opts.pop("calibrate", False)

    resume = opts.get("resume")
    trainer = WakeWordTrainer(arch=arch, featurizer=onnx_model, feature_dim=feat_dim,
                              wake_word=ww_name,
                              mlflow_uri=mlflow_uri, losses_cfg=losses_cfg,
                              use_amp=use_amp,
                              **opts)
    trainer.train(
        train_data=train_data,
        test_data=test_data,
        epochs=opts["epochs"],
        batch_size=opts["batch_size"],
        lr=opts["lr"],
        neg_threshold=opts["neg_threshold"],
        mine_fraction=opts["mine_sample"],
        mining_type=opts["mining_type"],
        patience=opts["patience"],
        save_best=opts["save_best"],
        metrics_log=opts["metrics_log"],
        tsne_every=opts["tsne_every"],
        pca_every=opts["pca_every"],
        umap_every=opts["umap_every"],
        output_dir=out_dir,
        base_hard=opts["base_hard"],
        max_hard=opts["max_hard"],
        base_easy=opts["base_easy"],
        min_easy=opts["min_easy"],
        base_random=opts["base_random"],
        total_ratio=opts["total_ratio"],
        blend_ratio=opts["blend_ratio"],
        use_amp=use_amp,
        accumulate_grad_batches=accumulate_grad_batches,
        resume=resume,
        neg_weight_schedule=neg_weight_schedule,
        max_neg_weight=max_neg_weight,
        target_fpr=target_fpr,
        ambient_dir=ambient_dir,
    )

    # Post-training: C header export
    if export_c_path:
        from ww_trainer.export_c import export_to_c_header
        try:
            export_to_c_header(trainer.model, export_c_path, wake_word=ww_name or "wake_word")
            click.secho(f"C header exported to {export_c_path}", fg="green")
        except TypeError as exc:
            click.secho(f"C export failed (FFN only): {exc}", fg="red")

    # Post-training: Platt calibration
    if calibrate:
        from ww_trainer.calibration import calibrate_model
        params = calibrate_model(trainer.model, test_data, out_dir, device=opts["device"])
        click.secho(f"Calibration: coef={params['coef']:.4f}, intercept={params['intercept']:.4f}", fg="green")

    meta = Path(out_dir) / f"{ww_name}_meta.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    with open(meta, "w") as f:
        json.dump(opts, f, indent=2)
    click.echo(click.style(f"Training complete. Model and config saved to {out_dir}", fg="green", bold=True))


if __name__ == "__main__":
    train()
