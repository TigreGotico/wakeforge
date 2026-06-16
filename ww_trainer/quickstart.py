"""Single-command wake-word training: string → trained ONNX model.

Public API:
- :func:`train_from_wakeword` — Python entry point
- :class:`QuickstartConfig` — all knobs with sensible defaults
- :class:`QuickstartResult` — paths and metrics of the trained model

CLI entry point: ``ww_trainer-quickstart`` (see :func:`cli_main`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ww_trainer.utils import read_dataset_csv

logger = logging.getLogger(__name__)


@dataclass
class QuickstartConfig:
    """All configuration for one quickstart run.

    Args:
        wake_word: The wake-word phrase, e.g. ``"hey jarvis"``.
        output_dir: Root directory; dataset and model subdirs are created here.
        n_positive: Number of positive (wake-word) audio samples to synthesise.
        lang: BCP-47 language tag for TTS synthesis.
        adversarial: Generate grapheme-level hard-negative confusables.
        adversarial_n: Number of adversarial confusable strings.
        vad_trim: Trim leading/trailing silence with Silero VAD.
        vc_refs_dir: Optional voice-conversion reference directory.
        download_augmentation: Download bg_noise/music/RIR datasets from HF.
        llm_url: Optional Ollama URL for LLM-based adversarial generation.
        tier: Hardware tier name (see ``ww_trainer-tiers``).
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.
        export_onnx: Export best model to ONNX after training.
        reuse_dataset: Skip datagen if the dataset directory already exists.
        seed: Random seed for reproducibility.
        losses_cfg: Loss configuration passed to :class:`~ww_trainer.trainer.WakeWordTrainer`.
            Each entry is a dict with at least ``"name"`` and ``"weight"`` keys, e.g.
            ``[{"name": "focal", "weight": 1.0}]``.  Defaults to BCE when ``None``.
        bg_noise_folder: Optional path to a background-noise directory for augmentation.
            When *None* the path is inferred from the datagen result (auto mode).
        music_folder: Optional path to a music directory for augmentation.
            When *None* the path is inferred from the datagen result (auto mode).
        rir_folder: Optional path to a room-impulse-response directory for augmentation.
            When *None* the path is inferred from the datagen result (auto mode).
    """

    wake_word: str
    output_dir: Path
    # datagen knobs
    n_positive: int = 1000
    lang: str = "en"
    adversarial: bool = True
    adversarial_n: int = 20
    vad_trim: bool = True
    vc_refs_dir: Optional[str] = None
    download_augmentation: bool = True
    llm_url: Optional[str] = None
    # training knobs
    tier: str = "small"
    epochs: int = 50
    batch_size: int = 16
    lr: float = 5e-4
    device: str = "auto"
    export_onnx: bool = True
    losses_cfg: Optional[List[Dict[str, Any]]] = None
    # augmentation overrides (auto-populated from datagen result when None)
    bg_noise_folder: Optional[str] = None
    music_folder: Optional[str] = None
    rir_folder: Optional[str] = None
    # flow
    reuse_dataset: bool = False
    seed: int = 42


@dataclass
class QuickstartResult:
    """Paths and metrics produced by a completed quickstart run.

    Args:
        dataset_dir: Root directory of the generated dataset.
        model_dir: Directory containing checkpoints and ONNX exports.
        train_csv: Path to the training metadata CSV.
        test_csv: Path to the test metadata CSV.
        best_model_path: Best F1 checkpoint (``best_f1.pt``), if saved.
        best_onnx_path: ONNX export of the best model, if ``export_onnx=True``.
        metrics: Final evaluation metrics (``f1``, ``precision``, ``recall``).
    """

    dataset_dir: Path
    model_dir: Path
    train_csv: Path
    test_csv: Path
    best_model_path: Optional[Path]
    best_onnx_path: Optional[Path]
    metrics: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_or_load_datagen(cfg: QuickstartConfig):  # noqa: ANN202
    """Run the datagen pipeline or reconstruct a result from existing paths.

    Args:
        cfg: Quickstart configuration.

    Returns:
        :class:`~ww_trainer.datagen.DatagenResult` instance.
    """
    from ww_trainer.datagen import DatagenConfig, DatagenResult, normalize_wake_word, run_datagen_pipeline

    slug = normalize_wake_word(cfg.wake_word)
    dataset_dir = Path(cfg.output_dir) / "dataset"
    train_csv = dataset_dir / "train" / "metadata.csv"
    test_csv = dataset_dir / "test" / "metadata.csv"

    if cfg.reuse_dataset and train_csv.exists() and test_csv.exists():
        logger.info("Reusing existing dataset at %s", dataset_dir)
        bg_noise_dir = dataset_dir / "augmentation" / "bg_noise"
        music_dir = dataset_dir / "augmentation" / "music"
        rir_dir = dataset_dir / "augmentation" / "rir"
        return DatagenResult(
            train_csv=train_csv,
            test_csv=test_csv,
            positives_dir=dataset_dir / slug / "positives",
            negatives_dir=dataset_dir / slug / "negatives",
            bg_noise_dir=bg_noise_dir if bg_noise_dir.exists() else None,
            music_dir=music_dir if music_dir.exists() else None,
            rir_dir=rir_dir if rir_dir.exists() else None,
        )

    datagen_cfg = DatagenConfig(
        wake_word=cfg.wake_word,
        output_dir=dataset_dir,
        n_positive=cfg.n_positive,
        lang=cfg.lang,
        adversarial=cfg.adversarial,
        adversarial_n=cfg.adversarial_n,
        vad_trim=cfg.vad_trim,
        vc_refs_dir=cfg.vc_refs_dir,
        download_augmentation=cfg.download_augmentation,
        llm_url=cfg.llm_url,
        seed=cfg.seed,
    )
    try:
        return run_datagen_pipeline(datagen_cfg)
    except ModuleNotFoundError as e:
        vc_modules = {"voiceclonnx"}
        if e.name in vc_modules or (cfg.vc_refs_dir and e.name and "voiceclonnx" in e.name):
            hint = (
                "Voice conversion dependency missing. Install the VC extra:\n"
                "    uv pip install -e \".[vc]\"   # pure-ONNX voiceclonnx\n"
                "Or drop --vc-refs to skip voice cloning."
            )
        else:
            hint = (
                "The quickstart requires the [datagen] and [torchcodec] extras:\n"
                "    uv pip install -e \".[dev,datagen,torchcodec]\"\n"
                "Or pass --reuse-dataset to skip datagen and train on an existing dataset."
            )
        raise ModuleNotFoundError(
            f"Quickstart datagen dependency missing: {e.name!r}. {hint}"
        ) from e


def _train_from_datagen_result(cfg: QuickstartConfig, datagen_result: Any) -> QuickstartResult:
    """Wire datagen output into a trainer and run training.

    This helper is the primary unit-test target: it can be called with a
    synthetic :class:`~ww_trainer.datagen.DatagenResult` without any TTS or
    HF downloads.

    Args:
        cfg: Quickstart configuration.
        datagen_result: A :class:`~ww_trainer.datagen.DatagenResult` with
            valid ``train_csv`` and ``test_csv`` paths.

    Returns:
        :class:`QuickstartResult` with paths and final metrics.
    """
    from ww_trainer.tiers import get_tier
    from ww_trainer.trainer import WakeWordTrainer

    model_dir = Path(cfg.output_dir) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ---- resolve tier ----
    tc = get_tier(cfg.tier)

    # ---- model kwargs from tier ----
    model_kwargs: Dict[str, Any] = {"hidden_dim": tc.hidden_dim}
    if tc.extractor_type == "mfcc":
        model_kwargs["n_mfcc"] = tc.n_mfcc
    if tc.head_arch == "gru":
        model_kwargs["bidirectional"] = tc.bidirectional
        model_kwargs["gru_n_layers"] = tc.gru_n_layers
    if tc.head_arch == "ocsvm" and tc.embed_dim is not None:
        model_kwargs["embed_dim"] = tc.embed_dim

    # ---- augmentation opts: explicit overrides take priority; fall back to datagen result ----
    augment_opts: Dict[str, str] = {}
    def _maybe(explicit: Optional[str], dr_attr: str) -> Optional[str]:
        candidate = explicit if explicit is not None else getattr(datagen_result, dr_attr, None)
        if candidate is not None and Path(candidate).exists():
            return str(candidate)
        return None

    for folder_key, dr_attr in (
        ("bg_noise_folder", "bg_noise_dir"),
        ("music_folder", "music_dir"),
        ("rir_folder", "rir_dir"),
    ):
        resolved = _maybe(getattr(cfg, folder_key), dr_attr)
        if resolved is not None:
            augment_opts[folder_key] = resolved

    losses: List[Dict[str, Any]] = cfg.losses_cfg if cfg.losses_cfg is not None else [{"name": "bce", "weight": 1.0}]

    trainer = WakeWordTrainer(
        arch=tc.head_arch,
        featurizer="",
        feature_dim=None,
        featurizer_type=tc.extractor_type,
        wake_word=cfg.wake_word,
        device=cfg.device,
        losses_cfg=losses,
        export_onnx=cfg.export_onnx,
        seed=cfg.seed,
        **model_kwargs,
        **augment_opts,
    )

    train_data = read_dataset_csv(datagen_result.train_csv)
    test_data = read_dataset_csv(datagen_result.test_csv)

    best_f1 = trainer.train(
        output_dir=model_dir,
        train_data=train_data,
        test_data=test_data,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
    )

    best_pt = model_dir / "best_f1.pt"
    best_onnx = model_dir / "best_f1.onnx"

    return QuickstartResult(
        dataset_dir=Path(cfg.output_dir) / "dataset",
        model_dir=model_dir,
        train_csv=datagen_result.train_csv,
        test_csv=datagen_result.test_csv,
        best_model_path=best_pt if best_pt.exists() else None,
        best_onnx_path=best_onnx if best_onnx.exists() else None,
        metrics={"f1": best_f1},
    )


# ---------------------------------------------------------------------------
# Public Python API
# ---------------------------------------------------------------------------

def train_from_wakeword(
    wake_word: str,
    output_dir: "str | Path",
    **kwargs: Any,
) -> QuickstartResult:
    """Train a wake-word detector end-to-end from a single string.

    Generates a synthetic dataset with :func:`~ww_trainer.datagen.run_datagen_pipeline`,
    then trains a :class:`~ww_trainer.trainer.WakeWordTrainer` with the tier
    and parameters specified in *kwargs*.  The result contains the paths to
    the best checkpoint and ONNX export.

    Args:
        wake_word: Wake-word phrase, e.g. ``"hey jarvis"``.
        output_dir: Root directory for dataset and model outputs.
        **kwargs: Any field of :class:`QuickstartConfig` (e.g. ``tier``,
            ``epochs``, ``n_positive``, ``reuse_dataset``).

    Returns:
        :class:`QuickstartResult` with paths and metrics.

    Example::

        from ww_trainer.quickstart import train_from_wakeword
        result = train_from_wakeword("hey jarvis", "./hey_jarvis", tier="micro", epochs=2)
        print(result.best_onnx_path)
    """
    cfg = QuickstartConfig(wake_word=wake_word, output_dir=Path(output_dir), **kwargs)
    datagen_result = _run_or_load_datagen(cfg)
    return _train_from_datagen_result(cfg, datagen_result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli_main() -> None:
    """Click entry point for ``ww_trainer-quickstart``."""
    import click

    @click.command("ww_trainer-quickstart")
    @click.option("--wake-word", required=True, help="Wake-word phrase, e.g. 'hey jarvis'.")
    @click.option("--output-dir", required=True, type=click.Path(), help="Root output directory.")
    @click.option("--tier", default="small", show_default=True, help="Hardware tier name.")
    @click.option("--epochs", default=50, show_default=True, type=int, help="Training epochs.")
    @click.option("--batch-size", default=16, show_default=True, type=int, help="Batch size.")
    @click.option("--lr", default=5e-4, show_default=True, type=float, help="Learning rate.")
    @click.option("--n-positive", default=1000, show_default=True, type=int,
                  help="Number of positive audio samples to synthesise.")
    @click.option("--lang", default="en", show_default=True, help="BCP-47 language tag for TTS.")
    @click.option("--adversarial/--no-adversarial", default=True, show_default=True,
                  help="Generate grapheme-level hard-negative confusables.")
    @click.option("--augmentation-data/--no-augmentation-data", default=True, show_default=True,
                  help="Download bg_noise/music/RIR augmentation datasets.")
    @click.option("--vc-refs", "vc_refs_dir", default=None, type=click.Path(exists=True),
                  help="Optional directory of reference voices (WAVs) for voice "
                       "conversion. Requires the [vc-onnx] or [vc-torch] extra. "
                       "When set, each synthesised positive is re-rendered in the "
                       "timbre of a random reference speaker for diversity.")
    @click.option("--reuse-dataset", is_flag=True, default=False,
                  help="Skip datagen if dataset directory already exists.")
    @click.option("--device", default="auto", show_default=True,
                  help="Compute device: 'auto', 'cpu', or 'cuda'.")
    @click.option("--export-onnx/--no-export-onnx", default=True, show_default=True,
                  help="Export best model to ONNX.")
    @click.option("--seed", default=42, show_default=True, type=int, help="Random seed.")
    def _cmd(
        wake_word: str,
        output_dir: str,
        tier: str,
        epochs: int,
        batch_size: int,
        lr: float,
        n_positive: int,
        lang: str,
        adversarial: bool,
        augmentation_data: bool,
        vc_refs_dir: Optional[str],
        reuse_dataset: bool,
        device: str,
        export_onnx: bool,
        seed: int,
    ) -> None:
        """Generate a synthetic dataset and train a wake-word model in one command."""
        result = train_from_wakeword(
            wake_word=wake_word,
            output_dir=output_dir,
            tier=tier,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            n_positive=n_positive,
            lang=lang,
            adversarial=adversarial,
            download_augmentation=augmentation_data,
            vc_refs_dir=vc_refs_dir,
            reuse_dataset=reuse_dataset,
            device=device,
            export_onnx=export_onnx,
            seed=seed,
        )
        click.echo(f"\n✓ Dataset: {result.dataset_dir}")
        f1_str = f"  (F1={result.metrics.get('f1', 0):.3f})" if result.metrics.get("f1") else ""
        if result.best_onnx_path and result.best_onnx_path.exists():
            click.echo(f"✓ Model:   {result.best_onnx_path}{f1_str}")
        elif result.best_model_path and result.best_model_path.exists():
            click.echo(f"✓ Model:   {result.best_model_path}{f1_str}")
        else:
            click.echo(f"✓ Model dir: {result.model_dir}{f1_str}")

    _cmd(standalone_mode=True)
