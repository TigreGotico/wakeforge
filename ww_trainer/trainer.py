#!/usr/bin/env python
"""Wake-word trainer core: model lifecycle, training loop, and inference."""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ww_trainer.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    save_intermediate_checkpoint,
)
from ww_trainer.factory import create_model

logger = logging.getLogger(__name__)


class WakeWordTrainer:
    """Full-featured wake word trainer.

    Encapsulates model setup, device management, MLflow integration, layer
    freezing, and the training loop.  The epoch-level loop logic lives in
    :mod:`ww_trainer.loop` and is invoked via :meth:`train`.

    Features:
      - BCE / Triplet / Pair loss management
      - Adaptive BCE+Triplet mixing
      - Hard-negative mining with a rolling cache
      - t-SNE logging of embeddings
      - MLflow integration
      - Checkpointing per metric
      - Mixed precision (optional) and LR scheduling
      - ROC / PR plots + FP/FN artifact logging
    """

    def __init__(
            self,
            arch: str,
            featurizer: str,
            wake_word: str = "hey_mycroft",
            feature_dim: int = None,
            device: str = "auto",
            mlflow_uri: Optional[str] = None,
            resume: Optional[str] = None,
            export_onnx: bool = False,
            losses_cfg: Optional[List[Dict[str, str]]] = None,
            featurizer_type: str = "onnx",
            shared_extractor=None,
            use_amp: bool = False,
            seed: Optional[int] = None,
            freeze_extractor: bool = False,
            freeze_layers: int = 0,
            unfreeze_at_epoch: Optional[int] = None,
            **model_kwargs: Any,
    ) -> None:
        if seed is not None:
            from ww_trainer.reproducibility import set_seed
            set_seed(seed)

        self.losses_cfg = losses_cfg
        self.wake_word = wake_word
        device_str: str = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.export_onnx = export_onnx
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None
        self.arch = arch
        self.training_params = model_kwargs
        self.model = self.create_model(
            arch, featurizer, feature_dim,
            featurizer_type=featurizer_type,
            shared_extractor=shared_extractor,
            device=self.device,
            **model_kwargs,
        )
        self.model.to(self.device)

        self.freeze_extractor = freeze_extractor
        self.freeze_layers = freeze_layers
        self.unfreeze_at_epoch = unfreeze_at_epoch
        if freeze_extractor or freeze_layers > 0:
            self._freeze()

        self.augment_opts = {
            k: v for k, v in model_kwargs.items()
            if k in [
                "bg_noise_folder", "mic_noise_folder", "bg_speech_folder", "music_folder",
                "rir_folder", "vc_folder", "snr_min", "snr_max",
                "pitch_min", "pitch_max", "speed_min", "speed_max",
                "vc_prob",
                "wake_word_over_speech_folder", "wow_prob", "wow_snr_min", "wow_snr_max",
            ]
        }
        # aug_prob is scheduled by the training loop — do not bake it in here
        if resume:
            self.load_checkpoint(resume)

        self.mlflow = None
        if mlflow_uri is not None:
            self._setup_mlflow(mlflow_uri, featurizer, arch, resume, losses_cfg, feature_dim,
                               {**model_kwargs, "featurizer_type": featurizer_type})

    def mlflow_log(self, fn_name: str, *args, **kwargs) -> None:
        """Call trainer.mlflow.<fn_name>(...) safely.

        On any MLflow error (e.g. deleted/finished run), logs a warning and
        disables MLflow for the rest of this training run so subsequent calls
        are no-ops rather than crashes.
        """
        if self.mlflow is None:
            return
        try:
            getattr(self.mlflow, fn_name)(*args, **kwargs)
        except Exception as exc:
            logger.warning("[MLflow] %s failed — disabling MLflow for this run: %s", fn_name, exc)
            self.mlflow = None

    # --------------------- MLflow Setup ---------------------

    def _setup_mlflow(
            self,
            mlflow_uri: str,
            featurizer: str,
            arch: str,
            resume: Optional[str],
            losses_cfg: Optional[List[Dict[str, str]]],
            feature_dim: Optional[int],
            model_kwargs: Dict[str, Any],
    ) -> None:
        """Initialise MLflow tracking for this run.

        Failures are non-fatal — training continues without MLflow if the
        server is unreachable or the experiment setup races with another process.
        """
        import time
        import mlflow
        from ww_trainer.env import load_env
        load_env()  # no-op if already loaded; ensures .env credentials are present
        mlflow.set_tracking_uri(mlflow_uri)

        # Experiment = one wake word.  All architectures/losses for the same
        # keyword land in the same experiment so they're directly comparable.
        wake = self.wake_word.replace("_", " ")   # "hey_mycroft" → "hey mycroft"
        exp_name = f"ww: {wake}"

        # Featurizer label: use the type kwarg if available, else the onnx filename
        featurizer_type = model_kwargs.get("featurizer_type")
        feat_type = featurizer_type or (
            featurizer.split("/")[-1].replace(".onnx", "") if featurizer else "unknown"
        )
        n_feat = (model_kwargs.get("n_mfcc")
                  or model_kwargs.get("n_mels")
                  or model_kwargs.get("n_filters")
                  or model_kwargs.get("n_features")
                  or feature_dim
                  or "?")
        hidden = model_kwargs.get("hidden_dim", "?")
        loss_label = "+".join(l["name"] for l in (losses_cfg or [])) or "bce"

        # Run name: human-readable at a glance in the MLflow UI
        # e.g.  "gru · mfcc-40 · h128 · focal"
        run_name = f"{arch} · {feat_type}-{n_feat} · h{hidden} · {loss_label}"

        # Resolve experiment id with retry — concurrent workers may race to create it.
        exp_id = None
        for attempt in range(5):
            try:
                exp = mlflow.get_experiment_by_name(exp_name)
                if exp is not None and exp.lifecycle_stage == "active":
                    exp_id = exp.experiment_id
                    break
                elif exp is None:
                    exp_id = mlflow.create_experiment(exp_name)
                    break
                else:
                    # Experiment exists but was deleted — create with a unique suffix
                    exp_id = mlflow.create_experiment(f"{exp_name} ({int(time.time())})")
                    break
            except Exception as exc:
                if attempt < 4:
                    time.sleep(0.5 * (attempt + 1))
                else:
                    logger.warning("[MLflow] Could not resolve experiment after 5 attempts: %s", exc)
                    return

        # Only log params that are meaningful in the MLflow UI — skip filesystem paths.
        _PATH_KEYS = {"bg_noise_folder", "music_folder", "rir_folder", "mic_noise_folder",
                      "bg_speech_folder", "vc_folder"}
        clean_kwargs = {k: v for k, v in model_kwargs.items() if k not in _PATH_KEYS}

        try:
            mlflow.start_run(experiment_id=exp_id, run_name=run_name,
                             nested=mlflow.active_run() is not None)
            mlflow.log_params({
                "wake_word":    wake,
                "arch":         arch,
                "featurizer":   feat_type,
                "n_features":   n_feat,
                "hidden_dim":   hidden,
                "loss":         loss_label,
                "device":       str(self.device),
                "resume":       resume or "",
                **clean_kwargs,
            })
            self.mlflow = mlflow
            logger.info("[MLflow] run '%s' in experiment '%s'", run_name, exp_name)
        except Exception as exc:
            logger.warning("[MLflow] Setup failed — continuing without tracking: %s", exc)

    # --------------------- Checkpoint I/O ---------------------

    def save_checkpoint(
            self,
            epoch: int,
            metrics: dict,
            optimizer: torch.optim.Optimizer,
            out_path: "Path | str",
    ) -> None:
        """Save full model + optimizer + trainer metadata."""
        save_checkpoint(self.model, epoch, metrics, optimizer, out_path)

    def load_checkpoint(
            self,
            path: "Path | str",
            optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Tuple[int, dict]:
        """Load model + optimizer + epoch state. Returns (start_epoch, metrics)."""
        return load_checkpoint(self.model, path, self.device, optimizer)

    def save_intermediate_ckpt(
            self,
            model_file: Path,
            optimizer: torch.optim.Optimizer = None,
            metrics: dict = None,
            epoch: int = -1,
    ) -> None:
        """Save checkpoint and optionally export to ONNX + log to MLflow."""
        save_intermediate_checkpoint(
            self.model, self.wake_word, self.arch,
            epoch, metrics or {}, optimizer, model_file,
            self.export_onnx, self.mlflow,
        )

    # --------------------- Model Factory ---------------------

    @staticmethod
    def create_model(
            arch_name: str,
            featurizer: str,
            feature_dim: int = None,
            featurizer_type: str = "onnx",
            sample_rate: int = 16000,
            device: str = "auto",
            shared_extractor=None,
            **kwargs,
    ):
        """Delegate to :func:`ww_trainer.factory.create_model`."""
        return create_model(
            arch_name, featurizer, feature_dim,
            featurizer_type=featurizer_type,
            sample_rate=sample_rate,
            device=device,
            shared_extractor=shared_extractor,
            **kwargs,
        )

    # --------------------- Layer Freezing ---------------------

    def _freeze(self) -> None:
        """Freeze extractor and/or first N classifier layers.

        Controlled by :attr:`freeze_extractor` and :attr:`freeze_layers`.
        """
        if self.freeze_extractor and hasattr(self.model, "feature_extractor"):
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False
            logger.info("[Freeze] Feature extractor frozen")

        if self.freeze_layers > 0 and hasattr(self.model, "classifier"):
            frozen = 0
            for name, param in self.model.classifier.named_parameters():
                if frozen >= self.freeze_layers:
                    break
                param.requires_grad = False
                frozen += 1
            logger.info("[Freeze] Froze %d classifier layer parameters", frozen)

    def _unfreeze(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("[Freeze] All layers unfrozen")

    # --------------------- Training ---------------------

    def train(
            self,
            output_dir: "str | Path",
            train_data: List[Tuple[str, str]],
            test_data: List[Tuple[str, str]],
            **kwargs: Any,
    ) -> float:
        """Run the full training loop. All kwargs forwarded to :func:`~ww_trainer.loop.training_loop`.

        Args:
            output_dir: Directory for checkpoints, plots, and CSV logs.
            train_data: List of ``(path, label)`` tuples for training.
            test_data: List of ``(path, label)`` tuples for evaluation.
            **kwargs: Forwarded to :func:`ww_trainer.loop.training_loop`.

        Returns:
            Best F1 score achieved during training.
        """
        from ww_trainer.loop import training_loop
        return training_loop(self, output_dir, train_data, test_data, **kwargs)

    # --------------------- Inference ---------------------

    def infer(self, audio_tensor: torch.Tensor) -> float:
        """Run single-sample inference, returning wake-word probability."""
        self.model.eval()
        with torch.no_grad():
            logit = self.model([audio_tensor.to(self.device)])
            return float(torch.sigmoid(logit).item())
