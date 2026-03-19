#!/usr/bin/env python
"""Wake-word trainer core: model lifecycle, training loop, and inference."""
import csv
import json
import os.path
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from colorama import Fore, Style
from torch.utils.data import DataLoader
from tqdm import tqdm

from ww_trainer.checkpoint import save_checkpoint, load_checkpoint
from ww_trainer.dataset import AudioDataset, collate_fn
from ww_trainer.evaluation import evaluate_model, log_metrics_csv
from ww_trainer.factory import create_model, EXTRACTOR_REGISTRY, HEAD_REGISTRY
from ww_trainer.loss import LossManager
from ww_trainer.mining import mine_hard_negatives, save_mining_cache, load_mining_cache
from ww_trainer.utils import timed
from ww_trainer.visualization import (
    log_confidence_histogram, log_pca, log_tsne, log_umap,
    log_embeddings_stats,
)


class WakeWordTrainer:
    """Full-featured wake word trainer.

    This class encapsulates the complete training loop, including data loading,
    model setup, loss management, hard-negative mining, evaluation, and logging.

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

    def __init__(self,
                 arch: str,
                 featurizer: str,
                 feature_dim: int = None,
                 device: str = "auto",
                 mlflow_uri: Optional[str] = None,
                 resume: Optional[str] = None,
                 export_onnx: bool = False,
                 losses_cfg: Optional[List[Dict[str, str]]] = None,
                 featurizer_type: str = "onnx",
                 shared_extractor=None,
                 use_amp: bool = False,
                 **model_kwargs: Any) -> None:
        self.losses_cfg = losses_cfg

        device_str: str = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.export_onnx = export_onnx
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None

        self.arch = arch
        self.training_params = model_kwargs
        self.model = self.create_model(arch, featurizer, feature_dim,
                                       featurizer_type=featurizer_type,
                                       shared_extractor=shared_extractor,
                                       device=self.device, **model_kwargs)
        self.model.to(self.device)

        self.augment_opts = {
            k: v for k, v in model_kwargs.items()
            if k in ["bg_noise_folder", "mic_noise_folder", "bg_speech_folder", "music_folder",
                     "rir_folder", "vc_folder",
                     "snr_min", "snr_max",
                     "pitch_min", "pitch_max", "speed_min", "speed_max",
                     "aug_prob", "vc_prob"]
        }
        if resume:
            self.load_checkpoint(resume)

        # MLflow setup
        self.mlflow = None
        if mlflow_uri is not None:
            import mlflow
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(f"WakeWord_Trainer:{self.arch}")
            feat = featurizer.split("/")[-1].split('.onnx')[0]
            mlflow.start_run(run_name=f"{feat}-{arch}")
            mlflow.log_params({
                "arch": arch,
                "device": str(self.device),
                "base_model": resume or "N/A",
                "loss_types": "+".join([l["name"] for l in losses_cfg]),
                "loss_weights": "+".join([str(l["weight"]) for l in losses_cfg]),
                "featurizer": feat,
                "feature_dim": feature_dim,
                **model_kwargs
            })
            self.mlflow = mlflow
            print(f"{Fore.GREEN}[MLflow]{Style.RESET_ALL} Enabled at {mlflow_uri}")

    # --------------------- Checkpoint I/O ---------------------
    def save_checkpoint(self, epoch: int, metrics: dict, optimizer: torch.optim.Optimizer, out_path: "Path | str"):
        """Save full model + optimizer + trainer metadata."""
        save_checkpoint(self.model, epoch, metrics, optimizer, out_path)

    def load_checkpoint(self, path: "Path | str", optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int, dict]:
        """Load model + optimizer + epoch state. Returns (start_epoch, metrics)."""
        return load_checkpoint(self.model, path, self.device, optimizer)

    @staticmethod
    def create_model(arch_name: str, featurizer: str, feature_dim: int = None,
                     featurizer_type: str = "onnx", sample_rate: int = 16000,
                     device: str = "auto",
                     shared_extractor=None, **kwargs):
        """Delegate to ``ww_trainer.factory.create_model``."""
        return create_model(arch_name, featurizer, feature_dim,
                            featurizer_type=featurizer_type,
                            sample_rate=sample_rate,
                            device=device,
                            shared_extractor=shared_extractor,
                            **kwargs)

    # ----------------- Evaluation & training -----------------
    @staticmethod
    def _compute_readiness(stats: dict) -> float:
        """Measures how much hard-negatives the embeddings can handle in next epoch."""
        intra_pos = stats.get("intra_pos_var", 1.0)
        total_var = stats.get("embed_var_total", 1.0)
        readiness = (total_var / (intra_pos + 1e-6))
        readiness = float(np.clip(readiness / 5.0, 0.0, 1.0))
        return readiness

    @timed
    def _evaluate(self, dataset, batch_size=128, threshold=0.5, epoch: int = 0, output_dir: Optional[Path] = None, aug_prob=0):
        """Delegate to ``ww_trainer.evaluation.evaluate_model``."""
        return evaluate_model(
            model=self.model,
            dataset=dataset,
            device=self.device,
            batch_size=batch_size,
            threshold=threshold,
            epoch=epoch,
            output_dir=output_dir,
            aug_prob=aug_prob,
            mlflow=self.mlflow,
        )

    @timed
    def _log_metrics_csv(self, path: str, epoch: int, loss: float, acc: float, prec: float, rec: float,
                         f1: float, auc: float) -> None:
        """Delegate to ``ww_trainer.evaluation.log_metrics_csv``."""
        log_metrics_csv(path, epoch, loss, acc, prec, rec, f1, auc)

    @timed
    def _log_embeddings_stats(self, dataset, epoch: int, batch_size: int = 128):
        return log_embeddings_stats(self.model, dataset, epoch, self.device, batch_size, self.mlflow)

    @timed
    def _log_confidence_histogram(self, targets, probs, epoch: int, outdir: Path, bins: int = 50) -> Optional[str]:
        return log_confidence_histogram(targets, probs, epoch, outdir, bins, self.mlflow)

    @timed
    def _log_pca(self, dataset, outdir: Path, epoch: int, sample_size: int = 200) -> str:
        return log_pca(self.model, dataset, outdir, epoch, self.device, sample_size, self.mlflow)

    @timed
    def _log_tsne(self, dataset, outdir: Path, epoch: int, sample_size: int = 200) -> Optional[str]:
        return log_tsne(self.model, dataset, outdir, epoch, self.device, sample_size, self.mlflow)

    @timed
    def _log_umap(self, dataset, outdir: Path, epoch: int, sample_size: int = 200) -> Optional[str]:
        return log_umap(self.model, dataset, outdir, epoch, self.device, sample_size, self.mlflow)

    # ----------------- Hard-negative mining -----------------
    @timed
    def _mine_hard_negatives(
            self,
            nonwakes: List[Tuple[str, str]],
            neg_threshold: float = 0.5,
            dataset_fraction: float = 0.2,
            cache_decay: float = 0.9,
            max_cache_size: int = 10000,
            use_embedding_mining: bool = True,
            embed_top_k: int = 1000,
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        if not hasattr(self, "hardness_cache"):
            self.hardness_cache = {}
        hard_negatives, easy_negatives, self.hardness_cache = mine_hard_negatives(
            model=self.model,
            nonwakes=nonwakes,
            device=self.device,
            hardness_cache=self.hardness_cache,
            neg_threshold=neg_threshold,
            dataset_fraction=dataset_fraction,
            cache_decay=cache_decay,
            max_cache_size=max_cache_size,
            use_embedding_mining=use_embedding_mining,
            embed_top_k=embed_top_k,
            wake_cache=getattr(self, "_wake_cache", []),
        )
        return hard_negatives, easy_negatives

    # ----------------- Training loop -----------------
    def train(self,
              output_dir: "str | Path",
              train_data: List[Tuple[str, str]],
              test_data: List[Tuple[str, str]],
              epochs: int = 30,
              batch_size: int = 8,
              lr: float = 1e-3,
              neg_threshold: float = 0.5,
              mine_fraction: float = 0.2,
              mining_type: str = "semihard",
              patience: int = 2,
              save_best: bool = True,
              metrics_log: str = "metrics_log.csv",
              pca_every: int = 0,
              tsne_every: int = 0,
              umap_every: int = 0,
              blend_ratio=0.7,
              base_hard=0.5,
              max_hard=5.0,
              base_easy=1.5,
              min_easy=0.2,
              base_random=0.1,
              total_ratio=5,
              use_amp: bool = False,
              accumulate_grad_batches: int = 1,
              resume: Optional[str] = None,
              ) -> None:
        """High-level training loop. Supports loss types: 'bce', 'triplet', 'pair'."""
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        # Effective AMP flag: use instance setting or override from argument
        effective_amp = use_amp or self.use_amp
        scaler = self.scaler if effective_amp else None
        if effective_amp and scaler is None:
            scaler = torch.amp.GradScaler(enabled=True)

        # Load mining cache if resuming
        hardness_cache: Dict[str, float] = {}
        if resume:
            cache_path = str(Path(resume).parent / "hardneg_cache.pt")
            hardness_cache = load_mining_cache(cache_path)
            if hardness_cache:
                print(f"[Mining] Loaded {len(hardness_cache)} cached hardness scores from {cache_path}")

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        initial_lr = lr

        eta_min = lr * (0.05 if len(train_data) > 5000 else 0.2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

        loss_manager = LossManager(loss_configs=self.losses_cfg, mining_type=mining_type, device=self.device)

        wakes = [x for x in train_data if x[1] == "1" and os.path.isfile(x[0])]
        nonwakes = [x for x in train_data if x[1] == "0" and os.path.isfile(x[0])]

        print(f"{Fore.GREEN}Total wake-word samples:{Style.RESET_ALL} {len(wakes)}")
        print(f"{Fore.YELLOW}Total not-wake-word samples:{Style.RESET_ALL} {len(nonwakes)}")

        best_metrics = {"loss": float("inf"), "precision": 0.0, "recall": 0.0, "f1": 0.0}
        epochs_no_new = 0

        hard_negatives: List[Tuple[str, str]] = []
        easy_negatives: List[Tuple[str, str]] = []
        ep = 0
        for ep in range(epochs):
            print(f"\n{Fore.CYAN}=== Epoch {ep + 1}/{epochs} ==={Style.RESET_ALL}")

            current_lr = optimizer.param_groups[0]['lr']
            lr_factor = current_lr / initial_lr
            progress = (ep + 1) / max(1, epochs)

            adaptive_phase = blend_ratio * progress + (1 - blend_ratio) * (1 - lr_factor)

            if ep == 0:
                readiness = 0.1
            else:
                stats = self._log_embeddings_stats(test_data, epoch=ep + 1)
                readiness = self._compute_readiness(stats)

            if not hasattr(self, "_readiness_ema"):
                self._readiness_ema = readiness
            else:
                self._readiness_ema = 0.9 * self._readiness_ema + 0.1 * readiness

            if self.mlflow:
                self.mlflow.log_metrics({"readiness": self._readiness_ema}, step=ep + 1)

            adaptive_phase = (adaptive_phase + self._readiness_ema) / 2.0

            hard_ratio = base_hard + (max_hard - base_hard) * adaptive_phase
            easy_ratio = base_easy - (base_easy - min_easy) * adaptive_phase
            random_ratio = max(base_random, total_ratio - (hard_ratio + easy_ratio))

            print(f"[Adaptive] readiness={self._readiness_ema:.3f} → hard_ratio={hard_ratio:.2f}")
            print(f"{Fore.GREEN}learning-rate={current_lr} hard-ratio={hard_ratio} easy-ratio={easy_ratio} random-ratio={random_ratio}{Style.RESET_ALL}")

            if self.mlflow:
                self.mlflow.log_metrics({
                    "learning-rate": current_lr,
                    "hard-ratio": hard_ratio,
                    "easy-ratio": easy_ratio,
                    "random-ratio": random_ratio,
                }, step=ep + 1)

            selected_hard = random.sample(
                hard_negatives, min(int(len(wakes) * hard_ratio), len(hard_negatives))
            ) if hard_negatives else []
            selected_easy = random.sample(
                easy_negatives, min(int(len(wakes) * easy_ratio), len(easy_negatives))
            ) if easy_negatives else []
            selected_random = random.sample(
                nonwakes, min(int(len(wakes) * random_ratio), len(nonwakes))
            ) if nonwakes else []

            epoch_data = wakes + selected_hard + selected_easy + selected_random
            random.shuffle(epoch_data)

            print(
                f"{Fore.MAGENTA}New data subset:{Style.RESET_ALL} total={len(epoch_data)} "
                f"{Fore.GREEN}wake={len(wakes)}{Style.RESET_ALL} "
                f"{Fore.YELLOW}nonwake={len(epoch_data) - len(wakes)}{Style.RESET_ALL}"
            )
            print(
                f"  {Fore.RED}hard={len(selected_hard)}{Style.RESET_ALL}  "
                f"{Fore.BLUE}easy={len(selected_easy)}{Style.RESET_ALL}  "
                f"{Fore.WHITE}random={len(selected_random)}{Style.RESET_ALL}"
            )

            loader = DataLoader(AudioDataset(epoch_data, device=self.device.type, **self.augment_opts),
                                batch_size=batch_size, shuffle=True,
                                collate_fn=lambda b: collate_fn(b, self.device))
            self.model.train()

            total_loss = 0.0
            loss_breakdown = {cfg["name"]: 0.0 for cfg in self.losses_cfg}

            optimizer.zero_grad()
            for batch_idx, (wavs, labels, _) in enumerate(tqdm(loader, desc="Training", leave=False)):
                with torch.amp.autocast(device_type=self.device.type, enabled=effective_amp):
                    loss, loss_dict = loss_manager.compute_loss(self.model, wavs, labels, loader.dataset)

                loss_scaled = loss / max(1, accumulate_grad_batches)

                if scaler is not None:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                if (batch_idx + 1) % accumulate_grad_batches == 0 or (batch_idx + 1) == len(loader):
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss_dict["total"]
                for k, v in loss_dict.items():
                    if k != "total":
                        loss_breakdown[k] += v

            avg_loss = total_loss / max(1, len(loader))
            for k in loss_breakdown:
                loss_breakdown[k] /= max(1, len(loader))

            print(f"Average total loss: {avg_loss:.4f}")
            for k, v in loss_breakdown.items():
                print(f"  {k} loss: {v:.4f}")

            metrics = {"total_loss": avg_loss, **loss_breakdown}
            if self.mlflow:
                self.mlflow.log_metrics(metrics, step=ep + 1)

            acc, prec, rec, f1, auc, fp_paths, fn_paths, paths_all, targets, preds, probs, det_report = self._evaluate(
                test_data, batch_size=batch_size, threshold=0.4, epoch=ep + 1, output_dir=output_dir)
            print(
                f"{Fore.GREEN}Loss={avg_loss:.4f} Epoch {ep + 1}: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} AUC={auc:.3f} EER={det_report.eer:.4f}{Style.RESET_ALL}")

            if metrics_log:
                self._log_metrics_csv(str(output_dir / metrics_log), ep + 1, avg_loss, acc, prec, rec, f1, auc)

            self._log_confidence_histogram(
                targets=targets,
                probs=probs,
                epoch=ep + 1,
                outdir=output_dir / "viz" / "confidence"
            )

            if pca_every and (ep + 1) % pca_every == 0:
                self._log_pca(test_data, outdir=output_dir / "viz" / "pca", epoch=ep + 1)

            if tsne_every and (ep + 1) % tsne_every == 0:
                self._log_tsne(test_data, outdir=output_dir / "viz" / "tsne", epoch=ep + 1)

            if umap_every and (ep + 1) % umap_every == 0:
                self._log_umap(test_data, outdir=output_dir / "viz" / "umap", epoch=ep + 1)

            if self.mlflow:
                try:
                    eval_metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}
                    self.mlflow.log_metrics(eval_metrics, step=ep + 1)

                    artifacts_dir = output_dir / "artifacts"
                    artifacts_dir.mkdir(parents=True, exist_ok=True)

                    fp_csv = artifacts_dir / f"fp_epoch_{ep + 1}.csv"
                    fn_csv = artifacts_dir / f"fn_epoch_{ep + 1}.csv"

                    with open(fp_csv, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["true_label", "predicted_label", "probability", "file_path"])
                        for path, true, pred, prob in zip(paths_all, targets, preds, probs):
                            if pred == 1 and true == 0:
                                writer.writerow([true, pred, f"{prob:.4f}", path])

                    with open(fn_csv, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["true_label", "predicted_label", "probability", "file_path"])
                        for path, true, pred, prob in zip(paths_all, targets, preds, probs):
                            if pred == 0 and true == 1:
                                writer.writerow([true, pred, f"{prob:.4f}", path])

                    self.mlflow.log_artifact(str(fp_csv), artifact_path="false_positives")
                    self.mlflow.log_artifact(str(fn_csv), artifact_path="false_negatives")
                except Exception as e:
                    print(f"Failed to log metrics/artifacts to MLflow: {e}")

            scheduler.step()

            if save_best:
                updated = []

                if avg_loss <= best_metrics["loss"]:
                    best_metrics["loss"] = avg_loss
                    self.save_intermediate_ckpt(
                        epoch=ep,
                        metrics=best_metrics,
                        optimizer=optimizer,
                        model_file=output_dir / "best_loss.pt")
                    updated.append(f"Loss={avg_loss:.4f}")

                if prec >= best_metrics["precision"]:
                    best_metrics["precision"] = prec
                    self.save_intermediate_ckpt(
                        epoch=ep,
                        metrics=best_metrics,
                        optimizer=optimizer,
                        model_file=output_dir / "best_precision.pt")
                    updated.append(f"Precision={prec:.3f}")

                if rec >= best_metrics["recall"]:
                    best_metrics["recall"] = rec
                    self.save_intermediate_ckpt(
                        epoch=ep,
                        metrics=best_metrics,
                        optimizer=optimizer,
                        model_file=output_dir / "best_recall.pt")
                    updated.append(f"Recall={rec:.3f}")

                if f1 >= best_metrics["f1"]:
                    best_metrics["f1"] = f1
                    self.save_intermediate_ckpt(
                        epoch=ep,
                        metrics=best_metrics,
                        optimizer=optimizer,
                        model_file=output_dir / "best_f1.pt")
                    updated.append(f"F1={f1:.3f}")

                if updated:
                    print(f"* Updated best model(s): {', '.join(updated)}")
            else:
                ckpt = output_dir / f"ep{ep + 1}.pt"
                self.save_intermediate_ckpt(
                    epoch=ep,
                    metrics=best_metrics,
                    optimizer=optimizer,
                    model_file=ckpt)
                print(f"Saved checkpoint: {ckpt}")

            if ep == epochs - 1:
                break

            new_hards, easy_negatives = self._mine_hard_negatives(nonwakes,
                                                                  neg_threshold=neg_threshold,
                                                                  dataset_fraction=mine_fraction,
                                                                  max_cache_size=3 * len(wakes))
            if new_hards:
                print(f"  -> Found {len(new_hards)} false positives")
                merged = {x[0]: x for x in new_hards}
                hard_neg_list = list(merged.values())
                max_samples = len(wakes) * 3 if len(wakes) > 0 else len(hard_neg_list)
                hard_negatives = hard_neg_list[-max_samples:]
                epochs_no_new = 0
            elif mine_fraction > 0:
                epochs_no_new += 1
                print(f"  -> No new false negatives found ({epochs_no_new}/{patience})")
                if epochs_no_new >= patience:
                    print("Early stopping — no new false negatives found.")
                    break

        # final save
        try:
            ckpt = output_dir / "final_model.pt"
            self.save_intermediate_ckpt(
                epoch=ep,
                metrics=best_metrics,
                optimizer=optimizer,
                model_file=ckpt)
            # Persist hard-negative mining cache
            cache_path = str(output_dir / "hardneg_cache.pt")
            save_mining_cache(getattr(self, "hardness_cache", {}), cache_path)
            print(f"Training complete. Saved to {ckpt}")
        except Exception as e:
            print(f"Failed to save final model: {e}")

        if self.mlflow:
            try:
                self.mlflow.end_run()
            except Exception:
                pass

        return best_metrics.get("f1", 0.0)

    def save_intermediate_ckpt(self, model_file: Path,
                               optimizer: torch.optim.Optimizer = None,
                               metrics: dict = None,
                               epoch: int = -1):
        """Save checkpoint and optionally export to ONNX + log to MLflow."""
        onnx_path = model_file.with_suffix(".onnx")

        self.save_checkpoint(
            epoch=epoch,
            metrics=metrics or {},
            optimizer=optimizer,
            out_path=model_file
        )

        if self.export_onnx:
            print(f"Exporting model to onnx: {onnx_path}")
            self.model.export_to_onnx(onnx_path)

        if self.mlflow:
            if self.export_onnx:
                try:
                    self.mlflow.log_artifact(onnx_path, artifact_path="checkpoints")
                except Exception as e:
                    print(f"Failed to log onnx model to MLflow: {e}")
            try:
                self.mlflow.log_artifact(model_file, artifact_path="checkpoints")
            except Exception as e:
                print(f"Failed to log model to MLflow: {e}")

    # --------------------- Inference API ---------------------
    def infer(self, audio_tensor: torch.Tensor) -> float:
        """Run single-sample inference, returning wake-word probability."""
        self.model.eval()
        with torch.no_grad():
            logit = self.model([audio_tensor.to(self.device)])
            return float(torch.sigmoid(logit).item())
