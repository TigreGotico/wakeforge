#!/usr/bin/env python
import csv
import inspect
import json
import os.path
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from colorama import Fore, Style
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, det_curve
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ww_trainer.dataset import AudioDataset, collate_fn
from ww_trainer.loss import LossManager
from ww_trainer.model import MfccGruWakeModel, MfccCnnWakeModel


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

    def __init__(self, arch: str,
                 device: str = "auto",
                 mlflow_uri: Optional[str] = None,
                 resume: Optional[str] = None,
                 export_onnx: bool = False,
                 losses_cfg: Optional[List[Dict[str, str]]] = None,
                 **model_kwargs: Any) -> None:
        """
        Initialize the WakeWordTrainer.

        Args:
            arch: The model architecture name
            device: The device to use for training ('auto', 'cuda', or 'cpu').
            mlflow_uri: URI for the MLflow tracking server.
            resume: Path to a checkpoint file to resume training from.
            export_onnx: Whether to export the best model to ONNX format.
            **model_kwargs: Additional keyword arguments passed to the model constructor.
        """
        self.losses_cfg = losses_cfg

        device_str: str = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.export_onnx = export_onnx

        self.arch = arch
        self.training_params = model_kwargs
        self.model = self.create_model(arch, device=self.device, **model_kwargs)
        self.model.to(self.device)

        self.augment_opts = {
            k: v for k, v in model_kwargs.items()
            if k in ["bg_noise_folder", "mic_noise_folder", "rir_folder", "vc_folder",
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
            mlflow.start_run(run_name=f"{arch}")
            mlflow.log_params({
                "arch": arch,
                "device": str(self.device),
                "base_model": resume or "N/A",
                "loss_types": "+".join([l["name"] for l in losses_cfg]),
                "loss_weights": "+".join([str(l["weight"]) for l in losses_cfg]),
                **model_kwargs
            })
            self.mlflow = mlflow
            print(f"{Fore.GREEN}[MLflow]{Style.RESET_ALL} Enabled at {mlflow_uri}")

    # --------------------- Checkpoint I/O ---------------------
    def save_checkpoint(self, epoch: int, metrics: dict, optimizer: torch.optim.Optimizer, out_path: Path | str):
        """Save full model + optimizer + trainer metadata using model.save_checkpoint."""
        out_path = Path(out_path) if isinstance(out_path, str) else out_path
        ckpt_dir = out_path.parent
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Let the model handle its internal saving (weights, etc.)
        model_ckpt_path = out_path.with_suffix(".pt")
        self.model.save_checkpoint(str(model_ckpt_path))

        trainer_state = {
            "epoch": epoch,
            "arch": self.arch,
            "metrics": metrics,
            "optimizer_state": optimizer.state_dict() if optimizer is not None else {},
            "device": str(self.device),
        }
        torch.save(trainer_state, out_path.with_suffix(".ts"))
        print(f"[Checkpoint] Saved model + trainer state at epoch {epoch} → {model_ckpt_path}")

    def load_checkpoint(self, path: Path | str, optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int, dict]:
        """Load model + optimizer + epoch state. Returns (start_epoch, metrics)."""
        model_path = Path(path) if isinstance(path, str) else path
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        if hasattr(self.model, "load_checkpoint"):
            self.model.load_checkpoint(str(model_path))
        else:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)

        start_epoch = 0
        metrics = {}
        trainer_state_path = model_path.with_suffix(".ts")
        if trainer_state_path.exists():
            state = torch.load(trainer_state_path, map_location=self.device)
            start_epoch = state.get("epoch", 0)
            metrics = state.get("metrics", {})
            if optimizer is not None and "optimizer_state" in state:
                optimizer.load_state_dict(state["optimizer_state"])
            print(f"[Resume] Loaded checkpoint from epoch {start_epoch}")
        else:
            print(f"[Resume] Loaded model weights only (trainer state missing)")

        return start_epoch, metrics

    @staticmethod
    def create_model(arch_name: str, device: str = "auto", **kwargs):
        arches = [MfccGruWakeModel, MfccCnnWakeModel]
        arch_map = {cls.__name__.lower(): cls for cls in arches}
        key = arch_name.lower()
        if key not in arch_map:
            raise ValueError(f"Unknown architecture '{arch_name}'. Available: {arches}")
        cls = arch_map[key]
        # introspect constructor and pass only accepted kwargs
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.values())[1:]
        allowed = {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        inst_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        if "device" in allowed and "device" not in inst_kwargs:
            inst_kwargs["device"] = device
        return cls(**inst_kwargs)

    # ----------------- Evaluation & training -----------------
    @staticmethod
    def _compute_readiness(stats: dict) -> float:
        intra_pos = stats.get("intra_pos_var", 1.0)
        intra_neg = stats.get("intra_neg_var", 1.0)
        total_var = stats.get("embed_var_total", 1.0)

        # tight wake clusters (low intra_pos) and good spread (high total_var)
        readiness = (total_var / (intra_pos + 1e-6))
        readiness = float(np.clip(readiness / 5.0, 0.0, 1.0))  # normalize to [0,1]
        return readiness

    def _evaluate(self, dataset, batch_size=8, threshold=0.5, epoch: int = 0, output_dir: Optional[Path] = None):
        if not dataset:
            return 0.0, 0.0, 0.0, 0.0, 0.0, [], []

        loader = DataLoader(AudioDataset(dataset), batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, self.device))

        preds, probs, targets = [], [], []
        paths_all = []
        self.model.eval()
        with torch.no_grad():
            for wavs, labels, paths in tqdm(loader, desc="Evaluating", leave=False):
                logits = self.model(wavs)
                prob = torch.sigmoid(logits).cpu().numpy().flatten()
                pred = (prob > threshold).astype(int)
                preds.extend(pred.tolist())
                probs.extend(prob.tolist())
                targets.extend(labels.cpu().int().numpy().tolist())
                paths_all.extend(paths)

        acc = accuracy_score(targets, preds) if len(targets) > 0 else 0.0
        prec = precision_score(targets, preds, zero_division=0) if len(targets) > 0 else 0.0
        rec = recall_score(targets, preds, zero_division=0) if len(targets) > 0 else 0.0
        f1 = f1_score(targets, preds, zero_division=0) if len(targets) > 0 else 0.0
        try:
            auc = roc_auc_score(targets, probs) if len(set(targets)) > 1 else 0.0
        except Exception:
            auc = 0.0

        # track false positives / false negatives (paths)
        fp_paths = [p for p, t, pr in zip(paths_all, targets, preds) if pr == 1 and t == 0]
        fn_paths = [p for p, t, pr in zip(paths_all, targets, preds) if pr == 0 and t == 1]

        if self.mlflow:
            try:
                pos_mean = np.mean(probs[targets == 1]) if np.any(targets == 1) else 0
                neg_mean = np.mean(probs[targets == 0]) if np.any(targets == 0) else 0
                separation = pos_mean - neg_mean
                self.mlflow.log_metrics({
                    "mean_conf_wake": pos_mean,
                    "mean_conf_nonwake": neg_mean,
                    "mean_conf_gap": separation,
                }, step=epoch)
            except Exception as e:
                print(f"Failed to log confidence stats to MLflow: {e}")

        if output_dir is not None:
            plot_dir = Path(output_dir) / "roc_pr_det"
            plot_dir.mkdir(parents=True, exist_ok=True)

            # ROC Curve - TODO add bool param to class __init__
            try:
                fpr, tpr, _ = roc_curve(targets, probs)
                plt.figure()
                plt.plot(fpr, tpr)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve (Epoch {epoch}) AUC={auc:.3f}')
                roc_path = plot_dir / f'roc_epoch_{epoch}.png'
                plt.tight_layout()
                plt.savefig(roc_path, dpi=200)
                plt.close()
            except Exception as e:
                roc_path = None
                print(f"Failed to create ROC plot: {e}")

            # PR Curve - TODO add bool param to class __init__
            try:
                prec_curve, rec_curve, _ = precision_recall_curve(targets, probs)
                plt.figure()
                plt.plot(rec_curve, prec_curve)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve (Epoch {epoch})')
                pr_path = plot_dir / f'pr_epoch_{epoch}.png'
                plt.tight_layout()
                plt.savefig(pr_path, dpi=200)
                plt.close()
            except Exception as e:
                pr_path = None
                print(f"Failed to create PR plot: {e}")

            # DET Curve - TODO add bool param to class __init__
            try:
                fpr_det, fnr_det, _ = det_curve(targets, probs)
                plt.figure()
                plt.plot(fpr_det, fnr_det)
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel('False Positive Rate')
                plt.ylabel('False Negative Rate')
                plt.title(f'DET Curve (Epoch {epoch})')
                det_path = plot_dir / f'det_epoch_{epoch}.png'
                plt.tight_layout()
                plt.savefig(det_path, dpi=200)
                plt.close()
            except Exception as e:
                det_path = None
                print(f"Failed to create DET plot: {e}")

            if self.mlflow:
                try:
                    if roc_path and roc_path.exists():
                        self.mlflow.log_artifact(str(roc_path), artifact_path="roc_pr_det/roc")
                    if pr_path and pr_path.exists():
                        self.mlflow.log_artifact(str(pr_path), artifact_path="roc_pr_det/pr")
                    if det_path and det_path.exists():
                        self.mlflow.log_artifact(str(det_path), artifact_path="roc_pr_det/det")
                except Exception as e:
                    print(f"Failed to log ROC/PR/DET plots to MLflow: {e}")

        return acc, prec, rec, f1, auc, fp_paths, fn_paths, paths_all, targets, preds, probs

    def _evaluate_streaming(self, dataset, epoch: int,
                            chunk_sec: float = 0.5, hop_sec: float = 0.25,
                            threshold: float = 0.7):
        """
        Simulate streaming evaluation:
          - Feed audio in overlapping chunks
          - Use model.forward_streaming()
          - Measure detection accuracy and latency
        """
        loader = DataLoader(AudioDataset(dataset), batch_size=1, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, self.device))
        model = self.model
        model.eval()

        total, correct, triggered = 0, 0, 0
        detection_latencies = []

        for wavs, labels, _ in loader:
            wav = wavs[0].cpu()
            label = labels[0].item()
            model.reset_streaming_state()

            sr = model.sample_rate
            chunk_len = int(chunk_sec * sr)
            hop_len = int(hop_sec * sr)

            triggered_time = None

            for start in range(0, len(wav), hop_len):
                chunk = wav[start:start + hop_len]
                prob = model.forward_streaming(chunk)

                # If wake probability exceeds threshold -> detection
                if prob > threshold and triggered_time is None:
                    triggered_time = start / sr
                    triggered += 1
                    break  # stop scanning this sample

            total += 1
            if label == 1 and triggered_time is not None:
                correct += 1
                detection_latencies.append(triggered_time)
            elif label == 0 and triggered_time is None:
                correct += 1  # correctly did not trigger

            model.reset_streaming_state()

        acc = correct / total
        avg_latency = np.mean(detection_latencies) if detection_latencies else None

        metrics = {
            "stream_acc": acc,
            "stream_avg_latency": avg_latency or 0.0,
            "stream_trigger_rate": triggered / total,
        }
        if self.mlflow:
            self.mlflow.log_metrics(metrics, step=epoch)
        print(f"[Streaming Eval] Acc={acc:.3f}, Avg Latency={avg_latency}, Trigger Rate={triggered}/{total}")

        return metrics

    def _log_confidence_histogram(
        self,
        targets: list[float],
        probs: list[float],
        epoch: int,
        outdir: Path,
        bins: int = 50
    ) -> Optional[str]:
        """
        Plot and optionally log the confidence distribution (sigmoid probabilities)
        for positive (wake) and negative (nonwake) samples.
        """
        if len(targets) == 0 or len(probs) == 0:
            return None

        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"confidence_hist_epoch_{epoch}.png"

        targets = np.array(targets)
        probs = np.array(probs)

        plt.figure(figsize=(6, 4))
        plt.hist(
            probs[targets == 1],
            bins=bins,
            range=(0, 1),
            alpha=0.6,
            label="Wake (label=1)",
            density=True,
            color="tab:blue"
        )
        plt.hist(
            probs[targets == 0],
            bins=bins,
            range=(0, 1),
            alpha=0.6,
            label="Nonwake (label=0)",
            density=True,
            color="tab:orange"
        )
        plt.title(f"Confidence Distribution (Epoch {epoch})")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Density")
        plt.legend(loc="upper center")
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()

        # Optional: log to MLflow
        if self.mlflow:
            try:
                self.mlflow.log_artifact(str(outpath), artifact_path="viz/confidence")
            except Exception as e:
                print(f"Failed to log confidence histogram to MLflow: {e}")

        return str(outpath)

    def _log_metrics_csv(self, path: str, epoch: int, loss: float, acc: float, prec: float, rec: float,
                         f1: float, auc: float) -> None:
        new = not Path(path).exists()
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(["epoch", "loss", "accuracy", "precision", "recall", "f1", "auc"])
            writer.writerow([epoch, loss, acc, prec, rec, f1, auc])

    def _log_embeddings_stats(self, dataset, epoch: int, batch_size: int = 8):
        loader = DataLoader(AudioDataset(dataset), batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, self.device))
        embeds_all, labels_all = [], []
        with torch.no_grad():
            for wavs, labels, _ in loader:
                embeds = self.model.embed(wavs)
                embeds_all.append(embeds.cpu());
                labels_all.append(labels.cpu())
        embeds_all = torch.cat(embeds_all);
        labels_all = torch.cat(labels_all)
        norms = embeds_all.norm(p=2, dim=1)
        stats = {
            "embed_norm_mean": norms.mean().item(),
            "embed_norm_std": norms.std().item(),
            "embed_var_total": embeds_all.var(dim=0).mean().item()
        }
        pos, neg = embeds_all[labels_all == 1], embeds_all[labels_all == 0]
        if len(pos) > 1: stats["intra_pos_var"] = pos.var(dim=0).mean().item()
        if len(neg) > 1: stats["intra_neg_var"] = neg.var(dim=0).mean().item()
        self.mlflow.log_metrics(stats, step=epoch)
        return stats

    def _log_pca(self, dataset, outdir: Path, epoch: int, sample_size: int = 200):
        subset = random.sample(dataset, min(len(dataset), sample_size))
        loader = DataLoader(AudioDataset(subset), batch_size=8, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, self.device))
        embeddings, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for wavs, lbls, _ in loader:
                emb = self.model.embed(wavs)
                embeddings.append(emb.cpu().numpy())
                labels.extend(lbls.cpu().numpy())
        embeddings = np.concatenate(embeddings)
        labels = np.array(labels)

        pca = PCA(n_components=2)
        proj = pca.fit_transform(embeddings)

        plt.figure(figsize=(6, 5))
        plt.scatter(proj[labels == 0, 0], proj[labels == 0, 1], s=15, alpha=0.5, label="Nonwake")
        plt.scatter(proj[labels == 1, 0], proj[labels == 1, 1], s=20, alpha=0.7, label="Wake")
        plt.legend();
        plt.title(f"PCA Embedding (Epoch {epoch})");
        plt.tight_layout()
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"pca_epoch_{epoch}.png"
        plt.savefig(outpath, dpi=200);
        plt.close()

        if self.mlflow:
            try:
                self.mlflow.log_artifact(str(outpath), artifact_path="viz/pca")
            except Exception as e:
                print(f"MLflow PCA log failed: {e}")
        return str(outpath)

    def _log_tsne(self, dataset: List[Tuple[str, str]], outdir: Path, epoch: int, sample_size: int = 200) -> Optional[
        str]:
        if len(dataset) == 0:
            return None
        subset = random.sample(dataset, min(len(dataset), sample_size))
        loader = DataLoader(AudioDataset(subset), batch_size=8, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, self.device))
        embeddings, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for wavs, lbls, _ in loader:
                emb = self.model.embed(wavs)
                embeddings.append(emb.cpu().numpy())
                labels.extend(lbls.cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.array(labels)
        try:
            tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
            proj = tsne.fit_transform(embeddings)
        except Exception as e:
            print(f"t-SNE failed: {e}")
            return None
        plt.figure(figsize=(6, 5))
        plt.scatter(proj[labels == 0, 0], proj[labels == 0, 1], alpha=0.5, s=15, label="Nonwake")
        plt.scatter(proj[labels == 1, 0], proj[labels == 1, 1], alpha=0.7, s=20, label="Wake")
        plt.legend()
        plt.title(f"t-SNE Embedding (Epoch {epoch})")
        plt.tight_layout()
        if isinstance(outdir, str):
            outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"tsne_epoch_{epoch}.png"
        plt.savefig(outpath, dpi=200)
        plt.close()
        if self.mlflow:
            try:
                self.mlflow.log_artifact(str(outpath), artifact_path="viz/tsne")
            except Exception as e:
                print(f"Failed to log t-SNE to MLflow: {e}")
        return str(outpath)

    def _log_umap(self, dataset, outdir: Path, epoch: int, sample_size: int = 200):
        try:
            import umap
        except ImportError:
            print("⚠️ UMAP not installed — skipping. Install with `pip install umap-learn`.")
            return None
        subset = random.sample(dataset, min(len(dataset), sample_size))
        loader = DataLoader(AudioDataset(subset), batch_size=8, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, self.device))
        embeddings, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for wavs, lbls, _ in loader:
                emb = self.model.embed(wavs)
                embeddings.append(emb.cpu().numpy())
                labels.extend(lbls.cpu().numpy())
        embeddings = np.concatenate(embeddings)
        labels = np.array(labels)

        reducer = umap.UMAP(n_components=2, random_state=42)
        proj = reducer.fit_transform(embeddings)

        plt.figure(figsize=(6, 5))
        plt.scatter(proj[labels == 0, 0], proj[labels == 0, 1], s=15, alpha=0.5, label="Nonwake")
        plt.scatter(proj[labels == 1, 0], proj[labels == 1, 1], s=20, alpha=0.7, label="Wake")
        plt.legend();
        plt.title(f"UMAP Embedding (Epoch {epoch})");
        plt.tight_layout()
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"umap_epoch_{epoch}.png"
        plt.savefig(outpath, dpi=200);
        plt.close()

        if self.mlflow:
            try:
                self.mlflow.log_artifact(str(outpath), artifact_path="viz/umap")
            except Exception as e:
                print(f"MLflow UMAP log failed: {e}")
        return str(outpath)

    # ----------------- Hard-negative mining -----------------
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
        """
        Mine hard negatives from nonwake samples based on model confidence and embedding similarity.

        Args:
            nonwakes: list of (path, label="0") pairs
            neg_threshold: probability below which samples are confidently nonwake
            dataset_fraction: fraction of dataset to evaluate for mining
            cache_decay: exponential decay factor for hardness cache
            max_cache_size: max number of cached samples
        """

        # Disable mining → use all nonwake samples directly
        if dataset_fraction <= 0.0:
            print("[HardNegMining] Disabled — using full nonwake dataset")
            return [], nonwakes

        if len(nonwakes) == 0:
            return [], []

        if not hasattr(self, "hardness_cache"):
            self.hardness_cache = {}

        # 1️⃣ Sample a subset of the dataset for efficiency
        sample_size = max(1, int(len(nonwakes) * dataset_fraction))
        subset = random.sample(nonwakes, min(sample_size, len(nonwakes)))

        loader = DataLoader(
            AudioDataset(subset),
            batch_size=32,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, self.device)
        )

        new_scores = {}
        self.model.eval()
        with torch.no_grad():
            for wavs, _, paths in tqdm(loader, desc="Mining negatives", leave=False):
                logits = self.model(wavs)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                for path, p in zip(paths, probs):
                    old = self.hardness_cache.get(path, 0.0)
                    # Samples near or above neg_threshold → hard
                    hardness = max(0.0, p - neg_threshold)
                    new_scores[path] = cache_decay * old + (1 - cache_decay) * hardness

        # 2️⃣ Update rolling cache and prune
        self.hardness_cache.update(new_scores)
        if len(self.hardness_cache) > max_cache_size:
            sorted_items = sorted(self.hardness_cache.items(), key=lambda kv: kv[1], reverse=True)
            self.hardness_cache = dict(sorted_items[:max_cache_size])

        # 3️⃣ Rank-based hard/easy split
        sorted_cache = sorted(self.hardness_cache.items(), key=lambda kv: kv[1], reverse=True)
        cutoff = int(len(sorted_cache) * dataset_fraction)
        hard_paths = [p for p, _ in sorted_cache[:cutoff]]
        easy_paths = [p for p, _ in sorted_cache[cutoff:]]

        hard_negatives = [(p, "0") for p in hard_paths]
        easy_negatives = [(p, "0") for p in easy_paths]

        # 4️⃣ Optional embedding-similarity refinement
        if use_embedding_mining and hasattr(self.model, "embed"):
            try:
                wakes_subset = random.sample(getattr(self, "_wake_cache", []),
                                             min(len(getattr(self, "_wake_cache", [])), 200))
            except Exception:
                wakes_subset = []

            if wakes_subset:
                wake_loader = DataLoader(AudioDataset(wakes_subset), batch_size=16, shuffle=False,
                                         collate_fn=lambda b: collate_fn(b, self.device))
                wake_embeds = []
                with torch.no_grad():
                    for wavs, _, _ in wake_loader:
                        wake_embeds.append(self.model.embed(wavs))
                wake_proto = torch.cat(wake_embeds, dim=0).mean(0, keepdim=True)

                emb_loader = DataLoader(AudioDataset(subset), batch_size=16, shuffle=False,
                                        collate_fn=lambda b: collate_fn(b, self.device))
                emb_sims = {}
                with torch.no_grad():
                    for wavs, _, paths in emb_loader:
                        emb = self.model.embed(wavs)
                        sim = torch.nn.functional.cosine_similarity(emb, wake_proto)
                        for path, s in zip(paths, sim):
                            emb_sims[path] = float(s.cpu())

                for path, s in emb_sims.items():
                    self.hardness_cache[path] = 0.5 * self.hardness_cache.get(path, 0.0) + 0.5 * s

                top_embed = sorted(self.hardness_cache.items(), key=lambda kv: kv[1], reverse=True)[:embed_top_k]
                hard_negatives = [(p, "0") for p, _ in top_embed]

        print(f"Mined {len(hard_negatives)} hard and {len(easy_negatives)} easy negatives (subset={sample_size})")
        return hard_negatives, easy_negatives

    # ----------------- Training loop -----------------
    def train(self,
              output_dir: str | Path,
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
              blend_ratio=0.7,  # 0.5 = equal weight, 0.7 = more progress-driven, 0.3 = more LR-driven
              base_hard=0.5,
              max_hard=5.0,
              base_easy=1.5,
              min_easy=0.2,
              base_random=0.1,
              total_ratio=5,
              ) -> None:
        """High-level training loop. Supports loss types: 'bce', 'triplet', 'pair'."""
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        model_json = output_dir / "model.json"
        if self.arch == "MfccCnnWakeModel":
            model_params = {
                "sample_rate": self.model.sample_rate,
                "n_mels": self.model.n_mels,
                "n_mfcc": self.model.n_mfcc,
                "n_fft": self.model.n_fft,
                "hop_length": self.model.hop_length,
                "feature_type": self.model.feature_type,
                "hidden_dim": self.model.hidden_dim,
            }
        elif self.arch == "MfccGruWakeModel":
            model_params = {
                "sample_rate": self.model.sample_rate,
                "n_mels": self.model.n_mels,
                "n_mfcc": self.model.n_mfcc,
                "n_fft": self.model.n_fft,
                "hop_length": self.model.hop_length,
                "feature_type": self.model.feature_type,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "gru_dropout": self.model.gru_dropout,
#                "batch_first": self.model.batch_first,
                "bidirectional": self.model.bidirectional,
            }
        else:
            raise ValueError(f"invalid architecture: {self.arch}")

        model_json.parent.mkdir(exist_ok=True, parents=True)
        with open(model_json, "w") as f:
            json.dump({
                "arch": self.arch,
                "model_params": model_params,
                "training_params": self.training_params

            }, f, ensure_ascii=False, indent=4)
        # optimizer / criteria
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

        # adaptive ratio schedule initialization
        best_metrics = {"loss": float("inf"), "precision": 0.0, "recall": 0.0, "f1": 0.0}
        epochs_no_new = 0

        hard_negatives = []
        easy_negatives = []
        ep = 0
        for ep in range(epochs):
            print(f"\n{Fore.CYAN}=== Epoch {ep + 1}/{epochs} ==={Style.RESET_ALL}")

            # --- Adaptive ratio schedule based on both LR and training progress ---
            current_lr = optimizer.param_groups[0]['lr']
            lr_factor = current_lr / initial_lr  # 1.0 → lower as LR decays
            progress = (ep + 1) / max(1, epochs)  # 0.0 → 1.0 across training

            # Blend progress and LR decay: smooth, always increasing adaptation
            adaptive_phase = blend_ratio * progress + (1 - blend_ratio) * (1 - lr_factor)

            if ep == 0:
                readiness = 0.5
            else:
                stats = self._log_embeddings_stats(test_data, epoch=ep + 1)
                readiness = self._compute_readiness(stats)

            if not hasattr(self, "_readiness_ema"):
                self._readiness_ema = readiness
            else:
                self._readiness_ema = 0.9 * self._readiness_ema + 0.1 * readiness

            if self.mlflow:
                self.mlflow.log_metrics({"readiness": self._readiness_ema}, step=ep + 1)

            # blend the two signals (phase ~0..1, readiness ~0..1)
            adaptive_phase = (adaptive_phase + self._readiness_ema) / 2.0

            # Adaptive interpolation
            hard_ratio = base_hard + (max_hard - base_hard) * adaptive_phase
            # Easy negatives taper off steadily
            easy_ratio = base_easy - (base_easy - min_easy) * adaptive_phase
            # Maintain at least minimal random exploration
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

            # Sample negatives adaptively
            selected_hard = random.sample(
                hard_negatives, min(int(len(wakes) * hard_ratio), len(hard_negatives))
            ) if hard_negatives else []
            selected_easy = random.sample(
                easy_negatives, min(int(len(wakes) * easy_ratio), len(easy_negatives))
            ) if easy_negatives else []
            selected_random = random.sample(
                nonwakes, min(int(len(wakes) * random_ratio), len(nonwakes))
            ) if nonwakes else []

            # Combine all samples for this epoch
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

            # Training pass
            loader = DataLoader(AudioDataset(epoch_data, **self.augment_opts),
                                batch_size=batch_size, shuffle=True,
                                collate_fn=lambda b: collate_fn(b, self.device))
            self.model.train()

            total_loss = 0.0
            loss_breakdown = {cfg["name"]: 0.0 for cfg in self.losses_cfg}

            for wavs, labels, _ in tqdm(loader, desc="Training", leave=False):
                optimizer.zero_grad()
                loss, loss_dict = loss_manager.compute_loss(self.model, wavs, labels, loader.dataset)
                loss.backward()
                optimizer.step()

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

            # evaluation on test subset
            acc, prec, rec, f1, auc, fp_paths, fn_paths, paths_all, targets, preds, probs = self._evaluate(
                test_data, batch_size=batch_size, threshold=0.4, epoch=ep + 1, output_dir=output_dir)
            print(
                f"{Fore.GREEN}Loss={avg_loss:.4f} Epoch {ep + 1}: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} AUC={auc:.3f}{Style.RESET_ALL}")

            #self._evaluate_streaming(test_data, ep + 1)

            if metrics_log:
                self._log_metrics_csv(str(output_dir / metrics_log), ep + 1, avg_loss, acc, prec, rec, f1, auc)

            # --- Visualization Logging ---
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

            # log to MLflow
            if self.mlflow:
                try:
                    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}
                    self.mlflow.log_metrics(metrics, step=ep + 1)

                    # write FP/FN lists to CSV and log them as artifacts
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

            # scheduler step (monitor F1)
            scheduler.step()

            # checkpoints: best per metric or per-epoch
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
                    print(f"★ Updated best model(s): {', '.join(updated)}")
            else:
                ckpt = output_dir / f"ep{ep + 1}.pt"
                self.save_intermediate_ckpt(
                    epoch=ep,
                    metrics=best_metrics,
                    optimizer=optimizer,
                    model_file=ckpt)
                print(f"Saved checkpoint: {ckpt}")

            # hard-negative mining
            if ep == epochs - 1:
                break

            new_hards, easy_negatives = self._mine_hard_negatives(nonwakes,
                                                                  neg_threshold=neg_threshold,
                                                                  dataset_fraction=mine_fraction,
                                                                  max_cache_size=3 * len(wakes))
            if new_hards:
                print(f"  → Found {len(new_hards)} false positives")
                # merge and deduplicate (by path)
                merged = {x[0]: x for x in new_hards}
                hard_neg_list = list(merged.values())
                max_samples = len(wakes) * 3 if len(wakes) > 0 else len(hard_neg_list)
                hard_negatives = hard_neg_list[-max_samples:]

                epochs_no_new = 0
            elif mine_fraction > 0:
                epochs_no_new += 1
                print(f"  → No new false negatives found ({epochs_no_new}/{patience})")
                if epochs_no_new >= patience:
                    print("Early stopping — no new false negatives found.")
                    break

        # final save
        try:
            ckpt = output_dir / f"final_model.pt"
            self.save_intermediate_ckpt(
                epoch=ep,
                metrics=best_metrics,
                optimizer=optimizer,
                model_file=ckpt)
            print(f"Training complete. Saved to {ckpt}")
        except Exception as e:
            print(f"Failed to save final model: {e}")

        # mlflow wrap-up
        if self.mlflow:
            try:
                self.mlflow.end_run()
            except Exception:
                pass

    def save_intermediate_ckpt(self, model_file: Path,
                               optimizer: torch.optim.Optimizer = None,
                               metrics: dict = None,
                               epoch: int = -1):

        onnx_path = model_file.with_suffix(".onnx")

        self.save_checkpoint(
            epoch=epoch,
            metrics=metrics or {},
            optimizer=optimizer,
            out_path=model_file
        )

        if self.export_onnx:
            print(f"Exporting model to onnx: {onnx_path}")  # TODO add color
            self.model.export_to_onnx(onnx_path)

        if self.mlflow:
            if self.export_onnx:
                try:
                    self.mlflow.log_artifact(onnx_path, artifact_path="checkpoints")
                except Exception as e:
                    print(f"Failed to log onnx model to MLflow: {e}")  # TODO add color
            try:
                self.mlflow.log_artifact(model_file, artifact_path="checkpoints")
            except Exception as e:
                print(f"Failed to log model to MLflow: {e}")  # TODO add color

    # --------------------- Inference API ---------------------
    def infer(self, audio_tensor: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            logit = self.model([audio_tensor.to(self.device)])
            return float(torch.sigmoid(logit).item())


# -------------------- CLI --------------------

@click.command(help="""
Train a wake-word detection model using the WakeWordTrainer.

This command handles full training — data loading, loss setup, adaptive sampling,
hard-negative mining, and evaluation — with optional MLflow tracking and ONNX export.
""")
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
@click.option("--arch", default="MfccGruWakeModel",
              help="Model architecture (e.g., MfccGruWakeModel, MfccCnnWakeModel).")
@click.option("--feature-type", "feature_type", type=click.Choice(['mel', 'mfcc']), default="mfcc",
              help="Depends on arch. Only for Mfcc/Mel feature extractors.")
@click.option("--device", type=click.Choice(["cpu", "cuda", "auto"]), default="auto",
              help="'cuda', 'cpu', or 'auto' (auto-selects CUDA if available).")
@click.option("--sample-rate", type=int, default=16000, help="Audio sample rate used for training.")
@click.option("--export-onnx", is_flag=True,  help="If set, export checkpoints to ONNX format.")
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
              help="Model confidence below which predictions are considered non-wake; used to score hard negatives.")
@click.option("--mine-sample", default=0.2, type=float,
              help="Fraction of the non-wake dataset to sample for mining. "
                   "Set to 0.0 to disable mining and use the full dataset instead.")
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
              help="Blend factor between progress-based and LR-based adaptation (0–1).")
# -------------------------- Augmentation --------------------------
@click.option('--aug-prob', default=0.8, type=float,
              help='Probability of applying any augmentation to each training sample.')
@click.option('--vc-prob', default=0.1, type=float,
              help='Probability of applying voice-cloning augmentation.')
@click.option('--bg-noise-folder', default=None,
              help='Folder with background noise samples for augmentation.')
@click.option('--mic-noise-folder', default=None,
              help='Folder with microphone or silence background clips.')
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
# -------------------------- Logging --------------------------
@click.option("--metrics-log", default="metrics_log.csv",
              help="Path to CSV file where per-epoch metrics will be appended.")
@click.option("--mlflow-uri", default=None,
              help="Optional MLflow tracking URI. Enables experiment logging when set.")
# -------------------------- Visualization --------------------------
@click.option("--pca-every", default=1, type=int,
              help="Run PCA visualization every N epochs (0 disables).")
@click.option("--tsne-every", default=0, type=int,
              help="Run t-SNE embedding visualization every N epochs (0 disables).")
@click.option("--umap-every", default=0, type=int,
              help="Run UMAP embedding visualization every N epochs (0 disables).")
def train(**opts):
    """Train a wake word model using WakeWordTrainer with optional ONNX export."""
    metadata = opts.pop("metadata")
    test_metadata = opts.pop("test_metadata")
    ww_name = opts.get("wake_word")
    mlflow_uri = opts.pop("mlflow_uri")
    arch = opts.pop("arch")
    out_dir = opts.pop("output_dir") or f"trained_models/{arch}/{ww_name}"

    # Set device automatically if needed
    if opts["device"] == "auto":
        opts["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    click.secho(f"Device: {opts['device']}", fg="green", bold=True)

    if opts["mine_sample"] == 0:
        click.secho(f"Hard-negative mining disabled", fg="yellow", bold=True)
    else:
        click.secho(f"Hard-negative mining enabled", fg="green", bold=True)
    # Parse loss config(s)
    loss_types = [x.strip().lower() for x in opts.pop("loss_type").split(",")]
    loss_weights = [float(x.strip()) for x in str(opts.pop("loss_weight")).split(",")]
    if len(loss_weights) == 1 and len(loss_types) > 1:
        loss_weights = [loss_weights[0]] * len(loss_types)
    if len(loss_weights) != len(loss_types):
        raise click.BadParameter("Number of loss weights must match number of loss types")

    losses_cfg = []
    for name, w in zip(loss_types, loss_weights):
        cfg = {"name": name, "weight": w}
        if name in ("triplet", "pair", "cn2pair"):
            cfg["margin"] = opts.get("triplet_margin", 1.0)
        losses_cfg.append(cfg)

    # Load metadata
    with open(metadata, "r", encoding="utf-8") as f:
        entries: List[Tuple[str, str]] = [tuple(line.strip().split(",", 1))
                                          for line in f if line.strip()]
    random.shuffle(entries)

    # Split or use separate test file
    if test_metadata:
        with open(test_metadata, "r", encoding="utf-8") as f:
            test_data = [tuple(line.strip().split(",", 1)) for line in f if line.strip()]
        train_data = entries
    else:
        split_idx = int(len(entries) * opts["split"])
        train_data, test_data = entries[:split_idx], entries[split_idx:]

    # Filter missing files
    train_data = [f for f in train_data if os.path.isfile(f[0])]
    test_data = [f for f in test_data if os.path.isfile(f[0])]

    click.secho(f"🎧 Training {arch} on {len(train_data)} samples", fg="blue", bold=True)

    # Initialize and run trainer
    trainer = WakeWordTrainer(arch=arch, mlflow_uri=mlflow_uri, losses_cfg=losses_cfg, **opts)
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
        blend_ratio=opts["blend_ratio"]
    )

    meta = Path(out_dir) / f"{ww_name}_meta.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    with open(meta, "w") as f:
        json.dump(opts, f, indent=2)
    click.echo(click.style(f"✅ Training complete. Model and config saved to {out_dir}", fg="green", bold=True))


if __name__ == "__main__":
    train()
