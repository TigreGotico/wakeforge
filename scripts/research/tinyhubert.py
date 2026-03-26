#!/usr/bin/env python3
"""
WakeHuBERT — A Minimal HuBERT Distillation Trainer (Streaming Version)

This script trains a small CNN+GRU "student" model to imitate
representations from the HuBERT "teacher" model (facebook/hubert-base-ls960).


Features:
- Streaming dataset loader for spoken-word audio (fixed-length crops/pads)
- MLflow experiment tracking (metrics, checkpoints, ONNX export)
- AUTOMATIC VISUALIZATION: Logs 2D/3D PCA and t-SNE plots of Student vs Teacher embeddings
- RESUME SUPPORT: Can stop and resume training from checkpoints
- CAUSAL ARCHITECTURE: Unidirectional GRU for streaming/wake-word deployment
- CONTRASTIVE LOSS (InfoNCE): Forces student to learn distinct, discriminative features
- The script uses datasets.load_dataset(streaming=True) to load the multi-TB
  MLCommons/ml_spoken_words dataset on the fly without downloading it.
- Training duration is controlled by STEPS_PER_EPOCH instead of full dataset length.

# https://github.com/huggingface/datasets/issues/7693
needs datasets==3.6.0
"""

import os
import random
import tempfile
import time
from typing import List, Tuple, Optional, Dict, Any

import click
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor


# ================================================================
# 1. HuBERT Teacher Wrapper
# ================================================================
class HuBERTTeacher:
    """
    Wrapper around HuggingFace's pretrained HuBERT model.
    """

    def __init__(self, hubert_checkpoint: str = "facebook/hubert-base-ls960", device: str = "cuda"):
        # Determine device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[Teacher] Loading {hubert_checkpoint} on {self.device}")

        # Load feature extractor and model
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_checkpoint)
        self.model = HubertModel.from_pretrained(hubert_checkpoint).to(self.device).eval()

        # Freeze all teacher parameters
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @torch.no_grad()
    def extract_batch(self, wavs: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        """
        Extracts HuBERT features from a batch of raw waveforms.
        Input wavs: [B, T] or [T]
        Output: [B, T', D]
        """
        if wavs.dim() == 2:
            raw = [w.cpu().numpy() for w in wavs]
        else:
            raw = [wavs.cpu().numpy()]

        # Process raw numpy arrays with feature extractor
        inputs = self.feature_extractor(raw, sampling_rate=sr, return_tensors="pt", padding=True)

        # Move inputs to device and run through model
        x = inputs.input_values.to(self.device)
        out = self.model(x).last_hidden_state.clone()
        return out


# ================================================================
# 2. Student Model — Tiny CNN + GRU network
# ================================================================
class CnnBlock(nn.Module):
    """
    A small residual 1D CNN block.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int, s: int, p: int = 0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, s, p, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

        if in_ch != out_ch or s != 1:
            self.res = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, s, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.res = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class WakeHuBERTStudent(nn.Module):
    """
    A small CNN + Unidirectional GRU student network.
    """

    def __init__(
            self,
            out_dim: int = 768,
            base_dim: int = 64,
            gru_hidden: int = 128,
            gru_layers: int = 2,
            device: str = "cuda",
            bidirectional: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # CNN encoder (downsamples time dimension)
        self.cnn = nn.Sequential(
            CnnBlock(1, base_dim, 10, 10),
            CnnBlock(base_dim, base_dim * 2, 4, 4),
            CnnBlock(base_dim * 2, base_dim * 4, 3, 2, 1),
            CnnBlock(base_dim * 4, base_dim * 4, 3, 2, 1),
        )

        # Unidirectional GRU for Causal Inference
        self.gru = nn.GRU(
            base_dim * 4,
            gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.ln = nn.LayerNorm(gru_hidden * 2 if bidirectional else gru_hidden)

        # Project to output dim
        gru_out_dim = gru_hidden * 2 if bidirectional else gru_hidden
        self.proj = nn.Linear(gru_out_dim, out_dim)
        self.out_dim = out_dim

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, 1, T] (1 channel raw audio)
        assert x.ndim == 3, f"Expected [B, 1, T], got {x.shape}"

        x = x.to(self.device)
        x = self.cnn(x)  # [B, C, T']
        x = x.transpose(1, 2)  # [B, T', C]
        x, _ = self.gru(x)  # [B, T', H]
        x = self.ln(x)  # LayerNorm
        return self.proj(x)  # [B, T', F]


# ================================================================
# 3. Dataset Loader (STREAMING VERSION)
# ================================================================

def get_streaming_data(
        lang: List[str],
        crop_seconds: float = 1.0,
        sr: int = 16000,
) -> Tuple[DataLoader, DataLoader]:
    """
    Loads MLCommons/ml_spoken_words dataset in streaming mode,
    applies preprocessing, and returns PyTorch DataLoaders.
    """
    DATASET_NAME = "MLCommons/ml_spoken_words"
    crop_len = int(sr * crop_seconds)
    SHUFFLE_BUFFER_SIZE = 1000

    print(f"[Dataset] Loading {DATASET_NAME} in streaming mode for languages: {lang}")

    # 1. Load the dataset stream (IterableDataset)
    train_stream = load_dataset(
        DATASET_NAME,
        languages=lang,
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    val_stream = load_dataset(
        DATASET_NAME,
        languages=lang,
        split="validation",
        streaming=True,
        trust_remote_code=True
    )

    # 2. Define Preprocessing Function
    def preprocess_function(example: Dict[str, Any]) -> Dict[str, Any]:
        """Loads, resamples, crops, and pads the audio data."""
        # Audio is loaded as a dict containing 'array' (float array) and 'sampling_rate'
        wav = torch.tensor(example["audio"]["array"]).float().unsqueeze(0)  # [1, T]

        # Resample if needed (datasets usually handles this, but safety check is good)
        if example["audio"]["sampling_rate"] != sr:
            wav = torchaudio.functional.resample(wav, example["audio"]["sampling_rate"], sr)

        # Apply fixed-length cropping or padding
        n = wav.shape[1]
        if n < crop_len:
            # Pad with zeros
            wav = F.pad(wav, (0, crop_len - n))
        else:
            # Random cropping
            start = random.randint(0, n - crop_len)
            wav = wav[:, start:start + crop_len]

        # The key we want to batch is 'input_values' ([1, T])
        example["input_values"] = wav
        return example

    # 3. Apply preprocessing and shuffle
    # Shuffle is mandatory for training streams
    train_stream = train_stream.map(preprocess_function).shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, seed=42)
    # Validation data is also streamed and can be lightly shuffled
    val_stream = val_stream.map(preprocess_function).shuffle(buffer_size=SHUFFLE_BUFFER_SIZE // 5, seed=42)

    def collate_fn(examples: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Custom collate function for stacking the preprocessed audio tensors.
        Input: list of dicts (examples). Output: [B, 1, T] tensor.
        """
        # We extracted the audio into 'input_values' in the preprocess step,
        # which are all now [1, T] due to fixed cropping/padding.
        return torch.stack([e["input_values"] for e in examples]).float()

    # 4. Create DataLoaders
    # The DataLoader is what allows us to iterate over the IterableDataset
    # without loading all data at once.
    train_loader = DataLoader(train_stream, batch_size=batch_size, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_stream, batch_size=batch_size, collate_fn=collate_fn, num_workers=1)

    return train_loader, val_loader


# ================================================================
# 4. Loss Functions
# ================================================================
def info_nce_loss(
        student_feats: torch.Tensor,
        teacher_feats: torch.Tensor,
        temperature: float = 0.1,
        max_samples: int = 2048
) -> torch.Tensor:
    """
    Computes Contrastive Loss (InfoNCE).
    """
    # Flatten time and batch dimensions
    s_flat = student_feats.reshape(-1, student_feats.shape[-1])  # [N, D]
    t_flat = teacher_feats.reshape(-1, teacher_feats.shape[-1])  # [N, D]

    # Subsample if too large to fit in VRAM for N^2 matrix
    N = s_flat.shape[0]
    if N > max_samples:
        indices = torch.randperm(N)[:max_samples].to(s_flat.device)
        s_flat = s_flat[indices]
        t_flat = t_flat[indices]

    # Normalize for cosine similarity
    s_norm = F.normalize(s_flat, dim=-1)
    t_norm = F.normalize(t_flat, dim=-1)

    # Similarity matrix: Student -> Teacher
    # Shape: [K, K] where K is num samples
    logits = torch.matmul(s_norm, t_norm.T) / temperature

    # Labels: The correct match for row i is column i
    labels = torch.arange(logits.shape[0], device=logits.device)

    return F.cross_entropy(logits, labels)


# ================================================================
# 5. Visualization Logic
# ================================================================
def log_embeddings_visualizations(
        student: WakeHuBERTStudent,
        teacher: HuBERTTeacher,
        val_loader: torch.utils.data.DataLoader,
        step: int,
        max_points: int = 2000
) -> None:
    print(f"[Viz] Generating embedding plots for step {step}...")
    student.eval()

    # Get a batch from the *streaming* validation loader
    try:
        batch = next(iter(val_loader))
    except StopIteration:
        # Happens if the validation stream runs out (unlikely with this dataset)
        print("[Viz] Validation stream temporarily empty. Skipping visualization.")
        return

    with torch.no_grad():
        # Input is [B, 1, T]
        wav = batch.to(student.device)
        stu_feats = student(wav)

        # Apply adapter if present
        stu_proj = student.adapter(stu_feats) if hasattr(student, "adapter") else stu_feats

        # Teacher input requires [B, T]
        wav_teacher = wav.squeeze(1).to(teacher.device)
        tea_feats = teacher.extract_batch(wav_teacher).to(student.device)

        # Time align the teacher features to match the student's sequence length
        T_s = stu_proj.size(1)
        tea_feats = tea_feats.permute(0, 2, 1)  # [B, D, T']
        tea_feats = F.interpolate(tea_feats, size=T_s, mode="linear", align_corners=False)
        tea_feats = tea_feats.permute(0, 2, 1)  # [B, T', D]

        # Flatten features
        stu_flat = stu_proj.reshape(-1, stu_proj.size(-1)).cpu().numpy()
        tea_flat = tea_feats.reshape(-1, tea_feats.size(-1)).cpu().numpy()

    # Subsample points for dimensionality reduction (PCA/t-SNE is slow otherwise)
    n_total = stu_flat.shape[0]
    n_samples = min(max_points // 2, n_total)

    # Randomly select indices
    idx = np.random.choice(n_total, n_samples, replace=False)
    X_stu_sub = stu_flat[idx]
    X_tea_sub = tea_flat[idx]

    X_combined = np.vstack([X_stu_sub, X_tea_sub])
    y_combined = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])  # 0=Student, 1=Teacher

    def plot_scatter(data, labels, title, mode="2d"):
        fig = plt.figure(figsize=(10, 8))
        if mode == "3d":
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[labels == 0, 0], data[labels == 0, 1], data[labels == 0, 2], c='blue', alpha=0.4,
                       label='Student', s=10)
            ax.scatter(data[labels == 1, 0], data[labels == 1, 1], data[labels == 1, 2], c='red', alpha=0.4,
                       label='Teacher', s=10)
        else:
            ax = fig.add_subplot(111)
            ax.scatter(data[labels == 0, 0], data[labels == 0, 1], c='blue', alpha=0.4, label='Student', s=10)
            ax.scatter(data[labels == 1, 0], data[labels == 1, 1], c='red', alpha=0.4, label='Teacher', s=10)
        ax.legend()
        ax.set_title(title)
        plt.tight_layout()
        safe_title = title.lower().replace(" ", "_").replace("-", "_")
        mlflow.log_figure(fig, f"plots/{safe_title}_step_{step}.png")
        plt.close(fig)

    # PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_combined)
    plot_scatter(X_pca, y_combined, "PCA 2D", mode="2d")
    plot_scatter(X_pca, y_combined, "PCA 3D", mode="3d")

    # t-SNE (slow, limit complexity)
    tsne_2d = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
    X_tsne_2 = tsne_2d.fit_transform(X_combined)
    plot_scatter(X_tsne_2, y_combined, "t-SNE 2D", mode="2d")

    student.train()  # Switch back to training mode


# ================================================================
# 6. Utils
# ================================================================
def export_onnx(student: WakeHuBERTStudent, path: str, sample_len: int = 16000) -> None:
    student.eval()
    dummy = torch.randn(1, 1, sample_len).to(student.device)
    torch.onnx.export(
        student, dummy, path,
        input_names=["waveform"], output_names=["embedding"],
        dynamic_axes={"waveform": {2: "n_samples"}, "embedding": {1: "time"}},
        opset_version=18, dynamo=False
    )
    print(f"[ONNX] exported to {path}")


def save_checkpoint(student, opt, sched, epoch, best_val, filename):
    state = {
        "epoch": epoch, "model_state": student.state_dict(),
        "optimizer_state": opt.state_dict(), "scheduler_state": sched.state_dict(),
        "best_val": best_val,
    }
    torch.save(state, filename)


# ================================================================
# 7. Training Loop
# ================================================================
def train(
        lang: List[str],  # Language list from CLI
        mlflow_uri: Optional[str],
        experiment: str,
        run_name: str,
        epochs: int,
        batch_size: int,
        lr: float,
        crop_seconds: float,
        cnn_dim: int,
        gru_hidden: int,
        gru_layers: int,
        output_dim: int,
        # max_files is no longer relevant for streaming
        device: str,
        plot_interval: int,
        resume: Optional[str],
) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params({
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "cnn_dim": cnn_dim, "gru_hidden": gru_hidden, "output_dim": output_dim,
        "languages": ", ".join(lang),
        "resumed_from": resume if resume else "None",
        "architecture": "Causal/Streaming",
        "loss_strategy": "MSE + InfoNCE (Contrastive)"
    })

    # --- STREAMING DATA LOADING ---
    train_loader, val_loader = get_streaming_data(lang, crop_seconds, sr=16000)

    # Since the streaming dataset has no inherent length, we define an epoch by steps
    STEPS_PER_EPOCH = 5000
    VAL_STEPS = 500
    print(f"[Config] Epoch defined as {STEPS_PER_EPOCH} steps. Validation is {VAL_STEPS} steps.")
    # ------------------------------

    teacher = HuBERTTeacher(device=device)
    teacher_dim = teacher.hidden_size

    # Use bidirectional=False for Causal/Streaming compatibility
    student = WakeHuBERTStudent(output_dim, cnn_dim, gru_hidden, gru_layers,
                                device=device, bidirectional=False)

    if output_dim != teacher_dim:
        print(f"[Distill] Mismatch: Student {output_dim} vs Teacher {teacher_dim}. Creating Adapter.")
        # Adapter ensures student features match teacher's hidden size (768)
        student.add_module("adapter", nn.Linear(output_dim, teacher_dim).to(device))
    else:
        print(f"[Distill] Dimensions Match ({teacher_dim}). No adapter needed.")

    # Optimizer and Scheduler
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    best_val = float("inf")
    start_epoch = 1

    # Resume logic (kept mostly the same, now handles adapter creation dynamically)
    if resume and os.path.isfile(resume):
        print(f"[Resume] Loading checkpoint from {resume}...")
        checkpoint = torch.load(resume, map_location=device)
        model_state = checkpoint["model_state"]

        # Attempt to create adapter if required by the checkpoint weights
        if "adapter.weight" in model_state and not hasattr(student, "adapter"):
            out_f, in_f = model_state["adapter.weight"].shape
            student.add_module("adapter", nn.Linear(in_f, out_f).to(device))
            opt = torch.optim.Adam(student.parameters(), lr=lr)  # Re-initialize optimizer to include adapter params

        try:
            student.load_state_dict(model_state)
            opt.load_state_dict(checkpoint["optimizer_state"])
            sched.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_val = checkpoint["best_val"]
            print(f"[Resume] Success! Continuing from Epoch {start_epoch}")
        except Exception as e:
            print(f"[Resume] Failed to load checkpoint state dict: {e}")
            print("Starting from scratch (check for architecture changes).")
            start_epoch = 1

    try:
        for ep in range(start_epoch, epochs + 1):
            student.train()
            tr_losses = []
            tr_nce = []
            tr_mse = []

            # --- START STREAMING EPOCH ---
            # IMPORTANT: Call set_epoch to ensure proper shuffling for the next epoch iteration
            if hasattr(train_loader.dataset, 'set_epoch'):
                train_loader.dataset.set_epoch(ep)

            step = 0
            for batch in tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [train]"):
                if batch.size(0) != batch_size:
                    # Skip incomplete last batch if the step count is low,
                    # but should be rare in streaming mode.
                    continue

                wav = batch.to(student.device)  # Input [B, 1, T]

                stu_feats = student(wav)
                stu_proj = student.adapter(stu_feats) if hasattr(student, "adapter") else stu_feats

                # Teacher requires [B, T]
                wav_teacher = wav.squeeze(1).to(teacher.device)
                tea_feats = teacher.extract_batch(wav_teacher).to(student.device)

                # Time align the teacher's output length to the student's output length
                T_s = stu_proj.size(1)
                tea_feats = tea_feats.permute(0, 2, 1)
                tea_feats = F.interpolate(tea_feats, size=T_s, mode="linear", align_corners=False)
                tea_feats = tea_feats.permute(0, 2, 1)

                # Distillation Losses
                loss_mse = F.mse_loss(stu_proj, tea_feats)

                # InfoNCE (Contrastive) - forces discriminative features
                loss_nce = info_nce_loss(stu_proj, tea_feats, temperature=0.07)

                # Total Loss (weight NCE slightly lower)
                loss = loss_mse + (0.1 * loss_nce)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
                opt.step()

                tr_losses.append(loss.item())
                tr_mse.append(loss_mse.item())
                tr_nce.append(loss_nce.item())

                step += 1
                if step >= STEPS_PER_EPOCH:
                    break
            # --- END STREAMING EPOCH ---

            sched.step()
            mlflow.log_metric("train_loss", np.mean(tr_losses), step=ep)
            mlflow.log_metric("train_nce", np.mean(tr_nce), step=ep)

            # Validation Loop (limited steps)
            student.eval()
            val_losses = []
            val_steps = 0
            with torch.no_grad():
                if hasattr(val_loader.dataset, 'set_epoch'):
                    val_loader.dataset.set_epoch(ep)  # Set epoch for validation stream too

                for batch in tqdm(val_loader, desc=f"Epoch {ep}/{epochs} [val]"):
                    if batch.size(0) != batch_size:
                        continue

                    wav = batch.to(student.device)
                    stu_feats = student(wav)
                    stu_proj = student.adapter(stu_feats) if hasattr(student, "adapter") else stu_feats

                    wav_teacher = wav.squeeze(1).to(teacher.device)
                    tea_feats = teacher.extract_batch(wav_teacher).to(student.device)

                    T_s = stu_proj.size(1)
                    tea_feats = tea_feats.permute(0, 2, 1)
                    tea_feats = F.interpolate(tea_feats, size=T_s, mode="linear", align_corners=False)
                    tea_feats = tea_feats.permute(0, 2, 1)

                    l_mse = F.mse_loss(stu_proj, tea_feats)
                    l_nce = info_nce_loss(stu_proj, tea_feats, temperature=0.07)
                    val_losses.append((l_mse + 0.1 * l_nce).item())

                    val_steps += 1
                    if val_steps >= VAL_STEPS:
                        break

            val_mean = np.mean(val_losses)
            mlflow.log_metric("val_loss", val_mean, step=ep)
            print(
                f"[Epoch {ep}] Loss: {np.mean(tr_losses):.4f} (MSE={np.mean(tr_mse):.3f}, NCE={np.mean(tr_nce):.3f}) | Val: {val_mean:.4f}")

            # Visualization and Checkpointing
            if ep % plot_interval == 0 and val_steps > 0:
                # Pass val_loader to the visualization function
                log_embeddings_visualizations(student, teacher, val_loader, step=ep)

            save_checkpoint(student, opt, sched, ep, best_val, "last_checkpoint.pt")
            if val_mean < best_val:
                best_val = val_mean
                # Save best checkpoint artifact to MLflow
                tmp = os.path.join(tempfile.mkdtemp(), "best_student.pt")
                save_checkpoint(student, opt, sched, ep, best_val, tmp)
                mlflow.log_artifact(tmp, artifact_path="best_model")

        # Final ONNX export
        onnx_path = os.path.join(tempfile.mkdtemp(), "tinyhubert.onnx")
        export_onnx(student, onnx_path)
        mlflow.log_artifact(onnx_path, artifact_path="onnx")

    finally:
        mlflow.end_run()
        print("[Training] Complete.")


# ================================================================
# 8. CLI (UPDATED)
# ================================================================
@click.command()
@click.option("--lang", required=True, multiple=True,
              help="List of languages to stream (e.g., --lang en --lang fr).")
@click.option("--mlflow-uri", default=None, help="MLflow tracking URI.")
@click.option("--experiment", default="RCNN_student", help="MLflow experiment name.")
@click.option("--run-name", default=None, help="Optional MLflow run name.")
@click.option("--epochs", default=100, show_default=True)
@click.option("--batch-size", default=32, show_default=True)
@click.option("--lr", default=1e-3, show_default=True)
@click.option("--crop-seconds", default=1.0, show_default=True)
@click.option("--cnn-dim", default=512, show_default=True)
@click.option("--gru-hidden", default=256, show_default=True)
@click.option("--gru-layers", default=1, show_default=True)
@click.option("--output-dim", default=768, show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--plot-interval", default=10, show_default=True)
@click.option("--resume", default=None, type=click.Path(exists=True, dir_okay=False))
def main(**kwargs):
    """
    Train Causal TinyHuBERT student model with InfoNCE loss using streaming.
    """
    global batch_size
    batch_size = kwargs.pop("batch_size")  # Extract batch size globally for data loader setup
    run_name = kwargs.pop("run_name") or f"run_{int(time.time())}_{'_'.join(kwargs['lang'])}"
    kwargs['lang'] = list(kwargs['lang'])  # wont accept tuples downstream
    train(run_name=run_name, batch_size=batch_size, **kwargs)


if __name__ == "__main__":
    main()
