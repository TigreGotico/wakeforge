"""Shared fixtures for ww-trainer tests.

All fixtures generate synthetic audio data (sine waves) so tests run
on CPU without real audio files or model weights.
"""
import json

import numpy as np
import pytest
import soundfile as sf
import torch


SAMPLE_RATE = 16000
DURATION_S = 0.5  # seconds — keep short for fast CI


@pytest.fixture
def sample_rate() -> int:
    return SAMPLE_RATE


@pytest.fixture
def wav_tensor() -> torch.Tensor:
    """Single 1-D float32 waveform tensor at 16 kHz."""
    t = torch.linspace(0, DURATION_S, int(SAMPLE_RATE * DURATION_S))
    return torch.sin(2 * torch.pi * 440 * t)


@pytest.fixture
def wav_np() -> np.ndarray:
    """Single 1-D float32 numpy waveform at 16 kHz."""
    t = np.linspace(0, DURATION_S, int(SAMPLE_RATE * DURATION_S))
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def wav_file(tmp_path, wav_np) -> str:
    """Write a synthetic WAV to a temp file and return the path string."""
    path = tmp_path / "test_audio.wav"
    sf.write(str(path), wav_np, SAMPLE_RATE)
    return str(path)


@pytest.fixture
def two_wav_files(tmp_path) -> list:
    """Two synthetic WAV files at different frequencies."""
    paths = []
    for freq in [440, 880]:
        t = np.linspace(0, DURATION_S, int(SAMPLE_RATE * DURATION_S))
        wav = np.sin(2 * np.pi * freq * t).astype(np.float32)
        p = tmp_path / f"audio_{freq}.wav"
        sf.write(str(p), wav, SAMPLE_RATE)
        paths.append(str(p))
    return paths


@pytest.fixture
def random_feats() -> torch.Tensor:
    """Random feature tensor [2, 50, 768] simulating HuBERT output."""
    torch.manual_seed(42)
    return torch.randn(2, 50, 768)


@pytest.fixture
def binary_labels() -> torch.Tensor:
    """Simple binary label tensor [1, 0] for batch size 2."""
    return torch.tensor([1, 0], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Stand-in pretrained featurizers (no network)
# ---------------------------------------------------------------------------

PRETRAINED_HOP = 320
PRETRAINED_DIM = 128


class _StandInFeaturizer(torch.nn.Module):
    """Causal stand-in: a stride-320 conv, then a conv over the 20 previous frames."""

    def __init__(self, seed, dim):
        super().__init__()
        torch.manual_seed(seed)
        self.frames = torch.nn.Conv1d(1, dim, kernel_size=PRETRAINED_HOP, stride=PRETRAINED_HOP)
        self.context = torch.nn.Conv1d(dim, dim, kernel_size=21)

    def forward(self, waveform):
        x = torch.tanh(self.frames(waveform.unsqueeze(1)))
        x = torch.tanh(self.context(torch.nn.functional.pad(x, (20, 0))))
        return x.transpose(1, 2)


def export_stand_in_featurizer(path, seed, dim=PRETRAINED_DIM):
    """Export a stand-in with the published I/O contract: waveform [B, N] -> features [B, N//320, dim]."""
    torch.onnx.export(_StandInFeaturizer(seed, dim).eval(), torch.zeros(1, 16000), str(path),
                      input_names=["waveform"], output_names=["features"],
                      dynamic_axes={"waveform": {0: "batch", 1: "samples"},
                                    "features": {0: "batch", 1: "frames"}},
                      opset_version=18, dynamo=False)


def _stand_in_config(name):
    from ww_trainer.pretrained import _REPOS
    licence = _REPOS[name][0]
    return {
        "family": "stand-in",
        "license": licence,
        "output": {"name": "features", "frame_rate_hz": 50.0, "hop_samples": PRETRAINED_HOP},
        "feature_dim": 256 if name.endswith("-wide") else PRETRAINED_DIM,
        "streaming": not name.endswith("bigru"),
        "files": {"float32": "wakehubert.onnx", "int8": "wakehubert_int8.onnx"},
    }


@pytest.fixture(scope="session")
def stand_in_repos(tmp_path_factory):
    """One directory per published repository, holding stand-in ONNX files."""
    from ww_trainer.pretrained import _REPOS
    root = tmp_path_factory.mktemp("hub")
    onnx = {}
    for dim in (PRETRAINED_DIM, 256):
        for variant, seed in (("wakehubert.onnx", 0), ("wakehubert_int8.onnx", 1)):
            onnx[dim, variant] = root / f"{dim}_{variant}"
            export_stand_in_featurizer(onnx[dim, variant], seed, dim)
    repos = {}
    for name in _REPOS:
        repo = root / name
        repo.mkdir()
        config = _stand_in_config(name)
        (repo / "config.json").write_text(json.dumps(config))
        for variant in config["files"].values():
            (repo / variant).symlink_to(onnx[config["feature_dim"], variant])
        repos[f"TigreGotico/{name}"] = repo
    return repos


@pytest.fixture
def fake_hub(stand_in_repos, monkeypatch):
    """Serve every registry repository from ``stand_in_repos`` instead of the Hub."""
    import ww_trainer.pretrained as pretrained
    calls = []

    def fake_download(repo_id, filename, revision=None, **kwargs):
        calls.append((repo_id, filename, revision))
        return str(stand_in_repos[repo_id] / filename)

    monkeypatch.setattr(pretrained, "hf_hub_download", fake_download)
    return stand_in_repos["TigreGotico/wakehubert-tiny"], calls


@pytest.fixture(autouse=True)
def fresh_feature_stores():
    """Every test starts and ends with no feature stores holding memory."""
    from ww_trainer.feature_store import FeatureStore
    FeatureStore.close_all()
    yield
    FeatureStore.close_all()
