"""Shared fixtures for smoke tests."""
import torch


def pytest_configure(config):
    """Set torch to use deterministic mode for reproducible smoke tests."""
    torch.manual_seed(42)
