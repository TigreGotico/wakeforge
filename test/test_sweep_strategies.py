"""Tests for grid, random, and genetic search strategies."""
import pytest

from ww_trainer.sweep import _build_search_space


class TestSearchSpace:
    def test_default_space_has_all_keys(self):
        space = _build_search_space()
        assert "arch" in space
        assert "hidden_dim" in space
        assert "lr" in space
        assert "batch_size" in space
        assert "dropout" in space

    def test_all_values_are_lists(self):
        space = _build_search_space()
        for k, v in space.items():
            assert isinstance(v, list), f"{k} is not a list"
            assert len(v) > 0

    def test_full_space_includes_architecture_search(self):
        space = _build_search_space(full=True)
        assert "featurizer_type" in space
        assert "loss" in space
        assert "n_features" in space
        assert len(space["arch"]) > 3  # more than just ffn/gru/cnn
        assert "mfcc" in space["featurizer_type"]
        assert "bcresnet" in space["arch"]
        assert "focal" in space["loss"]

    def test_full_space_has_all_classifiers(self):
        space = _build_search_space(full=True)
        for arch in ["ffn", "gru", "cnn", "bcresnet", "tcresnet",
                      "dscnn", "matchboxnet", "res15", "conformer", "crnn"]:
            assert arch in space["arch"], f"{arch} missing from full arch space"


class TestImports:
    """Verify all search functions are importable."""

    def test_import_grid(self):
        from ww_trainer.sweep import run_grid_search
        assert callable(run_grid_search)

    def test_import_random(self):
        from ww_trainer.sweep import run_random_search
        assert callable(run_random_search)

    def test_import_genetic(self):
        from ww_trainer.sweep import run_genetic_search
        assert callable(run_genetic_search)

    def test_import_optuna_sweep(self):
        from ww_trainer.sweep import run_sweep
        assert callable(run_sweep)
