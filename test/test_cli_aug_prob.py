import json

import pytest
from click.testing import CliRunner

from ww_trainer import cli


class _Stop(Exception):
    pass


def _invoke(monkeypatch, tmp_path, extra, finish=False):
    seen = {"train": []}

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            seen["init"] = kwargs
            extractor = type("E", (), {"feature_dim": 1})()
            self.model = type("M", (), {"feature_extractor": extractor})()

        def train(self, **kwargs):
            seen["train"].append(kwargs)
            if finish:
                return 0.0
            raise _Stop

    monkeypatch.setattr(cli, "WakeWordTrainer", FakeTrainer)
    monkeypatch.setattr("ww_trainer.feature_store.store_for", lambda *a, **k: None)
    for i in range(4):
        (tmp_path / f"c{i}.wav").write_bytes(b"x")
    meta = tmp_path / "m.csv"
    meta.write_text("".join(f"{tmp_path / f'c{i}.wav'},{i % 2}\n" for i in range(4)))
    out = tmp_path / "out"
    result = CliRunner().invoke(cli.train, ["--wake-word", "x", "--metadata", str(meta),
                                            "--featurizer-type", "mfcc", "--no-feature-cache",
                                            "--output-dir", str(out), *extra])
    if not finish:
        assert isinstance(result.exception, _Stop), result.output
    return seen, out, result


PATHS = [[], ["--training-stages", "1:1e-3,1:1e-4"]]


@pytest.mark.parametrize("stages", PATHS, ids=["plain", "multi-stage"])
def test_aug_flags_reach_the_training_loop_and_not_the_model(monkeypatch, tmp_path, stages):
    seen, _, _ = _invoke(monkeypatch, tmp_path,
                         ["--aug-prob", "0.35", "--aug-warmup-epochs", "5", *stages])
    assert seen["train"][0]["aug_prob"] == 0.35
    assert seen["train"][0]["aug_warmup_epochs"] == 5
    assert "aug_prob" not in seen["init"]
    assert "aug_warmup_epochs" not in seen["init"]


@pytest.mark.parametrize("stages", PATHS, ids=["plain", "multi-stage"])
def test_aug_defaults_reach_the_training_loop(monkeypatch, tmp_path, stages):
    seen, _, _ = _invoke(monkeypatch, tmp_path, stages)
    assert seen["train"][0]["aug_prob"] == 0.8
    assert seen["train"][0]["aug_warmup_epochs"] == 3


def test_the_run_record_keeps_the_augmentation_settings(monkeypatch, tmp_path):
    _, out, result = _invoke(monkeypatch, tmp_path,
                             ["--aug-prob", "0.35", "--aug-warmup-epochs", "5"], finish=True)
    assert result.exception is None, result.output
    record = json.loads((out / "x_meta.json").read_text())
    assert record["aug_prob"] == 0.35
    assert record["aug_warmup_epochs"] == 5
