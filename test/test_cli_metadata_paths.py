"""``ww_trainer-train`` resolves metadata paths beside the CSV and reports missing audio."""
import pytest
from click.testing import CliRunner

import ww_trainer.cli as cli


class _Reached(Exception):
    """Raised by the stand-in trainer: training would have started."""


@pytest.fixture(autouse=True)
def stop_at_trainer(monkeypatch):
    def refuse(*args, **kwargs):
        raise _Reached

    monkeypatch.setattr(cli, "WakeWordTrainer", refuse)


def _dataset(root, n_rows, n_missing, relative):
    clips = root / "clips"
    clips.mkdir()
    lines = []
    for i in range(n_rows):
        name = f"c{i}.wav"
        if i >= n_missing:
            (clips / name).write_bytes(b"x")
        lines.append(f"{'clips/' + name if relative else clips / name},{i % 2}")
    csv_path = root / "metadata.csv"
    csv_path.write_text("\n".join(lines) + "\n")
    return str(csv_path)


def _invoke(csv_path, *extra):
    return CliRunner().invoke(cli.train, [
        "--wake-word", "hey_test", "--metadata", csv_path, "--tier", "micro",
        "--device", "cpu", "--output-dir", str(cli.Path(csv_path).parent / "model"), *extra])


def test_relative_paths_resolve_against_csv_directory(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    csv_path = _dataset(data, 10, 0, relative=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    result = _invoke(csv_path)
    assert isinstance(result.exception, _Reached), result.output
    assert "on 8 samples" in result.output


def test_a_few_missing_files_warn_and_continue(tmp_path):
    csv_path = _dataset(tmp_path, 100, 1, relative=False)
    result = _invoke(csv_path)
    assert isinstance(result.exception, _Reached), result.output
    assert "1 of 100 audio files not found" in result.output
    assert "c0.wav" in result.output


def test_many_missing_files_refused(tmp_path):
    csv_path = _dataset(tmp_path, 100, 50, relative=False)
    result = _invoke(csv_path)
    assert result.exit_code != 0
    assert not isinstance(result.exception, _Reached)
    assert "50 of 100 audio files not found" in result.output
    assert "--allow-missing" in result.output


def test_allow_missing_overrides_the_refusal(tmp_path):
    csv_path = _dataset(tmp_path, 100, 50, relative=False)
    result = _invoke(csv_path, "--allow-missing")
    assert isinstance(result.exception, _Reached), result.output
    assert "50 of 100 audio files not found" in result.output


def test_all_missing_refused_even_for_a_tiny_csv(tmp_path):
    csv_path = _dataset(tmp_path, 3, 3, relative=False)
    result = _invoke(csv_path)
    assert result.exit_code != 0
    assert "3 of 3 audio files not found" in result.output


def test_one_base_per_csv_and_stragglers_are_reported(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    csv_path = _dataset(data, 100, 0, relative=True)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "clips").mkdir(parents=True)
    for i in range(50):
        (data / "clips" / f"c{i}.wav").unlink()
        (elsewhere / "clips" / f"c{i}.wav").write_bytes(b"x")
    monkeypatch.chdir(elsewhere)
    result = _invoke(csv_path)
    assert result.exit_code != 0
    assert "50 of 100 audio files not found" in result.output
    assert "50 of them exist relative to the working directory" in result.output
    result = _invoke(csv_path, "--allow-missing")
    assert isinstance(result.exception, _Reached), result.output
    assert "on 40 samples" in result.output
