"""Tests for the voiceclonnx-backed voice-conversion delegation layer."""
import pytest

from ww_trainer import vc_helpers


def test_list_engines_includes_known_engines():
    engines = vc_helpers.list_engines()
    assert isinstance(engines, list)
    # knnvc is the default; linacodec / chatterbox are notable members.
    for expected in ("knnvc", "linacodec", "chatterbox"):
        assert expected in engines


@pytest.mark.parametrize(
    "requested, resolved",
    [
        ("auto", "knnvc"),               # legacy "auto" → default engine
        ("voiceclonnx", "knnvc"),        # bare backend name → default engine
        ("chatterbox-onnx", "chatterbox"),  # legacy wrapper name → chatterbox engine
        ("chatterbox", "chatterbox"),
        ("linacodec", "linacodec"),      # back-compat alias preserved
        ("voiceclonnx:facodec", "facodec"),  # explicit engine selector
        ("knnvc", "knnvc"),
    ],
)
def test_load_vc_backend_resolves_engine(monkeypatch, requested, resolved):
    """``load_vc_backend`` maps legacy backend names to voiceclonnx engines
    without instantiating the (network-downloading) engine itself."""
    captured = {}

    class _StubConverter:
        def __init__(self, engine):
            captured["engine"] = engine
            self.engine = engine
            self.name = f"voiceclonnx:{engine}"
            self.sample_rate = 16000

    monkeypatch.setattr(vc_helpers, "VoiceConverter", _StubConverter)

    backend = vc_helpers.load_vc_backend(backend=requested)
    assert captured["engine"] == resolved
    assert backend.engine == resolved


def test_load_vc_backend_engine_kwarg_takes_precedence(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        vc_helpers, "VoiceConverter",
        lambda engine: type("S", (), {"engine": engine})(),
    )
    # engine= wins over the legacy backend= alias
    backend = vc_helpers.load_vc_backend(backend="linacodec", engine="freevc")
    assert backend.engine == "freevc"
