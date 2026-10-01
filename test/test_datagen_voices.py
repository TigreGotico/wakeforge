"""Synthesized positives come from many voices, not each plugin's default one.

A plugin called with no voice says the wake word in its default voice every
time, so 1,000 clips from edge-tts were 1,000 readings by one speaker. Each clip
now draws one of the plugin's voices, accents or rates, and a remote-server
plugin is only used when a host is configured, since it otherwise falls back
to public servers.
"""
import csv
import json
import random
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from ww_trainer import datagen


@pytest.fixture(autouse=True)
def _no_pacing(monkeypatch):
    monkeypatch.setattr(datagen, "_MIN_INTERVAL_S", {}, raising=False)


class _Recorder:
    """A TTS plugin that writes a placeholder file and records every call's options."""
    calls = []

    def __init__(self, config=None):
        self.config = config or {}

    def get_tts(self, sentence, wav_file, lang=None, voice=None, rate=None):
        type(self).calls.append({"plugin": type(self).__name__, "lang": lang, "voice": voice, "rate": rate,
                                 "config": self.config})
        sf.write(wav_file, np.zeros(1600, dtype=np.float32), 16000)
        return wav_file, None


def _plugins(monkeypatch, **classes):
    for c in classes.values():
        c.calls = []
    fake = types.ModuleType("ovos_plugin_manager.tts")
    fake.find_tts_plugins = lambda: dict(classes)
    monkeypatch.setitem(sys.modules, "ovos_plugin_manager.tts", fake)


def _edge_voices(monkeypatch, voices):
    mod = types.ModuleType("ovos_tts_plugin_edge_tts")
    mod.VOICES = voices
    monkeypatch.setitem(sys.modules, "ovos_tts_plugin_edge_tts", mod)


class Edge(_Recorder):
    pass


class Server(_Recorder):
    pass


class WithVoices(_Recorder):
    available_voices = ["a", "b", "c"]


def test_edge_clips_are_spread_over_its_voices_and_rates(monkeypatch, tmp_path):
    random.seed(0)
    _plugins(monkeypatch, **{"ovos-tts-plugin-edge-tts": Edge})
    _edge_voices(monkeypatch, {"en-US": ["en-US-A", "en-US-B"], "en-GB": ["en-GB-C"], "de-DE": ["de-DE-X"]})
    datagen.synthesize_positives("hey jarvis", tmp_path, n=60, lang="en-us")
    voices = {c["voice"] for c in Edge.calls}
    assert voices == {"en-US-A", "en-US-B", "en-GB-C"}, "every English voice, and no German one"
    assert len({c["rate"] for c in Edge.calls}) > 1


def test_a_plugin_listing_its_voices_uses_them(monkeypatch, tmp_path):
    random.seed(0)
    _plugins(monkeypatch, other=WithVoices)
    datagen.synthesize_positives("computer", tmp_path, n=40, lang="en")
    assert {c["voice"] for c in WithVoices.calls} == {"a", "b", "c"}


def test_a_server_plugin_without_a_host_is_not_used(monkeypatch, tmp_path):
    _plugins(monkeypatch, **{"ovos-tts-plugin-server": Server, "other": WithVoices})
    datagen.synthesize_positives("computer", tmp_path, n=20, lang="en")
    assert Server.calls == [], "no host configured, so no request may reach a public server"


def test_a_configured_server_uses_its_host_and_voice_list(monkeypatch, tmp_path):
    random.seed(0)
    _plugins(monkeypatch, **{"ovos-tts-plugin-server": Server})
    cfg = {"ovos-tts-plugin-server": {"host": "http://ser9:9666", "voices": ["vctk", "alba"]}}
    datagen.synthesize_positives("computer", tmp_path, n=30, lang="en", tts_config=cfg)
    assert {c["voice"] for c in Server.calls} == {"vctk", "alba"}
    assert all(c["config"]["host"] == "http://ser9:9666" for c in Server.calls)


def test_every_written_clip_has_a_manifest_row(monkeypatch, tmp_path):
    random.seed(0)
    _plugins(monkeypatch, other=WithVoices)
    written = datagen.synthesize_positives("computer", tmp_path, n=12, lang="en")
    with open(tmp_path / "synthesis.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert sorted(r["file"] for r in rows) == sorted(p.name for p in written)
    assert {json.loads(r["options"])["voice"] for r in rows} <= {"a", "b", "c"}


def test_the_tts_config_file_is_read(tmp_path):
    p = tmp_path / "tts.json"
    p.write_text(json.dumps({"ovos-tts-plugin-server": {"host": "http://h"}}))
    assert datagen.load_tts_config(str(p)) == {"ovos-tts-plugin-server": {"host": "http://h"}}
    assert datagen.load_tts_config(None) == {}


def test_the_cli_accepts_the_tts_config(monkeypatch):
    seen = {}
    monkeypatch.setattr(datagen, "run_datagen_pipeline", lambda cfg: seen.setdefault("cfg", cfg))
    monkeypatch.setattr(sys, "argv", ["ww_trainer-datagen", "--wake-word", "computer", "--output-dir", "/tmp/x",
                                      "--tts-config", "/tmp/tts.json"])
    datagen.cli_main()
    assert seen["cfg"].tts_config == "/tmp/tts.json"


class Flaky(_Recorder):
    """Fails twice, writing a partial file each time, then succeeds."""
    attempts = 0

    def get_tts(self, sentence, wav_file, lang=None, voice=None, rate=None):
        type(self).attempts += 1
        if type(self).attempts % 3:
            with open(wav_file, "wb") as fh:
                fh.write(b"RIFF partial")
            raise RuntimeError("429 Too Many Requests")
        return super().get_tts(sentence, wav_file, lang, voice, rate)


class Refuses(_Recorder):
    def get_tts(self, sentence, wav_file, lang=None, voice=None, rate=None):
        with open(wav_file, "wb") as fh:
            fh.write(b"RIFF partial")
        raise RuntimeError("429 Too Many Requests")


def test_a_refused_request_is_retried_and_leaves_no_file(monkeypatch, tmp_path):
    monkeypatch.setattr(datagen.time, "sleep", lambda s: None)
    _plugins(monkeypatch, flaky=Flaky)
    Flaky.attempts = 0
    written = datagen.synthesize_positives("computer", tmp_path, n=4, lang="en")
    assert len(written) == 4 and Flaky.attempts == 12
    assert sorted(p.name for p in tmp_path.glob("*.wav")) == sorted(p.name for p in written)


def test_a_clip_that_never_succeeds_leaves_nothing_behind(monkeypatch, tmp_path):
    monkeypatch.setattr(datagen.time, "sleep", lambda s: None)
    _plugins(monkeypatch, refuses=Refuses)
    assert datagen.synthesize_positives("computer", tmp_path, n=3, lang="en") == []
    assert list(tmp_path.glob("*.wav")) == []
    with open(tmp_path / "synthesis.csv", newline="") as fh:
        assert len(list(csv.DictReader(fh))) == 0


def test_requests_to_one_plugin_are_spaced_by_its_interval(monkeypatch, tmp_path):
    clock = [0.0]
    slept = []
    monkeypatch.setattr(datagen.time, "monotonic", lambda: clock[0])

    def sleep(s):
        slept.append(s)
        clock[0] += s

    monkeypatch.setattr(datagen.time, "sleep", sleep)
    _plugins(monkeypatch, other=WithVoices)
    cfg = {"other": {"min_interval_s": 2.0}}
    datagen.synthesize_positives("computer", tmp_path, n=4, lang="en", tts_config=cfg)
    assert slept == [2.0, 2.0, 2.0], "three waits of the full interval between four instant calls"


class NoVoiceKeyword(_Recorder):
    """A plugin whose get_tts takes no voice or rate keyword."""
    available_voices = ["a", "b"]

    def get_tts(self, sentence, wav_file, lang=None):
        return super().get_tts(sentence, wav_file, lang)


def test_options_a_plugin_does_not_accept_are_dropped(monkeypatch, tmp_path):
    monkeypatch.setattr(datagen.time, "sleep", lambda s: None)
    _plugins(monkeypatch, plain=NoVoiceKeyword)
    written = datagen.synthesize_positives("computer", tmp_path, n=3, lang="en")
    assert len(written) == 3 and len(NoVoiceKeyword.calls) == 3
    assert all(c["lang"] == "en" for c in NoVoiceKeyword.calls)


def test_a_plugin_that_never_succeeds_is_reported_with_its_count(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(datagen.time, "sleep", lambda s: None)
    _plugins(monkeypatch, refuses=Refuses)
    with caplog.at_level("WARNING", logger=datagen.logger.name):
        datagen.synthesize_positives("computer", tmp_path, n=3, lang="en")
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING" and "produced no audio" in r.getMessage()]
    assert len(warnings) == 1 and "3 clips" in warnings[0] and "429" in warnings[0]


class HeaderOnly(_Recorder):
    """Writes a 44-byte WAV header with no samples, as a refused request can."""

    def get_tts(self, sentence, wav_file, lang=None, voice=None, rate=None):
        sf.write(wav_file, np.zeros(0, dtype=np.int16), 16000, subtype="PCM_16")
        return wav_file, None


def test_a_header_with_no_audio_is_not_a_clip(monkeypatch, tmp_path):
    monkeypatch.setattr(datagen.time, "sleep", lambda s: None)
    _plugins(monkeypatch, empty=HeaderOnly)
    assert datagen.synthesize_positives("computer", tmp_path, n=2, lang="en") == []
    assert list(tmp_path.glob("*.wav")) == []


def test_google_clips_are_spread_over_its_accents_of_the_language(monkeypatch):
    mod = types.ModuleType("ovos_tts_plugin_google_tx")
    mod.REGIONAL_CONFIGS = {"en-US": {}, "en-GB": {}, "en-AU": {}, "pt-PT": {}}
    monkeypatch.setitem(sys.modules, "ovos_tts_plugin_google_tx", mod)
    opts = datagen.voice_options("ovos-tts-plugin-google-tx", object(), "en-us")
    assert opts == [{"lang": "en-US"}, {"lang": "en-GB"}, {"lang": "en-AU"}]
