"""The voice grid enumerates donor voices x rates x pitches x forms, writes them resumably and optionally clones and deduplicates them.

Every test uses a fake engine, a fake cloner and a fake speaker embedder: no network, no TTS model.
"""
import csv
import os
import threading
import types
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ww_trainer import voicegrid as vg


class FakeEngine(vg.GridEngine):
    def __init__(self, name="fake", voices=("v1", "v2"), rate=True, pitch=True, concurrency=4, fail_first=0):
        self.name = name
        self._voices = list(voices)
        self._rate = rate
        self._pitch = pitch
        self.max_concurrency = concurrency
        self.fail_first = fail_first
        self.calls = []
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def voices(self, lang):
        return list(self._voices)

    def has_rate(self, voice):
        return self._rate

    def has_pitch(self, voice):
        return self._pitch

    def synth(self, text, voice, rate, pitch):
        with self._lock:
            self.calls.append((text, voice, rate, pitch))
            self.active += 1
            self.peak = max(self.peak, self.active)
            if self.fail_first:
                self.fail_first -= 1
                self.active -= 1
                raise RuntimeError("service refused")
        time.sleep(0.01)
        with self._lock:
            self.active -= 1
        t = np.arange(8000) / 8000
        return 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32), 8000


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setattr(vg, "_RETRY_WAIT_S", 0.0)


def args(tmp_path, **over):
    base = dict(wake_word="hey_jarvis", output_dir=tmp_path / "out", lang="en", donors=None, phrases_file=None, omnivoice_clips=300, omnivoice_rates=vg.OMNIVOICE_RATES, omnivoice_style=[], require_asr_langs="",
                device="cpu", batch_size=4, verify="none", verify_percentile=5.0, verify_threshold=None, verify_all_donors=False,
                asr_model="m", asr_threshold=0.8, accept_spellings=[], tts_config=None, workers=3,
                rates=vg.RATES, pitches=vg.PITCHES, forms=vg.DEFAULT_FORMS, exclude_voices=[], no_default_excludes=False,
                max_voices_per_engine=None, max_clean_per_voice=None, vc_refs=None, vc_copies=1, vc_engine="chatterbox",
                vc_device="cpu", dedup_speaker_threshold=None, dedup_model="m", label=1, seed=0, dry_run=False)
    base.update(over)
    return Namespace(**base)


def manifest(tmp_path):
    with open(tmp_path / "out" / "manifest.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def test_grid_is_voices_times_rates_pitches_forms(tmp_path):
    engine = FakeEngine()
    cells = vg.enumerate_grid([engine], {"fake": ["v1", "v2"]}, "hey jarvis")
    assert len(cells) == 2 * 5 * 3 * 3
    assert len({c.filename for c in cells}) == 90
    assert {c.text for c in cells} == {"hey jarvis", "hey jarvis!", "hey jarvis?"}


def test_axes_an_engine_lacks_collapse(tmp_path):
    engine = FakeEngine(rate=False, pitch=False)
    cells = vg.enumerate_grid([engine], {"fake": ["v1"]}, "hey jarvis")
    assert len(cells) == 3
    assert all(c.rate is None and c.pitch is None for c in cells)
    edge_like = FakeEngine(rate=True, pitch=False)
    assert len(vg.enumerate_grid([edge_like], {"fake": ["v1"]}, "x")) == 5 * 3


def test_held_out_kokoro_speakers_are_excluded_across_versions():
    class Kokoro(FakeEngine):
        def voices(self, lang):
            return [f"{v}/{s}" for v in ("kokoro", "kokoro-v019", "kokoro-v11")
                    for s in ("af_heart", "af_kore", "am_echo", "am_puck", "bf_lily", "bm_fable", "af_bella", "am_adam")]

    kept = vg.select_voices(Kokoro(), "en", vg.DEFAULT_EXCLUDE_VOICES, None, 0)
    assert sorted(kept) == sorted(f"{v}/{s}" for v in ("kokoro", "kokoro-v019", "kokoro-v11") for s in ("af_bella", "am_adam"))


def test_exclude_option_adds_to_the_defaults(tmp_path):
    engine = FakeEngine(voices=("kokoro/af_heart", "kokoro/af_bella", "kokoro/am_adam"), rate=False, pitch=False)
    vg.run_voicegrid(args(tmp_path, exclude_voices=["kokoro/am_*"], forms=("plain",)), [engine])
    assert {r["voice"] for r in manifest(tmp_path)} == {"kokoro/af_bella"}
    vg.run_voicegrid(args(tmp_path, no_default_excludes=True, forms=("plain",), output_dir=tmp_path / "out2"), [engine])
    assert {r["voice"] for r in csv.DictReader(open(tmp_path / "out2" / "manifest.csv"))} == {
        "kokoro/af_heart", "kokoro/af_bella", "kokoro/am_adam"}


def test_same_speaker_twins_keep_one_name():
    names = ["en-US-AvaNeural", "en-US-AvaMultilingualNeural", "en-US-BrianMultilingualNeural", "en-US-EmmaNeural",
             "en-US-EmmaMultilingualNeural", "en-US-GuyNeural"]
    assert vg.drop_twins(names) == ["en-US-AvaNeural", "en-US-BrianMultilingualNeural", "en-US-EmmaNeural", "en-US-GuyNeural"]


def test_write_is_16k_mono_pcm_and_manifest_lists_every_parameter(tmp_path):
    engine = FakeEngine(voices=("v1",))
    assert vg.run_voicegrid(args(tmp_path), [engine]) == 0
    rows = manifest(tmp_path)
    assert tuple(rows[0]) == vg.MANIFEST_COLUMNS
    assert len(rows) == 45
    first = rows[0]
    info = sf.info(tmp_path / "out" / "clips" / first["file"])
    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
    assert first["stage"] == "tts" and first["engine"] == "fake" and first["lang"] == "en" and first["copy"] == "0"
    assert {r["rate"] for r in rows} == {"-25", "-12", "0", "12", "25"}
    assert {r["pitch"] for r in rows} == {"-10", "0", "10"}
    assert {r["form"] for r in rows} == {"plain", "exclaim", "question"}
    assert {r["text"] for r in rows} == {"hey jarvis", "hey jarvis!", "hey jarvis?"}


def test_metadata_csv_lists_kept_clips_with_the_label_and_no_header(tmp_path):
    vg.run_voicegrid(args(tmp_path, label=0, forms=("plain",), rates=(0,), pitches=(0, 10)), [FakeEngine(voices=("v1",))])
    lines = (tmp_path / "out" / "metadata.csv").read_text().splitlines()
    assert len(lines) == 2 and all(l.startswith("clips/hey-jarvis-") and "__fake__v1__" in l and l.endswith(".wav,0") for l in lines)
    assert all((tmp_path / "out" / l.rsplit(",", 1)[0]).exists() for l in lines)


def test_rerun_skips_existing_and_fills_only_the_gap(tmp_path, capsys):
    engine = FakeEngine(voices=("v1",))
    vg.run_voicegrid(args(tmp_path), [engine])
    assert len(engine.calls) == 45
    engine.calls.clear()
    vg.run_voicegrid(args(tmp_path), [engine])
    assert engine.calls == []
    clips = tmp_path / "out" / "clips"
    victim = sorted(clips.glob("*.wav"))[7]
    victim.unlink()
    (clips / "stray.wav.part").write_bytes(b"x")
    vg.run_voicegrid(args(tmp_path), [engine])
    assert len(engine.calls) == 1 and victim.exists()
    assert "stage=tts planned=45 existing=44 written=1 failed=0" in capsys.readouterr().out


def test_failed_clip_is_retried_then_counted_and_left_out(tmp_path):
    engine = FakeEngine(voices=("v1",), fail_first=2)
    assert vg.run_voicegrid(args(tmp_path), [engine]) == 0
    assert len(manifest(tmp_path)) == 45

    class Broken(FakeEngine):
        def synth(self, *a):
            raise RuntimeError("down")

    failed = vg.run_voicegrid(args(tmp_path, output_dir=tmp_path / "bad", forms=("plain",), rates=(0,), pitches=(0,)),
                              [Broken(voices=("v1",))])
    assert failed == 1
    assert list(csv.DictReader(open(tmp_path / "bad" / "manifest.csv"))) == []


def test_engine_concurrency_cap_is_respected(tmp_path):
    engine = FakeEngine(concurrency=2)
    vg.run_voicegrid(args(tmp_path, workers=8), [engine])
    assert engine.peak <= 2


def test_max_clean_per_voice_is_a_seeded_subset(tmp_path):
    engine = FakeEngine()
    a = vg.enumerate_grid([engine], {"fake": ["v1", "v2"]}, "x", max_clean_per_voice=5, seed=1)
    b = vg.enumerate_grid([engine], {"fake": ["v1", "v2"]}, "x", max_clean_per_voice=5, seed=1)
    c = vg.enumerate_grid([engine], {"fake": ["v1", "v2"]}, "x", max_clean_per_voice=5, seed=2)
    assert len(a) == 10 and a == b and a != c


def test_dry_run_writes_nothing(tmp_path, capsys):
    vg.run_voicegrid(args(tmp_path, dry_run=True), [FakeEngine()])
    assert not (tmp_path / "out").exists()
    assert "stage=plan cells=90" in capsys.readouterr().out


class FakeCloner:
    provider = "CUDAExecutionProvider"
    sample_rate = 24000

    def __init__(self, used=("CUDAExecutionProvider", "CUDAExecutionProvider")):
        self.used = list(used)
        self.pairs = []

    def clone(self, source, reference, out_path):
        self.pairs.append((Path(source).name, Path(reference).name))
        t = np.arange(12000) / 24000
        sf.write(out_path, 0.1 * np.sin(2 * np.pi * 300 * t).astype(np.float32), 24000)

    def providers_used(self):
        return self.used


def refs_dir(tmp_path, n=4):
    d = tmp_path / "refs"
    (d / "sub").mkdir(parents=True)
    for i in range(n):
        sf.write(d / ("sub" if i % 2 else ".") / f"ref{i}.wav", np.zeros(16000, np.float32), 16000)
    return d


def test_cloning_makes_copies_with_deterministic_references(tmp_path, monkeypatch):
    cloner = FakeCloner()
    monkeypatch.setattr(vg, "_load_cloner", lambda engine, device: cloner)
    kw = dict(vc_refs=refs_dir(tmp_path), vc_copies=2, forms=("plain",), rates=(0,), pitches=(0, 10))
    vg.run_voicegrid(args(tmp_path, **kw), [FakeEngine()])
    rows = manifest(tmp_path)
    cloned = [r for r in rows if r["stage"] == "vc"]
    assert len(rows) == 4 + 8 and len(cloned) == 8
    assert {r["copy"] for r in cloned} == {"1", "2"}
    assert all(r["source"].endswith("__c0.wav") and r["file"].endswith(("__c1.wav", "__c2.wav")) for r in cloned)
    assert all(r["reference"] for r in cloned) and len({r["file"] for r in cloned}) == 8
    first = {r["file"]: r["reference"] for r in cloned}
    cloner.pairs.clear()
    vg.run_voicegrid(args(tmp_path, **kw), [FakeEngine()])
    assert cloner.pairs == []
    other = tmp_path / "again"
    vg.run_voicegrid(args(tmp_path, output_dir=other, **{k: v for k, v in kw.items()}), [FakeEngine()])
    assert {r["file"]: r["reference"] for r in csv.DictReader(open(other / "manifest.csv")) if r["stage"] == "vc"} == first
    info = sf.info(tmp_path / "out" / "clips" / cloned[0]["file"])
    assert (info.samplerate, info.channels) == (16000, 1)


def test_cloning_refuses_when_the_requested_device_is_not_the_one_in_use(tmp_path, monkeypatch):
    monkeypatch.setattr(vg, "_load_cloner", lambda engine, device: FakeCloner(used=("CPUExecutionProvider", "CUDAExecutionProvider")))
    with pytest.raises(SystemExit, match="CUDAExecutionProvider"):
        vg.run_voicegrid(args(tmp_path, vc_refs=refs_dir(tmp_path), forms=("plain",), rates=(0,), pitches=(0,)), [FakeEngine()])


class FakeEmbedder:
    """Embeds by the pitch in the file name, so equal pitch means the same speaker."""

    def __call__(self, path):
        name = Path(path).name
        group = {"p-10": 0, "p+0": 1, "p+10": 2}[next(k for k in ("p-10", "p+0", "p+10") if f"__{k}__" in name)]
        v = np.zeros(8, np.float32)
        v[group] = 1.0
        return v


def test_dedup_keeps_cloned_clips_before_clean_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(vg, "_load_cloner", lambda engine, device: FakeCloner())
    monkeypatch.setattr(vg, "_load_embedder", lambda model: FakeEmbedder())
    vg.run_voicegrid(args(tmp_path, vc_refs=refs_dir(tmp_path), forms=("plain",), rates=(0,), pitches=(-10, 0, 10),
                          dedup_speaker_threshold=0.9), [FakeEngine(voices=("v1",))])
    with open(tmp_path / "out" / "dedup.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    kept = [r["file"] for r in rows if r["kept"] == "1"]
    assert len(rows) == 6 and len(kept) == 3
    listed = (tmp_path / "out" / "metadata.csv").read_text().splitlines()
    assert sorted(l.split(",")[0][len("clips/"):] for l in listed) == sorted(kept)
    assert all(f.endswith("__c1.wav") for f in kept), "each speaker group keeps its cloned clip"
    dropped = [r for r in rows if r["kept"] == "0"]
    assert all(r["duplicate_of"].endswith("__c1.wav") and float(r["similarity"]) >= 0.9 for r in dropped)


def test_dedup_is_seeded_and_a_threshold_above_one_keeps_everything(tmp_path, monkeypatch):
    files = [{"file": f"a__p{p}__c0.wav", "stage": "tts"} for p in ("-10", "+0", "+10")]
    monkeypatch.setattr(vg, "_load_embedder", lambda model: (lambda path: np.array([1.0, 0.0, 0.0], np.float32)))
    first = vg.dedup_speakers(files, tmp_path, 0.5, "m", 1, seed=3)
    assert first == vg.dedup_speakers(files, tmp_path, 0.5, "m", 1, seed=3)
    assert sum(r["kept"] for r in first) == 1
    assert sum(r["kept"] for r in vg.dedup_speakers(files, tmp_path, 1.01, "m", 1, seed=3)) == 3


def test_cli_dry_run_through_the_donor_builder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(vg, "build_donors", lambda a: [FakeEngine()])
    with pytest.raises(SystemExit) as exit_:
        vg.cli_main(["--wake-word", "hey jarvis", "--output-dir", str(tmp_path / "o"),
                     "--dry-run", "--rates=-10,10", "--pitches", "0", "--forms", "plain,question"])
    assert exit_.value.code == 0
    assert "stage=plan cells=8" in capsys.readouterr().out


def test_cli_rejects_an_unknown_form(tmp_path):
    with pytest.raises(SystemExit) as exit_:
        vg.cli_main(["--wake-word", "x", "--output-dir", str(tmp_path), "--forms", "shout"])
    assert exit_.value.code == 2


def test_language_matching():
    assert vg.lang_matches("en-GB", "en") and vg.lang_matches("en", "en-us") and vg.lang_matches("en-US", "en-us")
    assert not vg.lang_matches("en-GB", "en-us") and not vg.lang_matches("de-DE", "en")


CATALOG = tuple(
    vg.CatalogVoice(vid, lang, engine, index) for vid, lang, engine, index in [
        ("kokoro/af_bella", "en-US", "styletts2", "styletts2"),
        ("kokoro/af_heart", "en-US", "styletts2", "styletts2"),
        ("bsc/ca-styletts2", "ca-ES", "styletts2", "styletts2"),
        ("supertonic/F1/en", "en", "supertonic", "supertonic"),
        ("supertonic/F1/de", "de", "supertonic", "supertonic"),
        ("omnivoice/en", "en", "omnivoice", "omnivoice"),
        ("omnivoice/de", "de", "omnivoice", "omnivoice"),
        ("omnivoice/arb", "arb", "omnivoice", "omnivoice"),
        ("omnivoice/arz", "arz", "omnivoice", "omnivoice"),
        ("omnivoice/ary", "ary", "omnivoice", "omnivoice"),
        ("omnivoice/zh", "zh", "omnivoice", "omnivoice"),
    ]
)


@pytest.fixture
def catalog(monkeypatch):
    monkeypatch.setattr(vg, "_phoonnx_catalog", lambda: CATALOG)


def test_catalog_voices_filter_by_index_language_and_glob(catalog):
    assert vg.catalog_voices("styletts2", (), "en") == ["kokoro/af_bella", "kokoro/af_heart"]
    assert vg.catalog_voices("styletts2", ("bsc/*",), "ca") == ["bsc/ca-styletts2"]
    assert vg.catalog_voices("supertonic", (), "de") == ["supertonic/F1/de"]
    assert vg.catalog_voices("supertonic", (), "ko") == []


def test_clip_seeds_are_deterministic_per_index_and_differ_across_indexes_phrases_and_runs():
    seeds = [vg.clip_seed(0, "en", "okay nabu", i) for i in range(200)]
    assert seeds == [vg.clip_seed(0, "en", "okay nabu", i) for i in range(200)]
    assert len(set(seeds)) == 200
    assert vg.clip_seed(1, "en", "okay nabu", 0) != seeds[0] and vg.clip_seed(0, "en", "other", 0) != seeds[0]


def test_omnivoice_arabic_request_maps_to_the_arabic_codes(catalog):
    macro = vg.load_roster()["macrolanguages"]
    assert vg.omnivoice_codes("ar-SA", macro) == ["arb", "ary", "arz"]
    assert vg.omnivoice_codes("de", macro) == ["de"]
    assert vg.omnivoice_codes("tlh", macro) == []


def plan_for(phrase="okay nabu", n_codes=("en",), seed=0):
    return lambda i: vg.omnivoice_plan(i, phrase, 1, "en", n_codes, vg.DEFAULT_FORMS, vg.OMNIVOICE_RATES, seed)


def test_omnivoice_plan_gives_each_clip_its_own_speaker_and_rotates_forms_and_two_speeds():
    plan = plan_for()
    cells = [plan(i)[0] for i in range(12)]
    assert len({c.voice for c in cells}) == 12 and len({c.filename for c in cells}) == 12
    assert [c.form for c in cells[:6]] == ["plain", "exclaim", "question"] * 2
    assert [c.rate for c in cells] == [0, 0, 0, 15, 15, 15] * 2
    assert plan(5)[0] == plan_for()(5)[0] and plan(5)[1].seed == int(plan(5)[0].voice.split("#")[1])
    assert plan(0)[1].instruct is None, "the style is drawn at generation time, not planned"


class FakeBackend(vg.OmniBackend):
    """A second of tone; clips with an odd seed are quiet, which the fake aligner scores low."""

    def __init__(self):
        self.batches = []

    def generate(self, items):
        self.batches.append([i.seed for i in items])
        t = np.arange(16000) / 16000
        return [((0.02 if i.seed % 2 else 0.15) * np.sin(2 * np.pi * 200 * t).astype(np.float32), 16000) for i in items]


class FakeAligner:
    """Scores a clip by its peak level and puts the phrase between 0.1 s and 0.7 s."""

    def __init__(self, start=0.1, end=0.7):
        self.start, self.end, self.calls = start, end, 0

    def align(self, wav, phrase):
        self.calls += 1
        return vg.Alignment(float(np.abs(wav).max()) * 10 - 2, self.start, self.end)


def verifier(mode="align", threshold=-1.0, aligner=None, transcribe=None):
    v = vg.Verifier(mode, aligner or FakeAligner(), transcribe)
    v.default = None if threshold is None else (threshold, "manual")
    return v


def sink_for(tmp_path, v=None):
    return vg.ClipSink(tmp_path / "out", v, {"omnivoice"} if v else ())


def test_ctc_alignment_finds_where_the_tokens_are_and_scores_the_path():
    logp = np.full((12, 4), np.log(0.01))
    logp[:, 0] = np.log(0.97)
    logp[3:5, 1], logp[3:5, 0] = np.log(0.9), np.log(0.05)
    logp[7:9, 2], logp[7:9, 0] = np.log(0.9), np.log(0.05)
    score, first, last = vg.ctc_force_align(logp, [1, 2])
    assert first == 3 and last in (7, 8)
    assert -0.5 < score < 0
    assert vg.ctc_force_align(logp[:1], [1, 2]) is None
    worse = logp.copy()
    worse[3:9, 1:3] = np.log(0.01)
    assert vg.ctc_force_align(worse, [1, 2])[0] < score - 1


def test_a_clip_is_trimmed_to_the_phrase_plus_a_margin():
    wav = (0.15 * np.sin(2 * np.pi * 200 * np.arange(16000) / 16000)).astype(np.float32)
    verdict = verifier().check(wav, "okay nabu")
    assert verdict.accepted and len(verdict.audio) == pytest.approx(0.8 * 16000, abs=1)
    late = verifier(aligner=FakeAligner(0.5, 0.9)).check(wav, "okay nabu")
    assert len(late.audio) == pytest.approx(0.6 * 16000, abs=1)


def test_clips_outside_the_duration_window_or_without_voiced_energy_are_rejected():
    wav = (0.15 * np.sin(2 * np.pi * 200 * np.arange(16000) / 16000)).astype(np.float32)
    assert verifier(aligner=FakeAligner(0.5, 0.55)).check(wav, "x").reason == "duration"
    assert verifier(aligner=FakeAligner(0.0, 3.5)).check(np.tile(wav, 4), "x").reason == "duration"
    quiet = np.zeros(16000, np.float32)
    quiet[:100] = 0.5
    assert verifier(aligner=FakeAligner(0.0, 0.8), threshold=-9).check(quiet, "x").reason == "energy"
    assert verifier(threshold=-9).check(wav, "x").accepted


def test_a_clip_scoring_below_the_threshold_is_rejected_with_its_score():
    quiet = (0.02 * np.sin(2 * np.pi * 200 * np.arange(16000) / 16000)).astype(np.float32)
    verdict = verifier().check(quiet, "x")
    assert (verdict.accepted, verdict.reason) == (False, "score") and verdict.score < -1.0
    assert not verifier(threshold=None).check(quiet, "x").accepted


def test_asr_and_both_modes_use_the_transcript_and_alignment_does_not_need_one():
    wav = (0.15 * np.sin(2 * np.pi * 200 * np.arange(16000) / 16000)).astype(np.float32)
    assert verifier("asr", aligner=None, transcribe=lambda p: "Okay, nabu!").check(wav, "okay nabu").accepted
    bad = verifier("asr", transcribe=lambda p: "totally different").check(wav, "okay nabu")
    assert (bad.accepted, bad.reason, bad.transcript) == (False, "asr", "totally different")
    both = verifier("both", transcribe=lambda p: "totally different")
    assert both.check(wav, "okay nabu").reason == "asr"
    spelled = vg.Verifier("asr", None, lambda p: "oh kay nah boo", ["oh kay nah boo"])
    assert spelled.check(wav, "okay nabu").accepted


def write_tone(sink, cell, amp):
    t = np.arange(16000) / 16000
    vg.write_wav(sink.clips / cell.filename, (amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32))


def test_threshold_is_a_percentile_of_the_edge_clips_of_the_same_phrase_only(tmp_path):
    sink = sink_for(tmp_path)
    v = vg.Verifier("align", FakeAligner())
    edge = vg.enumerate_grid([FakeEngine(name="edge")], {"edge": ["v1", "v2"]}, "okay nabu", pitches=(0,), forms=("plain",))
    other = vg.enumerate_grid([FakeEngine(name="kokoro")], {"kokoro": ["v1", "v2"]}, "okay nabu", pitches=(0,), forms=("plain",))
    elsewhere = vg.enumerate_grid([FakeEngine(name="edge")], {"edge": ["v1", "v2"]}, "other phrase", pitches=(0,), forms=("plain",))
    assert len(edge) == 10
    for k, cell in enumerate(edge):
        write_tone(sink, cell, 0.05 + 0.01 * k)
    for cell in other + elsewhere:
        write_tone(sink, cell, 0.01)
    threshold, source, count = vg.reference_threshold(v, sink, "okay nabu", edge + other + elsewhere, 20.0)
    scored = [v.score(sf.read(sink.clips / c.filename, dtype="float32")[0], "x").score for c in edge]
    assert (source, count) == ("edge", 10) and threshold == pytest.approx(np.percentile(scored, 20.0), abs=1e-3)
    assert v.threshold_for("okay nabu") == (threshold, "edge") and len(sink.scores) == 10


def test_without_enough_edge_clips_the_fixed_fallback_threshold_is_used_and_recorded(tmp_path):
    sink = sink_for(tmp_path)
    v = vg.Verifier("align", FakeAligner())
    cells = vg.enumerate_grid([FakeEngine(name="edge")], {"edge": ["v1"]}, "okay nabu", pitches=(0,), forms=("plain",))
    for cell in cells:
        write_tone(sink, cell, 0.1)
    assert vg.reference_threshold(v, sink, "okay nabu", cells, 5.0) == (vg.FALLBACK_THRESHOLD, "fallback", 5)
    assert vg.reference_threshold(v, sink, "okay nabu", [], 5.0)[1:] == ("fallback", 0)
    assert vg.FALLBACK_THRESHOLD == -0.64


def omni_run(tmp_path, monkeypatch, **over):
    monkeypatch.setattr(vg, "MmsAligner", lambda device: FakeAligner())
    monkeypatch.setattr(vg, "omnivoice_codes", lambda lang, macro: ["en"])
    base = dict(verify="align", omnivoice_clips=6, forms=("plain",), rates=(0, 10), pitches=(0, 10, 20), dedup_speaker_threshold=None)
    base.update(over)
    return vg.run_voicegrid(args(tmp_path, **base), [FakeEngine(name="edge", voices=("v1", "v2")), vg.OmniVoiceDonor("omnivoice", {})], FakeBackend())


def test_omnivoice_clips_are_checked_against_a_threshold_from_edge_clips_and_recorded(tmp_path, monkeypatch, capsys):
    assert omni_run(tmp_path, monkeypatch) == 0
    out = tmp_path / "out"
    said = capsys.readouterr().out
    assert "stage=threshold lang=en phrase='hey jarvis' source=edge references=12 percentile=5.0" in said
    rows = manifest(tmp_path)
    omni = [r for r in rows if r["engine"] == "omnivoice"]
    assert len(omni) == 6 and all(r["score"] and r["threshold"] for r in omni)
    assert len({r["threshold"] for r in omni}) == 1 and {r["threshold_source"] for r in omni} == {"edge"}
    assert {r["style"] for r in omni} <= set(vg.OMNIVOICE_STYLES) and all(r["style"] for r in omni)
    assert "accepted=6" in said and "rejected=" in said
    with open(out / "manifest_rejected.csv", newline="") as fh:
        rejected = list(csv.DictReader(fh))
    assert rejected and all(r["reason"] == "score" and float(r["score"]) < float(r["threshold"]) for r in rejected)
    assert all(r["threshold_source"] == "edge" and r["style"] for r in rejected)
    assert len(list((out / "rejected").glob("*.wav"))) == len(rejected)
    for r in omni:
        assert sf.info(out / "clips" / r["file"]).duration == pytest.approx(0.8, abs=0.01), "accepted clips are trimmed"


def test_a_rerun_scores_nothing_again_and_generates_nothing(tmp_path, monkeypatch):
    omni_run(tmp_path, monkeypatch)
    rows = manifest(tmp_path)
    aligner = FakeAligner()
    monkeypatch.setattr(vg, "MmsAligner", lambda device: aligner)
    backend = FakeBackend()
    vg.run_voicegrid(args(tmp_path, verify="align", omnivoice_clips=6, forms=("plain",), rates=(0, 10), pitches=(0, 10, 20),
                          dedup_speaker_threshold=None), [FakeEngine(name="edge", voices=("v1", "v2")), vg.OmniVoiceDonor("omnivoice", {})], backend)
    assert backend.batches == [] and aligner.calls == 0
    assert manifest(tmp_path) == rows


def test_verification_off_accepts_everything(tmp_path, monkeypatch):
    omni_run(tmp_path, monkeypatch, verify="none")
    out = tmp_path / "out"
    assert not (out / "manifest_rejected.csv").exists()
    assert len([r for r in manifest(tmp_path) if r["engine"] == "omnivoice"]) == 6


def test_generation_stops_at_four_times_the_target_when_nothing_passes(tmp_path):
    sink = sink_for(tmp_path, verifier(threshold=5.0))
    result = vg.generate_omnivoice(plan_for(), 5, FakeBackend(), sink, 4)
    assert result["accepted"] == 0 and result["attempted"] == 20 and result["rejected"] == 20
    again = vg.generate_omnivoice(plan_for(), 5, FakeBackend(), sink, 4)
    assert again["attempted"] == 0, "rejected clips count as attempted, so a rerun does not redo them"


def test_generation_continues_until_the_target_is_accepted(tmp_path):
    sink = sink_for(tmp_path, verifier())
    result = vg.generate_omnivoice(plan_for(), 12, FakeBackend(), sink, 4)
    assert result["accepted"] == 12 and result["rejected"] > 0 and result["attempted"] == 12 + result["rejected"]


def test_omnivoice_resume_regenerates_nothing_and_a_missing_clip_keeps_its_seed(tmp_path):
    backend = FakeBackend()
    sink = sink_for(tmp_path)
    vg.generate_omnivoice(plan_for(), 6, backend, sink, 4)
    assert sum(len(b) for b in backend.batches) == 6
    backend.batches.clear()
    vg.generate_omnivoice(plan_for(), 6, backend, sink, 4)
    assert backend.batches == []
    gone = plan_for()(2)
    (sink.clips / gone[0].filename).unlink()
    vg.generate_omnivoice(plan_for(), 6, backend, sink, 4)
    assert backend.batches == [[gone[1].seed]]
    assert (sink.clips / gone[0].filename).exists()


def test_transcripts_are_normalised_before_matching():
    assert vg.transcript_match("Okay,  NABU!", "okay nabu") == 1.0
    assert vg.transcript_match("Ókáy nabú", "okay nabu") == 1.0
    assert vg.transcript_match("hello world", "okay nabu") < 0.5


def test_omnivoice_rows_carry_the_per_clip_seed_in_the_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(vg, "omnivoice_codes", lambda lang, macro: ["en"])
    vg.run_voicegrid(args(tmp_path, omnivoice_clips=4, dedup_speaker_threshold=None),
                     [vg.OmniVoiceDonor("omnivoice", {})], FakeBackend())
    rows = manifest(tmp_path)
    assert len(rows) == 4 and len({r["voice"] for r in rows}) == 4 and all(r["voice"].startswith("omnivoice/en#") for r in rows)
    assert {r["engine"] for r in rows} == {"omnivoice"}


def test_backend_is_the_official_package_when_importable_else_phoonnx(monkeypatch):
    made = []
    monkeypatch.setattr(vg, "TorchOmniBackend", lambda device: made.append(("torch", device)) or "t")
    monkeypatch.setattr(vg, "PhoonnxOmniBackend", lambda workers=1: made.append(("phoonnx", workers)) or "p")
    monkeypatch.setattr(vg, "_omnivoice_package", lambda: True)
    assert vg.omnivoice_backend("rocm", 3) == "t"
    monkeypatch.setattr(vg, "_omnivoice_package", lambda: False)
    assert vg.omnivoice_backend("rocm", 3) == "p"
    assert made == [("torch", "rocm"), ("phoonnx", 3)]


def test_package_detection_follows_importability(monkeypatch):
    monkeypatch.setattr(vg.importlib.util, "find_spec", lambda name: object() if name in ("omnivoice", "torch") else None)
    assert vg._omnivoice_package()
    monkeypatch.setattr(vg.importlib.util, "find_spec", lambda name: None if name == "omnivoice" else object())
    assert not vg._omnivoice_package()


def test_phrases_file_gives_every_phrase_the_grid_and_its_own_label(tmp_path):
    phrases = tmp_path / "phrases.txt"
    phrases.write_text("okay nabu\nokay nabi\t0\n\nokay mabu\t0\n", encoding="utf-8")
    vg.run_voicegrid(args(tmp_path, wake_word=None, phrases_file=phrases, forms=("plain",), rates=(0,), pitches=(0,)),
                     [FakeEngine(voices=("v1",))])
    rows = manifest(tmp_path)
    assert {(r["text"], r["label"]) for r in rows} == {("okay nabu", "1"), ("okay nabi", "0"), ("okay mabu", "0")}
    assert len({r["file"] for r in rows}) == 3
    labels = sorted(l.rsplit(",", 1)[1] for l in (tmp_path / "out" / "metadata.csv").read_text().splitlines())
    assert labels == ["0", "0", "1"]


def test_wake_word_and_phrases_file_combine_and_one_of_them_is_required(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("b phrase\n")
    assert vg.load_phrases(args(tmp_path, wake_word="a_phrase", phrases_file=f)) == [("a phrase", 1), ("b phrase", 1)]
    with pytest.raises(SystemExit) as e:
        vg.cli_main(["--output-dir", str(tmp_path)])
    assert e.value.code == 2


def test_a_language_without_edge_clips_gets_the_fallback_threshold(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(vg, "MmsAligner", lambda device: FakeAligner())
    monkeypatch.setattr(vg, "omnivoice_codes", lambda lang, macro: ["en"])
    vg.run_voicegrid(args(tmp_path, verify="align", omnivoice_clips=4, dedup_speaker_threshold=None),
                     [vg.OmniVoiceDonor("omnivoice", {})], FakeBackend())
    assert "source=fallback references=0" in capsys.readouterr().out
    assert {r["threshold_source"] for r in manifest(tmp_path)} == {"fallback"}
    assert {r["threshold"] for r in manifest(tmp_path)} == {"-0.6400"}


def test_required_asr_languages_need_both_the_alignment_and_the_transcript(tmp_path, monkeypatch):
    modes = []
    real = vg.Verifier
    monkeypatch.setattr(vg, "Verifier", lambda mode, *a, **k: modes.append(mode) or real(mode, *a, **k))
    monkeypatch.setattr(vg, "_load_asr", lambda model: (lambda path: "nothing like it"))
    omni_run(tmp_path, monkeypatch, lang="da", require_asr_langs="da, it,fr", output_dir=tmp_path / "da")
    assert modes == ["both"]
    assert [r for r in csv.DictReader(open(tmp_path / "da" / "manifest.csv")) if r["engine"] == "omnivoice"] == []
    reasons = {r["reason"] for r in csv.DictReader(open(tmp_path / "da" / "manifest_rejected.csv"))}
    assert reasons == {"score", "asr"}
    modes.clear()
    omni_run(tmp_path, monkeypatch, lang="en", require_asr_langs="da,it,fr", output_dir=tmp_path / "en")
    assert modes == ["align"]
    assert len([r for r in csv.DictReader(open(tmp_path / "en" / "manifest.csv")) if r["engine"] == "omnivoice"]) == 6


def test_style_weights_follow_accept_rates_and_never_drop_below_the_floor():
    styles = ["a", "b", "c", "d"]
    even = vg.style_weights(styles, {})
    assert even == pytest.approx([0.25] * 4)
    skewed = vg.style_weights(styles, {"a": (96, 100), "b": (54, 100), "c": (0, 100), "d": (0, 100)})
    assert sum(skewed) == pytest.approx(1.0) and min(skewed) >= vg.STYLE_FLOOR - 1e-9
    assert skewed[0] > skewed[1] > skewed[2] and skewed[2] == pytest.approx(skewed[3])
    assert skewed[2] == pytest.approx(vg.STYLE_FLOOR, abs=0.01)
    with pytest.raises(ValueError):
        vg.style_weights([str(i) for i in range(11)], {})


def test_style_draw_is_a_function_of_the_seed_and_the_weights():
    styles, weights = ["a", "b", "c"], [0.1, 0.1, 0.8]
    draws = [vg.pick_style(styles, weights, s) for s in range(2000)]
    assert draws == [vg.pick_style(styles, weights, s) for s in range(2000)]
    assert 0.05 < draws.count("a") / 2000 < 0.15 and draws.count("c") / 2000 > 0.7


class StyleBackend(FakeBackend):
    """Clips asked for in the style 'bad' are quiet, so the aligner scores them below the threshold."""

    def __init__(self):
        super().__init__()
        self.styles = []

    def generate(self, items):
        self.styles.append([i.instruct for i in items])
        t = np.arange(16000) / 16000
        return [((0.02 if i.instruct == "bad" else 0.15) * np.sin(2 * np.pi * 200 * t).astype(np.float32), 16000) for i in items]


def test_style_weights_are_recomputed_every_batch_and_persist_across_runs(tmp_path, monkeypatch):
    seen, real = [], vg.style_weights
    monkeypatch.setattr(vg, "style_weights", lambda styles, counts, *a: seen.append(dict(counts)) or real(styles, counts, *a))
    sink = sink_for(tmp_path, verifier())
    backend = StyleBackend()
    result = vg.generate_omnivoice(plan_for(), 60, backend, sink, 5, ["bad", "good", "auto"])
    assert result["accepted"] == 60
    assert len(seen) == len(backend.styles) and seen[0] == {} and seen[-1]["bad"][1] > seen[1]["bad"][1] > 0
    flat = [s for batch in backend.styles for s in batch]
    first, last = flat[:15], flat[-30:]
    assert last.count("bad") / len(last) < first.count("bad") / len(first)
    counts = sink.style_counts()
    assert counts["bad"][0] == 0 and counts["good"][0] == counts["good"][1]
    weights = vg.style_weights(["bad", "good", "auto"], counts)
    assert weights[0] == min(weights) and weights[0] >= vg.STYLE_FLOOR - 1e-9
    assert flat.count("bad") > 0, "the floor keeps every style in play"
    again = sink_for(tmp_path, verifier())
    assert again.style_counts() == counts


def test_torch_backend_groups_a_mixed_batch_by_style_and_keeps_the_order(monkeypatch):
    calls = []

    class Model:
        def generate(self, text, language, speed, instruct=None):
            calls.append((list(text), instruct))
            return [np.full(10, float(len(t))) for t in text]

    backend = object.__new__(vg.TorchOmniBackend)
    backend.model = Model()
    backend._torch = types.SimpleNamespace(manual_seed=lambda seed: calls.append(("seed", seed)))
    items = [vg.OmniItem("aa", "en", 0, 1, "elderly male"), vg.OmniItem("b", "en", 0, 2), vg.OmniItem("cccc", "en", 0, 3, "elderly male")]
    out = backend.generate(items)
    assert [len(a) for a, _ in out] == [10, 10, 10] and [a[0] for a, _ in out] == [2.0, 1.0, 4.0]
    assert calls == [("seed", 1), (["aa", "cccc"], ["elderly male"] * 2), ("seed", 2), (["b"], None)]


def test_cli_options_for_the_omnivoice_backend_and_verification():
    parse = vg.build_parser().parse_args
    a = parse(["--wake-word", "x", "--output-dir", "o"])
    assert (a.omnivoice_clips, a.batch_size, a.device, a.verify, a.verify_percentile, a.omnivoice_style, a.require_asr_langs) == (
        300, 10, "auto", "align", 5.0, [], "")
    b = parse(["--wake-word", "x", "--output-dir", "o", "--device", "rocm", "--verify", "both", "--verify-percentile", "10",
               "--omnivoice-clips", "50", "--accept-spellings", "a,b", "--omnivoice-style", "female, british accent",
               "--omnivoice-style", "auto", "--require-asr-langs", "da,it"])
    assert (b.device, b.verify, b.verify_percentile, b.omnivoice_clips, b.accept_spellings) == ("rocm", "both", 10.0, 50, ["a,b"])
    assert b.omnivoice_style == ["female, british accent", "auto"] and b.require_asr_langs == "da,it"
    with pytest.raises(SystemExit):
        vg.cli_main(["--wake-word", "x", "--output-dir", "o", "--verify-all-donors"])


def test_packaged_roster_gives_every_language_omnivoice_and_edge_and_leaves_old_engines_out():
    roster = vg.load_roster()
    old = {"mms", "mimic3", "glowtts", "coqui_vits", "coqui_community"}
    for lang, names in roster["languages"].items():
        assert names[0] == "omnivoice" and "edge" in names, lang
        assert set(names) <= set(roster["donors"]), lang
        assert not {roster["donors"][n].get("index") for n in names} & old, lang
    assert {l for l, n in roster["languages"].items() if "piper" in n} == {"de", "fr", "nl", "pl", "ru", "uk", "fa", "he", "ms", "no", "sw", "th"}
    assert vg.donors_for(roster, "pt-BR") == roster["languages"]["pt"]
    assert vg.donors_for(roster, "tlh") == roster["fallback"]


def test_roster_file_is_a_packaged_data_file():
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    assert 'ww_trainer = ["data/*.json"]' in pyproject
    assert 'ww_trainer-voicegrid = "ww_trainer.voicegrid:cli_main"' in pyproject
    assert vg.ROSTER_PATH.is_file()


def test_donors_option_names_donors_or_points_at_a_roster_file(tmp_path, monkeypatch):
    built = []
    monkeypatch.setattr(vg, "make_donor", lambda name, roster, lang, workers, cfg: built.append((name, sorted(roster["donors"]))) or FakeEngine(name=name))
    vg.build_donors(args(tmp_path, donors="edge,styletts2"))
    assert [b[0] for b in built] == ["edge", "styletts2"]
    built.clear()
    custom = tmp_path / "roster.json"
    custom.write_text('{"donors": {"mine": {"kind": "edge"}}, "languages": {"en": ["mine"]}, "fallback": ["mine"]}')
    vg.build_donors(args(tmp_path, donors=str(custom)))
    assert built == [("mine", ["mine"])]
    built.clear()
    vg.build_donors(args(tmp_path, lang="pt-BR"))
    assert [b[0] for b in built] == vg.load_roster()["languages"]["pt"]


def test_missing_roster_donor_package_is_skipped_but_a_named_one_raises(tmp_path, monkeypatch):
    def make(name, roster, lang, workers, cfg):
        if name == "google":
            raise KeyError("ovos-tts-plugin-google-tx")
        return FakeEngine(name=name)

    monkeypatch.setattr(vg, "make_donor", make)
    assert "google" not in [d.name for d in vg.build_donors(args(tmp_path))]
    with pytest.raises(KeyError):
        vg.build_donors(args(tmp_path, donors="google"))


def test_deduplication_is_on_by_default_and_can_be_turned_off():
    parser = vg.build_parser()
    base = ["--wake-word", "x", "--output-dir", "o"]
    assert parser.parse_args(base).dedup_speaker_threshold == vg.DEFAULT_DEDUP_THRESHOLD
    assert parser.parse_args(base + ["--no-dedup"]).dedup_speaker_threshold is None
    assert parser.parse_args(base + ["--dedup-speaker-threshold", "0.7"]).dedup_speaker_threshold == 0.7
    assert parser.parse_args(base).vc_refs is None, "cloning is an opt-in stage"


def test_help_renders_and_the_help_exit_path_works(capsys):
    text = vg.build_parser().format_help()
    assert "at least 10% each" in text and "--omnivoice-style" in text
    with pytest.raises(SystemExit) as done:
        vg.cli_main(["--help"])
    assert done.value.code == 0 and "ww_trainer-voicegrid" in capsys.readouterr().out


def test_held_out_edge_voices_and_their_twins_are_excluded_by_default():
    held = ["en-GB-MaisieNeural", "en-IE-ConnorNeural", "en-IN-PrabhatNeural", "en-KE-AsiliaNeural", "en-NZ-MollyNeural",
            "en-SG-WayneNeural"]
    twins = [h.replace("Neural", "MultilingualNeural") for h in held]
    kept = ["en-GB-RyanNeural", "en-US-AriaNeural", "en-IE-EmilyNeural", "en-US-AvaNeural"]
    engine = FakeEngine(name="edge", voices=held + twins + kept)
    assert vg.select_voices(engine, "en", vg.DEFAULT_EXCLUDE_VOICES, None, 0) == kept
    assert vg.select_voices(engine, "en", (), None, 0) == held + twins + kept


def test_a_failure_hidden_behind_a_library_message_names_the_missing_module(tmp_path, caplog):
    class Kokoro(FakeEngine):
        def synth(self, *a):
            try:
                raise ModuleNotFoundError("No module named 'num2words'", name="num2words")
            except ImportError as inner:
                raise ImportError("misaki is required for the Misaki phonemizer") from inner

    with caplog.at_level("WARNING"):
        failed = vg.run_voicegrid(args(tmp_path, forms=("plain",), rates=(0,), pitches=(0,)), [Kokoro(voices=("v1",))])
    assert failed == 1
    assert "misaki is required" in caplog.text and "(module num2words)" in caplog.text


def test_voicegrid_extra_installs_what_kokoro_imports():
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    extra = pyproject.split("voicegrid = [")[1].split("\n]")[0]
    assert '"num2words"' in extra and '"scriptconv[misaki]"' in extra


def test_hub_cache_symlinks_become_real_entries_without_copying_the_bytes(tmp_path):
    blobs = tmp_path / "blobs"
    snap = tmp_path / "snapshots" / "rev"
    blobs.mkdir()
    snap.mkdir(parents=True)
    (blobs / "aaa").write_bytes(b"graph")
    (blobs / "bbb").write_bytes(b"weights")
    (blobs / "ccc").write_text("{}")
    (snap / "model.onnx").symlink_to(blobs / "aaa")
    (snap / "model.onnx_data").symlink_to(blobs / "bbb")
    (snap / "config.json").symlink_to(blobs / "ccc")
    vg.realize_links(snap / "model.onnx")
    for name, data in (("model.onnx", b"graph"), ("model.onnx_data", b"weights")):
        assert not (snap / name).is_symlink() and (snap / name).read_bytes() == data
        assert (snap / name).resolve().parent == snap
    assert (snap / "config.json").is_symlink(), "only the graph and its sidecar are touched"
    assert (blobs / "aaa").read_bytes() == b"graph" and (blobs / "aaa").stat().st_nlink == 2
    vg.realize_links(snap / "model.onnx")
    assert (snap / "model.onnx").read_bytes() == b"graph"


def test_metadata_csv_trains_from_any_working_directory(tmp_path, monkeypatch):
    from ww_trainer.cli import _load_rows

    vg.run_voicegrid(args(tmp_path, forms=("plain",), rates=(0,), pitches=(0, 10)), [FakeEngine(voices=("v1",))])
    csv_path = str(tmp_path / "out" / "metadata.csv")
    for cwd in (tmp_path, tmp_path / "out", tmp_path / "out" / "clips"):
        monkeypatch.chdir(cwd)
        rows = _load_rows(csv_path, "--metadata", False)
        assert len(rows) == 2 and all(Path(p).is_file() and label == "1" for p, label in rows)
    monkeypatch.chdir(tmp_path.parent)
    assert len(_load_rows(os.path.relpath(csv_path), "--metadata", False)) == 2
