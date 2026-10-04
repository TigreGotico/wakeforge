"""scripts/research/wakephonehubert/train/train_pool.py: the pool readers and batch builders, on a synthetic pool.

The pool holds one MSWC-type source (1 s words with CTC targets) and one FMA-type source (10 s segments). The teacher
arrays carry their own row and frame index, so a batch shows which rows and frames it was cut from.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_DIR = Path(__file__).resolve().parent.parent / "scripts" / "research" / "wakephonehubert"
sys.path[:0] = [str(_DIR), str(_DIR / "pool")]
_S = importlib.util.spec_from_file_location("train_pool", _DIR / "train" / "train_pool.py")
tp = importlib.util.module_from_spec(_S)
sys.modules["train_pool"] = tp
_S.loader.exec_module(tp)

N_WORDS, N_MUSIC, K = 10, 6, 8


def write_source(d, name, kind, n, seg, forced_value):
    frames, t = seg // 320, seg // 320 - 1
    rng = np.random.default_rng(0)
    (d / f"{name}_audio.raw").write_bytes((rng.standard_normal((n, seg)) * 3000).astype(np.int16).tobytes())
    (d / f"{name}.done").write_text(str(n))
    np.save(d / f"{name}_silero.npy", rng.random((n, frames)).astype(np.float16))
    idx = np.broadcast_to(np.arange(t, dtype=np.int16)[None, :, None], (n, t, K)).copy()
    np.save(d / f"{name}_tk_idx.npy", idx)
    np.save(d / f"{name}_tk_p.npy", np.full((n, t, K), 1 / K, np.float16))
    np.save(d / f"{name}_forced.npy", np.full((n, frames), forced_value, bool))
    with open(d / f"{name}_meta.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"{name}{i}", "lang": "en", "word": "w"} if kind == "mswc" else {"id": f"{name}{i}", "track": i}) + "\n")
    prog = {"n": n, "chunk": 4096, "type": kind, "silero": 1, "ipa": 1, "voice": 1, "forced": 1}
    if kind == "mswc":
        prog["ids"] = 1
        with open(d / f"{name}_ipa_ids.jsonl", "w") as f:
            for i in range(n):
                f.write(json.dumps({"voice": "en-us", "ids": [4 + i % 5, 9] if i % 4 else []}) + "\n")
    (d / f"{name}_progress.json").write_text(json.dumps(prog))


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    d = tmp_path_factory.mktemp("pool")
    write_source(d, "words", "mswc", N_WORDS, 16000, False)
    write_source(d, "music", "fma", N_MUSIC, 160000, True)
    return {k: tp.Src(d, k, 0, 0.2) for k in ("words", "music")}


def test_sources_load_with_types_and_held_out_splits(sources):
    w, m = sources["words"], sources["music"]
    assert (w.type, m.type) == ("mswc", "fma")
    assert (w.n, m.n) == (N_WORDS, N_MUSIC)
    assert len(w.eval) == 2 and not set(w.train) & set(w.eval)
    assert list(m.eval) == [N_MUSIC - 1] and m.all_forced.all()
    assert len(w.labels) == N_WORDS


def test_a_source_without_a_finished_voice_stage_is_refused(tmp_path):
    write_source(tmp_path, "music", "fma", N_MUSIC, 160000, True)
    prog = json.loads((tmp_path / "music_progress.json").read_text())
    prog["voice"] = 0
    (tmp_path / "music_progress.json").write_text(json.dumps(prog))
    with pytest.raises(SystemExit, match="teacher stages incomplete"):
        tp.Src(tmp_path, "music", 0, 0.2)


def test_batch_a_is_noisy_words_with_their_teachers(sources):
    b = tp.batch_A(np.random.default_rng(0), sources, 6)
    assert b["kind"] == "A"
    assert b["wav"].shape == (6, 16000) and b["wav"].dtype == np.float32
    assert b["vad_y"].shape == (6, 50) and (b["vad_w"] == 1).all()
    assert b["tki"].shape == (6, 49, K) and b["tkp"].shape == (6, 49, K)
    assert len(b["ctc"]) == 6 and list(b["ctc_use"]) == [len(c) > 0 for c in b["ctc"]]
    assert set(b) == {"kind", "wav", "vad_y", "vad_w", "tki", "tkp", "ctc", "ctc_use"}


def test_batch_b_crops_start_on_a_320_ms_boundary(sources):
    b = tp.batch_B(np.random.default_rng(0), sources, 12)
    assert b["wav"].shape == (12, 150 * 320)
    assert b["vad_w"].shape == (12, 150) and (b["vad_y"] == 0).all()
    assert b["tki"].shape == (12, 150, K)
    first = b["tki"][:, 0, 0]
    assert (first % 16 == 0).all() and first.max() <= 500 - 150
    assert (b["tki"][:, :, 0] - first[:, None] == np.arange(150)).all()
    assert b["ctc_use"].all() and all(c == [] for c in b["ctc"])


def test_final_round_batch_mixes_words_with_non_speech_crops(sources):
    wav, targets = tp.batch_final(np.random.default_rng(0), {"words": sources["words"]}, sources, 20, 0.15)
    assert wav.shape == (20, 16000) and len(targets) == 20
    assert sum(1 for t in targets if t == []) >= 3
    assert np.abs(wav).max() <= 1.0


def test_edit_distance_counts_errors_against_the_reference():
    assert tp.per([1, 2, 3], [1, 3]) == (1, 3)
    assert tp.per([1, 2, 3], [4, 5, 6, 7]) == (4, 3)
    assert tp.per([], []) == (0, 0)
