"""Augmentation reads each noise, speech and room-response clip once, and reverb is computed by FFT.

A training run draws the same few hundred augmentation clips thousands of times; each draw re-read and
re-resampled the file. Reverb was a direct convolution, whose cost grows with clip length times impulse length.
"""
import numpy as np
import soundfile as sf

from ww_trainer import augment


def test_reverb_matches_direct_convolution(tmp_path):
    rng = np.random.default_rng(0)
    wav = rng.standard_normal(24000).astype(np.float32)
    rir = rng.standard_normal(8000).astype(np.float32) * np.exp(-np.arange(8000) / 800).astype(np.float32)
    direct = np.convolve(wav, rir)[:len(wav)]
    direct = direct * (np.sqrt(np.mean(wav ** 2) + 1e-9) / np.sqrt(np.mean(direct ** 2) + 1e-9)) * 0.5
    np.testing.assert_allclose(augment.apply_reverb(wav, rir), direct, rtol=0, atol=1e-4)


def test_a_clip_is_read_from_disk_once(tmp_path, monkeypatch):
    p = tmp_path / "noise.wav"
    sf.write(p, np.linspace(-0.5, 0.5, 1600, dtype=np.float32), 16000)
    reads = []
    real = sf.read
    monkeypatch.setattr(sf, "read", lambda *a, **k: reads.append(a[0]) or real(*a, **k))
    first = augment._load_audio_mono(p, 16000)
    second = augment._load_audio_mono(p, 16000)
    assert len(reads) == 1, f"read {len(reads)} times"
    np.testing.assert_array_equal(first, second)


def test_a_caller_changing_its_copy_does_not_change_the_next_draw(tmp_path):
    p = tmp_path / "noise.wav"
    sf.write(p, np.full(800, 0.25, dtype=np.float32), 16000)
    a = augment._load_audio_mono(p, 16000)
    a *= 4.0
    b = augment._load_audio_mono(p, 16000)
    np.testing.assert_allclose(b, 0.25, atol=1e-4)


def test_resampled_and_native_rates_are_cached_apart(tmp_path):
    p = tmp_path / "noise.wav"
    sf.write(p, np.zeros(1600, dtype=np.float32), 16000)
    assert len(augment._load_audio_mono(p, 16000)) == 1600
    assert len(augment._load_audio_mono(p, 8000)) == 800


def test_the_cache_is_bounded_by_bytes_not_by_clip_count(tmp_path, monkeypatch):
    monkeypatch.setattr(augment, "CLIP_CACHE_BYTES", 3 * 16000 * 4)  # three one-second clips
    for i in range(10):
        p = tmp_path / f"long{i}.wav"
        sf.write(p, np.zeros(16000, dtype=np.float32), 16000)
        augment._load_audio_mono(p, 16000)
    assert augment._clip_bytes <= augment.CLIP_CACHE_BYTES
    assert sum(w.nbytes for w in augment._clips.values()) == augment._clip_bytes


def test_a_clip_larger_than_the_budget_is_returned_but_not_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(augment, "CLIP_CACHE_BYTES", 1000)
    p = tmp_path / "big.wav"
    sf.write(p, np.zeros(16000, dtype=np.float32), 16000)
    assert len(augment._load_audio_mono(p, 16000)) == 16000
    assert (str(p), 16000) not in augment._clips
