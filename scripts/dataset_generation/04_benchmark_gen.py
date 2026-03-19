import os
import random
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
import click

def load_audio_mono(path, sr=16000):
    """Load an audio file, convert to mono, and resample to target sr."""
    try:
        wav, orig_sr = sf.read(str(path))
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if orig_sr != sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=orig_sr, target_sr=sr)
    return wav.astype(np.float32)

def save_wav(path, wav, sr=16000):
    """Save a numpy array as a WAV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr)

def mix_audio(clean, bg, snr_db):
    """Mix a background track into clean audio at a given SNR (dB)."""
    clean_len = len(clean)
    if len(bg) < clean_len:
        bg = np.tile(bg, int(np.ceil(clean_len / len(bg))))
    start = random.randint(0, len(bg) - clean_len)
    bg = bg[start:start + clean_len]

    rms_clean = np.sqrt(np.mean(clean ** 2) + 1e-9)
    rms_bg    = np.sqrt(np.mean(bg ** 2) + 1e-9)
    desired   = rms_clean / (10 ** (snr_db / 20.0))
    if rms_bg > 0:
        bg = bg * (desired / rms_bg)

    mixed = clean + bg
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)

def apply_reverb(wav, rir, attenuation=0.5):
    """Convolve with a Room Impulse Response."""
    out = np.convolve(wav, rir)[:len(wav)]
    rms_wav = np.sqrt(np.mean(wav ** 2) + 1e-9)
    rms_out = np.sqrt(np.mean(out ** 2) + 1e-9)
    out = out * (rms_wav / rms_out) * attenuation
    return out.astype(np.float32)

def collect_audio_files(base_folder):
    """Recursively collect audio files."""
    exts = [".wav", ".flac", ".mp3", ".m4a", ".ogg"]
    if not base_folder:
        return []
    p = Path(base_folder)
    if not p.exists():
        return []
    files = []
    for ext in exts:
        files.extend(p.rglob(f"*{ext}"))
    return sorted(files)

def _bench_mix(clean_files, bg_files, snr_list, output_dir, noise_type):
    if not bg_files: return
    for clean_path in tqdm(clean_files, desc=f"  {noise_type}"):
        clean_wav = load_audio_mono(clean_path)
        if clean_wav is None: continue
        bg_path = random.choice(bg_files)
        bg_wav = load_audio_mono(bg_path)
        if bg_wav is None: continue
        stem = clean_path.stem
        for snr in snr_list:
            mixed = mix_audio(clean_wav, bg_wav, snr)
            out_name = f"{stem}_{noise_type}_{int(snr)}db.wav"
            save_wav(output_dir / noise_type.lower() / out_name, mixed)

def _bench_variance(clean_files, rir_files, output_dir):
    PITCH_STEPS = [-3, -2, -1, 1, 2, 3]
    SPEED_FACTORS = [0.9, 1.1]
    for clean_path in tqdm(clean_files, desc="  Variance"):
        clean_wav = load_audio_mono(clean_path)
        if clean_wav is None: continue
        stem = clean_path.stem
        for step in PITCH_STEPS:
            out = librosa.effects.pitch_shift(clean_wav, sr=16000, n_steps=step)
            save_wav(output_dir / "variance" / f"{stem}_pitch_{step}.wav", out)
        for factor in SPEED_FACTORS:
            out = librosa.effects.time_stretch(clean_wav, rate=factor)
            save_wav(output_dir / "variance" / f"{stem}_speed_{factor:.1f}.wav", out)
        if rir_files:
            selected = random.sample(rir_files, k=min(3, len(rir_files)))
            for rir_path in selected:
                rir_wav = load_audio_mono(rir_path)
                if rir_wav is None: continue
                out = apply_reverb(clean_wav, rir_wav)
                rir_id = rir_path.stem[:8]
                save_wav(output_dir / "variance" / f"{stem}_reverb_{rir_id}.wav", out)

@click.command()
@click.option("--clean-dir", required=True, help="Clean wake-word samples")
@click.option("--out-dir", required=True, help="Output directory")
@click.option("--noise-dir", default="", help="Ambient noise dir")
@click.option("--music-dir", default="", help="Music noise dir")
@click.option("--speech-dir", default="", help="Competing-speaker dir")
@click.option("--rir-dir", default="", help="RIR files dir")
@click.option("--snrs", default="20,15,10,5,0", help="Comma-separated SNR list (dB)")
def main(clean_dir, out_dir, noise_dir, music_dir, speech_dir, rir_dir, snrs):
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clean_files = collect_audio_files(clean_dir)
    noise_files = collect_audio_files(noise_dir)
    music_files = collect_audio_files(music_dir)
    speech_files = collect_audio_files(speech_dir)
    rir_files = collect_audio_files(rir_dir)

    snr_list = [float(s.strip()) for s in snrs.split(",") if s.strip()]

    print(f"Found {len(clean_files)} clean files.")
    _bench_mix(clean_files, noise_files,  snr_list, output_path, "Noise")
    _bench_mix(clean_files, music_files,  snr_list, output_path, "Music")
    _bench_mix(clean_files, speech_files, snr_list, output_path, "Speech")
    _bench_variance(clean_files, rir_files, output_path)

if __name__ == "__main__":
    main()
