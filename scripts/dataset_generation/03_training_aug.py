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

@click.command()
@click.option("--input-dir", required=True, help="Preprocessed dataset dir (with metadata.csv)")
@click.option("--out-dir", required=True, help="Output directory")
@click.option("--bg-noise-dir", default="", help="Background noise dir")
@click.option("--music-dir", default="", help="Music folder")
@click.option("--speech-dir", default="", help="Competing-speaker dir")
@click.option("--mic-noise-dir", default="", help="Mic noise dir")
@click.option("--rir-dir", default="", help="RIR folder")
@click.option("--snr-min", default=0.0, help="Min SNR (dB)")
@click.option("--snr-max", default=20.0, help="Max SNR (dB)")
@click.option("--pitch-min", default=-1.0, help="Min pitch shift")
@click.option("--pitch-max", default=1.0, help="Max pitch shift")
@click.option("--speed-min", default=0.95, help="Min speed factor")
@click.option("--speed-max", default=1.05, help="Max speed factor")
@click.option("--prob", default=0.9, help="Augmentation probability")
def main(input_dir, out_dir, bg_noise_dir, music_dir, speech_dir, mic_noise_dir, rir_dir,
         snr_min, snr_max, pitch_min, pitch_max, speed_min, speed_max, prob):
    
    input_path = Path(input_dir)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_in = input_path / "metadata.csv"
    metadata_out = output_path / "metadata.csv"

    if not metadata_in.exists():
        raise FileNotFoundError(f"metadata.csv not found in {input_dir}")

    bg_files = collect_audio_files(bg_noise_dir)
    music_files = collect_audio_files(music_dir)
    speech_files = collect_audio_files(speech_dir)
    mic_files = collect_audio_files(mic_noise_dir)
    rir_files = collect_audio_files(rir_dir)

    with open(metadata_in, "r", encoding="utf-8") as f:
        source_data = [line.strip().split(",") for line in f]

    augmented = []
    for path_str, label in tqdm(source_data, desc="Augmenting"):
        src_path = Path(path_str)
        if not src_path.is_absolute():
            src_path = input_path.parent / src_path

        wav = load_audio_mono(src_path)
        if wav is None: continue

        was_augmented = False
        if random.random() < prob:
            # Random augmentations
            if bg_files and random.random() < 0.5:
                bg = load_audio_mono(random.choice(bg_files))
                if bg is not None:
                    wav = mix_audio(wav, bg, random.uniform(snr_min, snr_max))
                    was_augmented = True
            
            if mic_files and random.random() < 0.8:
                mic = load_audio_mono(random.choice(mic_files))
                if mic is not None:
                    wav = mix_audio(wav, mic, random.uniform(snr_min, snr_max))
                    was_augmented = True

            if music_files and random.random() < 0.5:
                music = load_audio_mono(random.choice(music_files))
                if music is not None:
                    wav = mix_audio(wav, music, random.uniform(0.0, 10.0))
                    was_augmented = True

            if rir_files and random.random() < 0.3:
                rir = load_audio_mono(random.choice(rir_files))
                if rir is not None:
                    wav = apply_reverb(wav, rir)
                    was_augmented = True

            if random.random() < 0.3:
                n_steps = random.uniform(pitch_min, pitch_max)
                wav = librosa.effects.pitch_shift(wav, sr=16000, n_steps=n_steps)
                was_augmented = True

            if random.random() < 0.3:
                rate = random.uniform(speed_min, speed_max)
                wav = librosa.effects.time_stretch(wav, rate=rate)
                was_augmented = True

        wav = wav / (np.max(np.abs(wav)) + 1e-9)
        rel_dir = "wakes_aug" if int(label) == 1 else "negatives_aug"
        dst_path = output_path / rel_dir / (src_path.stem + "_aug.wav")
        save_wav(dst_path, wav)
        augmented.append((str(dst_path), label))

    with open(metadata_out, "w", encoding="utf-8") as f:
        for p, lbl in augmented:
            f.write(f"{p},{lbl}\n")

if __name__ == "__main__":
    main()
