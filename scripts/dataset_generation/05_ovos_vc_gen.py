import os
import random
import tempfile
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_dataset
import click

TARGET_SR = 16000

def load_audio(path, sr=TARGET_SR, duration=None):
    try:
        audio, orig_sr = sf.read(str(path), dtype='float32')
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
        if duration:
            max_samples = int(duration * sr)
            audio = audio[:max_samples]
        if audio.max() > 0:
            audio = audio / np.abs(audio).max()
        return audio
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def load_audio_from_array(audio, sr_orig, sr=TARGET_SR, duration=None):
    try:
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr_orig != sr:
            audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=sr)
        if duration:
            max_samples = int(duration * sr)
            audio = audio[:max_samples]
        if audio.max() > 0:
            audio = audio / np.abs(audio).max()
        return audio
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def save_audio(path, audio, sr=TARGET_SR):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)

def apply_reverb(audio, ir, sr=TARGET_SR):
    if ir is None or len(ir) == 0:
        return audio
    ir = ir / np.abs(ir).max() if ir.max() > 0 else ir
    reverb = signal.fftconvolve(audio, ir, mode='same')
    if reverb.max() > 0:
        reverb = reverb / np.abs(reverb).max()
    return reverb.astype(np.float32)

def mix_audio_at_snr(clean, noise, snr_db):
    if noise is None or len(noise) == 0:
        return clean
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)[:len(clean)]
    else:
        start = random.randint(0, len(noise) - len(clean))
        noise = noise[start:start + len(clean)]
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return clean
    snr_linear = 10 ** (snr_db / 10)
    scale = np.sqrt(clean_power / (snr_linear * noise_power))
    mixed = clean + scale * noise
    if np.abs(mixed).max() > 1.0:
        mixed = mixed / np.abs(mixed).max() * 0.95
    return mixed.astype(np.float32)

def create_tts_plugin(engine, voice=None, lang="en"):
    if engine == "edge":
        from ovos_tts_plugin_edge_tts import EdgeTTSPlugin
        return EdgeTTSPlugin(config={"voice": voice})
    elif engine == "google":
        from ovos_tts_plugin_google_tx import GoogleTranslateTTS
        return GoogleTranslateTTS(config={"lang": lang})
    elif engine == "phoonnx":
        from phoonnx.opm import PhoonnxTTSPlugin
        cfg = {"lang": lang}
        if voice and voice != "default":
            cfg["voice"] = voice
        return PhoonnxTTSPlugin(config=cfg)
    return None

@click.command()
@click.option("--words", default="hey_mycroft", help="Wake words")
@click.option("--lang", default="en", help="Language")
@click.option("--out-dir", required=True, help="Output directory")
@click.option("--target-samples", default=1000, help="Target samples")
@click.option("--use-phoonnx", is_flag=True, default=True)
@click.option("--use-google", is_flag=True, default=False)
@click.option("--use-edge", is_flag=True, default=False)
def main(words, lang, out_dir, target_samples, use_phoonnx, use_google, use_edge):
    ww_list = words.split()
    output_path = Path(out_dir)
    ww_dir = output_path / "wake_word"
    notww_dir = output_path / "not_wake_word"
    
    # Simple generation for demonstration - full notebook has more complexity
    # with specific dataset loading. This ports the core logic.
    
    engines = []
    if use_phoonnx: engines.append("phoonnx")
    if use_google: engines.append("google")
    if use_edge: engines.append("edge")
    
    if not engines:
        print("No engines enabled.")
        return

    samples_per_word = target_samples // len(ww_list)
    
    for word in ww_list:
        print(f"Generating {word}...")
        word_out = ww_dir / word
        word_out.mkdir(parents=True, exist_ok=True)
        
        engine = random.choice(engines)
        plugin = create_tts_plugin(engine, lang=lang)
        
        for i in tqdm(range(samples_per_word)):
            text = word.replace("_", " ")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            plugin.get_tts(text, tmp_path, lang=lang)
            audio = load_audio(tmp_path)
            os.remove(tmp_path)
            if audio is not None:
                save_audio(word_out / f"{word}_{i:05d}.wav", audio)

if __name__ == "__main__":
    main()
