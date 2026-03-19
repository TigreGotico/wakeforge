import os
import sys
import time
import random
import warnings
from typing import List, Dict, Optional
from uuid import uuid4
from pathlib import Path

import torch
import torchaudio as ta
from tqdm import tqdm
import click

def _collect_tts_metadata(
    lang: str,
    edge: bool = True,
    google: bool = True,
    piper: bool = False,
) -> List[Dict]:
    """
    Scans enabled TTS plugins and returns a list of metadata dicts.
    """
    lang_prefix = lang.split("-")[0].lower()
    metadata = []

    # --- Edge TTS voices ---
    if edge:
        try:
            from ovos_tts_plugin_edge_tts import VOICES
            rate_variations = (
                [f"+{r}%" for r in range(1, 35)]
                + [f"-{r}%" for r in range(1, 30)]
            )
            for locale, voices in VOICES.items():
                if not locale.startswith(lang_prefix):
                    continue
                for v in voices:
                    rate = random.choice(rate_variations)
                    metadata.append({
                        "type": "edge",
                        "name": f"edge_{v}",
                        "config": {"voice": v, "rate": rate},
                    })
        except ImportError:
            print("  ovos-tts-plugin-edge-tts not installed.")

    # --- Google Translate TTS ---
    if google:
        try:
            metadata.append({
                "type": "google",
                "name": f"google_{lang}",
                "config": {"lang": lang, "slow": random.choice([True, False])},
            })
        except Exception:
            print("  Google TTS metadata collection failed.")

    # --- Piper TTS ---
    if piper:
        try:
            from ovos_tts_plugin_piper.voice_models import get_available_voices
            from ovos_utils.lang import standardize_lang_tag
            for voice, data in get_available_voices(update_voices=False).items():
                l = standardize_lang_tag(data["language"]["code"])
                if not l.startswith(lang_prefix):
                    continue
                n_speakers = len(data["speaker_id_map"])
                voices_list = (
                    [f"{voice}#{i}" for i in range(n_speakers)]
                    if n_speakers > 0
                    else [voice]
                )
                for v in voices_list:
                    metadata.append({
                        "type": "piper",
                        "name": f"piper_{l}_{v}",
                        "config": {"lang": l, "voice": v},
                    })
        except ImportError:
            print("  ovos-tts-plugin-piper not installed.")

    print(f"  Total TTS configurations: {len(metadata)}")
    return metadata

def _instantiate_plugin(meta: Dict):
    """Create a TTS plugin instance from a metadata dict."""
    if meta["type"] == "edge":
        from ovos_tts_plugin_edge_tts import EdgeTTSPlugin
        return EdgeTTSPlugin(config=meta["config"])
    if meta["type"] == "google":
        from ovos_tts_plugin_google_tx import GoogleTranslateTTS
        return GoogleTranslateTTS(config=meta["config"])
    if meta["type"] == "piper":
        from ovos_tts_plugin_piper import PiperTTSPlugin
        return PiperTTSPlugin(config=meta["config"])
    raise ValueError(f"Unknown TTS type: {meta['type']}")

def synthesize_and_convert(
    wake_word: str,
    lang: str,
    output_dir: str,
    reference_voices_dir: Optional[str],
    n: int,
    device: str = "cuda",
    edge: bool = True,
    google: bool = True,
    piper: bool = False,
):
    """
    Generate N samples, each with a randomly-chosen (TTS plugin, reference voice)
    combination.
    """
    all_meta = _collect_tts_metadata(lang, edge=edge, google=google, piper=piper)
    if not all_meta:
        print("  No TTS metadata found — aborting.")
        return

    # --- Load VC model + reference voices (optional) ---
    vc_model = None
    ref_voices: List[str] = []

    if reference_voices_dir:
        base_path = Path(reference_voices_dir)
        ref_voices = [str(p) for p in base_path.glob("**/*.wav")]
        if not ref_voices:
            print(f"  No reference voices (.wav) found in {reference_voices_dir}")
        else:
            print(f"  Found {len(ref_voices)} reference voices. Loading VC model...")
            try:
                from chatterbox_onnx import ChatterboxOnnx
                vc_model = ChatterboxOnnx(device=device)
                print("  VC model loaded.")
            except ImportError:
                print("  chatterbox-onnx not installed. VC skipped.")
    else:
        print("  Voice conversion not enabled (reference voices dir not provided).")

    os.makedirs(output_dir, exist_ok=True)
    success, fail = 0, 0
    start = time.time()

    for i in tqdm(range(n), total=n, desc=f"Synth {wake_word}", unit="sample"):
        plugin_meta = random.choice(all_meta)

        try:
            plugin = _instantiate_plugin(plugin_meta)
        except Exception:
            fail += 1
            continue

        base_name = str(uuid4())[10:]
        tts_path = os.path.join(output_dir, f"{base_name}_tts.wav")
        vc_path  = os.path.join(output_dir, f"{base_name}.wav")

        try:
            # 1) Generate TTS audio
            plugin.get_tts(
                wake_word.replace("_", " ").replace("-", " "),
                tts_path,
                lang=lang,
                voice=plugin_meta["config"].get("voice"),
            )

            # 2) Voice conversion (if model loaded)
            if vc_model is not None and ref_voices:
                ref_file = random.choice(ref_voices)
                try:
                    vc_model.voice_convert(
                        source_audio_path=tts_path,
                        target_voice_path=ref_file,
                        output_file_name=vc_path,
                    )
                    success += 1
                    if os.path.exists(tts_path):
                        os.remove(tts_path)
                except Exception:
                    fail += 1
            else:
                os.rename(tts_path, vc_path)
                success += 1

        except Exception:
            fail += 1

    elapsed = time.time() - start
    print(f"  Completed {success} ok / {fail} failed in {elapsed:.1f}s")

@click.command()
@click.option("--words", default=os.getenv("WW_WORDS", "hey_mycroft"), help="Space-separated wake words")
@click.option("--lang", default=os.getenv("WW_LANG", "en"), help="Language code")
@click.option("--out-dir", default=os.getenv("SYNTH_OUTPUT_DIR", "synth_output"), help="Output directory")
@click.option("--n-samples", default=int(os.getenv("SYNTH_N", "900")), help="Samples per wake word")
@click.option("--vc-refs", default=os.getenv("SYNTH_VC_REFS", ""), help="Reference voices dir")
@click.option("--vc-device", default=os.getenv("SYNTH_VC_DEVICE", "cuda"), help="VC device")
@click.option("--use-edge", is_flag=True, default=os.getenv("SYNTH_USE_EDGE", "1") == "1")
@click.option("--use-google", is_flag=True, default=os.getenv("SYNTH_USE_GOOGLE", "1") == "1")
@click.option("--use-piper", is_flag=True, default=os.getenv("SYNTH_USE_PIPER", "0") == "1")
@click.option("--seed", default=os.getenv("SYNTH_SEED", ""), help="Random seed")
def main(words, lang, out_dir, n_samples, vc_refs, vc_device, use_edge, use_google, use_piper, seed):
    warnings.filterwarnings("ignore")
    if seed:
        random.seed(int(seed))
        print(f"Using random seed: {seed}")

    ww_list = words.split()
    for ww in ww_list:
        ww_out = os.path.join(out_dir, ww)
        print(f"\nSynthesising: {ww} -> {ww_out}")
        synthesize_and_convert(
            wake_word=ww,
            lang=lang,
            output_dir=ww_out,
            reference_voices_dir=vc_refs if vc_refs else None,
            n=n_samples,
            device=vc_device,
            edge=use_edge,
            google=use_google,
            piper=use_piper,
        )

if __name__ == "__main__":
    main()
