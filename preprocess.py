import csv
from pathlib import Path
import webrtcvad
import click
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from sklearn.model_selection import train_test_split

AUDIO_EXTS = [".wav", ".flac", ".mp3", ".m4a", ".ogg"]


def collect_audio_files(folders):
    files = []
    for f in folders:
        p = Path(f)
        if not p.exists():
            click.echo(f"Warning: folder not found {f}")
            continue
        for ext in AUDIO_EXTS:
            files.extend(list(p.rglob(f"*{ext}")))
    return sorted(files)


def trim_silence_vad(wav: np.ndarray, sr: int, frame_ms: int = 30) -> np.ndarray:
    vad = webrtcvad.Vad(2) # TODO - use silero instead
    frame_len = int(sr * frame_ms / 1000)
    pcm = (wav * 32767).astype(np.int16).tobytes()
    voiced = []
    for i in range(0, len(pcm), frame_len * 2):
        frame = pcm[i:i + frame_len * 2]
        if len(frame) < frame_len * 2:
            break
        try:
            voiced.append(vad.is_speech(frame, sr))
        except Exception:
            voiced.append(False)
    if not any(voiced):
        return wav
    idx = np.where(voiced)[0]
    start = max(0, idx[0] * frame_len)
    end = min(len(wav), (idx[-1] + 1) * frame_len)
    return wav[start:end]


def convert_and_save(src_path: Path, dst_path: Path, sr: int, use_vad: bool):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        wav, orig_sr = torchaudio.load(str(src_path))
        wav = wav.mean(0).numpy()  # mono
        if orig_sr != sr:
            wav = torchaudio.functional.resample(torch.tensor(wav), orig_sr, sr).numpy()
        if use_vad:
            wav = trim_silence_vad(wav, sr)
        # Normalize: wav = wav / (np.max(np.abs(wav)) + 1e-9)
        # We need to manually normalize when using torch.tensor on numpy array (bug in some versions)
        max_abs = np.max(np.abs(wav))
        if max_abs > 0:
            wav = wav / max_abs

        # Ensure it's in a format torchaudio expects (float32, [1, n])
        torchaudio.save(str(dst_path), torch.tensor(wav).unsqueeze(0).float(), sr)
        return True
    except Exception as e:
        click.echo(f"Error converting {src_path}: {e}")
        return False


@click.command()
@click.option('--wake', 'wake_folders', multiple=True, required=True, help='Folders containing wake-word samples')
@click.option('--negative', 'neg_folders', multiple=True, required=True, help='Folders containing non-wake samples')
@click.option('--out', 'out_dir', required=True, help='Output folder to store processed audio and metadata')
@click.option('--sr', 'sample_rate', default=16000, show_default=True, help='Target sample rate for conversion')
@click.option('--vad', is_flag=True, help='Trim silence using VAD (requires webrtcvad)')
@click.option('--test-split', default=0.2, show_default=True, type=float,
              help='Fraction of data to use for the test set')
def main(wake_folders, neg_folders, out_dir, sample_rate, vad, test_split):
    out_dir = Path(out_dir)

    temp_processed_out = out_dir / 'processed'
    wake_temp_out = temp_processed_out / 'wakes'
    neg_temp_out = temp_processed_out / 'negatives'

    train_out = out_dir / 'train'
    test_out = out_dir / 'test'

    wake_temp_out.mkdir(parents=True, exist_ok=True)
    neg_temp_out.mkdir(parents=True, exist_ok=True)

    click.echo("--- Starting Preprocessing ---")
    click.echo(f"Output directory: {out_dir}")
    click.echo(f"Target sample rate: {sample_rate} Hz. VAD trimming: {'Enabled' if vad else 'Disabled'}")
    click.echo(f"Train/Test split ratio: {1.0 - test_split}/{test_split}")

    # --- COLLECTING FILES ---
    click.echo(f"\nCollecting wake files from: {wake_folders}")
    wake_files = collect_audio_files(wake_folders)
    click.echo(f"Found {len(wake_files)} potential wake files.")

    click.echo(f"\nCollecting negative files from: {neg_folders}")
    neg_files = collect_audio_files(neg_folders)
    click.echo(f"Found {len(neg_files)} potential negative files.")

    # --- PROCESSING TO TEMP FOLDER ---
    wake_metadata = []
    processed_wake_count = 0
    failed_wake_count = 0

    click.echo(f"\nProcessing {len(wake_files)} wake files to temporary location...")
    for f in tqdm(wake_files, desc="Processing Wakes", unit="file"):
        dst = wake_temp_out / (f.stem + '.wav')
        if convert_and_save(f, dst, sample_rate, use_vad=vad):
            # Store the temporary path (relative to out_dir) and original file name
            # We use the temporary path for splitting, and the original Path object for the move operation later.
            wake_metadata.append({'temp_path': str(dst.relative_to(out_dir)), 'label': 1, 'path_obj': dst})
            processed_wake_count += 1
        else:
            failed_wake_count += 1

    neg_metadata = []
    processed_neg_count = 0
    failed_neg_count = 0
    click.echo(f"\nProcessing {len(neg_files)} negative files to temporary location...")
    for f in tqdm(neg_files, desc="Processing Negatives", unit="file"):
        dst = neg_temp_out / (f.stem + '.wav')
        if convert_and_save(f, dst, sample_rate, use_vad=vad):
            neg_metadata.append({'temp_path': str(dst.relative_to(out_dir)), 'label': 0, 'path_obj': dst})
            processed_neg_count += 1
        else:
            failed_neg_count += 1

    # --- SPLITTING LOGIC ---

    click.echo(f"\nSplitting data into train ({1.0 - test_split:.1%}) and test ({test_split:.1%})...")

    # Split each class independently on the list of dictionaries
    wake_train, wake_test = train_test_split(wake_metadata, test_size=test_split, random_state=42)
    neg_train, neg_test = train_test_split(neg_metadata, test_size=test_split, random_state=42)

    # Combine the splits
    train_data = wake_train + neg_train
    test_data = wake_test + neg_test

    # Optional: shuffle the combined lists
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    # --- FILE MOVING AND METADATA CREATION ---

    def move_and_generate_metadata(data_list, split_dir, split_name):
        split_dir.mkdir(parents=True, exist_ok=True)
        # Create subdirectories for 'wakes' and 'negatives' within the split folder
        (split_dir / 'wakes').mkdir(parents=True, exist_ok=True)
        (split_dir / 'negatives').mkdir(parents=True, exist_ok=True)

        metadata = []

        click.echo(f"Moving {len(data_list)} files to {split_dir.name}/ ...")

        for item in tqdm(data_list, desc=f"Moving {split_name}", unit="file"):
            temp_path_obj = item['path_obj']
            label = item['label']

            # Determine the final destination based on label (1=wake, 0=negative)
            class_folder = 'wakes' if label == 1 else 'negatives'
            final_dst = split_dir / class_folder / temp_path_obj.name

            # Move the file
            temp_path_obj.rename(final_dst)

            # The path for the metadata.csv must be relative to the main out_dir, NOT the split_dir!
            # Example: 'train/wakes/file.wav'
            metadata_path = str(final_dst)
            metadata.append((metadata_path, label))

        return metadata

    # Move files and collect final metadata
    train_metadata_final = move_and_generate_metadata(train_data, train_out, 'Train')
    test_metadata_final = move_and_generate_metadata(test_data, test_out, 'Test')

    # Cleanup temporary directory
    import shutil
    shutil.rmtree(temp_processed_out, ignore_errors=True)

    # --- WRITING FINAL CSV FILES ---

    def write_csv(filepath, data):
        with filepath.open('w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header for compatibility
            writer.writerow(['path', 'label'])
            for path, label in data:
                writer.writerow([path, label])

    train_csv_path = train_out / 'metadata.csv'
    test_csv_path = test_out / 'metadata.csv'

    write_csv(train_csv_path, train_metadata_final)
    write_csv(test_csv_path, test_metadata_final)

    click.echo("\n--- Preprocessing Complete ---")
    click.echo(f"Successfully processed {processed_wake_count} wake files.")
    click.echo(f"Successfully processed {processed_neg_count} negative files.")
    if failed_wake_count > 0 or failed_neg_count > 0:
        click.echo(
            f"Warning: Failed to process {failed_wake_count + failed_neg_count} files (wake: {failed_wake_count}, negative: {failed_neg_count}). See conversion errors above.")

    click.echo(f"Done. Wrote Train metadata with {len(train_metadata_final)} entries at {train_csv_path}")
    click.echo(f"Done. Wrote Test metadata with {len(test_metadata_final)} entries at {test_csv_path}")


if __name__ == '__main__':
    main()