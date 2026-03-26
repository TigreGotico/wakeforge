"""Rewrite metadata CSVs to use local paths where files have been copied.

Replaces /mnt/hdd4/ww_datasets/vc_output/hey_mycroft/
      and /mnt/hdd4/ww_datasets/synth_output/en/hey_mycroft/
with their local equivalents under experiments/hey_mycroft/dataset/positives/.

Run after the rsync completes.
"""
from pathlib import Path

TRAIN_CSV = Path("experiments/hey_mycroft/dataset/train/metadata.csv")
TEST_CSV  = Path("experiments/hey_mycroft/dataset/test/metadata.csv")

REMAPS = {
    "/mnt/hdd4/ww_datasets/vc_output/hey_mycroft/":
        "experiments/hey_mycroft/dataset/positives/vc_output/",
    "/mnt/hdd4/ww_datasets/synth_output/en/hey_mycroft/":
        "experiments/hey_mycroft/dataset/positives/synth_output/",
    "/mnt/hdd4/ww_datasets/hf_datasets/not_wake_word_subset/":
        "experiments/hey_mycroft/dataset/negatives/not_wake_word_subset/",
}

for csv in [TRAIN_CSV, TEST_CSV]:
    lines = csv.read_text().splitlines()
    new_lines = []
    remapped = 0
    missing = 0
    for line in lines:
        if not line.strip():
            continue
        path, label = line.rsplit(",", 1)
        new_path = path
        for remote, local in REMAPS.items():
            if path.startswith(remote):
                candidate = path.replace(remote, local)
                if Path(candidate).exists():
                    new_path = candidate
                    remapped += 1
                else:
                    missing += 1  # copy not done yet — keep HDD path
                break
        new_lines.append(f"{new_path},{label}")
    csv.write_text("\n".join(new_lines) + "\n")
    print(f"{csv.name}: {remapped} remapped to local, {missing} still on HDD")
