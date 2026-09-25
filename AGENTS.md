# Training a wake-word model with wakeforge: what an agent must know

This file is for any coding agent asked to train, evaluate or advise on a
wake-word model with this repository. It states what a good model is, how the
data must be shaped, and the traps that have produced models which looked
perfect and were useless. Read it before running anything.

## A model is not good until it passes validation

A synthetic test split with F1 of 1.0 proves nothing. It shares every defect of
the training data. A wake-word detector is judged at one operating point
(threshold, patience, smoothing of the runtime plugin) on audio it never saw:

| measure | audio | bar |
|---|---|---|
| false reject rate | the phrase in voices that produced no training clip | under 10 % |
| near-miss acceptance | phonetic neighbours of the phrase ("hey microsoft" for "hey mycroft") in those voices | under 5 % |
| false accepts per hour | read speech by many speakers, conversational speech, environmental noise, music, each measured separately | under 1 per hour on each |
| chunk robustness | the runtime plugin fed at 1280, 1600, 2048 and 4096-sample chunks | no crash, same decisions |

Measure with `ww-benchmarks` (`eval_fn.py` for misses, `eval_fp.py` for false
accepts per hour with a two-second window and a half-second slide) through
its `wakeforge` adapter, and paste the command next to every number. A model
that passes only the synthetic held-out row has not been validated.

## Data shape decides everything

The first models trained from this repository scored 0.8 to 0.98 on any
continuous speech, noise or music and 0.0 on silence, giving thousands of
false accepts per hour, while the training negatives scored 0.0. The cause was
the shape of the data, not the network: the negatives were VAD-trimmed
fragments of 0.3 to 1.0 s and the positives 0.4 to 1.2 s, so the network
learned that sustained sound is the wake word. These rules follow.

1. Every training example is the same length (`--window-s`, 1.5 s by
   default). Positives are VAD trimmed then placed at a random offset in a
   zero-padded window. Negatives are windows cut at random offsets from the
   full clips, never trimmed, plus one short zero-padded fragment per clip so
   that padding marks neither class. `--window-s 0` restores raw clips and
   reintroduces the defect.
2. Negatives cover every class the detector will hear, at window length:
   speech that is not the phrase, environmental noise, music. AudioSet gives
   the last two and a speech corpus of many speakers gives the first. Words of
   one second are not sustained speech.
3. Near-miss phrases are negatives. Pass `--adversarial --adversarial-file
   near_miss.txt` with one phonetic neighbour per line; each is rendered by
   every installed TTS plugin `--adversarial-per-text` times. Without this,
   "hey microsoft" and "hey minecraft" are accepted like the phrase.
4. Held-out voices stay out. Keep a set of TTS voices that never render a
   training clip and use them only for the false reject and near-miss rows.
5. More positives means more voices, not more renderings of one voice.
   `synthesize_positives` picks a random installed plugin per clip with that
   plugin's default voice. Install several voices or plugins to get variety.

## Licences

Only datasets whose Hugging Face card states a permissive licence (CC0, CC BY,
Apache, MIT) may enter a training set that becomes a published model.
`DATASET_LICENSES` in `ww_trainer/datagen.py` is the record and a test refuses
any pipeline dataset missing from it. ESC-50 is CC BY-NC and is out. A dataset
whose card states no licence is out. Evaluation audio may be anything you may
lawfully use; state its licence in the model card.

## Runtime and hardware facts

- `datasets` 3 and later decode and encode audio through torchcodec, whose
  PyPI wheel links CUDA libraries. On a ROCm or CPU-only host it cannot load.
  `download_hf_audio_dataset` therefore fetches folder repos file by file with
  `huggingface_hub` and decodes parquet audio bytes with soundfile. Do not add
  a `datasets` Audio decode path back.
- torchaudio without torchcodec falls back to soundfile for load and save and
  logs one warning per file. The warnings are noise, not errors.
- On an AMD GPU, install torch from `https://download.pytorch.org/whl/rocm<ver>`
  into a venv on an executable filesystem (a `noexec` mount cannot map torch's
  shared objects) and set `HF_HOME` to a disk with space.
- The small tier trains at about 20 to 40 s per epoch on 16 000 windows on a
  consumer GPU with about 1 GB of VRAM. `scripts/train/train_default_chunked.py`
  runs training in wall-clock chunks with a checkpoint per epoch and resumes
  from the last complete epoch, so a killed run is a pause.
- The runtime plugin `ovos-ww-plugin-wakeforge` streams whatever chunk the
  listener gives it. At chunks that leave fewer than 200 samples for the
  featurizer (1280-sample chunks do) onnxruntime aborts on the STFT reflect
  pad, and at 1280 or 1600 samples the smoother fires on negatives. The
  listener's default is 2048 samples. Measure the plugin at the chunk size
  the deployment uses before writing a threshold.

## Publishing

A model is published only after it passes every validation row. The card
follows `docs/model_card_template.md`: training data by dataset id with
licence, evaluation rows with their commands, the operating point, the
model's own licence and the NGI0 funding block.
