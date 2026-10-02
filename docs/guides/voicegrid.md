# Voice-grid synthetic data

`ww_trainer-voicegrid` turns one phrase and one language into a large, speaker-diverse set of clean
synthetic clips. It says the phrase with every donor voice the language has, at several speaking
rates, several pitches and three text forms, using a bounded pool of workers, and writes 16 kHz mono
PCM files plus a `manifest.csv` that records every parameter. It is rerunnable: a second run skips
every clip that exists, so an interrupted run resumes and a finished one does nothing.

Use it instead of `ww_trainer-datagen` when the goal is breadth of speakers and delivery. The
datagen pipeline draws a random voice per clip and also downloads negatives, splits and trims; the
voice grid is exhaustive over the voices, resumable, and produces positives only.

## Install

```bash
uv pip install --pre "wakeforge[voicegrid]"
```

The extra brings `phoonnx` (OmniVoice, Kokoro and the other local engines), `edge-tts`, the OVOS TTS
plugins, `voiceclonnx`, `speakeronnx`, `uroman` and `onnx-asr`. For OmniVoice on a GPU install the official package as well
(`uv pip install omnivoice`); without it the slower ONNX export runs.

## One command

```bash
ww_trainer-voicegrid --wake-word "hey jarvis" --lang en --output-dir data/hey_jarvis --workers 4
```

The output directory then holds:

| Path | Content |
|---|---|
| `clips/*.wav` | 16 kHz mono 16-bit PCM clips |
| `manifest.csv` | one row per clip: `file, stage, engine, voice, lang, text, form, rate, pitch, copy, label, source, reference` |
| `manifest_rejected.csv`, `rejected/` | clips the intelligibility check refused, with reason and score |
| `dedup.csv` | one row per clip: `file, kept, duplicate_of, similarity` |
| `metadata.csv` | `path,label` rows, no header, for the kept clips; the format `ww_trainer-train --metadata` reads (`--label 0` for negative phrases) |

A clip name is built from its grid cell: `<phrase>-<hash>__<donor>__<voice>__r<rate>__p<pitch>__<form>__c<copy>.wav`.
`rate` is a percentage (`-25` to `+25`), `pitch` a shift in Hz, `copy` is `0` for a clean clip and
`1..N` for the clones of it. The name starts with the phrase and a short hash of it, so phrases never collide. `rate` and `pitch` read `na` and are empty in the manifest for a voice
that has no such control.

Each stage prints one summary line, for example
`VOICEGRID stage=tts planned=900 existing=0 written=897 failed=3 seconds=412.6`. The exit status is
non-zero when any clip failed; rerun the same command to retry them.

## The grid

For every donor voice the grid is the product of

- speaking rates: `-25,-12,0,12,25` percent (`--rates`),
- pitches: `-10,0,10` Hz (`--pitches`),
- text forms: `plain`, `exclaim` (`x!`) and `question` (`x?`) (`--forms`; `ellipsis` is also known).

Negative values need the `=` spelling: `--rates=-10,10`.

Rate and pitch are applied by each engine's own control. Kokoro (StyleTTS2) and Supertonic take a
speed multiplier, Piper-style and OmniVoice voices a length scale, and edge-tts takes both a rate and
a pitch. PocketTTS and the other engines without a speed control, and every local engine for pitch,
do not vary that axis: they contribute one value for it instead of repeated identical clips.

`--max-clean-per-voice K` keeps only a seeded subset of K grid cells per voice, so that a clean clip
is a small part of the data when cloning or other voices supply the variety. `--max-voices-per-engine N`
takes a seeded sample of N voices per donor, which is useful for a trial run. `--dry-run` prints the
number of clips the grid would produce and writes nothing.

## Donors

Which engines serve a language is data, not code. The packaged roster
(`ww_trainer/data/donor_roster.json`) gives each language a list of named donors, always led by
OmniVoice, and `--lang de` picks the German list. The roster covers English, the Iberian languages,
the large European and Asian languages, Arabic, and a long tail where OmniVoice and edge-tts are the
only modern engines; a language it does not list gets OmniVoice and edge-tts.

Each donor names how to build it: a `phoonnx` voice-index file with an optional list of voice-id
globs, the `omnivoice` donor, `edge`, or an OVOS TTS `plugin`. Override the roster with a file of the
same shape, or name donors directly for all languages:

```bash
ww_trainer-voicegrid --wake-word "hola mundo" --lang es --donors omnivoice,supertonic,edge ...
ww_trainer-voicegrid --wake-word "hola mundo" --lang es --donors my_roster.json ...
```

A roster donor whose Python package is not installed (for example the Google plugin) is skipped with
a warning; a donor named on the command line must be available.

### OmniVoice voices

OmniVoice has one entry per language and no fixed speakers. Called without a reference clip it
invents a voice, so its value is a new voice per clip. The OmniVoice donor does not enumerate
voices: it generates `--omnivoice-clips N` clips per phrase (default 300), each with its own seed
and therefore its own random speaker, rotating the text forms and two speaking rates
(`--omnivoice-rates`, default 0 and +15 percent). The seed of clip `i` is a function of `--seed`,
the language, the phrase and `i`, so a rerun produces the same clips and resumes where it stopped.
The seed is part of the voice id (`omnivoice/en#1873492011`) and so of every file name and manifest
row. Each clip also gets a voice-design style: `elderly male`, `female, british accent`,
`male, american accent`, `young adult female`, or `auto` (no description). Repeat `--omnivoice-style`
to give your own list. The styles are drawn with weights recomputed before every batch from each
style's accept rate so far, never below 10% each so that no style disappears, and the style of every
clip is recorded (`style` column of the manifest, `omnivoice_styles.csv`). In measurements on
"okay nabu", `elderly male` and `female, british accent` were accepted about 96% of the time against
54% for `auto`. A macrolanguage such
as `ar` rotates across its varieties (`arb`, `arz`, `ary`, ...; the mapping is the roster's
`macrolanguages` table).

Two backends run it, chosen automatically. When the official PyTorch package `omnivoice` is
importable it is used, in batches (`--batch-size`, default 10) on `--device` (`cuda`, `rocm`, which is
torch's `cuda` device on AMD, `cpu`, or `auto`); the generator stops if the model does not end up on
the device it asked for. Otherwise the ONNX export through `phoonnx` runs one clip at a time, which
on a CPU is about 40 times slower than real time and is only meant as a fallback.

OmniVoice does not hold a speaker across clips: a different seed gives a different speaker, and a
different rate or text form of the same seed sounds like another speaker too.

### Intelligibility check

OmniVoice garbles some phrases in a large share of its clips, so every OmniVoice clip is checked
before it is accepted (`--verify`, default `align`).

`align` forces a CTC alignment of the phrase to the clip with torchaudio's `MMS_FA` model
(multilingual, about 1,100 languages; the phrase is romanised with `uroman`) and scores the clip by
the mean per-frame log-probability along the alignment path. No per-phrase spelling list is needed.
The threshold is derived from the data, per phrase: the generator scores the edge-tts clips of the
same phrase (generated earlier in the same run) and rejects OmniVoice clips below the
`--verify-percentile` percentile of that distribution (default 5). Only edge-tts clips are used as
references: they score like clean speech, while other synthesizers can score lower than OmniVoice's
bad clips and make the threshold lenient. When edge-tts has fewer than 10 clips of the phrase (it
lacks the language, or `edge` is not among the donors) a fixed threshold of -0.64 is used, and
`threshold_source` in the manifest says `fallback` instead of `edge`; `--verify-threshold` sets the
value directly (`manual`). The alignment also gives where the phrase starts and ends, and an accepted
clip is trimmed to those boundaries plus 100 ms. Clips outside 0.3 to 3 seconds after trimming, or
with too little voiced energy, are rejected too.

`--require-asr-langs da,it,fr` makes the clips of the listed languages pass both the alignment and
the ASR check. Set it for languages where the alignment accepts clips that listeners or an ASR model
do not understand; in our measurements the alignment was lenient for Danish, Italian and French. It
is empty by default.

`asr` instead compares an `onnx_asr` transcript (`nemo-parakeet-tdt-0.6b-v3`, multilingual across the
European languages, CPU) with the phrase after normalising both (lower case, accents folded,
punctuation dropped): similarity at least `--asr-threshold` (default 0.8) to the phrase or to an
alternative spelling from `--accept-spellings` (for example `--accept-spellings "okay nahbu"`)
accepts it. `both` requires the alignment and the transcript. `--verify none` turns the check off.
`--verify-all-donors` checks every donor's clips and, for alignment, needs `--verify-threshold`.

Every score is stored: each clip's `score`, `threshold` and `threshold_source` are columns of
`manifest.csv`, and `verify_scores.csv` keeps scores so a rerun does not score a clip twice. A
rejected clip is never deleted: it moves to `rejected/` and is listed in `manifest_rejected.csv`
with its reason (`score`, `duration`, `energy`, `asr`, `align-failed`), score, threshold and
transcript, and it counts as attempted. Generation continues until `--omnivoice-clips` clips are
accepted or four times that many indexes have been tried.

### Several phrases

`--phrases-file FILE` takes one phrase per line, or `phrase<TAB>label`, and runs the whole grid for
each; with `--wake-word` as well, the wake word is the first phrase. Use it for the adversarial
negatives of a wake word: a line `okay nabi<TAB>0` writes `0` in `metadata.csv` for those clips
(`--label` is the label of a line without one).

### Held-out voices

The benchmark tests a model on voices it must never have heard in training, so the generator
excludes them by default. The held-out voices are the six Kokoro speakers `af_heart`, `af_kore`,
`am_echo`, `am_puck`, `bf_lily` and `bm_fable` in every Kokoro version (`kokoro/`, `kokoro-v019/`,
`kokoro-v11/`), and the six edge-tts voices that the OVOS edge-tts plugin's tests treat as held out,
`en-GB-MaisieNeural`, `en-IE-ConnorNeural`, `en-IN-PrabhatNeural`, `en-KE-AsiliaNeural`,
`en-NZ-MollyNeural` and `en-SG-WayneNeural`, each with its `Multilingual` twin. Without this a model
trained on the default output would be tested on its own training voices. Add more with
`--exclude-voices` (case-insensitive globs on the voice id, comma separated or repeated);
`--no-default-excludes` turns the built-in list off.

Edge voices that are one speaker under two names are reduced to one: `AvaNeural` and
`AvaMultilingualNeural`, `BrianNeural` and `BrianMultilingualNeural`, `EmmaNeural` and
`EmmaMultilingualNeural` keep the plain name.

## Optional stages

### Voice cloning

```bash
ww_trainer-voicegrid ... --vc-refs refs/ --vc-copies 2 --vc-device cuda --workers 4
```

Cloning is off by default. With `--vc-refs` every clean clip is converted `--vc-copies` times by
`voiceclonnx` (Chatterbox by default) onto reference speakers drawn from the `.wav` files under the
directory. The reference of each copy is a function of `--seed` and the clip name, so reruns agree.
The first clip runs alone and the stage then checks that the cloner's ONNX sessions really run on the
requested device (`--vc-device cuda`, `rocm`, `cpu` or `auto`); if they do not, it stops instead of
silently converting on the CPU.

### Speaker deduplication

Deduplication runs by default after synthesis (and cloning). It embeds every clip with `speakeronnx`
(`wespeaker-resnet34`, change it with `--dedup-model`) and walks the clips greedily in a seeded
order, cloned clips first, keeping a clip unless its cosine similarity to an already kept clip is at
least the threshold (`--dedup-speaker-threshold`, default 0.9; `--no-dedup` skips the stage).
Nothing is deleted: `dedup.csv` marks each clip `kept` or not, names the clip it duplicates and the
similarity, and `metadata.csv` lists only the kept clips.

The default 0.9 sits above the spread of one voice and below identical audio. On ten Kokoro clips of
two voices, clips of the same voice at different rates scored 0.73 to 0.89 against each other,
clips of different voices 0.18 to 0.36, and clips that Kokoro rendered identically (the `!` and `?`
forms of one voice at one rate came out the same in some cases) scored 1.00. Treat the default as a
starting point and lower it to thin a data set further.

## Resuming and parallelism

Clips are written under a temporary name and moved into place, so an interrupted run leaves no
truncated file under a final name. The manifest is rebuilt from the clips that exist at the end of
each run. `--workers` is the size of the worker pool for synthesis, cloning and embedding; each donor
also has its own limit, so a remote service such as edge-tts is paced to three concurrent requests
with a pause after each, however many workers there are, and a local model runs at most `--workers`
inferences at a time. Set `OMP_NUM_THREADS` and pin cores when sharing a machine: a local OmniVoice
or Kokoro inference uses every core the process may use.

## In a notebook

```python
from ww_trainer.voicegrid import cli_main

try:
    cli_main(["--wake-word", "hey jarvis", "--lang", "en", "--output-dir", "data/hey_jarvis"])
except SystemExit as done:
    assert done.code == 0
```

or `!ww_trainer-voicegrid --wake-word "hey jarvis" --lang en --output-dir data/hey_jarvis`.
