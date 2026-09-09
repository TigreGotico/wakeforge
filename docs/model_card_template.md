# Model card template for published wake-word models

Copy this file to `README.md` in the Hugging Face model repo and fill every
field in braces. A field that cannot be filled is a reason not to publish,
not a field to delete. The YAML header is what the Hub indexes; the body is
what a person reads.

```yaml
---
license: {apache-2.0 | cc-by-4.0}
language: {en}
tags:
  - wake-word
  - keyword-spotting
  - onnx
  - wakeforge
  - openvoiceos
library_name: wakeforge
datasets:
  - {every dataset id below, one per line}
---
```

# {wake phrase} wake-word model

A wake-word detector for the phrase "{wake phrase}", trained with
[wakeforge](https://github.com/TigreGotico/wakeforge) and run by
[ovos-ww-plugin-wakeforge](https://github.com/OpenVoiceOS/ovos-ww-plugin-wakeforge).
Two ONNX files: `featurizer.onnx` turns 16 kHz mono audio into features and
`model.onnx` scores them. Inference needs `onnxruntime` and `numpy` only.

## Use it in OpenVoiceOS

```json
{
  "listener": {"wake_word": "{phrase_id}"},
  "hotwords": {
    "{phrase_id}": {
      "module": "ovos-ww-plugin-wakeforge",
      "featurizer": "https://huggingface.co/OpenVoiceOS/{repo}/resolve/main/featurizer.onnx",
      "model": "https://huggingface.co/OpenVoiceOS/{repo}/resolve/main/model.onnx",
      "threshold": {threshold},
      "patience": {patience},
      "smoothing": "{ema | mean | max}",
      "listen": true
    }
  }
}
```

## Model

| field | value |
|---|---|
| wake phrase | {wake phrase} |
| tier | {tier name from ww_trainer/tiers.py} |
| featurizer | {name, and its parameters} |
| classifier head | {architecture, parameter count} |
| ONNX opset | {opset} |
| file sizes | featurizer {n} KB, model {n} KB |
| wakeforge version | {version that produced it} |
| training command | `{the exact command, pasted}` |
| checkpoint | {epoch selected and by which metric} |

## Training data

Every dataset the training run read, with its licence. Attribution for
CC BY sources is satisfied by this table.

| dataset | role | licence | samples used |
|---|---|---|---|
| [{id}](https://huggingface.co/datasets/{id}) | positives | {licence} | {n} |
| [{id}](https://huggingface.co/datasets/{id}) | speech negatives | {licence} | {n} |
| [{id}](https://huggingface.co/datasets/{id}) | general negatives | {licence} | {n} |
| [{id}](https://huggingface.co/datasets/{id}) | background noise augmentation | {licence} | {n} |
| [{id}](https://huggingface.co/datasets/{id}) | room impulse responses | {licence} | {n} |

Positives are synthetic: text-to-speech renderings of the phrase{, plus
voice-converted copies through voiceclonnx}. No real recording of a person
saying the phrase was used in training.

## Evaluation

| dataset | samples scored | error rate | false accept rate | false reject rate | operating point |
|---|---|---|---|---|---|
| [{id}](https://huggingface.co/datasets/{id}) | {n, at least 30} | {value} | {value} | {value} | threshold {t}, patience {p} |

Rows come from the
[OVOS Plugin Arena wake-word league](https://openvoiceos.github.io/ovos-plugin-arena)
and are reproducible with:

```
{the arena benchmark command, pasted}
```

The evaluation voices are disjoint from the training voices. A dataset
listed under training does not appear here.

## Limitations

Trained on synthetic speech, so far-field, accented and child speech are
under-represented. The operating point above is one point on the trade-off
curve; lower the threshold for fewer misses and more false accepts.

## Licence

The model files are released under {licence}. The training data licences
are listed above; each permits redistribution of a derived model with
attribution.

## Funding

Funded by the [NGI0 Commons Fund](https://nlnet.nl/project/OpenVoiceOS) /
[NLnet](https://nlnet.nl) under grant agreement No
[101135429](https://cordis.europa.eu/project/id/101135429), through the
European Commission's [Next Generation Internet](https://ngi.eu) programme.
