# Feature Enrichment Wrappers

Wrappers that decorate any `BaseExtractor`, appending extra feature channels without modifying the base extractor. All defined in `ww_trainer/feats.py:1318-1700`.

**Key constraint:** Wrappers cannot be ONNX-exported directly. Export the base extractor separately; enrichment features must be computed at runtime or replicated in a custom ONNX graph.

---

## Overview

| Wrapper | Class | Line | Extra Dims | Features Added |
|---------|-------|------|------------|----------------|
| VoiceActivity | `VoiceActivityExtractor` | `feats.py:1321` | +4 | Log RMS energy, ZCR, spectral flatness, VAD probability |
| Pitch | `PitchExtractor` | `feats.py:1443` | +3 | Normalized F0, voicing probability, F0 delta |
| MultiResolution | `MultiResolutionExtractor` | `feats.py:1577` | +coarse_dim | Fine + coarse extractor outputs concatenated |
| SNRAware | `SNRAwareExtractor` | `feats.py:1630` | +2 | Per-frame SNR estimate, noise floor estimate |

---

## VoiceActivityExtractor -- `feats.py:1321`

Appends 4 energy-based VAD signals computed per frame (`feats.py:1355`):

1. **Log RMS energy** -- normalized to [0,1] per utterance (`feats.py:1374-1380`)
2. **Zero-crossing rate** -- high for noise, low for voiced speech (`feats.py:1383-1384`)
3. **Spectral flatness** -- geometric/arithmetic mean ratio; high for noise, low for tonal (`feats.py:1386-1392`)
4. **VAD probability** -- soft sigmoid combination: `sigmoid(3*energy - 2*zcr - 2*flatness)` (`feats.py:1396-1398`)

Time alignment: if VAD frame count differs from base extractor output, linear interpolation aligns them (`feats.py:1427-1431`).

**When to use:** Datasets with significant silence or background noise. Lets the classifier explicitly attend to speech regions.

**When NOT to use:** Pre-trimmed, clean audio where all frames contain speech.

```python
from ww_trainer.feats import MfccExtractor, VoiceActivityExtractor

base = MfccExtractor(n_mfcc=40)
extractor = VoiceActivityExtractor(base)  # output dim: 44
```

---

## PitchExtractor -- `feats.py:1443`

Appends 3 pitch features via autocorrelation-based F0 estimation (`feats.py:1482`):

1. **Normalized F0** -- fundamental frequency scaled to [0,1] within `[f0_min, f0_max]` (`feats.py:1531-1532`)
2. **Voicing probability** -- autocorrelation peak value, clipped to [0,1] (`feats.py:1535`)
3. **F0 delta** -- finite difference of normalized F0 (pitch dynamics) (`feats.py:1538-1540`)

Pitch detection uses windowed autocorrelation via FFT (`feats.py:1506-1510`), searching for peaks in the lag range `[sr/f0_max, sr/f0_min]` (`feats.py:1516-1517`).

**When to use:** Multi-word wake phrases where prosody (intonation pattern) is discriminative. Speaker-dependent wake words.

**When NOT to use:** Single short wake words where pitch carries little discriminative information.

```python
from ww_trainer.feats import MfccExtractor, PitchExtractor

base = MfccExtractor(n_mfcc=40)
extractor = PitchExtractor(base, f0_min=50.0, f0_max=600.0)  # output dim: 43
```

---

## MultiResolutionExtractor -- `feats.py:1577`

Runs two extractors with different hop lengths and concatenates outputs. Coarse features are interpolated to match the fine extractor's time axis (`feats.py:1615-1619`).

**When to use:** Capture both fine temporal detail (short hop) and broad temporal patterns (long hop) simultaneously.

```python
from ww_trainer.feats import MfccExtractor, MultiResolutionExtractor

fine = MfccExtractor(n_mfcc=40, hop_length=160)    # 10ms hop
coarse = MfccExtractor(n_mfcc=40, hop_length=480)  # 30ms hop
extractor = MultiResolutionExtractor(fine, coarse)  # output dim: 80
```

---

## SNRAwareExtractor -- `feats.py:1630`

Appends 2 per-frame SNR features (`feats.py:1665`):

1. **Normalized SNR** -- frame energy minus noise floor (log domain), normalized to [0,1] (`feats.py:1695-1699`)
2. **Noise floor estimate** -- percentile-based noise floor tracking (`feats.py:1689-1692`). Default: 10th percentile of frame energies.

**When to use:** Noisy deployment environments (cars, kitchens, factories). Lets the classifier weight clean frames more heavily.

**When NOT to use:** Clean, controlled environments.

```python
from ww_trainer.feats import PNCCExtractor, SNRAwareExtractor

base = PNCCExtractor(n_pncc=13)
extractor = SNRAwareExtractor(base, noise_percentile=10.0)  # output dim: 15
```

---

## Stacking Wrappers

Wrappers can be composed. Each adds its features to the previous output.

```python
from ww_trainer.feats import MfccExtractor, VoiceActivityExtractor, PitchExtractor, SNRAwareExtractor

base = MfccExtractor(n_mfcc=40)                      # 40
vad = VoiceActivityExtractor(base)                     # 44
pitch = PitchExtractor(vad)                            # 47
snr = SNRAwareExtractor(pitch)                         # 49
```

**Recommended stacks by scenario:**

| Scenario | Stack | Output Dim |
|----------|-------|------------|
| Noisy environment | PNCC + SNRAware | 15 |
| Multi-word phrase | MFCC + Pitch | 43 |
| Noisy + speech detection | MFCC + VAD + SNRAware | 46 |
| Maximum enrichment | MFCC + VAD + Pitch + SNRAware | 49 |
| Multi-resolution noisy | MultiRes(MFCC, MFCC) + SNRAware | 82 |
