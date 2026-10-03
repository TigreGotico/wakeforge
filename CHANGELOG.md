# Changelog

## [0.17.0a1](https://github.com/TigreGotico/wakeforge/tree/0.17.0a1) (2026-10-03)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.16.5a1...0.17.0a1)

**Merged pull requests:**

- feat: add ww\_trainer-voicegrid, a resumable parallel voice-grid data generator [\#96](https://github.com/TigreGotico/wakeforge/pull/96) ([JarbasAl](https://github.com/JarbasAl))

## [0.16.5a1](https://github.com/TigreGotico/wakeforge/tree/0.16.5a1) (2026-10-03)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.16.4a1...0.16.5a1)

**Merged pull requests:**

- fix\(datagen\): silence and near-silence negative windows [\#57](https://github.com/TigreGotico/wakeforge/pull/57) ([openvoiceos-bot](https://github.com/openvoiceos-bot))

## [0.16.4a1](https://github.com/TigreGotico/wakeforge/tree/0.16.4a1) (2026-10-03)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.16.3a1...0.16.4a1)

**Merged pull requests:**

- fix: stream a capped dataset download instead of filling the whole cache [\#89](https://github.com/TigreGotico/wakeforge/pull/89) ([JarbasAl](https://github.com/JarbasAl))

## [0.16.3a1](https://github.com/TigreGotico/wakeforge/tree/0.16.3a1) (2026-10-02)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.16.2a1...0.16.3a1)

**Merged pull requests:**

- fix: read the CPU model from cpuinfo before platform.processor [\#93](https://github.com/TigreGotico/wakeforge/pull/93) ([JarbasAl](https://github.com/JarbasAl))

## [0.16.2a1](https://github.com/TigreGotico/wakeforge/tree/0.16.2a1) (2026-10-02)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.16.1a1...0.16.2a1)

**Merged pull requests:**

- fix: resolve train metadata paths beside the CSV and report missing audio [\#94](https://github.com/TigreGotico/wakeforge/pull/94) ([JarbasAl](https://github.com/JarbasAl))

## [0.16.1a1](https://github.com/TigreGotico/wakeforge/tree/0.16.1a1) (2026-10-02)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.16.0a1...0.16.1a1)

**Merged pull requests:**

- fix\(datagen\): fixed-length windows for every example and near-miss negatives from a phrase file [\#40](https://github.com/TigreGotico/wakeforge/pull/40) ([openvoiceos-bot](https://github.com/openvoiceos-bot))

## [0.16.0a1](https://github.com/TigreGotico/wakeforge/tree/0.16.0a1) (2026-10-02)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.15.1a2...0.16.0a1)

**Merged pull requests:**

- feat: head bench ablation axes \(pretrained featurizer, factory heads, losses\) [\#88](https://github.com/TigreGotico/wakeforge/pull/88) ([JarbasAl](https://github.com/JarbasAl))

## [0.15.1a2](https://github.com/TigreGotico/wakeforge/tree/0.15.1a2) (2026-10-02)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.15.1a1...0.15.1a2)

**Merged pull requests:**

- docs: honest first-run time, real datagen CSV layout, copy-paste-safe shell blocks [\#91](https://github.com/TigreGotico/wakeforge/pull/91) ([JarbasAl](https://github.com/JarbasAl))

## [0.15.1a1](https://github.com/TigreGotico/wakeforge/tree/0.15.1a1) (2026-10-02)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.15.0a1...0.15.1a1)

**Merged pull requests:**

- fix: key the bench feature store by runtime and anneal the cosine schedule to the floor [\#90](https://github.com/TigreGotico/wakeforge/pull/90) ([JarbasAl](https://github.com/JarbasAl))

## [0.15.0a1](https://github.com/TigreGotico/wakeforge/tree/0.15.0a1) (2026-10-02)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.14.0a1...0.15.0a1)

**Merged pull requests:**

- feat\(train\): keep the checkpoint with the best recall at zero false accepts on calibration audio [\#83](https://github.com/TigreGotico/wakeforge/pull/83) ([JarbasAl](https://github.com/JarbasAl))

## [0.14.0a1](https://github.com/TigreGotico/wakeforge/tree/0.14.0a1) (2026-10-02)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.13.0a1...0.14.0a1)

**Merged pull requests:**

- feat\(export\): ww\_trainer-export-plugin writes a head or an ensemble in the OVOS plugin's bundled-model format [\#85](https://github.com/TigreGotico/wakeforge/pull/85) ([JarbasAl](https://github.com/JarbasAl))

## [0.13.0a1](https://github.com/TigreGotico/wakeforge/tree/0.13.0a1) (2026-10-02)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.12.0a1...0.13.0a1)

**Merged pull requests:**

- feat\(research\): head\_bench content-addressed feature store and cosine learning-rate schedule [\#87](https://github.com/TigreGotico/wakeforge/pull/87) ([JarbasAl](https://github.com/JarbasAl))

## [0.12.0a1](https://github.com/TigreGotico/wakeforge/tree/0.12.0a1) (2026-10-02)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.11.2a2...0.12.0a1)

**Merged pull requests:**

- feat\(augment\): device response augmentation \(microphone band, colour, level, compression, clipping, self-noise\) [\#82](https://github.com/TigreGotico/wakeforge/pull/82) ([JarbasAl](https://github.com/JarbasAl))

## [0.11.2a2](https://github.com/TigreGotico/wakeforge/tree/0.11.2a2) (2026-10-02)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.11.2a1...0.11.2a2)

**Merged pull requests:**

- test\(feats\): deterministic DeltaExtractor ONNX parity check [\#86](https://github.com/TigreGotico/wakeforge/pull/86) ([JarbasAl](https://github.com/JarbasAl))

## [0.11.2a1](https://github.com/TigreGotico/wakeforge/tree/0.11.2a1) (2026-10-02)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.11.1a1...0.11.2a1)

**Merged pull requests:**

- fix\(cli\): --aug-prob reaches the training loop, and --aug-warmup-epochs is exposed [\#84](https://github.com/TigreGotico/wakeforge/pull/84) ([JarbasAl](https://github.com/JarbasAl))

## [0.11.1a1](https://github.com/TigreGotico/wakeforge/tree/0.11.1a1) (2026-10-02)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.11.0a3...0.11.1a1)

**Merged pull requests:**

- fix\(datagen\): a voice per synthesized clip, paced requests, and no file from a failed one [\#76](https://github.com/TigreGotico/wakeforge/pull/76) ([JarbasAl](https://github.com/JarbasAl))

## [0.11.0a3](https://github.com/TigreGotico/wakeforge/tree/0.11.0a3) (2026-10-02)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.11.0a2...0.11.0a3)

## [0.11.0a2](https://github.com/TigreGotico/wakeforge/tree/0.11.0a2) (2026-10-01)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.11.0a1...0.11.0a2)

**Merged pull requests:**

- perf\(feature\_store\): float16 features and a parallel prefill of augmented variants [\#80](https://github.com/TigreGotico/wakeforge/pull/80) ([JarbasAl](https://github.com/JarbasAl))
- perf\(augment\): FFT reverb and a bounded cache of decoded augmentation clips [\#79](https://github.com/TigreGotico/wakeforge/pull/79) ([JarbasAl](https://github.com/JarbasAl))

## [0.11.0a1](https://github.com/TigreGotico/wakeforge/tree/0.11.0a1) (2026-10-01)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.10.0a1...0.11.0a1)

**Merged pull requests:**

- feat: multi-speaker synthetic data and sound-alike negatives for the launch wake words [\#75](https://github.com/TigreGotico/wakeforge/pull/75) ([JarbasAl](https://github.com/JarbasAl))

## [0.10.0a1](https://github.com/TigreGotico/wakeforge/tree/0.10.0a1) (2026-10-01)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.9.0a1...0.10.0a1)

**Merged pull requests:**

- feat: temporal relation distillation for the distillation trainer [\#63](https://github.com/TigreGotico/wakeforge/pull/63) ([openvoiceos-bot](https://github.com/openvoiceos-bot))

## [0.9.0a1](https://github.com/TigreGotico/wakeforge/tree/0.9.0a1) (2026-10-01)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.8.0a1...0.9.0a1)

**Merged pull requests:**

- feat: margin-aware contrastive regularization loss [\#62](https://github.com/TigreGotico/wakeforge/pull/62) ([openvoiceos-bot](https://github.com/openvoiceos-bot))

## [0.8.0a1](https://github.com/TigreGotico/wakeforge/tree/0.8.0a1) (2026-10-01)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.7.1a3...0.8.0a1)

**Merged pull requests:**

- feat: event-level false activations and FRR at a FA/hour budget [\#61](https://github.com/TigreGotico/wakeforge/pull/61) ([openvoiceos-bot](https://github.com/openvoiceos-bot))

## [0.7.1a3](https://github.com/TigreGotico/wakeforge/tree/0.7.1a3) (2026-10-01)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.7.1a2...0.7.1a3)

**Merged pull requests:**

- chore\(ci\): add the Renovate rebase config, keep a smoke-job diagnostic step [\#53](https://github.com/TigreGotico/wakeforge/pull/53) ([openvoiceos-bot](https://github.com/openvoiceos-bot))

## [0.7.1a2](https://github.com/TigreGotico/wakeforge/tree/0.7.1a2) (2026-10-01)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.7.1a1...0.7.1a2)

**Merged pull requests:**

- chore\(scripts\): drop the internal host from four scripts [\#55](https://github.com/TigreGotico/wakeforge/pull/55) ([openvoiceos-bot](https://github.com/openvoiceos-bot))

## [0.7.1a1](https://github.com/TigreGotico/wakeforge/tree/0.7.1a1) (2026-10-01)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.7.0a1...0.7.1a1)

**Merged pull requests:**

- fix\(datagen\): split each label, so the test set is never one class [\#74](https://github.com/TigreGotico/wakeforge/pull/74) ([openvoiceos-bot](https://github.com/openvoiceos-bot))

## [0.7.0a1](https://github.com/TigreGotico/wakeforge/tree/0.7.0a1) (2026-09-30)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.5.0a1...0.7.0a1)

**Merged pull requests:**

- feat: pretrained featurizers across every training path [\#73](https://github.com/TigreGotico/wakeforge/pull/73) ([JarbasAl](https://github.com/JarbasAl))
- feat: WakeHuBERT pretrained featurizer and notebook [\#70](https://github.com/TigreGotico/wakeforge/pull/70) ([JarbasAl](https://github.com/JarbasAl))

## [0.5.0a1](https://github.com/TigreGotico/wakeforge/tree/0.5.0a1) (2026-09-30)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.5a1...0.5.0a1)

**Merged pull requests:**

- feat\(research\): supervised word objective beside distillation \(MSWC word classes\) [\#69](https://github.com/TigreGotico/wakeforge/pull/69) ([JarbasAl](https://github.com/JarbasAl))

## [0.4.5a1](https://github.com/TigreGotico/wakeforge/tree/0.4.5a1) (2026-09-30)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.4a1...0.4.5a1)

**Merged pull requests:**

- fix\(loss\): losses match their definitions, and their learned parameters train [\#65](https://github.com/TigreGotico/wakeforge/pull/65) ([JarbasAl](https://github.com/JarbasAl))

## [0.4.4a1](https://github.com/TigreGotico/wakeforge/tree/0.4.4a1) (2026-09-30)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.3a5...0.4.4a1)

**Merged pull requests:**

- fix\(ci\): make the notebook compile and smoke checks pass [\#71](https://github.com/TigreGotico/wakeforge/pull/71) ([JarbasAl](https://github.com/JarbasAl))

## [0.4.3a5](https://github.com/TigreGotico/wakeforge/tree/0.4.3a5) (2026-09-25)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.3a4...0.4.3a5)

**Merged pull requests:**

- docs: repoint 166 stale source anchors, and ship the checker [\#60](https://github.com/TigreGotico/wakeforge/pull/60) ([openvoiceos-bot](https://github.com/openvoiceos-bot))

## [0.4.3a4](https://github.com/TigreGotico/wakeforge/tree/0.4.3a4) (2026-09-23)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.3a3...0.4.3a4)

**Merged pull requests:**

- ci: stop proposing a stable release, this repository has no master branch [\#59](https://github.com/TigreGotico/wakeforge/pull/59) ([openvoiceos-bot](https://github.com/openvoiceos-bot))

## [0.4.3a3](https://github.com/TigreGotico/wakeforge/tree/0.4.3a3) (2026-09-19)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.3a2...0.4.3a3)

**Merged pull requests:**

- ci: delete python-support.yml, a duplicate Build Tests that never starts [\#58](https://github.com/TigreGotico/wakeforge/pull/58) ([openvoiceos-bot](https://github.com/openvoiceos-bot))

## [0.4.3a2](https://github.com/TigreGotico/wakeforge/tree/0.4.3a2) (2026-09-18)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.3a1...0.4.3a2)

## [0.4.3a1](https://github.com/TigreGotico/wakeforge/tree/0.4.3a1) (2026-09-18)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.2a8...0.4.3a1)

## [0.4.2a8](https://github.com/TigreGotico/wakeforge/tree/0.4.2a8) (2026-09-18)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.2a7...0.4.2a8)

## [0.4.2a7](https://github.com/TigreGotico/wakeforge/tree/0.4.2a7) (2026-09-18)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.2a6...0.4.2a7)

**Merged pull requests:**

- fix\(scripts\): select\_training\_subset.py compiles again, and CI compiles scripts/ [\#56](https://github.com/TigreGotico/wakeforge/pull/56) ([openvoiceos-bot](https://github.com/openvoiceos-bot))
- docs: state the measured ww-benchmarks baseline [\#52](https://github.com/TigreGotico/wakeforge/pull/52) ([openvoiceos-bot](https://github.com/openvoiceos-bot))
- docs: phonmatch.md points at orthography2ipa for the IPA [\#51](https://github.com/TigreGotico/wakeforge/pull/51) ([openvoiceos-bot](https://github.com/openvoiceos-bot))
- docs: model card template for published wake-word models [\#33](https://github.com/TigreGotico/wakeforge/pull/33) ([openvoiceos-bot](https://github.com/openvoiceos-bot))

## [0.4.2a6](https://github.com/TigreGotico/wakeforge/tree/0.4.2a6) (2026-09-17)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.2a5...0.4.2a6)

## [0.4.2a5](https://github.com/TigreGotico/wakeforge/tree/0.4.2a5) (2026-09-17)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.2a4...0.4.2a5)

**Merged pull requests:**

- docs: sweep — verify every code line-reference and strict build across the campaign [\#50](https://github.com/TigreGotico/wakeforge/pull/50) ([openvoiceos-bot](https://github.com/openvoiceos-bot))
- docs: add mkdocs.yml and a reader-path nav [\#49](https://github.com/TigreGotico/wakeforge/pull/49) ([openvoiceos-bot](https://github.com/openvoiceos-bot))

## [0.4.2a4](https://github.com/TigreGotico/wakeforge/tree/0.4.2a4) (2026-09-17)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.2a3...0.4.2a4)

**Merged pull requests:**

- docs: phonmatch.md/enrichment.md — attribute claims, fix stale line refs [\#48](https://github.com/TigreGotico/wakeforge/pull/48) ([openvoiceos-bot](https://github.com/openvoiceos-bot))
- docs: extractors.md/losses.md — attribute comparative claims, fix a fabricated citation [\#47](https://github.com/TigreGotico/wakeforge/pull/47) ([openvoiceos-bot](https://github.com/openvoiceos-bot))

## [0.4.2a3](https://github.com/TigreGotico/wakeforge/tree/0.4.2a3) (2026-09-17)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.2a2...0.4.2a3)

**Merged pull requests:**

- docs: classifiers.md — a paper link and measured/paper-reported tag per head [\#46](https://github.com/TigreGotico/wakeforge/pull/46) ([openvoiceos-bot](https://github.com/openvoiceos-bot))

## [0.4.2a2](https://github.com/TigreGotico/wakeforge/tree/0.4.2a2) (2026-09-16)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.2a1...0.4.2a2)

**Merged pull requests:**

- docs: add the expectations page — what a good wake word looks like [\#45](https://github.com/TigreGotico/wakeforge/pull/45) ([openvoiceos-bot](https://github.com/openvoiceos-bot))
- docs: settle the name as wakeforge, frame it as a research framework [\#44](https://github.com/TigreGotico/wakeforge/pull/44) ([openvoiceos-bot](https://github.com/openvoiceos-bot))

## [0.4.2a1](https://github.com/TigreGotico/wakeforge/tree/0.4.2a1) (2026-09-16)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.1a3...0.4.2a1)

**Merged pull requests:**

- fix\(ci\): turn dev green: deterministic tests and a license check that matches its inputs [\#32](https://github.com/TigreGotico/wakeforge/pull/32) ([openvoiceos-bot](https://github.com/openvoiceos-bot))

## [0.4.1a3](https://github.com/TigreGotico/wakeforge/tree/0.4.1a3) (2026-09-04)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.1a2...0.4.1a3)

**Merged pull requests:**

- chore: rename distribution to wakeforge [\#31](https://github.com/TigreGotico/wakeforge/pull/31) ([JarbasAl](https://github.com/JarbasAl))

## [0.4.1a2](https://github.com/TigreGotico/wakeforge/tree/0.4.1a2) (2026-08-10)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.1a1...0.4.1a2)

**Merged pull requests:**

- chore: add Apache-2.0 LICENSE [\#29](https://github.com/TigreGotico/wakeforge/pull/29) ([JarbasAl](https://github.com/JarbasAl))
- chore: voiceclonnx from PyPI; swap VAD to pure-ONNX vadonnx [\#26](https://github.com/TigreGotico/wakeforge/pull/26) ([JarbasAl](https://github.com/JarbasAl))
- feat\(inference\): true stateful O\(1\) streaming + fix live mic demo [\#25](https://github.com/TigreGotico/wakeforge/pull/25) ([JarbasAl](https://github.com/JarbasAl))
- fix\(cli\): two latent ww\_trainer-train bugs + regression test [\#24](https://github.com/TigreGotico/wakeforge/pull/24) ([JarbasAl](https://github.com/JarbasAl))
- fix: resume/seed/tier bugs, voiceclonnx-only VC, honest tests, notebook docs [\#23](https://github.com/TigreGotico/wakeforge/pull/23) ([JarbasAl](https://github.com/JarbasAl))
- fix: expose losses\_cfg and augmentation folders on QuickstartConfig [\#22](https://github.com/TigreGotico/wakeforge/pull/22) ([JarbasAl](https://github.com/JarbasAl))
- docs\(notebook\): kaggle quickstart rewrite — public user-facing wake-word training entry point [\#20](https://github.com/TigreGotico/wakeforge/pull/20) ([JarbasAl](https://github.com/JarbasAl))
- docs: upfront resource-budget page with real HF sizes [\#16](https://github.com/TigreGotico/wakeforge/pull/16) ([JarbasAl](https://github.com/JarbasAl))

## [0.4.1a1](https://github.com/TigreGotico/wakeforge/tree/0.4.1a1) (2026-05-13)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.0a3...0.4.1a1)

**Merged pull requests:**

- fix\(quickstart\): clearer datagen install + fail-fast errors \(\#12\) [\#13](https://github.com/TigreGotico/wakeforge/pull/13) ([JarbasAl](https://github.com/JarbasAl))

## [0.4.0a3](https://github.com/TigreGotico/wakeforge/tree/0.4.0a3) (2026-05-13)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.0a1...0.4.0a3)

**Merged pull requests:**

- docs: publish-ready overhaul + repo cleanup + CI fixes [\#11](https://github.com/TigreGotico/wakeforge/pull/11) ([JarbasAl](https://github.com/JarbasAl))

## [0.4.0a1](https://github.com/TigreGotico/wakeforge/tree/0.4.0a1) (2026-05-13)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.4.0a2...0.4.0a1)

## [0.4.0a2](https://github.com/TigreGotico/wakeforge/tree/0.4.0a2) (2026-05-13)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.3.0a1...0.4.0a2)

**Merged pull requests:**

- feat: add OCSVMHead — two-stage FFN backbone + One-Class SVM classifier [\#9](https://github.com/TigreGotico/wakeforge/pull/9) ([JarbasAl](https://github.com/JarbasAl))
- Update actions/setup-python action to v6 [\#4](https://github.com/TigreGotico/wakeforge/pull/4) ([renovate[bot]](https://github.com/apps/renovate))
- Update actions/checkout action to v6 [\#3](https://github.com/TigreGotico/wakeforge/pull/3) ([renovate[bot]](https://github.com/apps/renovate))

## [0.3.0a1](https://github.com/TigreGotico/wakeforge/tree/0.3.0a1) (2026-05-13)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.2.0a1...0.3.0a1)

**Merged pull requests:**

- feat: port ConvAttentionHead, AUT metric, and checkpoint averaging from livekit-wakeword [\#10](https://github.com/TigreGotico/wakeforge/pull/10) ([JarbasAl](https://github.com/JarbasAl))

## [0.2.0a1](https://github.com/TigreGotico/wakeforge/tree/0.2.0a1) (2026-04-23)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/0.1.1a1...0.2.0a1)

**Merged pull requests:**

- feat: add text modality — optional audio+text conditioning via OnnxTextExtractor [\#8](https://github.com/TigreGotico/wakeforge/pull/8) ([JarbasAl](https://github.com/JarbasAl))

## [0.1.1a1](https://github.com/TigreGotico/wakeforge/tree/0.1.1a1) (2026-03-20)

[Full Changelog](https://github.com/TigreGotico/wakeforge/compare/8f1a09ff3756ea8aa91009d395961885691a2207...0.1.1a1)



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
