# Model Card — rUv Neural Brain-to-Text (template)

> Fill this out before distributing any trained weights. A model trained on
> SpanishBCBL is **CC BY-NC 4.0** (non-commercial, attribution) — see
> [`WEIGHTS_LICENSE`](WEIGHTS_LICENSE). This is a **research artifact, not a
> medical device**, and makes no clinical or efficacy claim.

## Model details

- **Name / version:** `<name>` `<vX.Y.Z>`
- **Produced by:** `<author/org>` · contact `<email>`
- **Date:** `<YYYY-MM-DD>`
- **Architecture:** `ruv-neural-brain2text` pipeline — preprocess (bandpass +
  resample) → keystroke epoching → acoustic model (`Prototype` | `Linear` |
  `Mlp`) → character n-gram LM → beam search. Record the exact
  `Brain2TextConfig` used (it is embedded in the serialized artifact).
- **Artifact format:** JSON-serialized `TrainedPipeline` (config + acoustic
  weights + LM). Self-contained.
- **Code license:** MIT OR Apache-2.0.
- **Weights license:** `<MIT/Apache if trained on permissive data | CC BY-NC 4.0
  if trained on SpanishBCBL/Brain2Qwerty data>`.

## Intended use

- **In scope:** non-invasive brain-to-text research, benchmarking,
  reproducibility studies, methods development.
- **Out of scope:** clinical/diagnostic use; any deployment implying medical
  efficacy; commercial use of CC BY-NC weights; attempts to re-identify
  individuals.

## Training data

- **Dataset:** `<e.g. SpanishBCBL (DECOMEG)>` ·
  `<https://huggingface.co/datasets/bcbl190626/SpanishBCBL>`
- **Modality:** `<MEG 306-ch | EEG 64-ch>` · **Sample rate:** `<1 kHz>`
- **Subjects / sentences / keystrokes:** `<...>`
- **Data license:** `<CC BY-NC 4.0>`
- **Privacy:** de-identified; directly identifying material (MRI/T1, head-position
  video, eye-tracking, session video) excluded from the public release. Do not
  attempt re-identification.

## Evaluation

- **Split:** `<train/val/test fractions and how split>`
- **Metrics:** Character Error Rate (CER), Word Error Rate (WER).
- **Results:** CER `<...>` · WER `<...>` · per-subject range `<...>`.
- **Baseline compared against:** `<V1 default config / chance>`.

## Limitations & ethics

- MEG-dependent for strong accuracy; EEG is substantially weaker.
- Trained per-corpus; generalization across subjects/sessions is limited.
- Synthetic-data numbers (from the bundled generator) validate the machinery,
  **not** real decoding accuracy — never report them as real performance.
- The acoustic model in this crate is a clean-room stand-in for the deep
  Brain2Qwerty model; for the real model use the (opt-in) Python sidecar.

## Required attribution (if SpanishBCBL/Brain2Qwerty-derived)

- Lévy et al. (2026); Zhang et al. (2025).
- Dataset: https://huggingface.co/datasets/bcbl190626/SpanishBCBL (CC BY-NC 4.0)
- Brain2Qwerty: https://github.com/facebookresearch/brain2qwerty (CC BY-NC 4.0)
