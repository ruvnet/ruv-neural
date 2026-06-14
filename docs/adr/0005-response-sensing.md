# ADR-0005 — Physiological response sensing

## Status

Accepted

## Context

Closing the loop means measuring the subject's response. The task lists HRV,
breathing, sleep, and motion as response-sensing channels. These are cheap,
non-invasive, and reflect autonomic state and arousal — exactly what a wellness
neuromodulation loop needs to titrate against. (Neural response, e.g. the 40 Hz
SSVEP/ASSR entrainment index, is supplied optionally by the existing topology
pipeline.)

## Decision

`ruv-neural-biosense` turns raw peripheral biosignals into compact metrics:

- **HRV** from RR intervals: SDNN, RMSSD, pNN50 (Task-Force definitions) plus a
  dependency-free Goertzel LF/HF estimate and a normalized vagal-tone proxy.
- **Respiration** from a breathing waveform: rate, depth, regularity, and a
  resonance-breathing "calm index."
- **Motion** from accelerometry: movement index and stillness fraction.
- **Sleep** as a transparent HR+motion **proxy** (explicitly *not* PSG-grade),
  reusing the core `SleepStage` enum, used only to gate stimulation.
- **Fusion** (`PhysioMetrics`): arousal and relaxation indices that degrade
  gracefully when a channel is missing.

A deterministic `PhysioSimulator` generates windows consistent with a target
arousal so the loop is testable without hardware.

## Consequences

- The controller depends only on `PhysioMetrics`, decoupling it from sensors.
- Missing channels lower confidence but never crash the loop.
- The sleep proxy is honest about its limits and only ever *inhibits* stimulation.

## Evidence

- `ruv-neural-biosense/src/{hrv,respiration,motion,sleep,physio,simulator}.rs`
- Tests: `physio_fusion_calm_vs_aroused`, `hrv_high_variability_has_higher_rmssd`.
