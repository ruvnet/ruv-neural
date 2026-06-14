# ADR-0001 — Scope: research-grade wellness platform, **not** a medical device

## Status

Accepted

## Context

"Targeted brain stimulation for treatment is medical-device territory." Focused
ultrasound, TMS, tDCS/tACS, DBS, and VNS require clinical validation, dosing
controls, contraindication screening, and regulatory review (FDA/CE). Building
any of those here would be irresponsible and out of reach for an open-source
library.

At the same time, **safe external sensory channels** — light, sound, and touch —
can entrain cortical rhythms (e.g. 40 Hz gamma / GENUS) without breaching the
skin or the regulatory boundary, and are already used in consumer wellness and
research contexts.

## Decision

This subsystem is a **research-grade wellness and cognitive-state platform**, not
a disease-treatment device. Concretely:

1. **Only safe external sensory modalities** are implemented: 40 Hz light, audio,
   and haptics (ADR-0002). No transcranial or implanted modality is modeled.
2. The system **measures signal delivery, measures user response, adapts
   conservatively, and produces clean evidence** — it does not diagnose or treat.
3. All clinical/neural-data use remains **gated** on informed consent, IRB/ethics
   approval, and regulatory clearance, consistent with the repository's existing
   Ethics section.
4. The hard boundary is encoded in code: the stimulus crate has **no** API for
   any non-sensory modality, and the controller's safety envelope enforces
   conservative limits with fail-safe stops.

## Consequences

- The product thesis is *closed-loop sensory neuromodulation*, deliverable today.
- We forgo any treatment claim; outputs are wellness/cognitive-state oriented.
- Crate docs and the README cite this ADR as the scope boundary.

## Evidence

- `ruv-neural-stim/src/params.rs` — `Modality` has exactly `{Light, Audio, Haptic}`.
- `ruv-neural-loop` — controller commands only the stim crate.
