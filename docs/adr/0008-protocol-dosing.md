# ADR-0008 — Protocol selection & conservative dosing

## Status

Accepted

## Context

"Personalized neural dosing… adapt conservatively." We must decide *which*
modalities to drive and at *what* intensity, and the adaptation must be safe:
quick to back off, slow to escalate.

## Decision

A `Protocol` trait maps `(target, estimate, history) → StimulusPlan`. The
default `GammaEntrainmentProtocol` drives selected modalities at the target's
envelope frequency (40 Hz for entrainment, ~10 Hz alpha for relaxation) and
applies a conservative `DosingPolicy`:

- start low (`start_intensity = 0.15`);
- **titrate up gently** (`step_up = 0.05`) while the response is improving or on
  a noisy plateau;
- **back off fast** (`step_down = 0.10`) only when the response is *clearly*
  worsening (beyond `worsen_threshold`, i.e. not noise);
- never exceed `ceiling = 0.5` (and always subject to the safety clamp);
- hold, not escalate, once at target.

Genuine divergence is the safety envelope's job (ADR-0007); the dosing rule
handles ordinary titration. Modalities that cannot be delivered safely (e.g.
unscreened light) are skipped for the step while the others still drive.

## Consequences

- "Up gently, retreat fast" is encoded in asymmetric step sizes.
- The protocol is pluggable; new paradigms implement `Protocol`.
- Dosing and safety have clean, separate responsibilities.

## Evidence

- `ruv-neural-loop/src/protocol.rs`
- Tests: `controller_never_exceeds_safety_intensity`.
