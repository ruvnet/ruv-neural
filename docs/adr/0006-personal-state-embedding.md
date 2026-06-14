# ADR-0006 — Personal state embedding (ruVector)

## Status

Accepted

## Context

"Right signal, right time, right intensity, **right person**." Personalization
needs a per-person representation of state that fuses neural and physiological
signals into something comparable across time and against targets — the
ruVector personal state embedding.

## Decision

`PersonalStateEmbedding` is a fixed 9-dimensional `[0,1]` vector fusing:

```
[ arousal, relaxation, vagal_tone, hr_norm, resp_calm,
  stillness, gamma_index, alpha_index, connectivity ]
```

Missing neural features fall back to a neutral prior so the vector is always
full-dimensional. It exports cleanly to the core `NeuralEmbedding` (method tag
`personal-state-fusion`) for RVF storage and offline analysis.

A `PersonalBaseline` maintains an online (Welford) per-feature mean/variance so
live observations can be **z-scored against the person's own baseline**, and a
single `deviation()` scalar reports how far the current state is from baseline —
the substrate for personalized dosing and anomaly detection.

## Consequences

- One comparable state vector across simulator, recordings, and live use.
- Personalization is built in via the rolling baseline, not bolted on.
- Reuses the existing embedding/RVF ecosystem rather than inventing a new format.

## Evidence

- `ruv-neural-loop/src/embedding.rs`
- Tests: `personal_embedding_dimension_and_export`, `baseline_deviation_grows_with_change`.
