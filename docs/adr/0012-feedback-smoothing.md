# ADR-0012 — Feedback smoothing & divergence detection

## Status

Accepted

## Context

Per-window physiology is noisy. Early iterations of the controller titrated and
ran divergence checks on the *raw* per-step distance-to-target. The result was
two failure modes: (a) **dose chattering** — the dose backed off on every noisy
uptick, stalling convergence; and (b) **spurious fail-safe stops** — a single
noisy sample crossed the divergence tolerance and halted a healthy session.

## Decision

Low-pass the feedback before any decision. The controller maintains an
exponential moving average of the distance-to-target
(`smoothed = α·raw + (1−α)·prev`, `α = 0.5`) and uses the **smoothed** distance
for titration, completion, and divergence detection alike. The dosing rule
additionally only backs off on changes beyond a `worsen_threshold`, treating
small fluctuations as a plateau to climb through (ADR-0008).

Genuine divergence — e.g. an injected arousal spike — moves the smoothed distance
well beyond tolerance and still trips the fail-safe stop promptly; the smoothing
filters noise, not signal.

## Consequences

- Robust convergence on noisy simulated and (expected) real data.
- Fail-safe stops fire on real divergence, not on sensor noise.
- Standard, defensible closed-loop practice (filter the feedback path).

## Evidence

- `ruv-neural-loop/src/controller.rs` (EMA smoothing)
- Tests: `controller_reaches_target_and_delivers_verified_stimuli` (stable
  convergence) and `controller_safe_stops_on_perturbation` (real divergence still
  stops).
