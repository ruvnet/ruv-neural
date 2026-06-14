# ADR-0007 — Safety envelope & fail-safe stop

## Status

Accepted

## Context

The acceptance test's fourth clause: **"stop safely when response moves outside
the allowed envelope."** We need a precise definition of "the allowed envelope"
and a guarantee that leaving it halts stimulation.

## Decision

`SafetyEnvelope` encodes two complementary notions and is evaluated on **every**
control step **before** any stimulation decision:

1. **Absolute bounds** — hard physiological limits that must never be exceeded:
   heart-rate ceiling/floor, arousal ceiling, motion ceiling, and protected
   (deep) sleep.
2. **Response divergence** — the (smoothed) distance-to-target must not rise more
   than `divergence_tolerance` above the best distance achieved this session.
   This catches a response that is drifting *away* from the goal even when every
   individual reading is benign.

Any breach returns `EnvelopeStatus::Breach(reasons)`, which forces the controller
to phase `SafeStopped`, command **zero intensity**, and record the reasons in the
audit trail. `SafeStopped` is terminal — the loop does not silently resume.

## Consequences

- Fail-safe is the default: stimulation continues only while *proved* safe.
- Divergence detection makes "outside the envelope" meaningful beyond raw limits.
- Every stop is explainable (typed `BreachReason`s) and audited.

## Evidence

- `ruv-neural-loop/src/envelope.rs`, `controller.rs`
- Tests: `controller_safe_stops_on_perturbation`,
  `acceptance_safe_stop_outside_envelope`.
