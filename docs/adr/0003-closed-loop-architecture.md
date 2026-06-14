# ADR-0003 — Closed-loop control architecture

## Status

Accepted

## Context

Open-loop "bathe the brain in a rhythm" stimulation ignores the subject's state.
The valuable, frontier idea is **closed-loop state control**: detect the state,
stimulate only when needed, measure the response, and stop safely. This requires
a control loop that binds stimulus generation to response sensing with safety in
the middle of every cycle.

## Decision

Implement a deterministic state-machine controller (`ClosedLoopController`,
"Ruflo") whose `step(observation)` runs one loop iteration:

```
observe → embed (ruVector) → estimate state → SAFETY ENVELOPE
                                                  │
                         within ─────────────────┴──────── breach
                            │                                  │
                  select protocol & dose                 fail-safe STOP
                            │                              (intensity 0)
                  deliver verified stimulus
                            │
                  audit (hash-chained)
```

Phases: `Baselining → Stimulating ↔ Holding → {Completed | SafeStopped | Aborted}`.
Terminal phases are absorbing. Safety is evaluated **before** any stimulation
decision on every step, so a breach can never be out-voted by the protocol.

The controller is decoupled from hardware: it consumes `StateObservation`
(physiology + optional neural features) and emits `VerifiedStimulus` objects,
so it runs identically against a simulator, recorded data, or live sensors.

## Consequences

- Deterministic and unit-testable end to end (`LoopSimulation`).
- Safety is structurally first-class, not an afterthought.
- The same controller drives the acceptance test and the CLI demo.

## Evidence

- `ruv-neural-loop/src/controller.rs`, `sim.rs`
- Tests: `controller_reaches_target_and_delivers_verified_stimuli`.
