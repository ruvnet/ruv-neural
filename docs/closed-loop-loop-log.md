# Closed-Loop Neuromodulation — `/loop` iteration log

A record of the iterative develop → test → optimize → validate cycles used to
bring the closed-loop sensory neuromodulation subsystem to a fully implemented,
tested, and validated state. Each iteration is a focused pass with a concrete
exit check; the loop ran until the acceptance criterion (ADR-0011) passed
deterministically across modalities and the full workspace stayed green.

| # | Focus | Outcome | Validation |
|---|-------|---------|------------|
| 1 | **Architecture & boundary.** Studied the existing 12-crate pipeline; fixed scope to safe external sensory channels only (ADR-0001). Designed three new crates: `stim`, `biosense`, `loop`. | Crate skeletons + workspace wiring. | `cargo build` of empty crates. |
| 2 | **Stimulus synthesis.** 40 Hz light/audio/haptic envelope model, ramps, per-modality presets (ADR-0002). | `StimulusParams`, `StimulusWaveform`. | Waveform unit tests. |
| 3 | **Sensory safety.** Photosensitivity caution band, audio SPL, intensity ceilings; strict vs. clamped modes (ADR-0010). | `SensorySafetyLimits`, `StimulusGenerator`. | Safety unit tests. |
| 4 | **Verified delivery.** SHA-256 waveform binding + empirical envelope-frequency measurement → `verified` receipts (ADR-0004). | `DeliveryReceipt`. | `receipt_binds_to_waveform`. |
| 5 | **Response sensing.** HRV (SDNN/RMSSD/pNN50/LF-HF), respiration, motion, sleep proxy, autonomic fusion (ADR-0005). | `ruv-neural-biosense`. | 12 biosense tests. |
| 6 | **Personal state embedding.** 9-D ruVector fusion + online personal baseline; export to core `NeuralEmbedding`/RVF (ADR-0006). | `PersonalStateEmbedding`, `PersonalBaseline`. | Embedding/baseline tests. |
| 7 | **Closed-loop controller + safety envelope.** State machine, divergence-aware envelope, conservative dosing, hash-chained audit (ADR-0003/0007/0008/0009). | `ClosedLoopController`, `LoopSimulation`. | Controller end-to-end tests. |
| 8 | **Bug hunt.** Found & fixed: (a) envelope-frequency estimator biased by onset ramps; (b) zero-intensity light wrongly rejected by `check()`; (c) audio preset violated Nyquist at the default rate. | Correct, internally consistent presets. | All stim/biosense/loop tests pass. |
| 9 | **Stability optimization.** Raw per-window noise caused dose chatter and spurious safe-stops. Added EMA feedback smoothing + worsen-threshold back-off (ADR-0012). | Stable convergence; real divergence still stops. | Converge & safe-stop tests deterministic. |
| 10 | **Acceptance, integration & evidence.** Executable acceptance gate (ADR-0011), CLI `neuromod` command, witness attestations (+9 capabilities), ADR set, README. | End-to-end demos + signed sessions. | **392 workspace tests green**; acceptance PASS across modalities. |

## Exit criterion (met)

- `SessionReport::passes_acceptance()` is `true` for converging **and**
  safe-stopped sessions, across `audio-haptic` and `multimodal` protocols.
- Every delivered stimulus is cryptographically verified; every session carries an
  intact, Ed25519-attestable audit chain.
- `cargo test --workspace` → **392 passed, 0 failed.**

## What "SOTA for this scope" means here

This is deliberately **not** a treatment device (ADR-0001). Within the
research-grade wellness scope, the subsystem implements the full frontier loop —
*detect state → deliver verified stimulus → measure response → adapt
conservatively → stop safely → produce signed evidence* — which is exactly the
"closed-loop state control" row the task identified as the important frontier.
Further work (real-hardware drivers, clinical validation, richer neural decoding)
is gated on consent/IRB/regulatory review and is out of scope by design.
