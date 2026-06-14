# ADR-0004 — Stimulus verification & delivery receipts

## Status

Accepted

## Context

The acceptance test requires the system to **"deliver a *verified* stimulus."**
A command is not evidence of delivery; we need to confirm that the realized
waveform matches what was asked for, and to bind that confirmation to immutable
evidence.

## Decision

Every synthesized stimulus produces a `DeliveryReceipt` that:

1. **Binds to the waveform** via a SHA-256 digest of the emitted samples. Any
   later edit to the waveform breaks `receipt.matches(&waveform)`.
2. **Empirically measures** the realized entrainment frequency
   (`measured_envelope_hz`) by demodulating the envelope and counting mean
   crossings in the steady-state interior (excluding the ramp), independent of
   the requested parameter.
3. Records RMS / peak amplitude and start/end times.
4. Sets `verified = true` only when the measured envelope is within
   `ENVELOPE_TOLERANCE_HZ` (2 Hz) of the command **and** the peak is within unit
   range. A zero-intensity (disabled / safe-stop) stimulus is a legitimate
   verified no-op.

## Consequences

- "Verified stimulus" is a concrete, checkable property, not a claim.
- Receipt hashes feed the audit trail (ADR-0009), linking every step to exactly
  what was delivered.
- The frequency check is robust to onset ramps and to the audio carrier.

## Evidence

- `ruv-neural-stim/src/receipt.rs`, `waveform.rs`
- Tests: `receipt_binds_to_waveform`, `zero_intensity_safe_stop_is_verified_noop`.
