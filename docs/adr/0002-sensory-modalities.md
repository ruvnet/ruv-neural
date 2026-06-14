# ADR-0002 — Safe external sensory modalities (40 Hz light / audio / haptic)

## Status

Accepted

## Context

Gamma ENtrainment Using Sensory stimulation (GENUS) drives cortical 40 Hz
activity through the senses. The three practical, low-risk channels are:

| Channel | Cortical target | Mechanism |
|---------|-----------------|-----------|
| Light   | visual cortex   | luminance flicker at 40 Hz |
| Audio   | auditory cortex | 40 Hz amplitude-modulated tone / click train |
| Haptic  | somatosensory   | 40 Hz vibrotactile drive |

We want a single, uniform stimulus model across all three so the controller can
mix modalities and the verification/safety machinery is shared.

## Decision

Model every stimulus as a **modulation envelope at the entrainment frequency**,
optionally riding a carrier:

- **Light / haptic**: a unipolar `[0,1]` envelope *is* the drive signal.
- **Audio**: a bipolar carrier (default 1 kHz) amplitude-modulated by the `[0,1]`
  envelope, because auditory cortex follows the **envelope**, not the carrier.

Envelope shape is `Sine` (smooth) or `Square` (canonical GENUS light flicker,
duty-cycle configurable). A symmetric linear ramp avoids onset/offset transients.
The 40 Hz preset (`StimulusParams::gamma_40hz`) chooses sensible per-modality
defaults (carrier, sample rate, shape).

## Consequences

- One waveform type, one receipt type, one safety check for all modalities.
- Audio requires a higher sample rate (Nyquist for the carrier); the preset sets
  it automatically so presets are always internally valid.
- Multi-modal protocols are trivially expressible as a set of `StimulusParams`.

## Evidence

- `ruv-neural-stim/src/params.rs`, `waveform.rs`
- Tests: `waveform_measures_40hz_envelope_{audio,haptic}`.
