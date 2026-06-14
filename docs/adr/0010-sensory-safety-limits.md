# ADR-0010 — Photosensitivity & sensory safety limits

## Status

Accepted

## Context

The dominant hazard of visual flicker is **photosensitive (reflex) seizure**
provocation; for audio it is sound-pressure (hearing) exposure; for haptics it is
actuator over-drive. International consensus (Harding/Fisher et al.) puts the most
provocative full-field, high-contrast flicker at roughly **15–25 Hz**, with risk
across ~3–60 Hz. 40 Hz GENUS sits above the peak but still inside the cautionary
band, so it cannot be treated as automatically safe.

## Decision

`SensorySafetyLimits` enforces conservative, modality-specific guardrails that the
controller can never override:

- **Intensity ceiling** (`max_intensity`, default 0.6) on every modality.
- **Photic caution band** (3–60 Hz): light is **refused** unless a
  `photosensitivity_screen_cleared` flag is set, and even then luminance contrast
  is capped (`max_photic_contrast`, default 0.5).
- **Audio SPL** capped (`max_audio_db_spl`, default 75 dB) via a linear-headroom
  projection.
- A **zero-intensity** stimulus emits nothing physical and is always safe (the
  canonical safe-stop / disabled-channel case).

`check()` rejects unsafe requests (strict mode); `clamp()` limits them into the
safe region (conservative dosing), forcing unscreened light to zero rather than
aborting the whole session.

## Consequences

- Light is opt-in and contraindication-gated; audio/haptic are the defaults.
- Safety is enforced at synthesis time, below the controller, so no protocol can
  bypass it.
- Limits are explicit defaults, easy to review and to tighten per deployment.

## Evidence

- `ruv-neural-stim/src/safety.rs`
- Tests: `safety_blocks_unscreened_light_in_caution_band`, `clamp_disables_unscreened_light`,
  `safety_caps_loud_audio`.
