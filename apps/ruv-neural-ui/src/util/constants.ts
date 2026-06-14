/** Mirrors the Rust `SafetyEnvelope::default()` and platform metadata. */
export const SAFETY = {
  maxHrBpm: 100,
  minHrBpm: 45,
  maxArousal: 0.75,
  maxMovementIndex: 0.15,
  divergenceTolerance: 0.15,
  frequencyToleranceHz: 2.0,
  intensityCeiling: 0.6,
};

export const PLATFORM = {
  version: "0.1.0",
  workspaceTests: 398,
  attestations: 51,
  crates: 15,
  gammaHz: 40,
};

export const BOUNDARY =
  "This system is for wellness and cognitive-state research using safe external sensory channels. It is not a medical device. It does not diagnose, treat, cure, or prevent disease.";

/**
 * Disallowed medical-claim phrases (ADR-0014 §3). These are multi-word *claims*
 * — chosen so the not-a-medical-device disclaimer ("does not diagnose, treat,
 * cure, or prevent disease") never trips the lint. Enforced by content-lint.test.
 */
export const DISALLOWED_CLAIMS = [
  "treats alzheimer",
  "treat alzheimer",
  "cures cognitive",
  "cure cognitive decline",
  "brain therapy device",
  "clinical stimulation",
  "targeted treatment",
  "prescription protocol",
  "medical advice",
  "diagnoses disease",
];
