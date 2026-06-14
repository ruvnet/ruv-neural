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
  workspaceTests: 392,
  attestations: 50,
  crates: 15,
  gammaHz: 40,
};

export const BOUNDARY =
  "This system is for wellness and cognitive-state research using safe external sensory channels. It is not a medical device. It does not diagnose, treat, cure, or prevent disease.";
