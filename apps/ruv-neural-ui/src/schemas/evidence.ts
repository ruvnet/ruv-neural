import { z } from "zod";

/**
 * Zod schemas mirroring the Rust `EvidenceBundle` (`ruflo-evidence/1`).
 *
 * All imported reports are validated against these schemas and then treated as
 * untrusted *data* — never executed (ADR-0014 §13). The bundle's own fields are
 * camelCase; the embedded `report` is the Rust `SessionReport` in snake_case.
 */

export const SCHEMA_VERSION = "ruflo-evidence/1";

const optNum = z.number().nullable().optional();

export const ReceiptSchema = z.object({
  modality: z.enum(["light", "audio", "haptic"]),
  intendedFrequencyHz: z.number(),
  measuredFrequencyHz: z.number(),
  frequencyErrorHz: z.number(),
  dutyCycle: z.number(),
  intensity: z.number(),
  waveformSha256: z.string(),
  verified: z.boolean(),
});
export type Receipt = z.infer<typeof ReceiptSchema>;

export const BiosenseSchema = z.object({
  heartRateBpm: optNum,
  sdnnMs: optNum,
  rmssdMs: optNum,
  pnn50: optNum,
  lfHfRatio: optNum,
  respirationRateBpm: optNum,
  motionIndex: optNum,
  stillness: optNum,
  arousalScore: z.number(),
  relaxationScore: z.number(),
});
export type Biosense = z.infer<typeof BiosenseSchema>;

export const EnvelopeSchema = z.object({
  within: z.boolean(),
  breaches: z.array(z.string()),
});
export type Envelope = z.infer<typeof EnvelopeSchema>;

export const StepSchema = z.object({
  index: z.number(),
  timestampS: z.number(),
  phase: z.string(),
  auditKind: z.string(),
  distanceToTarget: z.number(),
  intensity: z.number(),
  embedding: z.array(z.number()),
  featureNames: z.array(z.string()),
  biosense: BiosenseSchema,
  receipts: z.array(ReceiptSchema),
  envelope: EnvelopeSchema,
  payloadSha256: z.string(),
  prevHash: z.string(),
  hash: z.string(),
});
export type Step = z.infer<typeof StepSchema>;

export const AcceptanceSchema = z.object({
  targetStateIdentified: z.boolean(),
  verifiedStimulusDelivered: z.boolean(),
  responseMeasured: z.boolean(),
  stoppedSafelyOutsideEnvelope: z.boolean(),
  goalReached: z.boolean(),
  passed: z.boolean(),
});
export type Acceptance = z.infer<typeof AcceptanceSchema>;

export const SignatureSchema = z.object({
  headHash: z.string(),
  signature: z.string(),
  publicKey: z.string(),
});
export type SignatureBlock = z.infer<typeof SignatureSchema>;

/** The embedded Rust `SessionReport` (snake_case); kept permissive. */
export const ReportSchema = z
  .object({
    protocol: z.string(),
    final_phase: z.string(),
    total_steps: z.number(),
    baseline_steps: z.number().optional(),
    stimulate_steps: z.number(),
    total_stimulation_s: z.number(),
    peak_intensity: z.number(),
    goal_reached: z.boolean(),
    safe_stopped: z.boolean(),
    stop_reasons: z.array(z.unknown()).default([]),
    best_distance: z.number(),
    final_distance: z.number(),
    num_receipts: z.number(),
    all_receipts_verified: z.boolean(),
    audit_head_hash: z.string(),
    audit_records: z.number(),
    audit_chain_valid: z.boolean(),
  })
  .passthrough();
export type Report = z.infer<typeof ReportSchema>;

export const BundleSchema = z.object({
  schemaVersion: z.literal(SCHEMA_VERSION),
  sessionId: z.string(),
  createdAt: z.string(),
  mode: z.string(),
  targetState: z.string(),
  protocol: z.string(),
  steps: z.array(StepSchema),
  acceptance: AcceptanceSchema,
  report: ReportSchema,
  bundleChainHead: z.string(),
  auditHeadHash: z.string(),
  auditRecords: z.number(),
  auditChainValid: z.boolean(),
  signature: SignatureSchema.nullable().optional(),
});
export type EvidenceBundle = z.infer<typeof BundleSchema>;

/** Parse + validate untrusted JSON text into a bundle, or throw with a message. */
export function parseBundle(text: string): EvidenceBundle {
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch {
    throw new Error("Invalid JSON: the file is not parseable.");
  }
  const result = BundleSchema.safeParse(json);
  if (!result.success) {
    const first = result.error.issues[0];
    throw new Error(
      `Schema validation failed at ${first.path.join(".") || "<root>"}: ${first.message}`,
    );
  }
  return result.data;
}
