import { sha256 } from "@noble/hashes/sha256";
import { bytesToHex, hexToBytes, utf8ToBytes } from "@noble/hashes/utils";
import { ed25519 } from "@noble/curves/ed25519";
import type { EvidenceBundle, Step } from "../schemas/evidence";

/**
 * Local, in-browser verification of a Ruflo evidence bundle (ADR-0014 §10).
 *
 * Nothing here trusts the bundle's self-reported flags: the per-step hash chain
 * is **recomputed** from the same fixed-precision canonical strings the Rust
 * exporter uses, so a matching head proves integrity rather than asserting it.
 */

/** Matches the Rust `ENVELOPE_TOLERANCE_HZ`. */
export const FREQUENCY_TOLERANCE_HZ = 2.0;

const GENESIS = "0".repeat(64);

export interface Check {
  id: string;
  label: string;
  ok: boolean;
  detail: string;
}

export interface VerificationResult {
  ok: boolean;
  checks: Check[];
}

/** SHA-256 of a UTF-8 string, lowercase hex. */
export function sha256Hex(text: string): string {
  return bytesToHex(sha256(utf8ToBytes(text)));
}

/** The canonical per-step payload string — must mirror Rust `canonical_payload`. */
export function canonicalPayload(step: Step): string {
  const receiptHashes = step.receipts.map((r) => r.waveformSha256).join(",");
  return [
    step.index,
    step.timestampS.toFixed(3),
    step.phase,
    step.distanceToTarget.toFixed(6),
    step.intensity.toFixed(6),
    receiptHashes,
  ].join("|");
}

function verifyChain(bundle: EvidenceBundle): Check {
  let prev = GENESIS;
  for (const step of bundle.steps) {
    if (step.prevHash !== prev) {
      return chain(false, `step ${step.index}: prevHash does not link to the chain`);
    }
    const payloadHash = sha256Hex(canonicalPayload(step));
    if (payloadHash !== step.payloadSha256) {
      return chain(false, `step ${step.index}: payload hash mismatch (recomputed ≠ stored)`);
    }
    const linkHash = sha256Hex(prev + payloadHash);
    if (linkHash !== step.hash) {
      return chain(false, `step ${step.index}: chain hash mismatch`);
    }
    prev = step.hash;
  }
  if (prev !== bundle.bundleChainHead) {
    return chain(false, "chain head does not match bundleChainHead");
  }
  return chain(true, `${bundle.steps.length} steps link cleanly to the head`);

  function chain(ok: boolean, detail: string): Check {
    return { id: "chain", label: "Audit hash chain", ok, detail };
  }
}

function verifyReceipts(bundle: EvidenceBundle): Check {
  let total = 0;
  let bad = 0;
  for (const step of bundle.steps) {
    for (const r of step.receipts) {
      total += 1;
      const hashOk = /^[0-9a-f]{64}$/.test(r.waveformSha256);
      const freqOk = r.frequencyErrorHz <= FREQUENCY_TOLERANCE_HZ;
      // verified must imply both a present hash and in-tolerance frequency.
      if (!hashOk || (r.verified && !freqOk) || (r.verified && !hashOk)) bad += 1;
    }
  }
  const ok = total >= 1 && bad === 0;
  return {
    id: "receipts",
    label: "Stimulus delivery receipts",
    ok,
    detail:
      total === 0
        ? "no stimulus receipts in this session"
        : `${total} receipt(s) verified, ${bad} invalid (tolerance ±${FREQUENCY_TOLERANCE_HZ} Hz)`,
  };
}

function verifyStepData(bundle: EvidenceBundle): Check {
  let bad = 0;
  for (const step of bundle.steps) {
    if (step.embedding.length !== step.featureNames.length || step.embedding.length === 0) bad += 1;
  }
  const ok = bundle.steps.length >= 1 && bad === 0;
  return {
    id: "stepData",
    label: "Per-step state embedding & biosense",
    ok,
    detail: ok
      ? `${bundle.steps.length} steps carry a ${bundle.steps[0]?.embedding.length}-D ruVector + biosense`
      : "some steps are missing embedding/biosense data",
  };
}

function verifySafeStop(bundle: EvidenceBundle): Check {
  const stops = bundle.steps.filter(
    (s) => s.phase === "SafeStopped" || s.auditKind === "safe_stop",
  );
  const violating = stops.filter((s) => s.intensity !== 0 || s.receipts.length > 0);
  const ok = violating.length === 0;
  return {
    id: "safeStop",
    label: "Fail-safe stop forces zero intensity",
    ok,
    detail:
      stops.length === 0
        ? "no safe-stop in this session (target reached)"
        : `${stops.length} safe-stop step(s); all command zero intensity`,
  };
}

function verifySignature(bundle: EvidenceBundle): Check | null {
  if (!bundle.signature) return null;
  const { headHash, signature, publicKey } = bundle.signature;
  try {
    if (headHash !== bundle.bundleChainHead) {
      return sig(false, "signed head hash ≠ bundle chain head");
    }
    const ok = ed25519.verify(hexToBytes(signature), utf8ToBytes(headHash), hexToBytes(publicKey));
    return sig(ok, ok ? "Ed25519 signature verifies against the included public key" : "signature does not verify");
  } catch (e) {
    return sig(false, `signature error: ${(e as Error).message}`);
  }
  function sig(ok: boolean, detail: string): Check {
    return { id: "signature", label: "Ed25519 session attestation", ok, detail };
  }
}

function verifyAcceptance(bundle: EvidenceBundle): Check {
  const a = bundle.acceptance;
  // The platform rule (a converged OR a safe-stopped session both pass).
  const recomputed =
    a.verifiedStimulusDelivered &&
    a.responseMeasured &&
    (a.stoppedSafelyOutsideEnvelope || a.goalReached) &&
    bundle.auditChainValid;
  const ok = recomputed === a.passed && a.passed;
  return {
    id: "acceptance",
    label: "Closed-loop acceptance test",
    ok,
    detail: a.passed
      ? a.goalReached
        ? "PASS — target reached"
        : "PASS — safe-stopped outside envelope"
      : "did not pass acceptance",
  };
}

/** Verify a bundle end-to-end, returning every check and an overall verdict. */
export function verifyBundle(bundle: EvidenceBundle): VerificationResult {
  const checks: Check[] = [
    { id: "schema", label: "Schema validity", ok: true, detail: `valid ${bundle.schemaVersion}` },
    verifyChain(bundle),
    verifyReceipts(bundle),
    verifyStepData(bundle),
    verifySafeStop(bundle),
    verifyAcceptance(bundle),
  ];
  const sig = verifySignature(bundle);
  if (sig) checks.push(sig);

  return { ok: checks.every((c) => c.ok), checks };
}
