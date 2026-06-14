import { describe, it, expect } from "vitest";
import { BundleSchema, parseBundle } from "../schemas/evidence";
import { verifyBundle, canonicalPayload, sha256Hex } from "./verify";
import relaxed from "../fixtures/relaxed.json";
import safestop from "../fixtures/relaxed-safestop.json";
import gamma from "../fixtures/gamma.json";

describe("evidence schema", () => {
  it("parses every bundled fixture", () => {
    for (const raw of [relaxed, safestop, gamma]) {
      expect(() => BundleSchema.parse(raw)).not.toThrow();
    }
  });

  it("rejects malformed JSON", () => {
    expect(() => parseBundle("{not json")).toThrow(/Invalid JSON/);
  });

  it("rejects a structurally invalid bundle", () => {
    expect(() => parseBundle(JSON.stringify({ schemaVersion: "ruflo-evidence/1" }))).toThrow(
      /Schema validation failed/,
    );
  });
});

describe("sha256 / canonical payload", () => {
  it("computes a known SHA-256", () => {
    // echo -n "" | sha256sum
    expect(sha256Hex("")).toBe(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
  });

  it("recomputes a fixture step's stored payload hash", () => {
    const b = BundleSchema.parse(relaxed);
    const step = b.steps[0];
    expect(sha256Hex(canonicalPayload(step))).toBe(step.payloadSha256);
  });
});

describe("verifyBundle", () => {
  it("verifies a converging session end-to-end", () => {
    const b = BundleSchema.parse(relaxed);
    const result = verifyBundle(b);
    expect(result.ok).toBe(true);
    expect(result.checks.find((c) => c.id === "chain")?.ok).toBe(true);
    expect(result.checks.find((c) => c.id === "acceptance")?.ok).toBe(true);
    expect(result.checks.find((c) => c.id === "receipts")?.ok).toBe(true);
    expect(result.checks.find((c) => c.id === "signature")?.ok).toBe(true);
  });

  it("verifies a safe-stopped session and its breach", () => {
    const b = BundleSchema.parse(safestop);
    const result = verifyBundle(b);
    expect(result.ok).toBe(true);
    expect(b.acceptance.stoppedSafelyOutsideEnvelope).toBe(true);
    const last = b.steps[b.steps.length - 1];
    expect(last.envelope.within).toBe(false);
    expect(last.intensity).toBe(0);
    expect(result.checks.find((c) => c.id === "safeStop")?.ok).toBe(true);
  });

  it("detects a tampered step (broken hash chain)", () => {
    const b = BundleSchema.parse(relaxed);
    b.steps[2].intensity += 0.2; // tamper
    const result = verifyBundle(b);
    expect(result.ok).toBe(false);
    expect(result.checks.find((c) => c.id === "chain")?.ok).toBe(false);
  });

  it("detects a tampered signature", () => {
    const b = BundleSchema.parse(relaxed);
    if (b.signature) {
      b.signature.signature = b.signature.signature.replace(/^.{2}/, "00");
      const result = verifyBundle(b);
      expect(result.checks.find((c) => c.id === "signature")?.ok).toBe(false);
    }
  });
});
