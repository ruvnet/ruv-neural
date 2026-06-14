import { describe, it, expect } from "vitest";
import { BundleSchema } from "../schemas/evidence";
import { canonicalPayload, sha256Hex, verifyBundle } from "./verify";
import relaxed from "../fixtures/relaxed.json";
import safestop from "../fixtures/relaxed-safestop.json";
import focused from "../fixtures/focused.json";
import gamma from "../fixtures/gamma.json";

/**
 * Cross-language verifier parity (ADR-0014 §6, Phase 3). The committed fixtures
 * carry the **Rust-computed** `payloadSha256` for every step. Re-deriving each
 * hash in TypeScript and matching it — across the whole corpus — proves the
 * browser verifier and the Rust reference verifier agree byte-for-byte. The
 * Rust side is exercised by `cli::commands::verify_bundle` and
 * `loop::evidence` tests.
 */

const FIXTURES = { relaxed, safestop, focused, gamma };

describe("Rust↔TS canonical-hash parity (whole corpus)", () => {
  for (const [name, raw] of Object.entries(FIXTURES)) {
    const bundle = BundleSchema.parse(raw);

    it(`${name}: every step's payload hash recomputes identically`, () => {
      expect(bundle.steps.length).toBeGreaterThan(0);
      for (const step of bundle.steps) {
        expect(sha256Hex(canonicalPayload(step))).toBe(step.payloadSha256);
      }
    });

    it(`${name}: the recomputed chain head matches the Rust head`, () => {
      const GENESIS = "0".repeat(64);
      let prev = GENESIS;
      for (const step of bundle.steps) {
        prev = sha256Hex(prev + step.payloadSha256);
      }
      expect(prev).toBe(bundle.bundleChainHead);
    });

    it(`${name}: full verification agrees with the bundle verdict`, () => {
      const result = verifyBundle(bundle);
      expect(result.ok).toBe(true);
      expect(result.checks.find((c) => c.id === "acceptance")?.ok).toBe(true);
    });
  }
});
