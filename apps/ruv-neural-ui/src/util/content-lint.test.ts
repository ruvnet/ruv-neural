import { describe, it, expect } from "vitest";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { DISALLOWED_CLAIMS, BOUNDARY } from "./constants";

/**
 * Content lint (ADR-0014 §3, §17): the UI must never make a medical-treatment
 * claim. This scans the TypeScript/TSX source for the disallowed claim phrases
 * and asserts the not-a-medical-device boundary statement is present.
 */

const SRC = join(dirname(fileURLToPath(import.meta.url)), "..");

function sourceFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const p = join(dir, entry);
    if (statSync(p).isDirectory()) {
      out.push(...sourceFiles(p));
    } else if (
      /\.(ts|tsx)$/.test(entry) &&
      !/\.test\.tsx?$/.test(entry) &&
      // constants.ts legitimately *defines* the disallowed-claim list.
      entry !== "constants.ts"
    ) {
      out.push(p);
    }
  }
  return out;
}

describe("content lint — no medical-treatment claims", () => {
  const files = sourceFiles(SRC);

  it("scans a non-trivial set of source files", () => {
    expect(files.length).toBeGreaterThan(5);
  });

  for (const claim of DISALLOWED_CLAIMS) {
    it(`never uses the disallowed claim "${claim}"`, () => {
      const offenders = files.filter((f) =>
        readFileSync(f, "utf8").toLowerCase().includes(claim),
      );
      expect(offenders, `disallowed claim found in: ${offenders.join(", ")}`).toEqual([]);
    });
  }

  it("keeps the not-a-medical-device boundary statement", () => {
    expect(BOUNDARY.toLowerCase()).toContain("not a medical device");
    expect(BOUNDARY.toLowerCase()).toContain("does not diagnose, treat, cure");
  });
});
