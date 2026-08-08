import { describe, expect, it } from "vitest";

import { DEFAULT_DFA_CONFIG, dfa } from "./dfa";

/**
 * Deterministic uniform pseudo-random numbers in (-0.5, 0.5).
 * Mirrors the 64-bit LCG used by the Rust tests in
 * `ruv-neural-signal/src/lrtc.rs` exactly (same seed => same stream).
 */
class Lcg {
  private state: bigint;
  private static readonly MASK = (1n << 64n) - 1n;

  constructor(seed: number) {
    this.state = BigInt(seed);
  }

  next(): number {
    this.state = (this.state * 6364136223846793005n + 1442695040888963407n) & Lcg.MASK;
    return Number(this.state >> 11n) / 2 ** 53 - 0.5;
  }
}

function whiteNoise(n: number, seed: number): Float64Array {
  const rng = new Lcg(seed);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = rng.next();
  }
  return out;
}

function brownian(n: number, seed: number): Float64Array {
  const noise = whiteNoise(n, seed);
  const out = new Float64Array(n);
  let acc = 0;
  for (let i = 0; i < n; i++) {
    acc += noise[i];
    out[i] = acc;
  }
  return out;
}

describe("dfa", () => {
  it("estimates alpha ~ 0.5 for white noise", () => {
    const result = dfa(whiteNoise(8192, 42));
    expect(result).not.toBeNull();
    expect(result!.alpha).toBeGreaterThan(0.4);
    expect(result!.alpha).toBeLessThan(0.6);
    expect(result!.rSquared).toBeGreaterThan(0.95);
    expect(result!.hurst).toBe(result!.alpha);
  });

  it("estimates alpha ~ 1.5 for Brownian motion", () => {
    const result = dfa(brownian(8192, 42));
    expect(result).not.toBeNull();
    expect(result!.alpha).toBeGreaterThan(1.35);
    expect(result!.alpha).toBeLessThan(1.65);
    expect(result!.hurst).toBeCloseTo(result!.alpha - 1, 12);
  });

  it("produces fluctuations that grow with scale", () => {
    const result = dfa(brownian(4096, 3));
    expect(result).not.toBeNull();
    for (let i = 1; i < result!.fluctuations.length; i++) {
      expect(result!.fluctuations[i]).toBeGreaterThan(result!.fluctuations[i - 1]);
    }
  });

  it("rejects degenerate input", () => {
    expect(dfa([])).toBeNull();
    expect(dfa(whiteNoise(8, 1))).toBeNull();
    expect(dfa(new Float64Array(4096).fill(1))).toBeNull();
    const withNan = whiteNoise(4096, 1);
    withNan[100] = Number.NaN;
    expect(dfa(withNan)).toBeNull();
    expect(dfa(whiteNoise(4096, 1), { ...DEFAULT_DFA_CONFIG, numScales: 1 })).toBeNull();
    expect(dfa(whiteNoise(4096, 1), { ...DEFAULT_DFA_CONFIG, minScale: 2 })).toBeNull();
  });

  it("matches the Rust implementation on the shared parity fixture", () => {
    // Same fixture as `lrtc::tests::parity_fixture` in
    // ruv-neural-signal/src/lrtc.rs: LCG seed 12345, n = 2048, defaults.
    // Both implementations perform the same IEEE-754 operations in the same
    // order, so agreement is expected to machine precision.
    const result = dfa(whiteNoise(2048, 12345));
    expect(result).not.toBeNull();
    expect(Math.abs(result!.alpha - 0.542472684045213)).toBeLessThan(1e-9);
  });
});
