import { describe, expect, it } from "vitest";

import {
  DecodeGate,
  DEFAULT_GATE_CONFIG,
  MAX_AUDIT_TRANSITIONS,
} from "./decodeGate";

// Semantics parity suite: each case mirrors a test in
// ruv-neural-core/src/gate.rs so the TS and Rust gates stay in lockstep.

describe("DecodeGate", () => {
  it("starts locked and suppresses output", () => {
    const gate = new DecodeGate();
    expect(gate.state).toEqual({ state: "locked" });
    expect(gate.release("secret")).toBeNull();
  });

  it("arms after the configured number of consecutive frames", () => {
    const gate = new DecodeGate();
    expect(gate.update(0.99, 0.0)).toEqual({ state: "arming", consecutive: 1 });
    expect(gate.update(0.99, 0.1)).toEqual({ state: "arming", consecutive: 2 });
    expect(gate.update(0.99, 0.2)).toEqual({ state: "armed", armedAtS: 0.2 });
    expect(gate.release("hello")).toBe("hello");
  });

  it("aborts arming on a below-threshold frame without keeping progress", () => {
    const gate = new DecodeGate();
    gate.update(0.99, 0.0);
    gate.update(0.99, 0.1);
    expect(gate.update(0.5, 0.2)).toEqual({ state: "locked" });
    expect(gate.update(0.99, 0.3)).toEqual({ state: "arming", consecutive: 1 });
  });

  it("re-locks when the armed window times out, ignoring confidence while armed", () => {
    const gate = new DecodeGate({ ...DEFAULT_GATE_CONFIG, armedTimeoutS: 10 });
    for (let i = 0; i < 3; i++) {
      gate.update(0.99, i * 0.1);
    }
    expect(gate.isArmed).toBe(true);
    expect(gate.update(0.0, 5.0).state).toBe("armed");
    expect(gate.update(0.99, 10.3)).toEqual({ state: "locked" });
    expect(gate.release("hello")).toBeNull();
  });

  it("disarms on explicit lock", () => {
    const gate = new DecodeGate();
    for (let i = 0; i < 3; i++) {
      gate.update(0.99, i * 0.1);
    }
    expect(gate.isArmed).toBe(true);
    expect(gate.lock(0.5)).toEqual({ state: "locked" });
    expect(gate.release("hello")).toBeNull();
  });

  it("treats out-of-domain confidence as zero (fail-locked)", () => {
    const gate = new DecodeGate();
    expect(gate.update(Number.NaN, 0.0)).toEqual({ state: "locked" });
    expect(gate.update(Number.POSITIVE_INFINITY, 0.1)).toEqual({ state: "locked" });
    // Finite but out-of-domain values (percent-scaled detectors) must not arm.
    expect(gate.update(50, 0.2)).toEqual({ state: "locked" });
    expect(gate.update(2, 0.3)).toEqual({ state: "locked" });
    expect(gate.update(-0.5, 0.4)).toEqual({ state: "locked" });
    expect(gate.update(0.99, 0.5)).toEqual({ state: "arming", consecutive: 1 });
  });

  it("locks on a non-finite clock and never arms under one", () => {
    const gate = new DecodeGate();
    for (let i = 0; i < 10; i++) {
      expect(gate.update(0.99, Number.NaN)).toEqual({ state: "locked" });
    }
    for (let i = 0; i < 3; i++) {
      gate.update(0.99, i * 0.1);
    }
    expect(gate.isArmed).toBe(true);
    expect(gate.update(0, Number.NaN)).toEqual({ state: "locked" });
    expect(gate.transitions.at(-1)?.reason).toBe("clock_anomaly");
    expect(gate.release("secret")).toBeNull();
  });

  it("locks on clock regression while armed and adopts the new timeline", () => {
    const gate = new DecodeGate({ ...DEFAULT_GATE_CONFIG, armedTimeoutS: 10 });
    for (let i = 0; i < 3; i++) {
      gate.update(0.99, 100 + i);
    }
    expect(gate.isArmed).toBe(true);
    // Regression is a safety event: fail locked instead of freezing the
    // armed countdown at zero elapsed time.
    expect(gate.update(0.0, 0.0)).toEqual({ state: "locked" });
    expect(gate.transitions.at(-1)?.reason).toBe("clock_anomaly");
    // Re-arming and the bounded timeout both work on the new timeline.
    for (let i = 0; i < 3; i++) {
      gate.update(0.99, 1 + i * 0.1);
    }
    expect(gate.isArmed).toBe(true);
    expect(gate.update(0, 20)).toEqual({ state: "locked" });
    expect(gate.transitions.at(-1)?.reason).toBe("timeout");
  });

  it("returns defensive copies so callers cannot corrupt the gate", () => {
    const gate = new DecodeGate({ ...DEFAULT_GATE_CONFIG, armedTimeoutS: 10 });
    for (let i = 0; i < 3; i++) {
      gate.update(0.99, i * 0.1);
    }
    expect(gate.isArmed).toBe(true);
    // Mutating the returned state must not disable the timeout.
    const s = gate.state as { state: string; armedAtS: number };
    s.armedAtS = Number.POSITIVE_INFINITY;
    expect(gate.update(0, 10.3)).toEqual({ state: "locked" });
    // Mutating a transition-log entry must not affect the gate either.
    for (let i = 0; i < 3; i++) {
      gate.update(0.99, 11 + i * 0.1);
    }
    const entry = gate.transitions.at(-1) as { to: { state: string; armedAtS: number } };
    entry.to.armedAtS = Number.POSITIVE_INFINITY;
    expect(gate.update(0, 30)).toEqual({ state: "locked" });
  });

  it("records transitions in the audit log", () => {
    const gate = new DecodeGate();
    for (let i = 0; i < 3; i++) {
      gate.update(0.99, i * 0.1);
    }
    gate.lock(0.5);
    const log = gate.transitions;
    expect(log).toHaveLength(4);
    expect(log[0].reason).toBe("password_detected");
    expect(log[2].to.state).toBe("armed");
    expect(log[3].reason).toBe("explicit_lock");
    expect(log[3].to).toEqual({ state: "locked" });
  });

  it("bounds the audit log", () => {
    const gate = new DecodeGate({ ...DEFAULT_GATE_CONFIG, armFrames: 1 });
    for (let i = 0; i < MAX_AUDIT_TRANSITIONS + 100; i++) {
      gate.update(0.99, i);
      gate.lock(i + 0.5);
    }
    expect(gate.transitions).toHaveLength(MAX_AUDIT_TRANSITIONS);
  });

  it("rejects invalid configuration", () => {
    expect(() => new DecodeGate({ ...DEFAULT_GATE_CONFIG, armThreshold: 0 })).toThrow();
    expect(() => new DecodeGate({ ...DEFAULT_GATE_CONFIG, armThreshold: 1.5 })).toThrow();
    expect(() => new DecodeGate({ ...DEFAULT_GATE_CONFIG, armFrames: 0 })).toThrow();
    expect(() => new DecodeGate({ ...DEFAULT_GATE_CONFIG, armedTimeoutS: 0 })).toThrow();
  });
});
