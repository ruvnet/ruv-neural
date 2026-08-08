/**
 * Decode-armed / decode-locked mental-privacy gate.
 *
 * TypeScript mirror of `ruv-neural-core/src/gate.rs` (the WASM export
 * `WasmDecodeGate` wraps the Rust implementation; this module is the
 * dependency-free fallback with identical semantics, verified by the parity
 * suite in `decodeGate.test.ts`).
 *
 * Pattern: Kunz et al. (Cell, 2025) — decoded output stays locked until the
 * user produces a chosen unlock signal (e.g. an imagined password phrase)
 * detected with high confidence, and re-locks automatically after a bounded
 * armed window. Decoding is opt-in per use, never ambient (ADR-0007/0022).
 */

export const MAX_AUDIT_TRANSITIONS = 1024;

export interface DecodeGateConfig {
  /** Password-detector confidence required to make arming progress. */
  armThreshold: number;
  /** Consecutive frames at/above threshold required to arm. */
  armFrames: number;
  /** Seconds after arming before the gate re-locks automatically. */
  armedTimeoutS: number;
}

export const DEFAULT_GATE_CONFIG: DecodeGateConfig = {
  armThreshold: 0.98,
  armFrames: 3,
  armedTimeoutS: 300,
};

export type GateState =
  | { state: "locked" }
  | { state: "arming"; consecutive: number }
  | { state: "armed"; armedAtS: number };

export type GateTransitionReason =
  | "password_detected"
  | "arming_aborted"
  | "timeout"
  | "explicit_lock";

export interface GateTransition {
  atS: number;
  from: GateState;
  to: GateState;
  reason: GateTransitionReason;
}

export class DecodeGate {
  private readonly config: DecodeGateConfig;
  private currentState: GateState = { state: "locked" };
  private lastNowS = Number.NEGATIVE_INFINITY;
  private readonly log: GateTransition[] = [];

  constructor(config: DecodeGateConfig = DEFAULT_GATE_CONFIG) {
    if (!(config.armThreshold > 0 && config.armThreshold <= 1)) {
      throw new Error(`armThreshold must be in (0, 1], got ${config.armThreshold}`);
    }
    if (!Number.isInteger(config.armFrames) || config.armFrames < 1) {
      throw new Error(`armFrames must be an integer >= 1, got ${config.armFrames}`);
    }
    if (!(config.armedTimeoutS > 0 && Number.isFinite(config.armedTimeoutS))) {
      throw new Error(`armedTimeoutS must be finite and > 0, got ${config.armedTimeoutS}`);
    }
    this.config = { ...config };
  }

  get state(): GateState {
    return this.currentState;
  }

  get isArmed(): boolean {
    return this.currentState.state === "armed";
  }

  /** Bounded audit log of state transitions, oldest first. */
  get transitions(): readonly GateTransition[] {
    return this.log;
  }

  /**
   * Advance the gate by one decode frame. Non-finite confidence is treated
   * as 0 (fail-locked); a regressing clock is clamped, never trusted.
   */
  update(passwordConfidence: number, nowS: number): GateState {
    const confidence = Number.isFinite(passwordConfidence)
      ? Math.min(Math.max(passwordConfidence, 0), 1)
      : 0;
    const now = Number.isFinite(nowS) ? Math.max(nowS, this.lastNowS) : this.lastNowS;
    this.lastNowS = now;

    const above = confidence >= this.config.armThreshold;
    const s = this.currentState;
    if (s.state === "locked") {
      if (above) {
        this.advanceArming(1, now);
      }
    } else if (s.state === "arming") {
      if (above) {
        this.advanceArming(s.consecutive + 1, now);
      } else {
        this.transition({ state: "locked" }, "arming_aborted", now);
      }
    } else {
      // Armed: the password only arms; only the bounded timeout (or an
      // explicit lock) disarms.
      if (now - s.armedAtS >= this.config.armedTimeoutS) {
        this.transition({ state: "locked" }, "timeout", now);
      }
    }
    return this.currentState;
  }

  /** Explicitly re-lock the gate (user action or safety-envelope trip). */
  lock(nowS: number): GateState {
    const now = Number.isFinite(nowS) ? Math.max(nowS, this.lastNowS) : this.lastNowS;
    this.lastNowS = now;
    if (this.currentState.state !== "locked") {
      this.transition({ state: "locked" }, "explicit_lock", now);
    }
    return this.currentState;
  }

  /** Pass decoded output through the gate: the value while armed, else null. */
  release(decoded: string): string | null {
    return this.isArmed ? decoded : null;
  }

  private advanceArming(consecutive: number, now: number): void {
    if (consecutive >= this.config.armFrames) {
      this.transition({ state: "armed", armedAtS: now }, "password_detected", now);
    } else {
      this.transition({ state: "arming", consecutive }, "password_detected", now);
    }
  }

  private transition(to: GateState, reason: GateTransitionReason, atS: number): void {
    const from = this.currentState;
    this.currentState = to;
    if (this.log.length >= MAX_AUDIT_TRANSITIONS) {
      this.log.shift();
    }
    this.log.push({ atS, from, to, reason });
  }
}
