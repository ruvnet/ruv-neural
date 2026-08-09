//! Decode-armed / decode-locked mental-privacy gate.
//!
//! Implements the "thought password" gating pattern demonstrated by Kunz et
//! al. (*Cell*, 2025): decoded output stays **locked** until the user produces
//! a chosen unlock signal (e.g. an imagined password phrase) detected with
//! high confidence, and re-locks automatically after a bounded armed window.
//! See ADR-0007 (safety envelope) and ADR-0022/0023 (privacy, neural data
//! governance): decoding is opt-in per use, never ambient.
//!
//! The gate is a small deterministic state machine driven by two inputs per
//! decode frame: the password-detector confidence and a monotonic timestamp.
//! Every transition is recorded in a bounded audit log (ADR-0009).
//!
//! ```
//! use ruv_neural_core::gate::{DecodeGate, DecodeGateConfig, GateState};
//!
//! let mut gate = DecodeGate::new(DecodeGateConfig::default()).unwrap();
//! assert!(!gate.is_armed());
//! // Password detector fires above threshold for the required frames:
//! gate.update(0.99, 0.0);
//! gate.update(0.99, 0.1);
//! gate.update(0.99, 0.2);
//! assert!(gate.is_armed());
//! // Decoded text is only released while armed:
//! assert_eq!(gate.release("hello"), Some("hello"));
//! gate.lock(0.3);
//! assert_eq!(gate.release("hello"), None);
//! ```

use serde::{Deserialize, Serialize};

use crate::error::{Result, RuvNeuralError};

/// Maximum number of transitions retained in the audit log. Oldest entries
/// are dropped first so memory stays bounded on long-running edge targets.
pub const MAX_AUDIT_TRANSITIONS: usize = 1024;

/// Configuration for the decode gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeGateConfig {
    /// Password-detector confidence required to make arming progress.
    /// Kunz et al. report >98% detection accuracy for an imagined password;
    /// the default threshold matches that operating point.
    pub arm_threshold: f64,
    /// Number of consecutive frames at or above `arm_threshold` required to
    /// arm. Values above 1 protect against single-frame false positives.
    pub arm_frames: u32,
    /// Seconds after arming before the gate re-locks automatically. A bounded
    /// armed window is what makes decoding opt-in per use rather than ambient.
    pub armed_timeout_s: f64,
}

impl Default for DecodeGateConfig {
    fn default() -> Self {
        Self {
            arm_threshold: 0.98,
            arm_frames: 3,
            armed_timeout_s: 300.0,
        }
    }
}

impl DecodeGateConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if !(self.arm_threshold > 0.0 && self.arm_threshold <= 1.0) {
            return Err(RuvNeuralError::Config(format!(
                "arm_threshold must be in (0, 1], got {}",
                self.arm_threshold
            )));
        }
        if self.arm_frames == 0 {
            return Err(RuvNeuralError::Config(
                "arm_frames must be >= 1".to_string(),
            ));
        }
        if !(self.armed_timeout_s > 0.0 && self.armed_timeout_s.is_finite()) {
            return Err(RuvNeuralError::Config(format!(
                "armed_timeout_s must be finite and > 0, got {}",
                self.armed_timeout_s
            )));
        }
        Ok(())
    }
}

/// The gate's externally visible state.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum GateState {
    /// Decoding output is suppressed. The default and fail-safe state.
    Locked,
    /// The password detector has fired but not yet for enough consecutive
    /// frames; output is still suppressed.
    Arming {
        /// Consecutive above-threshold frames observed so far.
        consecutive: u32,
    },
    /// Decoding output is released until timeout or explicit lock.
    #[serde(rename_all = "camelCase")]
    Armed {
        /// Timestamp (seconds) at which the gate armed.
        armed_at_s: f64,
    },
}

impl GateState {
    /// Short stable name for logs and cross-language bindings.
    pub fn name(&self) -> &'static str {
        match self {
            GateState::Locked => "locked",
            GateState::Arming { .. } => "arming",
            GateState::Armed { .. } => "armed",
        }
    }
}

/// Why a transition happened — recorded in the audit log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateTransitionReason {
    /// Password confidence crossed the arm threshold (Locked -> Arming, or
    /// Arming -> Armed on the final required frame).
    PasswordDetected,
    /// Confidence dropped below threshold before arming completed.
    ArmingAborted,
    /// The armed window expired.
    Timeout,
    /// `lock()` was called.
    ExplicitLock,
    /// The clock went non-finite or regressed while the gate was not locked;
    /// the gate fails locked rather than trusting the anomalous clock.
    ClockAnomaly,
    /// The gate was restored from persisted state; arming is opt-in per use
    /// and never survives persistence, so restore always re-locks.
    Restored,
}

/// One audited state transition.
///
/// Serializes with camelCase field names (`atS`) so the wire shape matches
/// the TypeScript mirror (`apps/ruv-neural-ui/src/safety/decodeGate.ts`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GateTransition {
    /// Timestamp (seconds) at which the transition occurred.
    pub at_s: f64,
    /// State before the transition.
    pub from: GateState,
    /// State after the transition.
    pub to: GateState,
    /// Why the transition happened.
    pub reason: GateTransitionReason,
}

/// Decode-armed / decode-locked gate state machine.
///
/// Drive it with [`DecodeGate::update`] once per decode frame and wrap all
/// decoder output in [`DecodeGate::release`]. The gate starts — and fails —
/// locked: out-of-domain confidence, a non-finite or regressing clock, the
/// armed-window timeout, and restore-from-persistence all resolve to
/// suppressed output.
#[derive(Debug, Clone, Serialize)]
pub struct DecodeGate {
    config: DecodeGateConfig,
    state: GateState,
    last_now_s: f64,
    transitions: Vec<GateTransition>,
}

// Restoring a gate must never resurrect an armed window (the opt-in is per
// use, not per process lifetime) and must never bypass config validation, so
// Deserialize is implemented manually instead of derived: the persisted
// config and audit log are kept, but the state always resumes Locked.
impl<'de> Deserialize<'de> for DecodeGate {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Repr {
            config: DecodeGateConfig,
            state: GateState,
            // serde_json writes non-finite floats as null; a fresh gate's
            // last_now_s is -inf, so tolerate null here.
            last_now_s: Option<f64>,
            transitions: Vec<GateTransition>,
        }
        let repr = Repr::deserialize(deserializer)?;
        repr.config.validate().map_err(serde::de::Error::custom)?;
        let mut transitions = repr.transitions;
        if transitions.len() > MAX_AUDIT_TRANSITIONS {
            let excess = transitions.len() - MAX_AUDIT_TRANSITIONS;
            transitions.drain(..excess);
        }
        let mut gate = DecodeGate {
            config: repr.config,
            state: GateState::Locked,
            last_now_s: f64::NEG_INFINITY,
            transitions,
        };
        if repr.state != GateState::Locked {
            let at_s = repr.last_now_s.filter(|t| t.is_finite()).unwrap_or(0.0);
            if gate.transitions.len() >= MAX_AUDIT_TRANSITIONS {
                gate.transitions.remove(0);
            }
            gate.transitions.push(GateTransition {
                at_s,
                from: repr.state,
                to: GateState::Locked,
                reason: GateTransitionReason::Restored,
            });
        }
        Ok(gate)
    }
}

impl DecodeGate {
    /// Create a new gate in the [`GateState::Locked`] state.
    pub fn new(config: DecodeGateConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            state: GateState::Locked,
            last_now_s: f64::NEG_INFINITY,
            transitions: Vec::new(),
        })
    }

    /// The active configuration.
    pub fn config(&self) -> &DecodeGateConfig {
        &self.config
    }

    /// Current state.
    pub fn state(&self) -> GateState {
        self.state
    }

    /// Whether decoded output may currently be released.
    pub fn is_armed(&self) -> bool {
        matches!(self.state, GateState::Armed { .. })
    }

    /// The audit log of state transitions (oldest first, bounded to
    /// [`MAX_AUDIT_TRANSITIONS`] entries).
    pub fn transitions(&self) -> &[GateTransition] {
        &self.transitions
    }

    /// Advance the gate by one decode frame.
    ///
    /// * `password_confidence` — detector confidence in `[0, 1]` that the
    ///   unlock signal (e.g. the imagined password phrase) is present in this
    ///   frame. Any value outside that domain — NaN, ±inf, or a finite value
    ///   below 0 or above 1 (e.g. a percent-scaled detector) — is treated as
    ///   `0.0` (fail-locked).
    /// * `now_s` — monotonic timestamp in seconds. A non-finite or regressing
    ///   clock is a safety event: any in-progress arming or armed window
    ///   fails locked (audited as [`GateTransitionReason::ClockAnomaly`]) and
    ///   the anomalous frame is dropped. After a regression the gate adopts
    ///   the new (earlier) timeline, so a stepped-back clock can never freeze
    ///   an armed countdown at zero elapsed time.
    ///
    /// Returns the state after the frame.
    pub fn update(&mut self, password_confidence: f64, now_s: f64) -> GateState {
        let confidence =
            if password_confidence.is_finite() && (0.0..=1.0).contains(&password_confidence) {
                password_confidence
            } else {
                0.0
            };
        if !now_s.is_finite() {
            if self.state != GateState::Locked {
                let at_s = self.last_now_s;
                self.transition(GateState::Locked, GateTransitionReason::ClockAnomaly, at_s);
            }
            return self.state;
        }
        if now_s < self.last_now_s {
            if self.state != GateState::Locked {
                let at_s = self.last_now_s;
                self.transition(GateState::Locked, GateTransitionReason::ClockAnomaly, at_s);
            }
            self.last_now_s = now_s;
            return self.state;
        }
        self.last_now_s = now_s;

        let above = confidence >= self.config.arm_threshold;
        match self.state {
            GateState::Locked => {
                if above {
                    self.advance_arming(1, now_s);
                }
            }
            GateState::Arming { consecutive } => {
                if above {
                    self.advance_arming(consecutive + 1, now_s);
                } else {
                    self.transition(
                        GateState::Locked,
                        GateTransitionReason::ArmingAborted,
                        now_s,
                    );
                }
            }
            GateState::Armed { armed_at_s } => {
                // The password only arms; while armed, confidence is ignored
                // and only the bounded timeout (or an explicit lock) disarms.
                if now_s - armed_at_s >= self.config.armed_timeout_s {
                    self.transition(GateState::Locked, GateTransitionReason::Timeout, now_s);
                }
            }
        }
        self.state
    }

    /// Explicitly re-lock the gate (user action or safety-envelope trip).
    pub fn lock(&mut self, now_s: f64) -> GateState {
        let now_s = if now_s.is_finite() {
            now_s.max(self.last_now_s)
        } else {
            self.last_now_s
        };
        self.last_now_s = now_s;
        if self.state != GateState::Locked {
            self.transition(GateState::Locked, GateTransitionReason::ExplicitLock, now_s);
        }
        self.state
    }

    /// Pass decoded output through the gate: `Some` only while armed.
    pub fn release<'a>(&self, decoded: &'a str) -> Option<&'a str> {
        if self.is_armed() {
            Some(decoded)
        } else {
            None
        }
    }

    fn advance_arming(&mut self, consecutive: u32, now_s: f64) {
        if consecutive >= self.config.arm_frames {
            self.transition(
                GateState::Armed { armed_at_s: now_s },
                GateTransitionReason::PasswordDetected,
                now_s,
            );
        } else {
            self.transition(
                GateState::Arming { consecutive },
                GateTransitionReason::PasswordDetected,
                now_s,
            );
        }
    }

    fn transition(&mut self, to: GateState, reason: GateTransitionReason, at_s: f64) {
        let from = self.state;
        self.state = to;
        if self.transitions.len() >= MAX_AUDIT_TRANSITIONS {
            self.transitions.remove(0);
        }
        self.transitions.push(GateTransition {
            at_s,
            from,
            to,
            reason,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> DecodeGate {
        DecodeGate::new(DecodeGateConfig::default()).unwrap()
    }

    #[test]
    fn starts_locked_and_suppresses_output() {
        let g = gate();
        assert_eq!(g.state(), GateState::Locked);
        assert_eq!(g.release("secret"), None);
    }

    #[test]
    fn arms_after_consecutive_frames() {
        let mut g = gate();
        assert_eq!(g.update(0.99, 0.0), GateState::Arming { consecutive: 1 });
        assert_eq!(g.update(0.99, 0.1), GateState::Arming { consecutive: 2 });
        assert!(matches!(g.update(0.99, 0.2), GateState::Armed { .. }));
        assert_eq!(g.release("hello"), Some("hello"));
    }

    #[test]
    fn below_threshold_frame_aborts_arming() {
        let mut g = gate();
        g.update(0.99, 0.0);
        g.update(0.99, 0.1);
        assert_eq!(g.update(0.50, 0.2), GateState::Locked);
        // Progress does not persist across the abort.
        assert_eq!(g.update(0.99, 0.3), GateState::Arming { consecutive: 1 });
    }

    #[test]
    fn armed_window_times_out() {
        let mut g = DecodeGate::new(DecodeGateConfig {
            armed_timeout_s: 10.0,
            ..DecodeGateConfig::default()
        })
        .unwrap();
        for i in 0..3 {
            g.update(0.99, i as f64 * 0.1);
        }
        assert!(g.is_armed());
        // Low confidence while armed does not disarm...
        assert!(matches!(g.update(0.0, 5.0), GateState::Armed { .. }));
        // ...but the bounded window does.
        assert_eq!(g.update(0.99, 10.3), GateState::Locked);
        assert_eq!(g.release("hello"), None);
    }

    #[test]
    fn explicit_lock_disarms() {
        let mut g = gate();
        for i in 0..3 {
            g.update(0.99, i as f64 * 0.1);
        }
        assert!(g.is_armed());
        assert_eq!(g.lock(0.5), GateState::Locked);
        assert_eq!(g.release("hello"), None);
    }

    #[test]
    fn out_of_domain_confidence_fails_locked() {
        let mut g = gate();
        assert_eq!(g.update(f64::NAN, 0.0), GateState::Locked);
        // +inf is non-finite too: treated as 0.0, not as "very confident".
        assert_eq!(g.update(f64::INFINITY, 0.1), GateState::Locked);
        // Finite but out-of-domain values (percent-scaled or corrupted
        // detectors) must not arm either.
        assert_eq!(g.update(50.0, 0.2), GateState::Locked);
        assert_eq!(g.update(2.0, 0.3), GateState::Locked);
        assert_eq!(g.update(1e308, 0.4), GateState::Locked);
        assert_eq!(g.update(-0.5, 0.5), GateState::Locked);
        assert_eq!(g.update(0.99, 0.6), GateState::Arming { consecutive: 1 });
    }

    #[test]
    fn non_finite_clock_locks_and_never_arms() {
        let mut g = gate();
        // A NaN clock can never make arming progress...
        for _ in 0..10 {
            assert_eq!(g.update(0.99, f64::NAN), GateState::Locked);
        }
        // ...and locks an armed gate instead of freezing its timeout.
        for i in 0..3 {
            g.update(0.99, i as f64 * 0.1);
        }
        assert!(g.is_armed());
        assert_eq!(g.update(0.0, f64::NAN), GateState::Locked);
        assert_eq!(
            g.transitions().last().unwrap().reason,
            GateTransitionReason::ClockAnomaly
        );
        assert_eq!(g.release("secret"), None);
        // Continued NaN clocks keep it locked.
        for _ in 0..100 {
            assert_eq!(g.update(0.99, f64::NAN), GateState::Locked);
        }
    }

    #[test]
    fn clock_regression_while_armed_locks() {
        let mut g = DecodeGate::new(DecodeGateConfig {
            armed_timeout_s: 10.0,
            ..DecodeGateConfig::default()
        })
        .unwrap();
        for i in 0..3 {
            g.update(0.99, 100.0 + i as f64);
        }
        assert!(g.is_armed());
        // A clock that jumps backwards is a safety event: fail locked rather
        // than freezing the armed countdown at zero elapsed time.
        assert_eq!(g.update(0.0, 0.0), GateState::Locked);
        assert_eq!(
            g.transitions().last().unwrap().reason,
            GateTransitionReason::ClockAnomaly
        );
        // The gate adopts the new timeline: re-arming and the bounded
        // timeout both work on the regressed clock.
        for i in 0..3 {
            g.update(0.99, 1.0 + i as f64 * 0.1);
        }
        assert!(g.is_armed());
        assert_eq!(g.update(0.0, 20.0), GateState::Locked);
        assert_eq!(
            g.transitions().last().unwrap().reason,
            GateTransitionReason::Timeout
        );
    }

    #[test]
    fn audit_log_records_transitions() {
        let mut g = gate();
        for i in 0..3 {
            g.update(0.99, i as f64 * 0.1);
        }
        g.lock(0.5);
        let log = g.transitions();
        assert_eq!(log.len(), 4);
        assert_eq!(log[0].reason, GateTransitionReason::PasswordDetected);
        assert!(matches!(log[2].to, GateState::Armed { .. }));
        assert_eq!(log[3].reason, GateTransitionReason::ExplicitLock);
        assert_eq!(log[3].to, GateState::Locked);
    }

    #[test]
    fn audit_log_is_bounded() {
        let mut g = DecodeGate::new(DecodeGateConfig {
            arm_frames: 1,
            ..DecodeGateConfig::default()
        })
        .unwrap();
        for i in 0..(MAX_AUDIT_TRANSITIONS + 100) {
            let t = i as f64;
            g.update(0.99, t);
            g.lock(t + 0.5);
        }
        assert_eq!(g.transitions().len(), MAX_AUDIT_TRANSITIONS);
    }

    #[test]
    fn config_validation_rejects_bad_values() {
        assert!(DecodeGate::new(DecodeGateConfig {
            arm_threshold: 0.0,
            ..DecodeGateConfig::default()
        })
        .is_err());
        assert!(DecodeGate::new(DecodeGateConfig {
            arm_threshold: 1.5,
            ..DecodeGateConfig::default()
        })
        .is_err());
        assert!(DecodeGate::new(DecodeGateConfig {
            arm_frames: 0,
            ..DecodeGateConfig::default()
        })
        .is_err());
        assert!(DecodeGate::new(DecodeGateConfig {
            armed_timeout_s: 0.0,
            ..DecodeGateConfig::default()
        })
        .is_err());
    }

    #[test]
    fn restore_forces_locked_and_keeps_audit_log() {
        let mut g = gate();
        for i in 0..3 {
            g.update(0.99, 3.0 + i as f64 * 0.1);
        }
        assert!(g.is_armed());
        let json = serde_json::to_string(&g).unwrap();
        let back: DecodeGate = serde_json::from_str(&json).unwrap();
        // Arming is opt-in per use: an armed state never survives restore,
        // and release() is suppressed immediately (no update() required).
        assert_eq!(back.state(), GateState::Locked);
        assert_eq!(back.release("secret"), None);
        let restored = back.transitions().last().unwrap();
        assert_eq!(restored.reason, GateTransitionReason::Restored);
        assert!(matches!(restored.from, GateState::Armed { .. }));
        // The prior audit history is preserved ahead of the restore entry.
        assert_eq!(back.transitions().len(), g.transitions().len() + 1);
        // A restarted (earlier) clock works: the gate adopted a fresh
        // timeline, so re-arming near t=0 is possible.
        let mut back = back;
        for i in 0..3 {
            back.update(0.99, 0.1 + i as f64 * 0.1);
        }
        assert!(back.is_armed());
    }

    #[test]
    fn restore_validates_config_and_roundtrips_fresh_gate() {
        // Deserialize must not bypass config validation.
        let bad = r#"{"config":{"arm_threshold":2.0,"arm_frames":0,"armed_timeout_s":-5.0},"state":{"state":"armed","armedAtS":0.0},"last_now_s":0.0,"transitions":[]}"#;
        assert!(serde_json::from_str::<DecodeGate>(bad).is_err());
        // A fresh gate (last_now_s = -inf, serialized as null) roundtrips.
        let fresh = gate();
        let json = serde_json::to_string(&fresh).unwrap();
        let back: DecodeGate = serde_json::from_str(&json).unwrap();
        assert_eq!(back.state(), GateState::Locked);
        assert!(back.transitions().is_empty());
    }

    #[test]
    fn transition_log_uses_camel_case_wire_shape() {
        let mut g = gate();
        for i in 0..3 {
            g.update(0.99, i as f64 * 0.1);
        }
        let json = serde_json::to_string(g.transitions()).unwrap();
        // Matches the TypeScript mirror's shape: atS / armedAtS, snake_case
        // state and reason tags.
        assert!(json.contains("\"atS\""), "json: {json}");
        assert!(json.contains("\"armedAtS\""), "json: {json}");
        assert!(json.contains("\"password_detected\""), "json: {json}");
        assert!(!json.contains("at_s"), "json: {json}");
    }
}
