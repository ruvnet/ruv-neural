//! The closed-loop controller (Ruflo core): a deterministic state machine that
//! turns observations into safe, audited stimulation decisions.
//!
//! Each [`ClosedLoopController::step`] runs one full loop iteration:
//!
//! ```text
//!   observe ─▶ embed (ruVector) ─▶ estimate state ─▶ SAFETY ENVELOPE
//!                                                       │
//!                              ┌────────── within ──────┴───── breach ──────┐
//!                              ▼                                            ▼
//!                    select protocol & dose                          fail-safe STOP
//!                              ▼                                       (intensity 0)
//!                    deliver verified stimulus
//!                              ▼
//!                         audit (hash-chained)
//! ```

use ruv_neural_biosense::PhysioMetrics;
use ruv_neural_stim::{StimulusGenerator, VerifiedStimulus};
use serde::{Deserialize, Serialize};

use crate::audit::{AuditEvent, AuditKind, AuditTrail};
use crate::embedding::{PersonalBaseline, PersonalStateEmbedding};
use crate::envelope::{EnvelopeStatus, SafetyEnvelope};
use crate::outcome::{build_report, SessionReport};
use crate::protocol::{Protocol, StimulusPlan};
use crate::state::{estimate_state, StateEstimate, StateObservation, TargetState};

/// The controller's lifecycle phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControllerPhase {
    /// Collecting a baseline before any stimulation.
    Baselining,
    /// Actively stimulating to move toward the target.
    Stimulating,
    /// At/near target, holding without escalation.
    Holding,
    /// Target reached; session completed normally. (terminal)
    Completed,
    /// Fail-safe stop after a safety-envelope breach. (terminal)
    SafeStopped,
    /// Aborted (e.g. step budget exhausted). (terminal)
    Aborted,
}

impl ControllerPhase {
    /// Whether this is a terminal phase (no further stimulation).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            ControllerPhase::Completed | ControllerPhase::SafeStopped | ControllerPhase::Aborted
        )
    }
}

/// Controller configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControllerConfig {
    /// Number of baseline steps before stimulation begins.
    pub baseline_steps: u64,
    /// Duration of each control step / stimulus window (s).
    pub step_duration_s: f64,
    /// Distance-to-target at/under which a step counts as "at target".
    pub completion_threshold: f64,
    /// Consecutive at-target steps required to complete the session.
    pub completion_hold_steps: u64,
    /// Hard cap on total steps before the session aborts.
    pub max_steps: u64,
    /// Optional subject identifier for embeddings / evidence.
    pub subject_id: Option<String>,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            baseline_steps: 2,
            step_duration_s: 10.0,
            completion_threshold: 0.12,
            completion_hold_steps: 2,
            max_steps: 64,
            subject_id: None,
        }
    }
}

/// The result of one control step.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Monotonic step index (matches the audit event index).
    pub index: u64,
    /// Session-relative timestamp of the observation (s).
    pub timestamp_s: f64,
    /// Phase after this step.
    pub phase: ControllerPhase,
    /// Audit event recorded for this step.
    pub audit_kind: AuditKind,
    /// State estimate at this step.
    pub estimate: StateEstimate,
    /// Safety-envelope status at this step.
    pub envelope: EnvelopeStatus,
    /// Verified stimuli emitted this step (empty for rest/hold/stop).
    pub emitted: Vec<VerifiedStimulus>,
    /// The plan that was selected.
    pub plan: StimulusPlan,
    /// The personal state embedding (ruVector) for this step.
    pub embedding: PersonalStateEmbedding,
    /// The fused physiological metrics observed this step.
    pub physio: PhysioMetrics,
    /// Human-readable message.
    pub message: String,
}

/// The closed-loop controller.
pub struct ClosedLoopController {
    config: ControllerConfig,
    target: TargetState,
    generator: StimulusGenerator,
    envelope: SafetyEnvelope,
    protocol: Box<dyn Protocol>,
    baseline: PersonalBaseline,
    audit: AuditTrail,

    phase: ControllerPhase,
    step_index: u64,
    prev_intensity: f64,
    prev_distance: Option<f64>,
    best_distance: f64,
    consecutive_at_target: u64,
    smoothed_distance: Option<f64>,
    smoothing_alpha: f64,

    // accumulators
    baseline_steps_run: u64,
    stimulate_steps_run: u64,
    total_stimulation_s: f64,
    peak_intensity: f64,
    num_receipts: u64,
    all_receipts_verified: bool,
    last_distance: f64,
    last_stop_reasons: Vec<crate::envelope::BreachReason>,
}

impl ClosedLoopController {
    /// Construct a controller. The audit trail is opened with a `SessionStart`
    /// record binding the target.
    pub fn new(
        config: ControllerConfig,
        target: TargetState,
        generator: StimulusGenerator,
        envelope: SafetyEnvelope,
        protocol: Box<dyn Protocol>,
    ) -> Self {
        let mut audit = AuditTrail::new();
        audit.append(AuditEvent {
            index: 0,
            timestamp_s: 0.0,
            kind: AuditKind::SessionStart,
            message: format!(
                "session start; target={:?}; protocol={}",
                target.label,
                protocol.name()
            ),
            intensity: 0.0,
            distance_to_target: 1.0,
            receipt_hashes: Vec::new(),
            breaches: Vec::new(),
        });

        Self {
            config,
            target,
            generator,
            envelope,
            protocol,
            baseline: PersonalBaseline::new(),
            audit,
            phase: ControllerPhase::Baselining,
            step_index: 0,
            prev_intensity: 0.0,
            prev_distance: None,
            best_distance: f64::INFINITY,
            consecutive_at_target: 0,
            smoothed_distance: None,
            smoothing_alpha: 0.5,
            baseline_steps_run: 0,
            stimulate_steps_run: 0,
            total_stimulation_s: 0.0,
            peak_intensity: 0.0,
            num_receipts: 0,
            all_receipts_verified: true,
            last_distance: 1.0,
            last_stop_reasons: Vec::new(),
        }
    }

    /// Current phase.
    pub fn phase(&self) -> ControllerPhase {
        self.phase
    }

    /// The audit trail.
    pub fn audit(&self) -> &AuditTrail {
        &self.audit
    }

    /// The running personal baseline.
    pub fn baseline(&self) -> &PersonalBaseline {
        &self.baseline
    }

    /// The target state.
    pub fn target(&self) -> &TargetState {
        &self.target
    }

    /// Run one control step against an observation.
    pub fn step(&mut self, obs: &StateObservation) -> StepResult {
        let embedding = PersonalStateEmbedding::from_observation(obs);

        // Terminal phases are absorbing: emit a no-op snapshot.
        if self.phase.is_terminal() {
            let estimate =
                estimate_state(obs, &self.target, self.config.completion_threshold);
            return StepResult {
                index: self.step_index,
                timestamp_s: obs.timestamp_s,
                phase: self.phase,
                audit_kind: AuditKind::Hold,
                estimate,
                envelope: EnvelopeStatus::Within,
                emitted: Vec::new(),
                plan: StimulusPlan::rest(),
                embedding,
                physio: obs.physio.clone(),
                message: "session terminal; no action".into(),
            };
        }

        let idx = self.step_index + 1; // audit index 0 is SessionStart
        self.step_index += 1;

        self.baseline.update(&embedding);
        let raw = estimate_state(obs, &self.target, self.config.completion_threshold);

        // Low-pass the feedback: per-window physiology is noisy, so titration,
        // completion, and divergence decisions all run on a smoothed distance.
        // This is standard closed-loop practice and avoids both chattering doses
        // and spurious fail-safe stops on single-sample noise.
        let smoothed = match self.smoothed_distance {
            Some(prev) => self.smoothing_alpha * raw.distance_to_target
                + (1.0 - self.smoothing_alpha) * prev,
            None => raw.distance_to_target,
        };
        self.smoothed_distance = Some(smoothed);
        let estimate = StateEstimate {
            label: raw.label,
            distance_to_target: smoothed,
            at_target: smoothed <= self.config.completion_threshold,
        };
        self.last_distance = smoothed;

        // Evaluate the safety envelope against the *running best* distance, so
        // a response that climbs back away from the best achieved is caught.
        let envelope = self.envelope.evaluate(
            &obs.physio,
            &estimate,
            self.best_distance,
            &self.target,
        );

        // Update running best after the envelope decision.
        if estimate.distance_to_target < self.best_distance {
            self.best_distance = estimate.distance_to_target;
        }

        // ── Decide the action ───────────────────────────────────────────
        let (kind, plan, emitted, message) = if envelope.is_breach() {
            self.phase = ControllerPhase::SafeStopped;
            self.last_stop_reasons = envelope.reasons().to_vec();
            (
                AuditKind::SafeStop,
                StimulusPlan::rest(),
                Vec::new(),
                format!(
                    "fail-safe stop: {} breach(es) — stimulation forced to zero",
                    envelope.reasons().len()
                ),
            )
        } else {
            match self.phase {
                ControllerPhase::Baselining => {
                    self.baseline_steps_run += 1;
                    if self.step_index >= self.config.baseline_steps {
                        self.phase = ControllerPhase::Stimulating;
                    }
                    (
                        AuditKind::Baseline,
                        StimulusPlan::rest(),
                        Vec::new(),
                        "collecting baseline".into(),
                    )
                }
                ControllerPhase::Stimulating | ControllerPhase::Holding => {
                    self.active_control(obs, &estimate)
                }
                // terminal handled above
                _ => unreachable!("terminal phase handled earlier"),
            }
        };

        // ── Accumulate & audit ──────────────────────────────────────────
        if plan.active {
            self.total_stimulation_s += self.config.step_duration_s;
            self.stimulate_steps_run += 1;
        }
        self.peak_intensity = self.peak_intensity.max(plan.intensity);
        self.prev_intensity = plan.intensity;
        self.prev_distance = Some(estimate.distance_to_target);

        let receipt_hashes: Vec<String> = emitted
            .iter()
            .map(|s| s.receipt.waveform_sha256.clone())
            .collect();
        self.num_receipts += emitted.len() as u64;
        for s in &emitted {
            self.all_receipts_verified &= s.receipt.verified;
        }

        self.audit.append(AuditEvent {
            index: idx,
            timestamp_s: obs.timestamp_s,
            kind: kind.clone(),
            message: message.clone(),
            intensity: plan.intensity,
            distance_to_target: estimate.distance_to_target,
            receipt_hashes,
            breaches: if kind == AuditKind::SafeStop {
                envelope.reasons().to_vec()
            } else {
                Vec::new()
            },
        });

        // Abort if we exhausted the step budget without completing.
        if !self.phase.is_terminal() && self.step_index >= self.config.max_steps {
            self.phase = ControllerPhase::Aborted;
            self.audit.append(AuditEvent {
                index: self.step_index + 1,
                timestamp_s: obs.timestamp_s,
                kind: AuditKind::Abort,
                message: "step budget exhausted".into(),
                intensity: 0.0,
                distance_to_target: estimate.distance_to_target,
                receipt_hashes: Vec::new(),
                breaches: Vec::new(),
            });
        }

        StepResult {
            index: idx,
            timestamp_s: obs.timestamp_s,
            phase: self.phase,
            audit_kind: kind,
            estimate,
            envelope,
            emitted,
            plan,
            embedding,
            physio: obs.physio.clone(),
            message,
        }
    }

    /// The stimulate/hold/complete decision when inside the envelope.
    fn active_control(
        &mut self,
        obs: &StateObservation,
        estimate: &StateEstimate,
    ) -> (AuditKind, StimulusPlan, Vec<VerifiedStimulus>, String) {
        if estimate.at_target {
            self.consecutive_at_target += 1;
        } else {
            self.consecutive_at_target = 0;
        }

        if self.consecutive_at_target >= self.config.completion_hold_steps {
            self.phase = ControllerPhase::Completed;
            return (
                AuditKind::Complete,
                StimulusPlan::rest(),
                Vec::new(),
                format!(
                    "target reached and held for {} steps",
                    self.consecutive_at_target
                ),
            );
        }

        let plan = self.protocol.next_plan(
            &self.target,
            estimate,
            self.prev_intensity,
            self.prev_distance,
            self.config.step_duration_s,
        );

        if !plan.active {
            self.phase = ControllerPhase::Holding;
            return (
                AuditKind::Hold,
                plan,
                Vec::new(),
                "holding (no active stimulation this step)".into(),
            );
        }

        // Generate verified, safety-clamped stimuli for each modality.
        let mut emitted = Vec::with_capacity(plan.stimuli.len());
        for p in &plan.stimuli {
            match self.generator.generate_clamped(p, obs.timestamp_s) {
                Ok(stim) => emitted.push(stim),
                Err(_e) => {
                    // A modality that cannot be safely delivered is simply
                    // skipped this step (e.g. unscreened light → zero); the
                    // remaining modalities still drive entrainment.
                }
            }
        }

        self.phase = if estimate.at_target {
            ControllerPhase::Holding
        } else {
            ControllerPhase::Stimulating
        };

        let msg = format!(
            "stimulate {} modality(ies) @ intensity {:.2}; distance {:.3}",
            emitted.len(),
            plan.intensity,
            estimate.distance_to_target
        );
        (AuditKind::Stimulate, plan, emitted, msg)
    }

    /// Sign the audit head, attesting the whole session with a fresh Ed25519
    /// key (delegates to [`AuditTrail::sign_head`]).
    pub fn sign_session(&self) -> crate::audit::SignedAuditHead {
        self.audit.sign_head()
    }

    /// Produce the session report.
    pub fn report(&self) -> SessionReport {
        build_report(
            self.target,
            self.protocol.name().to_string(),
            self.phase,
            self.step_index,
            self.baseline_steps_run,
            self.stimulate_steps_run,
            self.total_stimulation_s,
            self.peak_intensity,
            if self.best_distance.is_finite() {
                self.best_distance
            } else {
                1.0
            },
            self.last_distance,
            self.num_receipts,
            self.all_receipts_verified,
            self.last_stop_reasons.clone(),
            &self.audit,
        )
    }
}
