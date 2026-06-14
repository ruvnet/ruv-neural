//! Session reporting — the clean evidence artifact a closed-loop run produces.

use serde::{Deserialize, Serialize};

use crate::audit::AuditTrail;
use crate::controller::ControllerPhase;
use crate::envelope::BreachReason;
use crate::state::TargetState;

/// A summary of a completed (or stopped) closed-loop session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionReport {
    /// The target state the session pursued.
    pub target: TargetState,
    /// Protocol name used.
    pub protocol: String,
    /// Final controller phase.
    pub final_phase: ControllerPhase,
    /// Total control steps executed.
    pub total_steps: u64,
    /// Steps spent collecting baseline.
    pub baseline_steps: u64,
    /// Steps that actively stimulated.
    pub stimulate_steps: u64,
    /// Total active stimulation time (s).
    pub total_stimulation_s: f64,
    /// Peak commanded intensity reached.
    pub peak_intensity: f64,
    /// Whether the target was reached (normal completion).
    pub goal_reached: bool,
    /// Whether the session ended in a fail-safe stop.
    pub safe_stopped: bool,
    /// Breach reasons, if safe-stopped.
    pub stop_reasons: Vec<BreachReason>,
    /// Best (lowest) distance-to-target achieved.
    pub best_distance: f64,
    /// Final distance-to-target.
    pub final_distance: f64,
    /// Number of stimulus delivery receipts emitted.
    pub num_receipts: u64,
    /// Whether every emitted receipt verified.
    pub all_receipts_verified: bool,
    /// Audit chain head hash.
    pub audit_head_hash: String,
    /// Number of audit records.
    pub audit_records: u64,
    /// Whether the audit hash chain verifies intact.
    pub audit_chain_valid: bool,
}

impl SessionReport {
    /// A one-line human verdict for logs / CLI.
    pub fn verdict(&self) -> &'static str {
        if self.safe_stopped {
            "SAFE-STOPPED"
        } else if self.goal_reached {
            "TARGET-REACHED"
        } else {
            "INCOMPLETE"
        }
    }

    /// Whether this session satisfies the platform acceptance test:
    /// it identified a target, delivered at least one *verified* stimulus,
    /// measured a response (ran control steps), and either reached the target
    /// or stopped safely — with an intact audit chain.
    pub fn passes_acceptance(&self) -> bool {
        self.num_receipts >= 1
            && self.all_receipts_verified
            && self.total_steps >= 1
            && (self.goal_reached || self.safe_stopped)
            && self.audit_chain_valid
    }
}

/// Build a report from a finished trail and accumulated counters. Used
/// internally by the controller.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_report(
    target: TargetState,
    protocol: String,
    final_phase: ControllerPhase,
    total_steps: u64,
    baseline_steps: u64,
    stimulate_steps: u64,
    total_stimulation_s: f64,
    peak_intensity: f64,
    best_distance: f64,
    final_distance: f64,
    num_receipts: u64,
    all_receipts_verified: bool,
    stop_reasons: Vec<BreachReason>,
    audit: &AuditTrail,
) -> SessionReport {
    SessionReport {
        target,
        protocol,
        goal_reached: final_phase == ControllerPhase::Completed,
        safe_stopped: final_phase == ControllerPhase::SafeStopped,
        final_phase,
        total_steps,
        baseline_steps,
        stimulate_steps,
        total_stimulation_s,
        peak_intensity,
        stop_reasons,
        best_distance,
        final_distance,
        num_receipts,
        all_receipts_verified,
        audit_head_hash: audit.head_hash(),
        audit_records: audit.len() as u64,
        audit_chain_valid: audit.verify_chain(),
    }
}
