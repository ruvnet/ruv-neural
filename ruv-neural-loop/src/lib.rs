//! # ruv-neural-loop (Ruflo)
//!
//! The **closed-loop sensory neuromodulation controller** for the rUv Neural
//! platform. It binds the stimulus layer ([`ruv_neural_stim`]) and the response
//! layer ([`ruv_neural_biosense`]) into a single safe, auditable control loop:
//!
//! > **identify a target state → deliver a *verified* stimulus → measure the
//! > response → stop safely when the response leaves the allowed envelope.**
//!
//! That sentence is the platform's acceptance test, and
//! [`SessionReport::passes_acceptance`] checks it directly.
//!
//! ## Design pillars
//!
//! | Pillar | Type | ADR |
//! |--------|------|-----|
//! | Safe external modalities only | [`ruv_neural_stim`] | 0001, 0002 |
//! | Closed-loop control | [`ClosedLoopController`] | 0003 |
//! | Verified delivery | [`ruv_neural_stim::DeliveryReceipt`] | 0004 |
//! | Response sensing | [`ruv_neural_biosense`] | 0005 |
//! | Personal state embedding (ruVector) | [`PersonalStateEmbedding`] | 0006 |
//! | Safety envelope / fail-safe stop | [`SafetyEnvelope`] | 0007 |
//! | Conservative dosing | [`protocol::DosingPolicy`] | 0008 |
//! | Tamper-evident audit trail | [`AuditTrail`] | 0009 |
//!
//! ## Boundary
//!
//! This is a **research-grade wellness and cognitive-state platform, not a
//! disease-treatment device.** It commands only safe external sensory channels
//! and adapts conservatively with hard fail-safe stops. Transcranial/implanted
//! neuromodulation is out of scope (see `docs/adr/0001-scope.md`).
//!
//! ```
//! use ruv_neural_loop::*;
//! use ruv_neural_stim::StimulusGenerator;
//!
//! let controller = ClosedLoopController::new(
//!     ControllerConfig::default(),
//!     TargetState::relaxed(),
//!     StimulusGenerator::conservative(),
//!     SafetyEnvelope::default(),
//!     Box::new(GammaEntrainmentProtocol::audio_haptic()),
//! );
//! let mut controller = controller;
//! let mut sim = LoopSimulation::responsive(7, 10.0);
//! sim.run(&mut controller, 64);
//! let report = controller.report();
//! assert!(report.passes_acceptance());
//! ```

pub mod audit;
pub mod controller;
pub mod embedding;
pub mod envelope;
pub mod evidence;
pub mod federated;
pub mod outcome;
pub mod protocol;
pub mod sim;
pub mod state;

pub use audit::{AuditEvent, AuditKind, AuditRecord, AuditTrail, SignedAuditHead};
pub use controller::{ClosedLoopController, ControllerConfig, ControllerPhase, StepResult};
pub use embedding::{PersonalBaseline, PersonalStateEmbedding, EMBEDDING_DIM, FEATURE_NAMES};
pub use envelope::{BreachReason, EnvelopeStatus, SafetyEnvelope};
pub use evidence::{AcceptanceResult, EvidenceBundle, EvidenceStep, SCHEMA_VERSION};
pub use federated::{
    attach_federated_manifest, federated_average, read_federated_manifest, DpConfig,
    FederatedManifest, FederatedModel, FederatedUpdate,
};
pub use outcome::SessionReport;
pub use protocol::{DosingPolicy, GammaEntrainmentProtocol, Protocol, StimulusPlan};
pub use sim::LoopSimulation;
pub use state::{estimate_state, NeuralFeatures, StateEstimate, StateObservation, TargetState};

#[cfg(test)]
mod tests {
    use super::*;
    use ruv_neural_biosense::{PhysioMetrics, PhysioSimulator};
    use ruv_neural_core::topology::CognitiveState;
    use ruv_neural_stim::StimulusGenerator;

    fn observe(sim: &mut PhysioSimulator, t: f64, arousal: f64, gamma: f64) -> StateObservation {
        let w = sim.window(t, 10.0, arousal);
        let p = PhysioMetrics::from_window(&w).unwrap();
        StateObservation::from_physio(p).with_neural(NeuralFeatures {
            gamma_index: gamma,
            alpha_index: 1.0 - arousal,
            connectivity: 0.5,
        })
    }

    fn new_controller(target: TargetState) -> ClosedLoopController {
        ClosedLoopController::new(
            ControllerConfig::default(),
            target,
            StimulusGenerator::conservative(),
            SafetyEnvelope::default(),
            Box::new(GammaEntrainmentProtocol::audio_haptic()),
        )
    }

    // ── State estimation ────────────────────────────────────────────────

    #[test]
    fn relaxed_target_distance_small_when_calm() {
        let mut sim = PhysioSimulator::new(1);
        let obs = observe(&mut sim, 0.0, 0.1, 0.1);
        let est = estimate_state(&obs, &TargetState::relaxed(), 0.12);
        assert!(est.distance_to_target < 0.3);
    }

    #[test]
    fn gamma_target_needs_gamma() {
        let mut sim = PhysioSimulator::new(2);
        let low = observe(&mut sim, 0.0, 0.4, 0.1);
        let high = observe(&mut sim, 10.0, 0.4, 0.7);
        let t = TargetState::gamma_entrainment();
        let d_low = estimate_state(&low, &t, 0.12).distance_to_target;
        let d_high = estimate_state(&high, &t, 0.12).distance_to_target;
        assert!(d_high < d_low);
    }

    // ── Embedding / baseline ────────────────────────────────────────────

    #[test]
    fn personal_embedding_dimension_and_export() {
        let mut sim = PhysioSimulator::new(3);
        let obs = observe(&mut sim, 0.0, 0.5, 0.3);
        let e = PersonalStateEmbedding::from_observation(&obs);
        assert_eq!(e.features.len(), EMBEDDING_DIM);
        let ne = e.to_neural_embedding(Some("sub-01".into()));
        assert_eq!(ne.dimension, EMBEDDING_DIM);
    }

    #[test]
    fn baseline_deviation_grows_with_change() {
        let mut sim = PhysioSimulator::new(4);
        let mut base = PersonalBaseline::new();
        for i in 0..6 {
            let obs = observe(&mut sim, i as f64 * 10.0, 0.2, 0.1);
            base.update(&PersonalStateEmbedding::from_observation(&obs));
        }
        assert!(base.is_established());
        let calm = PersonalStateEmbedding::from_observation(&observe(&mut sim, 100.0, 0.2, 0.1));
        let aroused =
            PersonalStateEmbedding::from_observation(&observe(&mut sim, 110.0, 0.95, 0.1));
        assert!(base.deviation(&aroused) > base.deviation(&calm));
    }

    // ── Audit trail ─────────────────────────────────────────────────────

    #[test]
    fn audit_chain_verifies_and_detects_tampering() {
        let mut trail = AuditTrail::new();
        for i in 0..5 {
            trail.append(AuditEvent {
                index: i,
                timestamp_s: i as f64,
                kind: AuditKind::Stimulate,
                message: format!("step {i}"),
                intensity: 0.2,
                distance_to_target: 0.3,
                receipt_hashes: vec![],
                breaches: vec![],
            });
        }
        assert!(trail.verify_chain());
        // Tamper with an earlier event.
        trail.records[1].event.intensity = 0.9;
        assert!(!trail.verify_chain());
    }

    #[test]
    fn signed_audit_head_verifies() {
        let mut trail = AuditTrail::new();
        trail.append(AuditEvent {
            index: 0,
            timestamp_s: 0.0,
            kind: AuditKind::SessionStart,
            message: "start".into(),
            intensity: 0.0,
            distance_to_target: 1.0,
            receipt_hashes: vec![],
            breaches: vec![],
        });
        let signed = trail.sign_head();
        assert!(signed.verify());
        assert_eq!(signed.head_hash, trail.head_hash());
    }

    // ── Controller end-to-end ───────────────────────────────────────────

    #[test]
    fn controller_opens_with_session_start() {
        let c = new_controller(TargetState::relaxed());
        assert_eq!(c.audit().len(), 1);
        assert_eq!(c.audit().records[0].event.kind, AuditKind::SessionStart);
        assert_eq!(c.phase(), ControllerPhase::Baselining);
    }

    #[test]
    fn controller_reaches_target_and_delivers_verified_stimuli() {
        let mut c = new_controller(TargetState::relaxed());
        let mut sim = LoopSimulation::responsive(11, 10.0);
        sim.run(&mut c, 64);
        let report = c.report();
        assert!(
            report.num_receipts >= 1,
            "must deliver at least one stimulus"
        );
        assert!(report.all_receipts_verified, "all stimuli must verify");
        assert!(report.audit_chain_valid);
        assert!(report.goal_reached || report.safe_stopped);
        assert!(report.passes_acceptance());
    }

    #[test]
    fn controller_safe_stops_on_perturbation() {
        let mut c = new_controller(TargetState::relaxed());
        // Strongly responsive subject, then a big arousal spike mid-session.
        let mut sim = LoopSimulation::responsive(5, 10.0).with_perturbation(5, 0.9);
        sim.run(&mut c, 64);
        let report = c.report();
        assert!(
            report.safe_stopped,
            "perturbation must trigger a fail-safe stop"
        );
        assert!(!report.stop_reasons.is_empty());
        // Even a safe-stopped session must have delivered verified stimuli first.
        assert!(report.num_receipts >= 1);
        assert!(report.all_receipts_verified);
        assert!(report.passes_acceptance());
    }

    #[test]
    fn controller_never_exceeds_safety_intensity() {
        let mut c = new_controller(TargetState::gamma_entrainment());
        let mut sim = LoopSimulation::responsive(9, 10.0);
        let trace = sim.run(&mut c, 64);
        for step in &trace {
            for stim in &step.emitted {
                assert!(stim.waveform.params.intensity <= 0.6 + 1e-9);
            }
        }
    }

    #[test]
    fn terminal_phase_is_absorbing() {
        let mut c = new_controller(TargetState::relaxed());
        let mut sim = LoopSimulation::responsive(3, 10.0).with_perturbation(4, 0.95);
        sim.run(&mut c, 64);
        assert!(c.phase().is_terminal());
        let records_before = c.audit().len();
        // Stepping again after terminal must not add stimulation records.
        let mut s2 = PhysioSimulator::new(99);
        let obs = observe(&mut s2, 1000.0, 0.5, 0.3);
        let r = c.step(&obs);
        assert!(r.emitted.is_empty());
        assert_eq!(c.audit().len(), records_before);
    }

    #[test]
    fn report_serializes_to_json() {
        let mut c = new_controller(TargetState::relaxed());
        let mut sim = LoopSimulation::responsive(2, 10.0);
        sim.run(&mut c, 64);
        let json = serde_json::to_string_pretty(&c.report()).unwrap();
        assert!(json.contains("audit_head_hash"));
        let _: SessionReport = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn evidence_bundle_builds_and_chain_verifies() {
        let mut c = new_controller(TargetState::relaxed());
        let mut sim = LoopSimulation::responsive(11, 10.0);
        let trace = sim.run(&mut c, 64);
        let bundle = EvidenceBundle::build("relaxed", "demo", &trace, &c).signed();

        assert_eq!(bundle.schema_version, SCHEMA_VERSION);
        assert_eq!(bundle.steps.len(), trace.len());
        assert!(bundle.verify_chain(), "bundle step chain must verify");
        assert!(bundle.acceptance.passed);
        assert!(bundle.acceptance.verified_stimulus_delivered);
        // Signature present and well-formed.
        let sig = bundle.signature.as_ref().unwrap();
        assert_eq!(sig.head_hash, bundle.bundle_chain_head);

        // Round-trips through JSON (the wire format the UI consumes).
        let json = serde_json::to_string(&bundle).unwrap();
        let back: EvidenceBundle = serde_json::from_str(&json).unwrap();
        assert!(back.verify_chain());
        assert!(json.contains("schemaVersion"));
        assert!(json.contains("waveformSha256"));
    }

    #[test]
    fn evidence_bundle_chain_detects_tampering() {
        let mut c = new_controller(TargetState::relaxed());
        let mut sim = LoopSimulation::responsive(3, 10.0);
        let trace = sim.run(&mut c, 64);
        let mut bundle = EvidenceBundle::build("relaxed", "demo", &trace, &c);
        assert!(bundle.verify_chain());
        if bundle.steps.len() > 2 {
            bundle.steps[1].intensity += 0.2; // tamper
            assert!(!bundle.verify_chain());
        }
    }

    #[test]
    fn evidence_bundle_safe_stop_records_breach() {
        let mut c = new_controller(TargetState::relaxed());
        let mut sim = LoopSimulation::responsive(7, 10.0).with_perturbation(5, 0.9);
        let trace = sim.run(&mut c, 64);
        let bundle = EvidenceBundle::build("relaxed", "demo", &trace, &c);
        assert!(bundle.verify_chain());
        assert!(bundle.acceptance.stopped_safely_outside_envelope);
        let last = bundle.steps.last().unwrap();
        assert!(!last.envelope.within);
        assert!(!last.envelope.breaches.is_empty());
    }

    #[test]
    fn relaxed_label_when_calm() {
        let mut sim = PhysioSimulator::new(8);
        let obs = observe(&mut sim, 0.0, 0.05, 0.1);
        let est = estimate_state(&obs, &TargetState::relaxed(), 0.12);
        // Either at target (Rest) or inferred Rest from high relaxation.
        assert!(matches!(est.label, CognitiveState::Rest));
    }
}
