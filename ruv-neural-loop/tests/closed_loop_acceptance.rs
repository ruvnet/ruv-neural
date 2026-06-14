//! # Closed-loop neuromodulation acceptance test
//!
//! This is the platform's headline acceptance criterion, stated in the task:
//!
//! > Prove the system can **identify a target state**, **deliver a verified
//! > stimulus**, **measure a response**, and **stop safely** when the response
//! > moves outside the allowed envelope.
//!
//! Each of the four clauses is asserted explicitly below, across multiple
//! modalities, using only the public API of the workspace crates.

use ruv_neural_biosense::PhysioMetrics;
use ruv_neural_core::topology::CognitiveState;
use ruv_neural_loop::*;
use ruv_neural_stim::{Modality, SensorySafetyLimits, StimulusGenerator};

fn controller_for(target: TargetState, generator: StimulusGenerator) -> ClosedLoopController {
    ClosedLoopController::new(
        ControllerConfig::default(),
        target,
        generator,
        SafetyEnvelope::default(),
        Box::new(GammaEntrainmentProtocol::audio_haptic()),
    )
}

/// Clause 1 + 2 + 3: identify a target, deliver verified stimuli, measure a
/// response — and converge to the target.
#[test]
fn acceptance_identify_deliver_measure_converge() {
    let mut controller = controller_for(TargetState::relaxed(), StimulusGenerator::conservative());

    // (1) Identify a target state.
    assert_eq!(controller.target().label, CognitiveState::Rest);

    let mut sim = LoopSimulation::responsive(2024, 10.0);
    let trace = sim.run(&mut controller, 64);
    let report = controller.report();

    // (2) Deliver a *verified* stimulus — at least one, all cryptographically
    // verified, each bound to its waveform by SHA-256.
    let mut verified_emitted = 0;
    for step in &trace {
        for stim in &step.emitted {
            assert!(stim.receipt.verified, "every delivered stimulus must verify");
            assert!(stim.receipt.matches(&stim.waveform), "receipt must bind to waveform");
            assert!(
                (stim.receipt.measured_envelope_hz - stim.waveform.params.envelope_hz).abs() <= 2.0,
                "measured entrainment frequency must match the command"
            );
            verified_emitted += 1;
        }
    }
    assert!(verified_emitted >= 1, "the loop must deliver verified stimuli");
    assert_eq!(report.num_receipts, verified_emitted as u64);
    assert!(report.all_receipts_verified);

    // (3) Measure a response: the controller produced a personal state
    // embedding (ruVector) every step and converged toward the target.
    assert!(report.total_steps >= 1);
    assert!(
        report.best_distance < 0.5,
        "the measured response must move toward the target (best distance {:.3})",
        report.best_distance
    );
    assert!(report.goal_reached, "a responsive subject should reach the target");

    // Evidence integrity: the audit chain verifies and records the full story.
    assert!(report.audit_chain_valid);
    assert!(controller.audit().count_kind(&AuditKind::Stimulate) >= 1);
    assert!(controller.audit().count_kind(&AuditKind::Complete) >= 1);

    assert!(report.passes_acceptance());
    assert_eq!(report.verdict(), "TARGET-REACHED");
}

/// Clause 4: stop safely when the response moves outside the allowed envelope.
#[test]
fn acceptance_safe_stop_outside_envelope() {
    let mut controller = controller_for(TargetState::relaxed(), StimulusGenerator::conservative());

    // A subject who is settling nicely, then is perturbed (arousal spike) at
    // step 5 — pushing the response back outside the envelope.
    let mut sim = LoopSimulation::responsive(7, 10.0).with_perturbation(5, 0.9);
    let trace = sim.run(&mut controller, 64);
    let report = controller.report();

    // The loop delivered verified stimuli before the perturbation...
    assert!(report.num_receipts >= 1);
    assert!(report.all_receipts_verified);

    // ...and then stopped safely when the response left the envelope.
    assert!(report.safe_stopped, "perturbation must trigger a fail-safe stop");
    assert_eq!(controller.phase(), ControllerPhase::SafeStopped);
    assert!(!report.stop_reasons.is_empty(), "a stop must record its reasons");

    // After the stop, the controller emits no further stimulation.
    let last = trace.last().unwrap();
    assert!(last.emitted.is_empty());
    assert!(last.envelope.is_breach());
    assert_eq!(last.audit_kind, AuditKind::SafeStop);

    assert!(report.audit_chain_valid);
    assert!(report.passes_acceptance());
    assert_eq!(report.verdict(), "SAFE-STOPPED");
}

/// The acceptance criterion holds across every modality combination.
#[test]
fn acceptance_multimodal_coverage() {
    let combos: Vec<(&str, GammaEntrainmentProtocol)> = vec![
        ("audio+haptic", GammaEntrainmentProtocol::audio_haptic()),
        ("multimodal", GammaEntrainmentProtocol::multimodal()),
    ];

    for (name, protocol) in combos {
        // Light is included in `multimodal`; clear the photosensitivity screen
        // so the light channel is permitted (still contrast-capped).
        let generator = StimulusGenerator::new(SensorySafetyLimits::screened());
        let mut controller = ClosedLoopController::new(
            ControllerConfig::default(),
            TargetState::gamma_entrainment(),
            generator,
            SafetyEnvelope::default(),
            Box::new(protocol),
        );

        let mut sim = LoopSimulation::responsive(1234, 10.0);
        sim.run(&mut controller, 64);
        let report = controller.report();

        assert!(
            report.passes_acceptance(),
            "{name}: must satisfy the acceptance criterion (verdict {})",
            report.verdict()
        );
        assert!(report.all_receipts_verified, "{name}: stimuli must verify");
    }
}

/// The fused personal state embedding (ruVector) is well-formed and exportable
/// to the core neural-embedding / RVF ecosystem.
#[test]
fn acceptance_personal_state_embedding_exports() {
    let mut sim = ruv_neural_biosense::PhysioSimulator::new(99);
    let window = sim.window(0.0, 10.0, 0.3);
    let physio = PhysioMetrics::from_window(&window).unwrap();
    let obs = StateObservation::from_physio(physio).with_neural(NeuralFeatures {
        gamma_index: 0.5,
        alpha_index: 0.6,
        connectivity: 0.5,
    });

    let embedding = PersonalStateEmbedding::from_observation(&obs);
    assert_eq!(embedding.features.len(), EMBEDDING_DIM);
    assert_eq!(FEATURE_NAMES.len(), EMBEDDING_DIM);

    let neural = embedding.to_neural_embedding(Some("subject-acceptance".into()));
    assert_eq!(neural.dimension, EMBEDDING_DIM);
    assert_eq!(neural.metadata.embedding_method, "personal-state-fusion");
}

/// A session can be cryptographically attested end-to-end (Ed25519 over the
/// tamper-evident audit-chain head), independently verifiable by a third party.
#[test]
fn acceptance_session_is_independently_attestable() {
    let mut controller = controller_for(TargetState::relaxed(), StimulusGenerator::conservative());
    let mut sim = LoopSimulation::responsive(31337, 10.0);
    sim.run(&mut controller, 64);

    assert!(controller.audit().verify_chain());
    let signed = controller.sign_session();
    assert!(signed.verify(), "the signed session head must verify");
    assert_eq!(signed.head_hash, controller.audit().head_hash());

    // Tampering with the trail after signing invalidates the chain.
    let mut tampered = controller.audit().clone();
    if tampered.records.len() > 2 {
        tampered.records[1].event.intensity += 0.5;
        assert!(!tampered.verify_chain());
    }
}

/// Modality-tagging sanity: receipts carry the modality and a stable cortical
/// target label for downstream evidence/UI.
#[test]
fn acceptance_modality_metadata_is_present() {
    assert_eq!(Modality::Audio.cortical_target(), "auditory cortex");
    assert_eq!(Modality::Light.cortical_target(), "visual cortex");
    assert_eq!(Modality::Haptic.cortical_target(), "somatosensory cortex");

    let gen = StimulusGenerator::conservative();
    let params = ruv_neural_stim::StimulusParams::gamma_40hz(Modality::Haptic, 10.0);
    let stim = gen.generate_clamped(&params, 0.0).unwrap();
    assert_eq!(stim.waveform.params.modality, Modality::Haptic);
    assert!(stim.receipt.verified);
}
