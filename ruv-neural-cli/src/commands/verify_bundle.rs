//! Reference verifier for Ruflo evidence bundles (ADR-0014 §10, §17).
//!
//! This is the **Rust** side of cross-language verifier parity: it recomputes
//! the bundle's per-step hash chain and checks the Ed25519 signature and
//! acceptance result, exactly as the in-browser TypeScript verifier does — so a
//! bundle can be verified offline with the binary, and the two implementations
//! agree on real evidence.

use ruv_neural_loop::EvidenceBundle;
use std::path::PathBuf;

/// Run the verify-bundle command.
pub fn run(path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let json = std::fs::read_to_string(&path)?;
    let bundle: EvidenceBundle = serde_json::from_str(&json)?;

    let chain_ok = bundle.verify_chain();
    let sig = bundle.verify_signature();
    let accept = bundle.acceptance.passed;

    println!("=== rUv Neural — Evidence Bundle Verification ===\n");
    println!("  Schema:        {}", bundle.schema_version);
    println!("  Session:       {}", bundle.session_id);
    println!("  Target:        {}  ({})", bundle.target_state, bundle.protocol);
    println!("  Steps:         {}", bundle.steps.len());
    println!("  Receipts:      {}", bundle.report.num_receipts);
    println!();
    println!("  Hash chain:    {}", if chain_ok { "VALID" } else { "INVALID" });
    println!(
        "  Signature:     {}",
        match sig {
            Some(true) => "Ed25519 OK",
            Some(false) => "INVALID",
            None => "unsigned",
        }
    );
    println!("  Acceptance:    {}", if accept { "PASS" } else { "FAIL" });

    let ok = chain_ok && accept && sig != Some(false);
    println!("\n  VERDICT: {}", if ok { "PASS" } else { "FAIL" });

    if !ok {
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruv_neural_loop::{
        ClosedLoopController, ControllerConfig, GammaEntrainmentProtocol, LoopSimulation,
        SafetyEnvelope, TargetState,
    };
    use ruv_neural_stim::StimulusGenerator;

    fn write_bundle(perturb: Option<u64>) -> PathBuf {
        let mut c = ClosedLoopController::new(
            ControllerConfig::default(),
            TargetState::relaxed(),
            StimulusGenerator::conservative(),
            SafetyEnvelope::default(),
            Box::new(GammaEntrainmentProtocol::audio_haptic()),
        );
        let mut sim = LoopSimulation::responsive(11, 10.0);
        if let Some(p) = perturb {
            sim = sim.with_perturbation(p, 0.9);
        }
        let trace = sim.run(&mut c, 64);
        let bundle = EvidenceBundle::build("relaxed", "demo", &trace, &c).signed();
        let path = std::env::temp_dir().join(format!("ruflo_verify_{:?}.json", perturb));
        std::fs::write(&path, serde_json::to_string(&bundle).unwrap()).unwrap();
        path
    }

    #[test]
    fn verifies_a_generated_bundle() {
        let path = write_bundle(None);
        assert!(run(path.clone()).is_ok());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn detects_tampering() {
        let path = write_bundle(None);
        let mut bundle: EvidenceBundle =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        bundle.steps[2].intensity += 0.3;
        assert!(!bundle.verify_chain());
        let _ = std::fs::remove_file(path);
    }
}
