//! Closed-loop sensory neuromodulation session (Ruflo).
//!
//! Runs a deterministic simulated closed-loop session — identify a target
//! state, deliver verified 40 Hz stimuli, measure the physiological response,
//! and stop safely when the response leaves the allowed envelope — then prints
//! a session report and optionally writes the report and the tamper-evident
//! audit trail to disk.

use ruv_neural_loop::{
    ClosedLoopController, ControllerConfig, GammaEntrainmentProtocol, LoopSimulation,
    SafetyEnvelope, TargetState,
};
use ruv_neural_stim::{SensorySafetyLimits, StimulusGenerator};
use std::path::PathBuf;

/// Run the neuromod command.
#[allow(clippy::too_many_arguments)]
pub fn run(
    target: &str,
    protocol: &str,
    steps: u64,
    seed: u64,
    perturb: Option<u64>,
    screened: bool,
    report_out: Option<PathBuf>,
    audit_out: Option<PathBuf>,
    sign: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let target_state = match target {
        "relaxed" | "rest" => TargetState::relaxed(),
        "focused" => TargetState::focused(),
        "gamma" | "gamma-entrainment" => TargetState::gamma_entrainment(),
        other => return Err(format!("unknown target '{other}' (relaxed|focused|gamma)").into()),
    };

    let proto = match protocol {
        "audio-haptic" | "audio_haptic" => GammaEntrainmentProtocol::audio_haptic(),
        "multimodal" => GammaEntrainmentProtocol::multimodal(),
        other => {
            return Err(format!("unknown protocol '{other}' (audio-haptic|multimodal)").into())
        }
    };

    let limits = if screened {
        SensorySafetyLimits::screened()
    } else {
        SensorySafetyLimits::default()
    };

    let mut controller = ClosedLoopController::new(
        ControllerConfig::default(),
        target_state,
        StimulusGenerator::new(limits),
        SafetyEnvelope::default(),
        Box::new(proto),
    );

    let mut sim = LoopSimulation::responsive(seed, ControllerConfig::default().step_duration_s);
    if let Some(step) = perturb {
        sim = sim.with_perturbation(step, 0.9);
    }

    println!("=== rUv Neural — Closed-Loop Neuromodulation (Ruflo) ===\n");
    println!("  Target:    {target}  ({:?})", controller.target().label);
    println!("  Protocol:  {protocol}");
    println!(
        "  Channels:  safe external sensory only (light / audio / haptic, 40 Hz)"
    );
    println!(
        "  Screen:    photosensitivity {}",
        if screened { "CLEARED" } else { "not cleared (light disabled)" }
    );
    if let Some(p) = perturb {
        println!("  Perturb:   arousal spike injected at step {p}");
    }
    println!();

    let trace = sim.run(&mut controller, steps);

    // Per-step trace.
    for (i, step) in trace.iter().enumerate() {
        let breach = if step.envelope.is_breach() { "  [ENVELOPE BREACH]" } else { "" };
        println!(
            "  step {:>2} | {:<11} | dist {:.3} | intensity {:.2} | {} stim{}",
            i + 1,
            format!("{:?}", step.phase),
            step.estimate.distance_to_target,
            step.plan.intensity,
            step.emitted.len(),
            breach
        );
    }
    println!();

    let report = controller.report();
    println!("  ── Session report ──");
    println!("  Verdict:            {}", report.verdict());
    println!("  Total steps:        {}", report.total_steps);
    println!("  Stimulation steps:  {}", report.stimulate_steps);
    println!("  Total stim time:    {:.1} s", report.total_stimulation_s);
    println!("  Peak intensity:     {:.2}", report.peak_intensity);
    println!("  Best distance:      {:.3}", report.best_distance);
    println!("  Final distance:     {:.3}", report.final_distance);
    println!(
        "  Receipts:           {} ({})",
        report.num_receipts,
        if report.all_receipts_verified { "all verified" } else { "VERIFICATION FAILED" }
    );
    if report.safe_stopped {
        println!("  Stop reasons:");
        for r in &report.stop_reasons {
            println!("    - {r:?}");
        }
    }
    println!(
        "  Audit chain:        {} ({} records, head {}...)",
        if report.audit_chain_valid { "VALID" } else { "INVALID" },
        report.audit_records,
        &report.audit_head_hash[..16.min(report.audit_head_hash.len())]
    );
    println!(
        "  Acceptance test:    {}",
        if report.passes_acceptance() { "PASS" } else { "FAIL" }
    );

    if sign {
        let signed = controller.sign_session();
        println!(
            "  Signed head:        {} (sig {}...)",
            if signed.verify() { "Ed25519 OK" } else { "INVALID" },
            &signed.signature[..16]
        );
    }

    if let Some(path) = report_out {
        std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
        println!("\n  Report written to {}", path.display());
    }
    if let Some(path) = audit_out {
        std::fs::write(&path, serde_json::to_string_pretty(controller.audit())?)?;
        println!("  Audit trail written to {}", path.display());
    }

    if !report.passes_acceptance() {
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neuromod_relaxed_session_passes() {
        run("relaxed", "audio-haptic", 64, 7, None, false, None, None, false).unwrap();
    }

    #[test]
    fn neuromod_safe_stop_session() {
        run("relaxed", "audio-haptic", 64, 7, Some(5), false, None, None, false).unwrap();
    }

    #[test]
    fn neuromod_unknown_target_errors() {
        assert!(run("nope", "audio-haptic", 64, 7, None, false, None, None, false).is_err());
    }
}
