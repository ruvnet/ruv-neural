//! A deterministic end-to-end simulation harness that *closes the loop*: the
//! simulated subject's physiology responds to the stimulation the controller
//! commands. This is what the acceptance test and the CLI demo drive.
//!
//! The world model is intentionally simple and transparent:
//!   * stimulation nudges **arousal** toward the target's arousal floor
//!     (relaxation), and raises the **gamma** entrainment index;
//!   * both quantities drift gently back toward baseline without stimulation;
//!   * an optional **perturbation** injects an arousal spike at a chosen step
//!     to exercise the fail-safe stop.

use ruv_neural_biosense::{PhysioMetrics, PhysioSimulator};

use crate::controller::{ClosedLoopController, ControllerPhase, StepResult};
use crate::state::{NeuralFeatures, StateObservation};

/// Parameters of the simulated closed-loop world.
#[derive(Debug, Clone)]
pub struct LoopSimulation {
    physio: PhysioSimulator,
    /// Current arousal in `[0, 1]`.
    pub arousal: f64,
    /// Current gamma entrainment index in `[0, 1]`.
    pub gamma: f64,
    /// How strongly the subject responds to stimulation (per unit intensity).
    pub responsiveness: f64,
    /// Natural drift of arousal back toward `baseline_arousal` per step.
    pub drift: f64,
    /// Resting arousal the subject drifts toward without stimulation.
    pub baseline_arousal: f64,
    /// Gamma decay toward 0.1 per step without stimulation.
    pub gamma_decay: f64,
    /// Window/step length (s); must match the controller step duration.
    pub window_s: f64,
    /// Optional `(step_index, arousal_spike)` perturbation.
    pub perturbation: Option<(u64, f64)>,
}

impl LoopSimulation {
    /// A subject starting moderately aroused who responds to stimulation.
    pub fn responsive(seed: u64, window_s: f64) -> Self {
        Self {
            physio: PhysioSimulator::new(seed),
            arousal: 0.6,
            gamma: 0.1,
            responsiveness: 0.6,
            drift: 0.02,
            baseline_arousal: 0.55,
            gamma_decay: 0.15,
            window_s,
            perturbation: None,
        }
    }

    /// Add a perturbation that spikes arousal by `magnitude` at `step`.
    pub fn with_perturbation(mut self, step: u64, magnitude: f64) -> Self {
        self.perturbation = Some((step, magnitude));
        self
    }

    /// Build the current observation from the world state.
    fn observe(&mut self, t: f64) -> StateObservation {
        let w = self.physio.window(t, self.window_s, self.arousal);
        let physio = PhysioMetrics::from_window(&w).expect("simulated window has data");
        let neural = NeuralFeatures {
            gamma_index: self.gamma.clamp(0.0, 1.0),
            alpha_index: (1.0 - self.arousal).clamp(0.0, 1.0),
            connectivity: (0.4 + 0.3 * self.gamma).clamp(0.0, 1.0),
        };
        StateObservation::from_physio(physio).with_neural(neural)
    }

    /// Apply the physiological response to a commanded step.
    fn respond(&mut self, result: &StepResult, arousal_floor: f64) {
        // Realized drive = max emitted intensity this step.
        let drive = result
            .emitted
            .iter()
            .map(|s| s.waveform.params.intensity)
            .fold(0.0_f64, f64::max);

        // Relaxation: pull arousal toward the floor proportional to drive.
        self.arousal += -self.responsiveness * drive * (self.arousal - arousal_floor);
        // Natural drift back toward baseline.
        self.arousal += self.drift * (self.baseline_arousal - self.arousal);

        // Gamma entrainment rises with drive, decays otherwise.
        self.gamma += self.responsiveness * drive * (1.0 - self.gamma);
        self.gamma -= self.gamma_decay * (self.gamma - 0.1);

        self.arousal = self.arousal.clamp(0.0, 1.0);
        self.gamma = self.gamma.clamp(0.0, 1.0);
    }

    /// Drive the controller until it reaches a terminal phase or `max_ticks`
    /// elapse, returning the per-step trace.
    pub fn run(
        &mut self,
        controller: &mut ClosedLoopController,
        max_ticks: u64,
    ) -> Vec<StepResult> {
        let arousal_floor = controller.target().target_arousal;
        let mut trace = Vec::new();
        let mut t = 0.0;

        for tick in 0..max_ticks {
            // Apply perturbation just before observing, if scheduled.
            if let Some((step, mag)) = self.perturbation {
                if step == tick {
                    self.arousal = (self.arousal + mag).clamp(0.0, 1.0);
                }
            }

            let obs = self.observe(t);
            let result = controller.step(&obs);
            let terminal = result.phase.is_terminal();
            self.respond(&result, arousal_floor);
            trace.push(result);

            if terminal {
                break;
            }
            t += self.window_s;
        }

        // Ensure we never report a non-terminal dangling session.
        if controller.phase() == ControllerPhase::Stimulating
            || controller.phase() == ControllerPhase::Holding
            || controller.phase() == ControllerPhase::Baselining
        {
            // Out of ticks without completing — the controller itself aborts on
            // its own max_steps budget; nothing else to do here.
        }

        trace
    }
}
