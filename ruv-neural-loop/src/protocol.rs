//! Protocol selection and conservative dosing.
//!
//! A protocol decides *which* modalities to drive and at *what* intensity,
//! given the target and the live state estimate. Dosing is deliberately
//! conservative: intensity titrates **up slowly** while the response keeps
//! improving and is held or **backed off quickly** when it does not — the
//! "titrate up gently, retreat fast" principle (`docs/adr/0008-protocol-dosing.md`).

use ruv_neural_stim::{Modality, StimulusParams};
use serde::{Deserialize, Serialize};

use crate::state::{StateEstimate, TargetState};

/// A planned stimulus across one or more modalities for the next step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StimulusPlan {
    /// The commanded stimuli (pre-safety; the controller clamps them).
    pub stimuli: Vec<StimulusParams>,
    /// The commanded intensity this plan settled on, in `[0, 1]`.
    pub intensity: f64,
    /// Whether this plan intends active stimulation (false ⇒ rest/hold step).
    pub active: bool,
}

impl StimulusPlan {
    /// An explicit rest/no-stimulation plan.
    pub fn rest() -> Self {
        Self {
            stimuli: Vec::new(),
            intensity: 0.0,
            active: false,
        }
    }
}

/// Conservative dosing policy parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DosingPolicy {
    /// Starting intensity at session onset.
    pub start_intensity: f64,
    /// Per-step increase while improving.
    pub step_up: f64,
    /// Per-step decrease when not improving.
    pub step_down: f64,
    /// Hard ceiling the protocol may command (still subject to safety clamp).
    pub ceiling: f64,
    /// How much the distance-to-target must drop to count as "improving".
    pub improvement_epsilon: f64,
    /// How much the (smoothed) distance must rise to count as genuinely
    /// worsening (rather than noise) before the dose is backed off.
    pub worsen_threshold: f64,
}

impl Default for DosingPolicy {
    fn default() -> Self {
        Self {
            start_intensity: 0.15,
            step_up: 0.05,
            step_down: 0.10,
            ceiling: 0.5,
            improvement_epsilon: 0.005,
            worsen_threshold: 0.05,
        }
    }
}

/// A protocol maps (target, estimate, history) → next [`StimulusPlan`].
pub trait Protocol {
    /// Human-readable protocol name.
    fn name(&self) -> &str;

    /// Modalities this protocol drives.
    fn modalities(&self) -> &[Modality];

    /// Decide the next plan. `prev_intensity` is the last commanded intensity,
    /// `prev_distance` and `curr_distance` allow improvement detection.
    fn next_plan(
        &self,
        target: &TargetState,
        estimate: &StateEstimate,
        prev_intensity: f64,
        prev_distance: Option<f64>,
        step_duration_s: f64,
    ) -> StimulusPlan;
}

/// The 40 Hz GENUS gamma-entrainment protocol across selected sensory channels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GammaEntrainmentProtocol {
    /// Modalities to drive (e.g. audio + haptic; light requires a screen).
    pub modalities: Vec<Modality>,
    /// Dosing policy.
    pub dosing: DosingPolicy,
}

impl GammaEntrainmentProtocol {
    /// Default: audio + haptic 40 Hz (the channels safe without a
    /// photosensitivity screen).
    pub fn audio_haptic() -> Self {
        Self {
            modalities: vec![Modality::Audio, Modality::Haptic],
            dosing: DosingPolicy::default(),
        }
    }

    /// All three modalities (light included — requires a cleared screen, else
    /// the light channel is clamped to zero by the safety layer).
    pub fn multimodal() -> Self {
        Self {
            modalities: Modality::ALL.to_vec(),
            dosing: DosingPolicy::default(),
        }
    }

    /// Compute the next intensity under the conservative titration rule.
    fn titrate(
        &self,
        prev_intensity: f64,
        curr_distance: f64,
        prev_distance: Option<f64>,
        at_target: bool,
    ) -> f64 {
        let d = &self.dosing;
        if prev_intensity <= 0.0 {
            return d.start_intensity.min(d.ceiling);
        }
        // At target: hold, do not keep escalating.
        if at_target {
            return prev_intensity.min(d.ceiling);
        }
        match prev_distance {
            // Clearly worsening (beyond noise) → back off, faster than we climb.
            Some(prev) if curr_distance > prev + d.worsen_threshold => {
                (prev_intensity - d.step_down).max(0.0)
            }
            // Otherwise (improving or on a noisy plateau) keep titrating up
            // gently toward the ceiling. The safety envelope, not the dosing
            // rule, is responsible for catching genuine divergence.
            Some(_) => (prev_intensity + d.step_up).min(d.ceiling),
            None => prev_intensity.min(d.ceiling),
        }
    }
}

impl Protocol for GammaEntrainmentProtocol {
    fn name(&self) -> &str {
        "gamma-40hz-entrainment"
    }

    fn modalities(&self) -> &[Modality] {
        &self.modalities
    }

    fn next_plan(
        &self,
        target: &TargetState,
        estimate: &StateEstimate,
        prev_intensity: f64,
        prev_distance: Option<f64>,
        step_duration_s: f64,
    ) -> StimulusPlan {
        let intensity = self.titrate(
            prev_intensity,
            estimate.distance_to_target,
            prev_distance,
            estimate.at_target,
        );

        if intensity <= 0.0 {
            return StimulusPlan::rest();
        }

        let envelope_hz = if target.gamma_entrainment {
            ruv_neural_stim::GAMMA_ENTRAINMENT_HZ
        } else {
            // Relaxation targets use a slower ~10 Hz alpha-range envelope.
            10.0
        };

        let stimuli = self
            .modalities
            .iter()
            .map(|&m| {
                let mut p =
                    StimulusParams::gamma_40hz(m, step_duration_s).with_intensity(intensity);
                p.envelope_hz = envelope_hz;
                p
            })
            .collect();

        StimulusPlan {
            stimuli,
            intensity,
            active: true,
        }
    }
}
