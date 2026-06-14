//! Target states, observations, and state estimation for the closed loop.

use ruv_neural_biosense::PhysioMetrics;
use ruv_neural_core::topology::CognitiveState;
use serde::{Deserialize, Serialize};

/// Compact neural features supplied by the upstream topology pipeline
/// (sensor → signal → graph → mincut → embed). All optional so the loop can
/// run on physiology alone when no neural front-end is attached.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NeuralFeatures {
    /// 40 Hz gamma entrainment index in `[0, 1]`: how strongly the cortical
    /// response is following the stimulus envelope (e.g. SSVEP/ASSR power at
    /// the entrainment frequency, normalized).
    pub gamma_index: f64,
    /// Alpha-band relaxation index in `[0, 1]`.
    pub alpha_index: f64,
    /// Global network integration in `[0, 1]` (e.g. from min-cut / efficiency).
    pub connectivity: f64,
}

impl NeuralFeatures {
    /// A neutral baseline (no entrainment, mid relaxation/connectivity).
    pub fn neutral() -> Self {
        Self { gamma_index: 0.1, alpha_index: 0.4, connectivity: 0.4 }
    }
}

/// The state the controller is trying to drive the subject toward.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TargetState {
    /// Human-readable cognitive-state label this protocol aims for.
    pub label: CognitiveState,
    /// Desired relaxation index in `[0, 1]`.
    pub target_relaxation: f64,
    /// Desired (ceiling) arousal index in `[0, 1]`.
    pub target_arousal: f64,
    /// Whether 40 Hz gamma entrainment is part of the goal (drives whether the
    /// neural gamma index contributes to the distance metric).
    pub gamma_entrainment: bool,
    /// Desired gamma entrainment index when `gamma_entrainment` is set.
    pub target_gamma: f64,
}

impl TargetState {
    /// A calm / relaxed-rest target: high relaxation, low arousal, no gamma.
    pub fn relaxed() -> Self {
        Self {
            label: CognitiveState::Rest,
            target_relaxation: 0.8,
            target_arousal: 0.2,
            gamma_entrainment: false,
            target_gamma: 0.0,
        }
    }

    /// A focused-attention target with mild arousal and 40 Hz entrainment.
    pub fn focused() -> Self {
        Self {
            label: CognitiveState::Focused,
            target_relaxation: 0.5,
            target_arousal: 0.45,
            gamma_entrainment: true,
            target_gamma: 0.6,
        }
    }

    /// A pure 40 Hz gamma-entrainment target (the GENUS research paradigm).
    pub fn gamma_entrainment() -> Self {
        Self {
            label: CognitiveState::Focused,
            target_relaxation: 0.5,
            target_arousal: 0.4,
            gamma_entrainment: true,
            target_gamma: 0.7,
        }
    }
}

/// A fused snapshot of the subject at one time step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateObservation {
    /// Session-relative timestamp (s).
    pub timestamp_s: f64,
    /// Physiological metrics (always present).
    pub physio: PhysioMetrics,
    /// Neural features, when a neural front-end is attached.
    pub neural: Option<NeuralFeatures>,
}

impl StateObservation {
    /// Construct from physiology only.
    pub fn from_physio(physio: PhysioMetrics) -> Self {
        let timestamp_s = physio.timestamp_s;
        Self { timestamp_s, physio, neural: None }
    }

    /// Attach neural features.
    pub fn with_neural(mut self, neural: NeuralFeatures) -> Self {
        self.neural = Some(neural);
        self
    }
}

/// The controller's estimate of where the subject is relative to the target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateEstimate {
    /// Best-guess cognitive-state label.
    pub label: CognitiveState,
    /// Distance to the target state in `[0, 1]` (0 = at target).
    pub distance_to_target: f64,
    /// Whether this estimate is at/under the completion threshold.
    pub at_target: bool,
}

/// Estimate state and distance-to-target from an observation.
///
/// The distance is a weighted L1 over the dimensions the target cares about:
/// relaxation, arousal, and (optionally) gamma entrainment.
pub fn estimate_state(
    obs: &StateObservation,
    target: &TargetState,
    completion_threshold: f64,
) -> StateEstimate {
    let relax = obs.physio.relaxation_index;
    let arousal = obs.physio.arousal_index;

    let mut num = 0.0;
    let mut den = 0.0;

    // Relaxation and arousal always contribute.
    num += (relax - target.target_relaxation).abs();
    den += 1.0;
    num += (arousal - target.target_arousal).abs();
    den += 1.0;

    if target.gamma_entrainment {
        let gamma = obs.neural.map(|n| n.gamma_index).unwrap_or(0.0);
        // Gamma is the *primary* objective for entrainment targets → weight 2.
        num += 2.0 * (gamma - target.target_gamma).abs();
        den += 2.0;
    }

    let distance = (num / den).clamp(0.0, 1.0);

    // Label heuristic: prefer the target label when close, else infer from
    // autonomic balance.
    let label = if distance <= completion_threshold {
        target.label
    } else if arousal > 0.65 {
        CognitiveState::Stressed
    } else if relax > 0.65 {
        CognitiveState::Rest
    } else {
        CognitiveState::Unknown
    };

    StateEstimate {
        label,
        distance_to_target: distance,
        at_target: distance <= completion_threshold,
    }
}
