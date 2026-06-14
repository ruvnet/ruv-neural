//! The safety envelope: the allowed region of physiological response within
//! which stimulation may continue. Leaving the envelope triggers a fail-safe
//! stop.
//!
//! The envelope encodes two complementary ideas:
//!   1. **Absolute bounds** — hard physiological limits (e.g. heart rate,
//!      motion, protected sleep) that must never be exceeded.
//!   2. **Response divergence** — the response must not move *away* from the
//!      target beyond a tolerance. This is what lets the loop "stop safely
//!      when response moves outside the allowed envelope" even when every
//!      absolute reading is individually benign.
//!
//! See `docs/adr/0007-safety-envelope.md`.

use ruv_neural_biosense::{sleep, PhysioMetrics};
use serde::{Deserialize, Serialize};

use crate::state::{StateEstimate, TargetState};

/// A specific reason the safety envelope was breached.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BreachReason {
    /// Heart rate above the absolute ceiling.
    HeartRateHigh { bpm: f64, max: f64 },
    /// Heart rate below the absolute floor.
    HeartRateLow { bpm: f64, min: f64 },
    /// Arousal index above the ceiling.
    ArousalHigh { value: f64, max: f64 },
    /// Excessive motion (subject agitated / not settled).
    ExcessiveMotion { movement_index: f64, max: f64 },
    /// Subject appears to be in protected (deep) sleep.
    SleepInhibited,
    /// The response is diverging from the target rather than converging.
    ResponseDiverging { delta: f64, tolerance: f64 },
    /// Required data was missing to make a safety judgement.
    MissingData(String),
}

/// Result of evaluating the safety envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnvelopeStatus {
    /// Response is within the allowed envelope; stimulation may continue.
    Within,
    /// Response left the envelope; stimulation must stop.
    Breach(Vec<BreachReason>),
}

impl EnvelopeStatus {
    /// True if the envelope was breached.
    pub fn is_breach(&self) -> bool {
        matches!(self, EnvelopeStatus::Breach(_))
    }

    /// Breach reasons, if any.
    pub fn reasons(&self) -> &[BreachReason] {
        match self {
            EnvelopeStatus::Within => &[],
            EnvelopeStatus::Breach(r) => r,
        }
    }
}

/// Configurable safety envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetyEnvelope {
    /// Absolute heart-rate ceiling (bpm).
    pub max_hr_bpm: f64,
    /// Absolute heart-rate floor (bpm).
    pub min_hr_bpm: f64,
    /// Arousal-index ceiling.
    pub max_arousal: f64,
    /// Motion movement-index ceiling (g).
    pub max_movement_index: f64,
    /// Tolerance for response divergence: the distance-to-target may rise by at
    /// most this much from the running best before a divergence breach.
    pub divergence_tolerance: f64,
    /// Whether to inhibit stimulation during protected sleep.
    pub respect_sleep: bool,
}

impl Default for SafetyEnvelope {
    /// Conservative wellness-grade envelope.
    fn default() -> Self {
        Self {
            max_hr_bpm: 100.0,
            min_hr_bpm: 45.0,
            max_arousal: 0.75,
            max_movement_index: 0.15,
            divergence_tolerance: 0.15,
            respect_sleep: true,
        }
    }
}

impl SafetyEnvelope {
    /// Evaluate the envelope against the current physiology, the current state
    /// estimate, and the best (lowest) distance-to-target seen so far this
    /// session. `best_distance` enables divergence detection.
    pub fn evaluate(
        &self,
        physio: &PhysioMetrics,
        estimate: &StateEstimate,
        best_distance: f64,
        _target: &TargetState,
    ) -> EnvelopeStatus {
        let mut breaches = Vec::new();

        if let Some(h) = &physio.hrv {
            if h.mean_hr_bpm > self.max_hr_bpm {
                breaches.push(BreachReason::HeartRateHigh {
                    bpm: h.mean_hr_bpm,
                    max: self.max_hr_bpm,
                });
            }
            if h.mean_hr_bpm < self.min_hr_bpm {
                breaches.push(BreachReason::HeartRateLow {
                    bpm: h.mean_hr_bpm,
                    min: self.min_hr_bpm,
                });
            }
        }

        if physio.arousal_index > self.max_arousal {
            breaches.push(BreachReason::ArousalHigh {
                value: physio.arousal_index,
                max: self.max_arousal,
            });
        }

        if let Some(m) = &physio.motion {
            if m.movement_index > self.max_movement_index {
                breaches.push(BreachReason::ExcessiveMotion {
                    movement_index: m.movement_index,
                    max: self.max_movement_index,
                });
            }
        }

        if self.respect_sleep {
            if let (Some(h), Some(m)) = (&physio.hrv, &physio.motion) {
                let stage = sleep::estimate_from_metrics(h.mean_hr_bpm, h.rmssd_ms, m);
                if sleep::stimulation_inhibited(stage) {
                    breaches.push(BreachReason::SleepInhibited);
                }
            }
        }

        // Divergence: response moving away from the target beyond tolerance.
        let delta = estimate.distance_to_target - best_distance;
        if delta > self.divergence_tolerance {
            breaches.push(BreachReason::ResponseDiverging {
                delta,
                tolerance: self.divergence_tolerance,
            });
        }

        if breaches.is_empty() {
            EnvelopeStatus::Within
        } else {
            EnvelopeStatus::Breach(breaches)
        }
    }
}
