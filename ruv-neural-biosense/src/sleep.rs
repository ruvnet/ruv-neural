//! A lightweight, transparent sleep-stage **proxy** from heart rate and motion.
//!
//! This is **not** a polysomnography-grade scorer (which needs EEG/EOG/EMG).
//! It is a conservative actigraphy + cardiac heuristic used only to gate
//! stimulation (e.g. never stimulate during apparent deep sleep) and to label
//! sessions. It reuses the canonical [`SleepStage`] enum from the core crate.

use ruv_neural_core::topology::SleepStage;
use serde::{Deserialize, Serialize};

use crate::motion::MotionMetrics;

/// Inputs to the sleep proxy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SleepProxyInput {
    /// Mean heart rate (bpm).
    pub mean_hr_bpm: f64,
    /// RMSSD (ms) — short-term HRV; rises in REM and quiet rest.
    pub rmssd_ms: f64,
    /// Motion stillness fraction in `[0, 1]`.
    pub stillness: f64,
}

/// Estimate a coarse sleep stage. Thresholds are deliberately simple and
/// documented; they err toward [`SleepStage::Wake`] when motion is present.
pub fn estimate_stage(input: &SleepProxyInput) -> SleepStage {
    // Movement dominates: any meaningful motion ⇒ awake.
    if input.stillness < 0.6 {
        return SleepStage::Wake;
    }
    // Still, but cardiac markers distinguish depth.
    if input.mean_hr_bpm > 70.0 {
        // Still but elevated HR with high HRV resembles REM.
        if input.rmssd_ms > 45.0 {
            SleepStage::Rem
        } else {
            SleepStage::N1
        }
    } else if input.mean_hr_bpm > 58.0 {
        SleepStage::N2
    } else {
        // Low HR, very still ⇒ deep slow-wave sleep.
        SleepStage::N3
    }
}

/// Convenience: estimate from HRV/motion metric structs.
pub fn estimate_from_metrics(
    mean_hr_bpm: f64,
    rmssd_ms: f64,
    motion: &MotionMetrics,
) -> SleepStage {
    estimate_stage(&SleepProxyInput {
        mean_hr_bpm,
        rmssd_ms,
        stillness: motion.stillness(),
    })
}

/// Whether stimulation should be inhibited in this sleep state. Deep sleep
/// (N3) and unmonitored deep stages are protected from sensory stimulation.
pub fn stimulation_inhibited(stage: SleepStage) -> bool {
    matches!(stage, SleepStage::N3)
}
