//! # ruv-neural-biosense
//!
//! Physiological **response sensing** for the rUv Neural closed-loop
//! neuromodulation platform. This is the *"measure a response"* stage of the
//! closed loop: it turns raw peripheral biosignals into compact, interpretable
//! metrics the controller can act on.
//!
//! ## Channels
//!
//! | Module        | Signal                | Key metrics                          |
//! |---------------|-----------------------|--------------------------------------|
//! | [`hrv`]       | inter-beat intervals  | SDNN, RMSSD, pNN50, LF/HF, vagal tone |
//! | [`respiration`] | breathing waveform  | rate, depth, regularity, calm index  |
//! | [`motion`]    | tri-axial accel       | movement index, stillness            |
//! | [`sleep`]     | HR + motion           | coarse sleep-stage proxy             |
//! | [`physio`]    | fused window          | arousal / relaxation indices         |
//!
//! ```
//! use ruv_neural_biosense::{PhysioSimulator, PhysioMetrics};
//!
//! let mut sim = PhysioSimulator::new(42);
//! let calm = sim.window(0.0, 30.0, 0.1);
//! let m = PhysioMetrics::from_window(&calm).unwrap();
//! assert!(m.relaxation_index > m.arousal_index);
//! ```

pub mod hrv;
pub mod motion;
pub mod physio;
pub mod respiration;
pub mod simulator;
pub mod sleep;

pub use hrv::{HrvMetrics, HF_BAND_HZ, LF_BAND_HZ};
pub use motion::{MotionMetrics, STILLNESS_THRESHOLD_G};
pub use physio::{PhysioMetrics, PhysioWindow};
pub use respiration::RespirationMetrics;
pub use simulator::PhysioSimulator;
pub use sleep::{estimate_stage, stimulation_inhibited, SleepProxyInput};

use thiserror::Error;

/// Errors produced by physiological metric computation.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum BiosenseError {
    /// Not enough samples to compute a metric.
    #[error("Insufficient data: need {needed}, have {have}")]
    InsufficientData { needed: usize, have: usize },

    /// Structurally invalid input.
    #[error("Invalid physiological input: {0}")]
    Invalid(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ruv_neural_core::topology::SleepStage;

    #[test]
    fn hrv_rejects_short_series() {
        assert!(HrvMetrics::from_rr_ms(&[800.0]).is_err());
    }

    #[test]
    fn hrv_basic_time_domain() {
        // Alternating 800/840 ms → mean 820, successive diff 40 ms.
        let rr: Vec<f64> = (0..40).map(|i| if i % 2 == 0 { 800.0 } else { 840.0 }).collect();
        let h = HrvMetrics::from_rr_ms(&rr).unwrap();
        assert_abs_diff_eq!(h.mean_nn_ms, 820.0, epsilon = 0.5);
        assert_abs_diff_eq!(h.mean_hr_bpm, 60_000.0 / 820.0, epsilon = 0.1);
        assert_abs_diff_eq!(h.rmssd_ms, 40.0, epsilon = 1.0);
        // Successive diffs are all 40 ms (< 50 ms), so pnn50 is 0.
        assert_abs_diff_eq!(h.pnn50, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn hrv_pnn50_threshold() {
        // diffs of 40 ms are NOT > 50 ms, so pnn50 should be 0.
        let rr: Vec<f64> = (0..20).map(|i| if i % 2 == 0 { 800.0 } else { 840.0 }).collect();
        let h = HrvMetrics::from_rr_ms(&rr).unwrap();
        assert_abs_diff_eq!(h.pnn50, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn hrv_high_variability_has_higher_rmssd() {
        let calm: Vec<f64> = (0..60).map(|i| 1000.0 + 60.0 * ((i as f64) * 0.3).sin()).collect();
        let tense: Vec<f64> = (0..60).map(|i| 700.0 + 8.0 * ((i as f64) * 0.3).sin()).collect();
        let hc = HrvMetrics::from_rr_ms(&calm).unwrap();
        let ht = HrvMetrics::from_rr_ms(&tense).unwrap();
        assert!(hc.rmssd_ms > ht.rmssd_ms);
        assert!(hc.vagal_tone() > ht.vagal_tone());
    }

    #[test]
    fn respiration_detects_rate() {
        // 12 breaths/min = 0.2 Hz, 30 s window at 25 Hz.
        let fs = 25.0;
        let n = (30.0 * fs) as usize;
        let samples: Vec<f64> = (0..n)
            .map(|k| (2.0 * std::f64::consts::PI * 0.2 * (k as f64 / fs)).sin())
            .collect();
        let r = RespirationMetrics::from_waveform(&samples, fs).unwrap();
        assert_abs_diff_eq!(r.rate_bpm, 12.0, epsilon = 1.5);
    }

    #[test]
    fn respiration_calm_index_peaks_at_slow_paced() {
        let fs = 25.0;
        let n = (40.0 * fs) as usize;
        let slow: Vec<f64> = (0..n)
            .map(|k| (2.0 * std::f64::consts::PI * 0.1 * (k as f64 / fs)).sin())
            .collect();
        let fast: Vec<f64> = (0..n)
            .map(|k| (2.0 * std::f64::consts::PI * 0.4 * (k as f64 / fs)).sin())
            .collect();
        let rs = RespirationMetrics::from_waveform(&slow, fs).unwrap();
        let rf = RespirationMetrics::from_waveform(&fast, fs).unwrap();
        assert!(rs.calm_index() > rf.calm_index());
    }

    #[test]
    fn motion_stillness_and_movement() {
        let still: Vec<f64> = (0..100).map(|_| 1.0).collect();
        let m = MotionMetrics::from_magnitude(&still).unwrap();
        assert_abs_diff_eq!(m.stillness_fraction, 1.0);
        assert!(m.movement_index < 1e-9);

        let moving: Vec<f64> = (0..100).map(|i| 1.0 + 0.2 * (i as f64).sin()).collect();
        let mm = MotionMetrics::from_magnitude(&moving).unwrap();
        assert!(mm.movement_index > 0.0);
        assert!(mm.stillness_fraction < 1.0);
    }

    #[test]
    fn motion_triaxial_magnitude() {
        let accel = vec![[0.0, 0.0, 1.0]; 50];
        let m = MotionMetrics::from_triaxial(&accel).unwrap();
        assert_abs_diff_eq!(m.mean_magnitude_g, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn sleep_proxy_stages() {
        // Moving → Wake.
        assert_eq!(
            estimate_stage(&SleepProxyInput { mean_hr_bpm: 65.0, rmssd_ms: 30.0, stillness: 0.2 }),
            SleepStage::Wake
        );
        // Still + low HR → deep N3.
        assert_eq!(
            estimate_stage(&SleepProxyInput { mean_hr_bpm: 52.0, rmssd_ms: 30.0, stillness: 0.95 }),
            SleepStage::N3
        );
        // Still + elevated HR + high HRV → REM.
        assert_eq!(
            estimate_stage(&SleepProxyInput { mean_hr_bpm: 75.0, rmssd_ms: 55.0, stillness: 0.9 }),
            SleepStage::Rem
        );
        assert!(stimulation_inhibited(SleepStage::N3));
        assert!(!stimulation_inhibited(SleepStage::Wake));
    }

    #[test]
    fn physio_fusion_calm_vs_aroused() {
        let mut sim = PhysioSimulator::new(7);
        let calm = PhysioMetrics::from_window(&sim.window(0.0, 30.0, 0.1)).unwrap();
        let aroused = PhysioMetrics::from_window(&sim.window(30.0, 30.0, 0.9)).unwrap();
        assert!(calm.arousal_index < aroused.arousal_index);
        assert!(calm.relaxation_index > aroused.relaxation_index);
        assert!(calm.hrv.is_some() && calm.respiration.is_some() && calm.motion.is_some());
    }

    #[test]
    fn physio_empty_window_errors() {
        let w = PhysioWindow::default();
        assert!(PhysioMetrics::from_window(&w).is_err());
    }

    #[test]
    fn simulator_is_deterministic() {
        let mut a = PhysioSimulator::new(123);
        let mut b = PhysioSimulator::new(123);
        let wa = a.window(0.0, 10.0, 0.5);
        let wb = b.window(0.0, 10.0, 0.5);
        assert_eq!(wa.rr_ms.len(), wb.rr_ms.len());
        assert_abs_diff_eq!(wa.rr_ms[0], wb.rr_ms[0], epsilon = 1e-12);
    }
}
