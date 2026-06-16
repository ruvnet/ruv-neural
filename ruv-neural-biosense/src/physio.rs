//! Fused physiological window and derived autonomic-state indices.

use serde::{Deserialize, Serialize};

use crate::hrv::HrvMetrics;
use crate::motion::MotionMetrics;
use crate::respiration::RespirationMetrics;
use crate::BiosenseError;

/// Raw multi-signal physiological data for one analysis window.
///
/// Any field may be empty if that sensor is unavailable; metric computation
/// degrades gracefully, only producing the sub-metrics it has data for.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PhysioWindow {
    /// Session-relative timestamp of the window start (s).
    pub timestamp_s: f64,
    /// Inter-beat (RR) intervals in milliseconds.
    pub rr_ms: Vec<f64>,
    /// Respiration waveform samples.
    pub respiration: Vec<f64>,
    /// Respiration sample rate (Hz).
    pub respiration_fs: f64,
    /// Acceleration magnitudes (g-units).
    pub accel_magnitude_g: Vec<f64>,
}

/// Fused, derived physiological metrics for one window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysioMetrics {
    /// Window timestamp (s).
    pub timestamp_s: f64,
    /// HRV metrics, if RR data was available.
    pub hrv: Option<HrvMetrics>,
    /// Respiration metrics, if a respiration waveform was available.
    pub respiration: Option<RespirationMetrics>,
    /// Motion metrics, if accelerometer data was available.
    pub motion: Option<MotionMetrics>,
    /// Autonomic arousal index in `[0, 1]` (0 = deeply calm, 1 = high arousal).
    pub arousal_index: f64,
    /// Relaxation index in `[0, 1]` (vagal tone + calm breathing + stillness).
    pub relaxation_index: f64,
}

impl PhysioMetrics {
    /// Compute fused metrics from a [`PhysioWindow`].
    pub fn from_window(w: &PhysioWindow) -> Result<Self, BiosenseError> {
        let hrv = if w.rr_ms.len() >= 2 {
            Some(HrvMetrics::from_rr_ms(&w.rr_ms)?)
        } else {
            None
        };
        let respiration = if w.respiration.len() >= 4 && w.respiration_fs > 0.0 {
            Some(RespirationMetrics::from_waveform(
                &w.respiration,
                w.respiration_fs,
            )?)
        } else {
            None
        };
        let motion = if !w.accel_magnitude_g.is_empty() {
            Some(MotionMetrics::from_magnitude(&w.accel_magnitude_g)?)
        } else {
            None
        };

        if hrv.is_none() && respiration.is_none() && motion.is_none() {
            return Err(BiosenseError::InsufficientData { needed: 1, have: 0 });
        }

        let (arousal, relaxation) = autonomic_indices(&hrv, &respiration, &motion);

        Ok(Self {
            timestamp_s: w.timestamp_s,
            hrv,
            respiration,
            motion,
            arousal_index: arousal,
            relaxation_index: relaxation,
        })
    }
}

/// Map a heart rate (bpm) to a normalized arousal contribution in `[0, 1]`,
/// where ~55 bpm → 0 and ~100 bpm → 1.
fn hr_arousal(hr_bpm: f64) -> f64 {
    ((hr_bpm - 55.0) / 45.0).clamp(0.0, 1.0)
}

/// Derive arousal and relaxation indices from whichever metrics are present.
fn autonomic_indices(
    hrv: &Option<HrvMetrics>,
    respiration: &Option<RespirationMetrics>,
    motion: &Option<MotionMetrics>,
) -> (f64, f64) {
    // Weighted blend over available channels; weights renormalize when a
    // channel is missing.
    let mut arousal_num = 0.0;
    let mut arousal_den = 0.0;
    let mut relax_num = 0.0;
    let mut relax_den = 0.0;

    if let Some(h) = hrv {
        // High HR and low vagal tone → high arousal.
        arousal_num += 0.5 * hr_arousal(h.mean_hr_bpm) + 0.5 * (1.0 - h.vagal_tone());
        arousal_den += 1.0;
        relax_num += h.vagal_tone();
        relax_den += 1.0;
    }
    if let Some(r) = respiration {
        // Fast/irregular breathing → arousal; calm paced breathing → relax.
        let fast = ((r.rate_bpm - 6.0) / 24.0).clamp(0.0, 1.0);
        arousal_num += fast;
        arousal_den += 1.0;
        relax_num += r.calm_index();
        relax_den += 1.0;
    }
    if let Some(m) = motion {
        arousal_num += (m.movement_index / 0.1).clamp(0.0, 1.0);
        arousal_den += 1.0;
        relax_num += m.stillness();
        relax_den += 1.0;
    }

    let arousal = if arousal_den > 0.0 {
        (arousal_num / arousal_den).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let relaxation = if relax_den > 0.0 {
        (relax_num / relax_den).clamp(0.0, 1.0)
    } else {
        0.5
    };
    (arousal, relaxation)
}
