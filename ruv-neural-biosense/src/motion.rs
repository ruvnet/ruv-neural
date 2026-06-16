//! Motion / actigraphy metrics from a tri-axial accelerometer.

use serde::{Deserialize, Serialize};

use crate::BiosenseError;

/// Standard gravity in g-units used as the static baseline.
const GRAVITY_G: f64 = 1.0;

/// Motion metrics over one analysis window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotionMetrics {
    /// Mean acceleration magnitude (g).
    pub mean_magnitude_g: f64,
    /// Movement index: standard deviation of the gravity-removed magnitude (g).
    pub movement_index: f64,
    /// Fraction of samples below the stillness threshold, in `[0, 1]`.
    pub stillness_fraction: f64,
    /// Number of samples.
    pub n_samples: usize,
}

/// Threshold (g) on instantaneous |a-1g| below which a sample is "still".
pub const STILLNESS_THRESHOLD_G: f64 = 0.02;

impl MotionMetrics {
    /// Compute from tri-axial samples `accel[i] = [x, y, z]` in g-units.
    pub fn from_triaxial(accel: &[[f64; 3]]) -> Result<Self, BiosenseError> {
        if accel.is_empty() {
            return Err(BiosenseError::InsufficientData { needed: 1, have: 0 });
        }
        let mags: Vec<f64> = accel
            .iter()
            .map(|a| (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt())
            .collect();
        Self::from_magnitude(&mags)
    }

    /// Compute from precomputed acceleration magnitudes (g-units).
    pub fn from_magnitude(mags: &[f64]) -> Result<Self, BiosenseError> {
        if mags.is_empty() {
            return Err(BiosenseError::InsufficientData { needed: 1, have: 0 });
        }
        let n = mags.len();
        let mean = mags.iter().sum::<f64>() / n as f64;
        let dyn_mag: Vec<f64> = mags.iter().map(|m| (m - GRAVITY_G).abs()).collect();
        let dyn_mean = dyn_mag.iter().sum::<f64>() / n as f64;
        let movement_index =
            (dyn_mag.iter().map(|d| (d - dyn_mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        let still = dyn_mag
            .iter()
            .filter(|d| **d < STILLNESS_THRESHOLD_G)
            .count();
        Ok(Self {
            mean_magnitude_g: mean,
            movement_index,
            stillness_fraction: still as f64 / n as f64,
            n_samples: n,
        })
    }

    /// A normalized stillness proxy in `[0, 1]` (1.0 = perfectly still).
    pub fn stillness(&self) -> f64 {
        self.stillness_fraction.clamp(0.0, 1.0)
    }
}
