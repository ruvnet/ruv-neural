//! Respiration metrics from a breathing waveform (e.g. chest-belt, thermistor,
//! or respiratory-induced amplitude signal).

use serde::{Deserialize, Serialize};

use crate::BiosenseError;

/// Respiration metrics over one analysis window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RespirationMetrics {
    /// Breathing rate in breaths per minute.
    pub rate_bpm: f64,
    /// Peak-to-trough excursion (a depth proxy, signal units).
    pub depth: f64,
    /// Coefficient of variation of breath-to-breath interval (regularity:
    /// lower is more regular/paced).
    pub rate_cv: f64,
    /// Number of detected breaths.
    pub n_breaths: usize,
}

impl RespirationMetrics {
    /// Compute respiration metrics from a waveform sampled at `fs` Hz.
    pub fn from_waveform(samples: &[f64], fs: f64) -> Result<Self, BiosenseError> {
        if !fs.is_finite() || fs <= 0.0 {
            return Err(BiosenseError::Invalid(
                "fs must be finite and positive".into(),
            ));
        }
        if samples.len() < 4 {
            return Err(BiosenseError::InsufficientData {
                needed: 4,
                have: samples.len(),
            });
        }

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let max = samples.iter().cloned().fold(f64::MIN, f64::max);
        let min = samples.iter().cloned().fold(f64::MAX, f64::min);
        let depth = max - min;

        // Detect inspiratory peaks as upward crossings of the mean followed by
        // a local maximum, with a refractory period of 1.5 s (40 bpm ceiling).
        let refractory = (1.5 * fs) as usize;
        let mut peak_idxs: Vec<usize> = Vec::new();
        let mut i = 1;
        while i < samples.len() {
            if samples[i - 1] <= mean && samples[i] > mean {
                // climbed above mean: find the local peak ahead
                let mut j = i;
                let mut peak = i;
                while j < samples.len() && samples[j] >= mean {
                    if samples[j] > samples[peak] {
                        peak = j;
                    }
                    j += 1;
                }
                if peak_idxs
                    .last()
                    .map(|&p| peak - p >= refractory)
                    .unwrap_or(true)
                {
                    peak_idxs.push(peak);
                }
                i = j.max(i + 1);
            } else {
                i += 1;
            }
        }

        let duration_s = samples.len() as f64 / fs;
        let n_breaths = peak_idxs.len();
        let rate_bpm = if duration_s > 0.0 {
            n_breaths as f64 / duration_s * 60.0
        } else {
            0.0
        };

        // Breath-to-breath interval CV.
        let rate_cv = if peak_idxs.len() >= 3 {
            let intervals: Vec<f64> = peak_idxs
                .windows(2)
                .map(|w| (w[1] - w[0]) as f64 / fs)
                .collect();
            let m = intervals.iter().sum::<f64>() / intervals.len() as f64;
            let sd = (intervals.iter().map(|x| (x - m).powi(2)).sum::<f64>()
                / intervals.len() as f64)
                .sqrt();
            if m > 1e-9 {
                sd / m
            } else {
                0.0
            }
        } else {
            0.0
        };

        Ok(Self {
            rate_bpm,
            depth,
            rate_cv,
            n_breaths,
        })
    }

    /// A calm-breathing proxy in `[0, 1]`: highest near slow, regular paced
    /// breathing (~6 bpm) and decaying for faster or irregular breathing.
    pub fn calm_index(&self) -> f64 {
        // Gaussian around 6 bpm (resonance/coherence breathing), penalized by
        // irregularity.
        let rate_term = (-((self.rate_bpm - 6.0).powi(2)) / (2.0 * 6.0_f64.powi(2))).exp();
        let regularity = (1.0 - self.rate_cv).clamp(0.0, 1.0);
        (rate_term * regularity).clamp(0.0, 1.0)
    }
}
