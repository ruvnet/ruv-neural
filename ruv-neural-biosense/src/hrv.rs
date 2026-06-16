//! Heart-rate variability (HRV) metrics from inter-beat (RR) intervals.
//!
//! Time-domain metrics (SDNN, RMSSD, pNN50) follow the Task Force of the
//! European Society of Cardiology / NASPE standard definitions. Frequency-
//! domain LF/HF is estimated with a dependency-free Goertzel band-power on an
//! evenly resampled tachogram.

use serde::{Deserialize, Serialize};

use crate::BiosenseError;

/// Low-frequency HRV band (sympathetic + parasympathetic), Hz.
pub const LF_BAND_HZ: (f64, f64) = (0.04, 0.15);
/// High-frequency HRV band (parasympathetic / vagal, respiratory), Hz.
pub const HF_BAND_HZ: (f64, f64) = (0.15, 0.40);

/// Time- and frequency-domain HRV metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HrvMetrics {
    /// Mean heart rate in beats per minute.
    pub mean_hr_bpm: f64,
    /// Mean NN (RR) interval in milliseconds.
    pub mean_nn_ms: f64,
    /// SDNN: standard deviation of NN intervals (ms).
    pub sdnn_ms: f64,
    /// RMSSD: root mean square of successive NN differences (ms) — a robust
    /// short-term parasympathetic (vagal) marker.
    pub rmssd_ms: f64,
    /// pNN50: fraction of successive NN differences exceeding 50 ms.
    pub pnn50: f64,
    /// Low-frequency band power (ms^2).
    pub lf_power: f64,
    /// High-frequency band power (ms^2).
    pub hf_power: f64,
    /// LF/HF ratio (a coarse sympatho-vagal balance index).
    pub lf_hf_ratio: f64,
    /// Number of NN intervals used.
    pub n_intervals: usize,
}

impl HrvMetrics {
    /// Compute HRV metrics from a sequence of RR intervals in milliseconds.
    ///
    /// Requires at least two intervals; intervals must be finite and positive.
    pub fn from_rr_ms(rr_ms: &[f64]) -> Result<Self, BiosenseError> {
        if rr_ms.len() < 2 {
            return Err(BiosenseError::InsufficientData {
                needed: 2,
                have: rr_ms.len(),
            });
        }
        if rr_ms.iter().any(|&r| !r.is_finite() || r <= 0.0) {
            return Err(BiosenseError::Invalid(
                "RR intervals must be finite and positive".into(),
            ));
        }

        let n = rr_ms.len();
        let mean_nn = rr_ms.iter().sum::<f64>() / n as f64;
        let mean_hr = 60_000.0 / mean_nn;

        let var = rr_ms.iter().map(|r| (r - mean_nn).powi(2)).sum::<f64>() / n as f64;
        let sdnn = var.sqrt();

        let diffs: Vec<f64> = rr_ms.windows(2).map(|w| w[1] - w[0]).collect();
        let rmssd = (diffs.iter().map(|d| d * d).sum::<f64>() / diffs.len() as f64).sqrt();
        let pnn50 = diffs.iter().filter(|d| d.abs() > 50.0).count() as f64 / diffs.len() as f64;

        // Frequency domain on an evenly resampled tachogram.
        let (lf, hf) = lf_hf_power(rr_ms);
        let lf_hf_ratio = if hf > 1e-12 { lf / hf } else { 0.0 };

        Ok(Self {
            mean_hr_bpm: mean_hr,
            mean_nn_ms: mean_nn,
            sdnn_ms: sdnn,
            rmssd_ms: rmssd,
            pnn50,
            lf_power: lf,
            hf_power: hf,
            lf_hf_ratio,
            n_intervals: n,
        })
    }

    /// A normalized vagal-tone proxy in `[0, 1]` derived from RMSSD, saturating
    /// around 100 ms. Higher means more parasympathetic (calmer) activity.
    pub fn vagal_tone(&self) -> f64 {
        (self.rmssd_ms / 100.0).clamp(0.0, 1.0)
    }
}

/// Build an evenly sampled tachogram (instantaneous RR vs. time) and integrate
/// LF and HF band power via Goertzel over a frequency grid.
fn lf_hf_power(rr_ms: &[f64]) -> (f64, f64) {
    // Cumulative beat times (s).
    let mut t = Vec::with_capacity(rr_ms.len());
    let mut acc = 0.0;
    for &rr in rr_ms {
        acc += rr / 1000.0;
        t.push(acc);
    }
    let total = *t.last().unwrap();
    if total <= 0.0 {
        return (0.0, 0.0);
    }

    // Resample to 4 Hz with linear interpolation, mean-removed.
    let fs = 4.0;
    let n = (total * fs).floor() as usize;
    if n < 8 {
        return (0.0, 0.0);
    }
    let mean = rr_ms.iter().sum::<f64>() / rr_ms.len() as f64;
    let mut x = Vec::with_capacity(n);
    let mut j = 0usize;
    for k in 0..n {
        let tk = k as f64 / fs;
        while j + 1 < t.len() && t[j + 1] < tk {
            j += 1;
        }
        let (t0, t1) = (t[j], t[(j + 1).min(t.len() - 1)]);
        let (v0, v1) = (rr_ms[j], rr_ms[(j + 1).min(rr_ms.len() - 1)]);
        let val = if (t1 - t0).abs() < 1e-9 {
            v0
        } else {
            v0 + (v1 - v0) * ((tk - t0) / (t1 - t0))
        };
        x.push(val - mean);
    }

    let band_power = |lo: f64, hi: f64| -> f64 {
        let steps = 16;
        let mut p = 0.0;
        for s in 0..steps {
            let f = lo + (hi - lo) * (s as f64 + 0.5) / steps as f64;
            p += goertzel_power(&x, f, fs);
        }
        // Normalize by the number of probe frequencies and series length.
        p / steps as f64
    };

    (
        band_power(LF_BAND_HZ.0, LF_BAND_HZ.1),
        band_power(HF_BAND_HZ.0, HF_BAND_HZ.1),
    )
}

/// Goertzel power of `x` at frequency `f` (Hz), sampled at `fs` (Hz).
fn goertzel_power(x: &[f64], f: f64, fs: f64) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    let w = 2.0 * std::f64::consts::PI * f / fs;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &v in x {
        let s0 = v + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    let power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
    power / (n as f64).powi(2)
}
