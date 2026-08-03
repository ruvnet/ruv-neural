//! Mask-aware coherence for NeuroSleep.
//!
//! Kept separate from the legacy connectivity metrics because it is fallible,
//! configuration-driven, and refuses to estimate over samples the caller has not
//! declared simultaneously valid on both channels.

use super::{hann_window, FFT_PLANNER};
use crate::neurosleep::DspError;
use num_complex::Complex;

/// Explicit coherence configuration for NeuroSleep.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoherenceConfig {
    /// Samples per segment.
    pub window_samples: usize,
    /// Samples shared between consecutive segments.
    pub overlap_samples: usize,
    /// Subtract each segment's own mean before windowing. Enabling this keeps a
    /// per-channel DC offset out of the low-frequency bins; the same choice is
    /// applied to both channels so the cross-spectrum stays consistent.
    pub detrend_mean: bool,
}

impl CoherenceConfig {
    /// Reject window/overlap combinations that would alias or never advance.
    pub fn validate(self) -> Result<Self, DspError> {
        if self.window_samples < 4 {
            return Err(DspError::InvalidConfig(
                "coherence window must span >= 4 samples",
            ));
        }
        if self.overlap_samples >= self.window_samples {
            return Err(DspError::InvalidConfig(
                "coherence overlap must be shorter than the window",
            ));
        }
        Ok(self)
    }
}

/// Fallible magnitude-squared coherence using one identical validity mask for
/// both channels.
///
/// The mask is shared rather than per-channel on purpose: coherence is only
/// meaningful over samples both channels observed simultaneously, so a segment
/// touching an invalid sample is dropped from *both* channels rather than being
/// filled, shortened, or averaged around.
pub fn coherence_masked(
    signal_a: &[f64],
    signal_b: &[f64],
    shared_valid_mask: &[bool],
    sample_rate_hz: f64,
    config: CoherenceConfig,
) -> Result<Vec<(f64, f64)>, DspError> {
    let config = config.validate()?;
    if signal_a.len() != signal_b.len() || signal_a.len() != shared_valid_mask.len() {
        return Err(DspError::LengthMismatch);
    }
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return Err(DspError::InvalidConfig(
            "sample rate must be finite and > 0",
        ));
    }
    if signal_a.len() < config.window_samples {
        return Err(DspError::InsufficientSamples {
            needed: config.window_samples,
            actual: signal_a.len(),
        });
    }
    if let Some((index, _)) = signal_a
        .iter()
        .zip(signal_b)
        .zip(shared_valid_mask)
        .enumerate()
        .find(|(_, ((a, b), valid))| **valid && (!a.is_finite() || !b.is_finite()))
    {
        return Err(DspError::NonFiniteSample(index));
    }
    let n = config.window_samples;
    let hop = n - config.overlap_samples;
    let window = hann_window(n);
    let bins = n / 2 + 1;
    let fft = FFT_PLANNER.with(|planner| planner.borrow_mut().plan_fft_forward(n));
    let mut saa = vec![0.0; bins];
    let mut sbb = vec![0.0; bins];
    let mut sab = vec![Complex::new(0.0, 0.0); bins];
    let mut segments = 0usize;
    let mut start = 0usize;
    while start + n <= signal_a.len() {
        if shared_valid_mask[start..start + n]
            .iter()
            .all(|valid| *valid)
        {
            let mean = |signal: &[f64]| {
                if config.detrend_mean {
                    signal[start..start + n].iter().sum::<f64>() / n as f64
                } else {
                    0.0
                }
            };
            let (mean_a, mean_b) = (mean(signal_a), mean(signal_b));
            let mut fa: Vec<_> = (0..n)
                .map(|i| Complex::new((signal_a[start + i] - mean_a) * window[i], 0.0))
                .collect();
            let mut fb: Vec<_> = (0..n)
                .map(|i| Complex::new((signal_b[start + i] - mean_b) * window[i], 0.0))
                .collect();
            fft.process(&mut fa);
            fft.process(&mut fb);
            for bin in 0..bins {
                saa[bin] += fa[bin].norm_sqr();
                sbb[bin] += fb[bin].norm_sqr();
                sab[bin] += fa[bin] * fb[bin].conj();
            }
            segments += 1;
        }
        start += hop;
    }
    if segments == 0 {
        return Err(DspError::NoValidSegments);
    }
    Ok((0..bins)
        .map(|bin| {
            let denominator = saa[bin] * sbb[bin];
            let value = if denominator > 1e-30 {
                sab[bin].norm_sqr() / denominator
            } else {
                0.0
            };
            (
                bin as f64 * sample_rate_hz / n as f64,
                value.clamp(0.0, 1.0),
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const RATE: f64 = 250.0;

    fn config() -> CoherenceConfig {
        CoherenceConfig {
            window_samples: 250,
            overlap_samples: 125,
            detrend_mean: true,
        }
    }

    fn tone(frequency_hz: f64, phase: f64, offset: f64, samples: usize) -> Vec<f64> {
        (0..samples)
            .map(|index| offset + (2.0 * PI * frequency_hz * index as f64 / RATE + phase).sin())
            .collect()
    }

    /// Deterministic pseudo-random samples, so the test is reproducible.
    fn noise(seed: u64) -> Vec<f64> {
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (0..25_000)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    }

    fn at(pairs: &[(f64, f64)], frequency_hz: f64) -> f64 {
        pairs
            .iter()
            .min_by(|a, b| {
                (a.0 - frequency_hz)
                    .abs()
                    .total_cmp(&(b.0 - frequency_hz).abs())
            })
            .expect("non-empty spectrum")
            .1
    }

    #[test]
    fn phase_locked_channels_are_coherent_and_bounded() {
        let a = tone(10.0, 0.0, 0.0, 5_000);
        let b = tone(10.0, 0.7, 0.0, 5_000);
        let mask = vec![true; a.len()];
        let coherence = coherence_masked(&a, &b, &mask, RATE, config()).unwrap();
        assert!(at(&coherence, 10.0) > 0.99);
        assert!(coherence
            .iter()
            .all(|(_, value)| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn both_channels_share_one_mask_so_segments_drop_together() {
        let a = tone(10.0, 0.0, 0.0, 5_000);
        let b = tone(10.0, 0.7, 0.0, 5_000);
        let mut masked_a = a.clone();
        let mut mask = vec![true; a.len()];
        for index in 2_500..a.len() {
            masked_a[index] = f64::NAN;
            mask[index] = false;
        }
        let masked = coherence_masked(&masked_a, &b, &mask, RATE, config()).unwrap();
        let reference =
            coherence_masked(&a[..2_500], &b[..2_500], &vec![true; 2_500], RATE, config()).unwrap();
        assert_eq!(masked.len(), reference.len());
        for (left, right) in masked.iter().zip(&reference) {
            assert!((left.1 - right.1).abs() < 1e-12);
        }
    }

    #[test]
    fn mean_detrending_suppresses_dc_driven_low_frequency_coherence() {
        // Two channels carrying independent noise, each with a constant offset.
        // A constant is perfectly coherent with any other constant, so without
        // detrending the DC leakage manufactures near-unity coherence in the
        // lowest bins where these channels share nothing at all.
        let a: Vec<f64> = noise(1).iter().map(|value| 500.0 + value).collect();
        let b: Vec<f64> = noise(9).iter().map(|value| -300.0 + value).collect();
        let mask = vec![true; a.len()];
        let detrended = coherence_masked(&a, &b, &mask, RATE, config()).unwrap();
        let raw = coherence_masked(
            &a,
            &b,
            &mask,
            RATE,
            CoherenceConfig {
                detrend_mean: false,
                ..config()
            },
        )
        .unwrap();
        assert!(at(&raw, 1.0) > 0.9, "raw {}", at(&raw, 1.0));
        assert!(
            at(&detrended, 1.0) < 0.5,
            "detrended {}",
            at(&detrended, 1.0)
        );
    }

    #[test]
    fn masked_coherence_fails_closed_on_bad_inputs() {
        let a = tone(10.0, 0.0, 0.0, 5_000);
        let mask = vec![true; a.len()];
        assert_eq!(
            coherence_masked(&a, &a[..10], &mask, RATE, config()),
            Err(DspError::LengthMismatch)
        );
        assert!(coherence_masked(&a, &a, &mask, 0.0, config()).is_err());
        assert!(matches!(
            coherence_masked(&a[..10], &a[..10], &[true; 10], RATE, config()),
            Err(DspError::InsufficientSamples { needed: 250, .. })
        ));
        assert!(CoherenceConfig {
            overlap_samples: 250,
            ..config()
        }
        .validate()
        .is_err());

        let mut nan = a.clone();
        nan[7] = f64::NAN;
        assert_eq!(
            coherence_masked(&nan, &a, &mask, RATE, config()),
            Err(DspError::NonFiniteSample(7))
        );

        // Invalid every 200 samples leaves no fully valid 250-sample window.
        let mut sparse = vec![true; a.len()];
        for index in (0..a.len()).step_by(200) {
            sparse[index] = false;
        }
        assert_eq!(
            coherence_masked(&a, &a, &sparse, RATE, config()),
            Err(DspError::NoValidSegments)
        );
    }
}
