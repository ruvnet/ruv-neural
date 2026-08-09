//! Long-range temporal correlations (LRTC) via detrended fluctuation analysis.
//!
//! Computes the DFA scaling exponent α (and Hurst exponent) of a neural time
//! series. The DFA exponent is dimensionless, montage- and amplitude-invariant,
//! and cheap to compute — the class of feature shown (arXiv:2607.24834, 2026)
//! to transfer across populations where frozen EEG foundation-model embeddings
//! collapse below chance. Typical use on EEG is the amplitude envelope of a
//! band-limited oscillation (e.g. the alpha band), for which
//! [`band_envelope_dfa`] is provided.
//!
//! Interpretation of α (DFA-1, first-order detrending):
//! - α ≈ 0.5 — uncorrelated (white) noise
//! - 0.5 < α < 1.0 — persistent long-range correlations (healthy resting EEG
//!   alpha envelopes typically fall in ~0.6–0.9)
//! - α ≈ 1.0 — 1/f (pink) noise
//! - α ≈ 1.5 — Brownian motion (integrated white noise)
//!
//! For fractional-Gaussian-noise-like signals (α < 1) the Hurst exponent
//! equals α; for α > 1 the signal is non-stationary and H = α − 1.
//!
//! # Implementation notes
//!
//! The profile (cumulative sum of the mean-removed signal) is divided into
//! non-overlapping windows of each scale, taken from both the start and the
//! end of the record so the tail is not discarded (Kantelhardt et al., 2001).
//! Per-window linear detrending uses the closed-form least-squares solution
//! with precomputed index sums, so each scale costs O(n) with no per-window
//! allocation — the whole analysis is O(n · num_scales).

use ruv_neural_core::signal::FrequencyBand;

use crate::filter::BandpassFilter;
use crate::hilbert::instantaneous_amplitude;

/// Configuration for detrended fluctuation analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct DfaConfig {
    /// Smallest window scale in samples (≥ 4 for a meaningful linear fit).
    pub min_scale: usize,
    /// Largest window scale in samples. `None` selects `n / 4`, the standard
    /// upper bound for stable fluctuation statistics.
    pub max_scale: Option<usize>,
    /// Number of logarithmically spaced scales between min and max.
    pub num_scales: usize,
}

impl Default for DfaConfig {
    fn default() -> Self {
        Self {
            min_scale: 4,
            max_scale: None,
            num_scales: 16,
        }
    }
}

/// Result of a DFA computation.
#[derive(Debug, Clone, PartialEq)]
pub struct DfaResult {
    /// The DFA scaling exponent α (slope of log₂ F(s) vs log₂ s).
    pub alpha: f64,
    /// Coefficient of determination of the log-log fit. Values well below
    /// ~0.95 indicate the signal does not scale as a single power law over
    /// the chosen range and α should not be trusted blindly.
    pub r_squared: f64,
    /// The window scales used, in samples.
    pub scales: Vec<usize>,
    /// Root-mean-square fluctuation F(s) at each scale.
    pub fluctuations: Vec<f64>,
}

impl DfaResult {
    /// The Hurst exponent implied by α: `α` for stationary signals
    /// (α ≤ 1), `α − 1` for non-stationary signals (α > 1).
    pub fn hurst(&self) -> f64 {
        if self.alpha > 1.0 {
            self.alpha - 1.0
        } else {
            self.alpha
        }
    }
}

/// Compute the DFA scaling exponent of a time series (DFA-1).
///
/// Returns `None` when the input cannot support the analysis: fewer than
/// `4 * min_scale` samples, fewer than two distinct usable scales, a
/// non-finite sample, or a (near-)constant signal whose fluctuations are
/// numerically zero.
///
/// # Example
///
/// ```
/// use ruv_neural_signal::lrtc::{dfa, DfaConfig};
///
/// // A deterministic pseudo-random walk has strong persistence (alpha ~ 1.5).
/// let mut state = 1u64;
/// let mut walk = Vec::with_capacity(2048);
/// let mut acc = 0.0;
/// for _ in 0..2048 {
///     state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
///     acc += (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
///     walk.push(acc);
/// }
/// let result = dfa(&walk, &DfaConfig::default()).unwrap();
/// assert!(result.alpha > 1.2);
/// ```
pub fn dfa(signal: &[f64], config: &DfaConfig) -> Option<DfaResult> {
    let n = signal.len();
    if config.min_scale < 4 || config.num_scales < 2 || n < 4 * config.min_scale {
        return None;
    }
    if signal.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let max_scale = config.max_scale.unwrap_or(n / 4).min(n / 4);
    if max_scale <= config.min_scale {
        return None;
    }

    // Profile: cumulative sum of the mean-removed signal.
    let mean = signal.iter().sum::<f64>() / n as f64;
    let mut profile = Vec::with_capacity(n);
    let mut acc = 0.0;
    for &x in signal {
        acc += x - mean;
        profile.push(acc);
    }

    let scales = log_spaced_scales(config.min_scale, max_scale, config.num_scales);
    if scales.len() < 2 {
        return None;
    }

    let mut log_s = Vec::with_capacity(scales.len());
    let mut log_f = Vec::with_capacity(scales.len());
    let mut fluctuations = Vec::with_capacity(scales.len());
    for &s in &scales {
        let f = fluctuation_at_scale(&profile, s);
        if !(f.is_finite() && f > 0.0) {
            return None;
        }
        fluctuations.push(f);
        log_s.push((s as f64).log2());
        log_f.push(f.log2());
    }

    let (alpha, r_squared) = linear_fit(&log_s, &log_f)?;
    Some(DfaResult {
        alpha,
        r_squared,
        scales,
        fluctuations,
    })
}

/// DFA of the amplitude envelope of a band-limited oscillation.
///
/// Bandpass-filters the signal (4th-order Butterworth), takes the Hilbert
/// amplitude envelope, and runs [`dfa`] on the envelope — the standard
/// LRTC analysis for neural oscillations (e.g. the alpha band, 8–13 Hz).
pub fn band_envelope_dfa(
    signal: &[f64],
    sample_rate: f64,
    band: &FrequencyBand,
    config: &DfaConfig,
) -> Option<DfaResult> {
    if signal.is_empty() || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return None;
    }
    let (low_hz, high_hz) = band.range_hz();
    let filter = BandpassFilter::new(4, low_hz, high_hz, sample_rate);
    let filtered = filter.apply(signal);
    let envelope = instantaneous_amplitude(&filtered);
    dfa(&envelope, config)
}

/// Logarithmically spaced integer scales in `[min, max]`, deduplicated.
fn log_spaced_scales(min: usize, max: usize, count: usize) -> Vec<usize> {
    let log_min = (min as f64).ln();
    let log_max = (max as f64).ln();
    let mut scales = Vec::with_capacity(count);
    for i in 0..count {
        let t = i as f64 / (count - 1) as f64;
        let s = (log_min + t * (log_max - log_min)).exp().round() as usize;
        let s = s.clamp(min, max);
        if scales.last() != Some(&s) {
            scales.push(s);
        }
    }
    scales
}

/// RMS fluctuation F(s) over non-overlapping windows of length `s`, taken
/// from both the start and the end of the profile.
///
/// Each window is detrended with a closed-form least-squares line: with the
/// local index t = 0..s−1, precompute Σt and Σt² once per scale, accumulate
/// Σy, Σty, Σy² per window in a single pass, and use
/// RSS = Σy² − a·Σy − b·Σty for the residual sum of squares.
fn fluctuation_at_scale(profile: &[f64], s: usize) -> f64 {
    let n = profile.len();
    let windows = n / s;
    debug_assert!(windows >= 1);

    let sf = s as f64;
    // Index sums for t = 0..s-1.
    let st = sf * (sf - 1.0) / 2.0;
    let stt = (sf - 1.0) * sf * (2.0 * sf - 1.0) / 6.0;
    let denom = sf * stt - st * st;

    let window_rss = |start: usize| -> f64 {
        let mut sy = 0.0;
        let mut sty = 0.0;
        let mut syy = 0.0;
        for (t, &y) in profile[start..start + s].iter().enumerate() {
            sy += y;
            sty += t as f64 * y;
            syy += y * y;
        }
        let b = (sf * sty - st * sy) / denom;
        let a = (sy - b * st) / sf;
        // Rounding can push an exact-fit RSS slightly negative; clamp.
        (syy - a * sy - b * sty).max(0.0)
    };

    let mut total = 0.0;
    for k in 0..windows {
        total += window_rss(k * s);
    }
    // Mirror pass from the end so the tail (n mod s samples) contributes.
    let tail_offset = n - windows * s;
    if tail_offset > 0 {
        for k in 0..windows {
            total += window_rss(tail_offset + k * s);
        }
        (total / (2 * windows) as f64 / sf).sqrt()
    } else {
        (total / windows as f64 / sf).sqrt()
    }
}

/// Least-squares slope and R² of y against x. Returns `None` for degenerate x.
fn linear_fit(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;
    for (&xi, &yi) in x.iter().zip(y) {
        let dx = xi - mx;
        let dy = yi - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    if sxx <= 0.0 {
        return None;
    }
    let slope = sxy / sxx;
    let r_squared = if syy > 0.0 {
        (sxy * sxy) / (sxx * syy)
    } else {
        1.0
    };
    Some((slope, r_squared))
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;
    use rustfft::FftPlanner;

    /// Deterministic uniform pseudo-random numbers in (-0.5, 0.5).
    /// Mirrored exactly in the TypeScript reference implementation
    /// (`apps/ruv-neural-ui/src/lrtc/dfa.ts`) for cross-language parity.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        }
    }

    fn white_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Lcg(seed);
        (0..n).map(|_| rng.next()).collect()
    }

    fn brownian(n: usize, seed: u64) -> Vec<f64> {
        let mut acc = 0.0;
        white_noise(n, seed)
            .into_iter()
            .map(|x| {
                acc += x;
                acc
            })
            .collect()
    }

    /// 1/f (pink) noise via spectral shaping of deterministic white noise.
    fn pink_noise(n: usize, seed: u64) -> Vec<f64> {
        let white = white_noise(n, seed);
        let mut spectrum: Vec<Complex<f64>> = white.iter().map(|&x| Complex::new(x, 0.0)).collect();
        FftPlanner::new().plan_fft_forward(n).process(&mut spectrum);
        for (k, c) in spectrum.iter_mut().enumerate() {
            let freq_bin = if k <= n / 2 { k } else { n - k };
            if freq_bin == 0 {
                *c = Complex::new(0.0, 0.0);
            } else {
                *c /= (freq_bin as f64).sqrt();
            }
        }
        FftPlanner::new().plan_fft_inverse(n).process(&mut spectrum);
        spectrum.iter().map(|c| c.re / n as f64).collect()
    }

    #[test]
    fn white_noise_alpha_near_half() {
        let result = dfa(&white_noise(8192, 42), &DfaConfig::default()).unwrap();
        assert!(
            (result.alpha - 0.5).abs() < 0.1,
            "white noise alpha {} not near 0.5",
            result.alpha
        );
        assert!(result.r_squared > 0.95, "r² {}", result.r_squared);
        assert!((result.hurst() - result.alpha).abs() < f64::EPSILON);
    }

    #[test]
    fn brownian_alpha_near_three_halves() {
        let result = dfa(&brownian(8192, 42), &DfaConfig::default()).unwrap();
        assert!(
            (result.alpha - 1.5).abs() < 0.15,
            "brownian alpha {} not near 1.5",
            result.alpha
        );
        assert!((result.hurst() - (result.alpha - 1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn pink_noise_alpha_near_one() {
        let result = dfa(&pink_noise(8192, 7), &DfaConfig::default()).unwrap();
        assert!(
            (result.alpha - 1.0).abs() < 0.15,
            "pink noise alpha {} not near 1.0",
            result.alpha
        );
    }

    #[test]
    fn fluctuations_grow_with_scale() {
        let result = dfa(&brownian(4096, 3), &DfaConfig::default()).unwrap();
        for pair in result.fluctuations.windows(2) {
            assert!(pair[1] > pair[0], "fluctuations must increase with scale");
        }
    }

    #[test]
    fn rejects_degenerate_input() {
        let config = DfaConfig::default();
        assert!(dfa(&[], &config).is_none());
        assert!(dfa(&white_noise(8, 1), &config).is_none());
        assert!(dfa(&vec![1.0; 4096], &config).is_none(), "constant signal");
        let mut with_nan = white_noise(4096, 1);
        with_nan[100] = f64::NAN;
        assert!(dfa(&with_nan, &config).is_none());
        assert!(dfa(
            &white_noise(4096, 1),
            &DfaConfig {
                num_scales: 1,
                ..DfaConfig::default()
            }
        )
        .is_none());
        assert!(dfa(
            &white_noise(4096, 1),
            &DfaConfig {
                min_scale: 2,
                ..DfaConfig::default()
            }
        )
        .is_none());
    }

    /// Cross-language parity fixture: the same LCG-seeded white-noise input
    /// must produce the same alpha (to double precision) in the TypeScript
    /// reference implementation. Keep in sync with
    /// `apps/ruv-neural-ui/src/lrtc/dfa.test.ts`.
    #[test]
    fn parity_fixture() {
        let result = dfa(&white_noise(2048, 12345), &DfaConfig::default()).unwrap();
        let expected = 0.542472684045213;
        assert!(
            (result.alpha - expected).abs() < 1e-9,
            "parity alpha {} != {}",
            result.alpha,
            expected
        );
    }

    #[test]
    fn band_envelope_dfa_runs_on_alpha_band() {
        // 10 Hz oscillation with amplitude modulated by a persistent walk.
        let fs = 250.0;
        let n = 8192;
        let walk = brownian(n, 9);
        let max_walk = walk.iter().cloned().fold(f64::MIN, f64::max);
        let min_walk = walk.iter().cloned().fold(f64::MAX, f64::min);
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let amp = 1.0 + (walk[i] - min_walk) / (max_walk - min_walk);
                amp * (2.0 * std::f64::consts::PI * 10.0 * t).sin()
            })
            .collect();
        let band = FrequencyBand::Alpha;
        let result = band_envelope_dfa(&signal, fs, &band, &DfaConfig::default()).unwrap();
        // Modulated by an integrated process => strongly persistent envelope.
        assert!(
            result.alpha > 0.7,
            "modulated envelope alpha {} should be persistent",
            result.alpha
        );
        assert!(band_envelope_dfa(&signal, 0.0, &band, &DfaConfig::default()).is_none());
        assert!(band_envelope_dfa(&[], fs, &band, &DfaConfig::default()).is_none());
    }

    #[test]
    fn scales_are_log_spaced_and_unique() {
        let scales = log_spaced_scales(4, 1024, 16);
        assert_eq!(*scales.first().unwrap(), 4);
        assert_eq!(*scales.last().unwrap(), 1024);
        for pair in scales.windows(2) {
            assert!(pair[1] > pair[0]);
        }
    }
}
