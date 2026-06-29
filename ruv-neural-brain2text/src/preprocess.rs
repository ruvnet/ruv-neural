//! Brain2Qwerty V1 preprocessing: bandpass → resample → (baseline at epoching).
//!
//! Mirrors the published V1 recipe: bandpass 0.1–20 Hz, resample to 50 Hz, then
//! per-channel baseline correction over the pre-keypress interval (applied later
//! during epoch extraction). Filtering reuses the zero-phase Butterworth
//! bandpass from [`ruv_neural_signal`]; resampling is linear interpolation,
//! which is adequate because the signal is already low-passed below 20 Hz.

use ruv_neural_core::error::Result;
use ruv_neural_core::signal::MultiChannelTimeSeries;
use ruv_neural_signal::filter::{BandpassFilter, SignalProcessor};

use crate::config::Brain2TextConfig;

/// Apply bandpass filtering and resampling to a continuous recording.
pub fn preprocess(
    series: &MultiChannelTimeSeries,
    config: &Brain2TextConfig,
) -> Result<MultiChannelTimeSeries> {
    let sr = series.sample_rate_hz;

    // Zero-phase bandpass per channel. Clamp the high cutoff below Nyquist.
    let nyq = sr / 2.0;
    let high = config.bandpass_high_hz.min(nyq * 0.95);
    let low = config.bandpass_low_hz.min(high - 0.5).max(0.001);
    let filter = BandpassFilter::new(config.filter_order, low, high, sr);

    let filtered: Vec<Vec<f64>> = series
        .data
        .iter()
        .map(|ch| filter.process(ch))
        .collect();

    // Resample each channel to the target rate via linear interpolation.
    let resampled: Vec<Vec<f64>> = filtered
        .iter()
        .map(|ch| resample_linear(ch, sr, config.resample_hz))
        .collect();

    MultiChannelTimeSeries::new(resampled, config.resample_hz, series.timestamp_start)
}

/// Linear-interpolation resampler from `src_hz` to `dst_hz`.
fn resample_linear(signal: &[f64], src_hz: f64, dst_hz: f64) -> Vec<f64> {
    if signal.is_empty() || (src_hz - dst_hz).abs() < 1e-9 {
        return signal.to_vec();
    }
    let duration_s = signal.len() as f64 / src_hz;
    let n_out = (duration_s * dst_hz).floor() as usize;
    if n_out == 0 {
        return Vec::new();
    }
    let ratio = src_hz / dst_hz;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let src_pos = i as f64 * ratio;
        let i0 = src_pos.floor() as usize;
        let i1 = (i0 + 1).min(signal.len() - 1);
        let frac = src_pos - i0 as f64;
        out.push(signal[i0] * (1.0 - frac) + signal[i1] * frac);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine_series(freq: f64, sr: f64, secs: f64, channels: usize) -> MultiChannelTimeSeries {
        let n = (sr * secs) as usize;
        let data: Vec<Vec<f64>> = (0..channels)
            .map(|_| {
                (0..n)
                    .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
                    .collect()
            })
            .collect();
        MultiChannelTimeSeries::new(data, sr, 0.0).unwrap()
    }

    #[test]
    fn resample_changes_rate_and_length() {
        let s = sine_series(5.0, 1000.0, 1.0, 2);
        let cfg = Brain2TextConfig::default(); // resample to 50 Hz
        let out = preprocess(&s, &cfg).unwrap();
        assert_eq!(out.sample_rate_hz, 50.0);
        // ~1 second at 50 Hz.
        assert!((out.num_samples as i64 - 50).abs() <= 1);
        assert_eq!(out.num_channels, 2);
    }

    #[test]
    fn bandpass_attenuates_out_of_band() {
        // 40 Hz tone should be attenuated by a 0.1-20 Hz bandpass.
        let s = sine_series(40.0, 500.0, 2.0, 1);
        let cfg = Brain2TextConfig {
            resample_hz: 500.0, // keep rate to measure amplitude fairly
            ..Default::default()
        };
        let out = preprocess(&s, &cfg).unwrap();
        let in_amp = s.data[0].iter().map(|x| x.abs()).fold(0.0, f64::max);
        let out_amp = out.data[0].iter().map(|x| x.abs()).fold(0.0, f64::max);
        assert!(out_amp < in_amp * 0.5, "in={in_amp} out={out_amp}");
    }

    #[test]
    fn resample_linear_identity_when_equal() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let out = resample_linear(&v, 100.0, 100.0);
        assert_eq!(out, v);
    }
}
