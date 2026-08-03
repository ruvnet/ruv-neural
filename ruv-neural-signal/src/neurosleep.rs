//! Fallible, configuration-driven spectral primitives for NeuroSleep.
//!
//! Every entry point here is explicit about three things that silently differ
//! between EEG toolboxes and are therefore stated in the type system rather than
//! in prose:
//!
//! 1. **Masking.** Welch segments are accepted only when every sample they cover
//!    is marked valid by the caller-supplied mask. Invalid samples are never
//!    interpolated, zero-filled, or averaged over.
//! 2. **Units.** Power spectral density is uV^2/Hz; integrated band power is
//!    uV^2 ([`AbsoluteBandPower::UNIT`]); relative power is a dimensionless
//!    ratio that always carries the denominator band it was divided by.
//! 3. **Band edges.** Whether a band integral uses interpolated edges or only
//!    whole bins is a [`BandEdgePolicy`] the caller must choose.

use num_complex::Complex;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use thiserror::Error;

/// Typed, fail-closed DSP errors. No variant is recoverable by substituting a
/// default value; callers must decide to emit a typed null instead.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum DspError {
    /// A configuration field was non-finite or outside its declared domain.
    #[error("invalid DSP configuration: {0}")]
    InvalidConfig(&'static str),
    /// Fewer samples than one analysis window.
    #[error("insufficient samples: need {needed}, have {actual}")]
    InsufficientSamples { needed: usize, actual: usize },
    /// A sample the mask declared valid was NaN or infinite.
    #[error("non-finite sample at index {0}")]
    NonFiniteSample(usize),
    /// Paired inputs (signal/mask, frequency/density) had differing lengths.
    #[error("input lengths differ")]
    LengthMismatch,
    /// The mask left no fully valid analysis window.
    #[error("no fully valid analysis segment")]
    NoValidSegments,
    /// No spectrum bins overlap the requested band.
    #[error("no bins overlap the requested band")]
    EmptyBand,
    /// The denominator band integrated to zero or a negative value.
    #[error("relative power denominator is not positive")]
    NonPositiveDenominator,
    /// No periodic peak satisfied the frozen acceptance profile.
    #[error("no accepted periodic theta peak")]
    NoAcceptedPeak,
}

/// Welch averaging parameters. The window is a periodic Hann window, matching
/// the `fftbins=True` convention used by `scipy.signal.welch`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WelchConfig {
    /// Samples per Welch segment.
    pub window_samples: usize,
    /// Samples shared between consecutive segments.
    pub overlap_samples: usize,
    /// Subtract each segment's mean before windowing.
    pub detrend_mean: bool,
}

impl WelchConfig {
    /// Reject window/overlap combinations that would alias or never advance.
    pub fn validate(self) -> Result<Self, DspError> {
        if self.window_samples < 4 {
            return Err(DspError::InvalidConfig(
                "Welch window must span >= 4 samples",
            ));
        }
        if self.overlap_samples >= self.window_samples {
            return Err(DspError::InvalidConfig(
                "Welch overlap must be shorter than the window",
            ));
        }
        Ok(self)
    }

    /// Frequency resolution of the resulting spectrum, in Hz.
    pub fn bin_spacing_hz(self, sample_rate_hz: f64) -> f64 {
        sample_rate_hz / self.window_samples as f64
    }
}

/// Half-open-by-default band description. `include_low`/`include_high` only
/// affect [`BandEdgePolicy::BinInclusive`]; interpolated edges always integrate
/// the closed interval `[low_hz, high_hz]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrequencyBandConfig {
    /// Lower edge in Hz.
    pub low_hz: f64,
    /// Upper edge in Hz.
    pub high_hz: f64,
    /// Whether a bin exactly at `low_hz` counts under a bin-inclusive policy.
    pub include_low: bool,
    /// Whether a bin exactly at `high_hz` counts under a bin-inclusive policy.
    pub include_high: bool,
}

impl FrequencyBandConfig {
    /// Half-open `[low, high)` band, the convention used by the sleep profiles.
    pub const fn half_open(low_hz: f64, high_hz: f64) -> Self {
        Self {
            low_hz,
            high_hz,
            include_low: true,
            include_high: false,
        }
    }

    /// Reject non-finite, negative, or inverted bands.
    pub fn validate(self) -> Result<Self, DspError> {
        if !self.low_hz.is_finite()
            || !self.high_hz.is_finite()
            || self.low_hz < 0.0
            || self.high_hz <= self.low_hz
        {
            return Err(DspError::InvalidConfig("invalid frequency band"));
        }
        Ok(self)
    }

    /// Width in Hz, used when reducing a band integral to a band mean.
    pub fn width_hz(self) -> f64 {
        self.high_hz - self.low_hz
    }
}

/// How a band integral treats the band edges relative to the frequency grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BandEdgePolicy {
    /// Integrate the exact closed interval, linearly interpolating the density
    /// at `low_hz` and `high_hz` when they fall between bins.
    InterpolatedEdges,
    /// Integrate only over bin centres inside the band, honouring
    /// `include_low`/`include_high`. Edges are not interpolated.
    BinInclusive,
}

/// One-sided power spectral density in uV^2/Hz on a uniform frequency grid.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerSpectrum {
    /// Bin centre frequencies in Hz, strictly increasing.
    pub frequencies_hz: Vec<f64>,
    /// Power spectral density in uV^2/Hz.
    pub density: Vec<f64>,
    /// Number of fully valid Welch segments averaged.
    pub segments: usize,
}

impl PowerSpectrum {
    /// Physical unit of [`PowerSpectrum::density`].
    pub const DENSITY_UNIT: &'static str = "uV2_per_hz";

    /// Grid spacing in Hz, or `None` for a degenerate one-bin spectrum.
    pub fn bin_spacing_hz(&self) -> Option<f64> {
        (self.frequencies_hz.len() >= 2).then(|| self.frequencies_hz[1] - self.frequencies_hz[0])
    }
}

/// Welch PSD over an all-valid signal. Equivalent to [`welch_psd_masked`] with
/// a mask of all `true`.
pub fn welch_psd(
    signal: &[f64],
    sample_rate_hz: f64,
    config: WelchConfig,
) -> Result<PowerSpectrum, DspError> {
    welch_psd_masked(signal, &vec![true; signal.len()], sample_rate_hz, config)
}

/// Welch PSD restricted to segments whose samples are all marked valid.
///
/// Samples the mask marks invalid are never read, so a caller may leave NaN or
/// sentinel values in the gaps. A segment overlapping even one invalid sample is
/// dropped whole; it is not shortened, padded, or interpolated.
pub fn welch_psd_masked(
    signal: &[f64],
    valid_mask: &[bool],
    sample_rate_hz: f64,
    config: WelchConfig,
) -> Result<PowerSpectrum, DspError> {
    let config = config.validate()?;
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return Err(DspError::InvalidConfig(
            "sample rate must be finite and > 0",
        ));
    }
    if signal.len() != valid_mask.len() {
        return Err(DspError::LengthMismatch);
    }
    if signal.len() < config.window_samples {
        return Err(DspError::InsufficientSamples {
            needed: config.window_samples,
            actual: signal.len(),
        });
    }
    if let Some((index, _)) = signal
        .iter()
        .zip(valid_mask)
        .enumerate()
        .find(|(_, (value, valid))| **valid && !value.is_finite())
    {
        return Err(DspError::NonFiniteSample(index));
    }

    let n = config.window_samples;
    let hop = n - config.overlap_samples;
    let window = periodic_hann(n);
    let window_power: f64 = window.iter().map(|value| value * value).sum();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let bins = n / 2 + 1;
    let mut density = vec![0.0; bins];
    let mut segments = 0usize;
    let mut start = 0usize;
    while start + n <= signal.len() {
        if valid_mask[start..start + n].iter().all(|valid| *valid) {
            let mean = if config.detrend_mean {
                signal[start..start + n].iter().sum::<f64>() / n as f64
            } else {
                0.0
            };
            let mut values: Vec<Complex<f64>> = (0..n)
                .map(|index| Complex::new((signal[start + index] - mean) * window[index], 0.0))
                .collect();
            fft.process(&mut values);
            for (bin, accumulator) in density.iter_mut().enumerate() {
                let one_sided = if bin == 0 || (n % 2 == 0 && bin == n / 2) {
                    1.0
                } else {
                    2.0
                };
                *accumulator += values[bin].norm_sqr() * one_sided;
            }
            segments += 1;
        }
        start += hop;
    }
    if segments == 0 {
        return Err(DspError::NoValidSegments);
    }
    let normalization = segments as f64 * sample_rate_hz * window_power;
    for value in &mut density {
        *value /= normalization;
    }
    Ok(PowerSpectrum {
        frequencies_hz: (0..bins)
            .map(|bin| bin as f64 * sample_rate_hz / n as f64)
            .collect(),
        density,
        segments,
    })
}

/// Periodic (`fftbins=True`) Hann window.
fn periodic_hann(length: usize) -> Vec<f64> {
    (0..length)
        .map(|index| 0.5 * (1.0 - (2.0 * PI * index as f64 / length as f64).cos()))
        .collect()
}

/// Trapezoidally integrated absolute band power in uV^2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AbsoluteBandPower {
    /// Integrated power, in uV^2.
    pub micro_volts_squared: f64,
    /// Band that was integrated.
    pub band: FrequencyBandConfig,
    /// Edge treatment used for the integral.
    pub edge_policy: BandEdgePolicy,
}

impl AbsoluteBandPower {
    /// Physical unit tag carried into the evidence contract.
    pub const UNIT: &'static str = "uV2";

    /// Band-averaged density in uV^2/Hz, i.e. the integral over the band width.
    pub fn mean_density(&self) -> f64 {
        self.micro_volts_squared / self.band.width_hz()
    }
}

/// Integrate a PSD over a band with the trapezoidal rule.
///
/// The result is absolute power in uV^2: a density in uV^2/Hz integrated over
/// Hz. The frequency grid must be finite and strictly increasing.
pub fn integrate_band_trapezoidal(
    frequencies: &[f64],
    density: &[f64],
    band: FrequencyBandConfig,
    edge_policy: BandEdgePolicy,
) -> Result<AbsoluteBandPower, DspError> {
    let band = band.validate()?;
    if frequencies.len() != density.len() {
        return Err(DspError::LengthMismatch);
    }
    if frequencies.len() < 2
        || frequencies.iter().any(|value| !value.is_finite())
        || density.iter().any(|value| !value.is_finite())
        || frequencies.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err(DspError::InvalidConfig(
            "spectrum must be finite with a strictly increasing grid",
        ));
    }

    let mut points: Vec<(f64, f64)> = Vec::with_capacity(frequencies.len() + 2);
    if edge_policy == BandEdgePolicy::InterpolatedEdges {
        for boundary in [band.low_hz, band.high_hz] {
            if let Some(value) = interpolate(frequencies, density, boundary) {
                points.push((boundary, value));
            }
        }
    }
    for (&frequency, &power) in frequencies.iter().zip(density) {
        let inside = match edge_policy {
            BandEdgePolicy::InterpolatedEdges => {
                frequency > band.low_hz && frequency < band.high_hz
            }
            BandEdgePolicy::BinInclusive => {
                (frequency > band.low_hz || (band.include_low && frequency == band.low_hz))
                    && (frequency < band.high_hz
                        || (band.include_high && frequency == band.high_hz))
            }
        };
        if inside {
            points.push((frequency, power));
        }
    }
    points.sort_by(|a, b| a.0.total_cmp(&b.0));
    points.dedup_by(|a, b| a.0 == b.0);
    if points.len() < 2 {
        return Err(DspError::EmptyBand);
    }
    Ok(AbsoluteBandPower {
        micro_volts_squared: points
            .windows(2)
            .map(|pair| (pair[1].0 - pair[0].0) * (pair[0].1 + pair[1].1) * 0.5)
            .sum(),
        band,
        edge_policy,
    })
}

fn interpolate(frequencies: &[f64], density: &[f64], target: f64) -> Option<f64> {
    frequencies
        .windows(2)
        .enumerate()
        .find_map(|(index, pair)| {
            (target >= pair[0] && target <= pair[1]).then(|| {
                let fraction = (target - pair[0]) / (pair[1] - pair[0]);
                density[index] + fraction * (density[index + 1] - density[index])
            })
        })
}

/// Numerator and denominator bands plus the edge policy applied to both.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelativePowerConfig {
    /// Band whose power forms the numerator.
    pub numerator: FrequencyBandConfig,
    /// Explicit denominator band. There is no implicit "total power".
    pub denominator: FrequencyBandConfig,
    /// Edge treatment, applied identically to both integrals.
    pub edge_policy: BandEdgePolicy,
}

impl RelativePowerConfig {
    /// Validate both bands and require the denominator to contain the numerator.
    pub fn validate(self) -> Result<Self, DspError> {
        let numerator = self.numerator.validate()?;
        let denominator = self.denominator.validate()?;
        if numerator.low_hz < denominator.low_hz || numerator.high_hz > denominator.high_hz {
            return Err(DspError::InvalidConfig(
                "relative power numerator must lie inside the denominator band",
            ));
        }
        Ok(self)
    }
}

/// A relative power ratio that carries the exact denominator it used.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativeBandPower {
    /// Dimensionless numerator/denominator ratio.
    pub ratio: f64,
    /// Numerator power in uV^2.
    pub numerator: AbsoluteBandPower,
    /// Denominator power in uV^2.
    pub denominator: AbsoluteBandPower,
}

impl RelativeBandPower {
    /// Unit tag for [`RelativeBandPower::ratio`].
    pub const UNIT: &'static str = "ratio";
}

/// Relative band power with an explicit denominator band and edge policy.
pub fn relative_band_power(
    frequencies: &[f64],
    density: &[f64],
    config: RelativePowerConfig,
) -> Result<RelativeBandPower, DspError> {
    let config = config.validate()?;
    let numerator =
        integrate_band_trapezoidal(frequencies, density, config.numerator, config.edge_policy)?;
    let denominator =
        integrate_band_trapezoidal(frequencies, density, config.denominator, config.edge_policy)?;
    if denominator.micro_volts_squared <= 0.0 {
        return Err(DspError::NonPositiveDenominator);
    }
    Ok(RelativeBandPower {
        ratio: numerator.micro_volts_squared / denominator.micro_volts_squared,
        numerator,
        denominator,
    })
}

/// Acceptance profile for the periodic theta peak.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThetaPeakConfig {
    /// Lower search bound in Hz.
    pub low_hz: f64,
    /// Upper search bound in Hz.
    pub high_hz: f64,
    /// Minimum log10 power above the aperiodic background.
    pub minimum_log10_prominence: f64,
    /// Grid spacing the caller guarantees. Coarser spectra are rejected rather
    /// than reported with a centre frequency the grid cannot support.
    pub maximum_bin_spacing_hz: f64,
}

impl ThetaPeakConfig {
    /// Reject non-finite or inverted search bounds and non-positive gates.
    pub fn validate(self) -> Result<Self, DspError> {
        if !self.low_hz.is_finite()
            || !self.high_hz.is_finite()
            || self.low_hz <= 0.0
            || self.high_hz <= self.low_hz
            || !self.minimum_log10_prominence.is_finite()
            || self.minimum_log10_prominence < 0.0
            || !self.maximum_bin_spacing_hz.is_finite()
            || self.maximum_bin_spacing_hz <= 0.0
        {
            return Err(DspError::InvalidConfig("invalid theta peak configuration"));
        }
        Ok(self)
    }
}

/// A periodic peak refined off the discrete frequency grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodicPeak {
    /// Sub-bin centre frequency in Hz from parabolic vertex interpolation.
    pub center_frequency_hz: f64,
    /// log10 PSD at the winning bin, in log10(uV^2/Hz).
    pub log10_power: f64,
    /// log10 power above the aperiodic background at the winning bin.
    pub log10_prominence: f64,
    /// Grid spacing used, in Hz. Bounds the residual interpolation error.
    pub bin_spacing_hz: f64,
}

/// Locate the strongest periodic theta peak above a fitted aperiodic background.
///
/// The winning bin is the local maximum of the log10 residual inside the search
/// band; its centre frequency is then refined by parabolic vertex interpolation
/// over the residual, which resolves the peak to a fraction of one bin rather
/// than snapping it to the grid.
pub fn periodic_theta_peak(
    frequencies: &[f64],
    density: &[f64],
    background_log10: &[f64],
    config: ThetaPeakConfig,
) -> Result<PeriodicPeak, DspError> {
    let config = config.validate()?;
    if frequencies.len() != density.len() || density.len() != background_log10.len() {
        return Err(DspError::LengthMismatch);
    }
    if frequencies.len() < 3 {
        return Err(DspError::InsufficientSamples {
            needed: 3,
            actual: frequencies.len(),
        });
    }
    let spacing = frequencies[1] - frequencies[0];
    if !spacing.is_finite() || spacing <= 0.0 {
        return Err(DspError::InvalidConfig(
            "theta peak requires a uniform increasing frequency grid",
        ));
    }
    if spacing > config.maximum_bin_spacing_hz {
        return Err(DspError::InvalidConfig(
            "spectrum resolution is coarser than the theta peak profile allows",
        ));
    }

    let residual: Vec<f64> = density
        .iter()
        .zip(background_log10)
        .map(|(power, background)| {
            if *power > 0.0 && background.is_finite() {
                power.log10() - background
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();

    let mut best: Option<usize> = None;
    for index in 1..frequencies.len() - 1 {
        let frequency = frequencies[index];
        if frequency < config.low_hz || frequency > config.high_hz {
            continue;
        }
        if residual[index] < config.minimum_log10_prominence
            || residual[index] < residual[index - 1]
            || residual[index] < residual[index + 1]
        {
            continue;
        }
        if best.is_none_or(|previous| residual[index] > residual[previous]) {
            best = Some(index);
        }
    }
    let index = best.ok_or(DspError::NoAcceptedPeak)?;

    let (left, center, right) = (residual[index - 1], residual[index], residual[index + 1]);
    let curvature = left - 2.0 * center + right;
    let offset = if curvature < 0.0 {
        (0.5 * (left - right) / curvature).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    Ok(PeriodicPeak {
        center_frequency_hz: frequencies[index] + offset * spacing,
        log10_power: density[index].log10(),
        log10_prominence: center,
        bin_spacing_hz: spacing,
    })
}

#[cfg(test)]
mod tests;
