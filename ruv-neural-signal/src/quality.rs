//! Epoch admission gates for NeuroSleep. No interpolation is ever performed:
//! an epoch is either admitted as recorded or rejected with a typed reason.
//!
//! Clipping is judged **only** against the recorder's declared ADC rails. It is
//! deliberately not inferred from the data, because a quiet epoch's own extrema
//! are not rails — treating them as such rejects exactly the clean NREM epochs
//! the profile depends on.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Declared converter limits, in microvolts, taken from the acquisition
/// metadata rather than from the samples.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdcRails {
    /// Most negative representable value, in uV.
    pub minimum_uv: f64,
    /// Most positive representable value, in uV.
    pub maximum_uv: f64,
    /// Distance from a rail within which a sample counts as having hit it.
    pub tolerance_uv: f64,
}

impl AdcRails {
    /// Reject non-finite, inverted, or self-overlapping rail declarations.
    pub fn validate(self) -> Result<Self, QualityError> {
        if !self.minimum_uv.is_finite()
            || !self.maximum_uv.is_finite()
            || !self.tolerance_uv.is_finite()
        {
            return Err(QualityError::InvalidConfig("ADC rails must be finite"));
        }
        if self.maximum_uv <= self.minimum_uv {
            return Err(QualityError::InvalidConfig(
                "ADC maximum must exceed minimum",
            ));
        }
        if self.tolerance_uv < 0.0 || 2.0 * self.tolerance_uv >= self.maximum_uv - self.minimum_uv {
            return Err(QualityError::InvalidConfig(
                "ADC rail tolerance must be >= 0 and leave a non-empty interior",
            ));
        }
        Ok(self)
    }

    /// Whether a sample sits at or beyond either declared rail.
    pub fn is_at_rail(self, value_uv: f64) -> bool {
        value_uv <= self.minimum_uv + self.tolerance_uv
            || value_uv >= self.maximum_uv - self.tolerance_uv
    }
}

/// Typed epoch-assessment failures. These are input or configuration faults;
/// an epoch failing a *gate* is a successful assessment, not an error.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum QualityError {
    /// A configuration field was non-finite or outside its declared domain.
    #[error("invalid quality configuration: {0}")]
    InvalidConfig(&'static str),
    /// No channels were supplied.
    #[error("no channels supplied")]
    NoChannels,
}

/// Frozen epoch admission profile.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EpochQualityConfig {
    /// Declared converter limits used for the clipping gate.
    pub adc_rails: AdcRails,
    /// Maximum fraction of valid samples permitted to sit at a rail.
    pub maximum_clipped_fraction: f64,
    /// Duration of unchanging signal that counts as a flatline.
    pub flatline_seconds: f64,
    /// Peak-to-peak amplitude, in uV, below which a span counts as unchanging.
    pub flatline_epsilon_uv: f64,
    /// Longest run of consecutive invalid samples the epoch may contain.
    pub maximum_gap_seconds: f64,
    /// Maximum fraction of the epoch the mask may mark invalid.
    pub maximum_artifact_fraction: f64,
}

impl EpochQualityConfig {
    /// Reject non-finite fields, out-of-range fractions, and degenerate spans.
    pub fn validate(self) -> Result<Self, QualityError> {
        self.adc_rails.validate()?;
        let finite = [
            self.maximum_clipped_fraction,
            self.flatline_seconds,
            self.flatline_epsilon_uv,
            self.maximum_gap_seconds,
            self.maximum_artifact_fraction,
        ];
        if finite.iter().any(|value| !value.is_finite()) {
            return Err(QualityError::InvalidConfig(
                "every numeric field must be finite",
            ));
        }
        if !(0.0..=1.0).contains(&self.maximum_clipped_fraction)
            || !(0.0..=1.0).contains(&self.maximum_artifact_fraction)
        {
            return Err(QualityError::InvalidConfig(
                "fraction gates must lie in [0, 1]",
            ));
        }
        if self.flatline_seconds <= 0.0
            || self.flatline_epsilon_uv < 0.0
            || self.maximum_gap_seconds < 0.0
        {
            return Err(QualityError::InvalidConfig(
                "flatline span must be > 0 and epsilon/gap bounds >= 0",
            ));
        }
        Ok(self)
    }
}

/// Why an epoch was refused. Gates are evaluated in this declaration order and
/// the first failure is reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpochRejection {
    /// Channel lengths disagreed with each other or with the mask.
    MaskMismatch,
    /// A sample the mask declared valid was NaN or infinite.
    NonFinite,
    /// The mask invalidated more of the epoch than the profile allows.
    ArtifactCoverage,
    /// A single contiguous invalid run exceeded the profile's gap bound.
    Gap,
    /// Too many valid samples sat at a declared ADC rail.
    Clipped,
    /// A span of unchanging signal exceeded the profile's flatline bound.
    Flatline,
}

/// The measured facts behind an admission decision. All fractions are reported
/// whether or not their gate fired, so a rejection stays auditable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochQuality {
    /// Whether every gate passed.
    pub accepted: bool,
    /// Fraction of the epoch the mask marked invalid.
    pub artifact_fraction: f64,
    /// Worst per-channel fraction of valid samples sitting at a rail.
    pub clipped_fraction: f64,
    /// Longest contiguous invalid run, in seconds.
    pub longest_gap_seconds: f64,
    /// First gate that failed, if any.
    pub rejection: Option<EpochRejection>,
}

/// Assess one epoch against the frozen admission profile.
///
/// `shared_valid_mask` is a single mask covering every channel simultaneously,
/// so all channels are admitted or refused over exactly the same samples. Values
/// at invalid positions are never read, so callers may leave NaN in the gaps.
pub fn assess_epoch(
    channels: &[&[f64]],
    shared_valid_mask: &[bool],
    sample_rate_hz: f64,
    config: EpochQualityConfig,
) -> Result<EpochQuality, QualityError> {
    let config = config.validate()?;
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return Err(QualityError::InvalidConfig(
            "sample rate must be finite and > 0",
        ));
    }
    let Some(&first) = channels.first() else {
        return Err(QualityError::NoChannels);
    };
    let samples = first.len();
    if samples == 0 {
        return Err(QualityError::InvalidConfig("epoch must not be empty"));
    }
    if shared_valid_mask.len() != samples || channels.iter().any(|channel| channel.len() != samples)
    {
        return Ok(rejected(EpochRejection::MaskMismatch, 1.0, 1.0, f64::NAN));
    }

    let invalid = shared_valid_mask.iter().filter(|valid| !**valid).count();
    let artifact_fraction = invalid as f64 / samples as f64;
    let longest_gap_seconds = longest_invalid_run(shared_valid_mask) as f64 / sample_rate_hz;
    let valid_samples = samples - invalid;

    let clipped_fraction = if valid_samples == 0 {
        0.0
    } else {
        channels
            .iter()
            .map(|channel| {
                let at_rail = channel
                    .iter()
                    .zip(shared_valid_mask)
                    .filter(|(value, valid)| **valid && config.adc_rails.is_at_rail(**value))
                    .count();
                at_rail as f64 / valid_samples as f64
            })
            .fold(0.0, f64::max)
    };
    let report = |rejection| {
        Ok(rejected(
            rejection,
            artifact_fraction,
            clipped_fraction,
            longest_gap_seconds,
        ))
    };

    if channels.iter().any(|channel| {
        channel
            .iter()
            .zip(shared_valid_mask)
            .any(|(value, valid)| *valid && !value.is_finite())
    }) {
        return report(EpochRejection::NonFinite);
    }
    if artifact_fraction > config.maximum_artifact_fraction {
        return report(EpochRejection::ArtifactCoverage);
    }
    if longest_gap_seconds > config.maximum_gap_seconds {
        return report(EpochRejection::Gap);
    }
    if clipped_fraction > config.maximum_clipped_fraction {
        return report(EpochRejection::Clipped);
    }
    let flatline_samples = (config.flatline_seconds * sample_rate_hz).ceil() as usize;
    if flatline_samples > 0
        && flatline_samples <= samples
        && channels
            .iter()
            .any(|channel| has_flatline(channel, shared_valid_mask, flatline_samples, config))
    {
        return report(EpochRejection::Flatline);
    }

    Ok(EpochQuality {
        accepted: true,
        artifact_fraction,
        clipped_fraction,
        longest_gap_seconds,
        rejection: None,
    })
}

/// Longest run of consecutive `false` entries.
fn longest_invalid_run(mask: &[bool]) -> usize {
    let mut longest = 0usize;
    let mut current = 0usize;
    for valid in mask {
        current = if *valid { 0 } else { current + 1 };
        longest = longest.max(current);
    }
    longest
}

/// Scan only fully valid windows, so masked-out samples can neither hide a real
/// flatline nor manufacture a spurious one.
fn has_flatline(
    channel: &[f64],
    mask: &[bool],
    window_samples: usize,
    config: EpochQualityConfig,
) -> bool {
    channel
        .windows(window_samples)
        .zip(mask.windows(window_samples))
        .filter(|(_, valid)| valid.iter().all(|value| *value))
        .any(|(window, _)| {
            let (minimum, maximum) = window
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
                    (low.min(*value), high.max(*value))
                });
            maximum - minimum <= config.flatline_epsilon_uv
        })
}

fn rejected(
    rejection: EpochRejection,
    artifact_fraction: f64,
    clipped_fraction: f64,
    longest_gap_seconds: f64,
) -> EpochQuality {
    EpochQuality {
        accepted: false,
        artifact_fraction,
        clipped_fraction,
        longest_gap_seconds,
        rejection: Some(rejection),
    }
}

#[cfg(test)]
mod tests;
