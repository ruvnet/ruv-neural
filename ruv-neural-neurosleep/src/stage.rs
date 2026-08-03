//! Per-stage qEEG aggregation.
//!
//! Every metric is derived only from epochs the admission gates accepted, and
//! every metric that cannot be derived becomes an explicit typed null rather
//! than a substituted value.

use ruv_neural_core::neurosleep::{FeatureValue, NullReason, SleepState, StateQeegFeatures};
use ruv_neural_signal::aperiodic::{fit_aperiodic, AperiodicFit};
use ruv_neural_signal::connectivity::coherence_masked;
use ruv_neural_signal::neurosleep::{
    integrate_band_trapezoidal, periodic_theta_peak, relative_band_power, welch_psd_masked,
    AbsoluteBandPower, FrequencyBandConfig, PeriodicPeak, RelativeBandPower, RelativePowerConfig,
};

use crate::profile::NeuroSleepProfile;
use crate::{ExpertEpoch, NeuroSleepError};

/// Unit tag the V1 contract requires for `spectral_fit_error`. It differs from
/// the offset's unit on purpose: the fit error is a residual in log10 power, not
/// a spectral density.
const FIT_ERROR_UNIT: &str = "log10_power";

/// Compute one stage's features, or an all-null block if the stage is too short.
pub fn stage_features(
    stage: SleepState,
    epochs: &[ExpertEpoch],
    accepted: &[bool],
    stage_seconds: f64,
    profile: &NeuroSleepProfile,
) -> Result<StateQeegFeatures, NeuroSleepError> {
    if stage_seconds < profile.sufficiency.minimum_for(stage) {
        return Ok(null_features(stage, NullReason::InsufficientStageDuration));
    }
    let selected: Vec<&ExpertEpoch> = epochs
        .iter()
        .zip(accepted)
        .filter(|(epoch, admitted)| **admitted && epoch.state == stage)
        .map(|(epoch, _)| epoch)
        .collect();
    if selected.is_empty() {
        return Ok(null_features(stage, NullReason::InsufficientStageDuration));
    }

    let mut spectra = Vec::with_capacity(selected.len());
    let mut coherences = Vec::new();
    for epoch in &selected {
        spectra.push(welch_psd_masked(
            &epoch.channels_uv[0],
            &epoch.shared_valid_mask,
            profile.sample_rate_hz,
            profile.welch,
        )?);
        if epoch.channels_uv.len() >= 2 {
            coherences.push(coherence_masked(
                &epoch.channels_uv[0],
                &epoch.channels_uv[1],
                &epoch.shared_valid_mask,
                profile.sample_rate_hz,
                profile.coherence,
            )?);
        }
    }
    // Every epoch shares one profile, so every spectrum shares one grid.
    let frequencies = spectra[0].frequencies_hz.clone();
    let density = mean_of(
        &spectra
            .iter()
            .map(|s| s.density.clone())
            .collect::<Vec<_>>(),
    )?;

    let bands = &profile.bands;
    let absolute = |band: FrequencyBandConfig| -> Result<AbsoluteBandPower, NeuroSleepError> {
        Ok(integrate_band_trapezoidal(
            &frequencies,
            &density,
            band,
            bands.edge_policy,
        )?)
    };
    let relative = |band: FrequencyBandConfig| -> Result<RelativeBandPower, NeuroSleepError> {
        Ok(relative_band_power(
            &frequencies,
            &density,
            RelativePowerConfig {
                numerator: band,
                denominator: bands.relative_denominator,
                edge_policy: bands.edge_policy,
            },
        )?)
    };

    let fit = fit_aperiodic(&frequencies, &density, profile.aperiodic).ok();
    let peak = fit.as_ref().and_then(|fit| {
        periodic_theta_peak(
            &frequencies,
            &density,
            &fit.predicted_log10,
            profile.theta_peak,
        )
        .ok()
    });
    let coherence = stage_coherence(&coherences, profile)?;

    Ok(StateQeegFeatures {
        state: stage,
        delta_absolute_power: observed(
            absolute(bands.delta)?.micro_volts_squared,
            AbsoluteBandPower::UNIT,
        ),
        delta_relative_power: observed(relative(bands.delta)?.ratio, RelativeBandPower::UNIT),
        theta_absolute_power: observed(
            absolute(bands.theta)?.micro_volts_squared,
            AbsoluteBandPower::UNIT,
        ),
        theta_relative_power: observed(relative(bands.theta)?.ratio, RelativeBandPower::UNIT),
        alpha_absolute_power: observed(
            absolute(bands.alpha)?.micro_volts_squared,
            AbsoluteBandPower::UNIT,
        ),
        theta_peak_frequency: peak_value(peak, |peak| observed(peak.center_frequency_hz, "Hz")),
        theta_peak_power: peak_value(peak, |peak| {
            observed(peak.log10_power, AperiodicFit::OFFSET_UNIT)
        }),
        frontal_parietal_full_band_coherence: coherence.map_or_else(
            || null(NullReason::MissingSynchronousChannels),
            |(_, full_band)| observed(full_band, RelativeBandPower::UNIT),
        ),
        frontal_parietal_theta_coherence: coherence.map_or_else(
            || null(NullReason::MissingSynchronousChannels),
            |(theta, _)| observed(theta, RelativeBandPower::UNIT),
        ),
        aperiodic_exponent: aperiodic_value(stage, profile, fit.as_ref(), |fit| {
            observed(fit.exponent, AperiodicFit::EXPONENT_UNIT)
        }),
        aperiodic_offset: aperiodic_value(stage, profile, fit.as_ref(), |fit| {
            observed(fit.offset, AperiodicFit::OFFSET_UNIT)
        }),
        spectral_fit_error: aperiodic_value(stage, profile, fit.as_ref(), |fit| {
            observed(fit.rmse_log10, FIT_ERROR_UNIT)
        }),
    })
}

/// Band-mean coherence for the theta band and the full denominator band.
///
/// Per-epoch coherence spectra are averaged across epochs. Magnitude-squared
/// coherence is positively biased at small segment counts, so this stage value
/// is an upper-biased estimate; pooling cross-spectra across epochs instead of
/// averaging per-epoch estimates would remove that bias and is not implemented.
fn stage_coherence(
    coherences: &[Vec<(f64, f64)>],
    profile: &NeuroSleepProfile,
) -> Result<Option<(f64, f64)>, NeuroSleepError> {
    let Some(first) = coherences.first() else {
        return Ok(None);
    };
    let frequencies: Vec<f64> = first.iter().map(|(frequency, _)| *frequency).collect();
    let values = mean_of(
        &coherences
            .iter()
            .map(|spectrum| spectrum.iter().map(|(_, value)| *value).collect())
            .collect::<Vec<Vec<f64>>>(),
    )?;
    // The trapezoidal band integrator is unit-agnostic; dividing by the band
    // width turns the integral of a dimensionless coherence back into a
    // dimensionless band mean.
    let band_mean = |band: FrequencyBandConfig| -> Result<f64, NeuroSleepError> {
        Ok(
            integrate_band_trapezoidal(&frequencies, &values, band, profile.bands.edge_policy)?
                .micro_volts_squared
                / band.width_hz(),
        )
    };
    Ok(Some((
        band_mean(profile.bands.theta)?,
        band_mean(profile.bands.relative_denominator)?,
    )))
}

fn mean_of(vectors: &[Vec<f64>]) -> Result<Vec<f64>, NeuroSleepError> {
    let Some(first) = vectors.first() else {
        return Err(NeuroSleepError::InvalidEpoch("nothing to aggregate"));
    };
    if vectors.iter().any(|vector| vector.len() != first.len()) {
        return Err(NeuroSleepError::InvalidEpoch(
            "aggregated spectra have differing lengths",
        ));
    }
    Ok((0..first.len())
        .map(|index| vectors.iter().map(|vector| vector[index]).sum::<f64>() / vectors.len() as f64)
        .collect())
}

fn peak_value(
    peak: Option<PeriodicPeak>,
    project: impl Fn(PeriodicPeak) -> FeatureValue,
) -> FeatureValue {
    peak.map_or_else(|| null(NullReason::NoAcceptedPeak), project)
}

/// Aperiodic parameters are reported only for the stages the profile lists;
/// elsewhere they are an explicit `not_applicable` null rather than a value.
fn aperiodic_value(
    stage: SleepState,
    profile: &NeuroSleepProfile,
    fit: Option<&AperiodicFit>,
    project: impl Fn(&AperiodicFit) -> FeatureValue,
) -> FeatureValue {
    if !profile.report_aperiodic_for.contains(&stage) {
        return null(NullReason::NotApplicable);
    }
    fit.map_or_else(|| null(NullReason::FitQualityFailed), project)
}

pub(crate) fn observed(value: f64, unit: &str) -> FeatureValue {
    FeatureValue::Observed {
        value,
        unit: unit.into(),
    }
}

pub(crate) fn null(reason: NullReason) -> FeatureValue {
    FeatureValue::Null { reason }
}

fn null_features(state: SleepState, reason: NullReason) -> StateQeegFeatures {
    StateQeegFeatures {
        state,
        delta_absolute_power: null(reason),
        delta_relative_power: null(reason),
        theta_absolute_power: null(reason),
        theta_relative_power: null(reason),
        alpha_absolute_power: null(reason),
        theta_peak_frequency: null(reason),
        theta_peak_power: null(reason),
        frontal_parietal_full_band_coherence: null(reason),
        frontal_parietal_theta_coherence: null(reason),
        aperiodic_exponent: null(reason),
        aperiodic_offset: null(reason),
        spectral_fit_error: null(reason),
    }
}
