//! Annotation-first, stage-aware NeuroSleep nightly aggregation.
//!
//! This crate is **annotation-first**: it never infers sleep stages. Every epoch
//! arrives already carrying an expert scorer's label, and the crate's job is to
//! decide which of those epochs are clean enough to measure, to measure them,
//! and to say plainly when it cannot.
//!
//! Three properties hold throughout:
//!
//! - **No imputation.** An epoch is admitted as recorded or rejected with a
//!   typed reason. Masked-out samples are never interpolated or zero-filled.
//! - **No substituted values.** A metric that cannot be derived is emitted as an
//!   explicit typed null carrying the reason, never as a default number.
//! - **No hidden configuration.** The whole analysis depends on one
//!   [`NeuroSleepProfile`], validated in full before any epoch is touched.
//!
//! Signing uses an injected persistent signer; this crate never generates key
//! material. See [`analyze_and_sign`].

#![deny(missing_docs)]

mod aggregate;
mod bundle;
mod profile;
mod stage;

pub use aggregate::StageAggregate;
pub use bundle::{analyze_and_sign, NightBundleContext};
pub use profile::{BandProfile, NeuroSleepProfile, StageSufficiency};

use ruv_neural_core::attestation::NeuroSleepAttestationError;
use ruv_neural_core::neurosleep::{
    NeuroSleepContractError, NightQuality, SleepState, StageSummary, StateQeegFeatures,
};
use ruv_neural_signal::aperiodic::AperiodicError;
use ruv_neural_signal::neurosleep::DspError;
use ruv_neural_signal::quality::{assess_epoch, EpochRejection, QualityError};
use thiserror::Error;

/// One expert-scored epoch, with a single validity mask shared by every channel.
///
/// The mask is deliberately not per-channel: coherence and power must be derived
/// from exactly the same samples, so a sample is either valid for the whole
/// epoch or for none of it.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertEpoch {
    /// The expert scorer's label. Never inferred by this crate.
    pub state: SleepState,
    /// Channel signals in microvolts. Channel 0 is the spectral channel; when a
    /// second channel is present it is used as the coherence partner.
    pub channels_uv: Vec<Vec<f64>>,
    /// Samples every channel observed simultaneously. Values at `false`
    /// positions are never read and may be NaN.
    pub shared_valid_mask: Vec<bool>,
}

/// Fail-closed errors from night analysis.
#[derive(Debug, Error)]
pub enum NeuroSleepError {
    /// The frozen profile was internally inconsistent or out of domain.
    #[error("invalid NeuroSleep profile: {0}")]
    InvalidProfile(&'static str),
    /// An epoch's shape disagreed with the profile.
    #[error("invalid epoch: {0}")]
    InvalidEpoch(&'static str),
    /// A spectral primitive failed.
    #[error(transparent)]
    Dsp(#[from] DspError),
    /// Aperiodic fitting failed for a reason other than fit quality.
    #[error(transparent)]
    Aperiodic(#[from] AperiodicError),
    /// Epoch admission could not be assessed.
    #[error(transparent)]
    Quality(#[from] QualityError),
    /// Signing failed.
    #[error(transparent)]
    Attestation(#[from] NeuroSleepAttestationError),
    /// The derived payload did not satisfy the V1 evidence contract.
    #[error(transparent)]
    Contract(#[from] NeuroSleepContractError),
}

/// Everything derived from one night, before it is bound into a signed bundle.
#[derive(Debug, Clone, PartialEq)]
pub struct NightAnalysis {
    /// Night-level admission facts.
    pub quality: NightQuality,
    /// Stage durations and bouts.
    pub stage_summary: StageSummary,
    /// Per-stage qEEG features, in Wake/NREM/REM order.
    pub qeeg_by_stage: Vec<StateQeegFeatures>,
    /// Per-epoch admission decisions, in input order.
    pub accepted: Vec<bool>,
    /// Per-epoch rejection reasons, in input order.
    pub rejections: Vec<Option<EpochRejection>>,
}

/// Every stage the V1 contract reports, in a fixed order so the canonical
/// payload does not depend on which stages the night happened to contain.
const REPORTED_STAGES: [SleepState; 3] = [SleepState::Wake, SleepState::Nrem, SleepState::Rem];

/// Analyse one night of expert-scored epochs against a frozen profile.
pub fn analyze_night(
    epochs: &[ExpertEpoch],
    profile: &NeuroSleepProfile,
) -> Result<NightAnalysis, NeuroSleepError> {
    profile.validate()?;
    if epochs.is_empty() {
        return Err(NeuroSleepError::InvalidEpoch("no epochs"));
    }
    let expected = profile.epoch_samples();

    let mut accepted = Vec::with_capacity(epochs.len());
    let mut rejections = Vec::with_capacity(epochs.len());
    let mut artifact_sum = 0.0;
    for epoch in epochs {
        if epoch.channels_uv.is_empty()
            || epoch
                .channels_uv
                .iter()
                .any(|channel| channel.len() != expected)
            || epoch.shared_valid_mask.len() != expected
        {
            return Err(NeuroSleepError::InvalidEpoch(
                "epoch length or channel count disagrees with the profile",
            ));
        }
        let channels: Vec<&[f64]> = epoch.channels_uv.iter().map(Vec::as_slice).collect();
        let quality = assess_epoch(
            &channels,
            &epoch.shared_valid_mask,
            profile.sample_rate_hz,
            profile.quality,
        )?;
        artifact_sum += quality.artifact_fraction;
        accepted.push(quality.accepted);
        rejections.push(quality.rejection);
    }

    let coverage = accepted.iter().filter(|value| **value).count() as f64 / epochs.len() as f64;
    let mut reason_codes = Vec::new();
    if coverage < profile.minimum_valid_coverage {
        reason_codes.push("insufficient_valid_coverage".to_string());
    }

    let aggregate = StageAggregate::new(epochs, &accepted, profile.epoch_seconds);
    let qeeg_by_stage = REPORTED_STAGES
        .into_iter()
        .map(|state| {
            stage::stage_features(state, epochs, &accepted, aggregate.duration(state), profile)
        })
        .collect::<Result<_, _>>()?;

    Ok(NightAnalysis {
        quality: NightQuality {
            valid_coverage_fraction: coverage,
            artifact_fraction: artifact_sum / epochs.len() as f64,
            accepted: reason_codes.is_empty(),
            reason_codes,
        },
        stage_summary: aggregate.summary(profile),
        qeeg_by_stage,
        accepted,
        rejections,
    })
}

#[cfg(test)]
mod tests;
