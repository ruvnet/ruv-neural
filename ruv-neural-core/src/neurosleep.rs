//! Versioned NeuroSleep qEEG evidence contract.
//!
//! This module contains wire types and deterministic payload validation only.
//! Signal processing, file parsing, staging, and longitudinal interpretation
//! deliberately live outside `ruv-neural-core`.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

/// Exact schema identifier for the first NeuroSleep wire contract.
pub const NEUROSLEEP_SCHEMA_V1: &str = "ruv-neural/neurosleep/1";

/// Errors raised before a payload may be hashed, signed, or accepted.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NeuroSleepContractError {
    /// The wire schema is not the exact supported schema.
    #[error("unsupported NeuroSleep schema: {0}")]
    UnsupportedSchema(String),
    /// A required string is empty or otherwise malformed.
    #[error("invalid field {field}: {reason}")]
    InvalidField {
        /// Field path.
        field: &'static str,
        /// Stable reason.
        reason: &'static str,
    },
    /// A numeric value is NaN or infinite.
    #[error("non-finite numeric value in {0}")]
    NonFinite(&'static str),
    /// The detached payload digest did not match the canonical payload.
    #[error("payload digest mismatch")]
    DigestMismatch,
    /// RFC 8785 serialization failed.
    #[error("canonical JSON serialization failed: {0}")]
    CanonicalJson(String),
}

/// Species is explicit because evidence maturity differs across species.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Species {
    /// Human participant.
    Human,
    /// Laboratory mouse.
    Mouse,
}

/// Formats admitted by the version-one contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceFormat {
    /// European Data Format.
    Edf,
    /// EDF Plus.
    EdfPlus,
    /// BrainVision header/marker/binary set.
    BrainVision,
}

/// How sleep stages were assigned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum StageSource {
    /// Imported expert-scored hypnogram.
    ExpertHypnogram { scorer_type: String },
    /// Stages emitted by named device software.
    Device { name: String, version: String },
    /// Stages emitted by a frozen model.
    Model {
        name: String,
        version: String,
        model_sha256: String,
    },
    /// HR/motion context only; never paper-equivalent staging.
    ProxyContext { name: String, version: String },
}

/// Canonical sleep state used by nightly summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SleepState {
    /// Awake.
    Wake,
    /// Non-rapid-eye-movement sleep.
    Nrem,
    /// Rapid-eye-movement sleep.
    Rem,
}

/// Physiological channel modalities supported by the V1 contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcquisitionModality {
    /// Electroencephalography.
    Eeg,
    /// Electrooculography.
    Eog,
    /// Electromyography.
    Emg,
}

/// One acquisition channel and its reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcquisitionChannel {
    /// Channel label as imported.
    pub name: String,
    /// Physiological modality.
    pub modality: AcquisitionModality,
    /// Physical or derived reference.
    pub reference: String,
}

/// Acquisition facts that affect compatibility.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcquisitionMetadata {
    /// Device manufacturer/model, or a stable research rig identifier.
    pub device_model: String,
    /// Hardware revision when known.
    pub hardware_version: Option<String>,
    /// Firmware revision when known.
    pub firmware_version: Option<String>,
    /// Samples per second.
    pub sampling_rate_hz: f64,
    /// Ordered montage/channel description.
    pub channels: Vec<AcquisitionChannel>,
}

/// One fully declared DSP setting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlgorithmParameter {
    /// Stable parameter name.
    pub name: String,
    /// Deterministic textual value.
    pub value: String,
    /// Unit, or `dimensionless`.
    pub unit: String,
}

/// Complete identity of the extractor and its configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlgorithmManifest {
    /// Source commit used for the build.
    pub pipeline_commit: String,
    /// Released crate versions participating in extraction.
    pub crate_versions: Vec<String>,
    /// SHA-256 of the extractor binary.
    pub extractor_sha256: String,
    /// SHA-256 of the complete serialized configuration.
    pub configuration_sha256: String,
    /// Stage scorer identity.
    pub stage_source: StageSource,
    /// Ordered, explicit DSP parameter list.
    pub dsp_parameters: Vec<AlgorithmParameter>,
}

/// Machine-readable reason for a missing measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NullReason {
    /// Not enough valid stage time.
    InsufficientStageDuration,
    /// Artifact burden exceeded the profile gate.
    ExcessiveArtifact,
    /// A periodic peak did not satisfy the frozen profile.
    NoAcceptedPeak,
    /// Aperiodic fit quality did not satisfy the frozen profile.
    FitQualityFailed,
    /// Required synchronous channels were unavailable.
    MissingSynchronousChannels,
    /// This metric is not applicable to the stage/profile.
    NotApplicable,
}

/// A finite unit-tagged observation or an explicit typed null.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum FeatureValue {
    /// Successfully measured value.
    Observed { value: f64, unit: String },
    /// No value was emitted.
    Null { reason: NullReason },
}

impl FeatureValue {
    fn validate(&self, field: &'static str) -> Result<(), NeuroSleepContractError> {
        if let Self::Observed { value, unit } = self {
            ensure_finite(*value, field)?;
            ensure_nonempty(unit, field)?;
        }
        Ok(())
    }
}

/// Quality facts for the complete night.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NightQuality {
    /// Fraction of the requested recording with valid synchronized samples.
    pub valid_coverage_fraction: f64,
    /// Fraction rejected by artifact rules.
    pub artifact_fraction: f64,
    /// Whether the frozen profile's nightly engineering gates passed.
    pub accepted: bool,
    /// Stable reason codes, empty on acceptance.
    pub reason_codes: Vec<String>,
}

/// Nightly stage duration and bout summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageSummary {
    /// Wake duration.
    pub wake_duration: FeatureValue,
    /// NREM duration.
    pub nrem_duration: FeatureValue,
    /// NREM mean bout duration.
    pub nrem_mean_bout_duration: FeatureValue,
    /// REM duration.
    pub rem_duration: FeatureValue,
    /// REM bout count.
    pub rem_bout_count: FeatureValue,
}

/// Stage-specific fixed qEEG registry. No arbitrary metric map is accepted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateQeegFeatures {
    /// State represented by these features.
    pub state: SleepState,
    /// Absolute delta power.
    pub delta_absolute_power: FeatureValue,
    /// Relative delta power.
    pub delta_relative_power: FeatureValue,
    /// Absolute theta power.
    pub theta_absolute_power: FeatureValue,
    /// Relative theta power.
    pub theta_relative_power: FeatureValue,
    /// Alpha power.
    pub alpha_absolute_power: FeatureValue,
    /// Periodic theta center frequency.
    pub theta_peak_frequency: FeatureValue,
    /// Periodic theta peak power.
    pub theta_peak_power: FeatureValue,
    /// Frontal-to-parietal full-band magnitude-squared coherence.
    pub frontal_parietal_full_band_coherence: FeatureValue,
    /// Frontal-to-parietal theta magnitude-squared coherence.
    pub frontal_parietal_theta_coherence: FeatureValue,
    /// Aperiodic exponent.
    pub aperiodic_exponent: FeatureValue,
    /// Aperiodic offset.
    pub aperiodic_offset: FeatureValue,
    /// Spectral fit error.
    pub spectral_fit_error: FeatureValue,
}

impl StateQeegFeatures {
    fn validate(&self) -> Result<(), NeuroSleepContractError> {
        self.delta_absolute_power
            .validate("qeeg.delta_absolute_power")?;
        self.delta_relative_power
            .validate("qeeg.delta_relative_power")?;
        self.theta_absolute_power
            .validate("qeeg.theta_absolute_power")?;
        self.theta_relative_power
            .validate("qeeg.theta_relative_power")?;
        self.alpha_absolute_power
            .validate("qeeg.alpha_absolute_power")?;
        self.theta_peak_frequency
            .validate("qeeg.theta_peak_frequency")?;
        self.theta_peak_power.validate("qeeg.theta_peak_power")?;
        self.frontal_parietal_full_band_coherence
            .validate("qeeg.frontal_parietal_full_band_coherence")?;
        self.frontal_parietal_theta_coherence
            .validate("qeeg.frontal_parietal_theta_coherence")?;
        self.aperiodic_exponent
            .validate("qeeg.aperiodic_exponent")?;
        self.aperiodic_offset.validate("qeeg.aperiodic_offset")?;
        self.spectral_fit_error.validate("qeeg.spectral_fit_error")
    }
}

/// Public literature metadata carried for research context only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResearchCitation {
    /// Stable citation identifier, such as DOI or PMID.
    pub identifier: String,
    /// Citation title.
    pub title: String,
    /// Evidence maturity, for example `preclinical_mouse_model`.
    pub evidence_maturity: String,
}

/// The complete payload covered by canonical hashing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeuroSleepPayloadV1 {
    pub bundle_id: String,
    pub species: Species,
    pub study_id: String,
    pub subject_pseudonym: String,
    pub recording_id: String,
    pub night_start_ms: i64,
    pub night_end_ms: i64,
    pub nonce: String,
    pub consent_scope: Vec<String>,
    pub source_artifact_sha256: String,
    pub source_format: SourceFormat,
    pub source_byte_count: u64,
    pub acquisition: AcquisitionMetadata,
    pub algorithm: AlgorithmManifest,
    pub quality: NightQuality,
    pub stage_summary: StageSummary,
    pub qeeg_by_stage: Vec<StateQeegFeatures>,
    pub compatibility_fingerprint: String,
    pub literature_context: Vec<ResearchCitation>,
}

impl NeuroSleepPayloadV1 {
    /// Validate all contract invariants that precede hashing and signing.
    pub fn validate(&self) -> Result<(), NeuroSleepContractError> {
        for (field, value) in [
            ("bundle_id", self.bundle_id.as_str()),
            ("study_id", self.study_id.as_str()),
            ("subject_pseudonym", self.subject_pseudonym.as_str()),
            ("recording_id", self.recording_id.as_str()),
            ("nonce", self.nonce.as_str()),
            (
                "source_artifact_sha256",
                self.source_artifact_sha256.as_str(),
            ),
            (
                "compatibility_fingerprint",
                self.compatibility_fingerprint.as_str(),
            ),
        ] {
            ensure_nonempty(value, field)?;
        }
        ensure_sha256(&self.source_artifact_sha256, "source_artifact_sha256")?;
        ensure_sha256(&self.compatibility_fingerprint, "compatibility_fingerprint")?;
        if self.night_end_ms <= self.night_start_ms {
            return Err(invalid("night_end_ms", "must be after night_start_ms"));
        }
        if self.source_byte_count == 0 {
            return Err(invalid("source_byte_count", "must be positive"));
        }
        if self.consent_scope.is_empty() {
            return Err(invalid("consent_scope", "must not be empty"));
        }
        for scope in &self.consent_scope {
            ensure_nonempty(scope, "consent_scope")?;
        }
        ensure_finite(
            self.acquisition.sampling_rate_hz,
            "acquisition.sampling_rate_hz",
        )?;
        if !(64.0..=4096.0).contains(&self.acquisition.sampling_rate_hz) {
            return Err(invalid(
                "acquisition.sampling_rate_hz",
                "must be between 64 and 4096 Hz",
            ));
        }
        ensure_nonempty(&self.acquisition.device_model, "acquisition.device_model")?;
        for (field, value) in [
            (
                "acquisition.hardware_version",
                self.acquisition.hardware_version.as_deref(),
            ),
            (
                "acquisition.firmware_version",
                self.acquisition.firmware_version.as_deref(),
            ),
        ] {
            if let Some(value) = value {
                ensure_nonempty(value, field)?;
            }
        }
        if self.acquisition.channels.is_empty() {
            return Err(invalid("acquisition.channels", "must not be empty"));
        }
        for channel in &self.acquisition.channels {
            ensure_nonempty(&channel.name, "acquisition.channels.name")?;
            ensure_nonempty(&channel.reference, "acquisition.channels.reference")?;
        }
        self.validate_algorithm()?;
        ensure_fraction(
            self.quality.valid_coverage_fraction,
            "quality.valid_coverage_fraction",
        )?;
        ensure_fraction(self.quality.artifact_fraction, "quality.artifact_fraction")?;
        for reason in &self.quality.reason_codes {
            ensure_nonempty(reason, "quality.reason_codes")?;
        }
        self.stage_summary
            .wake_duration
            .validate("stage_summary.wake_duration")?;
        self.stage_summary
            .nrem_duration
            .validate("stage_summary.nrem_duration")?;
        self.stage_summary
            .nrem_mean_bout_duration
            .validate("stage_summary.nrem_mean_bout_duration")?;
        self.stage_summary
            .rem_duration
            .validate("stage_summary.rem_duration")?;
        self.stage_summary
            .rem_bout_count
            .validate("stage_summary.rem_bout_count")?;
        if self.qeeg_by_stage.is_empty() {
            return Err(invalid("qeeg_by_stage", "must not be empty"));
        }
        let mut states = BTreeSet::new();
        for features in &self.qeeg_by_stage {
            if !states.insert(features.state as u8) {
                return Err(invalid("qeeg_by_stage", "contains a duplicate state"));
            }
            features.validate()?;
        }
        for citation in &self.literature_context {
            ensure_nonempty(&citation.identifier, "literature_context.identifier")?;
            ensure_nonempty(&citation.title, "literature_context.title")?;
            ensure_nonempty(
                &citation.evidence_maturity,
                "literature_context.evidence_maturity",
            )?;
        }
        Ok(())
    }

    fn validate_algorithm(&self) -> Result<(), NeuroSleepContractError> {
        for (field, value) in [
            (
                "algorithm.pipeline_commit",
                self.algorithm.pipeline_commit.as_str(),
            ),
            (
                "algorithm.extractor_sha256",
                self.algorithm.extractor_sha256.as_str(),
            ),
            (
                "algorithm.configuration_sha256",
                self.algorithm.configuration_sha256.as_str(),
            ),
        ] {
            ensure_nonempty(value, field)?;
        }
        ensure_sha256(
            &self.algorithm.extractor_sha256,
            "algorithm.extractor_sha256",
        )?;
        ensure_sha256(
            &self.algorithm.configuration_sha256,
            "algorithm.configuration_sha256",
        )?;
        if self.algorithm.crate_versions.is_empty() {
            return Err(invalid("algorithm.crate_versions", "must not be empty"));
        }
        for version in &self.algorithm.crate_versions {
            ensure_nonempty(version, "algorithm.crate_versions")?;
        }
        match &self.algorithm.stage_source {
            StageSource::ExpertHypnogram { scorer_type } => {
                ensure_nonempty(scorer_type, "algorithm.stage_source.scorer_type")?;
            }
            StageSource::Device { name, version } | StageSource::ProxyContext { name, version } => {
                ensure_nonempty(name, "algorithm.stage_source.name")?;
                ensure_nonempty(version, "algorithm.stage_source.version")?;
            }
            StageSource::Model {
                name,
                version,
                model_sha256,
            } => {
                ensure_nonempty(name, "algorithm.stage_source.name")?;
                ensure_nonempty(version, "algorithm.stage_source.version")?;
                ensure_sha256(model_sha256, "algorithm.stage_source.model_sha256")?;
            }
        }
        for parameter in &self.algorithm.dsp_parameters {
            ensure_nonempty(&parameter.name, "algorithm.dsp_parameters.name")?;
            ensure_nonempty(&parameter.value, "algorithm.dsp_parameters.value")?;
            ensure_nonempty(&parameter.unit, "algorithm.dsp_parameters.unit")?;
        }
        Ok(())
    }

    /// RFC 8785-compatible canonical JSON bytes.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, NeuroSleepContractError> {
        self.validate()?;
        serde_json_canonicalizer::to_vec(self)
            .map_err(|error| NeuroSleepContractError::CanonicalJson(error.to_string()))
    }

    /// SHA-256 of the canonical payload bytes.
    pub fn payload_sha256(&self) -> Result<[u8; 32], NeuroSleepContractError> {
        let bytes = self.canonical_bytes()?;
        Ok(Sha256::digest(bytes).into())
    }
}

fn invalid(field: &'static str, reason: &'static str) -> NeuroSleepContractError {
    NeuroSleepContractError::InvalidField { field, reason }
}

fn ensure_nonempty(value: &str, field: &'static str) -> Result<(), NeuroSleepContractError> {
    if value.is_empty() {
        return Err(invalid(field, "must not be empty"));
    }
    if value.contains('\0') {
        return Err(invalid(field, "must not contain a null byte"));
    }
    Ok(())
}

fn ensure_finite(value: f64, field: &'static str) -> Result<(), NeuroSleepContractError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(NeuroSleepContractError::NonFinite(field))
    }
}

fn ensure_fraction(value: f64, field: &'static str) -> Result<(), NeuroSleepContractError> {
    ensure_finite(value, field)?;
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(invalid(field, "must be between zero and one"))
    }
}

fn ensure_sha256(value: &str, field: &'static str) -> Result<(), NeuroSleepContractError> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(invalid(field, "must be a 64-character hexadecimal SHA-256"))
    }
}
