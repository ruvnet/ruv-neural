//! Signed evidence bundle generation.
//!
//! This module never generates key material. The caller injects a persistent
//! signer whose key was enrolled out of band, and verifiers resolve the public
//! key from a trust profile rather than from the bundle.

use ruv_neural_core::attestation::{
    sign_neurosleep_bundle, NeuroSleepSigner, SignedNeuroSleepBundleV1,
};
use ruv_neural_core::neurosleep::{
    compatibility_fingerprint_v1, AcquisitionMetadata, AlgorithmManifest, NeuroSleepPayloadV1,
    ResearchCitation, SourceFormat, Species,
};

use crate::profile::NeuroSleepProfile;
use crate::{analyze_night, ExpertEpoch, NeuroSleepError};

/// Provenance the analysis cannot derive from the signal itself.
#[derive(Debug, Clone, PartialEq)]
pub struct NightBundleContext {
    /// Bundle identifier.
    pub bundle_id: String,
    /// Subject species.
    pub species: Species,
    /// Study identifier.
    pub study_id: String,
    /// Pseudonymous subject identifier. Never a direct identifier.
    pub subject_pseudonym: String,
    /// Recording identifier.
    pub recording_id: String,
    /// Recording window start, in milliseconds since the Unix epoch.
    pub night_start_ms: i64,
    /// Recording window end, in milliseconds since the Unix epoch.
    pub night_end_ms: i64,
    /// Replay-binding nonce.
    pub nonce: String,
    /// Consent scopes this derivation is covered by.
    pub consent_scope: Vec<String>,
    /// SHA-256 of the source recording artefact.
    pub source_artifact_sha256: String,
    /// Source container format.
    pub source_format: SourceFormat,
    /// Source artefact size in bytes.
    pub source_byte_count: u64,
    /// Recorder and channel montage description.
    pub acquisition: AcquisitionMetadata,
    /// Pipeline identity and parameters.
    pub algorithm: AlgorithmManifest,
    /// Literature the profile is derived from.
    pub literature_context: Vec<ResearchCitation>,
}

/// Analyse a night and bind the result into a signed V1 bundle.
///
/// The compatibility fingerprint is recomputed here from the acquisition
/// metadata and algorithm manifest actually carried by the payload, so a bundle
/// can never claim method compatibility it does not have. The payload is
/// validated by the contract before signing, so an internally inconsistent
/// bundle is never produced.
pub fn analyze_and_sign(
    context: NightBundleContext,
    epochs: &[ExpertEpoch],
    profile: &NeuroSleepProfile,
    signer: &impl NeuroSleepSigner,
) -> Result<SignedNeuroSleepBundleV1, NeuroSleepError> {
    let analysis = analyze_night(epochs, profile)?;
    let compatibility_fingerprint =
        compatibility_fingerprint_v1(&context.acquisition, &context.algorithm)?;
    let payload = NeuroSleepPayloadV1 {
        bundle_id: context.bundle_id,
        species: context.species,
        study_id: context.study_id,
        subject_pseudonym: context.subject_pseudonym,
        recording_id: context.recording_id,
        night_start_ms: context.night_start_ms,
        night_end_ms: context.night_end_ms,
        nonce: context.nonce,
        consent_scope: context.consent_scope,
        source_artifact_sha256: context.source_artifact_sha256,
        source_format: context.source_format,
        source_byte_count: context.source_byte_count,
        acquisition: context.acquisition,
        algorithm: context.algorithm,
        quality: analysis.quality,
        stage_summary: analysis.stage_summary,
        qeeg_by_stage: analysis.qeeg_by_stage,
        compatibility_fingerprint,
        literature_context: context.literature_context,
    };
    Ok(sign_neurosleep_bundle(payload, signer)?)
}
