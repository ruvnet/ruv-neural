//! Persistent-key attestation for NeuroSleep evidence.
//!
//! Unlike legacy witness and Ruflo evidence artifacts, a NeuroSleep bundle does
//! not embed a public key and never generates a key during signing. The caller
//! injects a persistent signer; verifiers obtain the key from an enrolled trust
//! profile.

use crate::neurosleep::{NeuroSleepContractError, NeuroSleepPayloadV1, NEUROSLEEP_SCHEMA_V1};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const SIGNATURE_SEPARATOR: u8 = 0;

/// A fully bound NeuroSleep nightly feature bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedNeuroSleepBundleV1 {
    /// Exact schema/domain identifier.
    pub schema: String,
    /// Canonically hashed evidence payload.
    pub payload: NeuroSleepPayloadV1,
    /// SHA-256 of RFC 8785 canonical payload JSON.
    pub payload_sha256: [u8; 32],
    /// Identifier resolved through an external trust store.
    pub signer_key_id: String,
    /// Detached Ed25519 signature.
    #[serde(with = "signature_bytes")]
    pub signature_ed25519: [u8; 64],
}

/// Signer interface implemented by persistent device or study key custody.
pub trait NeuroSleepSigner {
    /// Stable enrolled key identifier. It is bound into the signature.
    fn key_id(&self) -> &str;
    /// Sign the already domain-separated message.
    fn sign(&self, message: &[u8]) -> Result<[u8; 64], NeuroSleepAttestationError>;
}

/// Adapter for callers that already loaded a persistent Ed25519 key.
pub struct PersistentEd25519Signer {
    key_id: String,
    key: SigningKey,
}

impl PersistentEd25519Signer {
    /// Construct from persistent key bytes. No randomness or key generation is
    /// performed by this API.
    pub fn from_bytes(
        key_id: impl Into<String>,
        secret_key: &[u8; 32],
    ) -> Result<Self, NeuroSleepAttestationError> {
        let key_id = key_id.into();
        validate_key_id(&key_id)?;
        Ok(Self {
            key_id,
            key: SigningKey::from_bytes(secret_key),
        })
    }

    /// Public key bytes for explicit enrollment into a downstream trust store.
    pub fn verifying_key_bytes(&self) -> [u8; 32] {
        self.key.verifying_key().to_bytes()
    }
}

impl NeuroSleepSigner for PersistentEd25519Signer {
    fn key_id(&self) -> &str {
        &self.key_id
    }

    fn sign(&self, message: &[u8]) -> Result<[u8; 64], NeuroSleepAttestationError> {
        Ok(self.key.sign(message).to_bytes())
    }
}

/// Fail-closed attestation errors.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NeuroSleepAttestationError {
    /// Payload or canonical JSON validation failed.
    #[error(transparent)]
    Contract(#[from] NeuroSleepContractError),
    /// Signer identifier is malformed.
    #[error("invalid signer key identifier: {0}")]
    InvalidKeyId(&'static str),
    /// Injected signer failed.
    #[error("persistent signer failed: {0}")]
    Signer(String),
    /// Trusted Ed25519 public key bytes were invalid.
    #[error("invalid trusted Ed25519 public key")]
    InvalidVerifyingKey,
    /// Detached signature verification failed.
    #[error("NeuroSleep signature verification failed")]
    SignatureInvalid,
}

/// Create a bundle using the injected persistent signer.
pub fn sign_neurosleep_bundle(
    payload: NeuroSleepPayloadV1,
    signer: &impl NeuroSleepSigner,
) -> Result<SignedNeuroSleepBundleV1, NeuroSleepAttestationError> {
    validate_key_id(signer.key_id())?;
    let payload_sha256 = payload.payload_sha256()?;
    let message = signature_message(signer.key_id(), &payload_sha256)?;
    let signature = signer.sign(&message)?;
    Ok(SignedNeuroSleepBundleV1 {
        schema: NEUROSLEEP_SCHEMA_V1.to_string(),
        payload,
        payload_sha256,
        signer_key_id: signer.key_id().to_string(),
        signature_ed25519: signature,
    })
}

/// Independently recompute the digest and verify with a key obtained from a
/// trusted enrollment profile. Bundle-supplied key material is never used.
pub fn verify_neurosleep_bundle(
    bundle: &SignedNeuroSleepBundleV1,
    trusted_verifying_key: &[u8; 32],
) -> Result<(), NeuroSleepAttestationError> {
    if bundle.schema != NEUROSLEEP_SCHEMA_V1 {
        return Err(NeuroSleepContractError::UnsupportedSchema(bundle.schema.clone()).into());
    }
    validate_key_id(&bundle.signer_key_id)?;
    let recomputed = bundle.payload.payload_sha256()?;
    if recomputed != bundle.payload_sha256 {
        return Err(NeuroSleepContractError::DigestMismatch.into());
    }
    let key = VerifyingKey::from_bytes(trusted_verifying_key)
        .map_err(|_| NeuroSleepAttestationError::InvalidVerifyingKey)?;
    let message = signature_message(&bundle.signer_key_id, &bundle.payload_sha256)?;
    key.verify(&message, &Signature::from_bytes(&bundle.signature_ed25519))
        .map_err(|_| NeuroSleepAttestationError::SignatureInvalid)
}

/// Domain-separated bytes signed by every NeuroSleep V1 signer.
pub fn signature_message(
    signer_key_id: &str,
    payload_sha256: &[u8; 32],
) -> Result<Vec<u8>, NeuroSleepAttestationError> {
    validate_key_id(signer_key_id)?;
    let mut message = Vec::with_capacity(
        NEUROSLEEP_SCHEMA_V1.len() + signer_key_id.len() + payload_sha256.len() + 2,
    );
    message.extend_from_slice(NEUROSLEEP_SCHEMA_V1.as_bytes());
    message.push(SIGNATURE_SEPARATOR);
    message.extend_from_slice(signer_key_id.as_bytes());
    message.push(SIGNATURE_SEPARATOR);
    message.extend_from_slice(payload_sha256);
    Ok(message)
}

fn validate_key_id(key_id: &str) -> Result<(), NeuroSleepAttestationError> {
    if key_id.is_empty() {
        return Err(NeuroSleepAttestationError::InvalidKeyId(
            "must not be empty",
        ));
    }
    if key_id.contains('\0') {
        return Err(NeuroSleepAttestationError::InvalidKeyId(
            "must not contain a null byte",
        ));
    }
    if key_id.len() > 256 {
        return Err(NeuroSleepAttestationError::InvalidKeyId(
            "must not exceed 256 bytes",
        ));
    }
    Ok(())
}

mod signature_bytes {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S>(bytes: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        bytes.as_slice().serialize(serializer)
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 64], D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        bytes
            .try_into()
            .map_err(|_| serde::de::Error::custom("Ed25519 signature must be exactly 64 bytes"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neurosleep::{
        compatibility_fingerprint_v1, AcquisitionChannel, AcquisitionMetadata, AcquisitionModality,
        AlgorithmManifest, AlgorithmParameter, FeatureValue, NightQuality, NullReason,
        ResearchCitation, SourceFormat, Species, StageSource, StageSummary, StateQeegFeatures,
    };

    type PayloadMutation = Box<dyn Fn(&mut NeuroSleepPayloadV1)>;

    fn observed(value: f64, unit: &str) -> FeatureValue {
        FeatureValue::Observed {
            value,
            unit: unit.to_string(),
        }
    }

    fn payload() -> NeuroSleepPayloadV1 {
        let mut payload = NeuroSleepPayloadV1 {
            bundle_id: "bundle-001".into(),
            species: Species::Mouse,
            study_id: "constantino-method-fixture".into(),
            subject_pseudonym: "subject-random-001".into(),
            recording_id: "recording-001".into(),
            night_start_ms: 1_700_000_000_000,
            night_end_ms: 1_700_028_800_000,
            nonce: "nonce-001".into(),
            consent_scope: vec!["local_neurosleep_research_v1".into()],
            source_artifact_sha256: "11".repeat(32),
            source_format: SourceFormat::EdfPlus,
            source_byte_count: 4_096,
            acquisition: AcquisitionMetadata {
                device_model: "fixture-recorder".into(),
                hardware_version: Some("1".into()),
                firmware_version: Some("1.0.0".into()),
                sampling_rate_hz: 250.0,
                channels: vec![
                    AcquisitionChannel {
                        name: "frontal".into(),
                        modality: AcquisitionModality::Eeg,
                        reference: "cerebellar".into(),
                    },
                    AcquisitionChannel {
                        name: "parietal".into(),
                        modality: AcquisitionModality::Eeg,
                        reference: "cerebellar".into(),
                    },
                ],
            },
            algorithm: AlgorithmManifest {
                pipeline_commit: "caaa14144a70829293737b0ca717ebc818fcc523".into(),
                crate_versions: vec!["ruv-neural-core@0.1.0".into()],
                extractor_sha256: "22".repeat(32),
                configuration_sha256: "33".repeat(32),
                stage_source: StageSource::ExpertHypnogram {
                    scorer_type: "human_scored_10_second".into(),
                },
                dsp_parameters: vec![AlgorithmParameter {
                    name: "epoch_duration".into(),
                    value: "10".into(),
                    unit: "s".into(),
                }],
            },
            quality: NightQuality {
                valid_coverage_fraction: 0.95,
                artifact_fraction: 0.05,
                accepted: true,
                reason_codes: Vec::new(),
            },
            stage_summary: StageSummary {
                wake_duration: observed(7_200.0, "s"),
                nrem_duration: observed(18_000.0, "s"),
                nrem_mean_bout_duration: observed(900.0, "s"),
                rem_duration: observed(3_600.0, "s"),
                rem_bout_count: observed(5.0, "count"),
            },
            qeeg_by_stage: vec![StateQeegFeatures {
                state: crate::neurosleep::SleepState::Nrem,
                delta_absolute_power: observed(2.5, "uV2"),
                delta_relative_power: observed(0.4, "ratio"),
                theta_absolute_power: observed(1.5, "uV2"),
                theta_relative_power: observed(0.25, "ratio"),
                alpha_absolute_power: observed(0.8, "uV2"),
                theta_peak_frequency: observed(6.2, "Hz"),
                theta_peak_power: observed(0.3, "log10_uV2_per_hz"),
                frontal_parietal_full_band_coherence: observed(0.6, "ratio"),
                frontal_parietal_theta_coherence: observed(0.7, "ratio"),
                aperiodic_exponent: FeatureValue::Null {
                    reason: NullReason::NotApplicable,
                },
                aperiodic_offset: FeatureValue::Null {
                    reason: NullReason::NotApplicable,
                },
                spectral_fit_error: FeatureValue::Null {
                    reason: NullReason::NotApplicable,
                },
            }],
            compatibility_fingerprint: String::new(),
            literature_context: vec![ResearchCitation {
                identifier: "PMID:42252510".into(),
                title: "Mouse-model NeuroSleep fixture citation".into(),
                evidence_maturity: "preclinical_mouse_model".into(),
            }],
        };
        payload.compatibility_fingerprint =
            compatibility_fingerprint_v1(&payload.acquisition, &payload.algorithm).unwrap();
        payload
    }

    fn signer() -> PersistentEd25519Signer {
        PersistentEd25519Signer::from_bytes("study-key-2026-01", &[7; 32]).unwrap()
    }

    #[test]
    fn canonical_payload_has_stable_rfc8785_golden_digest() {
        let digest = payload().payload_sha256().unwrap();
        assert_eq!(
            hex(&digest),
            "171eaaaa654b5a16de4605d603aa5d7c97db6784c624354b6ee97ca2ac9b83b7",
            "intentional golden: update only with a reviewed schema change"
        );
    }

    #[test]
    fn persistent_signer_roundtrip_and_domain_are_exact() {
        let signer = signer();
        let bundle = sign_neurosleep_bundle(payload(), &signer).unwrap();
        verify_neurosleep_bundle(&bundle, &signer.verifying_key_bytes()).unwrap();

        let fixture: SignedNeuroSleepBundleV1 = serde_json::from_str(include_str!(
            "../tests/fixtures/neurosleep-v1/valid_bundle.json"
        ))
        .unwrap();
        assert_eq!(bundle, fixture);
        let trust: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/neurosleep-v1/trust_profile.json"
        ))
        .unwrap();
        let fixture_key: Vec<u8> = trust["verifying_key_ed25519"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u8)
            .collect();
        assert_eq!(fixture_key, signer.verifying_key_bytes());

        let message = signature_message(&bundle.signer_key_id, &bundle.payload_sha256).unwrap();
        let mut expected = b"ruv-neural/neurosleep/1\0study-key-2026-01\0".to_vec();
        expected.extend_from_slice(&bundle.payload_sha256);
        assert_eq!(message, expected);
    }

    #[test]
    fn tampering_and_wrong_trust_key_fail_closed() {
        let signer = signer();
        let mut bundle = sign_neurosleep_bundle(payload(), &signer).unwrap();
        bundle.payload.qeeg_by_stage[0].theta_relative_power = observed(0.26, "ratio");
        assert!(matches!(
            verify_neurosleep_bundle(&bundle, &signer.verifying_key_bytes()),
            Err(NeuroSleepAttestationError::Contract(
                NeuroSleepContractError::DigestMismatch
            ))
        ));

        let clean = sign_neurosleep_bundle(payload(), &signer).unwrap();
        let other = PersistentEd25519Signer::from_bytes("other", &[8; 32]).unwrap();
        assert_eq!(
            verify_neurosleep_bundle(&clean, &other.verifying_key_bytes()),
            Err(NeuroSleepAttestationError::SignatureInvalid)
        );
    }

    #[test]
    fn validation_rejects_nonfinite_unknown_and_null_key_id() {
        let mut invalid = payload();
        invalid.quality.artifact_fraction = f64::NAN;
        assert_eq!(
            invalid.validate(),
            Err(NeuroSleepContractError::NonFinite(
                "quality.artifact_fraction"
            ))
        );
        assert!(matches!(
            PersistentEd25519Signer::from_bytes("bad\0key", &[7; 32]),
            Err(NeuroSleepAttestationError::InvalidKeyId(_))
        ));

        let mut json = serde_json::to_value(payload()).unwrap();
        json.as_object_mut()
            .unwrap()
            .insert("unknown_metric".into(), serde_json::json!(1));
        assert!(serde_json::from_value::<NeuroSleepPayloadV1>(json).is_err());

        let mut unknown_modality = serde_json::to_value(payload()).unwrap();
        unknown_modality["acquisition"]["channels"][0]["modality"] =
            serde_json::json!("ambient_sensing");
        assert!(serde_json::from_value::<NeuroSleepPayloadV1>(unknown_modality).is_err());

        let signed = sign_neurosleep_bundle(payload(), &signer()).unwrap();
        let mut signed_json = serde_json::to_value(signed).unwrap();
        signed_json["signature_ed25519"] = serde_json::json!([1, 2, 3]);
        assert!(serde_json::from_value::<SignedNeuroSleepBundleV1>(signed_json).is_err());
    }

    #[test]
    fn every_top_level_payload_field_is_digest_bound() {
        let original = payload();
        let original_digest = original.payload_sha256().unwrap();
        let mutations: Vec<PayloadMutation> = vec![
            Box::new(|p| p.bundle_id.push('x')),
            Box::new(|p| p.species = Species::Human),
            Box::new(|p| p.study_id.push('x')),
            Box::new(|p| p.subject_pseudonym.push('x')),
            Box::new(|p| p.recording_id.push('x')),
            Box::new(|p| p.night_start_ms += 1),
            Box::new(|p| p.night_end_ms += 1),
            Box::new(|p| p.nonce.push('x')),
            Box::new(|p| p.consent_scope.push("export".into())),
            Box::new(|p| p.source_artifact_sha256 = "55".repeat(32)),
            Box::new(|p| p.source_format = SourceFormat::Edf),
            Box::new(|p| p.source_byte_count += 1),
            Box::new(|p| {
                p.acquisition.device_model.push('x');
                p.compatibility_fingerprint =
                    compatibility_fingerprint_v1(&p.acquisition, &p.algorithm).unwrap();
            }),
            Box::new(|p| {
                p.algorithm.pipeline_commit.push('x');
                p.compatibility_fingerprint =
                    compatibility_fingerprint_v1(&p.acquisition, &p.algorithm).unwrap();
            }),
            Box::new(|p| p.quality.artifact_fraction = 0.06),
            Box::new(|p| p.stage_summary.nrem_duration = observed(18_001.0, "s")),
            Box::new(|p| p.qeeg_by_stage[0].theta_relative_power = observed(0.26, "ratio")),
            Box::new(|p| p.literature_context[0].identifier.push('x')),
        ];
        for mutate in mutations {
            let mut changed = original.clone();
            mutate(&mut changed);
            assert_ne!(changed.payload_sha256().unwrap(), original_digest);
        }
    }

    #[test]
    fn rejects_inconsistent_fingerprint_and_wrong_registry_units() {
        let mut inconsistent = payload();
        inconsistent.compatibility_fingerprint = "66".repeat(32);
        assert!(inconsistent.validate().is_err());

        let mutations: Vec<PayloadMutation> = vec![
            Box::new(|p| p.stage_summary.wake_duration = observed(1.0, "ms")),
            Box::new(|p| p.stage_summary.rem_bout_count = observed(1.0, "ratio")),
            Box::new(|p| p.qeeg_by_stage[0].theta_peak_frequency = observed(6.0, "s")),
            Box::new(|p| p.qeeg_by_stage[0].theta_peak_power = observed(1.0, "uV2")),
            Box::new(|p| p.qeeg_by_stage[0].frontal_parietal_theta_coherence = observed(0.5, "Hz")),
            Box::new(|p| p.qeeg_by_stage[0].aperiodic_exponent = observed(1.0, "ratio")),
            Box::new(|p| p.qeeg_by_stage[0].aperiodic_offset = observed(1.0, "uV2")),
            Box::new(|p| p.qeeg_by_stage[0].spectral_fit_error = observed(0.1, "ratio")),
        ];
        for mutate in mutations {
            let mut changed = payload();
            mutate(&mut changed);
            assert!(changed.validate().is_err());
        }
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}
