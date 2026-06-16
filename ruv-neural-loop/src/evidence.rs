//! The **Ruflo evidence bundle** (`ruflo-evidence/1`): the canonical, portable
//! artifact a closed-loop session produces for the web console (ADR-0014).
//!
//! Unlike the internal [`crate::audit::AuditTrail`] (whose chain hashes
//! `serde_json` of an event struct), the evidence bundle's per-step hash chain
//! is built from **fixed-precision canonical strings**. This lets the
//! TypeScript verifier in the web UI recompute byte-identical hashes without
//! depending on language-specific JSON or float formatting — verification is
//! genuinely reproducible in the browser, not merely asserted.

use ruv_neural_biosense::PhysioMetrics;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::controller::{ClosedLoopController, StepResult};
use crate::envelope::{BreachReason, EnvelopeStatus};
use crate::outcome::SessionReport;

/// Schema identifier embedded in every bundle.
pub const SCHEMA_VERSION: &str = "ruflo-evidence/1";

/// All-zero genesis hash for the bundle step chain.
const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// One stimulus delivery receipt, UI-shaped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceReceipt {
    pub modality: String,
    pub intended_frequency_hz: f64,
    pub measured_frequency_hz: f64,
    pub frequency_error_hz: f64,
    pub duty_cycle: f64,
    pub intensity: f64,
    pub waveform_sha256: String,
    pub verified: bool,
}

/// Biosense metrics for one step (fields optional per available channel).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceBiosense {
    pub heart_rate_bpm: Option<f64>,
    pub sdnn_ms: Option<f64>,
    pub rmssd_ms: Option<f64>,
    pub pnn50: Option<f64>,
    pub lf_hf_ratio: Option<f64>,
    pub respiration_rate_bpm: Option<f64>,
    pub motion_index: Option<f64>,
    pub stillness: Option<f64>,
    pub arousal_score: f64,
    pub relaxation_score: f64,
}

impl EvidenceBiosense {
    fn from_physio(p: &PhysioMetrics) -> Self {
        let hrv = p.hrv.as_ref();
        Self {
            heart_rate_bpm: hrv.map(|h| h.mean_hr_bpm),
            sdnn_ms: hrv.map(|h| h.sdnn_ms),
            rmssd_ms: hrv.map(|h| h.rmssd_ms),
            pnn50: hrv.map(|h| h.pnn50),
            lf_hf_ratio: hrv.map(|h| h.lf_hf_ratio),
            respiration_rate_bpm: p.respiration.as_ref().map(|r| r.rate_bpm),
            motion_index: p.motion.as_ref().map(|m| m.movement_index),
            stillness: p.motion.as_ref().map(|m| m.stillness()),
            arousal_score: p.arousal_index,
            relaxation_score: p.relaxation_index,
        }
    }
}

/// Safety-envelope status for one step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceEnvelope {
    pub within: bool,
    pub breaches: Vec<String>,
}

/// One control step, with its bundle-chain hashes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceStep {
    pub index: u64,
    pub timestamp_s: f64,
    pub phase: String,
    pub audit_kind: String,
    pub distance_to_target: f64,
    pub intensity: f64,
    pub embedding: Vec<f64>,
    pub feature_names: Vec<String>,
    pub biosense: EvidenceBiosense,
    pub receipts: Vec<EvidenceReceipt>,
    pub envelope: EvidenceEnvelope,
    /// SHA-256 of this step's canonical payload string.
    pub payload_sha256: String,
    /// Previous step's `hash` (genesis for the first step).
    pub prev_hash: String,
    /// `SHA-256(prev_hash || payload_sha256)`.
    pub hash: String,
}

/// The four acceptance clauses plus the platform verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcceptanceResult {
    pub target_state_identified: bool,
    pub verified_stimulus_delivered: bool,
    pub response_measured: bool,
    pub stopped_safely_outside_envelope: bool,
    pub goal_reached: bool,
    pub passed: bool,
}

/// Detached Ed25519 attestation over the bundle chain head.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SignatureBlock {
    pub head_hash: String,
    pub signature: String,
    pub public_key: String,
}

/// The complete evidence bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceBundle {
    pub schema_version: String,
    pub session_id: String,
    pub created_at: String,
    pub mode: String,
    pub target_state: String,
    pub protocol: String,
    pub steps: Vec<EvidenceStep>,
    pub acceptance: AcceptanceResult,
    pub report: SessionReport,
    pub bundle_chain_head: String,
    pub audit_head_hash: String,
    pub audit_records: u64,
    pub audit_chain_valid: bool,
    pub signature: Option<SignatureBlock>,
}

impl EvidenceBundle {
    /// Build a bundle from a finished session.
    pub fn build(
        target_state: &str,
        mode: &str,
        trace: &[StepResult],
        controller: &ClosedLoopController,
    ) -> Self {
        let report = controller.report();

        let mut steps = Vec::with_capacity(trace.len());
        let mut prev_hash = GENESIS.to_string();
        for s in trace {
            let receipts: Vec<EvidenceReceipt> = s
                .emitted
                .iter()
                .map(|v| {
                    let p = &v.waveform.params;
                    EvidenceReceipt {
                        modality: p.modality.tag().to_string(),
                        intended_frequency_hz: p.envelope_hz,
                        measured_frequency_hz: v.receipt.measured_envelope_hz,
                        frequency_error_hz: (v.receipt.measured_envelope_hz - p.envelope_hz).abs(),
                        duty_cycle: p.duty_cycle,
                        intensity: p.intensity,
                        waveform_sha256: v.receipt.waveform_sha256.clone(),
                        verified: v.receipt.verified,
                    }
                })
                .collect();

            let payload = step_payload(
                s.index,
                s.timestamp_s,
                phase_tag(&s.phase),
                s.estimate.distance_to_target,
                s.plan.intensity,
                &receipts,
            );
            let payload_sha256 = sha256_hex(payload.as_bytes());
            let hash = sha256_hex(format!("{prev_hash}{payload_sha256}").as_bytes());

            steps.push(EvidenceStep {
                index: s.index,
                timestamp_s: s.timestamp_s,
                phase: phase_tag(&s.phase).to_string(),
                audit_kind: s.audit_kind.tag().to_string(),
                distance_to_target: s.estimate.distance_to_target,
                intensity: s.plan.intensity,
                embedding: s.embedding.features.to_vec(),
                feature_names: crate::embedding::FEATURE_NAMES
                    .iter()
                    .map(|f| f.to_string())
                    .collect(),
                biosense: EvidenceBiosense::from_physio(&s.physio),
                receipts,
                envelope: EvidenceEnvelope {
                    within: !s.envelope.is_breach(),
                    breaches: breach_strings(&s.envelope),
                },
                payload_sha256,
                prev_hash: prev_hash.clone(),
                hash: hash.clone(),
            });
            prev_hash = hash;
        }

        let bundle_chain_head = prev_hash;
        let acceptance = AcceptanceResult {
            target_state_identified: true,
            verified_stimulus_delivered: report.num_receipts >= 1 && report.all_receipts_verified,
            response_measured: report.total_steps >= 1,
            stopped_safely_outside_envelope: report.safe_stopped,
            goal_reached: report.goal_reached,
            passed: report.passes_acceptance(),
        };

        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            session_id: bundle_chain_head.chars().take(16).collect(),
            created_at: epoch_timestamp(),
            mode: mode.to_string(),
            target_state: target_state.to_string(),
            protocol: report.protocol.clone(),
            steps,
            acceptance,
            audit_head_hash: report.audit_head_hash.clone(),
            audit_records: report.audit_records,
            audit_chain_valid: report.audit_chain_valid,
            report,
            bundle_chain_head,
            signature: None,
        }
    }

    /// Attach an Ed25519 signature over the bundle chain head.
    pub fn signed(mut self) -> Self {
        use ed25519_dalek::{Signer, SigningKey};
        use rand::rngs::OsRng;
        let key = SigningKey::generate(&mut OsRng);
        let sig = key.sign(self.bundle_chain_head.as_bytes());
        self.signature = Some(SignatureBlock {
            head_hash: self.bundle_chain_head.clone(),
            signature: bytes_hex(sig.to_bytes().as_slice()),
            public_key: bytes_hex(key.verifying_key().to_bytes().as_slice()),
        });
        self
    }

    /// Recompute and verify the entire bundle step chain (the same check the
    /// browser performs).
    pub fn verify_chain(&self) -> bool {
        let mut prev = GENESIS.to_string();
        for s in &self.steps {
            if s.prev_hash != prev {
                return false;
            }
            let receipts_hashes: Vec<String> = s
                .receipts
                .iter()
                .map(|r| r.waveform_sha256.clone())
                .collect();
            let payload = canonical_payload(
                s.index,
                s.timestamp_s,
                &s.phase,
                s.distance_to_target,
                s.intensity,
                &receipts_hashes,
            );
            if s.payload_sha256 != sha256_hex(payload.as_bytes()) {
                return false;
            }
            let expect = sha256_hex(format!("{prev}{}", s.payload_sha256).as_bytes());
            if s.hash != expect {
                return false;
            }
            prev = s.hash.clone();
        }
        prev == self.bundle_chain_head
    }

    /// Verify the Ed25519 signature (if present) over the bundle chain head.
    /// Returns `None` when unsigned, `Some(true/false)` otherwise.
    pub fn verify_signature(&self) -> Option<bool> {
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};
        let sig = self.signature.as_ref()?;
        let ok = (|| -> Option<bool> {
            if sig.head_hash != self.bundle_chain_head {
                return Some(false);
            }
            let pk: [u8; 32] = unhex(&sig.public_key)?.try_into().ok()?;
            let sg: [u8; 64] = unhex(&sig.signature)?.try_into().ok()?;
            let vk = VerifyingKey::from_bytes(&pk).ok()?;
            Some(
                vk.verify(sig.head_hash.as_bytes(), &Signature::from_bytes(&sg))
                    .is_ok(),
            )
        })()
        .unwrap_or(false);
        Some(ok)
    }
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

// ── canonical hashing (must mirror the TypeScript verifier exactly) ─────────

fn step_payload(
    index: u64,
    timestamp_s: f64,
    phase: &str,
    distance: f64,
    intensity: f64,
    receipts: &[EvidenceReceipt],
) -> String {
    let hashes: Vec<String> = receipts.iter().map(|r| r.waveform_sha256.clone()).collect();
    canonical_payload(index, timestamp_s, phase, distance, intensity, &hashes)
}

/// The canonical per-step string. Fixed precision keeps Rust and TS identical.
fn canonical_payload(
    index: u64,
    timestamp_s: f64,
    phase: &str,
    distance: f64,
    intensity: f64,
    receipt_hashes: &[String],
) -> String {
    format!(
        "{}|{:.3}|{}|{:.6}|{:.6}|{}",
        index,
        timestamp_s,
        phase,
        distance,
        intensity,
        receipt_hashes.join(",")
    )
}

fn phase_tag(p: &crate::controller::ControllerPhase) -> &'static str {
    use crate::controller::ControllerPhase::*;
    match p {
        Baselining => "Baselining",
        Stimulating => "Stimulating",
        Holding => "Holding",
        Completed => "Completed",
        SafeStopped => "SafeStopped",
        Aborted => "Aborted",
    }
}

fn breach_strings(env: &EnvelopeStatus) -> Vec<String> {
    env.reasons().iter().map(breach_tag).collect()
}

fn breach_tag(r: &BreachReason) -> String {
    match r {
        BreachReason::HeartRateHigh { bpm, max } => {
            format!("HeartRateHigh({bpm:.1}>{max:.1})")
        }
        BreachReason::HeartRateLow { bpm, min } => format!("HeartRateLow({bpm:.1}<{min:.1})"),
        BreachReason::ArousalHigh { value, max } => format!("ArousalHigh({value:.2}>{max:.2})"),
        BreachReason::ExcessiveMotion {
            movement_index,
            max,
        } => {
            format!("ExcessiveMotion({movement_index:.3}>{max:.3})")
        }
        BreachReason::SleepInhibited => "SleepInhibited".to_string(),
        BreachReason::ResponseDiverging { delta, tolerance } => {
            format!("ResponseDiverging({delta:.3}>{tolerance:.3})")
        }
        BreachReason::MissingData(s) => format!("MissingData({s})"),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    bytes_hex(h.finalize().as_slice())
}

fn bytes_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn epoch_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("epoch:{secs}")
}
