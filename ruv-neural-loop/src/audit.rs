//! Tamper-evident audit trail for every closed-loop decision.
//!
//! Each decision the controller makes is appended as an [`AuditRecord`] whose
//! hash chains to the previous record (à la a hash list / lightweight
//! blockchain). Any post-hoc edit to an earlier record breaks every subsequent
//! hash, so the trail is verifiably append-only. The chain head can optionally
//! be Ed25519-signed to attest the whole session, mirroring the workspace's
//! existing witness mechanism.
//!
//! See `docs/adr/0009-audit-trail.md`.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::envelope::BreachReason;

/// The kind of decision recorded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuditKind {
    /// Session opened with a stated target.
    SessionStart,
    /// A baseline-collection step (no stimulation).
    Baseline,
    /// Active stimulation was delivered.
    Stimulate,
    /// A hold/rest step (target reached or settling, no escalation).
    Hold,
    /// The target state was reached and the session completed normally.
    Complete,
    /// A fail-safe stop: stimulation forced to zero due to envelope breach.
    SafeStop,
    /// The session was aborted (e.g. step budget exhausted).
    Abort,
}

impl AuditKind {
    /// Stable tag string.
    pub fn tag(&self) -> &'static str {
        match self {
            AuditKind::SessionStart => "session_start",
            AuditKind::Baseline => "baseline",
            AuditKind::Stimulate => "stimulate",
            AuditKind::Hold => "hold",
            AuditKind::Complete => "complete",
            AuditKind::SafeStop => "safe_stop",
            AuditKind::Abort => "abort",
        }
    }
}

/// The payload of a single audit event (the data that gets hashed).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Monotonic event index within the session.
    pub index: u64,
    /// Session-relative timestamp (s).
    pub timestamp_s: f64,
    /// Decision kind.
    pub kind: AuditKind,
    /// Human-readable message.
    pub message: String,
    /// Commanded intensity for this step.
    pub intensity: f64,
    /// Distance-to-target estimate at this step.
    pub distance_to_target: f64,
    /// SHA-256 digests of any stimulus delivery receipts emitted this step.
    pub receipt_hashes: Vec<String>,
    /// Envelope breach reasons, if this was a safe-stop.
    pub breaches: Vec<BreachReason>,
}

/// One chained record: an event plus the hashes binding it to the chain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditRecord {
    /// The event payload.
    pub event: AuditEvent,
    /// Hash of the previous record (`"0"*64` for the genesis record).
    pub prev_hash: String,
    /// Hash of this record = SHA-256(prev_hash || serialized(event)).
    pub hash: String,
}

/// The all-zero genesis predecessor hash.
pub const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// An append-only, hash-chained audit trail.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AuditTrail {
    /// The records, in order.
    pub records: Vec<AuditRecord>,
}

impl AuditTrail {
    /// A new, empty trail.
    pub fn new() -> Self {
        Self { records: Vec::new() }
    }

    /// Number of records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the trail is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The current head hash (or genesis if empty).
    pub fn head_hash(&self) -> String {
        self.records
            .last()
            .map(|r| r.hash.clone())
            .unwrap_or_else(|| GENESIS_HASH.to_string())
    }

    /// Append an event, chaining it to the current head.
    pub fn append(&mut self, event: AuditEvent) -> &AuditRecord {
        let prev_hash = self.head_hash();
        let hash = record_hash(&prev_hash, &event);
        self.records.push(AuditRecord { event, prev_hash, hash });
        self.records.last().unwrap()
    }

    /// Verify the integrity of the entire chain: every link recomputes and the
    /// `prev_hash` references line up.
    pub fn verify_chain(&self) -> bool {
        let mut expected_prev = GENESIS_HASH.to_string();
        for r in &self.records {
            if r.prev_hash != expected_prev {
                return false;
            }
            if r.hash != record_hash(&r.prev_hash, &r.event) {
                return false;
            }
            expected_prev = r.hash.clone();
        }
        true
    }

    /// Count records of a given kind.
    pub fn count_kind(&self, kind: &AuditKind) -> usize {
        self.records.iter().filter(|r| &r.event.kind == kind).count()
    }

    /// Sign the chain head with a freshly generated Ed25519 key, returning a
    /// detached, independently verifiable attestation of the whole session.
    pub fn sign_head(&self) -> SignedAuditHead {
        use ed25519_dalek::{Signer, SigningKey};
        use rand::rngs::OsRng;

        let head = self.head_hash();
        let signing_key = SigningKey::generate(&mut OsRng);
        let signature = signing_key.sign(head.as_bytes());
        SignedAuditHead {
            head_hash: head,
            num_records: self.records.len() as u64,
            signature: hex(signature.to_bytes().as_slice()),
            public_key: hex(signing_key.verifying_key().to_bytes().as_slice()),
        }
    }
}

/// A detached Ed25519 attestation over an audit-trail head hash.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignedAuditHead {
    /// The attested head hash.
    pub head_hash: String,
    /// Number of records covered.
    pub num_records: u64,
    /// Hex-encoded Ed25519 signature.
    pub signature: String,
    /// Hex-encoded Ed25519 public key.
    pub public_key: String,
}

impl SignedAuditHead {
    /// Verify the signature over the head hash.
    pub fn verify(&self) -> bool {
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};

        let (Ok(pk), Ok(sig)) = (unhex(&self.public_key), unhex(&self.signature)) else {
            return false;
        };
        let (Ok(pk), Ok(sig)): (Result<[u8; 32], _>, Result<[u8; 64], _>) =
            (pk.try_into(), sig.try_into())
        else {
            return false;
        };
        let Ok(vk) = VerifyingKey::from_bytes(&pk) else {
            return false;
        };
        vk.verify(self.head_hash.as_bytes(), &Signature::from_bytes(&sig))
            .is_ok()
    }
}

/// SHA-256(prev_hash || serialized(event)) as lowercase hex.
fn record_hash(prev_hash: &str, event: &AuditEvent) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prev_hash.as_bytes());
    hasher.update(serde_json::to_string(event).unwrap_or_default().as_bytes());
    hex(hasher.finalize().as_slice())
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn unhex(s: &str) -> Result<Vec<u8>, ()> {
    if s.len() % 2 != 0 {
        return Err(());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| ()))
        .collect()
}
