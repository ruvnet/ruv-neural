//! RVF `WITNESS` audit chains and `CRYPTO` signature segments.
//!
//! Implements the tamper-evident witness chain RuVector stores in `WITNESS`
//! segments and the Ed25519 attestation it keeps in `CRYPTO` segments, layered
//! on top of [`RvfContainer`]. This is the on-disk counterpart to the
//! workspace's in-memory hash-chained audit trail (ADR-0009): a verifier with
//! only the `.rvf` file and a public key can confirm both that the recorded
//! actions form an unbroken chain and that the container was signed.
//!
//! A witness entry is **73 bytes**, matching the upstream layout:
//!
//! | field         | size | notes                                   |
//! |---------------|------|-----------------------------------------|
//! | `prev_hash`   | 32   | hash of the previous entry (0 = genesis) |
//! | `action_hash` | 32   | hash of the witnessed action            |
//! | `timestamp_ns`| 8    | nanoseconds since the UNIX epoch        |
//! | `witness_type`| 1    | event-type code                         |
//!
//! Per the container profile note, hashes use SHA-256 in place of upstream's
//! SHAKE-256; sizes and offsets are unchanged.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use sha2::{Digest, Sha256};

use crate::error::{Result, RuvNeuralError};
use crate::rvf_container::{RvfContainer, SegmentType, FLAG_SEALED, FLAG_SIGNED};

/// Encoded length of a single witness entry, in bytes.
pub const WITNESS_ENTRY_LEN: usize = 73;

/// Witness event-type codes (subset of RuVector's enumeration).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WitnessType {
    /// Data provenance / origin attestation.
    Provenance,
    /// A computation step.
    Computation,
    /// A similarity search / query.
    Search,
    /// A deletion / redaction.
    Deletion,
    /// A computation correctness proof.
    ComputationProof,
    /// A derivation of new data from existing data.
    Derivation,
}

impl WitnessType {
    /// On-wire type code.
    pub fn to_code(self) -> u8 {
        match self {
            WitnessType::Provenance => 0x01,
            WitnessType::Computation => 0x02,
            WitnessType::Search => 0x03,
            WitnessType::Deletion => 0x04,
            WitnessType::ComputationProof => 0x07,
            WitnessType::Derivation => 0x09,
        }
    }

    /// Parse an on-wire type code.
    pub fn from_code(code: u8) -> Result<Self> {
        Ok(match code {
            0x01 => WitnessType::Provenance,
            0x02 => WitnessType::Computation,
            0x03 => WitnessType::Search,
            0x04 => WitnessType::Deletion,
            0x07 => WitnessType::ComputationProof,
            0x09 => WitnessType::Derivation,
            other => {
                return Err(RuvNeuralError::Serialization(format!(
                    "unknown witness type code: 0x{other:02x}"
                )))
            }
        })
    }
}

/// A single 73-byte witness-chain entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessEntry {
    /// SHA-256 of the previous entry's bytes (zero for the genesis entry).
    pub prev_hash: [u8; 32],
    /// SHA-256 of the action being witnessed.
    pub action_hash: [u8; 32],
    /// Nanoseconds since the UNIX epoch.
    pub timestamp_ns: u64,
    /// Event type.
    pub witness_type: WitnessType,
}

impl WitnessEntry {
    /// Encode the entry to its fixed 73-byte form.
    pub fn to_bytes(&self) -> [u8; WITNESS_ENTRY_LEN] {
        let mut b = [0u8; WITNESS_ENTRY_LEN];
        b[0..32].copy_from_slice(&self.prev_hash);
        b[32..64].copy_from_slice(&self.action_hash);
        b[64..72].copy_from_slice(&self.timestamp_ns.to_le_bytes());
        b[72] = self.witness_type.to_code();
        b
    }

    /// Decode an entry from a 73-byte slice.
    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        if b.len() < WITNESS_ENTRY_LEN {
            return Err(RuvNeuralError::Serialization(
                "witness entry too short".into(),
            ));
        }
        let mut prev_hash = [0u8; 32];
        prev_hash.copy_from_slice(&b[0..32]);
        let mut action_hash = [0u8; 32];
        action_hash.copy_from_slice(&b[32..64]);
        let timestamp_ns = u64::from_le_bytes(b[64..72].try_into().unwrap());
        let witness_type = WitnessType::from_code(b[72])?;
        Ok(Self {
            prev_hash,
            action_hash,
            timestamp_ns,
            witness_type,
        })
    }
}

/// A tamper-evident chain of witness entries.
#[derive(Debug, Clone, Default)]
pub struct WitnessChain {
    /// Entries in append order.
    pub entries: Vec<WitnessEntry>,
}

fn sha256(data: &[u8]) -> [u8; 32] {
    let d = Sha256::digest(data);
    let mut out = [0u8; 32];
    out.copy_from_slice(&d);
    out
}

impl WitnessChain {
    /// Create an empty chain.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an action, linking it to the previous entry.
    pub fn append(&mut self, action: &[u8], witness_type: WitnessType, timestamp_ns: u64) {
        let prev_hash = self
            .entries
            .last()
            .map(|e| sha256(&e.to_bytes()))
            .unwrap_or([0u8; 32]);
        self.entries.push(WitnessEntry {
            prev_hash,
            action_hash: sha256(action),
            timestamp_ns,
            witness_type,
        });
    }

    /// Verify that every entry links to the hash of its predecessor.
    pub fn verify(&self) -> bool {
        let mut expected_prev = [0u8; 32];
        for entry in &self.entries {
            if entry.prev_hash != expected_prev {
                return false;
            }
            expected_prev = sha256(&entry.to_bytes());
        }
        true
    }

    /// Encode the chain as a `WITNESS` segment payload.
    pub fn to_payload(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.entries.len() * WITNESS_ENTRY_LEN);
        for e in &self.entries {
            out.extend_from_slice(&e.to_bytes());
        }
        out
    }

    /// Decode a chain from a `WITNESS` segment payload.
    pub fn from_payload(payload: &[u8]) -> Result<Self> {
        if !payload.len().is_multiple_of(WITNESS_ENTRY_LEN) {
            return Err(RuvNeuralError::Serialization(format!(
                "WITNESS payload length {} is not a multiple of {WITNESS_ENTRY_LEN}",
                payload.len()
            )));
        }
        let entries = payload
            .chunks_exact(WITNESS_ENTRY_LEN)
            .map(WitnessEntry::from_bytes)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { entries })
    }
}

/// Attach a witness chain to a container as a sealed `WITNESS` segment.
pub fn attach_witness(container: &mut RvfContainer, chain: &WitnessChain) {
    container.add_segment(SegmentType::Witness, FLAG_SEALED, chain.to_payload());
}

/// Read and verify the `WITNESS` chain from a container, if present.
pub fn read_witness(container: &RvfContainer) -> Result<Option<WitnessChain>> {
    match container.find(SegmentType::Witness) {
        None => Ok(None),
        Some(seg) => {
            let chain = WitnessChain::from_payload(&seg.payload)?;
            if !chain.verify() {
                return Err(RuvNeuralError::Serialization(
                    "WITNESS chain failed verification (broken link)".into(),
                ));
            }
            Ok(Some(chain))
        }
    }
}

// ── CRYPTO segment: Ed25519 signature over the container ────────────────

/// Ed25519 signature-algorithm code in the `CRYPTO` segment.
pub const SIG_ALGO_ED25519: u16 = 0;

/// Message signed by [`sign_container`]: SHA-256 over the content hashes of
/// every non-`CRYPTO` segment, in order. Independent of the signature itself.
fn signing_message(container: &RvfContainer) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for seg in &container.segments {
        if seg.header.seg_type != SegmentType::Crypto {
            hasher.update(seg.header.content_hash);
        }
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&hasher.finalize());
    out
}

/// Sign the container with `signing_key`, appending a `CRYPTO` segment.
///
/// The `CRYPTO` payload is `sig_algo: u16 | sig_len: u16 | signature | pubkey`.
pub fn sign_container(container: &mut RvfContainer, signing_key: &SigningKey) {
    let msg = signing_message(container);
    let sig = signing_key.sign(&msg);
    let pubkey = signing_key.verifying_key();

    let sig_bytes = sig.to_bytes();
    let mut payload = Vec::with_capacity(4 + sig_bytes.len() + 32);
    payload.extend_from_slice(&SIG_ALGO_ED25519.to_le_bytes());
    payload.extend_from_slice(&(sig_bytes.len() as u16).to_le_bytes());
    payload.extend_from_slice(&sig_bytes);
    payload.extend_from_slice(pubkey.as_bytes());

    container.add_segment(SegmentType::Crypto, FLAG_SIGNED, payload);
}

/// Sign the container with a freshly generated ephemeral Ed25519 key
/// (a self-signed, tamper-evident artifact), returning the public key embedded
/// in the `CRYPTO` segment.
pub fn sign_container_ephemeral(container: &mut RvfContainer) -> VerifyingKey {
    use rand::rngs::OsRng;
    let key = SigningKey::generate(&mut OsRng);
    sign_container(container, &key);
    key.verifying_key()
}

/// Verify the container's `CRYPTO` Ed25519 signature.
///
/// Returns `Ok(true)` if a valid signature is present, `Ok(false)` if it is
/// present but invalid, and an error if the `CRYPTO` segment is missing or
/// malformed.
pub fn verify_container_signature(container: &RvfContainer) -> Result<bool> {
    let crypto = container
        .find(SegmentType::Crypto)
        .ok_or_else(|| RuvNeuralError::Serialization("container has no CRYPTO segment".into()))?;
    let p = &crypto.payload;
    if p.len() < 4 {
        return Err(RuvNeuralError::Serialization(
            "CRYPTO segment too short".into(),
        ));
    }
    let algo = u16::from_le_bytes([p[0], p[1]]);
    if algo != SIG_ALGO_ED25519 {
        return Err(RuvNeuralError::Serialization(format!(
            "unsupported signature algorithm: {algo}"
        )));
    }
    let sig_len = u16::from_le_bytes([p[2], p[3]]) as usize;
    if p.len() < 4 + sig_len + 32 {
        return Err(RuvNeuralError::Serialization(
            "CRYPTO segment truncated".into(),
        ));
    }
    let sig_arr: [u8; 64] = p[4..4 + sig_len]
        .try_into()
        .map_err(|_| RuvNeuralError::Serialization("Ed25519 signature must be 64 bytes".into()))?;
    let pk_arr: [u8; 32] = p[4 + sig_len..4 + sig_len + 32]
        .try_into()
        .map_err(|_| RuvNeuralError::Serialization("Ed25519 public key must be 32 bytes".into()))?;

    let verifying_key = VerifyingKey::from_bytes(&pk_arr)
        .map_err(|e| RuvNeuralError::Serialization(format!("invalid public key: {e}")))?;
    let signature = Signature::from_bytes(&sig_arr);
    let msg = signing_message(container);
    Ok(verifying_key.verify(&msg, &signature).is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::Atlas;
    use crate::embedding::{EmbeddingMetadata, NeuralEmbedding};
    use crate::rvf_container::embeddings_to_container;
    use crate::rvf_quant::VecDType;
    use rand::rngs::OsRng;

    fn embs() -> Vec<NeuralEmbedding> {
        let m = EmbeddingMetadata {
            subject_id: None,
            session_id: None,
            cognitive_state: None,
            source_atlas: Atlas::Custom(3),
            embedding_method: "test".into(),
        };
        vec![NeuralEmbedding::new(vec![1.0, 2.0, 3.0], 0.0, m).unwrap()]
    }

    #[test]
    fn witness_entry_is_73_bytes() {
        let e = WitnessEntry {
            prev_hash: [0u8; 32],
            action_hash: [1u8; 32],
            timestamp_ns: 42,
            witness_type: WitnessType::Provenance,
        };
        let b = e.to_bytes();
        assert_eq!(b.len(), WITNESS_ENTRY_LEN);
        assert_eq!(WitnessEntry::from_bytes(&b).unwrap(), e);
    }

    #[test]
    fn witness_chain_links_and_verifies() {
        let mut chain = WitnessChain::new();
        chain.append(b"ingest", WitnessType::Provenance, 1);
        chain.append(b"embed", WitnessType::Computation, 2);
        chain.append(b"query", WitnessType::Search, 3);
        assert!(chain.verify());

        // Genesis has a zero prev_hash; the rest link forward.
        assert_eq!(chain.entries[0].prev_hash, [0u8; 32]);
        assert_ne!(chain.entries[1].prev_hash, [0u8; 32]);

        let payload = chain.to_payload();
        assert_eq!(payload.len(), 3 * WITNESS_ENTRY_LEN);
        let restored = WitnessChain::from_payload(&payload).unwrap();
        assert!(restored.verify());
        assert_eq!(restored.entries, chain.entries);
    }

    #[test]
    fn witness_tampering_breaks_chain() {
        let mut chain = WitnessChain::new();
        chain.append(b"a", WitnessType::Provenance, 1);
        chain.append(b"b", WitnessType::Computation, 2);
        // Tamper with the first action after the fact.
        chain.entries[0].action_hash[0] ^= 0xFF;
        assert!(!chain.verify());
    }

    #[test]
    fn witness_segment_roundtrip_through_container() {
        let mut container = embeddings_to_container(&embs(), VecDType::F32).unwrap();
        let mut chain = WitnessChain::new();
        chain.append(b"provenance", WitnessType::Provenance, 10);
        chain.append(b"derivation", WitnessType::Derivation, 20);
        attach_witness(&mut container, &chain);

        let bytes = container.to_bytes();
        let back = RvfContainer::from_bytes(&bytes).unwrap();
        let recovered = read_witness(&back).unwrap().unwrap();
        assert_eq!(recovered.entries.len(), 2);
        assert!(recovered.verify());
    }

    #[test]
    fn sign_and_verify_container() {
        let mut container = embeddings_to_container(&embs(), VecDType::F32).unwrap();
        let key = SigningKey::generate(&mut OsRng);
        sign_container(&mut container, &key);

        // Round-trips through bytes and still verifies.
        let back = RvfContainer::from_bytes(&container.to_bytes()).unwrap();
        assert!(verify_container_signature(&back).unwrap());
    }

    #[test]
    fn tampered_vector_fails_signature() {
        let mut container = embeddings_to_container(&embs(), VecDType::F32).unwrap();
        let key = SigningKey::generate(&mut OsRng);
        sign_container(&mut container, &key);

        // Mutate the VEC payload and recompute its content hash so the CRC and
        // content-hash checks still pass — only the signature should catch it.
        let vec_idx = container
            .segments
            .iter()
            .position(|s| s.header.seg_type == SegmentType::Vec)
            .unwrap();
        container.segments[vec_idx].payload[30] ^= 0xFF;
        let new_payload = container.segments[vec_idx].payload.clone();
        container.segments[vec_idx].header.crc32c = crate::rvf_container::crc32c(&new_payload);
        container.segments[vec_idx].header.content_hash = {
            let d = Sha256::digest(&new_payload);
            let mut h = [0u8; 16];
            h.copy_from_slice(&d[..16]);
            h
        };

        assert!(!verify_container_signature(&container).unwrap());
    }
}
