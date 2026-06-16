//! RVF (RuVector Format) multi-segment container — the on-disk `.rvf` format.
//!
//! This is a faithful, dependency-free implementation of RuVector's RVF
//! container framing (`ruvnet/ruvector`, `crates/rvf`). Where the legacy
//! [`crate::rvf`] module stored a single typed blob, this module implements
//! the real **segmented** substrate: a sequence of self-describing 64-byte
//! segment headers, each followed by an aligned payload, so a single file can
//! carry vectors (`VEC`), a metadata directory (`META`/`MANIFEST`), a
//! tamper-evident audit chain (`WITNESS`), and an Ed25519 signature (`CRYPTO`).
//!
//! ## Wire format
//!
//! Magic `0x52564653` ("RVFS"), little-endian, segments 64-byte aligned. Each
//! segment begins with a 64-byte `repr(C)` header:
//!
//! | off  | size | field             |
//! |------|------|-------------------|
//! | 0x00 | 4    | `magic`           |
//! | 0x04 | 1    | `version`         |
//! | 0x05 | 1    | `seg_type`        |
//! | 0x06 | 2    | `flags`           |
//! | 0x08 | 8    | `segment_id`      |
//! | 0x10 | 8    | `payload_length`  |
//! | 0x18 | 8    | `timestamp_ns`    |
//! | 0x20 | 1    | `checksum_algo`   |
//! | 0x21 | 1    | `compression`     |
//! | 0x22 | 2    | `reserved_0`      |
//! | 0x24 | 4    | `crc32c`          |
//! | 0x28 | 16   | `content_hash`    |
//! | 0x38 | 4    | `uncompressed_len`|
//! | 0x3C | 4    | `alignment_pad`   |
//!
//! ### Profile note on hashing
//!
//! Upstream RVF uses SHAKE-256 for `content_hash` (`checksum_algo = 2`) and the
//! `WITNESS` chain. To keep this crate dependency-free, this profile computes
//! the **first 128 bits of SHA-256** for `content_hash` and SHA-256 for the
//! witness chain, while keeping the byte layout, field sizes, and `crc32c`
//! integrity word identical. The `checksum_algo` slot records this choice.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::embedding::{EmbeddingMetadata, NeuralEmbedding};
use crate::error::{Result, RuvNeuralError};
use crate::rvf_quant::{decode_vector, encode_vector, VecDType};

/// Magic number for the RVF container: ASCII "RVFS".
pub const RVFS_MAGIC: u32 = 0x5256_4653;

/// Container format version.
pub const RVF_CONTAINER_VERSION: u8 = 1;

/// Fixed segment-header length, in bytes.
pub const SEGMENT_HEADER_LEN: usize = 64;

/// Segment alignment boundary, in bytes.
pub const SEGMENT_ALIGN: usize = 64;

/// `checksum_algo` value for this profile: SHA-256 truncated to 128 bits.
/// (Occupies upstream's SHAKE-256 slot; see the module-level profile note.)
pub const CHECKSUM_SHA256_128: u8 = 2;

/// Maximum payload accepted for a single segment when reading (256 MiB).
pub const MAX_SEGMENT_PAYLOAD: u64 = 256 * 1024 * 1024;

// Segment flag bits.
/// Payload is LZ4/ZSTD compressed (not produced by this profile).
pub const FLAG_COMPRESSED: u16 = 1 << 0;
/// Payload is encrypted.
pub const FLAG_ENCRYPTED: u16 = 1 << 1;
/// Segment carries / is covered by a signature.
pub const FLAG_SIGNED: u16 = 1 << 2;
/// Segment is sealed (immutable).
pub const FLAG_SEALED: u16 = 1 << 3;

/// RVF segment types, matching RuVector's segment-type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SegmentType {
    /// Raw vector embeddings.
    Vec,
    /// HNSW adjacency / routing index.
    Index,
    /// Segment directory / epoch state.
    Manifest,
    /// Quantization dictionaries.
    Quant,
    /// Key-value metadata.
    Meta,
    /// Audit trail / attestation proofs.
    Witness,
    /// Domain profile declaration.
    Profile,
    /// Key material / signature chains.
    Crypto,
    /// WASM microkernel / bytecode.
    Wasm,
    /// Federated-learning manifest.
    FederatedManifest,
}

impl SegmentType {
    /// On-wire type code (matches the upstream `seg_type` table).
    pub fn to_code(self) -> u8 {
        match self {
            SegmentType::Vec => 0x01,
            SegmentType::Index => 0x02,
            SegmentType::Manifest => 0x05,
            SegmentType::Quant => 0x06,
            SegmentType::Meta => 0x07,
            SegmentType::Witness => 0x0A,
            SegmentType::Profile => 0x0B,
            SegmentType::Crypto => 0x0C,
            SegmentType::Wasm => 0x10,
            SegmentType::FederatedManifest => 0x40,
        }
    }

    /// Parse an on-wire type code.
    pub fn from_code(code: u8) -> Result<Self> {
        Ok(match code {
            0x01 => SegmentType::Vec,
            0x02 => SegmentType::Index,
            0x05 => SegmentType::Manifest,
            0x06 => SegmentType::Quant,
            0x07 => SegmentType::Meta,
            0x0A => SegmentType::Witness,
            0x0B => SegmentType::Profile,
            0x0C => SegmentType::Crypto,
            0x10 => SegmentType::Wasm,
            0x40 => SegmentType::FederatedManifest,
            other => {
                return Err(RuvNeuralError::Serialization(format!(
                    "unknown RVF segment type code: 0x{other:02x}"
                )))
            }
        })
    }
}

// ── CRC32C (Castagnoli, poly 0x82F63B78) ────────────────────────────────

/// Compute the CRC32C (Castagnoli) checksum of `data`.
pub fn crc32c(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0x82F6_3B78 & mask);
        }
    }
    !crc
}

/// First 128 bits of the SHA-256 of `data`.
fn content_hash_128(data: &[u8]) -> [u8; 16] {
    let digest = Sha256::digest(data);
    let mut out = [0u8; 16];
    out.copy_from_slice(&digest[..16]);
    out
}

/// Current wall-clock time in nanoseconds since the UNIX epoch (0 if before).
pub fn now_unix_nanos() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// A 64-byte self-describing segment header.
#[derive(Debug, Clone)]
pub struct SegmentHeader {
    /// Segment type.
    pub seg_type: SegmentType,
    /// Flag bitfield (`FLAG_*`).
    pub flags: u16,
    /// Monotonically increasing segment id within the container.
    pub segment_id: u64,
    /// Payload length in bytes.
    pub payload_length: u64,
    /// Creation time, nanoseconds since the UNIX epoch.
    pub timestamp_ns: u64,
    /// Checksum algorithm code for `content_hash`.
    pub checksum_algo: u8,
    /// Compression code (0 = none).
    pub compression: u8,
    /// CRC32C of the payload.
    pub crc32c: u32,
    /// First 128 bits of the payload hash.
    pub content_hash: [u8; 16],
    /// Uncompressed payload length (equals `payload_length` when uncompressed).
    pub uncompressed_len: u32,
}

impl SegmentHeader {
    /// Encode the header to its fixed 64-byte little-endian form.
    pub fn to_bytes(&self) -> [u8; SEGMENT_HEADER_LEN] {
        let mut b = [0u8; SEGMENT_HEADER_LEN];
        // Magic is a literal 4-byte tag ("RVFS"), stored big-endian so the
        // bytes read in order; all other fields are little-endian.
        b[0x00..0x04].copy_from_slice(&RVFS_MAGIC.to_be_bytes());
        b[0x04] = RVF_CONTAINER_VERSION;
        b[0x05] = self.seg_type.to_code();
        b[0x06..0x08].copy_from_slice(&self.flags.to_le_bytes());
        b[0x08..0x10].copy_from_slice(&self.segment_id.to_le_bytes());
        b[0x10..0x18].copy_from_slice(&self.payload_length.to_le_bytes());
        b[0x18..0x20].copy_from_slice(&self.timestamp_ns.to_le_bytes());
        b[0x20] = self.checksum_algo;
        b[0x21] = self.compression;
        // 0x22..0x24 reserved_0 (zero)
        b[0x24..0x28].copy_from_slice(&self.crc32c.to_le_bytes());
        b[0x28..0x38].copy_from_slice(&self.content_hash);
        b[0x38..0x3C].copy_from_slice(&self.uncompressed_len.to_le_bytes());
        // 0x3C..0x40 alignment_pad (zero)
        b
    }

    /// Decode a header from a 64-byte slice.
    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        if b.len() < SEGMENT_HEADER_LEN {
            return Err(RuvNeuralError::Serialization(format!(
                "RVF segment header too short: {} bytes (need {SEGMENT_HEADER_LEN})",
                b.len()
            )));
        }
        let magic = u32::from_be_bytes(b[0x00..0x04].try_into().unwrap());
        if magic != RVFS_MAGIC {
            return Err(RuvNeuralError::Serialization(
                "invalid RVF magic (expected RVFS)".into(),
            ));
        }
        let version = b[0x04];
        if version != RVF_CONTAINER_VERSION {
            return Err(RuvNeuralError::Serialization(format!(
                "unsupported RVF container version: {version}"
            )));
        }
        let seg_type = SegmentType::from_code(b[0x05])?;
        let flags = u16::from_le_bytes(b[0x06..0x08].try_into().unwrap());
        let segment_id = u64::from_le_bytes(b[0x08..0x10].try_into().unwrap());
        let payload_length = u64::from_le_bytes(b[0x10..0x18].try_into().unwrap());
        let timestamp_ns = u64::from_le_bytes(b[0x18..0x20].try_into().unwrap());
        let checksum_algo = b[0x20];
        let compression = b[0x21];
        let crc32c = u32::from_le_bytes(b[0x24..0x28].try_into().unwrap());
        let mut content_hash = [0u8; 16];
        content_hash.copy_from_slice(&b[0x28..0x38]);
        let uncompressed_len = u32::from_le_bytes(b[0x38..0x3C].try_into().unwrap());

        Ok(Self {
            seg_type,
            flags,
            segment_id,
            payload_length,
            timestamp_ns,
            checksum_algo,
            compression,
            crc32c,
            content_hash,
            uncompressed_len,
        })
    }
}

/// A single segment: header plus payload bytes.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Segment header.
    pub header: SegmentHeader,
    /// Raw payload bytes.
    pub payload: Vec<u8>,
}

/// An RVF container: an ordered collection of segments.
#[derive(Debug, Clone, Default)]
pub struct RvfContainer {
    /// Segments in write order.
    pub segments: Vec<Segment>,
}

impl RvfContainer {
    /// Create an empty container.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a segment, computing CRC32C and content hash over `payload`.
    pub fn add_segment(&mut self, seg_type: SegmentType, flags: u16, payload: Vec<u8>) {
        let segment_id = self.segments.len() as u64;
        let header = SegmentHeader {
            seg_type,
            flags,
            segment_id,
            payload_length: payload.len() as u64,
            timestamp_ns: now_unix_nanos(),
            checksum_algo: CHECKSUM_SHA256_128,
            compression: 0,
            crc32c: crc32c(&payload),
            content_hash: content_hash_128(&payload),
            uncompressed_len: payload.len() as u32,
        };
        self.segments.push(Segment { header, payload });
    }

    /// First segment of the given type, if present.
    pub fn find(&self, seg_type: SegmentType) -> Option<&Segment> {
        self.segments.iter().find(|s| s.header.seg_type == seg_type)
    }

    /// Serialize the container to bytes (segments 64-byte aligned).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for seg in &self.segments {
            out.extend_from_slice(&seg.header.to_bytes());
            out.extend_from_slice(&seg.payload);
            // Pad the payload up to the alignment boundary.
            let rem = seg.payload.len() % SEGMENT_ALIGN;
            if rem != 0 {
                out.resize(out.len() + (SEGMENT_ALIGN - rem), 0);
            }
        }
        out
    }

    /// Parse a container from bytes, verifying every segment's integrity.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut segments = Vec::new();
        let mut pos = 0usize;
        while pos < bytes.len() {
            if pos + SEGMENT_HEADER_LEN > bytes.len() {
                return Err(RuvNeuralError::Serialization(
                    "truncated RVF segment header".into(),
                ));
            }
            let header = SegmentHeader::from_bytes(&bytes[pos..pos + SEGMENT_HEADER_LEN])?;
            if header.payload_length > MAX_SEGMENT_PAYLOAD {
                return Err(RuvNeuralError::Serialization(format!(
                    "RVF segment payload {} exceeds maximum {MAX_SEGMENT_PAYLOAD}",
                    header.payload_length
                )));
            }
            let pstart = pos + SEGMENT_HEADER_LEN;
            let plen = header.payload_length as usize;
            if pstart + plen > bytes.len() {
                return Err(RuvNeuralError::Serialization(
                    "truncated RVF segment payload".into(),
                ));
            }
            let payload = bytes[pstart..pstart + plen].to_vec();

            // Integrity checks.
            if crc32c(&payload) != header.crc32c {
                return Err(RuvNeuralError::Serialization(format!(
                    "RVF CRC32C mismatch in segment {}",
                    header.segment_id
                )));
            }
            if content_hash_128(&payload) != header.content_hash {
                return Err(RuvNeuralError::Serialization(format!(
                    "RVF content-hash mismatch in segment {}",
                    header.segment_id
                )));
            }

            segments.push(Segment { header, payload });

            // Advance past the payload and its alignment padding.
            let mut next = pstart + plen;
            let rem = plen % SEGMENT_ALIGN;
            if rem != 0 {
                next += SEGMENT_ALIGN - rem;
            }
            pos = next;
        }
        Ok(Self { segments })
    }

    /// Recompute and verify the integrity of every segment.
    pub fn verify_integrity(&self) -> Result<()> {
        for seg in &self.segments {
            if crc32c(&seg.payload) != seg.header.crc32c {
                return Err(RuvNeuralError::Serialization(format!(
                    "CRC32C mismatch in segment {}",
                    seg.header.segment_id
                )));
            }
            if content_hash_128(&seg.payload) != seg.header.content_hash {
                return Err(RuvNeuralError::Serialization(format!(
                    "content-hash mismatch in segment {}",
                    seg.header.segment_id
                )));
            }
        }
        Ok(())
    }

    /// Write the container to any writer.
    pub fn write_to<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.to_bytes())
            .map_err(|e| RuvNeuralError::Serialization(e.to_string()))
    }

    /// Read a container from any reader.
    pub fn read_from<R: std::io::Read>(r: &mut R) -> Result<Self> {
        let mut buf = Vec::new();
        r.read_to_end(&mut buf)
            .map_err(|e| RuvNeuralError::Serialization(e.to_string()))?;
        Self::from_bytes(&buf)
    }
}

// ── Embedding ⇄ VEC + META mapping ──────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct RecordMeta {
    subject_id: Option<String>,
    session_id: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct VecMeta {
    method: String,
    dimension: usize,
    count: usize,
    dtype: String,
    records: Vec<RecordMeta>,
}

/// Build a `VEC` segment payload from embeddings at a chosen quantization.
fn build_vec_payload(embeddings: &[NeuralEmbedding], dim: usize, dtype: VecDType) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&(dim as u32).to_le_bytes());
    p.extend_from_slice(&0u32.to_le_bytes()); // reserved
    p.extend_from_slice(&(embeddings.len() as u64).to_le_bytes());
    p.push(dtype.to_code());
    p.extend_from_slice(&[0u8; 7]); // reserved
    for emb in embeddings {
        p.extend_from_slice(&emb.timestamp.to_le_bytes());
        p.extend_from_slice(&encode_vector(&emb.vector, dtype));
    }
    p
}

/// Construct a signed-ready RVF container holding `embeddings` as a `VEC`
/// segment (at quantization `dtype`) plus a `META` directory segment.
///
/// # Errors
/// Returns an error if `embeddings` is empty or dimensions are inconsistent.
pub fn embeddings_to_container(
    embeddings: &[NeuralEmbedding],
    dtype: VecDType,
) -> Result<RvfContainer> {
    if embeddings.is_empty() {
        return Err(RuvNeuralError::Embedding(
            "cannot build RVF container from empty embedding list".into(),
        ));
    }
    let dim = embeddings[0].dimension;
    if let Some(bad) = embeddings.iter().find(|e| e.dimension != dim) {
        return Err(RuvNeuralError::DimensionMismatch {
            expected: dim,
            got: bad.dimension,
        });
    }

    let meta = VecMeta {
        method: embeddings[0].metadata.embedding_method.clone(),
        dimension: dim,
        count: embeddings.len(),
        dtype: dtype.name().to_string(),
        records: embeddings
            .iter()
            .map(|e| RecordMeta {
                subject_id: e.metadata.subject_id.clone(),
                session_id: e.metadata.session_id.clone(),
            })
            .collect(),
    };
    let meta_json = serde_json::to_vec(&meta)
        .map_err(|e| RuvNeuralError::Serialization(e.to_string()))?;

    let mut container = RvfContainer::new();
    container.add_segment(SegmentType::Meta, FLAG_SEALED, meta_json);
    container.add_segment(
        SegmentType::Vec,
        FLAG_SEALED,
        build_vec_payload(embeddings, dim, dtype),
    );
    Ok(container)
}

/// Reconstruct embeddings from a container's `META` + `VEC` segments.
///
/// # Errors
/// Returns an error if either segment is missing or malformed.
pub fn container_to_embeddings(container: &RvfContainer) -> Result<Vec<NeuralEmbedding>> {
    let meta_seg = container.find(SegmentType::Meta).ok_or_else(|| {
        RuvNeuralError::Serialization("RVF container missing META segment".into())
    })?;
    let vec_seg = container.find(SegmentType::Vec).ok_or_else(|| {
        RuvNeuralError::Serialization("RVF container missing VEC segment".into())
    })?;
    let meta: VecMeta = serde_json::from_slice(&meta_seg.payload)
        .map_err(|e| RuvNeuralError::Serialization(e.to_string()))?;

    let p = &vec_seg.payload;
    if p.len() < 24 {
        return Err(RuvNeuralError::Serialization("VEC segment too short".into()));
    }
    let dim = u32::from_le_bytes(p[0..4].try_into().unwrap()) as usize;
    let count = u64::from_le_bytes(p[8..16].try_into().unwrap()) as usize;
    let dtype = VecDType::from_code(p[16])?;
    if dim != meta.dimension || count != meta.count {
        return Err(RuvNeuralError::Serialization(
            "VEC header disagrees with META directory".into(),
        ));
    }

    let rec_len = 8 + dtype.encoded_len(dim);
    let mut out = Vec::with_capacity(count);
    let mut off = 24usize;
    for i in 0..count {
        if off + rec_len > p.len() {
            return Err(RuvNeuralError::Serialization(
                "VEC segment truncated mid-record".into(),
            ));
        }
        let timestamp = f64::from_le_bytes(p[off..off + 8].try_into().unwrap());
        let vector = decode_vector(&p[off + 8..off + rec_len], dim, dtype)?;
        off += rec_len;

        let rec_meta = meta.records.get(i);
        let emb_meta = EmbeddingMetadata {
            subject_id: rec_meta.and_then(|r| r.subject_id.clone()),
            session_id: rec_meta.and_then(|r| r.session_id.clone()),
            cognitive_state: None,
            source_atlas: crate::brain::Atlas::Custom(dim),
            embedding_method: meta.method.clone(),
        };
        out.push(NeuralEmbedding::new(vector, timestamp, emb_meta)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::Atlas;
    use crate::rvf_quant::VecDType;

    fn meta(method: &str, dim: usize) -> EmbeddingMetadata {
        EmbeddingMetadata {
            subject_id: Some("sub-01".into()),
            session_id: Some("ses-01".into()),
            cognitive_state: None,
            source_atlas: Atlas::Custom(dim),
            embedding_method: method.into(),
        }
    }

    fn sample() -> Vec<NeuralEmbedding> {
        vec![
            NeuralEmbedding::new(vec![1.0, -2.0, 3.0, -4.0], 0.0, meta("spectral", 4)).unwrap(),
            NeuralEmbedding::new(vec![0.5, 0.25, -0.5, 0.75], 1.0, meta("spectral", 4)).unwrap(),
        ]
    }

    #[test]
    fn crc32c_known_vector() {
        // CRC32C of the ASCII string "123456789" is 0xE3069283.
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn segment_header_roundtrip() {
        let payload = b"hello world".to_vec();
        let header = SegmentHeader {
            seg_type: SegmentType::Vec,
            flags: FLAG_SEALED,
            segment_id: 7,
            payload_length: payload.len() as u64,
            timestamp_ns: 123_456_789,
            checksum_algo: CHECKSUM_SHA256_128,
            compression: 0,
            crc32c: crc32c(&payload),
            content_hash: content_hash_128(&payload),
            uncompressed_len: payload.len() as u32,
        };
        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), SEGMENT_HEADER_LEN);
        assert_eq!(&bytes[0..4], b"RVFS");
        assert_eq!(u32::from_be_bytes(bytes[0..4].try_into().unwrap()), RVFS_MAGIC);
        let back = SegmentHeader::from_bytes(&bytes).unwrap();
        assert_eq!(back.seg_type, SegmentType::Vec);
        assert_eq!(back.segment_id, 7);
        assert_eq!(back.payload_length, payload.len() as u64);
        assert_eq!(back.crc32c, header.crc32c);
    }

    #[test]
    fn segment_type_codes_match_spec() {
        assert_eq!(SegmentType::Vec.to_code(), 0x01);
        assert_eq!(SegmentType::Index.to_code(), 0x02);
        assert_eq!(SegmentType::Manifest.to_code(), 0x05);
        assert_eq!(SegmentType::Witness.to_code(), 0x0A);
        assert_eq!(SegmentType::Crypto.to_code(), 0x0C);
        for t in [
            SegmentType::Vec,
            SegmentType::Index,
            SegmentType::Manifest,
            SegmentType::Quant,
            SegmentType::Meta,
            SegmentType::Witness,
            SegmentType::Crypto,
            SegmentType::Wasm,
            SegmentType::FederatedManifest,
        ] {
            assert_eq!(SegmentType::from_code(t.to_code()).unwrap(), t);
        }
    }

    #[test]
    fn container_roundtrip_lossless_f64() {
        let embs = sample();
        let container = embeddings_to_container(&embs, VecDType::F64).unwrap();
        let bytes = container.to_bytes();
        // Every segment + payload is 64-byte aligned.
        assert!(bytes.len().is_multiple_of(SEGMENT_ALIGN));

        let back = RvfContainer::from_bytes(&bytes).unwrap();
        back.verify_integrity().unwrap();
        let restored = container_to_embeddings(&back).unwrap();

        assert_eq!(restored.len(), 2);
        assert_eq!(restored[0].metadata.embedding_method, "spectral");
        assert_eq!(restored[0].metadata.subject_id.as_deref(), Some("sub-01"));
        for (a, b) in embs.iter().zip(restored.iter()) {
            assert!((a.timestamp - b.timestamp).abs() < 1e-12);
            for (x, y) in a.vector.iter().zip(b.vector.iter()) {
                assert!((x - y).abs() < 1e-12, "f64 must be lossless");
            }
        }
    }

    #[test]
    fn container_quantized_sizes_shrink() {
        let embs: Vec<NeuralEmbedding> = (0..32)
            .map(|i| {
                let v: Vec<f64> = (0..128).map(|j| ((i + j) as f64 % 7.0) - 3.0).collect();
                NeuralEmbedding::new(v, i as f64, meta("spectral", 128)).unwrap()
            })
            .collect();

        let f32_len = embeddings_to_container(&embs, VecDType::F32)
            .unwrap()
            .find(SegmentType::Vec)
            .unwrap()
            .payload
            .len();
        let f16_len = embeddings_to_container(&embs, VecDType::F16)
            .unwrap()
            .find(SegmentType::Vec)
            .unwrap()
            .payload
            .len();
        let bin_len = embeddings_to_container(&embs, VecDType::Binary)
            .unwrap()
            .find(SegmentType::Vec)
            .unwrap()
            .payload
            .len();

        // f16 roughly halves the vector bytes; binary is far smaller again.
        assert!(f16_len < f32_len);
        assert!(bin_len < f16_len);
    }

    #[test]
    fn quantized_roundtrip_recovers_shape() {
        let embs = sample();
        for dtype in [VecDType::F16, VecDType::I8, VecDType::Binary] {
            let c = embeddings_to_container(&embs, dtype).unwrap();
            let back = RvfContainer::from_bytes(&c.to_bytes()).unwrap();
            let restored = container_to_embeddings(&back).unwrap();
            assert_eq!(restored.len(), embs.len());
            assert_eq!(restored[0].dimension, 4);
        }
    }

    #[test]
    fn corrupted_payload_is_detected() {
        let embs = sample();
        let mut bytes = embeddings_to_container(&embs, VecDType::F32)
            .unwrap()
            .to_bytes();
        // Flip a byte inside the first payload (just past the 64-byte header).
        bytes[SEGMENT_HEADER_LEN + 2] ^= 0xFF;
        assert!(RvfContainer::from_bytes(&bytes).is_err());
    }

    #[test]
    fn empty_embeddings_rejected() {
        assert!(embeddings_to_container(&[], VecDType::F32).is_err());
    }
}
