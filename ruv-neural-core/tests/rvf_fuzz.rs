//! Adversarial / property tests for the RVF container parser.
//!
//! A binary parser that gates `verify_integrity` and signature checks is a
//! **trust boundary**: it must never panic on hostile input, must reject
//! corruption, and must round-trip everything it accepts. These tests fuzz it
//! with both fully-random and structured-random bytes.

use proptest::prelude::*;

use ruv_neural_core::brain::Atlas;
use ruv_neural_core::embedding::{EmbeddingMetadata, NeuralEmbedding};
use ruv_neural_core::rvf_container::{
    crc32c, embeddings_to_container, RvfContainer, SegmentType, SEGMENT_HEADER_LEN,
};
use ruv_neural_core::rvf_quant::VecDType;

fn meta() -> EmbeddingMetadata {
    EmbeddingMetadata {
        subject_id: None,
        session_id: None,
        cognitive_state: None,
        source_atlas: Atlas::Custom(1),
        embedding_method: "fuzz".into(),
    }
}

fn dtype(code: u8) -> VecDType {
    match code % 5 {
        0 => VecDType::F64,
        1 => VecDType::F32,
        2 => VecDType::F16,
        3 => VecDType::I8,
        _ => VecDType::Binary,
    }
}

proptest! {
    // The parser must never panic on arbitrary bytes — only ever Ok/Err.
    #[test]
    fn never_panics_on_random_bytes(bytes in proptest::collection::vec(any::<u8>(), 0..4096)) {
        let _ = RvfContainer::from_bytes(&bytes);
    }

    // Structured-random bytes that *start* with the right magic still never panic
    // (exercises the header/length decode paths far more often).
    #[test]
    fn never_panics_on_magic_prefixed_bytes(
        tail in proptest::collection::vec(any::<u8>(), 0..2048)
    ) {
        let mut bytes = b"RVFS".to_vec();
        bytes.extend_from_slice(&tail);
        let _ = RvfContainer::from_bytes(&bytes);
    }

    // Every container we build round-trips and verifies.
    #[test]
    fn valid_containers_roundtrip(
        dt in any::<u8>(),
        dim in 1usize..8,
        count in 1usize..6,
        seed in any::<u64>(),
    ) {
        let mut s = seed;
        let mut embs = Vec::new();
        for i in 0..count {
            // Cheap deterministic pseudo-random vector.
            let v: Vec<f64> = (0..dim).map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((s >> 33) as f64 / u32::MAX as f64) * 20.0 - 10.0
            }).collect();
            embs.push(NeuralEmbedding::new(v, i as f64, meta()).unwrap());
        }

        let container = embeddings_to_container(&embs, dtype(dt)).unwrap();
        let bytes = container.to_bytes();
        // Segment stream is always 64-byte aligned.
        prop_assert_eq!(bytes.len() % SEGMENT_HEADER_LEN, 0);

        let back = RvfContainer::from_bytes(&bytes).unwrap();
        prop_assert!(back.verify_integrity().is_ok());
        prop_assert!(back.find(SegmentType::Vec).is_some());
        prop_assert!(back.find(SegmentType::Meta).is_some());
    }

    // Flipping any byte in the first segment's PAYLOAD must be caught by the
    // CRC32C or the content hash (corruption detection).
    #[test]
    fn payload_corruption_is_detected(pos in 0usize..32) {
        let embs = vec![
            NeuralEmbedding::new(vec![1.0, 2.0, 3.0, 4.0], 0.0, meta()).unwrap(),
            NeuralEmbedding::new(vec![5.0, 6.0, 7.0, 8.0], 1.0, meta()).unwrap(),
        ];
        let container = embeddings_to_container(&embs, VecDType::F64).unwrap();
        let mut bytes = container.to_bytes();

        // First segment payload starts right after its 64-byte header.
        let plen = container.segments[0].payload.len();
        let idx = SEGMENT_HEADER_LEN + (pos % plen);
        bytes[idx] ^= 0xFF;

        // Either the streaming parse rejects it, or a successful parse fails
        // verify_integrity — never silently accepted.
        match RvfContainer::from_bytes(&bytes) {
            Err(_) => {}
            Ok(c) => prop_assert!(c.verify_integrity().is_err()),
        }
    }

    // Any truncation of a valid container is handled without panic.
    #[test]
    fn truncation_never_panics(cut in 0usize..512) {
        let embs = vec![NeuralEmbedding::new(vec![1.0; 16], 0.0, meta()).unwrap()];
        let bytes = embeddings_to_container(&embs, VecDType::F32).unwrap().to_bytes();
        let n = cut.min(bytes.len());
        let _ = RvfContainer::from_bytes(&bytes[..n]);
    }
}

// ── Targeted adversarial cases ──────────────────────────────────────────

#[test]
fn rejects_bad_magic() {
    let mut bytes = vec![0u8; SEGMENT_HEADER_LEN];
    bytes[0..4].copy_from_slice(b"XXXX");
    assert!(RvfContainer::from_bytes(&bytes).is_err());
}

#[test]
fn rejects_oversized_payload_length() {
    // Valid magic + version + known segment type, but a colossal payload_length.
    let mut b = vec![0u8; SEGMENT_HEADER_LEN];
    b[0..4].copy_from_slice(b"RVFS");
    b[4] = 1; // version
    b[5] = SegmentType::Vec.to_code();
    b[0x10..0x18].copy_from_slice(&u64::MAX.to_le_bytes()); // payload_length
    assert!(RvfContainer::from_bytes(&b).is_err());
}

#[test]
fn rejects_payload_length_past_buffer() {
    // Claims a 1 KiB payload but provides none.
    let mut b = vec![0u8; SEGMENT_HEADER_LEN];
    b[0..4].copy_from_slice(b"RVFS");
    b[4] = 1;
    b[5] = SegmentType::Meta.to_code();
    b[0x10..0x18].copy_from_slice(&1024u64.to_le_bytes());
    assert!(RvfContainer::from_bytes(&b).is_err());
}

#[test]
fn rejects_unknown_segment_type() {
    let mut b = vec![0u8; SEGMENT_HEADER_LEN];
    b[0..4].copy_from_slice(b"RVFS");
    b[4] = 1;
    b[5] = 0xFE; // not a known segment code
    assert!(RvfContainer::from_bytes(&b).is_err());
}

#[test]
fn rejects_truncated_header() {
    let mut b = b"RVFS".to_vec();
    b.push(1);
    assert!(RvfContainer::from_bytes(&b).is_err());
}

#[test]
fn empty_input_is_empty_container() {
    // Zero bytes is a valid (empty) container, not an error.
    let c = RvfContainer::from_bytes(&[]).unwrap();
    assert!(c.segments.is_empty());
}

#[test]
fn detects_crc_field_tampering() {
    let embs = vec![NeuralEmbedding::new(vec![1.0, 2.0], 0.0, meta()).unwrap()];
    let mut bytes = embeddings_to_container(&embs, VecDType::F64)
        .unwrap()
        .to_bytes();
    // Corrupt the CRC32C field (offset 0x24) of the first header.
    bytes[0x24] ^= 0xFF;
    assert!(RvfContainer::from_bytes(&bytes).is_err());
}

#[test]
fn crc32c_matches_known_vector() {
    // Standard CRC32C check value for "123456789".
    assert_eq!(crc32c(b"123456789"), 0xE306_9283);
}
