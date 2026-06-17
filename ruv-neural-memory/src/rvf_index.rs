//! RVF `INDEX` segments: pack vectors and their HNSW ANN graph into one `.rvf`.
//!
//! Realizes ADR-0023 point 4 ("store & retrieve embeddings via HNSW") on top of
//! the [`RvfContainer`] substrate: [`build_indexed_container`] writes a `META`
//! directory, a `VEC` segment, and an `INDEX` segment holding the serialized
//! HNSW topology, so a single self-describing file carries both the vectors and
//! a ready-to-query approximate-nearest-neighbour graph.

use ruv_neural_core::embedding::NeuralEmbedding;
use ruv_neural_core::error::{Result, RuvNeuralError};
use ruv_neural_core::rvf_container::{
    container_to_embeddings, embeddings_to_container, RvfContainer, SegmentType, FLAG_SEALED,
};
use ruv_neural_core::rvf_quant::VecDType;

use crate::hnsw::{HnswGraph, HnswIndex};

/// Typical HNSW connectivity (`m`) used when building an indexed container.
pub const DEFAULT_M: usize = 16;

/// Typical HNSW construction search width (`ef_construction`).
pub const DEFAULT_EF_CONSTRUCTION: usize = 200;

/// Build an RVF container with `META` + `VEC` + `INDEX` segments from
/// `embeddings`, constructing an HNSW graph over them.
///
/// `dtype` controls how the `VEC` segment is stored. The HNSW graph is always
/// built from the full-precision vectors so its distances are exact regardless
/// of the chosen `VEC` quantization.
///
/// # Errors
/// Returns an error if `embeddings` is empty or dimensions are inconsistent.
pub fn build_indexed_container(
    embeddings: &[NeuralEmbedding],
    dtype: VecDType,
    m: usize,
    ef_construction: usize,
) -> Result<RvfContainer> {
    let mut container = embeddings_to_container(embeddings, dtype)?;

    let mut index = HnswIndex::new(m, ef_construction);
    for emb in embeddings {
        index.insert(&emb.vector);
    }

    let payload = bincode::serialize(&index.export_graph())
        .map_err(|e| RuvNeuralError::Serialization(format!("INDEX segment encode: {e}")))?;
    container.add_segment(SegmentType::Index, FLAG_SEALED, payload);
    Ok(container)
}

/// Load embeddings and a ready-to-query [`HnswIndex`] from a container that has
/// `VEC` + `INDEX` segments.
///
/// # Errors
/// Returns an error if either segment is missing or malformed.
pub fn load_indexed_container(
    container: &RvfContainer,
) -> Result<(Vec<NeuralEmbedding>, HnswIndex)> {
    let embeddings = container_to_embeddings(container)?;
    let index_seg = container.find(SegmentType::Index).ok_or_else(|| {
        RuvNeuralError::Serialization("RVF container missing INDEX segment".into())
    })?;
    let graph: HnswGraph = bincode::deserialize(&index_seg.payload)
        .map_err(|e| RuvNeuralError::Serialization(format!("INDEX segment decode: {e}")))?;

    let vectors: Vec<Vec<f64>> = embeddings.iter().map(|e| e.vector.clone()).collect();
    let index = HnswIndex::from_graph(graph, vectors);
    Ok((embeddings, index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruv_neural_core::brain::Atlas;
    use ruv_neural_core::embedding::EmbeddingMetadata;

    fn emb(vector: Vec<f64>, ts: f64) -> NeuralEmbedding {
        NeuralEmbedding::new(
            vector,
            ts,
            EmbeddingMetadata {
                subject_id: None,
                session_id: None,
                cognitive_state: None,
                source_atlas: Atlas::Custom(4),
                embedding_method: "spectral".into(),
            },
        )
        .unwrap()
    }

    fn corpus(n: usize, dim: usize) -> Vec<NeuralEmbedding> {
        (0..n)
            .map(|i| {
                // Distinct, well-separated vectors so each node is uniquely its
                // own nearest neighbour.
                let v: Vec<f64> = (0..dim).map(|j| i as f64 + j as f64 / 100.0).collect();
                emb(v, i as f64)
            })
            .collect()
    }

    #[test]
    fn indexed_container_has_all_segments() {
        let embs = corpus(40, 4);
        let c = build_indexed_container(&embs, VecDType::F64, DEFAULT_M, DEFAULT_EF_CONSTRUCTION)
            .unwrap();
        assert!(c.find(SegmentType::Meta).is_some());
        assert!(c.find(SegmentType::Vec).is_some());
        assert!(c.find(SegmentType::Index).is_some());
    }

    #[test]
    fn index_survives_container_roundtrip() {
        let embs = corpus(60, 8);
        let container =
            build_indexed_container(&embs, VecDType::F64, DEFAULT_M, DEFAULT_EF_CONSTRUCTION)
                .unwrap();

        // Round-trip through raw bytes (and integrity checks).
        let bytes = container.to_bytes();
        let reloaded = RvfContainer::from_bytes(&bytes).unwrap();
        reloaded.verify_integrity().unwrap();

        let (loaded_embs, loaded_index) = load_indexed_container(&reloaded).unwrap();
        assert_eq!(loaded_embs.len(), 60);
        assert_eq!(loaded_index.len(), 60);

        // The reloaded index must return the same nearest neighbour as a fresh
        // index built from scratch on the same data.
        let mut fresh = HnswIndex::new(DEFAULT_M, DEFAULT_EF_CONSTRUCTION);
        for e in &embs {
            fresh.insert(&e.vector);
        }
        let query = &embs[3].vector;
        let from_disk = loaded_index.search(query, 5, 50);
        let from_fresh = fresh.search(query, 5, 50);
        assert_eq!(from_disk[0].0, from_fresh[0].0);
        // The query vector itself is its own nearest neighbour (distance ~0).
        assert_eq!(from_disk[0].0, 3);
        assert!(from_disk[0].1 < 1e-9);
    }

    #[test]
    fn missing_index_segment_errors() {
        let embs = corpus(5, 4);
        // A plain VEC/META container has no INDEX segment.
        let c = embeddings_to_container(&embs, VecDType::F64).unwrap();
        assert!(load_indexed_container(&c).is_err());
    }
}
