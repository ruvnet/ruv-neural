//! RuVector-backed neural embedding index.
//!
//! This module is intentionally feature-gated. It provides an in-process
//! `ruvector-core::VectorDB` adapter for `NeuralEmbedding` so callers can use
//! the ruvnet/RuVector search stack without replacing the existing memory APIs.

use std::collections::HashMap;

use ruv_neural_core::embedding::NeuralEmbedding;
use ruv_neural_core::error::{Result, RuvNeuralError};
use ruv_neural_core::topology::CognitiveState;
use ruv_neural_core::traits::NeuralMemory;
use ruvector_core::{
    types::{DbOptions, DistanceMetric, HnswConfig, SearchQuery, VectorEntry},
    VectorDB,
};
use serde_json::Value;

/// RuVector-backed memory index for neural embeddings.
pub struct RuvectorMemoryStore {
    db: VectorDB,
    dimension: usize,
}

impl RuvectorMemoryStore {
    /// Create an in-memory RuVector index for embeddings of `dimension`.
    pub fn new(dimension: usize, capacity: usize) -> Result<Self> {
        if dimension == 0 {
            return Err(RuvNeuralError::Memory(
                "RuVector memory dimension must be positive".into(),
            ));
        }
        let options = DbOptions {
            dimensions: dimension,
            distance_metric: DistanceMetric::Cosine,
            storage_path: ":memory:".to_string(),
            hnsw_config: Some(HnswConfig {
                m: 16,
                ef_construction: 100,
                ef_search: 50,
                max_elements: capacity.max(1),
            }),
            quantization: None,
        };
        let db = VectorDB::new(options)
            .map_err(|e| RuvNeuralError::Memory(format!("failed to create RuVector DB: {e}")))?;
        Ok(Self { db, dimension })
    }

    /// Store an embedding and return the RuVector ID.
    pub fn insert(&self, embedding: &NeuralEmbedding) -> Result<String> {
        self.validate_embedding(embedding)?;
        let entry = VectorEntry {
            id: Some(embedding_id(embedding)),
            vector: to_f32_vector(&embedding.vector)?,
            metadata: Some(metadata_map(embedding)),
        };
        self.db
            .insert(entry)
            .map_err(|e| RuvNeuralError::Memory(format!("RuVector insert failed: {e}")))
    }

    /// Search nearest embeddings by vector, returning RuVector IDs and scores.
    pub fn search_ids(&self, embedding: &NeuralEmbedding, k: usize) -> Result<Vec<(String, f32)>> {
        self.validate_embedding(embedding)?;
        let results = self
            .db
            .search(SearchQuery {
                vector: to_f32_vector(&embedding.vector)?,
                k,
                filter: None,
                ef_search: None,
            })
            .map_err(|e| RuvNeuralError::Memory(format!("RuVector search failed: {e}")))?;
        Ok(results
            .into_iter()
            .map(|result| (result.id, result.score))
            .collect())
    }

    /// Number of vectors indexed by RuVector.
    pub fn len(&self) -> Result<usize> {
        self.db
            .len()
            .map_err(|e| RuvNeuralError::Memory(format!("RuVector len failed: {e}")))
    }

    /// Returns true when the RuVector index contains no embeddings.
    pub fn is_empty(&self) -> Result<bool> {
        self.db
            .is_empty()
            .map_err(|e| RuvNeuralError::Memory(format!("RuVector is_empty failed: {e}")))
    }

    fn validate_embedding(&self, embedding: &NeuralEmbedding) -> Result<()> {
        if embedding.dimension != self.dimension {
            return Err(RuvNeuralError::DimensionMismatch {
                expected: self.dimension,
                got: embedding.dimension,
            });
        }
        if !embedding.timestamp.is_finite() {
            return Err(RuvNeuralError::Memory(
                "embedding timestamp must be finite for RuVector indexing".into(),
            ));
        }
        if embedding.vector.iter().any(|v| !v.is_finite()) {
            return Err(RuvNeuralError::Memory(
                "embedding vector must be finite for RuVector indexing".into(),
            ));
        }
        Ok(())
    }
}

impl NeuralMemory for RuvectorMemoryStore {
    fn store(&mut self, embedding: &NeuralEmbedding) -> Result<()> {
        self.insert(embedding)?;
        Ok(())
    }

    fn query_nearest(
        &self,
        embedding: &NeuralEmbedding,
        k: usize,
    ) -> Result<Vec<NeuralEmbedding>> {
        self.validate_embedding(embedding)?;
        let results = self
            .db
            .search(SearchQuery {
                vector: to_f32_vector(&embedding.vector)?,
                k,
                filter: None,
                ef_search: None,
            })
            .map_err(|e| RuvNeuralError::Memory(format!("RuVector search failed: {e}")))?;

        results
            .into_iter()
            .filter_map(|result| result.vector)
            .map(|vector| {
                let values = vector.into_iter().map(f64::from).collect();
                NeuralEmbedding::new(values, embedding.timestamp, embedding.metadata.clone())
            })
            .collect()
    }

    fn query_by_state(&self, state: CognitiveState) -> Result<Vec<NeuralEmbedding>> {
        let _ = state;
        Err(RuvNeuralError::Memory(
            "RuVector metadata-filtered state reconstruction is not available yet".into(),
        ))
    }
}

fn to_f32_vector(vector: &[f64]) -> Result<Vec<f32>> {
    vector
        .iter()
        .map(|value| {
            if !value.is_finite() || *value > f32::MAX as f64 || *value < f32::MIN as f64 {
                Err(RuvNeuralError::Memory(
                    "embedding contains a value outside f32 range".into(),
                ))
            } else {
                Ok(*value as f32)
            }
        })
        .collect()
}

fn metadata_map(embedding: &NeuralEmbedding) -> HashMap<String, Value> {
    let mut metadata = HashMap::new();
    metadata.insert("timestamp".to_string(), Value::from(embedding.timestamp));
    metadata.insert(
        "embedding_method".to_string(),
        Value::from(embedding.metadata.embedding_method.clone()),
    );
    metadata.insert(
        "source_atlas".to_string(),
        Value::from(format!("{:?}", embedding.metadata.source_atlas)),
    );
    if let Some(subject_id) = &embedding.metadata.subject_id {
        metadata.insert("subject_id".to_string(), Value::from(subject_id.clone()));
    }
    if let Some(session_id) = &embedding.metadata.session_id {
        metadata.insert("session_id".to_string(), Value::from(session_id.clone()));
    }
    if let Some(state) = embedding.metadata.cognitive_state {
        metadata.insert("cognitive_state".to_string(), Value::from(format!("{state:?}")));
    }
    metadata
}

fn embedding_id(embedding: &NeuralEmbedding) -> String {
    let subject = embedding
        .metadata
        .subject_id
        .as_deref()
        .unwrap_or("unknown-subject");
    let session = embedding
        .metadata
        .session_id
        .as_deref()
        .unwrap_or("unknown-session");
    format!("{subject}:{session}:{:.6}", embedding.timestamp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruv_neural_core::brain::Atlas;
    use ruv_neural_core::embedding::EmbeddingMetadata;

    fn embedding(vector: Vec<f64>, timestamp: f64) -> NeuralEmbedding {
        NeuralEmbedding::new(
            vector,
            timestamp,
            EmbeddingMetadata {
                subject_id: Some("sub-01".to_string()),
                session_id: Some("tg-01".to_string()),
                cognitive_state: Some(CognitiveState::Focused),
                source_atlas: Atlas::Custom(3),
                embedding_method: "topology".to_string(),
            },
        )
        .unwrap()
    }

    #[test]
    fn insert_and_search_returns_indexed_id() {
        let store = RuvectorMemoryStore::new(3, 32).unwrap();
        let emb = embedding(vec![1.0, 0.0, 0.0], 1.0);
        let id = store.insert(&emb).unwrap();
        let hits = store.search_ids(&emb, 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, id);
    }

    #[test]
    fn rejects_non_finite_values() {
        let store = RuvectorMemoryStore::new(3, 32).unwrap();
        let emb = embedding(vec![1.0, f64::NAN, 0.0], 1.0);
        assert!(store.insert(&emb).is_err());
    }
}
