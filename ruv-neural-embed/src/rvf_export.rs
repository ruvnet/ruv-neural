//! Export neural embeddings to the RuVector File (.rvf) format.
//!
//! The canonical `.rvf` on-disk form is the **binary multi-segment container**
//! ([`ruv_neural_core::rvf_container`]), byte-compatible with RuVector's RVF
//! framing: a sealed `META` directory plus a `VEC` segment that can be stored
//! lossless (`f32`) or quantized (`f16`/`int8`/`binary`). [`export_rvf`] /
//! [`import_rvf`] read and write that container; [`to_rvf_string`] /
//! [`from_rvf_string`] expose a human-readable JSON form for debugging only.

use ruv_neural_core::brain::Atlas;
use ruv_neural_core::embedding::{EmbeddingMetadata, NeuralEmbedding};
use ruv_neural_core::error::{Result, RuvNeuralError};
use ruv_neural_core::rvf_container::{
    container_to_embeddings, embeddings_to_container, RvfContainer,
};
use ruv_neural_core::rvf_quant::VecDType;
use serde::{Deserialize, Serialize};

/// RVF file header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RvfHeader {
    /// Format version string.
    pub version: String,
    /// Number of embeddings in the file.
    pub count: usize,
    /// Embedding dimensionality.
    pub dimension: usize,
    /// Method used to generate embeddings.
    pub method: String,
    /// Optional description.
    pub description: Option<String>,
}

/// A single RVF record (embedding + metadata).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RvfRecord {
    /// Record index.
    pub index: usize,
    /// Timestamp of the source data.
    pub timestamp: f64,
    /// The embedding vector.
    pub values: Vec<f64>,
    /// Optional subject identifier.
    pub subject_id: Option<String>,
    /// Optional session identifier.
    pub session_id: Option<String>,
}

/// Complete RVF document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RvfDocument {
    /// File header.
    pub header: RvfHeader,
    /// Embedding records.
    pub records: Vec<RvfRecord>,
}

/// Export embeddings to a lossless (`f64`) binary RVF container file.
///
/// # Errors
/// Returns an error if the embedding list is empty or if file I/O fails.
pub fn export_rvf(embeddings: &[NeuralEmbedding], path: &str) -> Result<()> {
    export_rvf_quantized(embeddings, path, VecDType::F64)
}

/// Export embeddings to a binary RVF container at a chosen quantization.
///
/// `f16` roughly halves and `int8`/`binary` further shrink the `VEC` segment,
/// at the cost of reconstruction precision.
///
/// # Errors
/// Returns an error if the embedding list is empty or if file I/O fails.
pub fn export_rvf_quantized(
    embeddings: &[NeuralEmbedding],
    path: &str,
    dtype: VecDType,
) -> Result<()> {
    let container = embeddings_to_container(embeddings, dtype)?;
    let bytes = container.to_bytes();
    std::fs::write(path, bytes).map_err(|e| {
        RuvNeuralError::Serialization(format!("Failed to write RVF file '{}': {}", path, e))
    })
}

/// Import embeddings from a binary RVF container file.
///
/// # Errors
/// Returns an error if the file cannot be read, is not a valid container, or
/// fails an integrity check.
pub fn import_rvf(path: &str) -> Result<Vec<NeuralEmbedding>> {
    let bytes = std::fs::read(path).map_err(|e| {
        RuvNeuralError::Serialization(format!("Failed to read RVF file '{}': {}", path, e))
    })?;
    let container = RvfContainer::from_bytes(&bytes)?;
    container.verify_integrity()?;
    container_to_embeddings(&container)
}

/// Serialize embeddings to RVF JSON string (without writing to file).
pub fn to_rvf_string(embeddings: &[NeuralEmbedding]) -> Result<String> {
    if embeddings.is_empty() {
        return Err(RuvNeuralError::Embedding(
            "Cannot serialize empty embedding list".into(),
        ));
    }

    let dimension = embeddings[0].dimension;
    let method = embeddings[0].metadata.embedding_method.clone();

    let header = RvfHeader {
        version: "1.0".to_string(),
        count: embeddings.len(),
        dimension,
        method,
        description: None,
    };

    let records: Vec<RvfRecord> = embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| RvfRecord {
            index: i,
            timestamp: emb.timestamp,
            values: emb.vector.clone(),
            subject_id: emb.metadata.subject_id.clone(),
            session_id: emb.metadata.session_id.clone(),
        })
        .collect();

    let doc = RvfDocument { header, records };

    serde_json::to_string_pretty(&doc)
        .map_err(|e| RuvNeuralError::Serialization(format!("Failed to serialize RVF: {}", e)))
}

/// Deserialize embeddings from an RVF JSON string.
pub fn from_rvf_string(json: &str) -> Result<Vec<NeuralEmbedding>> {
    let doc: RvfDocument = serde_json::from_str(json)
        .map_err(|e| RuvNeuralError::Serialization(format!("Failed to parse RVF: {}", e)))?;

    doc.records
        .into_iter()
        .map(|rec| {
            let meta = EmbeddingMetadata {
                subject_id: rec.subject_id,
                session_id: rec.session_id,
                cognitive_state: None,
                source_atlas: Atlas::Custom(doc.header.dimension),
                embedding_method: doc.header.method.clone(),
            };
            NeuralEmbedding::new(rec.values, rec.timestamp, meta)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_metadata;

    #[test]
    fn test_rvf_string_roundtrip() {
        let embeddings = vec![
            NeuralEmbedding::new(
                vec![1.0, 2.0, 3.0],
                0.0,
                default_metadata("test", Atlas::Custom(3)),
            )
            .unwrap(),
            NeuralEmbedding::new(
                vec![4.0, 5.0, 6.0],
                0.5,
                default_metadata("test", Atlas::Custom(3)),
            )
            .unwrap(),
            NeuralEmbedding::new(
                vec![7.0, 8.0, 9.0],
                1.0,
                default_metadata("test", Atlas::Custom(3)),
            )
            .unwrap(),
        ];

        let json = to_rvf_string(&embeddings).unwrap();
        let restored = from_rvf_string(&json).unwrap();

        assert_eq!(restored.len(), 3);
        for (orig, rest) in embeddings.iter().zip(restored.iter()) {
            assert_eq!(orig.dimension, rest.dimension);
            assert!((orig.timestamp - rest.timestamp).abs() < 1e-10);
            for (a, b) in orig.vector.iter().zip(rest.vector.iter()) {
                assert!((a - b).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_rvf_file_roundtrip() {
        let embeddings = vec![
            NeuralEmbedding::new(
                vec![1.0, -2.5, 3.2],
                10.0,
                default_metadata("spectral", Atlas::Custom(3)),
            )
            .unwrap(),
            NeuralEmbedding::new(
                vec![0.0, 0.0, 0.0],
                10.5,
                default_metadata("spectral", Atlas::Custom(3)),
            )
            .unwrap(),
        ];

        let path = "/tmp/ruv_neural_embed_test.rvf";
        export_rvf(&embeddings, path).unwrap();
        let restored = import_rvf(path).unwrap();

        assert_eq!(restored.len(), 2);
        assert_eq!(restored[0].metadata.embedding_method, "spectral");
        assert!((restored[0].vector[0] - 1.0).abs() < 1e-10);
        assert!((restored[0].vector[1] - (-2.5)).abs() < 1e-10);
        assert!((restored[0].vector[2] - 3.2).abs() < 1e-10);
        assert!((restored[1].timestamp - 10.5).abs() < 1e-10);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_rvf_empty_fails() {
        assert!(to_rvf_string(&[]).is_err());
        assert!(export_rvf(&[], "/tmp/empty.rvf").is_err());
    }

    #[test]
    fn test_rvf_file_is_binary_container() {
        let embeddings = vec![NeuralEmbedding::new(
            vec![1.0, 2.0, 3.0, 4.0],
            0.0,
            default_metadata("spectral", Atlas::Custom(4)),
        )
        .unwrap()];
        let path = "/tmp/ruv_neural_embed_magic_test.rvf";
        export_rvf(&embeddings, path).unwrap();
        let bytes = std::fs::read(path).unwrap();
        // Canonical `.rvf` files start with the RVFS magic, not `{` (JSON).
        assert_eq!(&bytes[0..4], b"RVFS");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_rvf_quantized_roundtrip() {
        let embeddings = vec![
            NeuralEmbedding::new(
                vec![1.0, -2.0, 3.0, -4.0],
                0.0,
                default_metadata("spectral", Atlas::Custom(4)),
            )
            .unwrap(),
            NeuralEmbedding::new(
                vec![0.5, 0.5, -0.5, -0.5],
                1.0,
                default_metadata("spectral", Atlas::Custom(4)),
            )
            .unwrap(),
        ];
        let path = "/tmp/ruv_neural_embed_quant_test.rvf";
        export_rvf_quantized(&embeddings, path, VecDType::F16).unwrap();
        let restored = import_rvf(path).unwrap();
        assert_eq!(restored.len(), 2);
        // f16 keeps these exactly-representable values lossless.
        for (orig, rest) in embeddings.iter().zip(restored.iter()) {
            for (a, b) in orig.vector.iter().zip(rest.vector.iter()) {
                assert!((a - b).abs() < 1e-3);
            }
        }
        let _ = std::fs::remove_file(path);
    }
}
