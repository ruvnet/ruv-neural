//! Pluggable foundation-model embedding backend (ADR-0015).
//!
//! Self-supervised EEG/biosignal foundation models (LaBraM, NeuroLM, REVE, …)
//! produce rich representations, but they are heavy, often edge-infeasible, and —
//! per the critical literature — **not a guaranteed win** over compact baselines.
//! ADR-0015 therefore keeps the lightweight deterministic embeddings as the
//! default and exposes FMs as an *optional, inference-only* backend behind this
//! seam, producing a standard method-tagged [`NeuralEmbedding`] so downstream
//! code (distance, RVF export, the controller) stays method-agnostic.
//!
//! Real model backends (e.g. an ONNX-exported LaBraM/REVE) are gated behind the
//! `fm` Cargo feature so the core stays dependency-light. [`ReferenceFoundationEmbedder`]
//! is a dependency-free, deterministic **reference scaffold** — it is *not* a
//! trained model and makes no accuracy claim; per ADR-0015 point 4, any real FM
//! backend must beat the lightweight baselines **out-of-sample** before it is
//! promoted out of "Proposed".

use ruv_neural_core::brain::Atlas;
use ruv_neural_core::embedding::{EmbeddingMetadata, NeuralEmbedding};
use ruv_neural_core::error::{Result, RuvNeuralError};
use ruv_neural_core::signal::MultiChannelTimeSeries;

/// An inference-only foundation-model embedding backend.
///
/// Implementors map a windowed multichannel signal to a fixed-dimensional
/// embedding. They never train in-tree; they consume exported model outputs.
pub trait FoundationEmbedder {
    /// Method tag recorded on the embedding, e.g. `"foundation:labram"`.
    fn method_tag(&self) -> &str;

    /// Output embedding dimensionality.
    fn embedding_dim(&self) -> usize;

    /// SPDX-style license identifier of the backing model (ADR-0015 point 5).
    fn license(&self) -> &str;

    /// Run inference over a windowed multichannel signal, returning the raw
    /// embedding vector (length [`Self::embedding_dim`]).
    fn infer(&self, signal: &MultiChannelTimeSeries) -> Result<Vec<f64>>;

    /// Produce a method-tagged [`NeuralEmbedding`] ready for RVF export / memory.
    fn embed(
        &self,
        signal: &MultiChannelTimeSeries,
        timestamp: f64,
        subject_id: Option<String>,
    ) -> Result<NeuralEmbedding> {
        let vector = self.infer(signal)?;
        let meta = EmbeddingMetadata {
            subject_id,
            session_id: None,
            cognitive_state: None,
            source_atlas: Atlas::Custom(self.embedding_dim()),
            embedding_method: self.method_tag().to_string(),
        };
        NeuralEmbedding::new(vector, timestamp, meta)
    }
}

/// A dependency-free, deterministic **reference** backend.
///
/// It pools simple per-channel summary statistics (mean, variance, mean-absolute
/// amplitude, and line-length) into a fixed-dimensional, L2-normalized vector. It
/// exists to exercise the [`FoundationEmbedder`] seam end-to-end and to give real
/// model backends a drop-in target — it is **not** a trained foundation model and
/// asserts no representational advantage.
#[derive(Debug, Clone)]
pub struct ReferenceFoundationEmbedder {
    dim: usize,
}

impl ReferenceFoundationEmbedder {
    /// Create a reference backend producing `dim`-dimensional embeddings.
    pub fn new(dim: usize) -> Self {
        Self { dim: dim.max(1) }
    }

    /// Per-channel summary statistics: `[mean, variance, mean_abs, line_length]`.
    fn channel_features(samples: &[f64]) -> [f64; 4] {
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let mean_abs = samples.iter().map(|x| x.abs()).sum::<f64>() / n;
        let line_length = samples
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f64>();
        [mean, variance, mean_abs, line_length]
    }
}

impl Default for ReferenceFoundationEmbedder {
    fn default() -> Self {
        Self::new(64)
    }
}

impl FoundationEmbedder for ReferenceFoundationEmbedder {
    fn method_tag(&self) -> &str {
        "foundation:reference"
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }

    fn license(&self) -> &str {
        // Not a model; nothing licensable. Real backends report their license.
        "NONE (reference scaffold)"
    }

    fn infer(&self, signal: &MultiChannelTimeSeries) -> Result<Vec<f64>> {
        if signal.num_channels == 0 || signal.num_samples < 2 {
            return Err(RuvNeuralError::Embedding(
                "foundation embedder needs ≥1 channel and ≥2 samples".into(),
            ));
        }

        // Build a deterministic raw feature vector: 4 stats per channel.
        let mut feats: Vec<f64> = Vec::with_capacity(signal.num_channels * 4);
        for ch in 0..signal.num_channels {
            let samples = signal.channel(ch)?;
            feats.extend_from_slice(&Self::channel_features(samples));
        }

        // Pool the (variable-length) feature vector into a fixed `dim` by
        // averaging contiguous buckets — deterministic and montage-agnostic.
        let mut out = vec![0.0f64; self.dim];
        for (i, slot) in out.iter_mut().enumerate() {
            let start = i * feats.len() / self.dim;
            let end = ((i + 1) * feats.len() / self.dim).max(start + 1).min(feats.len());
            let bucket = &feats[start..end];
            *slot = bucket.iter().sum::<f64>() / bucket.len() as f64;
        }

        // L2-normalize so embeddings live on the unit sphere (cosine-friendly).
        let norm = out.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in out.iter_mut() {
                *x /= norm;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(seed: f64) -> MultiChannelTimeSeries {
        let ch0: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1 + seed).sin()).collect();
        let ch1: Vec<f64> = (0..128).map(|i| (i as f64 * 0.2 + seed).cos()).collect();
        MultiChannelTimeSeries::new(vec![ch0, ch1], 128.0, 0.0).unwrap()
    }

    #[test]
    fn reference_embedder_is_deterministic() {
        let fm = ReferenceFoundationEmbedder::new(32);
        let a = fm.infer(&signal(0.0)).unwrap();
        let b = fm.infer(&signal(0.0)).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 32);
    }

    #[test]
    fn output_is_unit_norm_and_tagged() {
        let fm = ReferenceFoundationEmbedder::new(48);
        let emb = fm.embed(&signal(1.0), 5.0, Some("sub-01".into())).unwrap();
        assert_eq!(emb.dimension, 48);
        assert_eq!(emb.metadata.embedding_method, "foundation:reference");
        assert_eq!(emb.metadata.subject_id.as_deref(), Some("sub-01"));
        assert!((emb.norm() - 1.0).abs() < 1e-9);
        assert!(fm.license().contains("scaffold"));
    }

    #[test]
    fn distinct_signals_give_distinct_embeddings() {
        let fm = ReferenceFoundationEmbedder::new(32);
        let a = fm.infer(&signal(0.0)).unwrap();
        let b = fm.infer(&signal(3.0)).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn exports_through_rvf_seam() {
        use crate::rvf_export::{export_rvf, import_rvf};
        let fm = ReferenceFoundationEmbedder::new(16);
        let e = fm.embed(&signal(2.0), 0.0, None).unwrap();
        let path = "/tmp/ruv_neural_fm_seam_test.rvf";
        export_rvf(&[e.clone()], path).unwrap();
        let back = import_rvf(path).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].metadata.embedding_method, "foundation:reference");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn degenerate_signal_errors() {
        let fm = ReferenceFoundationEmbedder::new(8);
        let one = MultiChannelTimeSeries::new(vec![vec![1.0]], 1.0, 0.0).unwrap();
        assert!(fm.infer(&one).is_err());
    }
}
