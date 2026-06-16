//! Persist a trained classifier into the signed RVF container substrate.
//!
//! This is the capstone that ties the learned [`LogisticRegression`] decoder to
//! the RVF format work: a model is stored as a `MODEL` segment (with a `META`
//! descriptor) inside an [`RvfContainer`], so it can be signed with Ed25519
//! ([`ruv_neural_core::rvf_witness::sign_container`]), shipped as one self-
//! describing `.rvf` file, then loaded and **verified** before use. The
//! container's CRC32C + content-hash catch corruption; the `CRYPTO` signature
//! catches tampering.

use serde::{Deserialize, Serialize};

use ruv_neural_core::error::{Result, RuvNeuralError};
use ruv_neural_core::rvf_container::{RvfContainer, SegmentType, FLAG_SEALED};

use crate::logistic::LogisticRegression;

/// Descriptor stored in the `META` segment alongside the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelDescriptor {
    kind: String,
    num_features: usize,
    format_version: u32,
}

/// Build an RVF container holding `model` as a `MODEL` segment plus a `META`
/// descriptor. The returned container is unsigned; sign it with
/// [`ruv_neural_core::rvf_witness::sign_container`] before distribution.
///
/// # Errors
/// Returns an error if the model cannot be serialized.
pub fn model_to_container(model: &LogisticRegression) -> Result<RvfContainer> {
    let descriptor = ModelDescriptor {
        kind: "logistic-regression".into(),
        num_features: model.num_features(),
        format_version: 1,
    };
    let meta = serde_json::to_vec(&descriptor)
        .map_err(|e| RuvNeuralError::Serialization(e.to_string()))?;
    let model_bytes =
        serde_json::to_vec(model).map_err(|e| RuvNeuralError::Serialization(e.to_string()))?;

    let mut container = RvfContainer::new();
    container.add_segment(SegmentType::Meta, FLAG_SEALED, meta);
    container.add_segment(SegmentType::Model, FLAG_SEALED, model_bytes);
    Ok(container)
}

/// Load a [`LogisticRegression`] from a container's `MODEL` segment.
///
/// Note: this only checks structural integrity. To trust the model, also call
/// [`ruv_neural_core::rvf_witness::verify_container_signature`] and
/// [`RvfContainer::verify_integrity`] first.
///
/// # Errors
/// Returns an error if the `MODEL` segment is missing or malformed.
pub fn container_to_model(container: &RvfContainer) -> Result<LogisticRegression> {
    let seg = container.find(SegmentType::Model).ok_or_else(|| {
        RuvNeuralError::Serialization("RVF container missing MODEL segment".into())
    })?;
    serde_json::from_slice(&seg.payload)
        .map_err(|e| RuvNeuralError::Serialization(format!("MODEL decode: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logistic::TrainConfig;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;
    use ruv_neural_core::rvf_witness::{sign_container, verify_container_signature};

    fn trained() -> (LogisticRegression, Vec<Vec<f64>>) {
        // A small separable problem so the model is non-trivial.
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..200 {
            let a = (i as f64 / 200.0) * 4.0 - 2.0;
            let b = ((i * 7 % 200) as f64 / 200.0) * 4.0 - 2.0;
            x.push(vec![a, b]);
            y.push(u8::from(a + b > 0.0));
        }
        let (model, _) = LogisticRegression::fit(&x, &y, &TrainConfig::default()).unwrap();
        (model, x)
    }

    #[test]
    fn model_roundtrips_through_container() {
        let (model, probe) = trained();
        let container = model_to_container(&model).unwrap();
        let bytes = container.to_bytes();

        let back = RvfContainer::from_bytes(&bytes).unwrap();
        back.verify_integrity().unwrap();
        let loaded = container_to_model(&back).unwrap();

        assert_eq!(loaded.num_features(), model.num_features());
        // Predictions are identical after the round trip.
        for row in probe.iter().take(50) {
            assert_eq!(loaded.predict(row), model.predict(row));
            assert!((loaded.predict_proba(row) - model.predict_proba(row)).abs() < 1e-12);
        }
    }

    #[test]
    fn signed_model_verifies_and_detects_tampering() {
        let (model, _) = trained();
        let mut container = model_to_container(&model).unwrap();
        let key = SigningKey::generate(&mut OsRng);
        sign_container(&mut container, &key);

        // Ships and reloads with a valid signature.
        let reloaded = RvfContainer::from_bytes(&container.to_bytes()).unwrap();
        assert!(verify_container_signature(&reloaded).unwrap());
        assert!(container_to_model(&reloaded).is_ok());

        // Tamper with the model weights (and fix CRC/content-hash so only the
        // signature catches it) → verification must fail.
        let mi = container
            .segments
            .iter()
            .position(|s| s.header.seg_type == SegmentType::Model)
            .unwrap();
        container.segments[mi].payload[10] ^= 0xFF;
        let p = container.segments[mi].payload.clone();
        container.segments[mi].header.crc32c = ruv_neural_core::rvf_container::crc32c(&p);
        container.segments[mi].header.content_hash = {
            use sha2::{Digest, Sha256};
            let d = Sha256::digest(&p);
            let mut h = [0u8; 16];
            h.copy_from_slice(&d[..16]);
            h
        };
        assert!(!verify_container_signature(&container).unwrap());
    }

    #[test]
    fn missing_model_segment_errors() {
        let container = RvfContainer::new();
        assert!(container_to_model(&container).is_err());
    }
}
