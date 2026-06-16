//! Browser-callable RVF verification and on-device model inference.
//!
//! Lets the web console (ADR-0014) load a signed `.rvf` produced by the CLI /
//! library, **verify it client-side** (CRC32C + content-hash integrity and the
//! Ed25519 `CRYPTO` signature), and run a trained classifier entirely in the
//! browser — no server round-trip, consistent with the local-first posture.
//!
//! The `*_inner` functions hold the pure logic and are unit-tested on the host;
//! the `#[wasm_bindgen]` wrappers expose them to JavaScript.

use serde::Serialize;
use wasm_bindgen::prelude::*;

use ruv_neural_core::rvf_container::RvfContainer;
use ruv_neural_core::rvf_witness::verify_container_signature;
use ruv_neural_decoder::container_to_model;

/// Result of inspecting a `.rvf` container, returned to JS as an object.
#[derive(Serialize)]
pub struct RvfReport {
    /// Number of segments parsed.
    pub num_segments: usize,
    /// Whether CRC32C + content-hash integrity holds.
    pub integrity_ok: bool,
    /// `"valid"`, `"invalid"`, or `"none"`.
    pub signature: String,
    /// Feature count if the container carries a loadable model, else `null`.
    pub model_features: Option<usize>,
}

/// Pure logic for [`verify_rvf`]: parse + verify a container.
pub fn rvf_report_inner(bytes: &[u8]) -> Result<RvfReport, String> {
    // `from_bytes` already validates CRC32C + content hash while parsing, so a
    // success here means integrity holds.
    let container = RvfContainer::from_bytes(bytes).map_err(|e| e.to_string())?;
    let integrity_ok = container.verify_integrity().is_ok();
    let signature = match verify_container_signature(&container) {
        Ok(true) => "valid",
        Ok(false) => "invalid",
        Err(_) => "none",
    }
    .to_string();
    let model_features = container_to_model(&container).ok().map(|m| m.num_features());
    Ok(RvfReport {
        num_segments: container.segments.len(),
        integrity_ok,
        signature,
        model_features,
    })
}

/// Pure logic for [`rvf_model_predict_proba`]: verify a signed model and score.
pub fn model_predict_inner(model_bytes: &[u8], features: &[f64]) -> Result<f64, String> {
    let container = RvfContainer::from_bytes(model_bytes).map_err(|e| e.to_string())?;
    container.verify_integrity().map_err(|e| e.to_string())?;
    if let Ok(false) = verify_container_signature(&container) {
        return Err("model signature is invalid".into());
    }
    let model = container_to_model(&container).map_err(|e| e.to_string())?;
    if features.len() != model.num_features() {
        return Err(format!(
            "feature count mismatch: got {}, model expects {}",
            features.len(),
            model.num_features()
        ));
    }
    Ok(model.predict_proba(features))
}

/// Inspect a `.rvf` container in the browser: integrity, signature, and whether
/// it carries a model. Returns a JS object `{ num_segments, integrity_ok,
/// signature, model_features }`.
#[wasm_bindgen]
pub fn verify_rvf(bytes: &[u8]) -> Result<JsValue, JsError> {
    let report = rvf_report_inner(bytes).map_err(|e| JsError::new(&e))?;
    serde_wasm_bindgen::to_value(&report).map_err(|e| JsError::new(&e.to_string()))
}

/// Verify a signed `.rvf` model and return the positive-class probability for a
/// feature row. Errors if integrity fails, the signature is invalid, or the
/// feature count is wrong.
#[wasm_bindgen]
pub fn rvf_model_predict_proba(model_bytes: &[u8], features: Vec<f64>) -> Result<f64, JsError> {
    model_predict_inner(model_bytes, &features).map_err(|e| JsError::new(&e))
}

/// As [`rvf_model_predict_proba`] but returns the 0/1 label at threshold 0.5.
#[wasm_bindgen]
pub fn rvf_model_predict_label(model_bytes: &[u8], features: Vec<f64>) -> Result<u8, JsError> {
    let p = model_predict_inner(model_bytes, &features).map_err(|e| JsError::new(&e))?;
    Ok(u8::from(p >= 0.5))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruv_neural_core::rvf_witness::sign_container_ephemeral;
    use ruv_neural_decoder::logistic::{LogisticRegression, TrainConfig};
    use ruv_neural_decoder::model_to_container;

    fn signed_model_bytes() -> (Vec<u8>, Vec<f64>) {
        // Train a tiny separable model.
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..200 {
            let a = (i as f64 / 200.0) * 4.0 - 2.0;
            let b = ((i * 7 % 200) as f64 / 200.0) * 4.0 - 2.0;
            x.push(vec![a, b]);
            y.push(u8::from(a + b > 0.0));
        }
        let (model, _) = LogisticRegression::fit(&x, &y, &TrainConfig::default()).unwrap();
        let mut c = model_to_container(&model).unwrap();
        sign_container_ephemeral(&mut c);
        (c.to_bytes(), vec![2.0, 2.0])
    }

    #[test]
    fn report_on_signed_model() {
        let (bytes, _) = signed_model_bytes();
        let r = rvf_report_inner(&bytes).unwrap();
        assert!(r.integrity_ok);
        assert_eq!(r.signature, "valid");
        assert_eq!(r.model_features, Some(2));
        assert!(r.num_segments >= 3); // META + MODEL + CRYPTO
    }

    #[test]
    fn predicts_from_signed_model() {
        let (bytes, feats) = signed_model_bytes();
        let p = model_predict_inner(&bytes, &feats).unwrap();
        assert!((0.0..=1.0).contains(&p));
        // (2, 2) is firmly on the positive side of x0 + x1 > 0.
        assert!(p > 0.9);
    }

    #[test]
    fn rejects_tampered_model() {
        let (mut bytes, feats) = signed_model_bytes();
        // Corrupt a deep payload byte → parse-time CRC failure.
        let i = bytes.len() / 2;
        bytes[i] ^= 0xFF;
        assert!(model_predict_inner(&bytes, &feats).is_err());
    }

    #[test]
    fn rejects_wrong_feature_count() {
        let (bytes, _) = signed_model_bytes();
        assert!(model_predict_inner(&bytes, &[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn garbage_bytes_error_cleanly() {
        assert!(rvf_report_inner(&[1, 2, 3, 4]).is_err());
        assert!(model_predict_inner(&[0u8; 8], &[1.0]).is_err());
    }
}
