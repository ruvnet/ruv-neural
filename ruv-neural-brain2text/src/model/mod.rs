//! Trainable acoustic models — the per-keystroke character classifiers.
//!
//! All models implement [`AcousticModel`] (inference) and are produced by
//! [`AcousticEnum::train`] (dispatch by [`ModelKind`]). Crucially, every model
//! is `serde`-serializable: the trained parameters *are* the distributable
//! artifact. These are clean-room, dependency-free implementations (MIT/Apache),
//! so weights trained on a permissively-licensed corpus can be distributed
//! freely; weights trained on CC BY-NC data inherit that data's terms (see
//! `MODEL_CARD.md` / `WEIGHTS_LICENSE`).
//!
//! - [`prototype::PrototypeModel`] — nearest-centroid (no gradient training)
//! - [`linear::LinearSoftmax`] — multinomial logistic regression (SGD)
//! - [`mlp::Mlp`] — one hidden layer, ReLU + softmax (SGD + backprop)

pub mod linear;
pub mod mlp;
pub mod prototype;

use serde::{Deserialize, Serialize};

pub use linear::LinearSoftmax;
pub use mlp::Mlp;
pub use prototype::PrototypeModel;

/// Inference interface shared by all acoustic models.
pub trait AcousticModel {
    /// Characters the model can emit.
    fn vocabulary(&self) -> Vec<char>;

    /// Per-character log-probabilities for one epoch's features, sorted by
    /// descending probability.
    fn logprobs(&self, features: &[f64]) -> Vec<(char, f64)>;

    /// Greedy single-character prediction.
    fn predict(&self, features: &[f64]) -> Option<char> {
        self.logprobs(features).first().map(|(c, _)| *c)
    }
}

/// Which acoustic model family to train.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelKind {
    /// Nearest-centroid prototype classifier.
    Prototype,
    /// Multinomial logistic regression (linear softmax).
    Linear,
    /// One-hidden-layer perceptron.
    Mlp,
}

/// Training hyperparameters for the gradient-trained models.
#[derive(Debug, Clone, Copy)]
pub struct TrainParams {
    /// Learning rate.
    pub learning_rate: f64,
    /// Number of full passes over the training set.
    pub epochs: usize,
    /// L2 regularization strength.
    pub l2: f64,
    /// Hidden layer width (MLP only).
    pub hidden_size: usize,
    /// RNG seed for shuffling/initialization.
    pub seed: u64,
}

impl Default for TrainParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.5,
            epochs: 80,
            l2: 1e-4,
            hidden_size: 32,
            seed: 0xACED,
        }
    }
}

/// A trained, serializable acoustic model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AcousticEnum {
    /// Prototype model.
    Prototype(PrototypeModel),
    /// Linear softmax model.
    Linear(LinearSoftmax),
    /// MLP model.
    Mlp(Mlp),
}

impl AcousticEnum {
    /// Train the requested model family on `(features, character)` samples.
    pub fn train(kind: ModelKind, samples: &[(Vec<f64>, char)], params: &TrainParams) -> Self {
        match kind {
            ModelKind::Prototype => AcousticEnum::Prototype(PrototypeModel::train(
                samples.iter().map(|(f, c)| (f.as_slice(), *c)),
            )),
            ModelKind::Linear => AcousticEnum::Linear(LinearSoftmax::train(samples, params)),
            ModelKind::Mlp => AcousticEnum::Mlp(Mlp::train(samples, params)),
        }
    }

    /// The model family.
    pub fn kind(&self) -> ModelKind {
        match self {
            AcousticEnum::Prototype(_) => ModelKind::Prototype,
            AcousticEnum::Linear(_) => ModelKind::Linear,
            AcousticEnum::Mlp(_) => ModelKind::Mlp,
        }
    }
}

impl AcousticModel for AcousticEnum {
    fn vocabulary(&self) -> Vec<char> {
        match self {
            AcousticEnum::Prototype(m) => m.vocabulary(),
            AcousticEnum::Linear(m) => m.vocabulary(),
            AcousticEnum::Mlp(m) => m.vocabulary(),
        }
    }

    fn logprobs(&self, features: &[f64]) -> Vec<(char, f64)> {
        match self {
            AcousticEnum::Prototype(m) => m.logprobs(features),
            AcousticEnum::Linear(m) => m.logprobs(features),
            AcousticEnum::Mlp(m) => m.logprobs(features),
        }
    }
}

/// Replace a non-finite value (NaN/±inf) with zero. Used to guarantee trained
/// parameters are always finite — important because `serde_json` serializes
/// non-finite floats as `null`, which then fails to deserialize.
pub(crate) fn finite_or_zero(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Clamp a logit into a range that keeps `exp` from overflowing during training.
pub(crate) fn clamp_logit(x: f64) -> f64 {
    x.clamp(-60.0, 60.0)
}

/// Numerically stable log-softmax of a logit vector (in place semantics: returns
/// a new vector of log-probabilities).
pub(crate) fn log_softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let denom: f64 = logits.iter().map(|l| (l - max).exp()).sum();
    let log_denom = denom.ln();
    logits.iter().map(|l| (l - max) - log_denom).collect()
}

/// Per-feature standardization statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Standardizer {
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl Standardizer {
    /// Fit mean/std over the training features.
    pub(crate) fn fit(samples: &[(Vec<f64>, char)], dim: usize) -> Self {
        let mut mean = vec![0.0; dim];
        let n = samples.len().max(1) as f64;
        for (f, _) in samples {
            for i in 0..dim.min(f.len()) {
                mean[i] += f[i];
            }
        }
        for m in &mut mean {
            *m /= n;
        }
        let mut var = vec![0.0; dim];
        for (f, _) in samples {
            for i in 0..dim.min(f.len()) {
                let d = f[i] - mean[i];
                var[i] += d * d;
            }
        }
        let std: Vec<f64> = var
            .iter()
            .map(|v| {
                let s = (v / n).sqrt();
                if s < 1e-8 {
                    1.0
                } else {
                    s
                }
            })
            .collect();
        Standardizer { mean, std }
    }

    /// Standardize a feature vector into a provided buffer.
    pub(crate) fn apply(&self, features: &[f64], out: &mut [f64]) {
        for i in 0..out.len() {
            let x = features.get(i).copied().unwrap_or(0.0);
            out[i] = (x - self.mean[i]) / self.std[i];
        }
    }

    /// Dimensionality.
    pub(crate) fn dim(&self) -> usize {
        self.mean.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_softmax_normalizes() {
        let lp = log_softmax(&[1.0, 2.0, 3.0]);
        let p: f64 = lp.iter().map(|l| l.exp()).sum();
        assert!((p - 1.0).abs() < 1e-9);
    }

    #[test]
    fn enum_dispatches_and_serializes() {
        let samples: Vec<(Vec<f64>, char)> = vec![
            (vec![1.0, 0.0], 'a'),
            (vec![0.9, 0.1], 'a'),
            (vec![0.0, 1.0], 'b'),
            (vec![0.1, 0.9], 'b'),
        ];
        for kind in [ModelKind::Prototype, ModelKind::Linear, ModelKind::Mlp] {
            let m = AcousticEnum::train(kind, &samples, &TrainParams::default());
            assert_eq!(m.kind(), kind);
            assert_eq!(m.predict(&[0.95, 0.05]), Some('a'));
            // Round-trip through JSON.
            let json = serde_json::to_string(&m).unwrap();
            let back: AcousticEnum = serde_json::from_str(&json).unwrap();
            assert_eq!(back.predict(&[0.95, 0.05]), Some('a'));
        }
    }
}
