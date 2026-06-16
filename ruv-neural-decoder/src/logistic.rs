//! Trainable linear classifier (regularized logistic regression).
//!
//! The other decoders in this crate are rule-based or instance-based; this is
//! the first *learned* model — a binary logistic-regression classifier trained
//! by full-batch gradient descent with feature standardization and L2
//! regularization. It is dependency-free (no BLAS/autograd) and deterministic,
//! so training is reproducible and the model serializes cleanly for RVF/JSON
//! storage.
//!
//! Per ADR-0015/0019, any reported accuracy must be **out-of-sample**; see
//! `examples/train_eeg_eye_state.rs` for an honest chronological-split benchmark
//! on the public UCI "EEG Eye State" dataset.

use serde::{Deserialize, Serialize};

use ruv_neural_core::error::{Result, RuvNeuralError};

/// Per-feature standardizer (zero mean, unit variance), fit on training data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScaler {
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl StandardScaler {
    /// Fit a scaler to the rows of `x`.
    pub fn fit(x: &[Vec<f64>]) -> Result<Self> {
        if x.is_empty() {
            return Err(RuvNeuralError::InsufficientData { needed: 1, have: 0 });
        }
        let dim = x[0].len();
        let n = x.len() as f64;
        let mut mean = vec![0.0; dim];
        for row in x {
            for (m, &v) in mean.iter_mut().zip(row.iter()) {
                *m += v;
            }
        }
        for m in mean.iter_mut() {
            *m /= n;
        }
        let mut std = vec![0.0; dim];
        for row in x {
            for (s, (&v, &m)) in std.iter_mut().zip(row.iter().zip(mean.iter())) {
                *s += (v - m).powi(2);
            }
        }
        for s in std.iter_mut() {
            *s = (*s / n).sqrt().max(1e-8);
        }
        Ok(Self { mean, std })
    }

    /// Standardize one row in place into a new vector.
    pub fn transform(&self, row: &[f64]) -> Vec<f64> {
        row.iter()
            .zip(self.mean.iter().zip(self.std.iter()))
            .map(|(&v, (&m, &s))| (v - m) / s)
            .collect()
    }
}

/// Hyperparameters for [`LogisticRegression::fit`].
#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    /// Gradient-descent step size.
    pub learning_rate: f64,
    /// L2 regularization strength (weight decay).
    pub l2: f64,
    /// Number of full-batch passes.
    pub epochs: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            l2: 1e-3,
            epochs: 300,
        }
    }
}

/// Classification metrics for a binary task (positive class = 1).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BinaryMetrics {
    /// Correct / total.
    pub accuracy: f64,
    /// TP / (TP + FP).
    pub precision: f64,
    /// TP / (TP + FN).
    pub recall: f64,
    /// Harmonic mean of precision and recall.
    pub f1: f64,
    /// True positives.
    pub tp: usize,
    /// False positives.
    pub fp: usize,
    /// True negatives.
    pub tn: usize,
    /// False negatives.
    pub fn_: usize,
}

/// A binary logistic-regression classifier with internal standardization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticRegression {
    weights: Vec<f64>,
    bias: f64,
    scaler: StandardScaler,
}

fn sigmoid(z: f64) -> f64 {
    // Numerically stable logistic function.
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

impl LogisticRegression {
    /// Train on standardized features `x` with binary labels `y` (0/1).
    ///
    /// Returns the trained model and the per-epoch mean log-loss history.
    ///
    /// # Errors
    /// Returns an error if the data is empty, ragged, or label/row counts differ.
    pub fn fit(x: &[Vec<f64>], y: &[u8], cfg: &TrainConfig) -> Result<(Self, Vec<f64>)> {
        if x.is_empty() {
            return Err(RuvNeuralError::InsufficientData { needed: 1, have: 0 });
        }
        if x.len() != y.len() {
            return Err(RuvNeuralError::DimensionMismatch {
                expected: x.len(),
                got: y.len(),
            });
        }
        let dim = x[0].len();
        if x.iter().any(|r| r.len() != dim) {
            return Err(RuvNeuralError::Serialization(
                "logistic regression: ragged feature matrix".into(),
            ));
        }

        let scaler = StandardScaler::fit(x)?;
        let xs: Vec<Vec<f64>> = x.iter().map(|r| scaler.transform(r)).collect();
        let n = xs.len() as f64;

        let mut weights = vec![0.0f64; dim];
        let mut bias = 0.0f64;
        let mut history = Vec::with_capacity(cfg.epochs);

        for _ in 0..cfg.epochs {
            let mut grad_w = vec![0.0f64; dim];
            let mut grad_b = 0.0f64;
            let mut loss = 0.0f64;

            for (row, &label) in xs.iter().zip(y.iter()) {
                let z = bias + dot(&weights, row);
                let p = sigmoid(z);
                let err = p - label as f64;
                for (g, &v) in grad_w.iter_mut().zip(row.iter()) {
                    *g += err * v;
                }
                grad_b += err;

                // Clamped cross-entropy for the reported loss.
                let pc = p.clamp(1e-12, 1.0 - 1e-12);
                loss += -(label as f64 * pc.ln() + (1.0 - label as f64) * (1.0 - pc).ln());
            }

            // Average gradients; L2 on weights only (not the bias).
            for (w, g) in weights.iter_mut().zip(grad_w.iter()) {
                let grad = g / n + cfg.l2 * *w;
                *w -= cfg.learning_rate * grad;
            }
            bias -= cfg.learning_rate * (grad_b / n);
            history.push(loss / n);
        }

        Ok((
            Self {
                weights,
                bias,
                scaler,
            },
            history,
        ))
    }

    /// Probability of the positive class for a raw (unscaled) feature row.
    pub fn predict_proba(&self, row: &[f64]) -> f64 {
        let xs = self.scaler.transform(row);
        sigmoid(self.bias + dot(&self.weights, &xs))
    }

    /// Predicted label (1 if probability ≥ `threshold`, else 0).
    pub fn predict_with_threshold(&self, row: &[f64], threshold: f64) -> u8 {
        u8::from(self.predict_proba(row) >= threshold)
    }

    /// Predicted label at the default 0.5 threshold.
    pub fn predict(&self, row: &[f64]) -> u8 {
        self.predict_with_threshold(row, 0.5)
    }

    /// Evaluate the model on a labeled set.
    pub fn evaluate(&self, x: &[Vec<f64>], y: &[u8]) -> BinaryMetrics {
        let (mut tp, mut fp, mut tn, mut fn_) = (0usize, 0usize, 0usize, 0usize);
        for (row, &label) in x.iter().zip(y.iter()) {
            match (self.predict(row), label) {
                (1, 1) => tp += 1,
                (1, 0) => fp += 1,
                (0, 0) => tn += 1,
                (0, 1) => fn_ += 1,
                _ => unreachable!("predict returns 0 or 1"),
            }
        }
        let total = (tp + fp + tn + fn_).max(1) as f64;
        let precision = safe_div(tp as f64, (tp + fp) as f64);
        let recall = safe_div(tp as f64, (tp + fn_) as f64);
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        BinaryMetrics {
            accuracy: (tp + tn) as f64 / total,
            precision,
            recall,
            f1,
            tp,
            fp,
            tn,
            fn_,
        }
    }

    /// Number of input features the model expects.
    pub fn num_features(&self) -> usize {
        self.weights.len()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn safe_div(a: f64, b: f64) -> f64 {
    if b > 0.0 {
        a / b
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A linearly separable 2-D problem: positive when x0 + x1 > 0.
    fn separable(n: usize) -> (Vec<Vec<f64>>, Vec<u8>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..n {
            let a = (i as f64 / n as f64) * 4.0 - 2.0;
            let b = ((i * 7 % n) as f64 / n as f64) * 4.0 - 2.0;
            x.push(vec![a, b]);
            y.push(u8::from(a + b > 0.0));
        }
        (x, y)
    }

    #[test]
    fn scaler_standardizes() {
        let x = vec![vec![1.0, 10.0], vec![3.0, 30.0], vec![5.0, 50.0]];
        let s = StandardScaler::fit(&x).unwrap();
        let t: Vec<Vec<f64>> = x.iter().map(|r| s.transform(r)).collect();
        // Column means ≈ 0.
        let m0: f64 = t.iter().map(|r| r[0]).sum::<f64>() / 3.0;
        assert!(m0.abs() < 1e-9);
    }

    #[test]
    fn learns_separable_problem() {
        let (x, y) = separable(400);
        let cfg = TrainConfig {
            learning_rate: 0.5,
            l2: 1e-4,
            epochs: 500,
        };
        let (model, history) = LogisticRegression::fit(&x, &y, &cfg).unwrap();

        // Loss decreases monotonically-ish: last < first.
        assert!(history.last().unwrap() < &history[0]);

        let m = model.evaluate(&x, &y);
        assert!(m.accuracy > 0.95, "accuracy was {}", m.accuracy);
        assert!(m.f1 > 0.95, "f1 was {}", m.f1);
    }

    #[test]
    fn deterministic_training() {
        let (x, y) = separable(200);
        let cfg = TrainConfig::default();
        let (a, _) = LogisticRegression::fit(&x, &y, &cfg).unwrap();
        let (b, _) = LogisticRegression::fit(&x, &y, &cfg).unwrap();
        // Same data + config ⇒ identical model.
        assert_eq!(a.bias, b.bias);
        assert_eq!(a.weights, b.weights);
    }

    #[test]
    fn serializes_roundtrip() {
        let (x, y) = separable(100);
        let (model, _) = LogisticRegression::fit(&x, &y, &TrainConfig::default()).unwrap();
        let json = serde_json::to_string(&model).unwrap();
        let back: LogisticRegression = serde_json::from_str(&json).unwrap();
        assert_eq!(back.num_features(), 2);
        assert_eq!(back.predict(&[1.0, 1.0]), model.predict(&[1.0, 1.0]));
    }

    #[test]
    fn rejects_mismatched_labels() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let y = vec![1u8];
        assert!(LogisticRegression::fit(&x, &y, &TrainConfig::default()).is_err());
    }
}
