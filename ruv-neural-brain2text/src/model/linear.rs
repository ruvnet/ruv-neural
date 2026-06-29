//! Multinomial logistic regression (linear softmax) trained with mini-batch
//! SGD on standardized features. A genuine *trained* model: the weight matrix
//! and bias are learned by gradient descent and are the distributable artifact.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use super::{clamp_logit, finite_or_zero, log_softmax, AcousticModel, Standardizer, TrainParams};

/// Trained linear-softmax classifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearSoftmax {
    /// Class characters (row order of `weights`).
    classes: Vec<char>,
    /// `weights[class][feature]`.
    weights: Vec<Vec<f64>>,
    /// Per-class bias.
    bias: Vec<f64>,
    /// Feature standardizer.
    standardizer: Standardizer,
}

impl LinearSoftmax {
    /// Train on `(features, character)` samples.
    pub fn train(samples: &[(Vec<f64>, char)], params: &TrainParams) -> Self {
        let dim = samples.iter().map(|(f, _)| f.len()).max().unwrap_or(0);
        let mut classes: Vec<char> = samples
            .iter()
            .map(|(_, c)| *c)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        classes.sort_unstable();
        let class_index: std::collections::HashMap<char, usize> =
            classes.iter().enumerate().map(|(i, c)| (*c, i)).collect();

        let standardizer = Standardizer::fit(samples, dim);
        let num_classes = classes.len();

        let mut weights = vec![vec![0.0f64; dim]; num_classes];
        let mut bias = vec![0.0f64; num_classes];

        if num_classes == 0 || dim == 0 || samples.is_empty() {
            return LinearSoftmax {
                classes,
                weights,
                bias,
                standardizer,
            };
        }

        // Pre-standardize all samples once.
        let mut xs: Vec<Vec<f64>> = Vec::with_capacity(samples.len());
        let mut ys: Vec<usize> = Vec::with_capacity(samples.len());
        for (f, c) in samples {
            let mut buf = vec![0.0; dim];
            standardizer.apply(f, &mut buf);
            xs.push(buf);
            ys.push(class_index[c]);
        }

        let mut rng = StdRng::seed_from_u64(params.seed);
        let mut order: Vec<usize> = (0..xs.len()).collect();
        let lr = params.learning_rate;

        for _ in 0..params.epochs {
            order.shuffle(&mut rng);
            for &i in &order {
                let x = &xs[i];
                let y = ys[i];
                // Forward.
                let mut logits = vec![0.0; num_classes];
                for k in 0..num_classes {
                    let mut s = bias[k];
                    for j in 0..dim {
                        s += weights[k][j] * x[j];
                    }
                    logits[k] = clamp_logit(s);
                }
                let logp = log_softmax(&logits);
                // Gradient of cross-entropy: p_k - 1{y==k}.
                for k in 0..num_classes {
                    let p = logp[k].exp();
                    let grad = p - if k == y { 1.0 } else { 0.0 };
                    bias[k] -= lr * grad;
                    let wk = &mut weights[k];
                    for j in 0..dim {
                        let g = grad * x[j] + params.l2 * wk[j];
                        wk[j] -= lr * g;
                    }
                }
            }
        }

        // Guarantee finite parameters (a diverged config must still serialize).
        for row in &mut weights {
            for w in row.iter_mut() {
                *w = finite_or_zero(*w);
            }
        }
        for b in &mut bias {
            *b = finite_or_zero(*b);
        }

        LinearSoftmax {
            classes,
            weights,
            bias,
            standardizer,
        }
    }

    fn raw_logits(&self, features: &[f64]) -> Vec<f64> {
        let dim = self.standardizer.dim();
        let mut x = vec![0.0; dim];
        self.standardizer.apply(features, &mut x);
        self.weights
            .iter()
            .zip(&self.bias)
            .map(|(w, b)| {
                let mut s = *b;
                for j in 0..dim {
                    s += w[j] * x[j];
                }
                s
            })
            .collect()
    }
}

impl AcousticModel for LinearSoftmax {
    fn vocabulary(&self) -> Vec<char> {
        self.classes.clone()
    }

    fn logprobs(&self, features: &[f64]) -> Vec<(char, f64)> {
        if self.classes.is_empty() {
            return Vec::new();
        }
        let logp = log_softmax(&self.raw_logits(features));
        let mut out: Vec<(char, f64)> =
            self.classes.iter().cloned().zip(logp).collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xor_ish() -> Vec<(Vec<f64>, char)> {
        // Linearly separable three-class problem.
        vec![
            (vec![2.0, 0.0], 'a'),
            (vec![1.8, 0.2], 'a'),
            (vec![0.0, 2.0], 'b'),
            (vec![0.2, 1.8], 'b'),
            (vec![-2.0, 0.0], 'c'),
            (vec![-1.8, -0.2], 'c'),
        ]
    }

    #[test]
    fn trains_and_classifies() {
        let m = LinearSoftmax::train(&xor_ish(), &TrainParams::default());
        assert_eq!(m.predict(&[1.9, 0.1]), Some('a'));
        assert_eq!(m.predict(&[0.1, 1.9]), Some('b'));
        assert_eq!(m.predict(&[-1.9, 0.0]), Some('c'));
        let p: f64 = m.logprobs(&[1.9, 0.1]).iter().map(|(_, l)| l.exp()).sum();
        assert!((p - 1.0).abs() < 1e-9);
    }

    #[test]
    fn empty_is_safe() {
        let m = LinearSoftmax::train(&[], &TrainParams::default());
        assert!(m.predict(&[1.0]).is_none());
    }
}
