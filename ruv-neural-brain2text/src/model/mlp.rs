//! One-hidden-layer perceptron (ReLU → softmax) trained with SGD + backprop on
//! standardized features. Captures nonlinear structure the linear model can't.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use super::{clamp_logit, finite_or_zero, log_softmax, AcousticModel, Standardizer, TrainParams};

/// Trained MLP classifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mlp {
    classes: Vec<char>,
    /// `w1[hidden][feature]`.
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    /// `w2[class][hidden]`.
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    standardizer: Standardizer,
}

impl Mlp {
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

        let h = params.hidden_size.max(1);
        let c = classes.len();
        let mut rng = StdRng::seed_from_u64(params.seed);

        // He-ish initialization.
        let scale1 = (2.0 / dim.max(1) as f64).sqrt();
        let scale2 = (2.0 / h as f64).sqrt();
        let mut w1: Vec<Vec<f64>> = (0..h)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0) * scale1).collect())
            .collect();
        let mut b1 = vec![0.0; h];
        let mut w2: Vec<Vec<f64>> = (0..c)
            .map(|_| (0..h).map(|_| rng.gen_range(-1.0..1.0) * scale2).collect())
            .collect();
        let mut b2 = vec![0.0; c];

        if c == 0 || dim == 0 || samples.is_empty() {
            return Mlp {
                classes,
                w1,
                b1,
                w2,
                b2,
                standardizer,
            };
        }

        let mut xs: Vec<Vec<f64>> = Vec::with_capacity(samples.len());
        let mut ys: Vec<usize> = Vec::with_capacity(samples.len());
        for (f, ch) in samples {
            let mut buf = vec![0.0; dim];
            standardizer.apply(f, &mut buf);
            xs.push(buf);
            ys.push(class_index[ch]);
        }

        let lr = params.learning_rate;
        let mut order: Vec<usize> = (0..xs.len()).collect();

        for _ in 0..params.epochs {
            order.shuffle(&mut rng);
            for &i in &order {
                let x = &xs[i];
                let y = ys[i];

                // Forward: hidden pre-activation z1, activation a1 = relu(z1).
                let mut z1 = vec![0.0; h];
                let mut a1 = vec![0.0; h];
                for k in 0..h {
                    let mut s = b1[k];
                    for j in 0..dim {
                        s += w1[k][j] * x[j];
                    }
                    z1[k] = s;
                    a1[k] = s.max(0.0);
                }
                // Output logits.
                let mut logits = vec![0.0; c];
                for k in 0..c {
                    let mut s = b2[k];
                    for j in 0..h {
                        s += w2[k][j] * a1[j];
                    }
                    logits[k] = clamp_logit(s);
                }
                let logp = log_softmax(&logits);

                // Output gradient: p - onehot.
                let mut dlogit = vec![0.0; c];
                for k in 0..c {
                    dlogit[k] = logp[k].exp() - if k == y { 1.0 } else { 0.0 };
                }
                // Backprop into hidden activation.
                let mut da1 = vec![0.0; h];
                for j in 0..h {
                    let mut s = 0.0;
                    for k in 0..c {
                        s += dlogit[k] * w2[k][j];
                    }
                    da1[j] = s;
                }
                // Update w2/b2.
                for k in 0..c {
                    b2[k] -= lr * dlogit[k];
                    for j in 0..h {
                        let g = dlogit[k] * a1[j] + params.l2 * w2[k][j];
                        w2[k][j] -= lr * g;
                    }
                }
                // ReLU gradient, then update w1/b1.
                for k in 0..h {
                    let dz = if z1[k] > 0.0 { da1[k] } else { 0.0 };
                    if dz == 0.0 {
                        continue;
                    }
                    b1[k] -= lr * dz;
                    for j in 0..dim {
                        let g = dz * x[j] + params.l2 * w1[k][j];
                        w1[k][j] -= lr * g;
                    }
                }
            }
        }

        // Guarantee finite parameters (a diverged config must still serialize).
        for row in w1.iter_mut().chain(w2.iter_mut()) {
            for v in row.iter_mut() {
                *v = finite_or_zero(*v);
            }
        }
        for v in b1.iter_mut().chain(b2.iter_mut()) {
            *v = finite_or_zero(*v);
        }

        Mlp {
            classes,
            w1,
            b1,
            w2,
            b2,
            standardizer,
        }
    }

    fn raw_logits(&self, features: &[f64]) -> Vec<f64> {
        let dim = self.standardizer.dim();
        let mut x = vec![0.0; dim];
        self.standardizer.apply(features, &mut x);
        let h = self.b1.len();
        let mut a1 = vec![0.0; h];
        for k in 0..h {
            let mut s = self.b1[k];
            for j in 0..dim {
                s += self.w1[k][j] * x[j];
            }
            a1[k] = s.max(0.0);
        }
        self.w2
            .iter()
            .zip(&self.b2)
            .map(|(w, b)| {
                let mut s = *b;
                for j in 0..h {
                    s += w[j] * a1[j];
                }
                s
            })
            .collect()
    }
}

impl AcousticModel for Mlp {
    fn vocabulary(&self) -> Vec<char> {
        self.classes.clone()
    }

    fn logprobs(&self, features: &[f64]) -> Vec<(char, f64)> {
        if self.classes.is_empty() {
            return Vec::new();
        }
        let logp = log_softmax(&self.raw_logits(features));
        let mut out: Vec<(char, f64)> = self.classes.iter().cloned().zip(logp).collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trains_on_separable_data() {
        let s: Vec<(Vec<f64>, char)> = vec![
            (vec![2.0, 0.0], 'a'),
            (vec![1.8, 0.2], 'a'),
            (vec![0.0, 2.0], 'b'),
            (vec![0.2, 1.8], 'b'),
        ];
        let m = Mlp::train(
            &s,
            &TrainParams {
                epochs: 150,
                ..Default::default()
            },
        );
        assert_eq!(m.predict(&[1.9, 0.1]), Some('a'));
        assert_eq!(m.predict(&[0.1, 1.9]), Some('b'));
    }

    #[test]
    fn empty_is_safe() {
        let m = Mlp::train(&[], &TrainParams::default());
        assert!(m.predict(&[1.0]).is_none());
    }
}
