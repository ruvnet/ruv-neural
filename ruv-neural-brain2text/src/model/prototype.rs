//! Nearest-centroid ("prototype") character classifier.
//!
//! Learns one centroid feature vector per character and scores an epoch by
//! (negative, scaled) squared distance to each centroid, turned into a
//! log-probability via softmax. No gradient training — a fast, robust baseline.

use serde::{Deserialize, Serialize};

use super::AcousticModel;

/// A trained prototype classifier (serializable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrototypeModel {
    centroids: Vec<(char, Vec<f64>)>,
    scale: f64,
}

impl PrototypeModel {
    /// Train from `(features, character)` samples.
    pub fn train<'a>(samples: impl IntoIterator<Item = (&'a [f64], char)>) -> Self {
        use std::collections::BTreeMap;
        let mut sums: BTreeMap<char, (Vec<f64>, usize)> = BTreeMap::new();
        for (feats, ch) in samples {
            let entry = sums.entry(ch).or_insert_with(|| (vec![0.0; feats.len()], 0));
            if entry.0.len() < feats.len() {
                entry.0.resize(feats.len(), 0.0);
            }
            for (acc, v) in entry.0.iter_mut().zip(feats) {
                *acc += v;
            }
            entry.1 += 1;
        }
        let centroids: Vec<(char, Vec<f64>)> = sums
            .into_iter()
            .map(|(ch, (mut sum, n))| {
                if n > 0 {
                    for v in &mut sum {
                        *v /= n as f64;
                    }
                }
                (ch, sum)
            })
            .collect();

        let mut scale = 1.0;
        if centroids.len() > 1 {
            let mut total = 0.0;
            let mut count = 0usize;
            for i in 0..centroids.len() {
                for j in (i + 1)..centroids.len() {
                    total += sq_dist(&centroids[i].1, &centroids[j].1);
                    count += 1;
                }
            }
            if count > 0 && total > 0.0 {
                scale = total / count as f64;
            }
        }
        PrototypeModel { centroids, scale }
    }
}

impl AcousticModel for PrototypeModel {
    fn vocabulary(&self) -> Vec<char> {
        self.centroids.iter().map(|(c, _)| *c).collect()
    }

    fn logprobs(&self, features: &[f64]) -> Vec<(char, f64)> {
        if self.centroids.is_empty() {
            return Vec::new();
        }
        let logits: Vec<(char, f64)> = self
            .centroids
            .iter()
            .map(|(c, cen)| (*c, -sq_dist(features, cen) / self.scale))
            .collect();
        let max = logits.iter().map(|(_, l)| *l).fold(f64::NEG_INFINITY, f64::max);
        let denom: f64 = logits.iter().map(|(_, l)| (l - max).exp()).sum();
        let log_denom = denom.ln();
        let mut out: Vec<(char, f64)> = logits
            .into_iter()
            .map(|(c, l)| (c, (l - max) - log_denom))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut s = 0.0;
    for i in 0..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_separable_classes() {
        let s: Vec<(Vec<f64>, char)> = vec![
            (vec![1.0, 0.0], 'a'),
            (vec![0.9, 0.1], 'a'),
            (vec![0.0, 1.0], 'b'),
            (vec![0.1, 0.9], 'b'),
        ];
        let dec = PrototypeModel::train(s.iter().map(|(f, c)| (f.as_slice(), *c)));
        assert_eq!(dec.predict(&[0.95, 0.05]), Some('a'));
        assert_eq!(dec.predict(&[0.05, 0.95]), Some('b'));
    }

    #[test]
    fn empty_is_safe() {
        let dec = PrototypeModel::train(std::iter::empty::<(&[f64], char)>());
        assert!(dec.predict(&[1.0]).is_none());
    }
}
