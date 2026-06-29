//! Nearest-centroid ("prototype") per-keystroke character classifier.
//!
//! This is the native, dependency-free stand-in for Brain2Qwerty's deep
//! Conv+Transformer acoustic model. It learns one centroid feature vector per
//! character and scores a new epoch by (negative, scaled) squared distance to
//! each centroid, turned into a log-probability via softmax. It is intentionally
//! simple — the point is a fully testable end-to-end pipeline and a meaningful
//! fitness signal for the optimizer, not state-of-the-art decoding. A real
//! deployment swaps this for the `python-sidecar` backend (see the integration
//! report).

use std::collections::BTreeMap;

/// A trained prototype classifier.
#[derive(Debug, Clone)]
pub struct PrototypeDecoder {
    /// `(character, centroid)` pairs.
    centroids: Vec<(char, Vec<f64>)>,
    /// Distance scale used to convert distances to logits.
    scale: f64,
}

impl PrototypeDecoder {
    /// Train from `(features, character)` samples.
    ///
    /// Returns a decoder with a uniform fallback if no samples are provided.
    pub fn train<'a>(samples: impl IntoIterator<Item = (&'a [f64], char)>) -> Self {
        let mut sums: BTreeMap<char, (Vec<f64>, usize)> = BTreeMap::new();
        let mut dim = 0usize;
        for (feats, ch) in samples {
            dim = dim.max(feats.len());
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

        // Scale = mean squared centroid-to-centroid distance (a typical spread).
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
        let _ = dim;
        PrototypeDecoder { centroids, scale }
    }

    /// Vocabulary the decoder can emit.
    pub fn vocabulary(&self) -> Vec<char> {
        self.centroids.iter().map(|(c, _)| *c).collect()
    }

    /// Per-character log-probabilities for one epoch's features (sorted by
    /// descending probability).
    pub fn logprobs(&self, features: &[f64]) -> Vec<(char, f64)> {
        if self.centroids.is_empty() {
            return Vec::new();
        }
        // logit_c = -dist^2 / scale
        let logits: Vec<(char, f64)> = self
            .centroids
            .iter()
            .map(|(c, cen)| (*c, -sq_dist(features, cen) / self.scale))
            .collect();
        let max = logits
            .iter()
            .map(|(_, l)| *l)
            .fold(f64::NEG_INFINITY, f64::max);
        let denom: f64 = logits.iter().map(|(_, l)| (l - max).exp()).sum();
        let log_denom = denom.ln();
        let mut out: Vec<(char, f64)> = logits
            .into_iter()
            .map(|(c, l)| (c, (l - max) - log_denom))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Greedy single-character prediction for one epoch.
    pub fn predict(&self, features: &[f64]) -> Option<char> {
        self.logprobs(features).first().map(|(c, _)| *c)
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
        // 'a' near (1,0), 'b' near (0,1).
        let a1 = [1.0, 0.0];
        let a2 = [0.9, 0.1];
        let b1 = [0.0, 1.0];
        let b2 = [0.1, 0.9];
        let dec = PrototypeDecoder::train([
            (&a1[..], 'a'),
            (&a2[..], 'a'),
            (&b1[..], 'b'),
            (&b2[..], 'b'),
        ]);
        assert_eq!(dec.predict(&[0.95, 0.05]), Some('a'));
        assert_eq!(dec.predict(&[0.05, 0.95]), Some('b'));
        let lp = dec.logprobs(&[0.95, 0.05]);
        // probabilities sum to ~1.
        let p: f64 = lp.iter().map(|(_, l)| l.exp()).sum();
        assert!((p - 1.0).abs() < 1e-9);
    }

    #[test]
    fn empty_decoder_is_safe() {
        let dec = PrototypeDecoder::train(std::iter::empty::<(&[f64], char)>());
        assert!(dec.predict(&[1.0]).is_none());
        assert!(dec.logprobs(&[1.0]).is_empty());
    }
}
