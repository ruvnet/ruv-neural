//! Keystroke epoching and feature extraction.
//!
//! For each keypress we cut a window from `epoch_pre_s` to `epoch_post_s`
//! relative to the keystroke onset (Brain2Qwerty V1 uses −0.2 s … +0.3 s),
//! baseline-correct each channel by subtracting its mean over the pre-keypress
//! interval, then reduce the window to a fixed-length per-channel feature vector
//! (mean and/or energy). The result is the `(feature, character)` supervision a
//! decoder trains on.

use serde::{Deserialize, Serialize};

use ruv_neural_core::signal::MultiChannelTimeSeries;

use crate::config::{Brain2TextConfig, FeatureKind};
use crate::events::EventTimeline;

/// A single keystroke epoch reduced to a feature vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Epoch {
    /// Feature vector of length `num_channels * feature.per_channel()`.
    pub features: Vec<f64>,
    /// Ground-truth character.
    pub character: char,
    /// Sentence this epoch belongs to.
    pub sentence_id: usize,
}

/// All epochs for one sentence, in typing order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceEpochs {
    /// Sentence id.
    pub id: usize,
    /// Ground-truth text.
    pub text: String,
    /// Epochs, one per keystroke.
    pub epochs: Vec<Epoch>,
}

/// Extract per-keystroke epochs from a *preprocessed* (resampled) recording.
///
/// `series` must already be filtered/resampled at `config.resample_hz`.
pub fn extract(
    series: &MultiChannelTimeSeries,
    timeline: &EventTimeline,
    config: &Brain2TextConfig,
) -> Vec<SentenceEpochs> {
    let sr = series.sample_rate_hz;
    let pre = config.epoch_pre_s;
    let post = config.epoch_post_s;

    let mut out = Vec::with_capacity(timeline.sentences.len());
    for sent in &timeline.sentences {
        let mut epochs = Vec::with_capacity(sent.keystrokes.len());
        for ks in &sent.keystrokes {
            let start = ((ks.onset_s + pre) * sr).round() as i64;
            let end = ((ks.onset_s + post) * sr).round() as i64;
            let baseline_end = (ks.onset_s * sr).round() as i64; // up to keypress
            let feats = window_features(series, start, end, baseline_end, config.feature);
            epochs.push(Epoch {
                features: feats,
                character: ks.character,
                sentence_id: sent.id,
            });
        }
        out.push(SentenceEpochs {
            id: sent.id,
            text: sent.text.clone(),
            epochs,
        });
    }
    out
}

/// Reduce one window (across all channels) to a feature vector with baseline
/// correction over `[start, baseline_end)`.
fn window_features(
    series: &MultiChannelTimeSeries,
    start: i64,
    end: i64,
    baseline_end: i64,
    feature: FeatureKind,
) -> Vec<f64> {
    let n = series.num_samples as i64;
    let per_ch = feature.per_channel();
    let mut feats = Vec::with_capacity(series.num_channels * per_ch);

    for ch in &series.data {
        // Baseline = mean over [start, baseline_end) clamped to valid range.
        let (b0, b1) = (start.max(0), baseline_end.clamp(0, n));
        let baseline = if b1 > b0 {
            let mut s = 0.0;
            for i in b0..b1 {
                s += ch[i as usize];
            }
            s / (b1 - b0) as f64
        } else {
            0.0
        };

        let (w0, w1) = (start.clamp(0, n), end.clamp(0, n));
        let len = (w1 - w0).max(1) as f64;
        let mut sum = 0.0;
        let mut sq = 0.0;
        for i in w0..w1 {
            let v = ch[i as usize] - baseline;
            sum += v;
            sq += v * v;
        }
        let mean = sum / len;
        let energy = sq / len;
        match feature {
            FeatureKind::Mean => feats.push(mean),
            FeatureKind::Energy => feats.push(energy),
            FeatureKind::MeanEnergy => {
                feats.push(mean);
                feats.push(energy);
            }
        }
    }
    feats
}

/// Deterministic split of sentences into (train, val, test) by ratio.
///
/// Sentences are assigned round-robin-by-modulo so the split is stable and
/// independent of ordering; `train_frac + val_frac` must be < 1.0 (the rest is
/// test).
pub fn split<'a>(
    sentences: &'a [SentenceEpochs],
    train_frac: f64,
    val_frac: f64,
) -> (
    Vec<&'a SentenceEpochs>,
    Vec<&'a SentenceEpochs>,
    Vec<&'a SentenceEpochs>,
) {
    let mut train = Vec::new();
    let mut val = Vec::new();
    let mut test = Vec::new();
    for s in sentences {
        // Stable bucket in [0,1) from the sentence id.
        let bucket = (s.id as f64 * 0.61803398875).fract();
        if bucket < train_frac {
            train.push(s);
        } else if bucket < train_frac + val_frac {
            val.push(s);
        } else {
            test.push(s);
        }
    }
    (train, val, test)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{generate_synthetic, SyntheticParams};
    use crate::preprocess::preprocess;

    #[test]
    fn extract_produces_one_epoch_per_keystroke() {
        let rec = generate_synthetic(&["hola", "mundo"], &SyntheticParams::default(), 3);
        let cfg = Brain2TextConfig::default();
        let pre = preprocess(&rec.series, &cfg).unwrap();
        let epochs = extract(&pre, &rec.timeline, &cfg);
        assert_eq!(epochs.len(), 2);
        assert_eq!(epochs[0].epochs.len(), 4); // "hola"
        assert_eq!(epochs[1].epochs.len(), 5); // "mundo"
        // Feature width = channels * 2 for MeanEnergy.
        let width = SyntheticParams::default().num_channels * 2;
        assert_eq!(epochs[0].epochs[0].features.len(), width);
    }

    #[test]
    fn split_partitions_disjointly() {
        let dummy: Vec<SentenceEpochs> = (0..100)
            .map(|i| SentenceEpochs {
                id: i,
                text: String::new(),
                epochs: vec![],
            })
            .collect();
        let (tr, va, te) = split(&dummy, 0.8, 0.1);
        assert_eq!(tr.len() + va.len() + te.len(), 100);
        assert!(!tr.is_empty() && !va.is_empty() && !te.is_empty());
    }
}
