//! Character-sequence decoding: acoustic model + language model via beam search.
//!
//! Mirrors Brain2Qwerty V1's structure — a per-keystroke character model whose
//! logits are fused with an n-gram language model through beam search:
//!
//! `score = acoustic_logp + lm_weight * lm_logp(char | context)`
//!
//! The acoustic model is any [`crate::model::AcousticModel`] (prototype, linear
//! softmax, or MLP), selected by [`Brain2TextConfig::model`]. The whole decoder
//! is serializable, so a trained instance is a self-contained, distributable
//! artifact.

pub mod ngram;

pub use ngram::CharNgram;

use serde::{Deserialize, Serialize};

use crate::config::Brain2TextConfig;
use crate::epoch::{Epoch, SentenceEpochs};
use crate::model::{AcousticEnum, AcousticModel};

/// A decoder that turns a sequence of keystroke epochs into text.
pub trait CharSequenceDecoder {
    /// Decode one sentence's epochs into a predicted string.
    fn decode_sentence(&self, epochs: &[Epoch]) -> String;
}

/// The native decoder: a trainable acoustic model + char n-gram LM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Brain2TextDecoder {
    acoustic: AcousticEnum,
    lm: CharNgram,
    lm_weight: f64,
    beam_size: usize,
    /// Max acoustic candidates considered per position.
    top_k: usize,
}

impl Brain2TextDecoder {
    /// Train the decoder from training sentences and a pipeline config.
    pub fn train(train: &[&SentenceEpochs], config: &Brain2TextConfig) -> Self {
        Self::train_seeded(train, config, 0xACED)
    }

    /// Train with an explicit RNG seed for the acoustic model.
    pub fn train_seeded(train: &[&SentenceEpochs], config: &Brain2TextConfig, seed: u64) -> Self {
        let samples: Vec<(Vec<f64>, char)> = train
            .iter()
            .flat_map(|s| s.epochs.iter())
            .map(|e| (e.features.clone(), e.character))
            .collect();
        let acoustic = AcousticEnum::train(config.model, &samples, &config.train_params(seed));
        let lm = CharNgram::train(train.iter().map(|s| s.text.as_str()), config.ngram_order);
        Brain2TextDecoder {
            acoustic,
            lm,
            lm_weight: config.lm_weight,
            beam_size: config.beam_size.max(1),
            top_k: 8,
        }
    }

    /// Access the trained acoustic model.
    pub fn acoustic(&self) -> &AcousticEnum {
        &self.acoustic
    }

    /// Greedy decode (acoustic argmax per position, no LM) — a baseline.
    pub fn decode_greedy(&self, epochs: &[Epoch]) -> String {
        epochs
            .iter()
            .filter_map(|e| self.acoustic.predict(&e.features))
            .collect()
    }
}

/// One hypothesis in the beam.
#[derive(Clone)]
struct Beam {
    text: String,
    score: f64,
}

impl CharSequenceDecoder for Brain2TextDecoder {
    fn decode_sentence(&self, epochs: &[Epoch]) -> String {
        if self.acoustic.vocabulary().is_empty() {
            return String::new();
        }
        let mut beams = vec![Beam {
            text: String::new(),
            score: 0.0,
        }];

        for epoch in epochs {
            let mut candidates = self.acoustic.logprobs(&epoch.features);
            candidates.truncate(self.top_k.max(1));
            if candidates.is_empty() {
                continue;
            }

            let mut next: Vec<Beam> = Vec::with_capacity(beams.len() * candidates.len());
            for beam in &beams {
                let ord = self.lm.order().saturating_sub(1);
                let ctx: String = if ord == 0 {
                    String::new()
                } else {
                    let chars: Vec<char> = beam.text.chars().collect();
                    let start = chars.len().saturating_sub(ord);
                    chars[start..].iter().collect()
                };
                for (ch, ac_lp) in &candidates {
                    let lm_lp = self.lm.logprob(&ctx, *ch);
                    let mut text = beam.text.clone();
                    text.push(*ch);
                    next.push(Beam {
                        text,
                        score: beam.score + ac_lp + self.lm_weight * lm_lp,
                    });
                }
            }
            next.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            next.truncate(self.beam_size);
            beams = next;
        }

        beams
            .into_iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
            .map(|b| b.text)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FeatureKind;
    use crate::dataset::{generate_synthetic, SyntheticParams};
    use crate::epoch::{extract, split};
    use crate::metrics::DecodeReport;
    use crate::model::ModelKind;
    use crate::preprocess::preprocess;

    fn corpus() -> Vec<&'static str> {
        vec![
            "hola mundo",
            "buenos dias",
            "hola amigo",
            "buenas noches",
            "como estas",
            "muy bien gracias",
            "hasta luego",
            "que tengas un buen dia",
            "hola que tal",
            "nos vemos manana",
        ]
    }

    #[test]
    fn decodes_synthetic_better_than_chance() {
        let sents = corpus();
        let rec = generate_synthetic(&sents, &SyntheticParams::default(), 42);
        let cfg = Brain2TextConfig::default();
        let pre = preprocess(&rec.series, &cfg).unwrap();
        let epochs = extract(&pre, &rec.timeline, &cfg);
        let (train, _val, _test) = split(&epochs, 0.7, 0.15);
        let train: Vec<&SentenceEpochs> = if train.is_empty() { epochs.iter().collect() } else { train };

        let dec = Brain2TextDecoder::train(&train, &cfg);
        let pairs: Vec<(String, String)> = train
            .iter()
            .map(|s| (dec.decode_sentence(&s.epochs), s.text.clone()))
            .collect();
        let report =
            DecodeReport::from_pairs(pairs.iter().map(|(p, t)| (p.as_str(), t.as_str())));
        assert!(report.mean_cer < 0.5, "CER too high: {}", report.mean_cer);
    }

    #[test]
    fn all_model_kinds_decode_and_serialize() {
        let rec = generate_synthetic(&corpus(), &SyntheticParams::default(), 7);
        for kind in [ModelKind::Prototype, ModelKind::Linear, ModelKind::Mlp] {
            let cfg = Brain2TextConfig {
                model: kind,
                feature: FeatureKind::Mean,
                ..Default::default()
            };
            let pre = preprocess(&rec.series, &cfg).unwrap();
            let epochs = extract(&pre, &rec.timeline, &cfg);
            let train: Vec<&SentenceEpochs> = epochs.iter().collect();
            let dec = Brain2TextDecoder::train(&train, &cfg);
            // Serializable artifact round-trip.
            let json = serde_json::to_string(&dec).unwrap();
            let back: Brain2TextDecoder = serde_json::from_str(&json).unwrap();
            let out = back.decode_sentence(&epochs[0].epochs);
            assert_eq!(out.chars().count(), epochs[0].epochs.len());
        }
    }
}
