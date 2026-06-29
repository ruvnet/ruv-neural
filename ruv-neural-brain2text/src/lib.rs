//! rUv Neural Brain-to-Text — a non-invasive brain-to-text decoding bridge
//! inspired by Meta AI's **Brain2Qwerty** and the **SpanishBCBL** (DECOMEG)
//! dataset, built natively on the rUv Neural sensor/signal types.
//!
//! # Pipeline
//!
//! ```text
//! Recording (MEG/EEG + keystroke timeline)
//!   -> preprocess  (bandpass 0.1-20 Hz, resample 50 Hz)        [preprocess]
//!   -> epoch       (-0.2..+0.3 s window, baseline, features)   [epoch]
//!   -> decode      (prototype acoustic model + n-gram LM beam) [decode]
//!   -> metrics     (CER / WER)                                 [metrics]
//! ```
//!
//! Every stage is configured by [`Brain2TextConfig`], and [`evolve`] searches
//! that configuration space with an evolutionary optimizer ("Darwin mode":
//! *freeze the model, evolve the harness*) using validation CER as fitness.
//!
//! The deep Conv+Transformer+LM model from upstream Brain2Qwerty is **not**
//! reimplemented here (it is PyTorch + CC BY-NC 4.0); the native
//! [`decode::PrototypeDecoder`] is a license-clean stand-in, and a Python
//! sidecar is the path to the real model (see `docs/research/`).

#![forbid(unsafe_code)]

pub mod config;
pub mod dataset;
pub mod decode;
pub mod epoch;
pub mod evolve;
pub mod events;
pub mod harness;
pub mod metrics;
pub mod model;
pub mod preprocess;

pub use config::{Brain2TextConfig, FeatureKind};
pub use dataset::{Modality, Recording};
pub use decode::{Brain2TextDecoder, CharSequenceDecoder};
pub use harness::{Harness, TrainedPipeline};
pub use metrics::{character_error_rate, word_error_rate, DecodeReport};
pub use model::{AcousticModel, ModelKind};

use ruv_neural_core::error::Result;

/// Which split to evaluate against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalSplit {
    /// Validation split (used as optimizer fitness).
    Validation,
    /// Held-out test split (used for final reporting).
    Test,
}

/// Result of an end-to-end evaluation.
#[derive(Debug, Clone)]
pub struct EvalResult {
    /// Decoding report on the chosen split.
    pub report: DecodeReport,
    /// Number of training sentences used.
    pub num_train: usize,
}

impl EvalResult {
    /// Fitness for the optimizer: higher is better. Equals `1 - mean_cer`.
    pub fn fitness(&self) -> f64 {
        1.0 - self.report.mean_cer
    }
}

/// Run the full pipeline for a config: preprocess, epoch, split, train on the
/// train split, and evaluate on the requested split.
///
/// This is the single entry point both the demo and the optimizer call.
pub fn evaluate(
    recording: &Recording,
    config: &Brain2TextConfig,
    split: EvalSplit,
    train_frac: f64,
    val_frac: f64,
) -> Result<EvalResult> {
    let pre = preprocess::preprocess(&recording.series, config)?;
    let epochs = epoch::extract(&pre, &recording.timeline, config);
    let (train, val, test) = epoch::split(&epochs, train_frac, val_frac);

    // Fall back to training on everything if a split came out empty (tiny corpora).
    let train_ref: Vec<&epoch::SentenceEpochs> = if train.is_empty() {
        epochs.iter().collect()
    } else {
        train
    };

    let decoder = Brain2TextDecoder::train(&train_ref, config);

    let eval_set = match split {
        EvalSplit::Validation if !val.is_empty() => val,
        EvalSplit::Test if !test.is_empty() => test,
        // If the requested split is empty, evaluate on training (keeps the
        // optimizer's fitness signal alive on small synthetic corpora).
        _ => train_ref.clone(),
    };

    let pairs: Vec<(String, String)> = eval_set
        .iter()
        .map(|s| (decoder.decode_sentence(&s.epochs), s.text.clone()))
        .collect();
    let report = DecodeReport::from_pairs(pairs.iter().map(|(p, t)| (p.as_str(), t.as_str())));

    Ok(EvalResult {
        report,
        num_train: train_ref.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dataset::{generate_synthetic, SyntheticParams};

    fn corpus() -> Vec<&'static str> {
        vec![
            "hola mundo",
            "buenos dias amigo",
            "como estas hoy",
            "muy bien gracias",
            "hasta luego",
            "que tengas buen dia",
            "nos vemos manana",
            "buenas noches a todos",
            "feliz cumpleanos",
            "muchas gracias por todo",
            "hola que tal estas",
            "todo esta bien aqui",
        ]
    }

    #[test]
    fn end_to_end_runs_and_reports() {
        let sents = corpus();
        let rec = generate_synthetic(&sents, &SyntheticParams::default(), 11);
        let cfg = Brain2TextConfig::default();
        let res = evaluate(&rec, &cfg, EvalSplit::Test, 0.7, 0.15).unwrap();
        assert!(res.report.num_sentences > 0);
        assert!(res.fitness() >= 0.0 && res.fitness() <= 1.0);
        // The synthetic structure is learnable, so we beat chance comfortably.
        assert!(res.report.mean_cer < 0.6, "CER {}", res.report.mean_cer);
    }
}
