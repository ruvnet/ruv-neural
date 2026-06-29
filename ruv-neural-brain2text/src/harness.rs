//! Composable training/decoding harness.
//!
//! This is the "harness for optimization and composability" layer: a small,
//! fluent builder that composes the pipeline stages (preprocess → epoch → train
//! → decode → score) into a single object, and a [`TrainedPipeline`] artifact
//! that bundles everything needed to decode — the config, the trained acoustic
//! model, and the language model — into one **serializable** unit you can save,
//! ship, and reload.
//!
//! The same harness is what [`crate::evolve`] drives: each candidate config is
//! `fit` and scored, so optimization and composition share one code path.

use serde::{Deserialize, Serialize};

use ruv_neural_core::error::Result;

use crate::config::Brain2TextConfig;
use crate::dataset::Recording;
use crate::decode::{Brain2TextDecoder, CharSequenceDecoder};
use crate::epoch::{self, Epoch};
use crate::metrics::DecodeReport;
use crate::preprocess;
use crate::{EvalResult, EvalSplit};

/// Fluent builder that composes a brain-to-text pipeline.
///
/// ```
/// use ruv_neural_brain2text::dataset::{generate_synthetic, SyntheticParams};
/// use ruv_neural_brain2text::harness::Harness;
/// use ruv_neural_brain2text::{Brain2TextConfig, EvalSplit};
///
/// let rec = generate_synthetic(&["hola mundo", "buenos dias"], &SyntheticParams::default(), 1);
/// let pipeline = Harness::new()
///     .with_config(Brain2TextConfig::default())
///     .with_split(0.7, 0.15)
///     .fit(&rec)
///     .unwrap();
/// let _report = pipeline.evaluate(&rec, EvalSplit::Test).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Harness {
    config: Brain2TextConfig,
    train_frac: f64,
    val_frac: f64,
    seed: u64,
}

impl Default for Harness {
    fn default() -> Self {
        Self {
            config: Brain2TextConfig::default(),
            train_frac: 0.7,
            val_frac: 0.15,
            seed: 0xACED,
        }
    }
}

impl Harness {
    /// Start a new harness with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the pipeline configuration (preprocessing + model + decoding).
    pub fn with_config(mut self, config: Brain2TextConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the train/val split fractions (the rest is test).
    pub fn with_split(mut self, train_frac: f64, val_frac: f64) -> Self {
        self.train_frac = train_frac;
        self.val_frac = val_frac;
        self
    }

    /// Set the RNG seed for model training.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The configured fractions.
    pub fn split(&self) -> (f64, f64) {
        (self.train_frac, self.val_frac)
    }

    /// Fit the pipeline on a recording's training split, returning a
    /// self-contained, serializable [`TrainedPipeline`].
    pub fn fit(&self, recording: &Recording) -> Result<TrainedPipeline> {
        let pre = preprocess::preprocess(&recording.series, &self.config)?;
        let epochs = epoch::extract(&pre, &recording.timeline, &self.config);
        let (train, _val, _test) = epoch::split(&epochs, self.train_frac, self.val_frac);
        let train_ref: Vec<&epoch::SentenceEpochs> = if train.is_empty() {
            epochs.iter().collect()
        } else {
            train
        };
        let decoder = Brain2TextDecoder::train_seeded(&train_ref, &self.config, self.seed);
        Ok(TrainedPipeline {
            config: self.config.clone(),
            decoder,
        })
    }
}

/// A trained, serializable pipeline: everything needed to decode brain signals
/// to text. Save it with [`TrainedPipeline::to_json`] and reload with
/// [`TrainedPipeline::from_json`] — this is the distributable model artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainedPipeline {
    /// The configuration this pipeline was trained with.
    pub config: Brain2TextConfig,
    /// The trained decoder (acoustic model + LM).
    pub decoder: Brain2TextDecoder,
}

impl TrainedPipeline {
    /// Decode a sequence of keystroke epochs to text.
    pub fn decode(&self, epochs: &[Epoch]) -> String {
        self.decoder.decode_sentence(epochs)
    }

    /// Preprocess + epoch a recording and evaluate this pipeline on a split.
    pub fn evaluate(&self, recording: &Recording, which: EvalSplit) -> Result<EvalResult> {
        let pre = preprocess::preprocess(&recording.series, &self.config)?;
        let epochs = epoch::extract(&pre, &recording.timeline, &self.config);
        // Use the same split geometry the harness would (70/15 by default here).
        let (train, val, test) = epoch::split(&epochs, 0.7, 0.15);
        let set = match which {
            EvalSplit::Validation if !val.is_empty() => val,
            EvalSplit::Test if !test.is_empty() => test,
            _ if !train.is_empty() => train,
            _ => epochs.iter().collect(),
        };
        let pairs: Vec<(String, String)> = set
            .iter()
            .map(|s| (self.decode(&s.epochs), s.text.clone()))
            .collect();
        let report = DecodeReport::from_pairs(pairs.iter().map(|(p, t)| (p.as_str(), t.as_str())));
        Ok(EvalResult {
            report,
            num_train: set.len(),
        })
    }

    /// Serialize the whole pipeline to JSON (the distributable artifact).
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| {
            ruv_neural_core::error::RuvNeuralError::Serialization(e.to_string())
        })
    }

    /// Deserialize a pipeline from JSON.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| {
            ruv_neural_core::error::RuvNeuralError::Serialization(e.to_string())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{generate_synthetic, SyntheticParams};

    fn corpus() -> Vec<&'static str> {
        vec![
            "hola mundo",
            "buenos dias amigo",
            "como estas hoy",
            "muy bien gracias",
            "hasta luego pronto",
            "que tengas buen dia",
            "nos vemos manana",
            "buenas noches a todos",
            "feliz cumpleanos hoy",
            "muchas gracias por todo",
        ]
    }

    #[test]
    fn fit_evaluate_and_serialize() {
        let rec = generate_synthetic(&corpus(), &SyntheticParams::default(), 9);
        let pipeline = Harness::new()
            .with_config(Brain2TextConfig::default())
            .with_split(0.7, 0.15)
            .fit(&rec)
            .unwrap();

        let report = pipeline.evaluate(&rec, EvalSplit::Test).unwrap();
        assert!(report.report.num_sentences > 0);

        // Round-trip the artifact and confirm identical decodes.
        let json = pipeline.to_json().unwrap();
        let restored = TrainedPipeline::from_json(&json).unwrap();

        let pre = preprocess::preprocess(&rec.series, &pipeline.config).unwrap();
        let epochs = epoch::extract(&pre, &rec.timeline, &pipeline.config);
        let a = pipeline.decode(&epochs[0].epochs);
        let b = restored.decode(&epochs[0].epochs);
        assert_eq!(a, b);
    }
}
