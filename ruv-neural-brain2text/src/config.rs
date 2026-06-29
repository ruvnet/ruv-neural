//! Hyperparameter configuration for the brain-to-text pipeline.
//!
//! These parameters mirror the tunable knobs of the Brain2Qwerty V1 pipeline
//! (bandpass, resample, keystroke window, language-model weight, beam size) and
//! are exactly the surface the [`crate::evolve`] optimizer mutates.

use serde::{Deserialize, Serialize};

use crate::model::ModelKind;

/// Feature extracted per channel within a keystroke epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureKind {
    /// Mean amplitude over the window (1 value / channel).
    Mean,
    /// Signal energy (mean of squares) over the window (1 value / channel).
    Energy,
    /// Concatenation of mean and energy (2 values / channel).
    MeanEnergy,
}

impl FeatureKind {
    /// Number of feature values produced per channel.
    pub fn per_channel(&self) -> usize {
        match self {
            FeatureKind::Mean | FeatureKind::Energy => 1,
            FeatureKind::MeanEnergy => 2,
        }
    }
}

/// End-to-end pipeline configuration.
///
/// All fields are bounded by [`Brain2TextConfig::clamp`] so that evolutionary
/// mutation can never produce an invalid pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Brain2TextConfig {
    /// Bandpass low cutoff in Hz (Brain2Qwerty V1 default: 0.1).
    pub bandpass_low_hz: f64,
    /// Bandpass high cutoff in Hz (Brain2Qwerty V1 default: 20.0).
    pub bandpass_high_hz: f64,
    /// Butterworth filter order.
    pub filter_order: usize,
    /// Target sample rate after resampling in Hz (V1 default: 50.0).
    pub resample_hz: f64,
    /// Keystroke epoch start relative to keypress in seconds (V1: -0.2).
    pub epoch_pre_s: f64,
    /// Keystroke epoch end relative to keypress in seconds (V1: +0.3).
    pub epoch_post_s: f64,
    /// Per-channel feature kind.
    pub feature: FeatureKind,
    /// Acoustic model family.
    pub model: ModelKind,
    /// SGD learning rate (gradient models).
    pub learning_rate: f64,
    /// Training epochs (gradient models).
    pub epochs: usize,
    /// Hidden layer width (MLP only).
    pub hidden_size: usize,
    /// L2 regularization strength (gradient models).
    pub l2: f64,
    /// Character n-gram order for the language model (V1: 9).
    pub ngram_order: usize,
    /// Language-model fusion weight (V1 alpha: 5.0).
    pub lm_weight: f64,
    /// Beam width for decoding (V1: 30).
    pub beam_size: usize,
}

impl Default for Brain2TextConfig {
    /// Defaults track the published Brain2Qwerty V1 hyperparameters.
    fn default() -> Self {
        Self {
            bandpass_low_hz: 0.1,
            bandpass_high_hz: 20.0,
            filter_order: 4,
            resample_hz: 50.0,
            epoch_pre_s: -0.2,
            epoch_post_s: 0.3,
            feature: FeatureKind::MeanEnergy,
            model: ModelKind::Linear,
            learning_rate: 0.5,
            epochs: 80,
            hidden_size: 32,
            l2: 1e-4,
            ngram_order: 9,
            lm_weight: 5.0,
            beam_size: 30,
        }
    }
}

impl Brain2TextConfig {
    /// Window length in seconds (post - pre).
    pub fn window_s(&self) -> f64 {
        (self.epoch_post_s - self.epoch_pre_s).max(0.0)
    }

    /// Training hyperparameters for the gradient-trained acoustic models.
    pub fn train_params(&self, seed: u64) -> crate::model::TrainParams {
        crate::model::TrainParams {
            learning_rate: self.learning_rate,
            epochs: self.epochs,
            l2: self.l2,
            hidden_size: self.hidden_size,
            seed,
        }
    }

    /// Clamp every field into a valid range, returning a self-consistent config.
    ///
    /// Used after random mutation so the optimizer can explore freely without
    /// ever constructing a degenerate pipeline.
    pub fn clamp(mut self) -> Self {
        self.bandpass_low_hz = self.bandpass_low_hz.clamp(0.01, 5.0);
        self.bandpass_high_hz = self.bandpass_high_hz.clamp(self.bandpass_low_hz + 1.0, 100.0);
        self.filter_order = self.filter_order.clamp(2, 8);
        self.resample_hz = self.resample_hz.clamp(20.0, 250.0);
        self.epoch_pre_s = self.epoch_pre_s.clamp(-0.5, -0.02);
        self.epoch_post_s = self.epoch_post_s.clamp(0.05, 0.8);
        self.learning_rate = self.learning_rate.clamp(1e-3, 3.0);
        self.epochs = self.epochs.clamp(5, 300);
        self.hidden_size = self.hidden_size.clamp(4, 256);
        self.l2 = self.l2.clamp(0.0, 0.1);
        self.ngram_order = self.ngram_order.clamp(1, 12);
        self.lm_weight = self.lm_weight.clamp(0.0, 20.0);
        self.beam_size = self.beam_size.clamp(1, 128);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_brain2qwerty_v1() {
        let c = Brain2TextConfig::default();
        assert_eq!(c.bandpass_low_hz, 0.1);
        assert_eq!(c.bandpass_high_hz, 20.0);
        assert_eq!(c.resample_hz, 50.0);
        assert_eq!(c.ngram_order, 9);
        assert_eq!(c.lm_weight, 5.0);
        assert_eq!(c.beam_size, 30);
        assert!((c.window_s() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn clamp_repairs_degenerate_config() {
        let bad = Brain2TextConfig {
            bandpass_low_hz: -10.0,
            bandpass_high_hz: 0.0,
            filter_order: 0,
            resample_hz: 1.0,
            epoch_pre_s: 5.0,
            epoch_post_s: -5.0,
            feature: FeatureKind::Mean,
            model: ModelKind::Mlp,
            learning_rate: 100.0,
            epochs: 0,
            hidden_size: 0,
            l2: 10.0,
            ngram_order: 0,
            lm_weight: -3.0,
            beam_size: 0,
        }
        .clamp();
        assert!(bad.bandpass_low_hz >= 0.01);
        assert!(bad.bandpass_high_hz > bad.bandpass_low_hz);
        assert!(bad.filter_order >= 2);
        assert!(bad.resample_hz >= 20.0);
        assert!(bad.epoch_pre_s < 0.0);
        assert!(bad.epoch_post_s > 0.0);
        assert!(bad.learning_rate <= 3.0);
        assert!(bad.epochs >= 5);
        assert!(bad.hidden_size >= 4);
        assert!(bad.l2 <= 0.1);
        assert!(bad.ngram_order >= 1);
        assert!(bad.lm_weight >= 0.0);
        assert!(bad.beam_size >= 1);
    }

    #[test]
    fn feature_widths() {
        assert_eq!(FeatureKind::Mean.per_channel(), 1);
        assert_eq!(FeatureKind::Energy.per_channel(), 1);
        assert_eq!(FeatureKind::MeanEnergy.per_channel(), 2);
    }
}
