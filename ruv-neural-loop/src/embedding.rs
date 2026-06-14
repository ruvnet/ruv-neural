//! Personal state embedding ("ruVector"): a normalized fusion of neural and
//! physiological features into a single per-person state vector.
//!
//! The embedding is the substrate for personalization — the controller can
//! compare the live state vector against a personal baseline and against
//! stored target exemplars, and it exports cleanly to the core
//! [`NeuralEmbedding`] / RVF ecosystem for evidence and offline analysis.

use ruv_neural_core::embedding::{EmbeddingMetadata, NeuralEmbedding};
use ruv_neural_core::brain::Atlas;
use serde::{Deserialize, Serialize};

use crate::state::StateObservation;

/// The ordered feature names of the personal state embedding.
pub const FEATURE_NAMES: [&str; 9] = [
    "arousal",
    "relaxation",
    "vagal_tone",
    "hr_norm",
    "resp_calm",
    "stillness",
    "gamma_index",
    "alpha_index",
    "connectivity",
];

/// Dimensionality of the personal state embedding.
pub const EMBEDDING_DIM: usize = FEATURE_NAMES.len();

/// A personal state embedding for one observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersonalStateEmbedding {
    /// Raw `[0, 1]` features in [`FEATURE_NAMES`] order.
    pub features: [f64; EMBEDDING_DIM],
    /// Timestamp (s).
    pub timestamp_s: f64,
}

impl PersonalStateEmbedding {
    /// Build the raw embedding from an observation. Missing neural features
    /// fall back to a neutral prior so the vector is always full-dimensional.
    pub fn from_observation(obs: &StateObservation) -> Self {
        let p = &obs.physio;
        let vagal = p.hrv.as_ref().map(|h| h.vagal_tone()).unwrap_or(0.5);
        let hr_norm = p
            .hrv
            .as_ref()
            .map(|h| ((h.mean_hr_bpm - 55.0) / 45.0).clamp(0.0, 1.0))
            .unwrap_or(0.5);
        let resp_calm = p.respiration.as_ref().map(|r| r.calm_index()).unwrap_or(0.5);
        let stillness = p.motion.as_ref().map(|m| m.stillness()).unwrap_or(0.5);
        let neural = obs.neural.unwrap_or_else(crate::state::NeuralFeatures::neutral);

        Self {
            features: [
                p.arousal_index,
                p.relaxation_index,
                vagal,
                hr_norm,
                resp_calm,
                stillness,
                neural.gamma_index,
                neural.alpha_index,
                neural.connectivity,
            ],
            timestamp_s: obs.timestamp_s,
        }
    }

    /// Euclidean distance to another embedding in raw feature space.
    pub fn distance(&self, other: &PersonalStateEmbedding) -> f64 {
        self.features
            .iter()
            .zip(other.features.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Export to the core [`NeuralEmbedding`] type for RVF / memory storage.
    pub fn to_neural_embedding(&self, subject_id: Option<String>) -> NeuralEmbedding {
        let meta = EmbeddingMetadata {
            subject_id,
            session_id: None,
            cognitive_state: None,
            source_atlas: Atlas::Custom(EMBEDDING_DIM),
            embedding_method: "personal-state-fusion".into(),
        };
        // Dimensions are validated by construction (fixed-size array), so this
        // cannot fail; fall back to an empty vector defensively.
        NeuralEmbedding::new(self.features.to_vec(), self.timestamp_s, meta)
            .unwrap_or_else(|_| panic!("personal embedding has fixed valid dimension"))
    }
}

/// A rolling per-person baseline (online mean / variance via Welford) used to
/// z-score live observations for personalization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersonalBaseline {
    mean: [f64; EMBEDDING_DIM],
    m2: [f64; EMBEDDING_DIM],
    count: u64,
}

impl Default for PersonalBaseline {
    fn default() -> Self {
        Self::new()
    }
}

impl PersonalBaseline {
    /// An empty baseline.
    pub fn new() -> Self {
        Self {
            mean: [0.0; EMBEDDING_DIM],
            m2: [0.0; EMBEDDING_DIM],
            count: 0,
        }
    }

    /// Number of observations folded in.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Whether enough samples exist to z-score meaningfully.
    pub fn is_established(&self) -> bool {
        self.count >= 3
    }

    /// Fold one embedding into the running statistics.
    pub fn update(&mut self, e: &PersonalStateEmbedding) {
        self.count += 1;
        let n = self.count as f64;
        for i in 0..EMBEDDING_DIM {
            let delta = e.features[i] - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = e.features[i] - self.mean[i];
            self.m2[i] += delta * delta2;
        }
    }

    /// Per-feature mean.
    pub fn mean(&self) -> [f64; EMBEDDING_DIM] {
        self.mean
    }

    /// Per-feature standard deviation (population), floored to avoid div-by-0.
    pub fn std(&self) -> [f64; EMBEDDING_DIM] {
        let mut s = [0.0; EMBEDDING_DIM];
        if self.count > 0 {
            for i in 0..EMBEDDING_DIM {
                s[i] = (self.m2[i] / self.count as f64).sqrt().max(1e-6);
            }
        } else {
            s = [1.0; EMBEDDING_DIM];
        }
        s
    }

    /// Z-score an embedding against this baseline.
    pub fn z_score(&self, e: &PersonalStateEmbedding) -> [f64; EMBEDDING_DIM] {
        let std = self.std();
        let mut z = [0.0; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            z[i] = (e.features[i] - self.mean[i]) / std[i];
        }
        z
    }

    /// Mahalanobis-like deviation magnitude (L2 of the z-scored vector): how
    /// far the live state is from this person's own baseline.
    pub fn deviation(&self, e: &PersonalStateEmbedding) -> f64 {
        self.z_score(e).iter().map(|z| z * z).sum::<f64>().sqrt()
    }
}
