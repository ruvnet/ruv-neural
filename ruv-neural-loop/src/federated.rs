//! Privacy-preserving federated personalization (ADR-0021) with an RVF
//! `FEDERATED_MANIFEST` segment (ADR-0023 point 5).
//!
//! Personalization never requires raw signals or even raw embeddings to leave a
//! device: each participant contributes only the summary statistics of its local
//! [`PersonalBaseline`] (a per-feature mean and a sample count). These updates
//! are combined with **federated averaging** (FedAvg); an optional
//! **differential-privacy** layer clips each contribution and adds calibrated
//! Gaussian noise (the Gaussian mechanism), bounding what the aggregate can
//! reveal about any single participant. The result, plus its privacy parameters,
//! is recorded in a self-describing `FEDERATED_MANIFEST` RVF segment.
//!
//! This is a coordination/accounting primitive — it deliberately does **not**
//! transport raw neural data, consistent with the ADR-0021/0022 data-minimization
//! and neurorights posture.

use rand::Rng;
use serde::{Deserialize, Serialize};

use ruv_neural_core::error::{Result, RuvNeuralError};
use ruv_neural_core::rvf_container::{RvfContainer, SegmentType, FLAG_SEALED};

use crate::embedding::{PersonalBaseline, FEATURE_NAMES};

/// One participant's contribution to a federated round: the per-feature mean of
/// its local baseline and the number of observations behind it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FederatedUpdate {
    /// Per-feature mean (same order as [`FEATURE_NAMES`]).
    pub mean: Vec<f64>,
    /// Number of local observations folded into `mean`.
    pub count: u64,
}

impl FederatedUpdate {
    /// Derive a contribution from a local baseline.
    pub fn from_baseline(baseline: &PersonalBaseline) -> Self {
        Self {
            mean: baseline.mean().to_vec(),
            count: baseline.count(),
        }
    }
}

/// Differential-privacy configuration for the Gaussian mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DpConfig {
    /// L2 clipping bound applied to each participant's update.
    pub clip_norm: f64,
    /// Noise multiplier `σ`: Gaussian noise std = `noise_multiplier * clip_norm`
    /// added to the summed (clipped) updates.
    pub noise_multiplier: f64,
    /// Target `δ` of the `(ε, δ)` guarantee.
    pub delta: f64,
}

impl DpConfig {
    /// Single-round `ε` of the analytic Gaussian mechanism:
    /// `ε = sqrt(2 ln(1.25/δ)) / σ`. (Classical bound, tight for `ε ≤ 1`;
    /// reported as a budget figure, not a substitute for full RDP accounting.)
    pub fn epsilon(&self) -> f64 {
        ((2.0 * (1.25 / self.delta).ln()).sqrt()) / self.noise_multiplier
    }
}

/// The aggregated output of a federated round.
#[derive(Debug, Clone, PartialEq)]
pub struct FederatedModel {
    /// Aggregated per-feature mean.
    pub mean: Vec<f64>,
    /// Number of participants in the round.
    pub num_participants: usize,
    /// Privacy budget spent this round, if differential privacy was applied.
    pub epsilon: Option<f64>,
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Clip `v` in place so its L2 norm does not exceed `clip_norm`.
fn clip_l2(v: &mut [f64], clip_norm: f64) {
    let norm = l2_norm(v);
    if norm > clip_norm && norm > 0.0 {
        let scale = clip_norm / norm;
        for x in v.iter_mut() {
            *x *= scale;
        }
    }
}

/// Sample a zero-mean Gaussian with standard deviation `std` (Box–Muller).
fn gaussian<R: Rng>(rng: &mut R, std: f64) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    let mag = (-2.0 * u1.ln()).sqrt();
    mag * (2.0 * std::f64::consts::PI * u2).cos() * std
}

/// Combine participant updates with federated averaging.
///
/// - Without DP (`dp = None`): a sample-count-weighted average (standard FedAvg).
/// - With DP: each update is L2-clipped to `clip_norm`, summed, perturbed with
///   Gaussian noise (std `noise_multiplier * clip_norm`), then divided by the
///   participant count — the Gaussian mechanism over the mean.
///
/// # Errors
/// Returns an error if `updates` is empty or the update dimensions disagree.
pub fn federated_average<R: Rng>(
    updates: &[FederatedUpdate],
    dp: Option<&DpConfig>,
    rng: &mut R,
) -> Result<FederatedModel> {
    if updates.is_empty() {
        return Err(RuvNeuralError::Embedding(
            "federated round needs at least one participant".into(),
        ));
    }
    let dim = updates[0].mean.len();
    if let Some(bad) = updates.iter().find(|u| u.mean.len() != dim) {
        return Err(RuvNeuralError::DimensionMismatch {
            expected: dim,
            got: bad.mean.len(),
        });
    }

    let n = updates.len();
    let mut agg = vec![0.0f64; dim];

    match dp {
        None => {
            let all_zero = updates.iter().all(|u| u.count == 0);
            let mut total_weight = 0.0;
            for u in updates {
                let w = if all_zero { 1.0 } else { u.count as f64 };
                total_weight += w;
                for (a, m) in agg.iter_mut().zip(u.mean.iter()) {
                    *a += w * m;
                }
            }
            for x in agg.iter_mut() {
                *x /= total_weight;
            }
            Ok(FederatedModel {
                mean: agg,
                num_participants: n,
                epsilon: None,
            })
        }
        Some(cfg) => {
            for u in updates {
                let mut m = u.mean.clone();
                clip_l2(&mut m, cfg.clip_norm);
                for (a, mi) in agg.iter_mut().zip(m.iter()) {
                    *a += mi;
                }
            }
            // One participant changes the summed vector by at most `clip_norm`,
            // so noise std on the sum is `noise_multiplier * clip_norm`.
            let noise_std = cfg.noise_multiplier * cfg.clip_norm;
            for x in agg.iter_mut() {
                *x = (*x + gaussian(rng, noise_std)) / n as f64;
            }
            Ok(FederatedModel {
                mean: agg,
                num_participants: n,
                epsilon: Some(cfg.epsilon()),
            })
        }
    }
}

/// A self-describing record of a federated round, stored in a
/// `FEDERATED_MANIFEST` RVF segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FederatedManifest {
    /// Round number.
    pub round: u64,
    /// Number of participants.
    pub num_participants: usize,
    /// Feature names for `aggregated_mean`.
    pub feature_names: Vec<String>,
    /// The aggregated per-feature mean.
    pub aggregated_mean: Vec<f64>,
    /// Whether differential privacy was applied.
    pub dp_applied: bool,
    /// L2 clip norm, if DP was applied.
    pub clip_norm: Option<f64>,
    /// Noise multiplier, if DP was applied.
    pub noise_multiplier: Option<f64>,
    /// Target `δ`, if DP was applied.
    pub delta: Option<f64>,
    /// Spent privacy budget `ε`, if DP was applied.
    pub epsilon: Option<f64>,
}

impl FederatedManifest {
    /// Build a manifest from an aggregated model and its DP configuration.
    pub fn new(round: u64, model: &FederatedModel, dp: Option<&DpConfig>) -> Self {
        Self {
            round,
            num_participants: model.num_participants,
            feature_names: FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
            aggregated_mean: model.mean.clone(),
            dp_applied: dp.is_some(),
            clip_norm: dp.map(|c| c.clip_norm),
            noise_multiplier: dp.map(|c| c.noise_multiplier),
            delta: dp.map(|c| c.delta),
            epsilon: model.epsilon,
        }
    }
}

/// Append a `FEDERATED_MANIFEST` segment to a container.
///
/// # Errors
/// Returns an error if the manifest cannot be serialized.
pub fn attach_federated_manifest(
    container: &mut RvfContainer,
    manifest: &FederatedManifest,
) -> Result<()> {
    let payload =
        serde_json::to_vec(manifest).map_err(|e| RuvNeuralError::Serialization(e.to_string()))?;
    container.add_segment(SegmentType::FederatedManifest, FLAG_SEALED, payload);
    Ok(())
}

/// Read the `FEDERATED_MANIFEST` segment from a container, if present.
///
/// # Errors
/// Returns an error if the segment is present but cannot be parsed.
pub fn read_federated_manifest(container: &RvfContainer) -> Result<Option<FederatedManifest>> {
    match container.find(SegmentType::FederatedManifest) {
        None => Ok(None),
        Some(seg) => {
            let m: FederatedManifest = serde_json::from_slice(&seg.payload)
                .map_err(|e| RuvNeuralError::Serialization(e.to_string()))?;
            Ok(Some(m))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use ruv_neural_core::rvf_container::RvfContainer;

    fn update(mean: Vec<f64>, count: u64) -> FederatedUpdate {
        FederatedUpdate { mean, count }
    }

    #[test]
    fn fedavg_is_count_weighted_mean() {
        let updates = vec![update(vec![0.0, 0.0], 1), update(vec![1.0, 2.0], 3)];
        let mut rng = StdRng::seed_from_u64(1);
        let model = federated_average(&updates, None, &mut rng).unwrap();
        // Weighted: (1*0 + 3*1)/4 = 0.75 ; (1*0 + 3*2)/4 = 1.5
        assert!((model.mean[0] - 0.75).abs() < 1e-12);
        assert!((model.mean[1] - 1.5).abs() < 1e-12);
        assert_eq!(model.num_participants, 2);
        assert!(model.epsilon.is_none());
    }

    #[test]
    fn dp_reports_budget_and_perturbs() {
        let updates = vec![
            update(vec![0.2, 0.4, 0.6], 10),
            update(vec![0.3, 0.5, 0.1], 10),
            update(vec![0.1, 0.9, 0.2], 10),
        ];
        let dp = DpConfig {
            clip_norm: 2.0,
            noise_multiplier: 1.0,
            delta: 1e-5,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let model = federated_average(&updates, Some(&dp), &mut rng).unwrap();

        // ε = sqrt(2 ln(1.25/δ)) / σ
        let expected_eps = (2.0 * (1.25 / 1e-5_f64).ln()).sqrt() / 1.0;
        assert!((model.epsilon.unwrap() - expected_eps).abs() < 1e-9);

        // DP output differs from the noise-free average (noise was added).
        let mut rng2 = StdRng::seed_from_u64(42);
        let plain = federated_average(&updates, None, &mut rng2).unwrap();
        assert!(model.mean != plain.mean);
        assert_eq!(model.mean.len(), 3);
    }

    #[test]
    fn dp_clipping_bounds_a_huge_update() {
        // A single enormous update is clipped, so it cannot dominate the mean.
        let updates = vec![update(vec![1000.0, 1000.0], 1), update(vec![0.0, 0.0], 1)];
        let dp = DpConfig {
            clip_norm: 1.0,
            noise_multiplier: 0.0, // isolate the clipping effect
            delta: 1e-5,
        };
        let mut rng = StdRng::seed_from_u64(7);
        let model = federated_average(&updates, Some(&dp), &mut rng).unwrap();
        // Clipped contribution has norm ≤ 1, averaged over 2 → mean norm ≤ 0.5.
        assert!(l2_norm(&model.mean) <= 0.5 + 1e-9);
    }

    #[test]
    fn empty_round_errors() {
        let mut rng = StdRng::seed_from_u64(0);
        assert!(federated_average(&[], None, &mut rng).is_err());
    }

    #[test]
    fn dimension_mismatch_errors() {
        let updates = vec![update(vec![0.0, 0.0], 1), update(vec![0.0], 1)];
        let mut rng = StdRng::seed_from_u64(0);
        assert!(federated_average(&updates, None, &mut rng).is_err());
    }

    #[test]
    fn manifest_roundtrips_through_container() {
        let updates = vec![update(vec![0.1; 9], 5), update(vec![0.2; 9], 5)];
        let dp = DpConfig {
            clip_norm: 1.5,
            noise_multiplier: 1.1,
            delta: 1e-6,
        };
        let mut rng = StdRng::seed_from_u64(99);
        let model = federated_average(&updates, Some(&dp), &mut rng).unwrap();
        let manifest = FederatedManifest::new(1, &model, Some(&dp));

        let mut container = RvfContainer::new();
        attach_federated_manifest(&mut container, &manifest).unwrap();

        let bytes = container.to_bytes();
        let back = RvfContainer::from_bytes(&bytes).unwrap();
        back.verify_integrity().unwrap();
        let recovered = read_federated_manifest(&back).unwrap().unwrap();

        assert_eq!(recovered.round, manifest.round);
        assert_eq!(recovered.num_participants, manifest.num_participants);
        assert_eq!(recovered.feature_names, manifest.feature_names);
        assert_eq!(
            recovered.aggregated_mean.len(),
            manifest.aggregated_mean.len()
        );
        for (a, b) in recovered
            .aggregated_mean
            .iter()
            .zip(manifest.aggregated_mean.iter())
        {
            assert!((a - b).abs() < 1e-9);
        }
        assert!(recovered.dp_applied);
        assert!(recovered.epsilon.unwrap() > 0.0);
    }

    #[test]
    fn no_manifest_returns_none() {
        let container = RvfContainer::new();
        assert!(read_federated_manifest(&container).unwrap().is_none());
    }
}
