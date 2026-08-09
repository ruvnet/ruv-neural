//! Executable promotion gates for foundation-model embedding claims.
//!
//! ADR-0016 point 4 says a foundation-model backend must beat the lightweight
//! baselines **out-of-sample** before promotion. This module makes that check
//! runnable, transcribing the negative-control protocol of the mid-2026 EEG
//! foundation-model benchmark (Khanday et al., arXiv:2607.27268) and the
//! imbalanced-detection protocol from LibriBrain KWS (arXiv:2510.21038):
//!
//! 1. **Random-init comparator.** The benchmark trains its baselines
//!    (EEGNet, ShallowFBCSPNet, …) *from random initialization* while the
//!    foundation models are fine-tuned from public checkpoints. Its headline
//!    result is that a **16K-parameter EEGNet reached 61.3%** against LaBraM
//!    (5.8M, 57.3%) and EEGMamba (8.3M, 56.6%) on 3-class speech-mode LOSO.
//!    A candidate that cannot beat a small from-scratch model has not earned
//!    promotion. See [`PromotionGate::comparator_max_params`].
//! 2. **Subject-disjoint evaluation.** Leave-one-subject-out: no trial from
//!    the test subject is ever seen in training. Under LOSO on BCIC2020-T3
//!    *all five* models sat at chance (balanced accuracy 0.189–0.202 against
//!    0.20 chance, κ ≈ 0), so a within-subject number proves nothing.
//!    See [`subject_disjoint_folds`] and [`PromotionGate::chance_margin_sd`].
//! 3. **Metrics and statistics.** Balanced accuracy, Cohen's κ, and weighted
//!    F1, reported as mean ± SD across folds/seeds, with a Wilcoxon
//!    signed-rank test at p < 0.05 for pairwise comparison and against
//!    chance.
//! 4. **Leakage check.** The benchmark explicitly verifies the evaluation
//!    dataset is absent from the candidate's pretraining corpus. Declared via
//!    [`CandidateReport::pretraining_corpora`] and checked by the gate.
//! 5. **Permutation null for imbalanced detection.** Report AUPRC against the
//!    base-rate null and gate on the *ratio*, not the raw value (LibriBrain:
//!    13.4× chance at a 0.00515 base rate). See
//!    [`ruv_neural_brain2text`-side helpers] — the ratio threshold lives in
//!    [`PromotionGate::min_auprc_lift`].
//!
//! **Designed here, not transcribed:** the *dataset-identity probe*
//! ([`dataset_identity_probe`]). The replicated finding that frozen
//! foundation embeddings separate datasets far better than clinical labels
//! motivates it, but no verified 2026 source specifies the protocol, so this
//! implementation is ruv-neural's own and is labelled as such rather than
//! cited.
//!
//! Nothing here trains a model: gates consume already-computed per-fold
//! scores, so they run on the edge and in CI without a DL toolchain.

use std::collections::{BTreeMap, BTreeSet};

use ruv_neural_core::error::{Result, RuvNeuralError};

/// Per-fold scores for one model arm.
#[derive(Debug, Clone, PartialEq)]
pub struct FoldScores {
    /// Balanced accuracy per fold, in `[0, 1]`.
    pub balanced_accuracy: Vec<f64>,
    /// Cohen's κ per fold (may be negative).
    pub cohens_kappa: Vec<f64>,
    /// Weighted F1 per fold, in `[0, 1]`.
    pub weighted_f1: Vec<f64>,
}

impl FoldScores {
    /// Build from equal-length metric vectors.
    pub fn new(
        balanced_accuracy: Vec<f64>,
        cohens_kappa: Vec<f64>,
        weighted_f1: Vec<f64>,
    ) -> Result<Self> {
        let n = balanced_accuracy.len();
        if n == 0 {
            return Err(RuvNeuralError::Config(
                "fold scores must be non-empty".to_string(),
            ));
        }
        if cohens_kappa.len() != n || weighted_f1.len() != n {
            return Err(RuvNeuralError::Config(
                "all metric vectors must have one entry per fold".to_string(),
            ));
        }
        let all = balanced_accuracy
            .iter()
            .chain(&cohens_kappa)
            .chain(&weighted_f1);
        if all.into_iter().any(|x| !x.is_finite()) {
            return Err(RuvNeuralError::Config(
                "fold scores must be finite".to_string(),
            ));
        }
        Ok(Self {
            balanced_accuracy,
            cohens_kappa,
            weighted_f1,
        })
    }

    /// Number of folds.
    pub fn num_folds(&self) -> usize {
        self.balanced_accuracy.len()
    }

    /// Mean and (population) SD of balanced accuracy — the "mean ± SD across
    /// splits/seeds" the benchmark reports.
    pub fn balanced_accuracy_mean_sd(&self) -> (f64, f64) {
        mean_sd(&self.balanced_accuracy)
    }
}

/// A candidate foundation-model backend's evaluation report.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateReport {
    /// Method tag, e.g. `"foundation:reve"`.
    pub method_tag: String,
    /// Trainable parameter count of the candidate.
    pub num_params: usize,
    /// Corpora the candidate was pretrained on (for the leakage check).
    pub pretraining_corpora: Vec<String>,
    /// Subject-disjoint per-fold scores.
    pub scores: FoldScores,
}

/// The random-init comparator arm.
#[derive(Debug, Clone, PartialEq)]
pub struct ComparatorReport {
    /// Comparator name, e.g. `"eegnet-16k"`.
    pub name: String,
    /// Trainable parameter count; the gate requires this to stay small.
    pub num_params: usize,
    /// Per-fold scores on the *same* folds as the candidate.
    pub scores: FoldScores,
}

/// Thresholds a candidate must clear to be promoted.
#[derive(Debug, Clone, PartialEq)]
pub struct PromotionGate {
    /// Maximum parameter count for a legitimate random-init comparator.
    /// The benchmark's bar is a 16K-parameter EEGNet; 100K leaves headroom
    /// while keeping the comparator genuinely small.
    pub comparator_max_params: usize,
    /// Chance level for the task (e.g. 0.20 for 5-class).
    pub chance_level: f64,
    /// How many SDs above chance the candidate's mean balanced accuracy must
    /// sit. The failure mode this catches is BCIC2020-T3's 0.189–0.202 at
    /// 0.20 chance.
    pub chance_margin_sd: f64,
    /// Significance level for the Wilcoxon signed-rank tests.
    pub alpha: f64,
    /// Minimum number of folds/seeds. LibriBrain used only 3 seeds, which is
    /// thin. The floor here is **6**, which is not arbitrary: under the exact
    /// two-sided Wilcoxon signed-rank test the smallest attainable p-value is
    /// `2 / 2^n`, so n = 5 tops out at 0.0625 and can *never* reach
    /// `alpha = 0.05` however large the effect. Six folds is the smallest
    /// count at which the significance check is satisfiable at all.
    pub min_folds: usize,
    /// Minimum AUPRC lift over the base-rate null for imbalanced detection
    /// claims (LibriBrain observed 13.4×). Checked by [`check_auprc_lift`].
    pub min_auprc_lift: f64,
    /// Maximum tolerated dataset-identity probe accuracy above chance; above
    /// this the representation is judged to encode acquisition identity
    /// rather than neural content.
    pub max_identity_margin: f64,
}

impl Default for PromotionGate {
    fn default() -> Self {
        Self {
            comparator_max_params: 100_000,
            chance_level: 0.5,
            chance_margin_sd: 1.0,
            alpha: 0.05,
            min_folds: 6,
            min_auprc_lift: 2.0,
            max_identity_margin: 0.25,
        }
    }
}

/// Why a candidate failed (or passed) the gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateCheck {
    /// The check passed, with a human-readable summary.
    Pass(String),
    /// The check failed, with the reason.
    Fail(String),
}

impl GateCheck {
    /// Whether this check passed.
    pub fn passed(&self) -> bool {
        matches!(self, GateCheck::Pass(_))
    }

    /// The check's message.
    pub fn message(&self) -> &str {
        match self {
            GateCheck::Pass(m) | GateCheck::Fail(m) => m,
        }
    }
}

/// The full verdict: promotion requires every check to pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromotionVerdict {
    /// Method tag of the evaluated candidate.
    pub method_tag: String,
    /// Individual checks, in evaluation order.
    pub checks: Vec<GateCheck>,
}

impl PromotionVerdict {
    /// True only when every check passed. Conjunctive by construction: a
    /// candidate cannot trade a failed control for a strong headline number.
    pub fn promoted(&self) -> bool {
        !self.checks.is_empty() && self.checks.iter().all(GateCheck::passed)
    }

    /// Messages of all failed checks.
    pub fn failures(&self) -> Vec<&str> {
        self.checks
            .iter()
            .filter(|c| !c.passed())
            .map(GateCheck::message)
            .collect()
    }
}

impl PromotionGate {
    /// Evaluate a candidate against its random-init comparator.
    ///
    /// `evaluation_corpus` is the dataset the folds come from; the leakage
    /// check fails if it appears in the candidate's pretraining corpora.
    pub fn evaluate(
        &self,
        candidate: &CandidateReport,
        comparator: &ComparatorReport,
        evaluation_corpus: &str,
    ) -> Result<PromotionVerdict> {
        if !(self.chance_level > 0.0 && self.chance_level < 1.0) {
            return Err(RuvNeuralError::Config(format!(
                "chance_level must be in (0, 1), got {}",
                self.chance_level
            )));
        }
        if !(self.alpha > 0.0 && self.alpha < 1.0) {
            return Err(RuvNeuralError::Config(format!(
                "alpha must be in (0, 1), got {}",
                self.alpha
            )));
        }
        if candidate.scores.num_folds() != comparator.scores.num_folds() {
            return Err(RuvNeuralError::Config(
                "candidate and comparator must be evaluated on the same folds".to_string(),
            ));
        }

        let mut checks = Vec::new();
        let n = candidate.scores.num_folds();

        // 1. Enough folds/seeds to say anything.
        checks.push(if n >= self.min_folds {
            GateCheck::Pass(format!("{n} folds >= minimum {}", self.min_folds))
        } else {
            GateCheck::Fail(format!("only {n} folds, need >= {}", self.min_folds))
        });

        // 2. The comparator must actually be a small random-init model.
        checks.push(if comparator.num_params <= self.comparator_max_params {
            GateCheck::Pass(format!(
                "comparator '{}' has {} params (<= {})",
                comparator.name, comparator.num_params, self.comparator_max_params
            ))
        } else {
            GateCheck::Fail(format!(
                "comparator '{}' has {} params, exceeding the {} limit — not a small random-init control",
                comparator.name, comparator.num_params, self.comparator_max_params
            ))
        });

        // 3. Pretraining leakage.
        let leaked = candidate
            .pretraining_corpora
            .iter()
            .any(|c| c.eq_ignore_ascii_case(evaluation_corpus));
        checks.push(if leaked {
            GateCheck::Fail(format!(
                "evaluation corpus '{evaluation_corpus}' appears in the candidate's pretraining corpora"
            ))
        } else {
            GateCheck::Pass(format!(
                "no leakage: '{evaluation_corpus}' absent from pretraining corpora"
            ))
        });

        // 4. Above chance by a real margin (the LOSO-at-chance failure mode).
        let (mean, sd) = candidate.scores.balanced_accuracy_mean_sd();
        let margin = self.chance_level + self.chance_margin_sd * sd;
        checks.push(if mean > margin {
            GateCheck::Pass(format!(
                "balanced accuracy {mean:.3} ± {sd:.3} exceeds chance {:.3} by > {:.1} SD",
                self.chance_level, self.chance_margin_sd
            ))
        } else {
            GateCheck::Fail(format!(
                "balanced accuracy {mean:.3} ± {sd:.3} is within {:.1} SD of chance {:.3}",
                self.chance_margin_sd, self.chance_level
            ))
        });

        // 5. Beats the small random-init comparator, significantly.
        let (cmean, _) = comparator.scores.balanced_accuracy_mean_sd();
        let p = wilcoxon_signed_rank_p(
            &candidate.scores.balanced_accuracy,
            &comparator.scores.balanced_accuracy,
        );
        checks.push(match p {
            Some(p) if mean > cmean && p < self.alpha => GateCheck::Pass(format!(
                "beats comparator '{}' ({mean:.3} vs {cmean:.3}, Wilcoxon p={p:.4} < {})",
                comparator.name, self.alpha
            )),
            Some(p) => GateCheck::Fail(format!(
                "does not significantly beat comparator '{}' ({mean:.3} vs {cmean:.3}, Wilcoxon p={p:.4})",
                comparator.name
            )),
            None => GateCheck::Fail(format!(
                "cannot compare with '{}': all fold differences are zero",
                comparator.name
            )),
        });

        Ok(PromotionVerdict {
            method_tag: candidate.method_tag.clone(),
            checks,
        })
    }

    /// Gate an imbalanced-detection claim on AUPRC lift over the base-rate
    /// null, the way LibriBrain reports it (raw AUPRC alone is misleading at
    /// low base rates).
    pub fn check_auprc_lift(&self, model_auprc: f64, null_auprc: f64) -> GateCheck {
        if !(model_auprc.is_finite() && null_auprc.is_finite() && null_auprc > 0.0) {
            return GateCheck::Fail("AUPRC inputs must be finite with a positive null".to_string());
        }
        let lift = model_auprc / null_auprc;
        if lift >= self.min_auprc_lift {
            GateCheck::Pass(format!(
                "AUPRC {model_auprc:.4} is {lift:.1}x the {null_auprc:.4} base-rate null"
            ))
        } else {
            GateCheck::Fail(format!(
                "AUPRC {model_auprc:.4} is only {lift:.1}x the {null_auprc:.4} null (need {:.1}x)",
                self.min_auprc_lift
            ))
        }
    }

    /// Gate a dataset-identity probe result.
    ///
    /// **ruv-neural's own design** (see module docs): no verified source
    /// specifies this protocol. The rationale is the replicated finding that
    /// frozen foundation embeddings separate *datasets* far better than
    /// clinical labels; a representation that predicts its acquisition source
    /// well above chance is encoding site/montage artifacts.
    pub fn check_identity_probe(&self, probe: &IdentityProbeResult) -> GateCheck {
        let margin = probe.accuracy - probe.chance_level;
        if margin <= self.max_identity_margin {
            GateCheck::Pass(format!(
                "dataset-identity probe {:.3} is {:.3} above chance {:.3} (<= {:.3})",
                probe.accuracy, margin, probe.chance_level, self.max_identity_margin
            ))
        } else {
            GateCheck::Fail(format!(
                "dataset-identity probe {:.3} is {:.3} above chance {:.3} — representation encodes acquisition identity",
                probe.accuracy, margin, probe.chance_level
            ))
        }
    }
}

/// Result of a dataset-identity probe.
#[derive(Debug, Clone, PartialEq)]
pub struct IdentityProbeResult {
    /// Leave-one-out accuracy of predicting the source dataset/session.
    pub accuracy: f64,
    /// Chance level, i.e. the largest class prior.
    pub chance_level: f64,
    /// Number of distinct source datasets/sessions.
    pub num_sources: usize,
}

/// Leave-one-out dataset-identity probe over labelled embeddings.
///
/// **ruv-neural's own design, not a transcribed protocol.** Runs a
/// nearest-centroid classifier (deterministic, no training loop, edge-safe)
/// predicting each embedding's source from the *other* embeddings' centroids.
/// High accuracy means the embedding space is organized by acquisition
/// source rather than by neural content.
///
/// `items` pairs each embedding vector with its source identifier.
pub fn dataset_identity_probe(items: &[(Vec<f64>, String)]) -> Result<IdentityProbeResult> {
    if items.len() < 2 {
        return Err(RuvNeuralError::Config(
            "identity probe needs >= 2 embeddings".to_string(),
        ));
    }
    let dim = items[0].0.len();
    if dim == 0 || items.iter().any(|(v, _)| v.len() != dim) {
        return Err(RuvNeuralError::DimensionMismatch {
            expected: dim,
            got: items
                .iter()
                .map(|(v, _)| v.len())
                .find(|&l| l != dim)
                .unwrap_or(0),
        });
    }
    if items.iter().any(|(v, _)| v.iter().any(|x| !x.is_finite())) {
        return Err(RuvNeuralError::Embedding(
            "identity probe inputs must be finite".to_string(),
        ));
    }

    // Per-source sums and counts, so a leave-one-out centroid is a subtraction.
    let mut sums: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (v, src) in items {
        let entry = sums.entry(src.as_str()).or_insert_with(|| vec![0.0; dim]);
        for (a, b) in entry.iter_mut().zip(v) {
            *a += b;
        }
        *counts.entry(src.as_str()).or_insert(0) += 1;
    }
    let sources: BTreeSet<&str> = counts.keys().copied().collect();
    if sources.len() < 2 {
        return Err(RuvNeuralError::Config(
            "identity probe needs >= 2 distinct sources".to_string(),
        ));
    }

    let mut correct = 0usize;
    for (v, src) in items {
        let mut best: Option<(&str, f64)> = None;
        for &candidate in &sources {
            let count = counts[candidate];
            // Leave-one-out: exclude this item from its own source centroid.
            let (n, subtract_self) = if candidate == src.as_str() {
                (count - 1, true)
            } else {
                (count, false)
            };
            if n == 0 {
                continue;
            }
            let sum = &sums[candidate];
            let mut dist = 0.0;
            for k in 0..dim {
                let centroid = if subtract_self {
                    (sum[k] - v[k]) / n as f64
                } else {
                    sum[k] / n as f64
                };
                let d = v[k] - centroid;
                dist += d * d;
            }
            if best.is_none_or(|(_, bd)| dist < bd) {
                best = Some((candidate, dist));
            }
        }
        if let Some((predicted, _)) = best {
            if predicted == src.as_str() {
                correct += 1;
            }
        }
    }

    let largest_prior = counts.values().copied().max().unwrap_or(1) as f64 / items.len() as f64;
    Ok(IdentityProbeResult {
        accuracy: correct as f64 / items.len() as f64,
        chance_level: largest_prior,
        num_sources: sources.len(),
    })
}

/// Build subject-disjoint (leave-one-subject-out) folds.
///
/// Returns one `(train_indices, test_indices)` pair per subject, with no
/// trial from the test subject appearing in training — the split the 2026
/// benchmark uses, and the only one under which its cross-subject
/// chance-level result is visible.
pub fn subject_disjoint_folds(subject_ids: &[String]) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if subject_ids.is_empty() {
        return Err(RuvNeuralError::Config(
            "need at least one labelled trial".to_string(),
        ));
    }
    let subjects: BTreeSet<&str> = subject_ids.iter().map(String::as_str).collect();
    if subjects.len() < 2 {
        return Err(RuvNeuralError::Config(
            "subject-disjoint folds need >= 2 subjects".to_string(),
        ));
    }
    Ok(subjects
        .into_iter()
        .map(|held_out| {
            let mut train = Vec::new();
            let mut test = Vec::new();
            for (i, s) in subject_ids.iter().enumerate() {
                if s == held_out {
                    test.push(i);
                } else {
                    train.push(i);
                }
            }
            (train, test)
        })
        .collect())
}

/// Mean and population standard deviation.
fn mean_sd(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Two-sided Wilcoxon signed-rank test p-value for paired samples.
///
/// Uses the exact null distribution for n <= 20 (the fold counts this gate
/// sees) and a normal approximation with continuity correction above that.
/// Zero differences are dropped (Wilcoxon's original treatment); ties share
/// average ranks. Returns `None` when every difference is zero.
pub fn wilcoxon_signed_rank_p(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let diffs: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(x, y)| x - y)
        .filter(|d| d.is_finite() && *d != 0.0)
        .collect();
    let n = diffs.len();
    if n == 0 {
        return None;
    }
    // Rank by absolute difference, averaging ties.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        diffs[i]
            .abs()
            .partial_cmp(&diffs[j].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && (diffs[order[j + 1]].abs() - diffs[order[i]].abs()).abs() < 1e-12 {
            j += 1;
        }
        let avg = ((i + j + 2) as f64) / 2.0; // ranks are 1-based
        for &k in &order[i..=j] {
            ranks[k] = avg;
        }
        i = j + 1;
    }
    let w_plus: f64 = ranks
        .iter()
        .zip(&diffs)
        .filter(|(_, d)| **d > 0.0)
        .map(|(r, _)| r)
        .sum();
    let total = (n * (n + 1)) as f64 / 2.0;
    let w = w_plus.min(total - w_plus);

    if n <= 20 {
        // Exact: count sign assignments with statistic <= w.
        let target = w;
        let mut at_or_below = 0u64;
        let combos = 1u64 << n;
        for mask in 0..combos {
            let mut sum = 0.0;
            for k in 0..n {
                if mask & (1 << k) != 0 {
                    sum += (k + 1) as f64;
                }
            }
            let stat = sum.min(total - sum);
            if stat <= target + 1e-12 {
                at_or_below += 1;
            }
        }
        Some((at_or_below as f64 / combos as f64).min(1.0))
    } else {
        let mean = total / 2.0;
        let sd = ((n * (n + 1) * (2 * n + 1)) as f64 / 24.0).sqrt();
        if sd == 0.0 {
            return None;
        }
        let z = (w - mean + 0.5) / sd;
        Some((2.0 * normal_cdf(z)).min(1.0))
    }
}

/// Standard normal CDF via the Abramowitz–Stegun erf approximation.
fn normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / std::f64::consts::SQRT_2)
}

fn erfc(x: f64) -> f64 {
    // Numerical Recipes' erfc approximation, |error| < 1.2e-7.
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 {
        ans
    } else {
        2.0 - ans
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(values: &[f64]) -> FoldScores {
        FoldScores::new(
            values.to_vec(),
            values.iter().map(|v| v - 0.2).collect(),
            values.to_vec(),
        )
        .unwrap()
    }

    fn candidate(values: &[f64]) -> CandidateReport {
        CandidateReport {
            method_tag: "foundation:reve".to_string(),
            num_params: 5_800_000,
            pretraining_corpora: vec!["TUEG".to_string()],
            scores: scores(values),
        }
    }

    fn comparator(values: &[f64]) -> ComparatorReport {
        ComparatorReport {
            name: "eegnet-16k".to_string(),
            num_params: 16_000,
            scores: scores(values),
        }
    }

    fn gate() -> PromotionGate {
        PromotionGate {
            chance_level: 0.333,
            ..PromotionGate::default()
        }
    }

    #[test]
    fn promotes_a_candidate_that_clears_every_control() {
        let v = gate()
            .evaluate(
                &candidate(&[0.71, 0.73, 0.70, 0.74, 0.72, 0.75]),
                &comparator(&[0.61, 0.62, 0.60, 0.63, 0.61, 0.64]),
                "UGR-MINDVOICE",
            )
            .unwrap();
        assert!(v.promoted(), "failures: {:?}", v.failures());
        assert_eq!(v.checks.len(), 5);
    }

    #[test]
    fn rejects_a_candidate_that_loses_to_a_16k_comparator() {
        // The published failure mode: EEGNet 61.3% vs LaBraM 57.3%.
        let v = gate()
            .evaluate(
                &candidate(&[0.573, 0.570, 0.575, 0.572, 0.574, 0.571]),
                &comparator(&[0.613, 0.610, 0.615, 0.612, 0.614, 0.611]),
                "UGR-MINDVOICE",
            )
            .unwrap();
        assert!(!v.promoted());
        assert!(v
            .failures()
            .iter()
            .any(|f| f.contains("does not significantly beat")));
    }

    #[test]
    fn rejects_chance_level_cross_subject_results() {
        // BCIC2020-T3 under LOSO: 0.189-0.202 against 0.20 chance.
        let g = PromotionGate {
            chance_level: 0.20,
            ..PromotionGate::default()
        };
        let v = g
            .evaluate(
                &candidate(&[0.189, 0.202, 0.195, 0.198, 0.191, 0.197]),
                &comparator(&[0.150, 0.160, 0.155, 0.158, 0.152, 0.157]),
                "BCIC2020-T3",
            )
            .unwrap();
        assert!(!v.promoted());
        assert!(v.failures().iter().any(|f| f.contains("within")));
    }

    #[test]
    fn rejects_pretraining_leakage_and_oversized_comparators() {
        let mut c = candidate(&[0.8, 0.82, 0.81, 0.79, 0.83, 0.84]);
        c.pretraining_corpora.push("UGR-MINDVOICE".to_string());
        let v = gate()
            .evaluate(&c, &comparator(&[0.6; 6]), "ugr-mindvoice")
            .unwrap();
        assert!(!v.promoted());
        assert!(
            v.failures()
                .iter()
                .any(|f| f.contains("leakage")
                    || f.contains("appears in the candidate's pretraining"))
        );

        let mut big = comparator(&[0.6; 6]);
        big.num_params = 5_000_000;
        let v = gate()
            .evaluate(
                &candidate(&[0.8, 0.82, 0.81, 0.79, 0.83, 0.84]),
                &big,
                "UGR-MINDVOICE",
            )
            .unwrap();
        assert!(!v.promoted());
        assert!(v
            .failures()
            .iter()
            .any(|f| f.contains("not a small random-init control")));
    }

    #[test]
    fn rejects_too_few_folds_and_mismatched_arms() {
        let v = gate()
            .evaluate(
                &candidate(&[0.8, 0.82, 0.81]),
                &comparator(&[0.6, 0.61, 0.62]),
                "UGR-MINDVOICE",
            )
            .unwrap();
        assert!(!v.promoted());
        assert!(v.failures().iter().any(|f| f.contains("only 3 folds")));
        // Arms evaluated on different fold counts is a hard error.
        assert!(gate()
            .evaluate(
                &candidate(&[0.8; 6]),
                &comparator(&[0.6; 4]),
                "UGR-MINDVOICE"
            )
            .is_err());
    }

    #[test]
    fn verdict_is_conjunctive() {
        // A strong headline number cannot buy back a failed control.
        let mut c = candidate(&[0.95, 0.96, 0.94, 0.97, 0.95, 0.98]);
        c.pretraining_corpora.push("UGR-MINDVOICE".to_string());
        let v = gate()
            .evaluate(&c, &comparator(&[0.6; 6]), "UGR-MINDVOICE")
            .unwrap();
        assert!(!v.promoted());
        assert_eq!(v.failures().len(), 1, "exactly the leakage check failed");
    }

    #[test]
    fn auprc_lift_gate_uses_the_ratio_not_the_raw_value() {
        let g = PromotionGate::default();
        // LibriBrain: 0.069 AUPRC at a 0.00515 base rate = 13.4x — a pass,
        // even though the raw value looks like failure.
        assert!(g.check_auprc_lift(0.069, 24.0 / 4660.0).passed());
        // A "better-looking" 0.30 AUPRC at a 0.25 base rate is only 1.2x.
        assert!(!g.check_auprc_lift(0.30, 0.25).passed());
        assert!(!g.check_auprc_lift(0.30, 0.0).passed());
        assert!(!g.check_auprc_lift(f64::NAN, 0.1).passed());
    }

    #[test]
    fn identity_probe_detects_source_clustered_embeddings() {
        // Embeddings clustered by acquisition source, not by content.
        let mut items = Vec::new();
        for i in 0..10 {
            items.push((vec![10.0 + i as f64 * 0.01, 0.0], "site-a".to_string()));
            items.push((vec![-10.0 - i as f64 * 0.01, 0.0], "site-b".to_string()));
        }
        let probe = dataset_identity_probe(&items).unwrap();
        assert_eq!(probe.num_sources, 2);
        assert!(probe.accuracy > 0.95, "accuracy {}", probe.accuracy);
        assert!(!PromotionGate::default()
            .check_identity_probe(&probe)
            .passed());
    }

    #[test]
    fn identity_probe_passes_on_source_agnostic_embeddings() {
        // Interleaved sources: the probe should be near chance.
        let mut items = Vec::new();
        for i in 0..20 {
            let src = if i % 2 == 0 { "site-a" } else { "site-b" };
            items.push((
                vec![(i as f64 * 0.7).sin(), (i as f64 * 0.3).cos()],
                src.to_string(),
            ));
        }
        let probe = dataset_identity_probe(&items).unwrap();
        assert!(probe.accuracy < 0.75, "accuracy {}", probe.accuracy);
        assert!(PromotionGate::default()
            .check_identity_probe(&probe)
            .passed());
    }

    #[test]
    fn identity_probe_rejects_degenerate_input() {
        assert!(dataset_identity_probe(&[]).is_err());
        assert!(dataset_identity_probe(&[(vec![1.0], "a".into())]).is_err());
        // One source only.
        assert!(
            dataset_identity_probe(&[(vec![1.0], "a".into()), (vec![2.0], "a".into())]).is_err()
        );
        // Ragged dimensions.
        assert!(
            dataset_identity_probe(&[(vec![1.0, 2.0], "a".into()), (vec![2.0], "b".into())])
                .is_err()
        );
        // Non-finite.
        assert!(
            dataset_identity_probe(&[(vec![f64::NAN], "a".into()), (vec![2.0], "b".into())])
                .is_err()
        );
    }

    #[test]
    fn subject_disjoint_folds_never_leak_the_test_subject() {
        let subjects: Vec<String> = ["s1", "s1", "s2", "s2", "s3"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let folds = subject_disjoint_folds(&subjects).unwrap();
        assert_eq!(folds.len(), 3);
        for (train, test) in &folds {
            assert!(!test.is_empty());
            let test_subject = &subjects[test[0]];
            assert!(
                train.iter().all(|&i| &subjects[i] != test_subject),
                "training set contains the held-out subject"
            );
            assert_eq!(train.len() + test.len(), subjects.len());
        }
        assert!(subject_disjoint_folds(&[]).is_err());
        assert!(subject_disjoint_folds(&["s1".to_string()]).is_err());
    }

    #[test]
    fn wilcoxon_matches_known_values() {
        // All differences positive with n=5: the minimum two-sided p is
        // 2/2^5 = 0.0625, so n=5 cannot reach p<0.05 — a real property of
        // the exact test worth knowing when setting min_folds.
        let p = wilcoxon_signed_rank_p(&[2.0, 3.0, 4.0, 5.0, 6.0], &[1.0; 5]).unwrap();
        assert!((p - 0.0625).abs() < 1e-12, "p = {p}");
        // n=6 all-positive: 2/64 = 0.03125 < 0.05.
        let p = wilcoxon_signed_rank_p(&[2.0; 6], &[1.0; 6]).unwrap();
        assert!((p - 0.03125).abs() < 1e-12, "p = {p}");
        // Symmetric differences give a large p.
        let p = wilcoxon_signed_rank_p(&[1.0, -1.0, 2.0, -2.0, 3.0, -3.0], &[0.0; 6]).unwrap();
        assert!(p > 0.5, "p = {p}");
        // Identical inputs: undefined.
        assert!(wilcoxon_signed_rank_p(&[1.0; 5], &[1.0; 5]).is_none());
        assert!(wilcoxon_signed_rank_p(&[1.0], &[1.0, 2.0]).is_none());
        // Large n uses the normal approximation and stays in [0, 1].
        let a: Vec<f64> = (0..30).map(|i| 1.0 + i as f64 * 0.01).collect();
        let b: Vec<f64> = (0..30).map(|i| i as f64 * 0.01).collect();
        let p = wilcoxon_signed_rank_p(&a, &b).unwrap();
        assert!((0.0..=1.0).contains(&p) && p < 0.01, "p = {p}");
    }

    #[test]
    fn fold_scores_validate_inputs() {
        assert!(FoldScores::new(vec![], vec![], vec![]).is_err());
        assert!(FoldScores::new(vec![0.5], vec![], vec![0.5]).is_err());
        assert!(FoldScores::new(vec![f64::NAN], vec![0.0], vec![0.5]).is_err());
        let s = FoldScores::new(vec![0.5, 0.7], vec![0.1, 0.2], vec![0.5, 0.7]).unwrap();
        let (m, sd) = s.balanced_accuracy_mean_sd();
        assert!((m - 0.6).abs() < 1e-12);
        assert!((sd - 0.1).abs() < 1e-12);
    }
}
