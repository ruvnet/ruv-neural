//! Night-level stage duration and bout aggregation.

use ruv_neural_core::neurosleep::{FeatureValue, NullReason, SleepState, StageSummary};
use std::collections::BTreeMap;

use crate::profile::NeuroSleepProfile;
use crate::stage::{null, observed};
use crate::ExpertEpoch;

/// Accepted stage time and bout lengths, both in seconds.
///
/// A *bout* is a maximal run of consecutive epochs that were admitted **and**
/// carry the same expert stage label. A rejected epoch ends the run it sits in
/// and does not start a new one, so an artefact-corrupted epoch splits a bout
/// rather than silently bridging it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StageAggregate {
    durations: BTreeMap<u8, f64>,
    bouts: BTreeMap<u8, Vec<f64>>,
}

impl StageAggregate {
    /// Aggregate accepted epochs into per-stage durations and bouts.
    pub fn new(epochs: &[ExpertEpoch], accepted: &[bool], epoch_seconds: f64) -> Self {
        let mut aggregate = Self::default();
        let mut run: Option<(u8, f64)> = None;
        for (epoch, admitted) in epochs.iter().zip(accepted) {
            let key = admitted.then_some(epoch.state as u8);
            if let Some(stage) = key {
                *aggregate.durations.entry(stage).or_insert(0.0) += epoch_seconds;
            }
            if key == run.map(|(stage, _)| stage) {
                if let Some(current) = run.as_mut() {
                    current.1 += epoch_seconds;
                }
            } else {
                if let Some((stage, seconds)) = run.take() {
                    aggregate.bouts.entry(stage).or_default().push(seconds);
                }
                run = key.map(|stage| (stage, epoch_seconds));
            }
        }
        if let Some((stage, seconds)) = run {
            aggregate.bouts.entry(stage).or_default().push(seconds);
        }
        aggregate
    }

    /// Total admitted seconds for a stage.
    pub fn duration(&self, stage: SleepState) -> f64 {
        self.durations
            .get(&(stage as u8))
            .copied()
            .unwrap_or_default()
    }

    /// Bout lengths in seconds, in order of occurrence.
    pub fn bouts(&self, stage: SleepState) -> &[f64] {
        self.bouts
            .get(&(stage as u8))
            .map_or(&[][..], Vec::as_slice)
    }

    /// Whether a stage cleared its sufficiency threshold.
    fn sufficient(&self, stage: SleepState, profile: &NeuroSleepProfile) -> bool {
        self.duration(stage) >= profile.sufficiency.minimum_for(stage)
    }

    /// Build the contract's stage summary, nulling anything the sufficiency
    /// thresholds do not support.
    pub fn summary(&self, profile: &NeuroSleepProfile) -> StageSummary {
        StageSummary {
            wake_duration: self.duration_value(SleepState::Wake, profile),
            nrem_duration: self.duration_value(SleepState::Nrem, profile),
            nrem_mean_bout_duration: self.mean_bout_value(SleepState::Nrem, profile),
            rem_duration: self.duration_value(SleepState::Rem, profile),
            rem_bout_count: self.bout_count_value(SleepState::Rem, profile),
        }
    }

    fn duration_value(&self, stage: SleepState, profile: &NeuroSleepProfile) -> FeatureValue {
        if self.sufficient(stage, profile) {
            observed(self.duration(stage), "s")
        } else {
            null(NullReason::InsufficientStageDuration)
        }
    }

    fn mean_bout_value(&self, stage: SleepState, profile: &NeuroSleepProfile) -> FeatureValue {
        let bouts = self.bouts(stage);
        if !self.sufficient(stage, profile) || bouts.is_empty() {
            return null(NullReason::InsufficientStageDuration);
        }
        observed(bouts.iter().sum::<f64>() / bouts.len() as f64, "s")
    }

    fn bout_count_value(&self, stage: SleepState, profile: &NeuroSleepProfile) -> FeatureValue {
        if !self.sufficient(stage, profile) {
            return null(NullReason::InsufficientStageDuration);
        }
        observed(self.bouts(stage).len() as f64, "count")
    }
}
