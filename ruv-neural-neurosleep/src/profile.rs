//! The frozen, fully validated NeuroSleep analysis profile.

use ruv_neural_core::neurosleep::SleepState;
use ruv_neural_signal::aperiodic::{AperiodicConfig, AperiodicMode};
use ruv_neural_signal::connectivity::CoherenceConfig;
use ruv_neural_signal::neurosleep::{
    BandEdgePolicy, FrequencyBandConfig, ThetaPeakConfig, WelchConfig,
};
use ruv_neural_signal::quality::{AdcRails, EpochQualityConfig};

use crate::NeuroSleepError;

/// Named frequency bands and the denominator every relative power divides by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandProfile {
    /// Delta band.
    pub delta: FrequencyBandConfig,
    /// Theta band, also the coherence band.
    pub theta: FrequencyBandConfig,
    /// Alpha band.
    pub alpha: FrequencyBandConfig,
    /// The one explicit denominator for every relative power and band mean.
    /// There is no implicit "total power".
    pub relative_denominator: FrequencyBandConfig,
    /// Edge treatment applied identically to every band integral.
    pub edge_policy: BandEdgePolicy,
}

impl BandProfile {
    fn validate(&self) -> Result<(), NeuroSleepError> {
        for band in [
            self.delta,
            self.theta,
            self.alpha,
            self.relative_denominator,
        ] {
            band.validate()?;
            if band.low_hz < self.relative_denominator.low_hz
                || band.high_hz > self.relative_denominator.high_hz
            {
                return Err(NeuroSleepError::InvalidProfile(
                    "every reported band must lie inside the relative denominator",
                ));
            }
        }
        Ok(())
    }
}

/// Minimum accepted stage time before a stage's features are reported at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageSufficiency {
    /// Minimum accepted wake time, in seconds.
    pub wake_seconds: f64,
    /// Minimum accepted NREM time, in seconds.
    pub nrem_seconds: f64,
    /// Minimum accepted REM time, in seconds.
    pub rem_seconds: f64,
}

impl StageSufficiency {
    /// Threshold for one stage.
    pub fn minimum_for(&self, stage: SleepState) -> f64 {
        match stage {
            SleepState::Wake => self.wake_seconds,
            SleepState::Nrem => self.nrem_seconds,
            SleepState::Rem => self.rem_seconds,
        }
    }

    fn validate(&self) -> Result<(), NeuroSleepError> {
        if [self.wake_seconds, self.nrem_seconds, self.rem_seconds]
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(NeuroSleepError::InvalidProfile(
                "stage sufficiency thresholds must be finite and >= 0",
            ));
        }
        Ok(())
    }
}

/// Everything the night analysis is allowed to depend on. Constructed once,
/// validated once, and then frozen for the whole recording.
#[derive(Debug, Clone, PartialEq)]
pub struct NeuroSleepProfile {
    /// Acquisition sample rate in Hz.
    pub sample_rate_hz: f64,
    /// Expert scoring epoch length in seconds.
    pub epoch_seconds: f64,
    /// Welch averaging parameters.
    pub welch: WelchConfig,
    /// Masked coherence parameters.
    pub coherence: CoherenceConfig,
    /// Epoch admission gates.
    pub quality: EpochQualityConfig,
    /// Aperiodic fitting profile.
    pub aperiodic: AperiodicConfig,
    /// Periodic theta peak acceptance profile.
    pub theta_peak: ThetaPeakConfig,
    /// Band definitions and the relative-power denominator.
    pub bands: BandProfile,
    /// Stages for which aperiodic parameters are reported. Every other stage
    /// reports them as an explicit `not_applicable` null.
    pub report_aperiodic_for: Vec<SleepState>,
    /// Minimum fraction of epochs that must be admitted for the night to pass.
    pub minimum_valid_coverage: f64,
    /// Per-stage minimum accepted duration.
    pub sufficiency: StageSufficiency,
}

impl NeuroSleepProfile {
    /// Samples per expert scoring epoch. For the paper profile this is exactly
    /// 2500: 250 Hz for 10 seconds.
    pub fn epoch_samples(&self) -> usize {
        (self.sample_rate_hz * self.epoch_seconds).round() as usize
    }

    /// The 250 Hz, 10-second-epoch research profile.
    ///
    /// The Welch window is 1250 samples (5 s), giving a 0.2 Hz grid: fine enough
    /// for the theta peak profile's 0.25 Hz resolution bound while still leaving
    /// three averaging segments inside each 2500-sample epoch.
    ///
    /// The ADC rails below are a placeholder. Clipping is judged against the
    /// *recorder's* declared rails, so callers must overwrite them from their
    /// acquisition metadata rather than relying on this default.
    pub fn constantino_250hz() -> Self {
        Self {
            sample_rate_hz: 250.0,
            epoch_seconds: 10.0,
            welch: WelchConfig {
                window_samples: 1250,
                overlap_samples: 625,
                detrend_mean: true,
            },
            coherence: CoherenceConfig {
                window_samples: 1250,
                overlap_samples: 625,
                detrend_mean: true,
            },
            quality: EpochQualityConfig {
                adc_rails: AdcRails {
                    minimum_uv: -1000.0,
                    maximum_uv: 1000.0,
                    tolerance_uv: 0.5,
                },
                maximum_clipped_fraction: 0.001,
                flatline_seconds: 1.0,
                flatline_epsilon_uv: 1e-6,
                maximum_gap_seconds: 1.0,
                maximum_artifact_fraction: 0.20,
            },
            aperiodic: AperiodicConfig {
                low_hz: 2.0,
                high_hz: 40.0,
                mode: AperiodicMode::Knee,
                minimum_exponent: 0.1,
                maximum_exponent: 4.0,
                maximum_knee: 30.0,
                grid_points: 13,
                refinement_rounds: 8,
                peak_removal_iterations: 2,
                peak_removal_percentile: 0.025,
                peak_removal_tolerance_log10: 0.01,
                minimum_r_squared: 0.90,
                maximum_rmse_log10: 0.10,
            },
            theta_peak: ThetaPeakConfig {
                low_hz: 4.0,
                high_hz: 8.0,
                minimum_log10_prominence: 0.05,
                maximum_bin_spacing_hz: 0.25,
            },
            bands: BandProfile {
                delta: FrequencyBandConfig::half_open(0.5, 4.0),
                theta: FrequencyBandConfig::half_open(4.0, 8.0),
                alpha: FrequencyBandConfig::half_open(8.0, 13.0),
                relative_denominator: FrequencyBandConfig::half_open(0.5, 40.0),
                edge_policy: BandEdgePolicy::InterpolatedEdges,
            },
            report_aperiodic_for: vec![SleepState::Wake],
            minimum_valid_coverage: 0.90,
            sufficiency: StageSufficiency {
                wake_seconds: 1800.0,
                nrem_seconds: 1800.0,
                rem_seconds: 600.0,
            },
        }
    }

    /// Validate every nested configuration and every cross-configuration
    /// relationship, so no downstream stage can be reached with an unchecked
    /// parameter.
    pub fn validate(&self) -> Result<(), NeuroSleepError> {
        if !self.sample_rate_hz.is_finite() || self.sample_rate_hz <= 0.0 {
            return Err(NeuroSleepError::InvalidProfile(
                "sample rate must be finite and > 0",
            ));
        }
        if !self.epoch_seconds.is_finite() || self.epoch_seconds <= 0.0 {
            return Err(NeuroSleepError::InvalidProfile(
                "epoch duration must be finite and > 0",
            ));
        }
        let samples = self.epoch_samples();
        if samples == 0 {
            return Err(NeuroSleepError::InvalidProfile(
                "epoch must span at least one sample",
            ));
        }
        self.welch.validate()?;
        self.coherence.validate()?;
        self.quality.validate()?;
        self.aperiodic.validate()?;
        self.theta_peak.validate()?;
        self.bands.validate()?;
        self.sufficiency.validate()?;

        if !(0.0..=1.0).contains(&self.minimum_valid_coverage) {
            return Err(NeuroSleepError::InvalidProfile(
                "minimum valid coverage must lie in [0, 1]",
            ));
        }
        if self.welch.window_samples > samples || self.coherence.window_samples > samples {
            return Err(NeuroSleepError::InvalidProfile(
                "analysis window must fit inside one epoch",
            ));
        }
        // The theta peak is refined off a grid the Welch window produces, so the
        // two configurations have to agree before any epoch is touched.
        let spacing = self.welch.bin_spacing_hz(self.sample_rate_hz);
        if spacing > self.theta_peak.maximum_bin_spacing_hz {
            return Err(NeuroSleepError::InvalidProfile(
                "Welch resolution is coarser than the theta peak profile allows",
            ));
        }
        if self.bands.relative_denominator.high_hz > self.sample_rate_hz / 2.0
            || self.aperiodic.high_hz > self.sample_rate_hz / 2.0
        {
            return Err(NeuroSleepError::InvalidProfile(
                "requested bands exceed the Nyquist frequency",
            ));
        }
        Ok(())
    }
}
