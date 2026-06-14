//! The stimulus generator: validate → enforce safety → synthesize → receipt.

use serde::{Deserialize, Serialize};

use crate::params::StimulusParams;
use crate::receipt::DeliveryReceipt;
use crate::safety::SensorySafetyLimits;
use crate::waveform::StimulusWaveform;
use crate::StimError;

/// A verified, safety-checked stimulus ready to emit, paired with its receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifiedStimulus {
    /// The synthesized waveform.
    pub waveform: StimulusWaveform,
    /// The delivery receipt.
    pub receipt: DeliveryReceipt,
    /// True if safety limits clamped the commanded parameters.
    pub clamped: bool,
}

/// Generates verified stimuli under a fixed set of sensory-safety limits.
#[derive(Debug, Clone)]
pub struct StimulusGenerator {
    limits: SensorySafetyLimits,
}

impl StimulusGenerator {
    /// Create a generator with the given safety limits.
    pub fn new(limits: SensorySafetyLimits) -> Self {
        Self { limits }
    }

    /// Create a generator with conservative default limits.
    pub fn conservative() -> Self {
        Self::new(SensorySafetyLimits::default())
    }

    /// The active safety limits.
    pub fn limits(&self) -> &SensorySafetyLimits {
        &self.limits
    }

    /// Strict mode: synthesize only if the request is already within safety
    /// limits; otherwise return [`StimError::SafetyViolation`].
    pub fn generate_strict(
        &self,
        params: &StimulusParams,
        start_s: f64,
    ) -> Result<VerifiedStimulus, StimError> {
        params.validate()?;
        self.limits.check(params)?;
        let waveform = StimulusWaveform::synthesize(params);
        let receipt = DeliveryReceipt::for_waveform(&waveform, start_s);
        Ok(VerifiedStimulus {
            waveform,
            receipt,
            clamped: false,
        })
    }

    /// Conservative mode: clamp the request into the safe region (never
    /// rejecting on intensity/contrast), then synthesize. This is what the
    /// closed-loop controller uses for dosing so a too-strong command is
    /// *limited* rather than aborting the session.
    pub fn generate_clamped(
        &self,
        params: &StimulusParams,
        start_s: f64,
    ) -> Result<VerifiedStimulus, StimError> {
        params.validate()?;
        let mut safe = params.clone();
        let clamped = self.limits.clamp(&mut safe);
        // After clamping, the structural request must still be safe.
        self.limits.check(&safe)?;
        let waveform = StimulusWaveform::synthesize(&safe);
        let receipt = DeliveryReceipt::for_waveform(&waveform, start_s);
        Ok(VerifiedStimulus {
            waveform,
            receipt,
            clamped,
        })
    }
}
