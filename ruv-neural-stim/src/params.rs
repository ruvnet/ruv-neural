//! Stimulus parameters: modality, envelope shape, intensity, and timing.
//!
//! All stimulation in this crate is **non-invasive sensory** entrainment —
//! light, sound, and touch delivered through ordinary external channels. No
//! transcranial or implanted modality (TMS, tDCS/tACS, focused ultrasound,
//! DBS, VNS) is modeled here; those are medical-device territory and are
//! explicitly out of scope (see `docs/adr/0001-scope.md`).

use serde::{Deserialize, Serialize};

/// A safe, external sensory channel used for entrainment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Visual flicker (e.g. an LED panel or screen) driving visual cortex.
    Light,
    /// Amplitude-modulated audio envelope driving auditory cortex.
    Audio,
    /// Vibrotactile actuator driving somatosensory cortex.
    Haptic,
}

impl Modality {
    /// All modalities, in a stable order.
    pub const ALL: [Modality; 3] = [Modality::Light, Modality::Audio, Modality::Haptic];

    /// Short stable identifier used in receipts and audit logs.
    pub fn tag(&self) -> &'static str {
        match self {
            Modality::Light => "light",
            Modality::Audio => "audio",
            Modality::Haptic => "haptic",
        }
    }

    /// Human-readable cortical target for documentation/UI.
    pub fn cortical_target(&self) -> &'static str {
        match self {
            Modality::Light => "visual cortex",
            Modality::Audio => "auditory cortex",
            Modality::Haptic => "somatosensory cortex",
        }
    }
}

/// Shape of the modulation envelope applied at the entrainment frequency.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EnvelopeShape {
    /// Sinusoidal amplitude modulation — smooth, low harmonic content.
    Sine,
    /// Square (on/off) modulation at the configured duty cycle — strongest
    /// entrainment drive but richest in harmonics. Used by classic 40 Hz
    /// GENUS light flicker.
    Square,
}

/// The canonical gamma-entrainment frequency (GENUS) in Hz.
pub const GAMMA_ENTRAINMENT_HZ: f64 = 40.0;

/// Parameters fully describing one stimulus to synthesize.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StimulusParams {
    /// Which external sensory channel.
    pub modality: Modality,
    /// Entrainment (envelope) frequency in Hz — 40 Hz for gamma GENUS.
    pub envelope_hz: f64,
    /// Carrier frequency in Hz. For [`Modality::Audio`] this is the audible
    /// tone that is amplitude-modulated by `envelope_hz`. For light/haptic
    /// the carrier is the envelope itself, so this is `None`.
    pub carrier_hz: Option<f64>,
    /// Envelope shape.
    pub shape: EnvelopeShape,
    /// Duty cycle in `[0, 1]` for [`EnvelopeShape::Square`] (fraction of each
    /// cycle that is "on"). Ignored for sine.
    pub duty_cycle: f64,
    /// Commanded intensity in `[0, 1]`, a fraction of the modality's
    /// safety-capped maximum (not an absolute physical unit).
    pub intensity: f64,
    /// Linear ramp-in/ramp-out duration in seconds, to avoid startle / onset
    /// transients. Applied symmetrically at start and end.
    pub ramp_s: f64,
    /// Total stimulus duration in seconds.
    pub duration_s: f64,
    /// Output sample rate in Hz for the synthesized waveform.
    pub sample_rate_hz: f64,
}

impl StimulusParams {
    /// A conservative 40 Hz GENUS preset for the given modality at a modest
    /// intensity, with a 0.5 s ramp and a 50 % duty cycle.
    pub fn gamma_40hz(modality: Modality, duration_s: f64) -> Self {
        // Audio needs an audible carrier and a sample rate that satisfies
        // Nyquist for it; light/haptic drive the envelope directly at a modest
        // rate.
        let (carrier_hz, sample_rate_hz) = match modality {
            Modality::Audio => (Some(1_000.0), 8_000.0),
            Modality::Light | Modality::Haptic => (None, 1_000.0),
        };
        let shape = match modality {
            // Square flicker is the canonical GENUS light drive; audio/haptic
            // use a smoother sinusoidal envelope.
            Modality::Light => EnvelopeShape::Square,
            Modality::Audio | Modality::Haptic => EnvelopeShape::Sine,
        };
        Self {
            modality,
            envelope_hz: GAMMA_ENTRAINMENT_HZ,
            carrier_hz,
            shape,
            duty_cycle: 0.5,
            intensity: 0.4,
            ramp_s: 0.5,
            duration_s,
            sample_rate_hz,
        }
    }

    /// Number of output samples this stimulus will produce.
    pub fn num_samples(&self) -> usize {
        (self.duration_s * self.sample_rate_hz).round().max(0.0) as usize
    }

    /// Override the commanded intensity, clamped to `[0, 1]`, returning self.
    pub fn with_intensity(mut self, intensity: f64) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Validate structural sanity of the parameters (independent of safety
    /// limits, which are enforced separately by the safety module).
    pub fn validate(&self) -> Result<(), crate::StimError> {
        if !self.sample_rate_hz.is_finite() || self.sample_rate_hz <= 0.0 {
            return Err(crate::StimError::Params(
                "sample_rate_hz must be finite and positive".into(),
            ));
        }
        if !self.envelope_hz.is_finite() || self.envelope_hz <= 0.0 {
            return Err(crate::StimError::Params(
                "envelope_hz must be finite and positive".into(),
            ));
        }
        // Nyquist: the envelope (and any carrier) must be representable.
        let max_freq = self
            .carrier_hz
            .unwrap_or(self.envelope_hz)
            .max(self.envelope_hz);
        if max_freq * 2.0 > self.sample_rate_hz {
            return Err(crate::StimError::Params(format!(
                "sample_rate_hz {} too low for {} Hz content (Nyquist)",
                self.sample_rate_hz, max_freq
            )));
        }
        if !(0.0..=1.0).contains(&self.duty_cycle) {
            return Err(crate::StimError::Params(
                "duty_cycle must be in [0,1]".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.intensity) {
            return Err(crate::StimError::Params(
                "intensity must be in [0,1]".into(),
            ));
        }
        if !self.duration_s.is_finite() || self.duration_s <= 0.0 {
            return Err(crate::StimError::Params(
                "duration_s must be finite and positive".into(),
            ));
        }
        if self.ramp_s < 0.0 || self.ramp_s * 2.0 > self.duration_s {
            return Err(crate::StimError::Params(
                "ramp_s must be >= 0 and total ramp must not exceed duration".into(),
            ));
        }
        Ok(())
    }
}
