//! Sensory-safety limits for non-invasive stimulation.
//!
//! These limits are **conservative engineering guardrails**, not clinical
//! dosing. They exist so the closed-loop controller can never command a
//! physically unsafe sensory stimulus. The dominant hazard for visual
//! flicker is photosensitive (reflex) seizure provocation; for audio it is
//! sound-pressure (hearing) exposure; for haptics it is actuator drive.
//!
//! See `docs/adr/0010-sensory-safety-limits.md`.

use serde::{Deserialize, Serialize};

use crate::params::{Modality, StimulusParams};
use crate::StimError;

/// The frequency band most provocative for photosensitive epilepsy.
///
/// International consensus (Harding/Fisher et al.) identifies roughly
/// **15–25 Hz** full-field high-contrast flicker as maximally provocative,
/// with risk extending across ~3–60 Hz. 40 Hz GENUS sits above the peak but
/// is still within the cautionary band, so luminance contrast is capped and
/// an explicit photosensitivity screen is required.
pub const PHOTIC_PROVOCATIVE_HZ: (f64, f64) = (15.0, 25.0);

/// Wider cautionary flicker band where luminance must be limited.
pub const PHOTIC_CAUTION_HZ: (f64, f64) = (3.0, 60.0);

/// Per-modality safety limits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorySafetyLimits {
    /// Maximum normalized intensity `[0,1]` the controller may command.
    pub max_intensity: f64,
    /// Maximum luminance contrast `[0,1]` for visual flicker inside the
    /// cautionary band. Full-field 100 % contrast flicker is the hazard, so
    /// this is held well below 1.0.
    pub max_photic_contrast: f64,
    /// Maximum sustained sound exposure for audio, in dB SPL.
    pub max_audio_db_spl: f64,
    /// Maximum fraction of a session that may be actively stimulating
    /// (duty limit across the whole protocol), in `[0,1]`.
    pub max_session_duty: f64,
    /// Whether a photosensitivity contraindication screen has been completed
    /// and cleared for this subject. Light stimulation in the cautionary band
    /// is refused unless this is `true`.
    pub photosensitivity_screen_cleared: bool,
}

impl Default for SensorySafetyLimits {
    /// Conservative wellness-grade defaults.
    fn default() -> Self {
        Self {
            max_intensity: 0.6,
            max_photic_contrast: 0.5,
            max_audio_db_spl: 75.0,
            max_session_duty: 0.5,
            photosensitivity_screen_cleared: false,
        }
    }
}

impl SensorySafetyLimits {
    /// Limits for a subject who has completed and cleared a photosensitivity
    /// screen (still conservative on intensity/contrast).
    pub fn screened() -> Self {
        Self {
            photosensitivity_screen_cleared: true,
            ..Self::default()
        }
    }

    /// True if `envelope_hz` falls in the cautionary photic band.
    pub fn in_photic_caution_band(hz: f64) -> bool {
        hz >= PHOTIC_CAUTION_HZ.0 && hz <= PHOTIC_CAUTION_HZ.1
    }

    /// True if `envelope_hz` falls in the maximally provocative band.
    pub fn in_photic_provocative_band(hz: f64) -> bool {
        hz >= PHOTIC_PROVOCATIVE_HZ.0 && hz <= PHOTIC_PROVOCATIVE_HZ.1
    }

    /// Estimate the SPL that audio at the given commanded intensity would
    /// produce, under a simple linear-headroom model where `intensity == 1.0`
    /// maps to `max_audio_db_spl + headroom`.
    fn projected_audio_db(&self, intensity: f64) -> f64 {
        // 20*log10(intensity) below the cap's reference; intensity 1.0 == cap.
        if intensity <= 0.0 {
            return 0.0;
        }
        self.max_audio_db_spl + 20.0 * intensity.log10()
    }

    /// Check a stimulus against these limits. Returns `Ok(())` if safe, or a
    /// descriptive [`StimError::SafetyViolation`] otherwise.
    pub fn check(&self, params: &StimulusParams) -> Result<(), StimError> {
        if params.intensity > self.max_intensity + 1e-9 {
            return Err(StimError::SafetyViolation(format!(
                "intensity {:.3} exceeds max_intensity {:.3}",
                params.intensity, self.max_intensity
            )));
        }

        match params.modality {
            // A zero-intensity stimulus emits nothing physical, so it is always
            // safe (this is the canonical safe-stop / disabled-channel case).
            _ if params.intensity <= 0.0 => return Ok(()),
            Modality::Light => {
                if Self::in_photic_caution_band(params.envelope_hz) {
                    if !self.photosensitivity_screen_cleared {
                        return Err(StimError::SafetyViolation(format!(
                            "light flicker at {:.1} Hz is in the photosensitivity caution band \
                             ({:.0}-{:.0} Hz) and requires a cleared photosensitivity screen",
                            params.envelope_hz, PHOTIC_CAUTION_HZ.0, PHOTIC_CAUTION_HZ.1
                        )));
                    }
                    // Even when screened, cap effective luminance contrast.
                    if params.intensity > self.max_photic_contrast + 1e-9 {
                        return Err(StimError::SafetyViolation(format!(
                            "photic contrast {:.3} exceeds max_photic_contrast {:.3} in caution band",
                            params.intensity, self.max_photic_contrast
                        )));
                    }
                }
            }
            Modality::Audio => {
                let db = self.projected_audio_db(params.intensity);
                if db > self.max_audio_db_spl + 1e-9 {
                    return Err(StimError::SafetyViolation(format!(
                        "projected {db:.1} dB SPL exceeds max {:.1} dB SPL",
                        self.max_audio_db_spl
                    )));
                }
            }
            Modality::Haptic => { /* intensity cap already enforced above */ }
        }
        Ok(())
    }

    /// Clamp a stimulus into the safe region in-place, returning whether any
    /// clamping occurred. Used by the controller's conservative dosing so a
    /// request is *limited* rather than rejected where that is safe to do.
    /// Note: this cannot clear a missing photosensitivity screen — a
    /// light stimulus in the caution band without a screen is forced to zero
    /// intensity (effectively disabled) and reported.
    pub fn clamp(&self, params: &mut StimulusParams) -> bool {
        let original = params.intensity;
        params.intensity = params.intensity.min(self.max_intensity);

        if params.modality == Modality::Light && Self::in_photic_caution_band(params.envelope_hz) {
            if !self.photosensitivity_screen_cleared {
                params.intensity = 0.0;
            } else {
                params.intensity = params.intensity.min(self.max_photic_contrast);
            }
        }
        (original - params.intensity).abs() > 1e-12
    }
}
