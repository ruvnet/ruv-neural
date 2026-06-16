//! # ruv-neural-stim
//!
//! Multi-modal **non-invasive sensory** neuromodulation stimulus synthesis for
//! the rUv Neural closed-loop platform.
//!
//! This crate generates verified entrainment stimuli on three safe external
//! channels — **40 Hz light**, **40 Hz amplitude-modulated audio**, and
//! **40 Hz haptics** — under explicit sensory-safety limits, and emits a
//! cryptographic [`DeliveryReceipt`] proving exactly what was delivered.
//!
//! ## Scope & boundary
//!
//! Only safe external sensory channels are modeled. Transcranial / implanted
//! modalities (TMS, tDCS/tACS, focused ultrasound, DBS, VNS) are **out of
//! scope** — they are medical-device territory requiring clinical validation,
//! dosing controls, contraindication screening, and regulatory review. See
//! `docs/adr/0001-scope.md` and `docs/adr/0002-sensory-modalities.md`.
//!
//! ## Pipeline
//!
//! ```text
//!   StimulusParams ──validate──> SensorySafetyLimits ──synthesize──> StimulusWaveform
//!                                                                           │
//!                                                                           v
//!                                                                   DeliveryReceipt
//!                                                                  (SHA-256 + verified)
//! ```
//!
//! ```
//! use ruv_neural_stim::{StimulusGenerator, StimulusParams, Modality};
//!
//! let gen = StimulusGenerator::conservative();
//! let params = StimulusParams::gamma_40hz(Modality::Audio, 2.0);
//! let stim = gen.generate_clamped(&params, 0.0).unwrap();
//! assert!(stim.receipt.verified);
//! assert!((stim.receipt.measured_envelope_hz - 40.0).abs() < 2.0);
//! ```

pub mod generator;
pub mod params;
pub mod receipt;
pub mod safety;
pub mod waveform;

pub use generator::{StimulusGenerator, VerifiedStimulus};
pub use params::{EnvelopeShape, Modality, StimulusParams, GAMMA_ENTRAINMENT_HZ};
pub use receipt::{DeliveryReceipt, ENVELOPE_TOLERANCE_HZ};
pub use safety::{SensorySafetyLimits, PHOTIC_CAUTION_HZ, PHOTIC_PROVOCATIVE_HZ};
pub use waveform::StimulusWaveform;

use thiserror::Error;

/// Errors produced by stimulus synthesis and safety enforcement.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum StimError {
    /// Structurally invalid parameters (bad rate, Nyquist violation, etc.).
    #[error("Stimulus parameter error: {0}")]
    Params(String),

    /// The request violates a sensory-safety limit.
    #[error("Sensory safety violation: {0}")]
    SafetyViolation(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn gamma_preset_is_40hz() {
        for m in Modality::ALL {
            let p = StimulusParams::gamma_40hz(m, 1.0);
            assert_abs_diff_eq!(p.envelope_hz, 40.0);
        }
    }

    #[test]
    fn audio_carrier_present_light_haptic_absent() {
        assert!(StimulusParams::gamma_40hz(Modality::Audio, 1.0)
            .carrier_hz
            .is_some());
        assert!(StimulusParams::gamma_40hz(Modality::Light, 1.0)
            .carrier_hz
            .is_none());
        assert!(StimulusParams::gamma_40hz(Modality::Haptic, 1.0)
            .carrier_hz
            .is_none());
    }

    #[test]
    fn params_validate_rejects_nyquist_violation() {
        let mut p = StimulusParams::gamma_40hz(Modality::Audio, 1.0);
        p.sample_rate_hz = 100.0; // far below 2*10kHz carrier
        assert!(p.validate().is_err());
    }

    #[test]
    fn params_validate_rejects_overlong_ramp() {
        let mut p = StimulusParams::gamma_40hz(Modality::Haptic, 1.0);
        p.ramp_s = 0.8; // 2*0.8 > 1.0
        assert!(p.validate().is_err());
    }

    #[test]
    fn waveform_num_samples_matches_duration() {
        let p = StimulusParams::gamma_40hz(Modality::Haptic, 2.0);
        let w = StimulusWaveform::synthesize(&p);
        assert_eq!(w.len(), 2000); // 2 s * 1000 Hz
    }

    #[test]
    fn waveform_measures_40hz_envelope_haptic() {
        let p = StimulusParams::gamma_40hz(Modality::Haptic, 3.0);
        let w = StimulusWaveform::synthesize(&p);
        assert_abs_diff_eq!(w.measured_envelope_hz(), 40.0, epsilon = 2.0);
    }

    #[test]
    fn waveform_measures_40hz_envelope_audio() {
        // The auditory entrainment drive is the 40 Hz envelope, not the carrier.
        let mut p = StimulusParams::gamma_40hz(Modality::Audio, 3.0);
        p.sample_rate_hz = 44_100.0;
        let w = StimulusWaveform::synthesize(&p);
        assert_abs_diff_eq!(w.measured_envelope_hz(), 40.0, epsilon = 2.0);
    }

    #[test]
    fn waveform_peak_within_unit_range() {
        let p = StimulusParams::gamma_40hz(Modality::Audio, 1.0).with_intensity(1.0);
        let w = StimulusWaveform::synthesize(&p);
        assert!(w.peak() <= 1.0 + 1e-9);
    }

    #[test]
    fn ramp_starts_and_ends_at_zero() {
        let p = StimulusParams::gamma_40hz(Modality::Haptic, 2.0);
        let w = StimulusWaveform::synthesize(&p);
        assert_abs_diff_eq!(w.samples[0], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(*w.samples.last().unwrap(), 0.0, epsilon = 1e-3);
    }

    #[test]
    fn safety_blocks_unscreened_light_in_caution_band() {
        let limits = SensorySafetyLimits::default(); // screen not cleared
        let p = StimulusParams::gamma_40hz(Modality::Light, 1.0);
        assert!(limits.check(&p).is_err());
    }

    #[test]
    fn safety_allows_screened_light_with_capped_contrast() {
        let limits = SensorySafetyLimits::screened();
        let p = StimulusParams::gamma_40hz(Modality::Light, 1.0).with_intensity(0.4);
        assert!(limits.check(&p).is_ok());
    }

    #[test]
    fn safety_caps_loud_audio() {
        let limits = SensorySafetyLimits::default();
        let p = StimulusParams::gamma_40hz(Modality::Audio, 1.0).with_intensity(1.0);
        // intensity 1.0 maps to exactly the cap → ok; raise the cap model by
        // exceeding max_intensity instead.
        let _ = p;
        let mut loud = StimulusParams::gamma_40hz(Modality::Audio, 1.0);
        loud.intensity = 0.95; // above default max_intensity 0.6
        assert!(limits.check(&loud).is_err());
    }

    #[test]
    fn clamp_disables_unscreened_light() {
        let limits = SensorySafetyLimits::default();
        let mut p = StimulusParams::gamma_40hz(Modality::Light, 1.0);
        let clamped = limits.clamp(&mut p);
        assert!(clamped);
        assert_eq!(p.intensity, 0.0);
        assert!(limits.check(&p).is_ok());
    }

    #[test]
    fn generator_strict_rejects_unsafe() {
        let gen = StimulusGenerator::conservative();
        let mut p = StimulusParams::gamma_40hz(Modality::Haptic, 1.0);
        p.intensity = 0.99;
        assert!(gen.generate_strict(&p, 0.0).is_err());
    }

    #[test]
    fn generator_clamped_limits_and_verifies() {
        let gen = StimulusGenerator::conservative();
        let mut p = StimulusParams::gamma_40hz(Modality::Haptic, 2.0);
        p.intensity = 0.99;
        let stim = gen.generate_clamped(&p, 0.0).unwrap();
        assert!(stim.clamped);
        assert!(stim.waveform.params.intensity <= gen.limits().max_intensity + 1e-9);
        assert!(stim.receipt.verified);
    }

    #[test]
    fn receipt_binds_to_waveform() {
        let gen = StimulusGenerator::conservative();
        let p = StimulusParams::gamma_40hz(Modality::Haptic, 1.0);
        let stim = gen.generate_clamped(&p, 5.0).unwrap();
        assert!(stim.receipt.matches(&stim.waveform));
        assert_abs_diff_eq!(stim.receipt.start_s, 5.0);
        assert_abs_diff_eq!(stim.receipt.duration_s(), 1.0, epsilon = 1e-9);

        // Tampering with the waveform breaks the binding.
        let mut tampered = stim.waveform.clone();
        tampered.samples[10] += 0.5;
        assert!(!stim.receipt.matches(&tampered));
    }

    #[test]
    fn zero_intensity_safe_stop_is_verified_noop() {
        // A disabled stimulus (safe-stop) is a legitimate verified delivery.
        let p = StimulusParams::gamma_40hz(Modality::Haptic, 1.0).with_intensity(0.0);
        let w = StimulusWaveform::synthesize(&p);
        let r = DeliveryReceipt::for_waveform(&w, 0.0);
        assert!(r.verified);
        assert_abs_diff_eq!(r.rms, 0.0);
    }
}
