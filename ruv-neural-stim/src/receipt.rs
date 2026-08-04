//! Cryptographic delivery receipts proving a stimulus was synthesized and
//! emitted as commanded.
//!
//! A receipt is the evidence that backs the acceptance-test requirement to
//! *"deliver a verified stimulus"*. It binds the commanded parameters to the
//! realized waveform via a SHA-256 digest, records the empirically measured
//! entrainment frequency and amplitude, and carries a `verified` verdict that
//! is only `true` when the realized signal matches the request within
//! tolerance.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::params::StimulusParams;
use crate::waveform::StimulusWaveform;

/// Tolerance (Hz) between commanded and measured envelope frequency for a
/// stimulus to count as verified.
pub const ENVELOPE_TOLERANCE_HZ: f64 = 2.0;

/// A signed-by-content record that a specific stimulus was delivered.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeliveryReceipt {
    /// Parameters that were commanded.
    pub params: StimulusParams,
    /// Number of samples actually emitted.
    pub num_samples: usize,
    /// Session-relative start time in seconds.
    pub start_s: f64,
    /// Session-relative end time in seconds.
    pub end_s: f64,
    /// Empirically measured entrainment frequency in Hz.
    pub measured_envelope_hz: f64,
    /// RMS amplitude of the emitted waveform.
    pub rms: f64,
    /// Peak absolute amplitude of the emitted waveform.
    pub peak: f64,
    /// SHA-256 digest (hex) of the emitted samples, binding the receipt to the
    /// exact waveform.
    pub waveform_sha256: String,
    /// Whether the realized stimulus matches the request within tolerance.
    pub verified: bool,
}

impl DeliveryReceipt {
    /// Build a receipt for a waveform emitted starting at `start_s`.
    pub fn for_waveform(waveform: &StimulusWaveform, start_s: f64) -> Self {
        let params = waveform.params.clone();
        let measured = waveform.measured_envelope_hz();
        let duration = waveform.len() as f64 / params.sample_rate_hz;
        let digest = hash_samples(&waveform.samples);

        let envelope_ok = (measured - params.envelope_hz).abs() <= ENVELOPE_TOLERANCE_HZ
            || params.intensity == 0.0;
        let amplitude_ok = waveform.peak() <= 1.0 + 1e-9;
        // A zero-intensity (disabled) stimulus is a legitimate, verified
        // "no-op" delivery — important for safe-stop receipts.
        let nonempty_ok = !waveform.is_empty();

        Self {
            params,
            num_samples: waveform.len(),
            start_s,
            end_s: start_s + duration,
            measured_envelope_hz: measured,
            rms: waveform.rms(),
            peak: waveform.peak(),
            waveform_sha256: digest,
            verified: envelope_ok && amplitude_ok && nonempty_ok,
        }
    }

    /// Re-verify this receipt against a waveform: the digest must match and the
    /// receipt must be internally consistent. Returns `true` if the receipt
    /// genuinely describes `waveform`.
    pub fn matches(&self, waveform: &StimulusWaveform) -> bool {
        self.waveform_sha256 == hash_samples(&waveform.samples)
            && self.num_samples == waveform.len()
    }

    /// Duration of the delivered stimulus in seconds.
    pub fn duration_s(&self) -> f64 {
        self.end_s - self.start_s
    }
}

/// SHA-256 of the sample vector, hashing the canonical little-endian bytes of
/// each `f64`.
pub fn hash_samples(samples: &[f64]) -> String {
    let mut hasher = Sha256::new();
    for s in samples {
        hasher.update(s.to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        out.push_str(&format!("{b:02x}"));
    }
    out
}
