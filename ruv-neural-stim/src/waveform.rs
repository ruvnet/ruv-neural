//! Sample-accurate stimulus waveform synthesis.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::params::{EnvelopeShape, Modality, StimulusParams};

/// A synthesized stimulus waveform: the normalized drive signal in `[-1, 1]`
/// (audio) or `[0, 1]` (light/haptic envelope), one sample per element.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StimulusWaveform {
    /// The realized parameters used to synthesize this waveform.
    pub params: StimulusParams,
    /// Output samples.
    pub samples: Vec<f64>,
}

impl StimulusWaveform {
    /// Synthesize a waveform from validated parameters.
    ///
    /// - Light / haptic: a unipolar envelope in `[0, 1]` at `envelope_hz`.
    /// - Audio: a bipolar carrier in `[-1, 1]` amplitude-modulated by a
    ///   `[0, 1]` envelope at `envelope_hz` (so the *envelope* carries the
    ///   40 Hz entrainment drive, which is what auditory cortex follows).
    ///
    /// A symmetric linear ramp of `ramp_s` is applied at onset and offset.
    pub fn synthesize(params: &StimulusParams) -> Self {
        let n = params.num_samples();
        let fs = params.sample_rate_hz;
        let mut samples = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f64 / fs;
            let env = envelope_value(params, t);
            let drive = match params.modality {
                Modality::Audio => {
                    let carrier_hz = params.carrier_hz.unwrap_or(params.envelope_hz);
                    let carrier = (2.0 * PI * carrier_hz * t).sin();
                    // AM with full modulation depth around the envelope.
                    carrier * env
                }
                Modality::Light | Modality::Haptic => env,
            };
            let ramp = ramp_gain(params, t, n, fs);
            samples.push(drive * params.intensity * ramp);
        }

        Self {
            params: params.clone(),
            samples,
        }
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Root-mean-square amplitude of the waveform.
    pub fn rms(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = self.samples.iter().map(|s| s * s).sum();
        (sum_sq / self.samples.len() as f64).sqrt()
    }

    /// Peak absolute amplitude.
    pub fn peak(&self) -> f64 {
        self.samples.iter().fold(0.0f64, |m, s| m.max(s.abs()))
    }

    /// Estimate the dominant envelope frequency (Hz) actually present in the
    /// waveform, by counting envelope zero-up-crossings around the mean of the
    /// rectified signal. This is the empirical check that backs a "verified
    /// stimulus": the realized entrainment frequency, independent of the
    /// requested parameter.
    pub fn measured_envelope_hz(&self) -> f64 {
        let fs = self.params.sample_rate_hz;
        // Work on the analytic envelope proxy: rectified signal smoothed by a
        // short moving average to suppress the audio carrier, leaving the
        // modulation envelope.
        let rect: Vec<f64> = self.samples.iter().map(|s| s.abs()).collect();
        if rect.len() < 4 {
            return 0.0;
        }
        let win = ((fs / (self.params.envelope_hz * 4.0)).round() as usize).clamp(1, rect.len());
        let env_full = moving_average(&rect, win);

        // Exclude the onset/offset ramp regions: their reduced amplitude can
        // sit below the global mean and drop otherwise-valid cycles, biasing
        // the estimate low. Analyze the steady-state interior only.
        let ramp_n = (self.params.ramp_s * fs).round() as usize;
        let (lo, hi) = if env_full.len() > 2 * ramp_n + 4 {
            (ramp_n, env_full.len() - ramp_n)
        } else {
            (0, env_full.len())
        };
        let env = &env_full[lo..hi];
        let mean = env.iter().sum::<f64>() / env.len() as f64;

        // Count rising crossings of the mean.
        let mut crossings = 0usize;
        for w in env.windows(2) {
            if w[0] <= mean && w[1] > mean {
                crossings += 1;
            }
        }
        let duration = env.len() as f64 / fs;
        if duration <= 0.0 {
            0.0
        } else {
            crossings as f64 / duration
        }
    }
}

/// The unipolar `[0,1]` modulation envelope at time `t`.
fn envelope_value(params: &StimulusParams, t: f64) -> f64 {
    let phase = (params.envelope_hz * t).fract(); // [0,1) within one cycle
    match params.shape {
        EnvelopeShape::Sine => 0.5 * (1.0 - (2.0 * PI * params.envelope_hz * t).cos()),
        EnvelopeShape::Square => {
            if phase < params.duty_cycle {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Linear onset/offset ramp gain in `[0,1]`.
fn ramp_gain(params: &StimulusParams, t: f64, n: usize, fs: f64) -> f64 {
    if params.ramp_s <= 0.0 {
        return 1.0;
    }
    let total = n as f64 / fs;
    let up = (t / params.ramp_s).min(1.0);
    let down = ((total - t) / params.ramp_s).min(1.0);
    up.min(down).max(0.0)
}

/// Centered moving average with a window of `win` samples.
fn moving_average(x: &[f64], win: usize) -> Vec<f64> {
    if win <= 1 {
        return x.to_vec();
    }
    let mut out = Vec::with_capacity(x.len());
    let half = win / 2;
    for i in 0..x.len() {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(x.len());
        let slice = &x[lo..hi];
        out.push(slice.iter().sum::<f64>() / slice.len() as f64);
    }
    out
}
