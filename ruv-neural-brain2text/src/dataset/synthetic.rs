//! Synthetic SpanishBCBL-like recording generator.
//!
//! The real SpanishBCBL data is ~262 GB and CC BY-NC 4.0, so it cannot live in
//! the repo. To exercise and test the full pipeline we generate recordings with
//! the same *structure* (a continuous multi-channel signal plus a keystroke
//! timeline) and, crucially, with **learnable** structure: each character is
//! assigned a fixed per-channel spatial template, so a decoder that captures
//! the spatial pattern around each keypress can recover the typed text. This is
//! a deliberately easy stand-in for real neural data — it validates the wiring,
//! the metrics, and that the optimizer actually improves accuracy, not that the
//! published Brain2Qwerty numbers reproduce.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use ruv_neural_core::signal::MultiChannelTimeSeries;

use super::{Modality, Recording};
use crate::events::{EventTimeline, KeystrokeEvent, Sentence};

/// Parameters controlling synthetic generation.
#[derive(Debug, Clone)]
pub struct SyntheticParams {
    /// Number of sensor channels.
    pub num_channels: usize,
    /// Sampling rate in Hz (real EEG/MEG is 1 kHz).
    pub sample_rate_hz: f64,
    /// Inter-keystroke interval in seconds.
    pub keystroke_interval_s: f64,
    /// Gap between sentences in seconds.
    pub sentence_gap_s: f64,
    /// Per-channel additive Gaussian noise standard deviation.
    pub noise_std: f64,
    /// Amplitude of the per-character spatial template.
    pub signal_amplitude: f64,
    /// Modality label to stamp onto the recording.
    pub modality: Modality,
}

impl Default for SyntheticParams {
    fn default() -> Self {
        Self {
            num_channels: 32,
            sample_rate_hz: 1000.0,
            keystroke_interval_s: 0.3,
            sentence_gap_s: 1.0,
            noise_std: 0.4,
            signal_amplitude: 1.0,
            modality: Modality::Eeg,
        }
    }
}

/// Approximate Gaussian sample via sum of uniforms (Irwin–Hall, n=12).
fn gauss(rng: &mut StdRng) -> f64 {
    let mut s = 0.0;
    for _ in 0..12 {
        s += rng.gen::<f64>();
    }
    s - 6.0
}

/// Generate a synthetic recording from a list of target sentences.
///
/// Determinism is controlled by `seed`. Each distinct character in the corpus
/// gets a stable random spatial template; keystrokes are laid out at a fixed
/// cadence and a temporal bump of that template is written into the signal.
pub fn generate(sentences: &[&str], params: &SyntheticParams, seed: u64) -> Recording {
    let mut rng = StdRng::seed_from_u64(seed);

    // Assign a per-channel template to every character we will need.
    let mut vocab: Vec<char> = sentences
        .iter()
        .flat_map(|s| s.chars())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    vocab.sort_unstable();

    let mut templates: std::collections::HashMap<char, Vec<f64>> =
        std::collections::HashMap::new();
    for &c in &vocab {
        let t: Vec<f64> = (0..params.num_channels)
            .map(|_| gauss(&mut rng) * params.signal_amplitude)
            .collect();
        templates.insert(c, t);
    }

    let sr = params.sample_rate_hz;
    let interval = params.keystroke_interval_s;
    let gap = params.sentence_gap_s;

    // First pass: lay out keystroke onsets and total duration.
    let mut timeline = EventTimeline::default();
    let mut cursor_s = 0.5; // small lead-in
    for (sid, text) in sentences.iter().enumerate() {
        let mut keystrokes = Vec::new();
        for ch in text.chars() {
            keystrokes.push(KeystrokeEvent {
                onset_s: cursor_s,
                character: ch,
                sentence_id: sid,
            });
            cursor_s += interval;
        }
        cursor_s += gap;
        timeline.sentences.push(Sentence {
            id: sid,
            text: (*text).to_string(),
            keystrokes,
        });
    }
    let total_s = cursor_s + 0.5;
    let num_samples = (total_s * sr).ceil() as usize;

    // Second pass: synthesize the signal.
    let mut data = vec![vec![0.0f64; num_samples]; params.num_channels];
    for ch in 0..params.num_channels {
        for s in 0..num_samples {
            data[ch][s] = gauss(&mut rng) * params.noise_std;
        }
    }

    // Temporal bump: a short raised-cosine centered just after the keypress.
    let bump_half = (0.08 * sr) as i64; // ~80 ms each side
    for ks in timeline.all_keystrokes() {
        let template = &templates[&ks.character];
        let center = (ks.onset_s * sr) as i64 + (0.05 * sr) as i64; // peak ~50 ms post
        for d in -bump_half..=bump_half {
            let idx = center + d;
            if idx < 0 || idx as usize >= num_samples {
                continue;
            }
            let frac = d as f64 / bump_half as f64; // -1..1
            let w = 0.5 * (1.0 + (std::f64::consts::PI * frac).cos()); // raised cosine
            for ch in 0..params.num_channels {
                data[ch][idx as usize] += template[ch] * w;
            }
        }
    }

    let series = MultiChannelTimeSeries::new(data, sr, 0.0)
        .expect("synthetic series dimensions are valid by construction");

    Recording {
        series,
        timeline,
        modality: params.modality,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_structured() {
        let sents = ["hola mundo", "que tal"];
        let p = SyntheticParams::default();
        let a = generate(&sents, &p, 7);
        let b = generate(&sents, &p, 7);
        // Determinism.
        assert_eq!(a.series.data[0][100], b.series.data[0][100]);
        // Structure: keystroke count matches characters.
        let nchars: usize = sents.iter().map(|s| s.chars().count()).sum();
        assert_eq!(a.timeline.num_keystrokes(), nchars);
        assert_eq!(a.series.num_channels, p.num_channels);
        assert!(a.series.num_samples > 0);
    }

    #[test]
    fn different_seeds_differ() {
        let sents = ["abc"];
        let p = SyntheticParams::default();
        let a = generate(&sents, &p, 1);
        let b = generate(&sents, &p, 2);
        assert_ne!(a.series.data[0][50], b.series.data[0][50]);
    }
}
