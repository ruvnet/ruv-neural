//! Dataset loading: real BrainVision EEG + synthetic SpanishBCBL-like data.
//!
//! A [`Recording`] pairs a continuous [`MultiChannelTimeSeries`] with the
//! [`EventTimeline`] of keystrokes — the two inputs every downstream stage
//! needs. Real MEG `.fif` files require an external `mne` export step
//! (documented in the integration report); BrainVision EEG is parsed natively
//! here via [`brainvision::read_vhdr`].

pub mod brainvision;
pub mod synthetic;

use serde::{Deserialize, Serialize};

use ruv_neural_core::error::Result;
use ruv_neural_core::sensor::SensorType;
use ruv_neural_core::signal::MultiChannelTimeSeries;

use crate::events::{EventTimeline, KeystrokeEvent, Sentence};

pub use brainvision::{read_vhdr, BrainVisionRecording, ChannelInfo, Marker};
pub use synthetic::{generate as generate_synthetic, SyntheticParams};

/// Recording modality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Modality {
    /// Magnetoencephalography (306-ch Megin in SpanishBCBL).
    Meg,
    /// Electroencephalography (64-ch BrainVision in SpanishBCBL).
    Eeg,
}

impl Modality {
    /// Map to the core sensor type.
    pub fn sensor_type(&self) -> SensorType {
        match self {
            Modality::Meg => SensorType::SquidMeg,
            Modality::Eeg => SensorType::Eeg,
        }
    }
}

/// A single recording: signal plus keystroke timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recording {
    /// Continuous multi-channel signal.
    pub series: MultiChannelTimeSeries,
    /// Behavioral keystroke/sentence timeline.
    pub timeline: EventTimeline,
    /// Modality of this recording.
    pub modality: Modality,
}

impl Recording {
    /// Load a BrainVision EEG recording, deriving the keystroke timeline from
    /// markers.
    ///
    /// `is_keystroke` decides which markers are keypresses and maps each to its
    /// typed character; `sentence_breaks` markers (matched by predicate) start a
    /// new sentence. This keeps the SpanishBCBL-specific marker convention out
    /// of the parser while still producing a usable [`EventTimeline`].
    pub fn from_brainvision(
        vhdr_path: impl AsRef<std::path::Path>,
        is_keystroke: impl Fn(&Marker) -> Option<char>,
        is_sentence_break: impl Fn(&Marker) -> bool,
    ) -> Result<Self> {
        let rec = read_vhdr(vhdr_path)?;
        let sr = rec.series.sample_rate_hz;

        let mut sentences: Vec<Sentence> = Vec::new();
        let mut current: Vec<KeystrokeEvent> = Vec::new();
        let mut sid = 0usize;

        let flush = |sid: &mut usize,
                     current: &mut Vec<KeystrokeEvent>,
                     sentences: &mut Vec<Sentence>| {
            if !current.is_empty() {
                let text: String = current.iter().map(|k| k.character).collect();
                sentences.push(Sentence {
                    id: *sid,
                    text,
                    keystrokes: std::mem::take(current),
                });
                *sid += 1;
            }
        };

        for m in &rec.markers {
            if is_sentence_break(m) {
                flush(&mut sid, &mut current, &mut sentences);
                continue;
            }
            if let Some(ch) = is_keystroke(m) {
                current.push(KeystrokeEvent {
                    onset_s: m.onset_s(sr),
                    character: ch,
                    sentence_id: sid,
                });
            }
        }
        flush(&mut sid, &mut current, &mut sentences);

        Ok(Recording {
            series: rec.series,
            timeline: EventTimeline { sentences },
            modality: Modality::Eeg,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modality_maps_to_sensor_type() {
        assert_eq!(Modality::Meg.sensor_type(), SensorType::SquidMeg);
        assert_eq!(Modality::Eeg.sensor_type(), SensorType::Eeg);
    }

    #[test]
    fn synthetic_roundtrips_through_recording() {
        let rec = generate_synthetic(&["hola"], &SyntheticParams::default(), 1);
        assert_eq!(rec.modality, Modality::Eeg);
        assert_eq!(rec.timeline.sentences.len(), 1);
        assert_eq!(rec.timeline.sentences[0].text, "hola");
    }
}
