//! Behavioral event model: keystrokes and sentences.
//!
//! Brain2Qwerty trials follow a read → wait → type structure. The typing phase
//! produces a timeline of keystrokes, each with an onset time (seconds, relative
//! to the recording start) and the typed character. Sentences group the
//! keystrokes that belong to one typed sentence — the unit over which Character
//! Error Rate (CER) and Word Error Rate (WER) are computed.

use serde::{Deserialize, Serialize};

/// A single keypress event during the typing phase.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeystrokeEvent {
    /// Onset time of the keypress in seconds from recording start.
    pub onset_s: f64,
    /// The character that was typed.
    pub character: char,
    /// Index of the sentence this keystroke belongs to.
    pub sentence_id: usize,
}

/// A typed sentence: its target text plus the keystrokes that produced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sentence {
    /// Unique sentence index within the session.
    pub id: usize,
    /// Ground-truth target text.
    pub text: String,
    /// Keystrokes belonging to this sentence, in typing order.
    pub keystrokes: Vec<KeystrokeEvent>,
}

impl Sentence {
    /// The target string reconstructed from the keystroke characters.
    pub fn typed_text(&self) -> String {
        self.keystrokes.iter().map(|k| k.character).collect()
    }
}

/// A full session timeline: all sentences typed in one recording.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventTimeline {
    /// Sentences in presentation order.
    pub sentences: Vec<Sentence>,
}

impl EventTimeline {
    /// Total number of keystrokes across all sentences.
    pub fn num_keystrokes(&self) -> usize {
        self.sentences.iter().map(|s| s.keystrokes.len()).sum()
    }

    /// Every keystroke across all sentences, flattened in order.
    pub fn all_keystrokes(&self) -> impl Iterator<Item = &KeystrokeEvent> {
        self.sentences.iter().flat_map(|s| s.keystrokes.iter())
    }

    /// The set of distinct characters appearing in the timeline (sorted).
    pub fn vocabulary(&self) -> Vec<char> {
        let mut v: Vec<char> = self
            .all_keystrokes()
            .map(|k| k.character)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        v.sort_unstable();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ks(t: f64, c: char, sid: usize) -> KeystrokeEvent {
        KeystrokeEvent {
            onset_s: t,
            character: c,
            sentence_id: sid,
        }
    }

    #[test]
    fn typed_text_reconstructs_string() {
        let s = Sentence {
            id: 0,
            text: "hola".into(),
            keystrokes: vec![
                ks(0.0, 'h', 0),
                ks(0.2, 'o', 0),
                ks(0.4, 'l', 0),
                ks(0.6, 'a', 0),
            ],
        };
        assert_eq!(s.typed_text(), "hola");
    }

    #[test]
    fn timeline_aggregates() {
        let tl = EventTimeline {
            sentences: vec![
                Sentence {
                    id: 0,
                    text: "ab".into(),
                    keystrokes: vec![ks(0.0, 'a', 0), ks(0.1, 'b', 0)],
                },
                Sentence {
                    id: 1,
                    text: "ba".into(),
                    keystrokes: vec![ks(1.0, 'b', 1), ks(1.1, 'a', 1)],
                },
            ],
        };
        assert_eq!(tl.num_keystrokes(), 4);
        assert_eq!(tl.vocabulary(), vec!['a', 'b']);
    }
}
