//! Decoding accuracy metrics: Character Error Rate (CER) and Word Error Rate
//! (WER), both based on Levenshtein edit distance — the same metrics reported
//! by Brain2Qwerty.

use serde::{Deserialize, Serialize};

/// Levenshtein edit distance between two token slices.
///
/// Generic over the token type so it serves both characters (CER) and words
/// (WER). Uses the standard two-row dynamic-programming formulation, O(n·m)
/// time and O(min(n,m)) space.
pub fn levenshtein<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    // Ensure `b` is the shorter to minimize the row buffer.
    let (a, b) = if a.len() >= b.len() { (a, b) } else { (b, a) };

    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];

    for (i, ai) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, bj) in b.iter().enumerate() {
            let cost = if ai == bj { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1) // deletion
                .min(curr[j] + 1) // insertion
                .min(prev[j] + cost); // substitution
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// Character Error Rate = edit_distance(pred, target) / len(target).
///
/// An empty target returns 0.0 when the prediction is also empty, else 1.0.
pub fn character_error_rate(prediction: &str, target: &str) -> f64 {
    let p: Vec<char> = prediction.chars().collect();
    let t: Vec<char> = target.chars().collect();
    if t.is_empty() {
        return if p.is_empty() { 0.0 } else { 1.0 };
    }
    levenshtein(&p, &t) as f64 / t.len() as f64
}

/// Word Error Rate = word_edit_distance(pred, target) / word_count(target).
///
/// Words are whitespace-delimited.
pub fn word_error_rate(prediction: &str, target: &str) -> f64 {
    let p: Vec<&str> = prediction.split_whitespace().collect();
    let t: Vec<&str> = target.split_whitespace().collect();
    if t.is_empty() {
        return if p.is_empty() { 0.0 } else { 1.0 };
    }
    levenshtein(&p, &t) as f64 / t.len() as f64
}

/// Aggregated decoding report over a set of decoded sentences.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecodeReport {
    /// Number of sentences scored.
    pub num_sentences: usize,
    /// Macro-averaged CER across sentences.
    pub mean_cer: f64,
    /// Macro-averaged WER across sentences.
    pub mean_wer: f64,
    /// Per-sentence (prediction, target, cer) triples.
    pub per_sentence: Vec<(String, String, f64)>,
}

impl DecodeReport {
    /// Build a report from `(prediction, target)` pairs.
    pub fn from_pairs<'a>(pairs: impl IntoIterator<Item = (&'a str, &'a str)>) -> Self {
        let mut per_sentence = Vec::new();
        let mut cer_sum = 0.0;
        let mut wer_sum = 0.0;
        for (pred, target) in pairs {
            let cer = character_error_rate(pred, target);
            let wer = word_error_rate(pred, target);
            cer_sum += cer;
            wer_sum += wer;
            per_sentence.push((pred.to_string(), target.to_string(), cer));
        }
        let n = per_sentence.len();
        DecodeReport {
            num_sentences: n,
            mean_cer: if n > 0 { cer_sum / n as f64 } else { 0.0 },
            mean_wer: if n > 0 { wer_sum / n as f64 } else { 0.0 },
            per_sentence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn levenshtein_basic() {
        assert_eq!(levenshtein(b"kitten", b"sitting"), 3);
        assert_eq!(levenshtein(b"", b"abc"), 3);
        assert_eq!(levenshtein(b"abc", b"abc"), 0);
        assert_eq!(levenshtein(b"flaw", b"lawn"), 2);
    }

    #[test]
    fn cer_perfect_and_empty() {
        assert_relative_eq!(character_error_rate("hola", "hola"), 0.0);
        assert_relative_eq!(character_error_rate("", ""), 0.0);
        assert_relative_eq!(character_error_rate("x", ""), 1.0);
        // one substitution out of four chars.
        assert_relative_eq!(character_error_rate("hila", "hola"), 0.25);
    }

    #[test]
    fn wer_counts_words() {
        assert_relative_eq!(word_error_rate("the cat sat", "the cat sat"), 0.0);
        assert_relative_eq!(word_error_rate("the dog sat", "the cat sat"), 1.0 / 3.0);
    }

    #[test]
    fn report_aggregates() {
        let r = DecodeReport::from_pairs([("hola", "hola"), ("hila", "hola")]);
        assert_eq!(r.num_sentences, 2);
        assert_relative_eq!(r.mean_cer, 0.125);
    }
}
