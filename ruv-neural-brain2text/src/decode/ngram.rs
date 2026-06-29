//! Character n-gram language model with add-k smoothing and backoff.
//!
//! Plays the role of Brain2Qwerty V1's 9-gram KenLM module: it scores candidate
//! character sequences during beam search so the acoustic model's per-keystroke
//! guesses are nudged toward plausible text. Trained on the target sentences'
//! characters (in the real system: a large Spanish corpus).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A character n-gram model of order `n` (uses contexts up to `n-1` chars).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharNgram {
    order: usize,
    /// counts[context_string][next_char] = count
    counts: HashMap<String, HashMap<char, u32>>,
    /// Total count per context (for normalization).
    context_totals: HashMap<String, u32>,
    /// Vocabulary size (for add-k smoothing denominator).
    vocab_size: usize,
    /// Smoothing constant.
    k: f64,
}

impl CharNgram {
    /// Train an order-`n` model on a set of texts.
    pub fn train<'a>(texts: impl IntoIterator<Item = &'a str>, order: usize) -> Self {
        let order = order.max(1);
        let mut counts: HashMap<String, HashMap<char, u32>> = HashMap::new();
        let mut context_totals: HashMap<String, u32> = HashMap::new();
        let mut vocab = std::collections::BTreeSet::new();

        for text in texts {
            let chars: Vec<char> = text.chars().collect();
            for c in &chars {
                vocab.insert(*c);
            }
            for i in 0..chars.len() {
                // For each backoff length 0..order-1, count the (context, next).
                let next = chars[i];
                let max_ctx = (order - 1).min(i);
                for clen in 0..=max_ctx {
                    let ctx: String = chars[i - clen..i].iter().collect();
                    *counts
                        .entry(ctx.clone())
                        .or_default()
                        .entry(next)
                        .or_insert(0) += 1;
                    *context_totals.entry(ctx).or_insert(0) += 1;
                }
            }
        }

        CharNgram {
            order,
            counts,
            context_totals,
            vocab_size: vocab.len().max(1),
            k: 0.1,
        }
    }

    /// The model order.
    pub fn order(&self) -> usize {
        self.order
    }

    /// log P(next | context) with stupid-backoff and add-k smoothing.
    ///
    /// Tries the longest available context (up to `order-1`), backing off to
    /// shorter contexts when the longer one was never seen.
    pub fn logprob(&self, context: &str, next: char) -> f64 {
        let ctx_chars: Vec<char> = context.chars().collect();
        let max_ctx = (self.order - 1).min(ctx_chars.len());
        for clen in (0..=max_ctx).rev() {
            let ctx: String = ctx_chars[ctx_chars.len() - clen..].iter().collect();
            if let Some(total) = self.context_totals.get(&ctx) {
                let c = self
                    .counts
                    .get(&ctx)
                    .and_then(|m| m.get(&next))
                    .copied()
                    .unwrap_or(0) as f64;
                let p = (c + self.k) / (*total as f64 + self.k * self.vocab_size as f64);
                return p.ln();
            }
        }
        // Fully unseen: uniform over vocabulary.
        (1.0 / self.vocab_size as f64).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_seen_transitions() {
        let lm = CharNgram::train(["abab", "abab", "abab"], 2);
        // After 'a', 'b' is far more likely than 'a'.
        let p_ab = lm.logprob("a", 'b');
        let p_aa = lm.logprob("a", 'a');
        assert!(p_ab > p_aa, "p(b|a)={p_ab} p(a|a)={p_aa}");
    }

    #[test]
    fn backoff_for_unseen_context() {
        let lm = CharNgram::train(["hello world"], 5);
        // Unseen long context backs off; should still return a finite logprob.
        let lp = lm.logprob("zzzz", 'h');
        assert!(lp.is_finite());
    }

    #[test]
    fn serde_round_trip() {
        let lm = CharNgram::train(["hola mundo", "buenos dias"], 4);
        let json = serde_json::to_string(&lm).unwrap();
        let back: CharNgram = serde_json::from_str(&json).unwrap();
        assert_eq!(back.order(), lm.order());
        // Probabilities preserved.
        assert!((back.logprob("hol", 'a') - lm.logprob("hol", 'a')).abs() < 1e-12);
    }

    #[test]
    fn order_one_is_unigram() {
        let lm = CharNgram::train(["aaab"], 1);
        let p_a = lm.logprob("", 'a');
        let p_b = lm.logprob("", 'b');
        assert!(p_a > p_b);
    }
}
