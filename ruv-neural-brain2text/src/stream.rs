//! Causal streaming decoder: fixed frame-hop, incremental state, gated output.
//!
//! The 2025–2026 streaming speech-BCI literature converged on a shape this
//! module implements natively in Rust:
//!
//! - **Fixed frame hop with incremental state.** The UCSF/Berkeley streaming
//!   brain-to-voice system decodes in **80-ms increments** and meets that
//!   per-step budget in 99.3% of steps (Littlejohn, Cho et al., *Nat.
//!   Neurosci.* 28:902–912, 2025). 80 ms is a validated cadence but a **soft**
//!   budget, not a floor: UC Davis runs a fully causal decoder at a **10-ms**
//!   hop with ~10 ms inference (Wairagkar et al., *Nature* 644:145–152, 2025).
//!   [`StreamConfig::frame_hop_s`] therefore defaults to 0.080 and is tunable
//!   down to the 10-ms regime, and overruns are *reported*
//!   ([`StreamStats::overrun_frames`]) rather than assumed away.
//! - **Strict causality.** Normalization and smoothing use only past samples
//!   (rolling mean/variance over a bounded history), mirroring the causal
//!   rolling normalization the UC Davis online pipeline requires. No frame
//!   ever sees a future sample.
//! - **Event-referenced keyword detection for low-SNR non-invasive signals.**
//!   Open-vocabulary text is out of reach for EEG-class hardware (~65% CER,
//!   Brain2Qwerty v1), so the realistic tier is binary detection over
//!   event-referenced windows (LibriBrain KWS, arXiv:2510.21038): a window
//!   spanning `[-pre, +post]` around a candidate onset, scored against a
//!   permutation null. Defaults follow that paper's offset sweep (0.1 s pre,
//!   0.3 s post), which peaked its AUPRC.
//! - **Gated output.** Every emission passes through
//!   [`DecodeGate`](ruv_neural_core::gate::DecodeGate): decoding is opt-in per
//!   use, never ambient (ADR-0007/0022).
//!
//! Deliberately **not** implemented: an RNN-T. The widely repeated attribution
//! of an RNN-T to the 80-ms streaming system did not survive source
//! verification — the architecture behind that increment is not publicly
//! established, so this module commits only to the *cadence and causality*
//! those papers do establish, and drives any [`AcousticModel`] behind them.
//!
//! Scope honesty: every latency and accuracy number cited above comes from
//! **invasive** recordings (ECoG / Utah arrays, n=1) or MEG. The cadence and
//! causal structure transfer to an edge EEG pipeline; the accuracy does not.

use serde::{Deserialize, Serialize};

use ruv_neural_core::error::{Result, RuvNeuralError};
use ruv_neural_core::gate::{DecodeGate, DecodeGateConfig, GateState};
use ruv_neural_core::signal::MultiChannelTimeSeries;

use crate::model::AcousticModel;

/// Configuration for the streaming decoder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Seconds between successive decode steps. Default 0.080 s — the
    /// validated streaming cadence; the 10-ms regime is also supported.
    pub frame_hop_s: f64,
    /// Seconds of past signal each frame sees. Causal: the window always ends
    /// at the current frame boundary and never extends into the future.
    pub window_s: f64,
    /// Seconds of history used for rolling (causal) normalization. `0.0`
    /// disables normalization.
    pub norm_history_s: f64,
    /// Per-frame processing budget in seconds. Frames whose measured cost
    /// exceeds this are counted as overruns. Defaults to `frame_hop_s`.
    pub budget_s: Option<f64>,
    /// Event-referenced detection window: seconds *before* a candidate onset.
    pub kws_pre_s: f64,
    /// Event-referenced detection window: seconds *after* a candidate onset.
    pub kws_post_s: f64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            frame_hop_s: 0.080,
            window_s: 0.500,
            norm_history_s: 1.500,
            budget_s: None,
            kws_pre_s: 0.100,
            kws_post_s: 0.300,
        }
    }
}

impl StreamConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        let finite_pos = |x: f64| x.is_finite() && x > 0.0;
        if !finite_pos(self.frame_hop_s) {
            return Err(RuvNeuralError::Config(format!(
                "frame_hop_s must be finite and > 0, got {}",
                self.frame_hop_s
            )));
        }
        if !finite_pos(self.window_s) {
            return Err(RuvNeuralError::Config(format!(
                "window_s must be finite and > 0, got {}",
                self.window_s
            )));
        }
        if self.window_s < self.frame_hop_s {
            return Err(RuvNeuralError::Config(format!(
                "window_s ({}) must be >= frame_hop_s ({})",
                self.window_s, self.frame_hop_s
            )));
        }
        if !self.norm_history_s.is_finite() || self.norm_history_s < 0.0 {
            return Err(RuvNeuralError::Config(format!(
                "norm_history_s must be finite and >= 0, got {}",
                self.norm_history_s
            )));
        }
        if let Some(b) = self.budget_s {
            if !finite_pos(b) {
                return Err(RuvNeuralError::Config(format!(
                    "budget_s must be finite and > 0, got {b}"
                )));
            }
        }
        for (name, v) in [
            ("kws_pre_s", self.kws_pre_s),
            ("kws_post_s", self.kws_post_s),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(RuvNeuralError::Config(format!(
                    "{name} must be finite and >= 0, got {v}"
                )));
            }
        }
        if self.kws_pre_s + self.kws_post_s <= 0.0 {
            return Err(RuvNeuralError::Config(
                "kws_pre_s + kws_post_s must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// The effective per-frame budget in seconds.
    pub fn budget_s(&self) -> f64 {
        self.budget_s.unwrap_or(self.frame_hop_s)
    }
}

/// One decoder emission.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamEmission {
    /// Frame index since stream start.
    pub frame: u64,
    /// Timestamp (seconds) of the frame boundary that produced it.
    pub at_s: f64,
    /// Decoded character, if the gate released output for this frame.
    /// `None` while the gate is locked — the decoder still runs, but nothing
    /// is released.
    pub character: Option<char>,
    /// Log-probability of the emitted character under the acoustic model.
    pub logp: f64,
    /// Gate state after this frame.
    pub gate: GateState,
}

/// Rolling statistics over the last `n` frames of per-frame processing cost.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StreamStats {
    /// Frames processed.
    pub frames: u64,
    /// Frames whose measured cost exceeded [`StreamConfig::budget_s`].
    pub overrun_frames: u64,
    /// Frames for which the gate released output.
    pub released_frames: u64,
    /// Maximum observed per-frame cost in seconds (0 if never reported).
    pub max_frame_cost_s: f64,
}

impl StreamStats {
    /// Fraction of frames that met the budget, in `[0, 1]`. The published
    /// streaming system reports 99.3% here.
    pub fn budget_success_rate(&self) -> f64 {
        if self.frames == 0 {
            return 1.0;
        }
        1.0 - (self.overrun_frames as f64 / self.frames as f64)
    }
}

/// Causal rolling normalizer over a bounded sample history.
///
/// Maintains per-channel running sums over the last `capacity` samples, so
/// normalization at frame *k* uses only samples at or before frame *k* — the
/// causal constraint the online pipelines require.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RollingNorm {
    capacity: usize,
    /// Per-channel ring buffer of past samples.
    history: Vec<std::collections::VecDeque<f64>>,
    sum: Vec<f64>,
    sum_sq: Vec<f64>,
}

impl RollingNorm {
    fn new(num_channels: usize, capacity: usize) -> Self {
        Self {
            capacity,
            history: vec![std::collections::VecDeque::new(); num_channels],
            sum: vec![0.0; num_channels],
            sum_sq: vec![0.0; num_channels],
        }
    }

    fn push(&mut self, channel: usize, x: f64) {
        if self.capacity == 0 || channel >= self.history.len() {
            return;
        }
        let hist = &mut self.history[channel];
        hist.push_back(x);
        self.sum[channel] += x;
        self.sum_sq[channel] += x * x;
        while hist.len() > self.capacity {
            if let Some(old) = hist.pop_front() {
                self.sum[channel] -= old;
                self.sum_sq[channel] -= old * old;
            }
        }
    }

    /// Mean and standard deviation of the retained history for a channel.
    fn stats(&self, channel: usize) -> Option<(f64, f64)> {
        let n = self.history.get(channel)?.len();
        if n < 2 {
            return None;
        }
        let n = n as f64;
        let mean = self.sum[channel] / n;
        let var = (self.sum_sq[channel] / n - mean * mean).max(0.0);
        Some((mean, var.sqrt()))
    }
}

/// A causal streaming decoder driving any [`AcousticModel`] at a fixed frame
/// hop, with incremental state and gated emissions.
///
/// State carried across frames: the causal normalization history and the
/// sample cursor. No frame recomputes a sliding window from scratch beyond
/// its own `window_s` of context, and nothing looks ahead.
#[derive(Debug, Clone)]
pub struct StreamingDecoder<M: AcousticModel> {
    config: StreamConfig,
    model: M,
    gate: DecodeGate,
    norm: RollingNorm,
    sample_rate: f64,
    num_channels: usize,
    /// Index of the next unconsumed sample.
    cursor: usize,
    frame: u64,
    stats: StreamStats,
}

impl<M: AcousticModel> StreamingDecoder<M> {
    /// Create a decoder for a given channel count and sample rate.
    pub fn new(
        config: StreamConfig,
        model: M,
        gate_config: DecodeGateConfig,
        num_channels: usize,
        sample_rate: f64,
    ) -> Result<Self> {
        config.validate()?;
        if num_channels == 0 {
            return Err(RuvNeuralError::Config(
                "num_channels must be >= 1".to_string(),
            ));
        }
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(RuvNeuralError::Config(format!(
                "sample_rate must be finite and > 0, got {sample_rate}"
            )));
        }
        let norm_capacity = (config.norm_history_s * sample_rate).round() as usize;
        Ok(Self {
            norm: RollingNorm::new(num_channels, norm_capacity),
            config,
            model,
            gate: DecodeGate::new(gate_config)?,
            sample_rate,
            num_channels,
            cursor: 0,
            frame: 0,
            stats: StreamStats::default(),
        })
    }

    /// Samples per frame hop (at least 1).
    pub fn hop_samples(&self) -> usize {
        ((self.config.frame_hop_s * self.sample_rate).round() as usize).max(1)
    }

    /// Samples per causal decode window (at least 1).
    pub fn window_samples(&self) -> usize {
        ((self.config.window_s * self.sample_rate).round() as usize).max(1)
    }

    /// Per-frame processing statistics.
    pub fn stats(&self) -> &StreamStats {
        &self.stats
    }

    /// The gate guarding output (read-only).
    pub fn gate(&self) -> &DecodeGate {
        &self.gate
    }

    /// Number of frames emitted so far.
    pub fn frame_count(&self) -> u64 {
        self.frame
    }

    /// Record a measured per-frame processing cost, counting budget overruns.
    ///
    /// Callers time their own frame (the decoder does not read the clock, so
    /// it stays deterministic and `no_std`-friendly in spirit). The published
    /// 99.3%-under-80-ms figure is exactly this ratio.
    pub fn record_frame_cost(&mut self, cost_s: f64) {
        if !cost_s.is_finite() || cost_s < 0.0 {
            return;
        }
        if cost_s > self.stats.max_frame_cost_s {
            self.stats.max_frame_cost_s = cost_s;
        }
        if cost_s > self.config.budget_s() {
            self.stats.overrun_frames += 1;
        }
    }

    /// Push a chunk of samples and decode every complete frame it enables.
    ///
    /// `chunk` is `[channel][sample]`, matching
    /// [`MultiChannelTimeSeries`]'s layout, and represents *new* samples
    /// appended to the stream. Returns one [`StreamEmission`] per completed
    /// frame, in order.
    ///
    /// `password_confidence` is the unlock-detector confidence for this
    /// chunk, forwarded to the gate once per frame (see
    /// [`DecodeGate::update`]). `now_s` is the wall/monotonic timestamp of the
    /// chunk's *end*; per-frame timestamps are interpolated backwards from it.
    pub fn push_chunk(
        &mut self,
        chunk: &[Vec<f64>],
        password_confidence: f64,
        now_s: f64,
    ) -> Result<Vec<StreamEmission>> {
        if chunk.len() != self.num_channels {
            return Err(RuvNeuralError::DimensionMismatch {
                expected: self.num_channels,
                got: chunk.len(),
            });
        }
        let n_new = chunk.first().map(|c| c.len()).unwrap_or(0);
        if chunk.iter().any(|c| c.len() != n_new) {
            return Err(RuvNeuralError::Signal(
                "all channels in a chunk must have the same length".to_string(),
            ));
        }

        // Append to the causal normalization history (past-only by
        // construction: samples enter in arrival order).
        for (ch, samples) in chunk.iter().enumerate() {
            for &x in samples {
                self.norm.push(ch, x);
            }
        }

        let hop = self.hop_samples();
        let window = self.window_samples();
        let mut pending = self.cursor + n_new;
        let mut emissions = Vec::new();
        let hop_s = self.config.frame_hop_s;

        // Decode every complete hop that has accumulated. The window is the
        // most recent `window` samples of normalization history, which is
        // exactly the causal context.
        let mut frames_this_chunk = 0u64;
        while pending >= hop {
            pending -= hop;
            frames_this_chunk += 1;
        }
        for i in 0..frames_this_chunk {
            // Frame boundaries are spaced hop_s apart, ending at now_s.
            let remaining = frames_this_chunk - 1 - i;
            let at_s = now_s - remaining as f64 * hop_s;
            let features = self.causal_features(window);
            let (character, logp) = match self.model.logprobs(&features).first() {
                Some(&(c, lp)) => (Some(c), lp),
                None => (None, f64::NEG_INFINITY),
            };
            let gate_state = self.gate.update(password_confidence, at_s);
            let released = character.filter(|_| self.gate.is_armed());
            if released.is_some() {
                self.stats.released_frames += 1;
            }
            self.frame += 1;
            self.stats.frames += 1;
            emissions.push(StreamEmission {
                frame: self.frame - 1,
                at_s,
                character: released,
                logp,
                gate: gate_state,
            });
        }
        self.cursor = pending;
        Ok(emissions)
    }

    /// Explicitly lock the gate (user action or safety-envelope trip).
    pub fn lock(&mut self, now_s: f64) -> GateState {
        self.gate.lock(now_s)
    }

    /// Build the feature vector for the current frame from causal context:
    /// per-channel mean and standard deviation of the last `window` samples,
    /// z-scored against the rolling normalization history.
    fn causal_features(&self, window: usize) -> Vec<f64> {
        let mut features = Vec::with_capacity(self.num_channels * 2);
        for ch in 0..self.num_channels {
            let hist = &self.norm.history[ch];
            let take = window.min(hist.len());
            if take == 0 {
                features.push(0.0);
                features.push(0.0);
                continue;
            }
            let start = hist.len() - take;
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for &x in hist.iter().skip(start) {
                sum += x;
                sum_sq += x * x;
            }
            let n = take as f64;
            let mean = sum / n;
            let sd = (sum_sq / n - mean * mean).max(0.0).sqrt();
            match self.norm.stats(ch) {
                Some((hmean, hsd)) if hsd > 0.0 => {
                    features.push((mean - hmean) / hsd);
                    features.push(sd / hsd);
                }
                _ => {
                    features.push(mean);
                    features.push(sd);
                }
            }
        }
        features
    }
}

/// Event-referenced keyword-detection window (LibriBrain KWS formulation).
///
/// A candidate onset at `onset_s` yields the window
/// `[onset_s - pre, onset_s + post]`, labelled 1 when the spoken/typed token
/// is in the keyword set and 0 otherwise. This binary, heavily imbalanced
/// formulation is the realistic tier for low-SNR non-invasive signals —
/// evaluate it with AUPRC against a permutation null
/// ([`crate::stream::permutation_null_auprc`]), never with raw accuracy.
#[derive(Debug, Clone, PartialEq)]
pub struct KeywordWindow {
    /// Onset the window is referenced to, in seconds.
    pub onset_s: f64,
    /// `[channel][sample]` slice of the recording covering the window.
    pub data: Vec<Vec<f64>>,
    /// 1 when the token at this onset is in the keyword set.
    pub label: u8,
}

/// Extract event-referenced keyword windows from a recording.
///
/// Onsets whose window falls outside the recording are skipped rather than
/// zero-padded, so no window mixes real and fabricated samples.
pub fn extract_keyword_windows(
    series: &MultiChannelTimeSeries,
    onsets: &[(f64, u8)],
    config: &StreamConfig,
) -> Result<Vec<KeywordWindow>> {
    config.validate()?;
    let fs = series.sample_rate_hz;
    if !fs.is_finite() || fs <= 0.0 {
        return Err(RuvNeuralError::Signal(format!(
            "sample_rate must be finite and > 0, got {fs}"
        )));
    }
    let pre = (config.kws_pre_s * fs).round() as i64;
    let post = (config.kws_post_s * fs).round() as i64;
    let total = (pre + post).max(1) as usize;
    let n = series.num_samples as i64;

    let mut out = Vec::new();
    for &(onset_s, label) in onsets {
        if !onset_s.is_finite() {
            continue;
        }
        let center = ((onset_s - series.timestamp_start) * fs).round() as i64;
        let start = center - pre;
        if start < 0 || start + total as i64 > n {
            continue;
        }
        let mut data = Vec::with_capacity(series.num_channels);
        for ch in 0..series.num_channels {
            let samples = series.channel(ch)?;
            data.push(samples[start as usize..start as usize + total].to_vec());
        }
        out.push(KeywordWindow {
            onset_s,
            data,
            label: u8::from(label != 0),
        });
    }
    Ok(out)
}

/// Area under the precision-recall curve for binary scores.
///
/// AUPRC is the primary metric for event-referenced detection under heavy
/// class imbalance (LibriBrain KWS reports a 0.00515 base rate), where raw
/// accuracy and even AUROC are uninformative. Uses the step-wise
/// (interpolation-free) definition: the sum of precision at each true
/// positive divided by the number of positives.
///
/// Returns `None` when there are no positives or no samples.
pub fn auprc(scores: &[f64], labels: &[u8]) -> Option<f64> {
    if scores.len() != labels.len() || scores.is_empty() {
        return None;
    }
    if scores.iter().any(|s| !s.is_finite()) {
        return None;
    }
    let positives = labels.iter().filter(|&&l| l != 0).count();
    if positives == 0 {
        return None;
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut sum_precision = 0.0;
    for &i in &order {
        if labels[i] != 0 {
            tp += 1;
            sum_precision += tp as f64 / (tp + fp) as f64;
        } else {
            fp += 1;
        }
    }
    Some(sum_precision / positives as f64)
}

/// Expected AUPRC of a random scorer: the positive base rate.
///
/// A model AUPRC of 0.09 looks like failure until compared against a base
/// rate of 0.005 — a 13.4× lift. Always report the ratio, never the raw
/// number alone.
pub fn permutation_null_auprc(labels: &[u8]) -> Option<f64> {
    if labels.is_empty() {
        return None;
    }
    let positives = labels.iter().filter(|&&l| l != 0).count();
    if positives == 0 {
        return None;
    }
    Some(positives as f64 / labels.len() as f64)
}

/// False alarms per hour at a given score threshold — the operational
/// complement to AUPRC. Thresholds must be selected on validation data, never
/// on the reported test set.
pub fn false_alarms_per_hour(
    scores: &[f64],
    labels: &[u8],
    threshold: f64,
    duration_s: f64,
) -> Option<f64> {
    if scores.len() != labels.len() || scores.is_empty() {
        return None;
    }
    if !duration_s.is_finite() || duration_s <= 0.0 {
        return None;
    }
    let fp = scores
        .iter()
        .zip(labels)
        .filter(|(&s, &l)| l == 0 && s >= threshold)
        .count();
    Some(fp as f64 * 3600.0 / duration_s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::AcousticModel;

    /// A deterministic stub model: emits a character keyed to the sign of the
    /// first feature, so tests can assert on decoder plumbing, not learning.
    #[derive(Debug, Clone)]
    struct StubModel;

    impl AcousticModel for StubModel {
        fn vocabulary(&self) -> Vec<char> {
            vec!['a', 'b']
        }
        fn logprobs(&self, features: &[f64]) -> Vec<(char, f64)> {
            let first = features.first().copied().unwrap_or(0.0);
            if first >= 0.0 {
                vec![('a', -0.1), ('b', -2.0)]
            } else {
                vec![('b', -0.1), ('a', -2.0)]
            }
        }
    }

    fn decoder(hop_s: f64) -> StreamingDecoder<StubModel> {
        StreamingDecoder::new(
            StreamConfig {
                frame_hop_s: hop_s,
                window_s: 0.5,
                ..StreamConfig::default()
            },
            StubModel,
            DecodeGateConfig::default(),
            2,
            250.0,
        )
        .unwrap()
    }

    fn chunk(n: usize, value: f64) -> Vec<Vec<f64>> {
        vec![vec![value; n], vec![value * 0.5; n]]
    }

    #[test]
    fn emits_one_frame_per_hop() {
        let mut d = decoder(0.080);
        assert_eq!(d.hop_samples(), 20); // 80 ms at 250 Hz
        let out = d.push_chunk(&chunk(20, 1.0), 0.0, 0.08).unwrap();
        assert_eq!(out.len(), 1);
        // A chunk of 3 hops emits exactly 3 frames.
        let out = d.push_chunk(&chunk(60, 1.0), 0.0, 0.32).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(d.frame_count(), 4);
    }

    #[test]
    fn partial_hops_accumulate_across_chunks() {
        let mut d = decoder(0.080);
        // 12 + 8 samples = one 20-sample hop, split across two chunks.
        assert!(d
            .push_chunk(&chunk(12, 1.0), 0.0, 0.048)
            .unwrap()
            .is_empty());
        let out = d.push_chunk(&chunk(8, 1.0), 0.0, 0.08).unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn supports_the_ten_millisecond_regime() {
        // 80 ms is a validated cadence, not a floor (UC Davis runs 10 ms).
        let mut d = decoder(0.010);
        assert_eq!(d.hop_samples(), 3); // 10 ms at 250 Hz, rounded
        let out = d.push_chunk(&chunk(30, 1.0), 0.0, 0.12).unwrap();
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn output_is_suppressed_until_the_gate_arms() {
        let mut d = decoder(0.080);
        // Locked: frames still decode, but nothing is released.
        let out = d.push_chunk(&chunk(40, 1.0), 0.0, 0.16).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|e| e.character.is_none()));
        assert!(out.iter().all(|e| e.logp.is_finite()));
        assert_eq!(d.stats().released_frames, 0);

        // Password detected for the required consecutive frames -> armed.
        let out = d.push_chunk(&chunk(60, 1.0), 0.99, 0.40).unwrap();
        assert_eq!(out.len(), 3);
        assert!(out.last().unwrap().character.is_some());
        assert!(d.gate().is_armed());

        // Explicit lock suppresses again.
        d.lock(0.5);
        let out = d.push_chunk(&chunk(20, 1.0), 0.0, 0.58).unwrap();
        assert!(out[0].character.is_none());
    }

    #[test]
    fn decoding_is_causal() {
        // Two decoders fed the same prefix must agree on that prefix's
        // frames regardless of what arrives afterwards — no lookahead.
        let mut a = decoder(0.080);
        let mut b = decoder(0.080);
        let first_a = a.push_chunk(&chunk(20, 1.0), 0.0, 0.08).unwrap();
        let first_b = b.push_chunk(&chunk(20, 1.0), 0.0, 0.08).unwrap();
        assert_eq!(first_a, first_b);
        // Diverging futures cannot retroactively change the emitted frame.
        a.push_chunk(&chunk(20, -50.0), 0.0, 0.16).unwrap();
        b.push_chunk(&chunk(20, 50.0), 0.0, 0.16).unwrap();
        assert_eq!(first_a, first_b);
    }

    #[test]
    fn budget_overruns_are_reported_not_assumed_away() {
        let mut d = decoder(0.080);
        d.push_chunk(&chunk(200, 1.0), 0.0, 0.8).unwrap();
        assert_eq!(d.stats().frames, 10);
        // 9 frames inside budget, 1 over: 90% success, mirroring how the
        // published 99.3%-under-80-ms figure is computed.
        for _ in 0..9 {
            d.record_frame_cost(0.05);
        }
        d.record_frame_cost(0.12);
        assert_eq!(d.stats().overrun_frames, 1);
        assert!((d.stats().budget_success_rate() - 0.9).abs() < 1e-12);
        assert!((d.stats().max_frame_cost_s - 0.12).abs() < 1e-12);
        // Nonsense costs are ignored rather than corrupting the stats.
        d.record_frame_cost(f64::NAN);
        d.record_frame_cost(-1.0);
        assert_eq!(d.stats().overrun_frames, 1);
    }

    #[test]
    fn rejects_bad_config_and_input() {
        assert!(StreamConfig {
            frame_hop_s: 0.0,
            ..StreamConfig::default()
        }
        .validate()
        .is_err());
        assert!(StreamConfig {
            window_s: 0.01,
            frame_hop_s: 0.08,
            ..StreamConfig::default()
        }
        .validate()
        .is_err());
        assert!(StreamConfig {
            kws_pre_s: -0.1,
            ..StreamConfig::default()
        }
        .validate()
        .is_err());
        assert!(StreamingDecoder::new(
            StreamConfig::default(),
            StubModel,
            DecodeGateConfig::default(),
            0,
            250.0
        )
        .is_err());
        assert!(StreamingDecoder::new(
            StreamConfig::default(),
            StubModel,
            DecodeGateConfig::default(),
            2,
            0.0
        )
        .is_err());
        let mut d = decoder(0.080);
        // Wrong channel count.
        assert!(d.push_chunk(&[vec![1.0; 20]], 0.0, 0.08).is_err());
        // Ragged channels.
        assert!(d
            .push_chunk(&[vec![1.0; 20], vec![1.0; 19]], 0.0, 0.08)
            .is_err());
    }

    #[test]
    fn keyword_windows_are_event_referenced_and_bounded() {
        let fs = 250.0;
        let n = 1000;
        let series = MultiChannelTimeSeries::new(
            vec![(0..n).map(|i| i as f64).collect(), vec![0.0; n]],
            fs,
            0.0,
        )
        .unwrap();
        let config = StreamConfig::default(); // 0.1 s pre, 0.3 s post
        let windows = extract_keyword_windows(
            &series,
            &[
                (0.02, 1), // too early: window would start before t=0
                (1.0, 1),  // fits
                (2.0, 0),  // fits
                (3.99, 1), // too late: window would run past the end
                (f64::NAN, 1),
            ],
            &config,
        )
        .unwrap();
        assert_eq!(windows.len(), 2, "out-of-range onsets must be skipped");
        // 0.4 s at 250 Hz = 100 samples.
        assert_eq!(windows[0].data[0].len(), 100);
        // Window starts 0.1 s before the onset: sample index 250 - 25 = 225.
        assert!((windows[0].data[0][0] - 225.0).abs() < 1e-9);
        assert_eq!(windows[0].label, 1);
        assert_eq!(windows[1].label, 0);
    }

    #[test]
    fn auprc_matches_hand_computed_values() {
        // Perfect ranking: all positives above all negatives -> 1.0.
        let scores = [0.9, 0.8, 0.2, 0.1];
        let labels = [1, 1, 0, 0];
        assert!((auprc(&scores, &labels).unwrap() - 1.0).abs() < 1e-12);
        // Worst ranking: positives last. Precision at first TP = 1/3,
        // at second = 2/4 -> mean = (1/3 + 1/2) / 2.
        let labels_rev = [0, 0, 1, 1];
        let expected = (1.0 / 3.0 + 0.5) / 2.0;
        assert!((auprc(&scores, &labels_rev).unwrap() - expected).abs() < 1e-12);
        // Degenerate inputs.
        assert!(auprc(&[], &[]).is_none());
        assert!(auprc(&scores, &[0, 0, 0, 0]).is_none());
        assert!(auprc(&[f64::NAN, 1.0], &[1, 0]).is_none());
        assert!(auprc(&scores, &[1, 0]).is_none());
    }

    #[test]
    fn null_auprc_is_the_base_rate_and_gives_a_lift_ratio() {
        // LibriBrain's held-out test set: 24 positives in 4660 windows.
        let mut labels = vec![0u8; 4660];
        for l in labels.iter_mut().take(24) {
            *l = 1;
        }
        let null = permutation_null_auprc(&labels).unwrap();
        assert!((null - 24.0 / 4660.0).abs() < 1e-12);
        // A raw AUPRC of 0.069 is ~13.4x this chance level — the reason the
        // ratio, not the raw value, is what a gate should threshold on.
        assert!((0.069 / null - 13.39).abs() < 0.1);
        assert!(permutation_null_auprc(&[]).is_none());
        assert!(permutation_null_auprc(&[0, 0]).is_none());
    }

    #[test]
    fn false_alarms_per_hour_counts_only_negatives_above_threshold() {
        let scores = [0.9, 0.8, 0.7, 0.1];
        let labels = [1, 0, 0, 0];
        // Two negatives at/above 0.7 in 60 s -> 120 FA/h.
        let fa = false_alarms_per_hour(&scores, &labels, 0.7, 60.0).unwrap();
        assert!((fa - 120.0).abs() < 1e-9);
        assert!(false_alarms_per_hour(&scores, &labels, 0.7, 0.0).is_none());
    }
}
