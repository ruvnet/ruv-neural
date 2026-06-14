//! A deterministic physiological simulator for testing and demos.
//!
//! Produces [`PhysioWindow`]s consistent with a target *arousal* level in
//! `[0, 1]`, so the closed-loop controller can be exercised end-to-end without
//! hardware. It is seeded and fully reproducible (no `rand` dependency).

use crate::physio::PhysioWindow;

/// Deterministic physiological signal generator.
#[derive(Debug, Clone)]
pub struct PhysioSimulator {
    rng: u64,
    /// Respiration sample rate (Hz).
    pub respiration_fs: f64,
    /// Accelerometer sample rate (Hz).
    pub accel_fs: f64,
    /// Phase accumulator for respiration continuity across windows (s).
    phase_s: f64,
}

impl PhysioSimulator {
    /// Create a simulator with the given RNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1),
            respiration_fs: 25.0,
            accel_fs: 50.0,
            phase_s: 0.0,
        }
    }

    /// Next uniform pseudo-random value in `[0, 1)` (xorshift64*).
    fn next_unit(&mut self) -> f64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        ((x.wrapping_mul(0x2545_F491_4F6C_DD1D)) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate one `window_s`-second window for the given target arousal in
    /// `[0, 1]`. Higher arousal ⇒ faster HR, lower HRV, faster/irregular
    /// breathing, and more motion.
    pub fn window(&mut self, timestamp_s: f64, window_s: f64, arousal: f64) -> PhysioWindow {
        let arousal = arousal.clamp(0.0, 1.0);

        // ── Heart rate / RR intervals ───────────────────────────────────
        // HR ranges ~55 bpm (calm) → ~100 bpm (aroused).
        let base_hr = 55.0 + 45.0 * arousal;
        let base_rr_ms = 60_000.0 / base_hr;
        // HRV (RR jitter) shrinks with arousal: ~60 ms calm → ~10 ms aroused.
        let rr_jitter = 60.0 * (1.0 - arousal) + 10.0;

        let mut rr_ms = Vec::new();
        let mut elapsed = 0.0;
        while elapsed < window_s * 1000.0 {
            let jitter = (self.next_unit() - 0.5) * 2.0 * rr_jitter;
            let rr = (base_rr_ms + jitter).max(300.0);
            rr_ms.push(rr);
            elapsed += rr;
        }

        // ── Respiration ────────────────────────────────────────────────
        // Rate ~6 bpm (calm/paced) → ~22 bpm (aroused).
        let resp_rate_bpm = 6.0 + 16.0 * arousal;
        let resp_hz = resp_rate_bpm / 60.0;
        let n_resp = (window_s * self.respiration_fs) as usize;
        let mut respiration = Vec::with_capacity(n_resp);
        for k in 0..n_resp {
            let t = self.phase_s + k as f64 / self.respiration_fs;
            // Irregularity noise grows with arousal.
            let noise = (self.next_unit() - 0.5) * 0.3 * arousal;
            respiration.push((2.0 * std::f64::consts::PI * resp_hz * t).sin() + noise);
        }
        self.phase_s += window_s;

        // ── Motion ─────────────────────────────────────────────────────
        // Movement index ~0.005 g (still) → ~0.12 g (restless).
        let move_sd = 0.005 + 0.115 * arousal;
        let n_acc = (window_s * self.accel_fs) as usize;
        let mut accel_magnitude_g = Vec::with_capacity(n_acc);
        for _ in 0..n_acc {
            let n = (self.next_unit() - 0.5) * 2.0 * move_sd;
            accel_magnitude_g.push(1.0 + n);
        }

        PhysioWindow {
            timestamp_s,
            rr_ms,
            respiration,
            respiration_fs: self.respiration_fs,
            accel_magnitude_g,
        }
    }
}
