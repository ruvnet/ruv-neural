//! Criterion benchmarks for DFA/Hurst long-range temporal correlation analysis.
//!
//! Benchmarks the optimized closed-form DFA against a naive reference
//! implementation (per-window allocation + explicit residual pass) to quantify
//! the optimization, and measures scaling across signal lengths.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use ruv_neural_signal::lrtc::{dfa, DfaConfig};

/// Deterministic uniform pseudo-random numbers in (-0.5, 0.5).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

fn white_noise(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    (0..n).map(|_| rng.next()).collect()
}

/// Naive DFA reference: same math, but materializes each window, fits, and
/// subtracts the trend sample-by-sample. Used only as a benchmark baseline.
fn dfa_naive(signal: &[f64], config: &DfaConfig) -> Option<f64> {
    let n = signal.len();
    if n < 4 * config.min_scale {
        return None;
    }
    let mean = signal.iter().sum::<f64>() / n as f64;
    let mut profile = Vec::with_capacity(n);
    let mut acc = 0.0;
    for &x in signal {
        acc += x - mean;
        profile.push(acc);
    }
    let max_scale = config.max_scale.unwrap_or(n / 4).min(n / 4);
    let log_min = (config.min_scale as f64).ln();
    let log_max = (max_scale as f64).ln();
    let mut scales = Vec::new();
    for i in 0..config.num_scales {
        let t = i as f64 / (config.num_scales - 1) as f64;
        let s = (log_min + t * (log_max - log_min)).exp().round() as usize;
        if scales.last() != Some(&s) {
            scales.push(s);
        }
    }
    let mut log_s = Vec::new();
    let mut log_f = Vec::new();
    for &s in &scales {
        let windows = n / s;
        let mut total = 0.0;
        let mut count = 0usize;
        let starts: Vec<usize> = (0..windows)
            .map(|k| k * s)
            .chain((0..windows).map(|k| n - (k + 1) * s))
            .collect();
        for &start in &starts {
            let window: Vec<f64> = profile[start..start + s].to_vec();
            let ts: Vec<f64> = (0..s).map(|t| t as f64).collect();
            let mt = ts.iter().sum::<f64>() / s as f64;
            let my = window.iter().sum::<f64>() / s as f64;
            let mut sxx = 0.0;
            let mut sxy = 0.0;
            for (t, y) in ts.iter().zip(&window) {
                sxx += (t - mt) * (t - mt);
                sxy += (t - mt) * (y - my);
            }
            let b = sxy / sxx;
            let a = my - b * mt;
            let rss: f64 = ts
                .iter()
                .zip(&window)
                .map(|(t, y)| {
                    let r = y - (a + b * t);
                    r * r
                })
                .sum();
            total += rss / s as f64;
            count += 1;
        }
        log_s.push((s as f64).log2());
        log_f.push((total / count as f64).sqrt().log2());
    }
    let m = log_s.len() as f64;
    let mx = log_s.iter().sum::<f64>() / m;
    let my = log_f.iter().sum::<f64>() / m;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (x, y) in log_s.iter().zip(&log_f) {
        sxx += (x - mx) * (x - mx);
        sxy += (x - mx) * (y - my);
    }
    Some(sxy / sxx)
}

fn bench_dfa_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dfa");
    let config = DfaConfig::default();
    for &n in &[1024usize, 4096, 16384, 65536] {
        let signal = white_noise(n, 42);
        group.bench_with_input(BenchmarkId::new("optimized", n), &signal, |b, sig| {
            b.iter(|| dfa(black_box(sig), black_box(&config)))
        });
    }
    group.finish();
}

fn bench_dfa_naive_vs_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("dfa_naive_vs_optimized");
    let config = DfaConfig::default();
    let signal = white_noise(16384, 42);
    group.bench_function("optimized_16384", |b| {
        b.iter(|| dfa(black_box(&signal), black_box(&config)))
    });
    group.bench_function("naive_16384", |b| {
        b.iter(|| dfa_naive(black_box(&signal), black_box(&config)))
    });
    group.finish();
}

criterion_group!(benches, bench_dfa_scaling, bench_dfa_naive_vs_optimized);
criterion_main!(benches);
