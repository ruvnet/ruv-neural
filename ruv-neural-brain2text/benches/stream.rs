//! Criterion benchmarks for the causal streaming decoder.
//!
//! Measures per-frame decode cost against the published streaming budget:
//! the 2025 streaming brain-to-voice system meets an 80-ms per-step budget in
//! 99.3% of steps, and UC Davis runs a causal decoder at a 10-ms hop. These
//! benchmarks report the margin an edge Rust implementation has against both
//! cadences at realistic EEG channel counts.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use ruv_neural_brain2text::model::AcousticModel;
use ruv_neural_brain2text::stream::{StreamConfig, StreamingDecoder};
use ruv_neural_core::gate::DecodeGateConfig;

/// Deterministic stub model: isolates decoder plumbing cost from model cost.
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

fn chunk(channels: usize, n: usize) -> Vec<Vec<f64>> {
    (0..channels)
        .map(|c| {
            (0..n)
                .map(|i| ((i + c) as f64 * 0.01).sin())
                .collect::<Vec<f64>>()
        })
        .collect()
}

/// Per-frame cost across channel counts, at the 80-ms cadence.
fn bench_frame_cost(c: &mut Criterion) {
    let mut group = c.benchmark_group("stream_frame");
    let fs = 250.0;
    for &channels in &[1usize, 8, 64, 306] {
        let mut decoder = StreamingDecoder::new(
            StreamConfig::default(),
            StubModel,
            DecodeGateConfig::default(),
            channels,
            fs,
        )
        .unwrap();
        // 80 ms at 250 Hz = 20 samples: exactly one frame per push.
        let data = chunk(channels, 20);
        let mut t = 0.0;
        group.bench_with_input(BenchmarkId::new("channels", channels), &data, |b, data| {
            b.iter(|| {
                t += 0.08;
                decoder.push_chunk(black_box(data), 0.0, t)
            })
        });
    }
    group.finish();
}

/// Per-frame cost at the 10-ms cadence (the UC Davis regime).
fn bench_ten_ms_cadence(c: &mut Criterion) {
    let mut group = c.benchmark_group("stream_cadence");
    let fs = 1000.0;
    for &(label, hop_s) in &[("80ms", 0.080), ("10ms", 0.010)] {
        let channels = 64;
        let mut decoder = StreamingDecoder::new(
            StreamConfig {
                frame_hop_s: hop_s,
                window_s: 0.5,
                ..StreamConfig::default()
            },
            StubModel,
            DecodeGateConfig::default(),
            channels,
            fs,
        )
        .unwrap();
        let hop_samples = (hop_s * fs) as usize;
        let data = chunk(channels, hop_samples);
        let mut t = 0.0;
        group.bench_function(label, |b| {
            b.iter(|| {
                t += hop_s;
                decoder.push_chunk(black_box(&data), 0.0, t)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_frame_cost, bench_ten_ms_cadence);
criterion_main!(benches);
