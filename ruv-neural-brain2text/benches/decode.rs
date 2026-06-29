//! Criterion micro-benchmarks for the hot paths: training and decoding.
//!
//! Run with: `cargo bench -p ruv-neural-brain2text`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use ruv_neural_brain2text::dataset::{generate_synthetic, SyntheticParams};
use ruv_neural_brain2text::epoch::{extract, split, SentenceEpochs};
use ruv_neural_brain2text::harness::Harness;
use ruv_neural_brain2text::preprocess::preprocess;
use ruv_neural_brain2text::{Brain2TextConfig, Brain2TextDecoder, CharSequenceDecoder, ModelKind};

fn corpus() -> Vec<&'static str> {
    vec![
        "hola mundo",
        "buenos dias amigo",
        "como estas hoy",
        "muy bien gracias",
        "hasta luego pronto",
        "que tengas buen dia",
        "nos vemos manana",
        "buenas noches a todos",
        "feliz cumpleanos hoy",
        "muchas gracias por todo",
    ]
}

fn bench_train(c: &mut Criterion) {
    let rec = generate_synthetic(&corpus(), &SyntheticParams::default(), 1);
    let cfg = Brain2TextConfig::default();
    let pre = preprocess(&rec.series, &cfg).unwrap();
    let epochs = extract(&pre, &rec.timeline, &cfg);
    let (train, _v, _t) = split(&epochs, 0.7, 0.15);
    let train: Vec<&SentenceEpochs> = if train.is_empty() { epochs.iter().collect() } else { train };

    let mut group = c.benchmark_group("train");
    for kind in [ModelKind::Prototype, ModelKind::Linear, ModelKind::Mlp] {
        let cfg = Brain2TextConfig {
            model: kind,
            ..Default::default()
        };
        group.bench_with_input(BenchmarkId::from_parameter(format!("{kind:?}")), &cfg, |b, cfg| {
            b.iter(|| Brain2TextDecoder::train(&train, cfg));
        });
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let rec = generate_synthetic(&corpus(), &SyntheticParams::default(), 1);
    let pipeline = Harness::new().fit(&rec).unwrap();
    let pre = preprocess(&rec.series, &pipeline.config).unwrap();
    let epochs = extract(&pre, &rec.timeline, &pipeline.config);
    let sentence = &epochs[0].epochs;

    c.bench_function("decode_sentence", |b| {
        b.iter(|| pipeline.decoder.decode_sentence(sentence));
    });
}

criterion_group!(benches, bench_train, bench_decode);
criterion_main!(benches);
