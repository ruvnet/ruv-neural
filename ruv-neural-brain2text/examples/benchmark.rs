//! Benchmark harness: compare acoustic models and baseline-vs-evolved configs.
//!
//! Run with: `cargo run --release -p ruv-neural-brain2text --example benchmark`
//!
//! Reports, on synthetic SpanishBCBL-like data:
//!   1. per-model test CER/WER + train time for the Brain2Qwerty V1 default config
//!   2. the evolved (Darwin-mode) best config across all model families
//!   3. decode throughput of the winning pipeline
//!
//! Numbers are on synthetic, learnable data — they benchmark the machinery and
//! the optimizer, not the published Brain2Qwerty accuracy.

use std::time::Instant;

use ruv_neural_brain2text::dataset::{generate_synthetic, SyntheticParams};
use ruv_neural_brain2text::evolve::{evolve, EvolveConfig};
use ruv_neural_brain2text::harness::Harness;
use ruv_neural_brain2text::{evaluate, Brain2TextConfig, EvalSplit, ModelKind};

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
        "hola que tal estas",
        "todo esta bien aqui",
        "vamos a la playa",
        "el sol brilla mucho",
        "me gusta el cafe",
        "la luna esta llena",
        "el cielo es azul",
        "tengo mucha hambre",
        "donde esta la salida",
        "manana sera otro dia",
    ]
}

fn main() {
    let sentences = corpus();
    let rec = generate_synthetic(&sentences, &SyntheticParams::default(), 2025);

    println!("== rUv Neural Brain-to-Text benchmark ==");
    println!(
        "data: {} ch, {:.0}s, {} sentences, {} keystrokes (synthetic)\n",
        rec.series.num_channels,
        rec.series.duration_s(),
        rec.timeline.sentences.len(),
        rec.timeline.num_keystrokes(),
    );

    println!("1) Per-model, Brain2Qwerty V1 default config (test split)");
    println!("   {:<11} {:>8} {:>8} {:>12}", "model", "CER", "WER", "train(ms)");
    println!("   {}", "-".repeat(41));
    for kind in [ModelKind::Prototype, ModelKind::Linear, ModelKind::Mlp] {
        let cfg = Brain2TextConfig {
            model: kind,
            ..Default::default()
        };
        let t0 = Instant::now();
        let res = evaluate(&rec, &cfg, EvalSplit::Test, 0.7, 0.15).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!(
            "   {:<11} {:>8.3} {:>8.3} {:>12.1}",
            format!("{:?}", kind),
            res.report.mean_cer,
            res.report.mean_wer,
            ms
        );
    }

    println!("\n2) Evolved best config (Darwin mode, across all model families)");
    let ec = EvolveConfig {
        population: 16,
        generations: 12,
        ..Default::default()
    };
    let t0 = Instant::now();
    let result = evolve(&rec, &ec).unwrap();
    let evolve_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let tuned = evaluate(&rec, &result.best.config, EvalSplit::Test, 0.7, 0.15).unwrap();
    println!(
        "   model={:?} val_fitness={:.3}  ->  test CER={:.3} WER={:.3}  (search {:.0} ms)",
        result.best.config.model, result.best.fitness, tuned.report.mean_cer, tuned.report.mean_wer, evolve_ms
    );
    println!(
        "   tuned: lr={:.3} epochs={} ngram={} lm_weight={:.2} beam={} feature={:?}",
        result.best.config.learning_rate,
        result.best.config.epochs,
        result.best.config.ngram_order,
        result.best.config.lm_weight,
        result.best.config.beam_size,
        result.best.config.feature,
    );

    println!("\n   improvement curve (best val fitness per generation):");
    for (g, f) in result.history.iter().enumerate() {
        println!("     gen {:>2}: {:.3} {}", g, f, "#".repeat((f * 40.0) as usize));
    }

    println!("\n3) Decode throughput (winning pipeline)");
    let pipeline = Harness::new()
        .with_config(result.best.config.clone())
        .fit(&rec)
        .unwrap();
    let pre = ruv_neural_brain2text::preprocess::preprocess(&rec.series, &pipeline.config).unwrap();
    let epochs = ruv_neural_brain2text::epoch::extract(&pre, &rec.timeline, &pipeline.config);
    let total_keys: usize = epochs.iter().map(|s| s.epochs.len()).sum();
    let t0 = Instant::now();
    let reps = 50;
    for _ in 0..reps {
        for s in &epochs {
            std::hint::black_box(pipeline.decode(&s.epochs));
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let keys_per_s = (total_keys * reps) as f64 / secs;
    println!(
        "   {:.0} keystrokes/sec decoded ({} keys x {} reps in {:.3}s)",
        keys_per_s, total_keys, reps, secs
    );

    // The trained pipeline is the distributable artifact.
    let json = pipeline.to_json().unwrap();
    println!("\nserialized pipeline artifact: {} bytes", json.len());
}
