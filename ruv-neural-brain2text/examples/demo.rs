//! End-to-end brain-to-text demo on synthetic SpanishBCBL-like data.
//!
//! Run with: `cargo run -p ruv-neural-brain2text --example brain2text_demo`
//!
//! It (1) generates a synthetic recording, (2) evaluates the Brain2Qwerty V1
//! default config, (3) evolves the config with the in-crate Darwin-mode
//! optimizer, and (4) reports the CER improvement.

use ruv_neural_brain2text::dataset::{generate_synthetic, SyntheticParams};
use ruv_neural_brain2text::evolve::{evolve, EvolveConfig};
use ruv_neural_brain2text::{evaluate, Brain2TextConfig, EvalSplit};

fn main() {
    let sentences = [
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
    ];

    println!("== rUv Neural Brain-to-Text demo (synthetic SpanishBCBL-like) ==\n");
    let rec = generate_synthetic(&sentences, &SyntheticParams::default(), 2025);
    println!(
        "recording: {} channels, {:.1}s, {} sentences, {} keystrokes\n",
        rec.series.num_channels,
        rec.series.duration_s(),
        rec.timeline.sentences.len(),
        rec.timeline.num_keystrokes(),
    );

    // Baseline: Brain2Qwerty V1 default config.
    let baseline = evaluate(&rec, &Brain2TextConfig::default(), EvalSplit::Test, 0.7, 0.15).unwrap();
    println!(
        "baseline (V1 defaults):   test CER = {:.3}  WER = {:.3}",
        baseline.report.mean_cer, baseline.report.mean_wer
    );

    // Evolve the config (Darwin mode).
    let ec = EvolveConfig {
        population: 16,
        generations: 12,
        ..Default::default()
    };
    let result = evolve(&rec, &ec).unwrap();
    println!(
        "evolved best (val fitness {:.3}): {:?}\n",
        result.best.fitness, result.best.config
    );

    // Final test-set evaluation with the evolved config.
    let tuned = evaluate(&rec, &result.best.config, EvalSplit::Test, 0.7, 0.15).unwrap();
    println!(
        "evolved config:           test CER = {:.3}  WER = {:.3}",
        tuned.report.mean_cer, tuned.report.mean_wer
    );

    println!("\nimprovement curve (best val fitness per generation):");
    for (g, f) in result.history.iter().enumerate() {
        let bars = (f * 40.0).round() as usize;
        println!("  gen {:>2}: {:.3} {}", g, f, "#".repeat(bars));
    }

    println!("\nsample decodes (evolved config, test split):");
    for (pred, target, cer) in tuned.report.per_sentence.iter().take(5) {
        println!("  cer {:.2}  target={:?}  pred={:?}", cer, target, pred);
    }
}
