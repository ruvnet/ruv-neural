//! Train a pipeline and export the distributable model artifact.
//!
//! Run with: `cargo run -p ruv-neural-brain2text --example train -- out/model.json`
//!
//! Fits a `TrainedPipeline` and writes it as JSON weights. With real SpanishBCBL
//! data the resulting weights inherit the dataset's **CC BY-NC 4.0** license —
//! see `WEIGHTS_LICENSE` and fill out `MODEL_CARD.md` before distributing.

use std::path::PathBuf;

use ruv_neural_brain2text::dataset::{generate_synthetic, SyntheticParams};
use ruv_neural_brain2text::harness::Harness;
use ruv_neural_brain2text::{Brain2TextConfig, EvalSplit};

fn main() {
    let out = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("brain2text_model.json"));

    // Stand-in data. Swap in a real Recording (BrainVision loader or sidecar) here.
    let sentences = [
        "hola mundo",
        "buenos dias amigo",
        "como estas hoy",
        "muy bien gracias",
        "hasta luego pronto",
        "que tengas buen dia",
        "nos vemos manana",
        "buenas noches a todos",
    ];
    let rec = generate_synthetic(&sentences, &SyntheticParams::default(), 1);

    let pipeline = Harness::new()
        .with_config(Brain2TextConfig::default())
        .with_split(0.8, 0.1)
        .fit(&rec)
        .unwrap();

    let report = pipeline.evaluate(&rec, EvalSplit::Test).unwrap();
    println!(
        "trained model={:?}  test CER={:.3} WER={:.3}",
        pipeline.config.model, report.report.mean_cer, report.report.mean_wer
    );

    let json = pipeline.to_json().unwrap();
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    std::fs::write(&out, &json).expect("write model artifact");
    println!("wrote {} ({} bytes)", out.display(), json.len());
    println!(
        "NOTE: if trained on SpanishBCBL, this artifact is CC BY-NC 4.0 \
         (non-commercial) — see WEIGHTS_LICENSE and MODEL_CARD.md."
    );
}
