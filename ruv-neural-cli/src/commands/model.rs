//! `train`, `model-info`, and `predict` — the trainable-model lifecycle, with
//! models persisted as signed RVF containers.

use std::error::Error;

use ruv_neural_core::rvf_container::RvfContainer;
use ruv_neural_core::rvf_witness::{sign_container_ephemeral, verify_container_signature};
use ruv_neural_decoder::logistic::{LogisticRegression, TrainConfig};
use ruv_neural_decoder::{container_to_model, model_to_container};

type Rows = (Vec<Vec<f64>>, Vec<u8>);

/// Parse a numeric CSV/ARFF table into feature rows + binary labels.
///
/// Lines starting with `@`, `%`, or `#` (ARFF / comments) and blank lines are
/// skipped. `skip_cols` leading columns are dropped (e.g. an id column). The
/// last remaining column is the label, mapped to 1 iff it equals `positive`.
fn parse_table(path: &str, skip_cols: usize, positive: i64) -> Result<Rows, Box<dyn Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut dim: Option<usize> = None;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('@') || line.starts_with('%') || line.starts_with('#')
        {
            continue;
        }
        let fields: Vec<&str> = line.split(',').skip(skip_cols).collect();
        if fields.len() < 2 {
            continue;
        }
        let (label_str, feat_strs) = fields.split_last().unwrap();
        // A header row (non-numeric features) is silently skipped.
        let feats: Option<Vec<f64>> = feat_strs
            .iter()
            .map(|s| s.trim().trim_matches('"').parse::<f64>().ok())
            .collect();
        let (Some(feats), Ok(label)) = (feats, label_str.trim().parse::<f64>()) else {
            continue;
        };
        if let Some(d) = dim {
            if feats.len() != d {
                continue;
            }
        } else {
            dim = Some(feats.len());
        }
        x.push(feats);
        y.push(u8::from(label.round() as i64 == positive));
    }

    if x.is_empty() {
        return Err(format!("no numeric rows parsed from '{path}'").into());
    }
    Ok((x, y))
}

/// Deterministic in-place Fisher–Yates shuffle (no external RNG dependency).
fn shuffle_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state % (i as u64 + 1)) as usize;
        idx.swap(i, j);
    }
    idx
}

/// `train` — fit a logistic-regression model and save it as a signed `.rvf`.
#[allow(clippy::too_many_arguments)]
pub fn train(
    input: &str,
    output: &str,
    skip_cols: usize,
    positive: i64,
    test_frac: f64,
    shuffle: bool,
    seed: u64,
    epochs: usize,
) -> Result<(), Box<dyn Error>> {
    let (x, y) = parse_table(input, skip_cols, positive)?;
    let n = x.len();
    let pos = y.iter().filter(|&&v| v == 1).count();
    println!("=== rUv Neural \u{2014} train ===");
    println!("  rows={n}  features={}  positives={pos} ({:.1}%)", x[0].len(), 100.0 * pos as f64 / n as f64);

    let order = if shuffle {
        shuffle_indices(n, seed)
    } else {
        (0..n).collect()
    };
    let cut = ((n as f64) * (1.0 - test_frac)) as usize;
    let cut = cut.clamp(1, n.saturating_sub(1).max(1));
    let pick = |ids: &[usize]| -> Rows {
        (ids.iter().map(|&i| x[i].clone()).collect(), ids.iter().map(|&i| y[i]).collect())
    };
    let (xt, yt) = pick(&order[..cut]);
    let (xe, ye) = pick(&order[cut..]);

    let cfg = TrainConfig { learning_rate: 0.5, l2: 1e-3, epochs };
    let (model, history) = LogisticRegression::fit(&xt, &yt, &cfg)?;
    println!("  log-loss {:.4} -> {:.4} over {epochs} epochs", history[0], history.last().unwrap());

    if !xe.is_empty() {
        let m = model.evaluate(&xe, &ye);
        let base_pos = ye.iter().filter(|&&v| v == 1).count();
        let baseline = base_pos.max(ye.len() - base_pos) as f64 / ye.len() as f64;
        println!(
            "  holdout test ({} rows{}): acc={:.4} prec={:.4} rec={:.4} f1={:.4}  (majority baseline={:.4})",
            ye.len(),
            if shuffle { ", shuffled" } else { ", chronological" },
            m.accuracy, m.precision, m.recall, m.f1, baseline
        );
        if shuffle {
            println!("  note: a shuffled holdout can leak temporal autocorrelation; see docs/benchmarks.");
        }
    }

    // Persist as a signed RVF model.
    let mut container = model_to_container(&model)?;
    let pubkey = sign_container_ephemeral(&mut container);
    std::fs::write(output, container.to_bytes())?;
    println!(
        "  wrote signed model to {output} ({} features, self-signed key {})",
        model.num_features(),
        &hex8(pubkey.as_bytes())
    );
    Ok(())
}

/// `model-info` — load a `.rvf` model, verify it, and print its descriptor.
pub fn info(input: &str) -> Result<(), Box<dyn Error>> {
    let bytes = std::fs::read(input)?;
    let container = RvfContainer::from_bytes(&bytes)?;
    println!("=== rUv Neural \u{2014} model-info ===");
    println!("  file: {input} ({} bytes, {} segments)", bytes.len(), container.segments.len());

    match container.verify_integrity() {
        Ok(()) => println!("  integrity: OK (CRC32C + content-hash)"),
        Err(e) => println!("  integrity: FAILED \u{2014} {e}"),
    }
    match verify_container_signature(&container) {
        Ok(true) => println!("  signature: VALID (Ed25519)"),
        Ok(false) => println!("  signature: INVALID"),
        Err(_) => println!("  signature: (none)"),
    }
    match container_to_model(&container) {
        Ok(model) => println!("  model: logistic-regression, {} features", model.num_features()),
        Err(e) => println!("  model: not loadable \u{2014} {e}"),
    }
    Ok(())
}

/// `predict` — load+verify a `.rvf` model and score rows from a CSV.
pub fn predict(model_path: &str, input: &str, skip_cols: usize, proba: bool) -> Result<(), Box<dyn Error>> {
    let container = RvfContainer::from_bytes(&std::fs::read(model_path)?)?;
    container.verify_integrity()?;
    if let Ok(false) = verify_container_signature(&container) {
        return Err("model signature is invalid \u{2014} refusing to run".into());
    }
    let model = container_to_model(&container)?;

    // Reuse the table parser with a dummy trailing label column requirement: here
    // the input has no labels, so append a 0 per row by reading raw features.
    let text = std::fs::read_to_string(input)?;
    let mut count = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('@') || line.starts_with('%') || line.starts_with('#') {
            continue;
        }
        let feats: Option<Vec<f64>> = line
            .split(',')
            .skip(skip_cols)
            .map(|s| s.trim().trim_matches('"').parse::<f64>().ok())
            .collect();
        let Some(feats) = feats else { continue };
        if feats.len() != model.num_features() {
            continue;
        }
        if proba {
            println!("{:.6}", model.predict_proba(&feats));
        } else {
            println!("{}", model.predict(&feats));
        }
        count += 1;
    }
    eprintln!("scored {count} rows");
    Ok(())
}

fn hex8(bytes: &[u8]) -> String {
    bytes.iter().take(4).map(|b| format!("{b:02x}")).collect()
}
