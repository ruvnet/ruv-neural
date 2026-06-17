//! Train and evaluate the logistic-regression decoder on the public
//! "Epileptic Seizure Recognition" dataset (the Bonn EEG corpus, reshaped):
//! 500 single-channel 23.6 s recordings, each cut into 23 one-second, 178-sample
//! chunks → 11,500 rows, 5 classes. Binary task here: **seizure (class 1) vs
//! the rest** (majority-class baseline ≈ 0.80).
//!
//! Honest-evaluation notes (ADR-0015/0019): the 23 chunks of one recording are
//! correlated, so a random *row* split leaks. Every protocol below is **grouped
//! by source recording** (all 23 chunks stay on one side), which removes
//! temporal-autocorrelation leakage. NOTE: grouping is by *recording*, not by
//! subject — the reshaped CSV has only segment IDs, so this is **not**
//! patient-independent (the clinical gold standard); it shows the trainer on a
//! separable public task, not generalization to unseen patients.
//! A random-row split is also shown to quantify the inflation.
//!
//! Usage:
//!
//! ```text
//! curl -L -o /tmp/esr.csv \
//!   "https://raw.githubusercontent.com/QiuyiWu/Epileptic-Seizure-Recognition-Data/master/A%26B%26C%26D%26E.csv"
//! cargo run -p ruv-neural-decoder --example train_seizure -- /tmp/esr.csv
//! ```

use std::collections::BTreeSet;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use ruv_neural_decoder::logistic::{BinaryMetrics, LogisticRegression, TrainConfig};

const N_RAW: usize = 178;

struct Sample {
    group: u32,
    raw: Vec<f64>,
    y: u8,
}

fn parse_csv(path: &str) -> Vec<Sample> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue; // header
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != N_RAW + 2 {
            continue;
        }
        // id like "X21.V1.791" → group = the trailing number (source recording).
        let id = f[0].trim_matches('"');
        let group = id
            .rsplit('.')
            .next()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);
        let raw: Vec<f64> = f[1..=N_RAW]
            .iter()
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect();
        if raw.len() != N_RAW {
            continue;
        }
        let label = f[N_RAW + 1].trim().parse::<u8>().unwrap_or(0);
        out.push(Sample {
            group,
            raw,
            y: u8::from(label == 1), // class 1 = seizure
        });
    }
    out
}

/// Compact, physiologically-motivated per-chunk features. Seizure EEG has much
/// larger amplitude swings, so power/line-length/range separate it strongly.
fn features(raw: &[f64]) -> Vec<f64> {
    let n = raw.len() as f64;
    let mean = raw.iter().sum::<f64>() / n;
    let var = raw.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let line_length: f64 = raw.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
    let max = raw.iter().cloned().fold(f64::MIN, f64::max);
    let min = raw.iter().cloned().fold(f64::MAX, f64::min);
    let mav = raw.iter().map(|x| (x - mean).abs()).sum::<f64>() / n;
    // Zero-crossing rate about the mean (rhythmicity proxy).
    let zc = raw
        .windows(2)
        .filter(|w| (w[0] - mean).signum() != (w[1] - mean).signum())
        .count() as f64
        / n;
    vec![
        (var + 1.0).ln(),
        (line_length + 1.0).ln(),
        (max - min).ln().max(0.0),
        mav.ln().max(0.0),
        zc,
    ]
}

fn balanced_accuracy(m: &BinaryMetrics) -> f64 {
    let rp = if m.tp + m.fn_ > 0 {
        m.tp as f64 / (m.tp + m.fn_) as f64
    } else {
        0.0
    };
    let rn = if m.tn + m.fp > 0 {
        m.tn as f64 / (m.tn + m.fp) as f64
    } else {
        0.0
    };
    0.5 * (rp + rn)
}

fn build(samples: &[Sample], engineered: bool) -> (Vec<Vec<f64>>, Vec<u8>, Vec<u32>) {
    let mut x = Vec::with_capacity(samples.len());
    let mut y = Vec::with_capacity(samples.len());
    let mut g = Vec::with_capacity(samples.len());
    for s in samples {
        x.push(if engineered {
            features(&s.raw)
        } else {
            s.raw.clone()
        });
        y.push(s.y);
        g.push(s.group);
    }
    (x, y, g)
}

fn report(tag: &str, model: &LogisticRegression, x: &[Vec<f64>], y: &[u8]) {
    let m = model.evaluate(x, y);
    println!(
        "  {tag:<34} acc={:.4} bal-acc={:.4} prec={:.4} rec={:.4} f1={:.4}",
        m.accuracy,
        balanced_accuracy(&m),
        m.precision,
        m.recall,
        m.f1
    );
}

/// `(x_train, y_train, x_test, y_test)`.
type Split = (Vec<Vec<f64>>, Vec<u8>, Vec<Vec<f64>>, Vec<u8>);

/// Leakage-free grouped split: whole recordings go to train or test.
fn grouped_split(x: &[Vec<f64>], y: &[u8], g: &[u32], train_frac: f64, seed: u64) -> Split {
    let mut groups: Vec<u32> = g
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    groups.shuffle(&mut StdRng::seed_from_u64(seed));
    let cut = (groups.len() as f64 * train_frac) as usize;
    let train_groups: BTreeSet<u32> = groups[..cut].iter().copied().collect();

    let (mut xt, mut yt, mut xe, mut ye) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..x.len() {
        if train_groups.contains(&g[i]) {
            xt.push(x[i].clone());
            yt.push(y[i]);
        } else {
            xe.push(x[i].clone());
            ye.push(y[i]);
        }
    }
    (xt, yt, xe, ye)
}

/// Grouped k-fold CV: each fold holds out a disjoint set of recordings.
fn grouped_cv(x: &[Vec<f64>], y: &[u8], g: &[u32], k: usize, cfg: &TrainConfig) -> (f64, f64) {
    let mut groups: Vec<u32> = g
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    groups.shuffle(&mut StdRng::seed_from_u64(7));
    let (mut acc, mut bal) = (0.0, 0.0);
    for fold in 0..k {
        let test_groups: BTreeSet<u32> = groups.iter().copied().skip(fold).step_by(k).collect();
        let (mut xt, mut yt, mut xe, mut ye) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..x.len() {
            if test_groups.contains(&g[i]) {
                xe.push(x[i].clone());
                ye.push(y[i]);
            } else {
                xt.push(x[i].clone());
                yt.push(y[i]);
            }
        }
        let (model, _) = LogisticRegression::fit(&xt, &yt, cfg).unwrap();
        let m = model.evaluate(&xe, &ye);
        acc += m.accuracy;
        bal += balanced_accuracy(&m);
    }
    (acc / k as f64, bal / k as f64)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/esr.csv".to_string());
    let samples = parse_csv(&path);
    let pos = samples.iter().filter(|s| s.y == 1).count();
    println!(
        "=== Epileptic Seizure Recognition — seizure vs rest ===\n\
         {} chunks, {} recordings, seizure={} ({:.1}%), majority baseline={:.4}",
        samples.len(),
        samples
            .iter()
            .map(|s| s.group)
            .collect::<BTreeSet<_>>()
            .len(),
        pos,
        100.0 * pos as f64 / samples.len() as f64,
        (samples.len() - pos) as f64 / samples.len() as f64,
    );

    let cfg = TrainConfig {
        learning_rate: 0.5,
        l2: 1e-3,
        epochs: 600,
    };

    for (name, engineered) in [("raw 178-sample", false), ("engineered 5-feature", true)] {
        let (x, y, g) = build(&samples, engineered);
        println!("\n--- {name} features ---");

        // Honest: grouped 70/30 split (no recording spans train and test).
        let (xt, yt, xe, ye) = grouped_split(&x, &y, &g, 0.7, 1);
        let (model, hist) = LogisticRegression::fit(&xt, &yt, &cfg).unwrap();
        println!("  log-loss {:.4} -> {:.4}", hist[0], hist.last().unwrap());
        report("GROUPED split TEST [honest]", &model, &xe, &ye);

        // Honest: grouped 5-fold CV.
        let (acc, bal) = grouped_cv(&x, &y, &g, 5, &cfg);
        println!("  GROUPED 5-fold CV [honest]         acc={acc:.4} bal-acc={bal:.4}");

        // Optimistic contrast: random-row split (chunks leak across the split).
        let mut idx: Vec<usize> = (0..x.len()).collect();
        idx.shuffle(&mut StdRng::seed_from_u64(1));
        let cut = idx.len() * 7 / 10;
        let xt: Vec<_> = idx[..cut].iter().map(|&i| x[i].clone()).collect();
        let yt: Vec<_> = idx[..cut].iter().map(|&i| y[i]).collect();
        let xe: Vec<_> = idx[cut..].iter().map(|&i| x[i].clone()).collect();
        let ye: Vec<_> = idx[cut..].iter().map(|&i| y[i]).collect();
        let (model, _) = LogisticRegression::fit(&xt, &yt, &cfg).unwrap();
        report("random-row split [leaky/optimistic]", &model, &xe, &ye);
    }

    // ── End-to-end: persist the trained model in a SIGNED RVF container ──
    persist_signed(&samples, &cfg);

    println!(
        "\nThe GROUPED numbers are the honest ones — no recording is split across\n\
         train and test. Seizure detection has real, strongly separable signal, so\n\
         here the leakage-free model genuinely beats the baseline (contrast the\n\
         EEG-eye-state benchmark, where it did not)."
    );
}

/// Train on a grouped split, write the model to a signed `.rvf`, reload it,
/// verify the signature, and confirm the reloaded model reproduces the test
/// predictions — exercising the whole stack (trainer → RVFS container → Ed25519).
fn persist_signed(samples: &[Sample], cfg: &TrainConfig) {
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;
    use ruv_neural_core::rvf_container::RvfContainer;
    use ruv_neural_core::rvf_witness::{sign_container, verify_container_signature};
    use ruv_neural_decoder::{container_to_model, model_to_container};

    println!("\n--- persist trained model in a signed .rvf ---");
    let (x, y, g) = build(samples, true);
    let (xt, yt, xe, ye) = grouped_split(&x, &y, &g, 0.7, 1);
    let (model, _) = LogisticRegression::fit(&xt, &yt, cfg).unwrap();
    let test_acc = model.evaluate(&xe, &ye).accuracy;

    // Serialize → sign → bytes.
    let mut container = model_to_container(&model).unwrap();
    let key = SigningKey::generate(&mut OsRng);
    sign_container(&mut container, &key);
    let bytes = container.to_bytes();
    let path = "/tmp/seizure_model.rvf";
    std::fs::write(path, &bytes).unwrap();

    // Reload → verify → predict.
    let reloaded = RvfContainer::from_bytes(&std::fs::read(path).unwrap()).unwrap();
    reloaded.verify_integrity().unwrap();
    let sig_ok = verify_container_signature(&reloaded).unwrap();
    let loaded = container_to_model(&reloaded).unwrap();
    let loaded_acc = loaded.evaluate(&xe, &ye).accuracy;

    println!(
        "  wrote {} bytes to {path}; integrity+CRC ok, signature verified={sig_ok}",
        bytes.len()
    );
    println!(
        "  in-memory test acc={test_acc:.4}  ==  reloaded-model test acc={loaded_acc:.4}  (match={})",
        (test_acc - loaded_acc).abs() < 1e-12
    );
    let _ = std::fs::remove_file(path);
}
