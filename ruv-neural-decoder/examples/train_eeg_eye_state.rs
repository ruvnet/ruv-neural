//! Train and evaluate the logistic-regression decoder on the public UCI
//! "EEG Eye State" dataset (14 Emotiv channels → eye open/closed).
//!
//! Honest-evaluation notes (ADR-0015/0019):
//! - The dataset is a **single continuous recording**, so a random shuffle leaks
//!   information across adjacent (highly autocorrelated) samples and inflates
//!   accuracy. The headline number here uses a **chronological** split (train on
//!   the earlier portion, test on the later portion); the shuffled number is
//!   printed alongside, explicitly labeled as optimistic.
//! - Hyperparameters are tuned on a validation slice carved from the *training*
//!   portion only — never on the test set.
//!
//! Usage (download the public ARFF, then run):
//!
//! ```text
//! curl -L -o /tmp/eeg_eye_state.arff \
//!   "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
//! cargo run -p ruv-neural-decoder --example train_eeg_eye_state -- /tmp/eeg_eye_state.arff
//! ```

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use ruv_neural_decoder::logistic::{BinaryMetrics, LogisticRegression, TrainConfig};

const N_FEATURES: usize = 14;
// Plausible Emotiv amplitude range (µV); a handful of glitch rows sit far
// outside this and are dropped before standardization.
const LO: f64 = 1000.0;
const HI: f64 = 6000.0;

fn parse_arff(path: &str) -> (Vec<Vec<f64>>, Vec<u8>) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut dropped = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('@') || line.starts_with('%') {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != N_FEATURES + 1 {
            continue;
        }
        let feats: Vec<f64> = parts[..N_FEATURES]
            .iter()
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect();
        if feats.len() != N_FEATURES {
            continue;
        }
        if feats.iter().any(|&v| !(LO..=HI).contains(&v)) {
            dropped += 1;
            continue;
        }
        let label = parts[N_FEATURES].trim().parse::<u8>().unwrap_or(0);
        x.push(feats);
        y.push(label);
    }
    eprintln!(
        "parsed {} samples ({} glitch rows dropped), positives = {}",
        x.len(),
        dropped,
        y.iter().filter(|&&v| v == 1).count()
    );
    (x, y)
}

/// Trailing-window per-channel variance features.
///
/// Instantaneous voltage is a poor eye-state feature and drifts over the
/// recording; eyes-closed instead raises occipital **alpha-band power**, which
/// shows up as higher per-channel *variance*. For each time `t >= w` we emit the
/// variance of each channel over the causal window `[t-w, t)` (drift-robust,
/// uses only past samples — safe for a chronological split), keeping the label
/// `y[t]`.
fn windowed_variance_features(raw: &[Vec<f64>], y: &[u8], w: usize) -> (Vec<Vec<f64>>, Vec<u8>) {
    let mut x = Vec::with_capacity(raw.len().saturating_sub(w));
    let mut yo = Vec::with_capacity(raw.len().saturating_sub(w));
    for t in w..raw.len() {
        let mut feats = Vec::with_capacity(N_FEATURES);
        for c in 0..N_FEATURES {
            let mut mean = 0.0;
            for row in &raw[t - w..t] {
                mean += row[c];
            }
            mean /= w as f64;
            let mut var = 0.0;
            for row in &raw[t - w..t] {
                var += (row[c] - mean).powi(2);
            }
            // Log band-power: alpha power is heavy-tailed, and log linearizes it
            // for the linear classifier (the classic band-power pipeline).
            feats.push((var / w as f64 + 1.0).ln());
        }
        x.push(feats);
        yo.push(y[t]);
    }
    (x, yo)
}

/// Non-overlapping log-power windows: stride == w, so windows share **no**
/// samples — a cleaner unit for cross-validation than the stride-1 features.
fn nonoverlapping_features(raw: &[Vec<f64>], y: &[u8], w: usize) -> (Vec<Vec<f64>>, Vec<u8>) {
    let mut x = Vec::new();
    let mut yo = Vec::new();
    let mut t = w;
    while t <= raw.len() {
        let mut feats = Vec::with_capacity(N_FEATURES);
        for c in 0..N_FEATURES {
            let mean = raw[t - w..t].iter().map(|r| r[c]).sum::<f64>() / w as f64;
            let var = raw[t - w..t]
                .iter()
                .map(|r| (r[c] - mean).powi(2))
                .sum::<f64>()
                / w as f64;
            feats.push((var + 1.0).ln());
        }
        x.push(feats);
        // Label the window by its majority eye-state.
        let pos = y[t - w..t].iter().filter(|&&v| v == 1).count();
        yo.push(u8::from(pos * 2 >= w));
        t += w;
    }
    (x, yo)
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

/// Stratified-ish k-fold CV on non-overlapping windows (shuffled, fixed seed).
fn kfold_cv(x: &[Vec<f64>], y: &[u8], k: usize, cfg: &TrainConfig) {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.shuffle(&mut StdRng::seed_from_u64(7));

    let (mut acc_sum, mut bal_sum) = (0.0, 0.0);
    for fold in 0..k {
        let test_ids: Vec<usize> = idx.iter().copied().skip(fold).step_by(k).collect();
        let test_set: std::collections::HashSet<usize> = test_ids.iter().copied().collect();

        let (mut xtr, mut ytr, mut xte, mut yte) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..x.len() {
            if test_set.contains(&i) {
                xte.push(x[i].clone());
                yte.push(y[i]);
            } else {
                xtr.push(x[i].clone());
                ytr.push(y[i]);
            }
        }
        let (model, _) = LogisticRegression::fit(&xtr, &ytr, cfg).unwrap();
        let m = model.evaluate(&xte, &yte);
        acc_sum += m.accuracy;
        bal_sum += balanced_accuracy(&m);
    }
    println!(
        "  {k}-fold CV: mean acc={:.4}  mean balanced-acc={:.4}",
        acc_sum / k as f64,
        bal_sum / k as f64
    );
}

fn print_metrics(label: &str, m: &BinaryMetrics) {
    println!(
        "  {label:<26} acc={:.4} prec={:.4} rec={:.4} f1={:.4}  [tp={} fp={} tn={} fn={}]",
        m.accuracy, m.precision, m.recall, m.f1, m.tp, m.fp, m.tn, m.fn_
    );
}

/// Tune over a small grid using a validation slice carved from `train`.
fn tune(x_train: &[Vec<f64>], y_train: &[u8]) -> TrainConfig {
    let n = x_train.len();
    let cut = (n as f64 * 0.85) as usize;
    let (xt, xv) = x_train.split_at(cut);
    let (yt, yv) = y_train.split_at(cut);

    let mut best = TrainConfig::default();
    let mut best_acc = -1.0;
    for &lr in &[0.05, 0.1, 0.3] {
        for &l2 in &[1e-4, 1e-3, 1e-2] {
            let cfg = TrainConfig {
                learning_rate: lr,
                l2,
                epochs: 400,
            };
            let (model, _) = LogisticRegression::fit(xt, yt, &cfg).unwrap();
            let m = model.evaluate(xv, yv);
            // Balanced accuracy: robust to class imbalance in the val slice.
            let rec_pos = if m.tp + m.fn_ > 0 {
                m.tp as f64 / (m.tp + m.fn_) as f64
            } else {
                0.0
            };
            let rec_neg = if m.tn + m.fp > 0 {
                m.tn as f64 / (m.tn + m.fp) as f64
            } else {
                0.0
            };
            let bal = 0.5 * (rec_pos + rec_neg);
            if bal > best_acc {
                best_acc = bal;
                best = cfg;
            }
        }
    }
    println!(
        "  tuned: lr={} l2={} epochs={} (val bal-acc={:.4})",
        best.learning_rate, best.l2, best.epochs, best_acc
    );
    best
}

fn run_split(name: &str, x: &[Vec<f64>], y: &[u8]) {
    let n = x.len();
    let cut = (n as f64 * 0.7) as usize;
    let (x_train, x_test) = x.split_at(cut);
    let (y_train, y_test) = y.split_at(cut);
    println!("\n{name} (train={}, test={}):", x_train.len(), x_test.len());

    let cfg = tune(x_train, y_train);
    let (model, history) = LogisticRegression::fit(x_train, y_train, &cfg).unwrap();
    println!(
        "  log-loss: {:.4} -> {:.4} over {} epochs",
        history.first().unwrap(),
        history.last().unwrap(),
        history.len()
    );

    // Majority-class baseline for honest context.
    let pos = y_test.iter().filter(|&&v| v == 1).count();
    let majority = pos.max(y_test.len() - pos) as f64 / y_test.len() as f64;
    println!("  majority-class baseline acc={:.4}", majority);

    print_metrics("train", &model.evaluate(x_train, y_train));
    print_metrics("TEST (out-of-sample)", &model.evaluate(x_test, y_test));
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/eeg_eye_state.arff".to_string());

    let (x, y) = parse_arff(&path);
    if x.len() < 100 {
        eprintln!("not enough data");
        std::process::exit(1);
    }

    println!("=== UCI EEG Eye State — logistic-regression decoder ===");

    // Raw instantaneous amplitude (weak, drift-prone) for contrast.
    run_split("RAW amplitude, CHRONOLOGICAL [weak baseline]", &x, &y);

    // Optimized features: trailing-window per-channel variance (alpha power).
    let w = 64;
    let (xf, yf) = windowed_variance_features(&x, &y, w);
    println!(
        "\n--- optimized features: {w}-sample trailing variance ({} windows) ---",
        xf.len()
    );

    // Honest protocol: chronological split (no temporal leakage).
    run_split("VARIANCE, CHRONOLOGICAL [honest]", &xf, &yf);

    // Comparison: shuffled split (optimistic — leaks autocorrelation).
    let mut idx: Vec<usize> = (0..xf.len()).collect();
    idx.shuffle(&mut StdRng::seed_from_u64(42));
    let xs: Vec<Vec<f64>> = idx.iter().map(|&i| xf[i].clone()).collect();
    let ys: Vec<u8> = idx.iter().map(|&i| yf[i]).collect();
    run_split(
        "VARIANCE, SHUFFLED [optimistic — temporal leakage]",
        &xs,
        &ys,
    );

    // Cleanest "does it learn?" evidence: k-fold CV on NON-overlapping windows
    // (no shared samples between folds), swept over window size, beside baseline.
    println!("\n--- leakage-free k-fold CV on non-overlapping windows ---");
    for &ww in &[64usize, 128, 256] {
        let (xn, yn) = nonoverlapping_features(&x, &y, ww);
        let pos = yn.iter().filter(|&&v| v == 1).count();
        let baseline = pos.max(yn.len() - pos) as f64 / yn.len() as f64;
        print!(
            "  w={ww:<4} ({:>4} windows, baseline acc={:.4}): ",
            xn.len(),
            baseline
        );
        kfold_cv(
            &xn,
            &yn,
            5,
            &TrainConfig {
                learning_rate: 0.3,
                l2: 1e-3,
                epochs: 600,
            },
        );
    }

    println!(
        "\nNote: the chronological TEST number is the strict, leakage-free bound; the\n\
         shuffled split and stride-1 windows share autocorrelated neighbours and\n\
         overstate skill. The non-overlapping k-fold CV is the fair 'does it learn'\n\
         estimate — compare it against the majority-class baseline."
    );
}
