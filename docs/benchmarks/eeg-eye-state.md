# Benchmark — UCI EEG Eye State (logistic-regression decoder)

A trained-on-public-data benchmark for the `LogisticRegression` decoder
(`ruv-neural-decoder`), run under the claims-discipline of ADR-0015 (honest,
out-of-sample evaluation) and ADR-0019 (surface null/contested results).

- **Data:** UCI "EEG Eye State" — 14 Emotiv channels, eye open/closed,
  14,980 samples from a **single continuous recording**
  (`archive.ics.uci.edu/.../00264/EEG Eye State.arff`).
- **Model:** binary logistic regression, full-batch gradient descent, feature
  standardization, L2; hyperparameters tuned on a validation slice of the
  *training* portion only.
- **Reproduce:** `cargo run -p ruv-neural-decoder --example train_eeg_eye_state -- <arff>`

## Results

| Protocol | Features | Test acc | Majority baseline |
|---|---|---:|---:|
| Chronological split (strict, leakage-free) | raw amplitude | 0.40 | 0.74 |
| Chronological split (strict, leakage-free) | log band-power (var, w=64) | 0.43 | 0.74 |
| Shuffled split (per-sample) | log band-power | 0.64 | 0.55 |
| **5-fold CV, non-overlapping windows** | log band-power, w∈{64,128,256} | **0.47–0.50** | 0.55–0.57 |

## The honest finding

With **temporal-autocorrelation leakage removed** — non-overlapping windows in
cross-validation, or a strict chronological split — a linear log-power model on
this single-session recording performs **at chance** and does not beat the
majority-class baseline. The much higher accuracies commonly reported on this
dataset come largely from **per-sample shuffling**, which places near-identical
neighbouring samples in both train and test.

This is a textbook illustration of the evaluation pitfalls ADR-0015 cites for
EEG models (in-sample leakage; simple baselines competitive; unsolved
distribution shift) — and exactly why this project keeps embeddings small,
auditable, and benchmarked out-of-sample rather than chasing leaky headline
numbers. The model itself is verified correct on a separable problem
(`logistic::tests::learns_separable_problem`, >0.95 accuracy); the limitation
is the dataset-under-honest-evaluation, not the trainer.
