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

---

# Benchmark — Epileptic Seizure Recognition (the honest win)

The same trainer on a task with **real, separable signal** and a naturally
**leakage-free split**, for contrast.

- **Data:** "Epileptic Seizure Recognition" (the Bonn EEG corpus, reshaped) —
  500 single-channel 23.6 s recordings, each cut into 23 one-second 178-sample
  chunks → 11,500 rows, 5 classes. Task: **seizure (class 1) vs rest**.
- **Leakage control:** the 23 chunks of a recording are correlated, so every
  honest protocol is **grouped by source recording** (all chunks stay on one
  side). A random-row split is shown only for contrast.
- **Reproduce:** `cargo run -p ruv-neural-decoder --example train_seizure -- <csv>`

## Results (seizure vs rest; majority baseline = 0.80)

| Features | Protocol | Accuracy | Balanced acc | F1 |
|---|---|---:|---:|---:|
| raw 178-sample | grouped 5-fold CV | 0.81 | 0.53 | — |
| **engineered 5-feature** | **grouped 5-fold CV [honest]** | **0.96** | **0.92** | — |
| engineered 5-feature | grouped held-out split [honest] | 0.97 | 0.95 | 0.93 |
| engineered 5-feature | random-row split [leaky] | 0.96 | 0.93 | 0.90 |

The 5 features are compact band-power/morphology summaries per 1-second chunk:
`[ln variance, ln line-length, ln range, ln mean-abs-deviation, zero-crossing
rate]`. Two honest takeaways:

1. **Feature engineering is the optimization.** Raw amplitude barely beats the
   baseline (balanced-acc 0.53 ≈ chance); the band-power features lift it to
   ~0.92 balanced accuracy out-of-sample.
2. **Leakage didn't help here** (grouped 0.96 ≈ random-row 0.96) because summary
   statistics don't carry the per-sample autocorrelation that inflated the
   eye-state benchmark — so the seizure win is real, not a protocol artefact.

## End-to-end: ship the trained model as a signed `.rvf`

The example's final step persists the trained model through the full stack:
`model_to_container` writes `MODEL` + `META` segments, `sign_container` adds an
Ed25519 `CRYPTO` segment, and the bytes are written to disk. On reload, the
container's CRC32C/content-hash **and** the signature are verified, and the
reloaded model reproduces the held-out accuracy **exactly** (0.9727 == 0.9727)
in an 832-byte file — closing the loop from training to a tamper-evident,
self-describing artifact (ADR-0023).

