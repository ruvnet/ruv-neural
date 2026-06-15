# ADR-0021 — Privacy-preserving, on-device & federated personalization

## Status

Proposed — design principles for any future personalization/learning path. The
data-minimization stance is binding now; federated/DP mechanisms are a roadmap.

## Context

Personalization ("right person," ADR-0006) and any cross-user learning collide
with a hard fact: **neural and physiological signals are biometric identifiers,
not anonymous data.**

- **Re-identification risk:** functional connectomes act as **"brain
  fingerprints"** that identify individuals across a population, in health and
  disease. "Anonymized EEG" is a contestable claim. (Fraunhofer's NEMO project
  exists specifically to build utility-preserving EEG anonymization.)
- **Federated learning (FL) is the credible cross-user pattern** (2023–2026):
  cross-subject FL for EEG (e.g. *mixEEG*, CogSci 2025, shares only averaged
  unlabeled data); cross-device FL for wearable ECG anomaly detection
  (AUC ≈ 0.82 without centralizing data); personalization-via-distillation
  (FedL2T for seizure prediction) is favored because inter-subject distribution
  shift defeats a single global model.
- **Differential privacy (DP)** (DP-SGD, adaptive clipping; SMPC+DP defense in
  depth) protects shared updates, but the **privacy–utility tradeoff is acute
  for the small datasets** typical of neuro/wellness work; **DP synthetic data**
  is emerging for sharing/benchmarking rather than DP on a tiny live loop.

The project already minimizes by design: it stores compact metrics and
embeddings (HRV/respiration/motion → ruVector → `NeuralEmbedding`), not raw
traces, and runs the controller locally/deterministically.

## Decision

Adopt **privacy-by-architecture**, in this order of preference:

1. **On-device feature extraction & local-first by default.** Raw biosignals are
   processed to compact metrics/embeddings on the edge (ESP32/WASM/local);
   **raw traces are not exported by default.** This is the strongest privacy
   control and is already the grain of the design.
2. **Data minimization & short retention.** Persist features/embeddings (RVF),
   not raw signal; treat any stored neural data as a biometric identifier with
   correspondingly short retention and explicit consent (ties to ADR-0014's
   consent-gated Research workflow).
3. **Federation before centralization.** If cross-user learning is ever added,
   use **federated learning with personalization layers** (cross-silo for
   institutions, cross-device for consumers) — never a raw-data lake. Expect
   per-person adaptation, not one global model.
4. **Differential privacy as a governed parameter.** Treat the DP budget (ε) as
   an explicit, recorded governance setting on any update-sharing path; prefer
   **DP synthetic data** for benchmarking/sharing over DP on the live small-N
   loop. Combine with secure aggregation where feasible.
5. **No "anonymized neural data" claim** is made without a documented method and
   its threat model — re-identification is assumed possible by default.

## Consequences

- The project's existing local, minimal design is reframed as a privacy
  guarantee, not just an engineering choice.
- A future learning path has guardrails (FL + personalization + DP budget)
  before any code is written, so privacy is not retrofitted.
- We avoid the common, false comfort of "anonymized EEG."

## Evidence

- `ruv-neural-biosense/src/{hrv,respiration,motion,physio}.rs` — on-device
  feature extraction; the controller consumes metrics, not raw traces.
- `ruv-neural-loop/src/embedding.rs`, `ruv-neural-embed/src/rvf_export.rs` —
  compact `NeuralEmbedding`/RVF persistence (features, not raw signal).
- `docs/adr/0006-personal-state-embedding.md`, `docs/adr/0014-web-console.md` —
  ruVector personalization and the consent-gated, local-first Research workflow.

## References

1. "mixEEG: cross-subject federated EEG," CogSci 2025 — arxiv.org/abs/2504.07987
2. Federated unsupervised anomaly detection on wearable ECG — pmc.ncbi.nlm.nih.gov/articles/PMC11975069
3. FedL2T (two-teacher distillation for seizure prediction) — arxiv.org/pdf/2510.08984
4. Functional connectomes as "brain fingerprints" (re-identification) — biorxiv.org/content/10.1101/2023.03.11.532201
5. Fraunhofer NEMO — biosignal anonymization — idmt.fraunhofer.de/en/Press_and_Media/press_releases/2023/data-protection-of-biosignals.html
6. SMPC+DP defense-in-depth for healthcare FL — arxiv.org/pdf/2410.02462
7. DP synthetic EHR time series — academic.oup.com/jamia/article/31/11/2529/7747780
