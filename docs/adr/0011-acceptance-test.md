# ADR-0011 — Acceptance test definition

## Status

Accepted

## Context

The task fixes a single, concrete acceptance criterion:

> Prove the system can **identify a target state**, **deliver a verified
> stimulus**, **measure a response**, and **stop safely** when the response moves
> outside the allowed envelope.

We make this an executable, non-negotiable gate rather than prose.

## Decision

Encode the four clauses as `SessionReport::passes_acceptance()`:

```rust
num_receipts >= 1                 // delivered a stimulus
  && all_receipts_verified        // …and it was verified (ADR-0004)
  && total_steps >= 1             // measured a response over ≥1 loop
  && (goal_reached || safe_stopped) // converged or stopped safely
  && audit_chain_valid            // with intact evidence (ADR-0009)
```

The integration test `ruv-neural-loop/tests/closed_loop_acceptance.rs` asserts
each clause directly, across modalities, and includes a dedicated test that a
mid-session perturbation triggers a fail-safe stop. The CLI `neuromod` command
prints the `PASS/FAIL` verdict and exits non-zero on failure, so CI and operators
share one definition of "working."

## Consequences

- "Done" is mechanically checkable and CI-enforced.
- The criterion spans all the other ADRs (verification, sensing, envelope, audit).
- Both convergence and safe-stop are first-class passing outcomes.

## Evidence

- `ruv-neural-loop/src/outcome.rs` (`passes_acceptance`)
- `ruv-neural-loop/tests/closed_loop_acceptance.rs`
