# ADR-0009 — Tamper-evident audit trail

## Status

Accepted

## Context

"With logs and feedback… produce clean evidence." A neuromodulation session must
leave an auditable, tamper-evident record of every decision: what state was
estimated, what was delivered (and verified), and why the loop stopped.

## Decision

`AuditTrail` is an append-only **hash chain**. Each `AuditRecord` stores an
`AuditEvent` plus `prev_hash` and `hash = SHA-256(prev_hash || serialized(event))`.
Editing any earlier record changes its hash and breaks every subsequent link, so
`verify_chain()` detects any post-hoc tampering.

Events capture the full story: `SessionStart`, `Baseline`, `Stimulate` (with the
SHA-256 receipt hashes from ADR-0004), `Hold`, `Complete`, `SafeStop` (with typed
breach reasons), `Abort`. The chain head can be **Ed25519-signed**
(`sign_head → SignedAuditHead`), giving a detached attestation any third party
can verify — mirroring the workspace's existing capability-witness mechanism.

## Consequences

- Every session is reconstructable and independently verifiable.
- Receipt hashes in the trail link decisions to exact delivered waveforms.
- Reuses the existing Ed25519/SHA-256 toolchain; no new crypto dependencies.

## Evidence

- `ruv-neural-loop/src/audit.rs`
- Tests: `audit_chain_verifies_and_detects_tampering`, `signed_audit_head_verifies`,
  `acceptance_session_is_independently_attestable`.
