# ADR-0000 — Architecture Decision Record process & template

## Status

Accepted

## Context

The closed-loop sensory neuromodulation subsystem (`ruv-neural-stim`,
`ruv-neural-biosense`, `ruv-neural-loop`) touches safety, ethics, signal
processing, and control. Decisions need to be **traceable** and **reviewable**,
especially the boundary between a research-grade wellness platform and a medical
device. We record significant decisions as Architecture Decision Records (ADRs).

## Decision

Each ADR is a numbered Markdown file in `docs/adr/` with the sections:

- **Status** — Proposed / Accepted / Superseded by ADR-XXXX / Deprecated
- **Context** — the forces at play and why a decision is needed
- **Decision** — what we decided, in the active voice
- **Consequences** — what becomes easier/harder, and follow-ups
- **Evidence** — code/tests that implement or validate the decision

ADRs are immutable once accepted; revisions are new ADRs that supersede.

## Index

| ADR | Title |
|-----|-------|
| [0001](0001-scope.md) | Scope: research-grade wellness, **not** a medical device |
| [0002](0002-sensory-modalities.md) | Safe external sensory modalities (40 Hz light/audio/haptic) |
| [0003](0003-closed-loop-architecture.md) | Closed-loop control architecture |
| [0004](0004-stimulus-verification.md) | Stimulus verification & delivery receipts |
| [0005](0005-response-sensing.md) | Physiological response sensing |
| [0006](0006-personal-state-embedding.md) | Personal state embedding (ruVector) |
| [0007](0007-safety-envelope.md) | Safety envelope & fail-safe stop |
| [0008](0008-protocol-dosing.md) | Protocol selection & conservative dosing |
| [0009](0009-audit-trail.md) | Tamper-evident audit trail |
| [0010](0010-sensory-safety-limits.md) | Photosensitivity & sensory safety limits |
| [0011](0011-acceptance-test.md) | Acceptance test definition |
| [0012](0012-feedback-smoothing.md) | Feedback smoothing & divergence detection |
| [0014](0014-web-console.md) | rUv Neural UI — web console for Ruflo (Demo + Replay) |

## Consequences

Reviewers can audit the *why* behind the subsystem without reverse-engineering
the code. The ethics/scope boundary (ADR-0001) is explicit and cited from every
crate's top-level docs.
