# ADR-0015 — Signed NeuroSleep qEEG phenotype export

## Status

Proposed

Date: 2026-08-03 · Canonical downstream owner: Helix ADR-051 ·
Pinned base: `caaa14144a70829293737b0ca717ebc818fcc523`

## Context

Constantino and colleagues reported sleep and qEEG changes in APP/PS1 mice,
including NREM loss, altered delta/theta power, reduced theta coherence, a lower
aperiodic exponent, and EEG slowing. This is causal evidence in that mouse model,
not a validated human Alzheimer, amyloid, microglial, treatment-response, or risk
marker. The first integration is therefore local-first research evidence
infrastructure, not diagnosis or intervention selection.

rUv Neural already owns EEG abstractions, spectral/connectivity computation,
sleep-state types, and Ed25519/SHA-256 dependencies. Helix owns longitudinal
health interpretation and user-facing language. Existing `ruflo-evidence/1`
artifacts are intentionally unchanged: they generate a fresh key and embed its
public key, which establishes internal consistency but not enrolled device or
study identity, and their canonical chain does not bind every NeuroSleep field.

## Decision

rUv Neural owns acquisition, bounded parsing, preprocessing, artifact masking,
stage-aware qEEG computation, quality decisions, and signed evidence generation.
Helix independently verifies the released contract using a trusted enrolled key,
stores provenance and compatible scalar time series, abstains when gates fail,
and renders deterministic research-only copy.

The authoritative V1 wire types live in `ruv-neural-core`. The signed envelope is
`SignedNeuroSleepBundleV1` with schema `ruv-neural/neurosleep/1`, a
`NeuroSleepPayloadV1`, its SHA-256 digest, a signer key identifier, and a detached
Ed25519 signature. Payload JSON is canonicalized according to RFC 8785 before
hashing. The signature message is exactly:

```text
ruv-neural/neurosleep/1 \0 signer_key_id \0 payload_sha256
```

The spaces above illustrate concatenation and are not signed. A null byte in a
key identifier is rejected. Every payload field is inside the canonical digest;
the digest is not recursively stored in the payload. Numeric observations are
finite and unit-tagged. Failed or insufficient measurements are typed nulls,
never zeros. Structs reject unknown fields and enums reject unknown variants.

Signing is injected through `NeuroSleepSigner`. The core API never generates a
key for NeuroSleep and never embeds a public key in its bundle. The supplied
`PersistentEd25519Signer` only adapts key bytes already loaded from persistent
custody. Its verifying key must be enrolled separately under the active personal
device, laboratory, or study trust profile. A bundle-supplied key is not a trust
root.

The fixed V1 qEEG registry includes stage-specific absolute/relative delta and
theta power, alpha power, theta peak frequency/power, frontal-parietal full-band
and theta coherence, aperiodic exponent/offset, and fit error, plus nightly stage
duration/bout and artifact/coverage facts. The algorithm manifest binds source
commit, crate versions, extractor/configuration digests, stage source, and DSP
parameters. A compatibility fingerprint binds all acquisition and analytic facts
that affect comparison.

This N1 foundation adds contract and attestation primitives only. EDF/BrainVision
I/O, DSP replacements, staging, CLI analysis, and WASM verification are separate
follow-up changes. Raw EEG never belongs in this derived bundle.

## Safety and interpretation boundary

NeuroSleep output is observational research evidence. It must not diagnose,
screen for, rule out, or estimate risk for Alzheimer disease, cognitive
impairment, amyloid pathology, neuroinflammation, or microglial activation. It
must not recommend medication, supplements, gamma entrainment, or another
intervention. It has no dependency path to `ruv-neural-loop`, stimulation,
protocol selection, an actuator, a hosted model, or a health score.

The existing HR/motion sleep proxy remains appropriate only for contextual or
stimulation gating use and cannot be represented as paper-equivalent expert
NREM/REM/Wake scoring. The existing 40 Hz research loop remains separate; the
mouse study used CSF1R-mediated microglial depletion and did not test gamma
entrainment efficacy.

## Consequences

- One released Rust contract becomes the numeric/evidence source of truth.
- Helix can reject tampering and unknown fields before creating any record.
- Persistent key enrollment and revocation become deployment requirements.
- Coordinated releases and cross-repository golden fixtures are required.
- Conservative typed nulls and downstream abstention reduce early visible data.
- Existing Ruflo evidence and heart-rate/motion proxy behavior remain unchanged.

## Evidence

- `ruv-neural-core/src/neurosleep.rs`
- `ruv-neural-core/src/attestation.rs`
- Golden RFC 8785 payload digest and exact domain-separation tests
- Tamper, wrong-trust-key, unknown-field, null-key-id, and non-finite tests
- `.github/workflows/rust.yml`
- Committed workspace `Cargo.lock`

Downstream integration, policy gates, and the complete acceptance matrix are
owned by Helix ADR-051.
