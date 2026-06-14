# ADR-0014 — rUv Neural UI: Web Console for Ruflo Closed-Loop Sensory Neuromodulation

## Status

Accepted — **v1 (Demo + Replay) implemented** in `apps/ruv-neural-ui`. Real mode
and Research mode remain feature-gated / deferred per the phased plan below.

Date: 2026-06-13 · Deciders: rUv ·
Tags: ruv-neural, Ruflo, neuromodulation, web-ui, GitHub Pages, verifier,
simulator, audit, safety, ruVector, WebSerial, WASM

## 1. Context

rUv Neural now includes a complete closed-loop sensory neuromodulation platform
on **safe external sensory channels only**:

- **ruv-neural-stim** — 40 Hz light/audio/haptic stimuli, sensory-safety limits,
  verified delivery receipts (SHA-256 waveform binding, measured entrainment Hz).
- **ruv-neural-biosense** — HRV, respiration, motion, sleep proxy, autonomic
  arousal/relaxation fusion, deterministic simulation.
- **ruv-neural-loop (Ruflo)** — ruVector personal state embeddings, divergence-
  aware safety envelopes, conservative dosing, EMA feedback smoothing, fail-safe
  stop, hash-chained Ed25519-attestable audit trails.

The platform passes the core acceptance test (identify target → deliver verified
stimulus → measure response → stop safely outside the envelope). The missing
layer is a **public, operator-friendly, research-friendly management interface**:
a static web console deployable to GitHub Pages that demonstrates Ruflo without
compiling Rust, attaching hardware, or running a local service.

## 2. Decision

Create **rUv Neural UI**, a static web console for Ruflo, deployable to GitHub
Pages, supporting four modes:

1. **Demo** — deterministic, browser-only; plays back real Rust-generated
   evidence bundles (one per target × perturbation).
2. **Replay** — imports a Ruflo evidence bundle (JSON) and verifies receipts,
   hash chains, signatures, stimulus integrity, controller decisions, safe-stop
   events, and the acceptance result **locally** in the browser.
3. **Real** — explicit, user-gated local hardware/service bridges
   (WebSerial/WebUSB/WebBluetooth/localhost/file import). **Excluded from v1.**
4. **Research** — guided study workflows. Introduced after Demo/Replay stabilise.

**v1 ships Demo + Replay only.** Real mode is feature-gated; Research mode follows.

## 3. Product positioning

rUv Neural UI is **not a treatment interface**. It is a research-grade web console
for demonstrating closed-loop sensory neuromodulation, verifying stimulus
receipts, visualising physiological response and state-embedding drift, showing
safety-envelope decisions, replaying signed sessions, and exporting evidence.

**Approved language:** closed-loop sensory neuromodulation research · cognitive
state experimentation · wellness research · verified sensory dosing · safety
envelope · local-first session replay · not a medical device.

**Disallowed language:** treats Alzheimer's · cures cognitive decline · brain
therapy device · clinical stimulation without medical supervision · targeted
treatment · diagnosis · prescription protocol. *(Enforced by a content-lint test.)*

## 4. Goals / 5. Non-goals

The UI must make Ruflo understandable in five minutes: show the whole loop, let
anyone run a deterministic demo, let reviewers import and verify a signed report,
make safety visible, keep all verification local, and preserve the hard
not-a-medical-device boundary. It must **not** diagnose, recommend treatment,
store health data in a cloud backend, require login for the public demo, control
real hardware by default, hide safety thresholds, or make efficacy claims.

## 6. Architecture

Static TypeScript app: **Vite + React + TypeScript + Zustand + Zod**, Web-Crypto /
`@noble` for SHA-256 + Ed25519 verification, lightweight SVG charts, **Vitest**
unit tests, **Playwright** E2E, GitHub Pages hosting. A future **WASM** package
can reuse the Rust verifier directly.

```
apps/ruv-neural-ui
  src/{schemas, verifier, simulator, store, components, pages, styles}
  public/fixtures/*.json   ← real Rust-generated evidence bundles
  tests/                   ← Playwright E2E
```

The canonical artifact is the **Ruflo evidence bundle** (`ruflo-evidence/1`),
emitted by `ruv-neural neuromod --bundle <path>` and consumed by both modes. Its
per-step hash chain uses fixed-precision canonical strings so the **TypeScript
verifier recomputes the same hashes as Rust** without depending on language-
specific JSON/float formatting.

## 7–8. Modes & screens

Demo (target/seed/perturbation → bundled deterministic session) and Replay
(import → verify) in v1. Screens: Overview, Live Session, Stimulus Verifier,
Biosense, ruVector State, Safety Envelope, Audit Trail, Witness Bundle, Export.

## 9. Data model

The shared `EvidenceBundle` schema (Rust `serde` ⇄ TS `zod`):

```ts
EvidenceBundle = {
  schemaVersion: "ruflo-evidence/1"
  sessionId, createdAt, mode, targetState, protocol
  steps: EvidenceStep[]          // phase, distance, intensity, embedding,
                                 // biosense, receipts, envelope, hash chain
  acceptance: AcceptanceResult   // 4 clauses + passed
  report: SessionReport          // Rust SessionReport
  bundleChainHead, auditHeadHash, auditChainValid
  signature?: { headHash, signature, publicKey }  // Ed25519 over bundleChainHead
}
```

## 10. Verification logic

The verifier evaluates, locally and on untrusted input:

1. Every receipt carries a waveform SHA-256.
2. Every `verified` receipt has measured frequency within tolerance
   (`ENVELOPE_TOLERANCE_HZ = 2.0`, matching the Rust receipt rule; the ADR's
   draft 0.1 Hz reflected an idealised synthesiser, not the crossing estimator).
3. Every step has biosense metrics and a ruVector state.
4. Every safe-stop sets intensity to zero.
5. Every hash-chain event links to the previous (recomputed in-browser).
6. Signed bundles verify against the included public key (Ed25519).
7. Acceptance passes only if all four core clauses hold.

## 11–13. Safety, privacy, security

Safety is a first-class state: visible boundary statement, intensity-ceiling and
photosensitivity cautions, fail-safe-stop reasons, no hidden unsafe override in
public mode, no auto-restart after stop. **Local-first:** public demo stores no
personal data; imported reports stay in browser memory unless downloaded; no
cloud backend in v1. **Security:** all imported JSON validated with Zod and
treated as untrusted data (never executed), strict CSP, Web-Crypto/`@noble`
digests, schema version pinned, immutable demo bundles per release.

## 14–15. Build, deployment & phased plan

GitHub Pages via `npm run build`; CI does type-check, unit tests, verifier tests,
fixture replay, Playwright demo flow, static build, Pages publish.

- **Phase 1 (this ADR):** static Demo + Replay, simulator, live visualisation,
  safe-stop demo, JSON import/export, schema validation, receipt + audit-chain
  verifier, Pages deployment.
- **Phase 2:** witness/evidence UX, CLI report compatibility.
- **Phase 3:** WASM verifier with Rust↔TS parity tests.
- **Phase 4:** Real-mode bridges (gated, emergency stop, hardware validation).
- **Phase 5:** Research workflow (consent, baseline, survey, evidence export).

## 16. Acceptance test

A visitor can open the Pages URL, run a deterministic Ruflo demo, observe target
identification, verified 40 Hz delivery, measured response, a perturbation-driven
safe stop, hash-chained/signed evidence export, report reload, and **local replay
verification with no backend** — asserted by Vitest (verifier/simulator) and
Playwright (demo flow).

## 18. Alternatives considered

CLI-only (weak for public/non-technical validation), backend dashboard (premature
privacy/compliance burden), native desktop (deferred to real-hardware era), and
real-hardware-first (rejected — prove the loop and verifier before device
control). Static, local-first GitHub Pages won for distribution and transparency.

## 19. Consequences

Ruflo becomes understandable without compiling Rust; a public proof surface
improves credibility; researchers can replay evidence; verification becomes
central to the product. Costs: more surface to maintain, browser-crypto
compatibility, careful UI-language governance — mitigated by shipping Demo+Replay
first, gating Real mode, and content-linting medical claims.

## 23. Final decision

Create rUv Neural UI as a static, local-first, GitHub-Pages-deployable web console
for Ruflo. v1 focuses on deterministic demo, replay verification, safety
visualisation, audit-chain verification, and evidence export. Real hardware
control is added only after the verifier, safety envelope, and boundary language
are stable. **The UI is the public proof surface for Ruflo:** verified stimulus,
measured response, conservative control, safe stop, replayable evidence.
