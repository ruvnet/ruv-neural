# ADR-0020 — Adaptive/closed-loop *therapeutic* neuromodulation is explicitly out of scope

## Status

Accepted — concretizes the ADR-0001 boundary against the specific implanted,
closed-loop *treatment* devices that share architectural vocabulary with this
project.

## Context

This project implements a **closed-loop controller** (ADR-0003) and could be
mistaken for — or pressured toward — the regulated medical devices that use the
same words ("closed-loop," "adaptive," "biomarker-driven"). The 2024–2026 SOTA
in *therapeutic* closed-loop neuromodulation is firmly in implanted Class III
device territory:

- **Adaptive DBS (Medtronic *BrainSense* / Percept):** **FDA-approved 24 Feb
  2025** — first adaptive DBS for Parkinson's. Senses beta-band local field
  potentials and **auto-titrates intracranial stimulation**. Pivotal evidence:
  ADAPT-PD (*JAMA Neurology* 2025) — improved On-time without troublesome
  dyskinesia; ~98% of patients chose to continue aDBS.
- **Responsive neurostimulation (NeuroPace *RNS*):** **FDA PMA approved 2014** —
  detects electrographic seizure onset and delivers responsive intracranial
  stimulation. 9-year data (*Neurology* 2020): median 75% seizure reduction.

These are **implanted, surgical, intracranial** systems making **treatment
claims for diagnosed disease**, driven by closed-loop control on neural
biomarkers — the exact combination (implant + treatment claim) that triggers
FDA Class III / PMA regulation.

## Decision

1. **No therapeutic closed-loop modality is modeled.** rUv Neural's controller
   commands **only** the safe external sensory modalities of ADR-0002
   (40 Hz light / audio / haptic). There is no API — and there will be none —
   for transcranial (TMS/tDCS/tACS), vagus-nerve, or implanted stimulation.
2. **Architectural lessons are welcome; the device line is not.** We may *learn*
   from aDBS/RNS control design (biomarker-gated actuation, fail-safe behavior,
   evidence logging) — and indeed the safety envelope (ADR-0007), conservative
   dosing (ADR-0008), and audit trail (ADR-0009) echo good closed-loop practice —
   **without** adopting their modality, claims, or invasiveness.
3. **The boundary is encoded, not just stated.** The stim crate's `Modality`
   admits exactly `{Light, Audio, Haptic}`; the controller can drive nothing
   else. Any PR adding a non-sensory actuator must be rejected as out of scope.
4. **Wellness framing is non-negotiable.** No diagnosis, no treatment claim, no
   implant — keeping the project on the non-device side of the line that even
   Cognito's *non-invasive therapeutic* program (ADR-0019) only crosses via a
   Breakthrough designation plus a pivotal trial.

## Consequences

- The most regulation-attracting feature anyone might request — "make it treat
  X by stimulating the brain directly" — has a documented, principled refusal.
- The project can borrow control-engineering rigor from medical closed-loop
  systems while staying a research-grade wellness platform.
- Reviewers have a clear test for scope creep: any actuator beyond external
  senses is out.

## Evidence

- `ruv-neural-stim/src/params.rs` — `Modality` is exactly `{Light, Audio,
  Haptic}` (per ADR-0001/0002); no transcranial or implanted variant.
- `ruv-neural-loop/src/controller.rs` — controller emits `VerifiedStimulus` to
  the stim crate only; `envelope.rs` enforces fail-safe stop (ADR-0007).
- `docs/adr/0001-scope.md`, `docs/adr/0007-safety-envelope.md`,
  `docs/adr/0009-audit-trail.md` — the wellness/safety/evidence stance this ADR
  defends against therapeutic creep.

## References

1. Medtronic adaptive DBS (BrainSense) FDA approval, 24 Feb 2025 — news.medtronic.com/2025-02-24-Medtronic-earns-U-S-FDA-approval
2. NeurologyLive, FDA approves Medtronic adaptive DBS for Parkinson's — neurologylive.com/view/fda-approves-medtronic-adaptive-deep-brain-stimulation-parkinson-disease
3. ADAPT-PD chronic aDBS pivotal data, *Neurology* / *JAMA Neurology* 2025 — neurology.org/doi/10.1212/WNL.0000000000204762
4. NeuroPace RNS FDA PMA approval (2014) — epilepsy.com/stories/fda-approves-responsive-neurostimulation-therapy-neuropace
5. Nine-year RNS outcomes, *Neurology* 2020 — neurology.org/doi/10.1212/WNL.0000000000010154
