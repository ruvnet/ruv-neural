# ADR-0019 — Clinical evidence base for 40 Hz GENUS & claims discipline

## Status

Accepted — defines how the project may describe the science behind its core
mechanism, reinforcing ADR-0001 (wellness, not treatment) and ADR-0011
(acceptance test).

## Context

The project's stimulation thesis rests on **GENUS** (Gamma ENtrainment Using
Sensory stimulation). The evidence is genuinely promising but **contested and
not yet definitive in humans**, and the project must describe it honestly:

- **Foundational science (MIT / Tsai lab):** Iaccarino et al., *Nature* 2016 —
  40 Hz visual flicker entrained gamma and reduced amyloid-β in mice via
  microglia. Murdock et al., *Nature* 2024 — multisensory 40 Hz drove
  **glymphatic** amyloid clearance via VIP-interneuron peptide release (mouse).
- **Clinical program (Cognito Therapeutics, *Spectris*):** OVERTURE (Phase 2,
  NCT03556280) reported reduced atrophy / preserved myelin vs sham; a 2025
  post-hoc "time saved" analysis (*Alzheimer's & Dementia: TRCI*) is favorable
  **but post-hoc / partly open-label**. HOPE (Phase 3 pivotal, NCT05637801,
  670 participants) **completed enrollment June 2025; pivotal readout pending**.
  Regulatory status is **FDA Breakthrough Device Designation — not clearance or
  approval.**
- **Contested / null findings (balance):** failed mouse replications (Yang & Lai
  2023; Soula et al., *Nat. Neurosci.* 2023 — no reliable entrainment beyond
  V1); small human study showed entrainment but **no reduction in cerebral
  amyloid**; reviews (2024–2025) call the field heterogeneous, underpowered, and
  not yet definitively efficacious.

The risk is twofold: under-selling (ignoring real mechanistic work) or
over-selling (implying GENUS is a proven Alzheimer's therapy). Either misleads.

## Decision

Adopt a **claims-discipline** standard for any documentation, UI copy, or
research material the project ships:

1. **No disease/treatment claims.** GENUS is described as a sensory-entrainment
   *mechanism under active investigation*, never as a treatment for Alzheimer's
   or any condition. This is the ADR-0001 boundary applied to language.
2. **Cite primary literature with its evidence grade.** Mouse vs human;
   peer-reviewed vs press release; controlled vs post-hoc/open-label; designation
   vs clearance. Pivotal human efficacy is **pending**, and must be labeled so.
3. **Surface the contested findings**, not only the supportive ones — failed
   replications and null human amyloid results are part of the honest picture.
4. **The acceptance test (ADR-0011) is about *delivery*, not *efficacy*.** The
   system proves it can identify a target state, deliver a *verified* 40 Hz
   stimulus, measure a response, and stop safely — it makes **no claim** that
   doing so treats anything.

## Consequences

- The project can responsibly motivate 40 Hz stimulation without crossing into
  unproven medical claims.
- Copy and docs have a single, citable standard reviewers can check against.
- If HOPE (or independent replication) reads out, this ADR is superseded by one
  that records the new evidence grade — claims never silently drift upward.

## Evidence

- `docs/adr/0001-scope.md` — the wellness/not-treatment boundary this ADR
  enforces in language.
- `docs/adr/0011-acceptance-test.md` — delivery-not-efficacy acceptance gate.
- `ruv-neural-stim` / `ruv-neural-loop` — deliver and *verify* the stimulus
  (ADR-0002, ADR-0004); they make no efficacy claim.

## References

1. Iaccarino et al., "Gamma frequency entrainment attenuates amyloid load," *Nature* 2016 — nature.com/articles/nature20587
2. Murdock et al., "Multisensory gamma stimulation promotes glymphatic clearance of amyloid," *Nature* 2024 — nature.com/articles/s41586-024-07132-6
3. MIT News, "Evidence that 40 Hz gamma stimulation promotes brain health," 2025 — news.mit.edu/2025/evidence-40hz-gamma-stimulation-promotes-brain-health-expanding-0314
4. Cognito Therapeutics HOPE enrollment completion, June 2025 — medicaleconomics.com/view/cognito-therapeutics-completes-enrollment-in-landmark-alzheimer-s-trial
5. "The Mystery of 40 Hz" critical review — pmc.ncbi.nlm.nih.gov/articles/PMC11178681
6. Phase 1 feasibility (entrainment without amyloid reduction), *PLOS One* — journals.plos.org/plosone/article?id=10.1371/journal.pone.0278412
7. *Front. Aging Neurosci.* 2025 review — frontiersin.org/articles/10.3389/fnagi.2025.1710041
