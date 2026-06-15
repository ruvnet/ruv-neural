# ADR-0017 — Wearable OPM-MEG front-end (driver abstraction, external shielding assumed)

## Status

Proposed — research front-end. EEG remains the validated path (ADR-0001); OPM-MEG
is the realistic next high-fidelity modality and is modeled as a sensor driver,
not yet an owned/validated acquisition stack.

## Context

Optically pumped magnetometers (OPMs) have turned magnetoencephalography from a
cryogenic, fixed-helmet instrument into a **wearable, room-temperature** one,
and the technology is now commercially mature (research/clinical-research grade):

- **Sensitivity (demonstrated):** QuSpin QZFM Gen-3 dual-axis **<15 fT/√Hz**
  (typ. 7–10) over 3–100 Hz; triaxial **<23 fT/√Hz**. Comparable to cryogenic
  SQUID (~2–5 fT/√Hz), and because sensors sit **on the scalp** the measured
  signal is larger.
- **Systems & vendors:** QuSpin (sensor OEM), **Cerca Magnetics** (full helmet
  systems), FieldLine. Typical research systems run ~48 OPMs → ~118 channels;
  triaxial ~168 channels; a 384-channel triaxial system is a 2025 research
  prototype, not a product.
- **Deployments 2023–2026:** SQUID-task replication; mesial-temporal and
  pediatric/ictal epilepsy detection (incl. an infant); connectome test–retest.
  Still research, not routine clinical.
- **The hard constraint:** OPMs have a tiny dynamic range (**±5 nT**), so they
  still require a **magnetically shielded room** (fields <1 nT, gradients
  <1 nT/m over the movement volume) plus active field nulling. "Lighter" MSRs
  exist; **MSR-free OPM-MEG is not yet practical.** Movement is tolerated only
  *within* the nulled field.

OPM-MEG offers far better spatial resolution than EEG (no volume-conduction
smearing) — directly valuable to a connectivity/min-cut topology pipeline — but
it is a five-figure-per-sensor instrument that this project will never own.

## Decision

Model OPM-MEG as a **`SensorSource` driver/abstraction**, not as hardware the
project provides or validates:

1. Keep `SensorType::Opm` as a first-class modality behind the same
   `SensorSource` interface as EEG, so the DSP → graph → min-cut → decode
   pipeline runs **modality-agnostically** against OPM samples (units: fT, not
   µV) once a device is attached.
2. **Assume external shielding and field nulling.** The framework does not model
   MSR control; it consumes already-nulled, in-range samples and surfaces
   dynamic-range/saturation flags via the existing signal-quality layer.
3. **Validate on EEG and the simulator now.** OPM remains a simulated/driver
   path until real OPM data is available; no clinical/quantitative OPM claim is
   made until then.
4. Document the **±5 nT range and MSR requirement** as the gating realities, so
   no reader mistakes "wearable MEG" for "shielding-free MEG."

## Consequences

- The pipeline gains a credible high-resolution upgrade path without the project
  pretending to own magnetometry hardware.
- Modality-agnostic acquisition means OPM "just works" through the connectivity
  stack the day a real device + MSR are available.
- We avoid over-claiming: OPM-MEG here is a driver target and a simulator, not a
  validated clinical capability.

## Evidence

- `ruv-neural-sensor/src/opm.rs` — OPM array driver behind the shared
  `SensorSource` interface.
- `ruv-neural-sensor/src/quality.rs` — signal-quality / saturation flags that
  surface OPM dynamic-range limits.
- `ruv-neural-core/src/sensor.rs` — `SensorType::Opm` as a first-class modality.

## References

1. QuSpin QZFM Gen-3 specifications — quspin.com/products-qzfm
2. Cerca Magnetics (wearable OPM-MEG helmet systems), 2024–25 — sci-tech-today.com/news/cerca-magnetics-secures-4-3m
3. Boto et al. / Brookes et al., OPM-MEG vs SQUID replication, *Sci. Rep.* 2024 — nature.com/articles/s41598-024-56878-6
4. OPM-MEG epilepsy detection (incl. pediatric/infant), medRxiv 2023 — medrxiv.org/content/10.1101/2023.10.03.23296442
5. Shielding requirements for movement in OPM-MEG, *NeuroImage* — ncbi.nlm.nih.gov/pmc/articles/PMC9248349
6. 384-channel triaxial OPM-MEG prototype, arXiv 2509.03107 (2025) — ncbi.nlm.nih.gov/pmc/articles/PMC12723407
7. "On-scalp MEG" review, *Trends in Neurosciences* 2022 — cell.com/trends/neurosciences/fulltext/S0166-2236(22)00102-3
