# ADR-0018 — NV-diamond quantum magnetometry: aspirational front-end with an honest MEG gap

## Status

Proposed — experimental/aspirational front-end, kept behind a feature flag and
clearly labeled. The README's "NV reality check" is hereby formalized as a
decision.

## Context

Nitrogen-vacancy (NV) ensembles in diamond promise a **room-temperature,
mm-scale, vector** magnetometer — attractive because it could, in principle, put
magnetometry on a chip instead of in a shielded room. The honest 2024–2026
state of the art:

- **Best biomagnetic sensitivity (demonstrated):** ~**9.4 pT/√Hz** over
  5–100 Hz (Tokyo Tech, 2024). Photon-shot-noise-limited demos reach
  **~460–670 fT/√Hz** but with large bias fields / small bandwidth.
- **Biological demonstrations:** rat magnetocardiography (~20 pT R-wave);
  **human MCG** with heavy averaging and sub-mm³ sensing (early 2026);
  magnetomyography/neurography remains phantom + simulation stage.
- **The MEG gap:** MEG signals are **~fT to low-pT**. Current NV is ~pT/√Hz —
  roughly **1–3 orders of magnitude short** of single-trial MEG. **MEG-scale NV
  is a projection, not a result.** Active labs/companies: Tokyo Tech, Stuttgart,
  MIT/Harvard lineage, **Fraunhofer IAF** (integrated vector magnetometer
  prototype, 2025), Element Six (diamond), QDTI.
- **Buildability:** a *crude* NV magnetometer (µT–nT) is genuinely hobbyist-
  buildable; **bio-grade (pT) is a serious lab build; fT MEG-grade is not
  routinely buildable by anyone.**

## Decision

Treat NV-diamond as an **experimental research front-end**, never as a delivered
sensing capability:

1. Keep `SensorType::NvDiamond` and the NV driver, but gate any NV acquisition
   path behind an explicit experimental flag, and keep the **simulator** as the
   default so the pipeline is exercisable without hardware.
2. **State the ~1000× MEG gap in the open**, both in code docs and the README:
   NV today reaches MCG-scale (pT) biomagnetism with heavy averaging; MEG-scale
   neural fields (fT) are out of reach for the foreseeable roadmap.
3. The honest path for NV in this project is **education and method validation**
   (ODMR, calibration, demodulation) and, at most, **MCG-class** biomagnetism —
   not MEG. EEG (today) and OPM-MEG (ADR-0017) are the real connectivity paths.
4. No NV-based neural/clinical claim is made; the BOM and reality check in the
   README remain the canonical, non-marketing description.

## Consequences

- The project keeps a credible, inspiring quantum-sensing component without
  misrepresenting its physics.
- Contributors interested in NV have a sandbox (driver + simulator + ODMR math)
  that cannot be mistaken for a working brain magnetometer.
- If NV sensitivity closes the gap in future, the modality-agnostic pipeline is
  already wired to accept it.

## Evidence

- `ruv-neural-sensor/src/nv_diamond.rs` — NV driver, ODMR/gyromagnetic constants
  (`GAMMA_NV_GHZ_PER_T = 28.024`), and configuration; paired with
  `simulator.rs` as the default exercisable path.
- `ruv-neural-sensor/src/calibration.rs` — per-channel calibration the NV path
  depends on.
- `README.md` — "NV-diamond magnetometry is research-grade hardware" reality
  check and BOM, now formalized by this ADR.

## References

1. Sekiguchi et al., NV biomagnetic sensitivity <10 pT/√Hz, Tokyo Tech 2024 (arXiv 2309.04093) — titech.ac.jp/english/news/2024/069398
2. NV dynamic-range/sensitivity demo (~670 fT/√Hz, 280 µT range) — quantumzeitgeist.com/280-670-diamond-center-magnetometry
3. NV flux-concentrator ~460 fT/√Hz, arXiv 2411.10437 — arxiv.org/html/2411.10437v2
4. NV rat magnetocardiography, *Phys. Rev. Applied* 21, 064028 (2024) — link.aps.org/doi/10.1103/PhysRevApplied.21.064028
5. Human cardiac measurements with diamond magnetometers, arXiv 2601.18843 (2026) — arxiv.org/abs/2601.18843
6. Fraunhofer IAF integrated NV vector magnetometer prototype, 2025 — eurekalert.org/news-releases/1087820
7. NV magnetomyography/neurography (phantom/simulation), *Front. Neurosci.* 2022 — frontiersin.org/articles/10.3389/fnins.2022.1034391
