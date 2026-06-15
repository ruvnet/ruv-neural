# ADR-0016 — Invasive BCI landscape & the non-invasive scope boundary (interop, not parity)

## Status

Accepted — reinforces and extends ADR-0001. The interoperability targets
(NWB / LSL / BIDS) are **Proposed** as a roadmap; the scope boundary is binding now.

## Context

The high-performance frontier of brain-computer interfaces in 2024–2026 is
overwhelmingly **invasive**, and it achieves signal quality that non-invasive
EEG/OPM cannot approach:

- **Implant programs (mostly company/press, not peer-reviewed):** Neuralink
  *PRIME* (≈1024 electrodes on 64 threads, intracortical single-unit + LFP;
  first human implant Jan 2024); Synchron *Stentrode* (endovascular, ~16
  contacts, LFP-class, FDA Breakthrough Device, COMMAND feasibility study met
  1-yr safety); **Precision Neuroscience *Layer 7*** (1024-microelectrode
  thin-film surface array — **FDA 510(k) cleared K242618, 30 Mar 2025**, the
  first such clearance for a next-gen BCI company); Paradromics *Connexus*
  (first-in-human recording May 2025); Blackrock *Utah/NeuroPort* array
  (96–128 penetrating electrodes; underlies most BrainGate work).
- **Peer-reviewed neuroprostheses:** UC Davis speech BCI (*NEJM* 2024 — 97.5%
  accuracy, 125k-word vocabulary, ~32 wpm conversational; intracortical);
  UCSF/Chang ECoG speech-and-avatar (*Nature* 2023 — 78 wpm, 25% WER) and
  streaming brain-to-voice (*Nat. Neurosci.* 2025 — ~1 s latency); Stanford/
  BrainGate handwriting (*Nature* 2021 — 90 char/min, >99% with autocorrect).

These results depend on **single-unit / ECoG bandwidth from surgically placed
electrodes**. They are not a "scaled-up" version of what scalp sensors can do —
they are a different problem class. Building toward them would put this project
squarely inside implanted-device regulation (ADR-0001 forbids exactly this).

## Decision

1. **Acquisition stays strictly non-invasive.** rUv Neural models only scalp/
   external sensing (EEG today; OPM-MEG and NV-diamond as research front-ends,
   ADR-0017 / ADR-0018). No intracortical, ECoG, subdural, or endovascular
   acquisition path is implemented. The `SensorType` enum is deliberately
   restricted to non-invasive and external magnetometry modalities.
2. **Interoperate by data format, never by signal parity.** Where the project
   touches the broader BCI ecosystem it does so through open standards, as a
   **roadmap**:
   - **Archival/offline:** export sessions and embeddings to the existing
     RuVector `.rvf` format and target **NWB (Neurodata Without Borders)**
     compatibility for neurophysiology archives; align dataset layout with
     **BIDS** (incl. the iEEG extension's conventions) for tooling reuse.
   - **Runtime:** consume and produce **Lab Streaming Layer (LSL)** streams so
     rUv Neural can sit in a standard real-time BCI rig as one node.
   - **Terminology:** align with the **IEEE P2731** neurotech glossary.
3. **Frame EEG/OPM honestly** in docs as a distinct, lower-SNR problem class —
   never as a non-invasive route to speech/handwriting decoding.

## Consequences

- The scope boundary that keeps this an open-source, non-device project is now
  explicit against the *specific* invasive systems people will compare it to.
- Choosing format-level interop (NWB/LSL/BIDS) means the project can exchange
  data with invasive-BCI tooling without ever acquiring invasive signals.
- We will not chase speech/motor-decoding benchmarks that are physically
  unreachable from the scalp; cognitive-state *topology* remains the thesis.

## Evidence

- `ruv-neural-core/src/sensor.rs` — `SensorType` enumerates only non-invasive /
  external magnetometry modalities (`Eeg`, `Opm`, `NvDiamond`, `SquidMeg`,
  `AtomInterferometer`); no intracortical/ECoG variant exists.
- `ruv-neural-embed/src/rvf_export.rs` — `.rvf` export, the current interop seam
  that NWB/BIDS export would extend.

## References

1. Precision Neuroscience FDA 510(k) clearance (K242618), 17 Apr 2025 — globenewswire.com/news-release/2025/04/17/3063418
2. Card et al., "An Accurate and Rapidly Calibrating Speech Neuroprosthesis," *NEJM* 2024 — nejm.org/doi/full/10.1056/NEJMoa2314132
3. Metzger et al., "A high-performance neuroprosthesis for speech decoding and avatar control," *Nature* 2023 — nature.com/articles/s41586-023-06443-4
4. "Brain-computer interface restores natural speech" (streaming brain-to-voice), NIH, *Nat. Neurosci.* 2025 — nih.gov/news-events/nih-research-matters/brain-computer-interface-restores-natural-speech-after-paralysis
5. Willett et al., "High-performance brain-to-text communication via handwriting," *Nature* 2021 — nature.com/articles/s41586-021-03506-2
6. Synchron COMMAND study results, 2024 — businesswire.com/news/home/20240930433219
7. Neurodata Without Borders — nwb.org ; Lab Streaming Layer — github.com/sccn/labstreaminglayer
8. IEEE Standards Roadmap on Neurotechnologies (incl. P2731) — brain.ieee.org
