# Research Report — Brain-AI State of the Art, 2025–2026

> **Status:** Deep-research sweep — agent-swarm workflow output. **Two of the
> §9 "adopt now" actions are implemented** (Rust + WASM + TypeScript, tested,
> validated, and benchmarked): DFA/Hurst LRTC features (§9 row 2 —
> `ruv-neural-signal/src/lrtc.rs`, `ruv-neural-wasm::compute_dfa`,
> `apps/ruv-neural-ui/src/lrtc/dfa.ts`) and the decode-armed/decode-locked
> mental-privacy gate (§9 row 6 — `ruv-neural-core/src/gate.rs`,
> `ruv-neural-wasm::WasmDecodeGate`, `apps/ruv-neural-ui/src/safety/decodeGate.ts`).
> **Date:** 2026-08-08 · **Author:** rUv (ruv@ruv.net)
> **Method:** 7 parallel web-research agents (one per SOTA dimension, ≥5–8 sourced
> queries each) + 1 synthesis agent, orchestrated as a deterministic agent-swarm
> workflow (8 agents, ~437k research tokens, 124 tool calls). The continuous
> version of this loop — scaffold → evolve → verify/promote → remember → repeat —
> is specified in [§10 Flywheel playbook](#10-flywheel-playbook-metaharness--darwin--flywheel--agenticow)
> using the npm `metaharness` family and related flywheels.
> **Scope:** Latest (2025–2026) SOTA across brain-to-text, neural foundation
> models, invasive BCI, non-invasive sensing, neuromorphic/SNN, whole-brain
> simulation & brain-inspired algorithms — plus the ruvnet npm flywheel tooling —
> mapped onto the `ruv-neural` workspace (crates + ADRs 0001–0024).

---

## TL;DR

| Question | Answer |
|---|---|
| Where did brain-AI cross a threshold in 2025–2026? | **Invasive speech BCI became a product-in-waiting.** ~2 years / 3,800+ hours of *independent home use* at 56 wpm, ~99% word accuracy on a 125k-word vocabulary (UC Davis/BrainGate2, *Nature Medicine* 2026); open Brain-to-Text Benchmark WER down to **5.77%**; streaming brain-to-voice at **80-ms increments** with sub-second latency. |
| What is the winning decode architecture? | **Streaming phoneme/diphone decoder + LLM rescoring**, with **cross-subject pretraining + minutes-scale per-user calibration** (>50% relative WER reduction from multi-user pretraining). Uniform across labs. |
| Where does non-invasive stand? | Far behind: MEG ~29% CER vs **EEG ~65% CER** (Brain2Qwerty v1, *Nat. Neurosci.* 2026); v2 hits 61% word accuracy but needs a room-sized MEG. For EEG-class hardware the honest tier is **keyword/intent decoding**, not open vocabulary. |
| Did EEG foundation models win? | **No — a replicated negative-results wave hit in mid-2026**: dataset-identity leakage (frozen REVE decodes *which dataset* at 100% AUROC but diagnosis at 52.8%), random-init sometimes beating pretrained weights, no FM recovers DFA/Hurst long-range correlations. DIVER-1's scaling laws put parameter count last. **Compact, transparent, honestly-benchmarked models are gaining ground** — vindicating ADR-0016's default. |
| What's the biggest hardware shift? | **Compress/embed at the sensor is now consensus** (Neuralink on-implant spike detection; Paradromics 100 Mbit/s bandwidth wall; FEMBA 2-bit FM on a RISC-V MCU) — and the bottleneck across sensing modalities moved **from physics to real-time edge software**, the layer a deterministic Rust DSP workspace can own. |
| What should ruv-neural do first? | (1) Add **DFA/Hurst features** to `ruv-neural-embed` (cheap, montage-invariant, transfers where FMs collapse). (2) Codify the published **negative controls as promotion gates** for any FM backend (ADR-0016 pt 4). (3) Architect brain2text as **streaming 80-ms phoneme/keyword decoding + off-edge rescoring**. (4) Consider a **`no_std` Rust SNN runtime with NIR import** — genuinely unclaimed territory. |
| What is the metaharness flywheel and does it apply? | The ruvnet npm stack consolidated (mid-2026) into a verifiable self-improvement loop: `metaharness` scaffolds, `@metaharness/darwin` evolves the harness around frozen models, `@metaharness/flywheel` gates + Ed25519-signs promotions, `agenticow`/RVF branch memory in ~0.5 ms. Its **auditable-promotion pattern is directly transferable** to evolving ruv-neural's stimulation/staging policies under a safety case (ADR-0007/0009/0011). |

---

## 1. Brain-to-text and speech neuroprostheses

Invasive speech BCIs crossed the practicality threshold. The field's decode
architecture is now uniform, and the biggest 2026 shift is cross-subject
pretraining.

| Finding | Date | Headline |
|---|---|---|
| ~2 years independent home use of a speech BCI — UC Davis/BrainGate2, *Nature Medicine* | 2026-06 | 3,800+ h, 183k+ sentences, **56 wpm at ~99% word accuracy** on a 125k-word vocabulary; month-over-month decoder stability demonstrated |
| Multi-user pretraining halves WER — BrainGate2, bioRxiv 10.64898/2026.07.23.739430 | 2026-07 | Transformer trained across 6 intracortical participants: **>50% relative WER reduction** for every participant vs per-subject models |
| Inner-speech decoding with a thought-password — Stanford, *Cell* | 2025-08 | 74% accuracy on 125k words from *imagined* sentences; decoding stays **locked until an imagined password** is detected (>98% accuracy) |
| Streaming brain-to-voice — UCSF/Berkeley, *Nat. Neurosci.* | 2025-03 | RNN-transducer over ECoG in **80-ms increments**, sub-second latency vs ~8 s utterance-level systems |
| Instantaneous closed-loop voice with prosody — UC Davis, *Nature* 644:145 | 2025-06 | ~10-ms-scale synthesis with closed-loop audio feedback; **code + data open** (GitHub/Dryad) |
| DCoND diphone decoding + LLM rescoring — *J. Neural Eng.* | 2025-10 | Brain-to-Text Benchmark '24 WER **9.7% → 5.77%**; all top entrants used decoder ensembling merged by a fine-tuned LLM |
| Brain2Qwerty v1 in *Nat. Neurosci.* (arXiv:2502.17480) | 2026-06 | Non-invasive typing decoding: **MEG ~29% CER vs EEG ~65% CER** — the honest MEG/EEG gap quantified; code open |
| Brain2Qwerty v2 (Meta) | 2026-06 | Keystroke-free sentence generation from continuous MEG at 10× data: **61% mean / 78% best word accuracy**, with *predictable scaling laws* in training data |
| Imagined-handwriting Chinese decoding — Zhejiang U., *Adv. Science* | 2025 | Trajectory regression with DILATE time-warp loss; 76.6% character classification; BCI Award finalist |
| fMRI semantic decoding without per-subject training — Tang & Huth, *Curr. Biol.* | 2025-02 | Cross-subject functional alignment removes per-subject linguistic training entirely |
| LibriBrain / PNPL competition | 2025-06 | Largest within-subject MEG speech dataset (~50 h); non-invasive **keyword spotting** demonstrated — the realistic task tier for low-SNR signals |

**Takeaways for ruv-neural**

- The winning decomposition — on-edge streaming phoneme/keyword decoder + LLM
  rescoring — maps cleanly onto a Rust/ESP32/WASM edge decoder with off-edge (or
  quantized) rescoring. Target the 80-ms frame-hop, incremental-state pattern in
  `ruv-neural-brain2text` / `ruv-neural-decoder`.
- Design the embedding layer as **pretrained shared latent + minutes-scale
  per-user calibration** (`ruv-neural-embed`, ADR-0006) — but keep per-subject
  calibration mandatory: leave-one-subject-out *imagined speech* remains at chance.
- Scope the EEG tier honestly: keyword/intent decoding validated against
  LibriBrain, with Brain2Qwerty's ~65% EEG CER as the open-vocabulary calibration
  point.
- Adopt the Stanford **thought-password** as a decode-armed/decode-locked gating
  state machine in the safety envelope (ADR-0007) and privacy ADRs (0022/0023) —
  it is the first published, quantified mental-privacy control.
- Open assets to validate Rust reimplementations against: Brain-to-Text
  Benchmark '24/'25, UC Davis brain-to-voice code+data, `facebookresearch/brain2qwerty`,
  LibriBrain.

## 2. Neural foundation models (delta since ADR-0016)

Scale leadership changed hands, but the strongest 2026 story is a **replicated
wave of negative results** — which vindicates rather than overturns ADR-0016's
compact-model default.

| Finding | Date | Headline |
|---|---|---|
| DIVER-1 — new scale leader + first systematic scaling laws (arXiv:2512.19097) | 2025-12 | 54k h EEG + 5.3k h iEEG, 17.7k subjects, up to 1.82B params; **data-constrained scaling: smaller models trained longer beat larger models**; parameter count subordinate |
| Stress-test with negative controls (arXiv:2607.24519) | 2026-07 | Frozen REVE separates *datasets* at 100% AUROC but decodes dementia diagnosis at 52.8%; **random-init REVE (65.9%) beats pretrained (57.0%)** on CAUEEG; classical features +20 pp |
| FMs blind to long-range temporal correlations (arXiv:2607.24834) | 2026-07 | No FM recovers the alpha-envelope DFA exponent (R²≈0 vs 0.32–0.38 classical); dimensionless **DFA transfers cross-population (AUROC 0.58–0.74) where frozen REVE collapses below chance** |
| FEMBA on the edge (arXiv:2603.26716) | 2026-03 | 2-bit **QAT** Mamba EEG FM real-time on GAP9 RISC-V MCU: ~2 MB, 1.70 s per 5 s window, 27× fewer FLOPs; **PTQ loses ~30% — QAT is mandatory** |
| POSSM — spike tokenizer + SSM core (NeurIPS 2025) | 2025 | Causal 20–50 ms chunks, **9× faster than SOTA transformers** at parity, monkey→human transfer; sits with POYO+/UniBCI in the spike-FM line |
| Benchmark proliferation, convergent negative verdicts | 2025–2026 | AdaBrain-Bench, EEG-Bench, NeuroAtlas, OmniEEG-Bench; a **16.6K-param EEGNet beats 5.8M LaBraM** on speech decoding; LOSO imagined speech at chance for all models |
| Multi-teacher cross-modal distillation (arXiv:2603.04478) | 2026-03 | **4M-param student** (DINOv3 + Chronos teachers → CBraMod) beats SSL baseline on 10/12 datasets with 25% of the pretraining data |
| REVE weights released (`brain-bzh/reve-base`) | 2025-10→ | Most practical open EEG FM for the ADR-0016 `fm` backend — but must clear the negative-control gate first |
| Sleep-only pretraining transfers beyond sleep (arXiv:2605.02500); Stanford Sleep Bench | 2025-12→2026-05 | PSG contrastive pretraining improves **eight non-sleep EEG/ECG tasks**; standardized PSG-pretraining evaluation now exists |
| MEG-GPT (arXiv:2510.18080) | 2025-10 | First dedicated MEG FM; lossless tokenizer generates realistic **synthetic MEG** — usable as DSP test fixtures |
| LUNA/PanLUNA latent-query FMs (arXiv:2510.22257) | 2025–2026 | Fixed-size latent from arbitrary montages, **linear channel scaling**; LUNA-Base (~7M) is ONNX/WASM-feasible |

**Takeaways for ruv-neural**

- **Cheap win, adopt now:** implement DFA/Hurst long-range-correlation features
  in `ruv-neural-embed`/`ruv-neural-signal` — edge-computable, montage- and
  amplitude-invariant, and they transfer where 100M+-param FMs fail.
- **Codify the published negative controls as promotion gates** (random-init
  comparator, dataset-identity probe, subject-disjoint splits) before any FM
  backend clears ADR-0016 point 4.
- If piloting an FM backend, prefer a **compact distilled or LUNA-Base-class
  (~4–7M) model** behind the opt-in ONNX `FoundationEmbedder`; REVE is the most
  practical open candidate but carries the documented failure modes.
- Edge constraint: any MCU-class deployment must budget for **QAT, not PTQ**,
  with SSM-aware activation handling; FEMBA's ~2 MB / real-time-on-GAP9 numbers
  are the benchmark to publish against (`ruv-neural-esp32`/`-wasm`).
- For NeuroSleep: a compact **sleep-PSG-pretrained encoder** could serve both
  qEEG staging and general embedding duty; benchmark against Stanford Sleep Bench.
- For closed-loop paths: the spike-FM line converged on **event tokenizers +
  cheap recurrent/SSM cores** — causal, fixed-state, millisecond-latency — a far
  better fit for Rust/ESP32 than full-attention FMs.

## 3. Invasive BCI hardware and clinical programs

The field entered the trial-to-product transition in 2025–2026.

| Program | Status (2025–2026) | Key numbers |
|---|---|---|
| **Neuralink** PRIME/Telepathy | 21 participants (Jan 2026), ~26 by mid-2026, 4 countries; VOICE speech trial (FDA Breakthrough) implanting; Blindsight first humans expected 2026; $650M Series E at $9B pre-money | N1: 1,024 electrodes / 64 threads; next-gen ~3× channels late 2026; 0 serious adverse device events reported |
| **Synchron** | COMMAND EFS **completed Nov 2025, primary safety endpoint met** — first FDA-IDE BCI feasibility study; $200M Series D; 2026 pivotal toward **first BCI PMA**; Apple **BCI-HID** native input; NVIDIA "Chiral" brain foundation model | Stentrode: 16-electrode endovascular array |
| **Precision Neuroscience** | Layer 7 got the **first FDA 510(k) for a next-gen BCI electrode component** (Apr 2025, ≤30-day implants); 37 patients tested | 1,024 electrodes in ~1.6 cm²; human record 4,096 simultaneous cortical electrodes |
| **Paradromics** | Connect-One IDE (Nov 2025); **first-in-human Connexus implant** (U. Michigan, Jun 2026) | 421 microwires/module, up to 1,684 channels; **100 Mbit/s transcutaneous IR link**; preclinical ITR >200 bits/s at 56 ms |
| **ONWARD** ARC-BCI | 7 brain-to-spine implants (Jan 2026) — clinical decode-then-stimulate closed loop, FDA Breakthrough | WIMAGINE ECoG cortical implant → targeted epidural spinal stimulation |
| **BrainGate/Blackrock** longevity | medRxiv 2025: 14 participants, 20 Utah arrays, 2,319 sessions | Spikes on 35.6% of electrodes; **~7% decline over up to 7.6 years** |
| **China** | NMPA approved **NEO — world's first commercial invasive BCI** (Mar 2026): 8 epidural electrodes, external decoding, insurance-coded, ~$900/procedure target; NeuCyber Beinao-1 (7+ patients), StairMed ($73M, Green Channel); 50+ BCI trials underway | Commercial BCI today = low channel count + modest compute — squarely ESP32/WASM-servable |

**Takeaways for ruv-neural**

- Two philosophies win simultaneously: **minimal-channel non-penetrating**
  (16-ch stentrode, 8-ch epidural NEO) clearing regulators fastest, and
  **high-bandwidth penetrating/high-density** (1,024–4,096 ch) chasing speech-rate
  decoding. A hardware-agnostic ingestion layer should serve both field-potential
  regimes (ADR-0017 boundary intact; epidural/ECoG are EEG-adjacent).
- **Bandwidth is the binding constraint**: Paradromics pushes 100 Mbit/s through
  skin and still preprocesses on-chip; Neuralink detects spikes on-implant.
  Compress/embed-at-the-sensor is standard architecture — direct validation of
  ruv-neural's edge-first thesis.
- Drift is quantified: assume slow per-channel yield decay (~7%/years) in
  recalibration design; vendors' answer (Synchron Chiral) is self-supervised
  cross-session foundation models.
- Track **Apple BCI-HID** as the OS-level output sink a Rust/WASM middleware
  could target.

## 4. Non-invasive and minimally-invasive sensing

Sensor physics matured; the bottleneck moved to real-time edge software.

| Finding | Date | Headline |
|---|---|---|
| 384-channel triaxial OPM-MEG (arXiv:2509.03107) | 2025-09 | First OPM array to **exceed cryogenic-MEG channel density**; dipole localization <1 mm |
| OPM matches SQUID for epilepsy spikes (NeuroImage 312:121232) | 2025 | 46-patient cohort, no significant detection difference; OPM sits 19.5 mm vs 30.1 mm from scalp → higher SNR (+27–60% pediatric) |
| Cerca Magnetics commercialization | 2025→2026-04 | Installs in 3 countries, £14M+£3.1M Innovate UK, **£3.8M Series A for clinical approval** — regulatory clearance is the open frontier |
| NV-diamond magnetometry | 2025 | 670 fT/√Hz at 280 µT dynamic range (laser-threshold); ambient hybrid 195 fT/√Hz; still **1–2 orders short** of the 5–100 Hz physiological band → 3–5 year sensor class (ADR-0019 unchanged) |
| In-ear EEG sleep staging (arXiv:2509.07896); NextSense launch | 2025 | Single dry in-ear channel: 90.5% wake/sleep, 65.1% 4-class; best dry ear-EEG 5-class κ 0.61 vs PSG; $399 consumer earbuds Q4 2025 |
| Kernel Flow 2 + edge inference (NeurIPS 2025 demo) | 2025-12 | ~3,500 TD-fNIRS channels whole-head; **real-time on-edge ML inference** (Jetson Thor/Holoscan) |
| Ambulatory human fUS (Sci. Adv. eadu9133) | 2025-06 | Walking subject, 20-month stability through sonolucent implant; ~200 µm functional resolution class |
| Consumer: Emotiv MW20, Muse S Athena | 2025 | Dual-channel in-ear EEG at 128 Hz in ANC earphones; **first consumer EEG+fNIRS** headband at $475 (claims 88–96% PSG staging agreement) |
| Dry electrodes close the wet gap | 2025–2026 | 512-lead graphene dry cap (3.8–6.5 kΩ, stable 103 days); MXene dry-vs-gel ERP ρ≥0.95; learned denoising recovers up to **+40% SNR** |

**Takeaways for ruv-neural**

- Treat **1–8 dry-electrode channels at 64–256 Hz** as the first-class ESP32
  regime (`ruv-neural-biosense`/`-sensor`, ADR-0002), with published ear-EEG
  staging baselines (κ ~0.6–0.73, 65–74% multi-class) as beatable on-device
  targets for NeuroSleep.
- Size the OPM-MEG path (ADR-0018) for **hundreds of triaxial channels at kHz
  rates**, and prioritize deterministic real-time interference-suppression /
  motion-correction DSP — the software layer the field currently lacks.
- Ingestion schemas should assume **mixed electrical + hemodynamic channels**
  (very different rates/latencies): EEG+fNIRS fusion already ships at $475.
- A reference **dry-electrode denoising model** (the +40% SNR-recovery class)
  directly raises usable channel quality on cheap hardware.

## 5. Neuromorphic computing and SNNs

Edge SNN silicon became purchasable, and training is no longer the bottleneck.

| Finding | Date | Headline |
|---|---|---|
| MatMul-free LLM on Loihi 2 (arXiv:2503.18002); Hala Point | 2025-03 | 370M-param LLM: 3× throughput at 2× less energy vs edge GPU; system-level **15 TOPS/W** (8-bit, sparse, batch-free) |
| SpiNNaker2 commercial (Sandia, Leipzig); EventProp on-chip | 2025 | 650M+-neuron Leipzig system; **on-chip event-based backprop**, up to 32× energy reduction at GPU-parity latency |
| IBM NorthPole appliance (arXiv:2511.15950) | 2025-11 | 115 INT4 peta-ops at 30 kW, 2.8 ms/token; single-card **72.7× tokens/s/W vs 4nm GPU** in the latency-bound regime |
| **Innatera Pulsar** — first mass-market neuromorphic MCU | 2025-05 | SNN engine + RISC-V + DSP; claims up to 100× lower latency / 500× lower energy; **sub-mW always-on**; COMPUTEX Best-in-Show |
| BrainChip Akida 2 + TENNs | 2025-08→2026-03 | SSM-derived **Temporal Event-based Neural Networks** in silicon — arguably the best-matched architecture class for continuous EEG/MEG streams |
| SpikingBrain-1.0 7B/76B (arXiv:2509.05276) | 2025-09 | Spiking-hybrid LLMs; **26.5× faster TTFT at 1M-token context**; trained entirely on non-NVIDIA (MetaX) GPUs |
| E-SpikeFormer (T-PAMI 2025) | 2025 | Directly-trained SNN at **86.2% ImageNet top-1** (173M params); 19M model: 79.8% at 5.9 mJ (~9× energy reduction) |
| ANN→SNN conversion matured | 2025 | Training-free transformer conversion; **one-timestep** conversion; near-lossless at 1–4 timesteps |
| Sub-mW EEG seizure detection on SynSense Xylo | 2024–2025 | 93% ictal/interictal accuracy at **~375 µW total** — the efficiency ceiling for continuous EEG classification |
| Darwin Monkey (Zhejiang U.) | 2025-08 | 2B spiking neurons / 100B+ synapses at ~2 kW — new scale leader over Hala Point |
| **NIR interchange standard**; Rust SNN ecosystem thin | 2024–2025 | NIR spans 11 platforms (Loihi, SpiNNaker, Xylo, major frameworks) — but **no production `no_std` Rust SNN runtime exists** |

**Takeaways for ruv-neural**

- The practical pipeline is now: **train a quantized ANN in mainstream tools →
  mechanically convert to a near-lossless 1–4-timestep SNN** for event-driven MCU
  execution — no exotic training loop needed in the embedded toolchain.
- **Strategic opportunity:** a `no_std` Rust SNN inference runtime with an NIR
  importer targeting ESP32/WASM is unclaimed first-mover territory squarely on
  ruv-neural's targets. Xylo's ~375 µW / 93% seizure detection is the efficiency
  ceiling to benchmark against.
- The neuromorphic advantage grows as workloads get more streaming, sparse, and
  latency-bound — exactly the EEG/closed-loop regime. TENN/SSM-style event-driven
  operators map to both Akida-class silicon and WASM.
- Innatera Pulsar (SNN + RISC-V + conventional core for safety-critical logic)
  is a plausible companion target beside the ESP32 tier.

## 6. Whole-brain simulation, connectomics, organoids, brain-inspired algorithms

Connectomes went from *mapped* to *running*; biological computing went
commercial; local-update learning matured.

| Finding | Date | Headline |
|---|---|---|
| MICrONS mm³ functional connectome (*Nature* 640:435) | 2025-04 | ~120k neurons, >523M synapses, ~75k functionally imaged; "like-to-like" wiring rule; **digital-twin** predictive models; EM connectomics = Nature Methods **Method of the Year 2025** |
| First embodied whole-brain fly emulation (Eon Systems) | 2026-03 | Full FlyWire connectome (~140k neurons / 50M synapses) driving a body in MuJoCo at **15 ms brain-body sync**; emergent foraging/grooming, no training |
| Fly connectome on Loihi 2 (arXiv:2508.16792) | 2025-08 | Whole connectome on 12 chips, orders faster than conventional simulation; advantage grows with spike sparsity |
| Cortical Labs **CL1** ships (~$35k) | 2025 | First commercial biological computer; ~115 units; Wetware-as-a-Service cloud; closed-loop stimulate-record-adapt with the same safety-envelope needs as ruv-neural's loop |
| FinalSpark Neuroplatform | 2025 | 16 cloud organoids, ~100-day lifespan, claimed up to 10⁶× lower energy/op |
| Thousand Brains / Monty | 2025–2026 | Independent nonprofit; *Neural Computation* paper: modular cortical-column learning modules with reference frames, rapid continual learning without deep-learning-scale data |
| Predictive coding scales (µPC, arXiv:2505.13124) | 2025-05 | **100+ layer PC networks** competitive with backprop via local, layer-parallel updates |
| Active inference in practical robotics (VERSES) | 2025 | Long-horizon mobile manipulation; hierarchical world models on physical robots — a principled bounded control law for closed loops |
| Sleep-replay consolidation (SESLR, arXiv:2507.02901 and successors) | 2025–2026 | Noise-injected unsupervised "sleep" phase + latent replay prevents catastrophic forgetting at **edge-compatible cost** |
| The Virtual Brain goes native | 2026 | **TVB C++ backend** (*Adv. Science*); €10M Virtual Brain Twin program pushing personalized EEG/MEG-fitted brain models into clinics |

**Takeaways for ruv-neural**

- **Directly actionable for the NeuroSleep rail:** gate on-device latent-replay
  consolidation of embeddings on detected sleep/idle windows from qEEG staging
  (`ruv-neural-neurosleep` + `ruv-neural-memory`) — the SESLR line shows this
  prevents catastrophic forgetting on edge budgets.
- Eon's 15-ms brain-body sync and ONWARD's clinical loop supply concrete
  **latency-budget references for ADR-0003**; active inference offers a bounded,
  uncertainty-aware control law compatible with the safety envelope (ADR-0012).
- TVB's pivot to native compiled simulation validates the Rust-first bet; TVB's
  patient-fitted neural-mass models are a natural downstream consumer of
  ruv-neural ingestion/DSP outputs.
- Local-update learning (µPC-style predictive coding) suits on-device adaptation
  where backprop through large graphs is infeasible (WASM/ESP32).

## 7. The ruvnet npm flywheel ecosystem (metaharness family)

The ecosystem consolidated in mid-2026 into an explicit, cross-depending
flywheel with one slogan: **"freeze the model, evolve the harness."** Versions
verified on the npm registry as of 2026-08-08.

| Package | Version (date) | Role in the flywheel |
|---|---|---|
| `metaharness` (lib: `@ruvnet/agent-harness-generator`) | 0.4.3 (2026-08-02) | **Scaffold** — mints a governed agent harness from any repo (CLI + Studio; host adapters incl. Claude Code and **RVM**, a hardware-isolated WASM guest with capability tables); `harness sign/verify` with Ed25519 witness manifests |
| `@metaharness/darwin` | 0.8.2 (2026-08-08) | **Evolve** — Darwin Gödel Machine-style evolution of one harness surface at a time under sandbox testing; self-reported 34.0% SWE-bench Lite at ~$0.005/instance, 55.6% Verified (cascade), 68.3% Lite *with acceptance test given* (explicitly not a leaderboard number) |
| `@metaharness/flywheel` | 0.1.9 (2026-08-08) | **Verify/promote** — frozen conjunctive promotion gate with recorded fingerprint, anti-Goodhart holdout + never-optimized anchor, **Ed25519-signed replayable promotion lineage** ("git for operating policies"); ADR-226 ablation: advisor loops gave 0 lift at 5.4× cost — evolve *executor* policies |
| `agenticow` (+ `@ruvector/rvf-node`) | 0.2.4 (2026-07) | **Remember** — copy-on-write vector branching: ~0.5 ms / 162 B per branch regardless of base size (vs 496 MB / 67 ms full copy at 1M vectors); instant rollback of poisoned branches; Rust single-file **RVF** core (ADR-0024's lineage) |
| `claude-flow` / `ruflo` | 3.34.0 (2026-07-31) | **Orchestrate** — v3 rebuild (TS + WASM), 60+ agents in coordinated swarms; renamed Ruflo Feb 2026; `@claude-flow/neural` carries a WASM RL/continual-learning stack |
| `agentic-flow` | 2.1.2 (2026-07-30) | **Runtime hub** — "the agentic meta-harness": cost-optimal routing, self-repair; pulls in `@ruvector/edge-full` (browser-WASM vector/neural/ONNX — the closest ruvnet artifact to ruv-neural's WASM target) and `agentdb` |
| `@metaharness/router`, `weight-eft`, `redblue` | 0.3.3 / 0.1.1 / 0.1.4 | **Satellites** — route to the cheapest adequate model; LoRA-distill wins back into cheap models; adversarial red-team/auto-patch/retest (prompt injection, tool misuse, denial-of-wallet) |
| `ruv-swarm`, `flow-nexus` | dormant since 2025-09 | **Legacy** — ruv-FANN-era lineage; new integration effort should target the ruvector/RVF + ruflo v3 generation |

**Context:** the family explicitly cites Sakana AI's Darwin Gödel Machine
(arXiv:2505.22954, v3 revised 2026-03; SWE-bench 20.0%→50.0% self-improvement)
but diverges by freezing the model and mutating one harness surface at a time
under a frozen gate. 2026 successors validate the direction: **Group-Evolving
Agents** (arXiv:2602.04837) reach 71.0% SWE-bench Verified via cross-branch
experience sharing — evidence that agenticow-style branchable shared memory is
the right substrate for group evolution.

**Takeaways for ruv-neural**

- The most transferable idea is **verifiability economics**: frozen promotion
  gates, anti-Goodhart holdouts, and Ed25519-signed replayable lineages are
  exactly the auditable evidence trail a neurotech safety case needs
  (ADR-0007/0009/0011).
- Back the embeddings store with **RVF CoW branches** (ADR-0024): per-subject /
  per-session forks of a shared base, checkpoint-before-stimulation rollback.
- Run **redblue-style adversarial passes** against the safety envelope each cycle.
- Caveat: headline Darwin/metaharness benchmark numbers are self-reported on npm
  READMEs (with linked official-harness eval artifacts and Wilson CIs) but have
  **no third-party leaderboard placement**; never quote the 68.3% with-test
  figure as a leaderboard result.

---

## 8. Cross-cutting trends

1. **Compress/embed at the sensor is consensus** across every regime — on-implant
   spike detection, on-chip preprocessing behind bandwidth walls, 2-bit FMs on
   MCUs, Jetson-class fNIRS inference — validating ruv-neural's edge-first thesis.
2. **Pretrain-across-subjects, calibrate-per-user-in-minutes** dominates decoding
   (invasive >50% WER gain; fMRI alignment; Synchron Chiral) — while imagined-speech
   EEG transfer stays at chance, so per-subject calibration remains in the loop.
3. **A replicated negative-results wave hit EEG FMs**; DIVER-1's scaling laws put
   parameter count last. Small, transparent, honestly-benchmarked models are
   *gaining* ground.
4. **Small beats large repeatedly**: 16.6K-param EEGNet > 5.8M LaBraM on speech;
   4M distilled students match full SSL; classical DFA transfers where FMs collapse.
5. **Streaming, causal, fixed-state inference converges** everywhere: 80-ms
   brain-to-voice, POSSM's 20–50 ms SSM chunks, TENNs, neuromorphic advantage
   growing precisely in the sparse/latency-bound regime.
6. **Closed-loop decode-then-stimulate is clinically real and regulator-supervised**
   — safety envelopes, mental-privacy gating, and auditable evidence trails are
   regulatory requirements, not features.
7. **The sensing bottleneck moved from physics to real-time edge software** —
   OPM field-nulling, dry-electrode learned denoising, on-device staging — a layer
   a deterministic Rust streaming-DSP workspace can own.
8. **Verifiable self-improvement loops matured** on both the academic (DGM, GEA)
   and ruvnet-flywheel fronts, providing machinery to evolve pipeline policies
   under a safety case.

## 9. Mapping to ruv-neural (recommended actions)

| # | Finding | Crate / ADR | Action |
|---|---|---|---|
| 1 | FM negative controls replicate (identity leakage, random-init wins, DFA blindness) | ADR-0016, `validation/` | **Adopt:** codify random-init comparator, dataset-identity probe, subject-disjoint splits as mandatory FM promotion gates |
| 2 | DFA/Hurst features transfer cross-population where FMs collapse | `ruv-neural-embed`, `-signal` | **Adopt now (cheap win):** implement DFA/Hurst in the feature set |
| 3 | REVE weights open; compact distilled/LUNA-Base models viable | ADR-0016, ADR-0024, `-embed` | **Watch/pilot:** ~4–7M-param model behind opt-in ONNX backend, gated by #1 |
| 4 | FEMBA: QAT mandatory (PTQ −30%), ~2 MB real-time on MCU | `-esp32`, `-wasm`, `-embed` | **Adopt as constraint:** QAT-only quantization plans; publish against FEMBA numbers |
| 5 | Streaming phoneme/keyword decoder + LLM rescoring wins; EEG tier = keyword/intent | `-brain2text`, `-decoder`, `-signal` | **Adopt:** 80-ms frame-hop streaming edge decoder + off-edge rescoring; validate on LibriBrain |
| 6 | Thought-password mental-privacy gating (>98% detection) | ADR-0007/0022/0023, `-brain2text` | **Adopt:** decode-armed/locked gating state machine |
| 7 | Cross-subject pretraining + minutes-scale calibration | `-embed`, ADR-0006/0022 | **Adopt the pattern:** shared latent + on-device per-user calibration; keep per-subject calibration for speech/intent |
| 8 | Sleep-PSG pretraining transfers; ear-EEG staging baselines published | `-neurosleep`, ADR-0015 | **Adopt:** compact sleep-pretrained encoder; publish vs κ 0.61–0.73 baselines and Stanford Sleep Bench |
| 9 | Sleep-replay consolidation prevents forgetting on edge budgets | NeuroSleep rail + `-memory` | **Watch/prototype:** gate on-device latent replay on detected sleep/idle windows |
| 10 | Event tokenizer + SSM cores: causal, 9× faster, cross-species | `-loop`, `-decoder`, `-stim` | **Adopt pattern:** fixed-state SSM/recurrent closed-loop decode path; 15-ms sync as ADR-0003 latency reference |
| 11 | OPM-MEG at clinical parity; bottleneck = real-time software | ADR-0018, `-sensor`, `-io`, `-signal` | **Adopt:** size for 100s of triaxial channels @ kHz; deterministic interference-suppression DSP |
| 12 | NV-diamond still 1–2 orders short in physiological band | ADR-0019 | **Watch:** keep as 3–5-year abstraction; no implementation yet |
| 13 | Consumer regime: 1–8 dry channels @ 64–256 Hz; EEG+fNIRS ships | `-biosense`, `-sensor`, ADR-0002 | **Adopt:** first-class dry-electrode regime; mixed electrical+hemodynamic schemas; reference denoising model |
| 14 | Sub-mW SNN EEG on Xylo; NIR standard; no `no_std` Rust runtime | `-esp32`, `-wasm`, `-core` | **Adopt as strategic opportunity:** `no_std` Rust SNN runtime + NIR importer (train ANN → 1-timestep SNN) |
| 15 | Invasive spans 8–4,096 ch; Utah decay ~7%/yr-scale; Apple BCI-HID | ADR-0017, `-io`, `-embed` | **Watch (mostly out of scope):** hardware-agnostic field-potential ingestion; drift-robust recalibration; track BCI-HID as sink |
| 16 | Flywheel promotion gates + signed lineage; redblue red-teaming | ADR-0007/0009/0011, `-loop`/`-stim` | **Adopt:** gate-fingerprinted, Ed25519-signed promotion for stimulation/staging policy changes; adversarial passes each cycle |
| 17 | agenticow CoW branching over RVF; GEA cross-branch sharing | ADR-0024, `-memory`, `-embed` | **Adopt:** RVF CoW-branched embedding store; checkpoint-before-stimulation rollback |
| 18 | µPC / active inference: local updates, bounded control | `-loop`, ADR-0003/0012 | **Watch:** PC uncertainty signals and free-energy control laws as safety-envelope inputs |
| 19 | Synthetic MEG (MEG-GPT) + open code/datasets | `-signal` tests, `validation/`, `-brain2text` | **Adopt:** use as test fixtures and validation baselines |

## 10. Flywheel playbook (metaharness + Darwin + flywheel + agenticow)

> **Status: steps 1–2 executed against this repo on 2026-08-08** with
> `metaharness` 0.4.3 / `@metaharness/darwin` 0.8.2 (see [`harness/`](../../harness)):
>
> - `npx metaharness score .` → harnessFit 67, compileConfidence 100,
>   toolSafety 100, est. $0.048/run, archetype `rust-crate-harness`,
>   scaffoldReady true. `npx metaharness genome .` → verdict **READY**
>   (risk 21%, test confidence 80%, publish readiness 75%).
> - Scaffolded `harness/` with the tool-recommended `vertical:coding`
>   template targeting the `claude-code` host: 4 agents
>   (architect/implementer/reviewer/test-writer), `/plan-change` + `evolve`
>   skills, `doctor`/`review-diff` commands, default-deny permission posture
>   (no `git push`, no `rm -rf`, no `.env` reads). Smoke tests 4/4 green;
>   `harness doctor` → **HEALTHY**.
> - `harness sign` + `verify` → 19-entry SHA-256 witness manifest at
>   `harness/.harness/witness.json`, verdict VALID. *Caveat:* the Ed25519
>   signature is degraded in this environment (kernel `witnessVerify`
>   unavailable, placeholder public key) — the manifest is content-hash
>   witness only until CI signs with a real key per the package's GCP
>   Secret Manager flow.
> - Darwin Mode dry-run (`npm run evolve:dry`, mock sandbox, 2 generations ×
>   3 children): winner `g2_v5` mutated the `contextBuilder` surface,
>   **+0.110 over baseline** (0.875 vs 0.765), lineage
>   `baseline → g1_v0 → g2_v5` committed in
>   `harness/.metaharness/{archive,lineage}.json` as audit-trail evidence.
>   A real (non-mock) evolve run against cargo-test fitness is the next step.

How to run this research-and-optimization loop continuously, on a
monthly-or-faster cadence:

1. **Scaffold** — mint a research harness for ruv-neural and a pipeline-tuning
   harness targeting the RVM hardware-isolated WASM host; sign both (Ed25519
   witness manifests feed ADR-0009's audit trail):

   ```bash
   npx metaharness ruv-neural-research --template vertical:research --host claude-code
   npx metaharness ruv-neural-tuner --template minimal --host rvm
   npx harness sign && npx harness verify
   ```

2. **Evolve** — point `@metaharness/darwin` at *measurable pipeline surfaces*,
   one surface per generation (DSP parameters, embedding feature sets such as the
   new DFA/Hurst features, sleep-staging thresholds, decoder policies), scored
   against fixed benchmark suites (LibriBrain, Brain-to-Text Benchmark, ear-EEG
   staging baselines, Stanford Sleep Bench) *and* the ADR-0016 negative controls:

   ```bash
   npx ruflo metaharness evolve --repo . --confirm --generations 5 --children 4
   ```

   Heed the ADR-226 ablation: evolve **executor** policies — advisor loops gave
   zero lift at 5.4× cost.

3. **Verify/promote** — wrap every candidate in `@metaharness/flywheel`'s frozen
   conjunctive gate with an anti-Goodhart holdout and a never-optimized anchor.
   Promotions emit Ed25519-signed, replay-auditable receipts
   (`verifyReplayBundle()` replays the lineage to gen-0) — the safety-case
   evidence chain for anything touching the stimulation envelope (ADR-0007/0011).
   Run `redblue` adversarially against the safety envelope each cycle.

4. **Remember** — store per-generation embeddings and run artifacts in
   `agenticow`/RVF copy-on-write branches: branch before each risky experiment
   (~0.5 ms / 162 B), roll back poisoned branches instantly, and share the base
   across parallel Darwin lineages (the GEA experience-sharing pattern).

5. **Repeat** — a recurring research-agent swarm (ruflo/agentic-flow
   orchestrated, router-selected cheap models) re-sweeps arXiv/npm/vendor sources
   like this sweep did, diffs findings against ADRs 0015–0024, and files
   ADR-update candidates. Each cycle re-bases on the last promoted winner,
   compounding a verifiable lift curve.

## 11. Gaps and caveats

This sweep could **not** verify:

- **Metaharness/Darwin benchmark numbers are self-reported** (npm READMEs with
  linked eval artifacts, Wilson CIs) with no third-party leaderboard placement or
  independent replication; the 68.3% figure is a with-acceptance-test number.
- **No agent audited ruv-neural's actual crate code** — the §9 mapping is based
  on the workspace's stated architecture and ADR titles, not on verifying current
  implementations (e.g. whether DFA/Hurst or QAT paths already exist).
- Several key results are **preprints or vendor claims**: the multi-user
  speech-BCI pretraining result (bioRxiv), Utah-array longevity (medRxiv), Muse
  Athena's 88–96% staging claim, FinalSpark's 10⁶× energy figure, Innatera's
  100×/500× claims, Synchron's Chiral (no published benchmarks).
- **No hands-on verification** of hardware availability/pricing (Pulsar, Xylo,
  GAP9) or that FEMBA-class / NIR toolchains work from Rust today.
- EEG FM benchmarking still lacks a single accepted community benchmark; clinical
  coverage beyond seizure/dementia/sleep is thin.
- **Regulatory coverage is FDA + NMPA only** — EU MDR pathways for consumer EEG,
  OPM-MEG, and closed-loop stimulation were not researched.
- **Licenses of recommended open assets** (REVE weights, brain-to-voice
  code/data, brain2qwerty, LibriBrain) were not verified for commercial use.
- Adversarial robustness of neural decoders themselves, and the broader
  neurorights/neural-data-law landscape (beyond the single Stanford
  thought-password result), were not swept.

## 12. Method note

This report was produced by a deterministic agent-swarm workflow: seven
parallel research agents (brain-to-text, foundation models, invasive BCI,
non-invasive sensing, neuromorphic/SNN, brain simulation & algorithms, ruvnet
flywheel tooling), each running independent live web research with structured
(schema-validated) outputs, followed by a synthesis agent that produced the
cross-cutting trends, the §9 mapping, and the §10 playbook. Totals: 8 agents,
~437k research tokens, 124 tool calls. §10 describes how to make this loop
self-sustaining with the npm `metaharness` family. All claims carry their
sources inline; preprint/vendor-claim status is flagged in §11.
