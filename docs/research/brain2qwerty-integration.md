# Research Report — Integrating Brain2Qwerty + SpanishBCBL into rUv Neural

> **Status:** Research / feasibility study — **Phases 1–2 + the optimizer are now
> implemented** in the [`ruv-neural-brain2text`](../../ruv-neural-brain2text) crate
> (see [§9 Implementation status](#9-implementation-status)).
> **Date:** 2026-06-29 · **Author:** rUv (ruv@ruv.net)
> **Scope:** Evaluate whether Meta AI's *Brain2Qwerty* brain-to-text system and the
> *SpanishBCBL* (DECOMEG) dataset can be integrated into the `ruv-neural` Rust
> workspace, and whether the ruvnet tooling — `metaharness`, **Darwin Mode**
> (`@metaharness/darwin`), and `agenticow` — can be used to optimize the result.

---

## TL;DR

| Question | Answer |
|---|---|
| Can we integrate Brain2Qwerty into rUv Neural? | **Yes, but as a *bridge*, not a port.** Brain2Qwerty is Python/PyTorch + MEG-centric and **CC BY-NC 4.0**. The realistic path is a new `ruv-neural-brain2text` crate that ingests the SpanishBCBL data into rUv Neural's existing sensor→signal pipeline, plus an optional Python sidecar that runs the upstream model for inference. A full Rust re-implementation of the deep model is a large, separate effort. |
| Can we use the SpanishBCBL dataset? | **Yes** for research/non-commercial use. ~262 GB, MEG (`.fif`) + EEG (BrainVision), 1 kHz, CC BY-NC 4.0. A loader maps cleanly onto rUv Neural's `SensorType::SquidMeg` / `SensorType::Eeg` and `MultiChannelTimeSeries`. |
| Can `metaharness` / Darwin / `agenticow` optimize it? | **Partially, and not in the way the names suggest.** **Darwin Mode is the actual optimizer** (evolves the *harness/agent strategy*, "freeze the model, evolve the harness"). `metaharness` is the scaffold factory. `agenticow` is a copy-on-write vector store for experiment memory — not an optimizer. They optimize the *agentic workflow around* training/tuning, not model weights directly. |
| Biggest blockers | (1) **License** — both code and data are **CC BY-NC 4.0** (non-commercial). (2) **MEG dependency** — strong results need a shielded MEG scanner; EEG is much weaker (67% vs 32% CER). (3) **Language/runtime gap** — PyTorch model vs Rust workspace. |

---

## 1. Background: what we're integrating

### 1.1 Brain2Qwerty (Meta AI / FAIR, with BCBL)

Brain2Qwerty is a non-invasive **brain-to-text** system: it decodes full sentences a
person *types* on a QWERTY keyboard directly from their brain activity, recorded with
a wearable/desktop scanner rather than a surgical implant. Participants read a briefly
memorized sentence, wait, then type it from memory; the model reconstructs the typed
text. Stated goal: a safer alternative to invasive neuroprostheses for people who have
lost the ability to speak or move.

There are **two versions** in the same repo:

- **V1** — *"Non-invasive decoding of typed sentences from human brain activity"*
  (Nature Neuroscience 2026; arXiv:2502.17480, Feb 2025). Synchronous,
  keystroke-windowed decoding. **This is the runnable, reproducible path today.**
- **V2** — *"Accurate Decoding of Natural Sentences from Non-Invasive Brain
  Recordings"* (preprint, 2026). Asynchronous, whole-sentence decoding. Trained on
  an **EnglishBCBL** dataset that is **embargoed until paper acceptance**, so V2 is
  **not reproducible on original data yet**.

**Input modality.** Both MEG and EEG were tested; **MEG dramatically outperforms EEG**.
All headline numbers are MEG.

**V1 architecture — three stages (Conv → Transformer → Language Model):**

1. **Convolutional module (keystroke encoder).** Input is a 0.5 s window around each
   keypress (−0.2 s → +0.3 s). A subject-specific linear layer absorbs inter-subject
   variation; a spatial-attention channel-merger encodes relative sensor positions;
   then **8 residual conv blocks** (kernel 3, dilation period 3, skip connections,
   dropout, GELU). Output dim **h = 2048**.
2. **Transformer module.** **4 layers, 2 heads**, single-sentence receptive field →
   per-character logits.
3. **Language model module.** A **9-gram character LM** (KenLM, trained on Spanish
   Wikipedia), applied as post-processing via **beam search (beam 30)** over a
   weighted mix of transformer logits and LM probabilities (**α = 5**).

**V1 preprocessing.** Bandpass **0.1–20 Hz**, resample to **50 Hz**, baseline-correct
by subtracting the per-channel mean over (−0.2, 0) s.

**V2 architecture (for context).** Conv + Conformer encoder with a character-level
**CTC** head, a word-level **contrastive (SigLIP-style)** aligner, and a
**LoRA-adapted LLM** that autoregressively generates the sentence. Staged training
schedule (CTC @ epoch 0, contrastive @ 150, LLM @ 225).

**Results.**

| Metric | MEG | EEG |
|---|---|---|
| **V1 average Character Error Rate (CER)** | **32% (±0.6)** | **67% (±1.5)** |
| V1 best participant CER | **19% (±1.1)** | — |
| V2 word accuracy (avg) | **~61%** (WER ~39%) | — |
| V2 best participant word accuracy | **78%** | — |

Accuracy improves **log-linearly with data volume** — the non-invasive gap to surgical
implants may narrow with scale rather than being closed today.

**Repo.** `facebookresearch/brain2qwerty`, 100% Python. Layout: `brain2qwerty_v1/`,
`brain2qwerty_v2/`, `studies/` (registers `Pinet2024Meg` / `Pinet2024Eeg`), plus
`scripts/`, `tests/`, `pyproject.toml`, `requirements.lock`. **License: CC BY-NC 4.0**
(non-commercial; pyproject classifies as "Other/Proprietary"). Key deps:
`torch==2.6.0`, `lightning==2.5.2`, `x-transformers`, `mne==1.11.0`, `transformers` +
`peft` (V2 LoRA), optional `kenlm` (`[lm]` extra). `requires-python >=3.12`. Runs via
`python -m brain2qwerty_v1.main {cache,debug,train,eval}` on PyTorch Lightning
(1 node / 8 GPU default, single-GPU fallback). Infra libs `neuralset`/`neuralfetch`/
`neuraltrain` come from the separate `facebookresearch/neuroai` repo.

> ⚠️ A third-party `The-Swarm-Corporation/Brain2Qwerty` exists — an **independent EEG
> re-implementation**, not Meta's official code. Use the `facebookresearch` repo.

**Limitations** (carry these into any rUv Neural framing — consistent with the
project's "not a medical device, no efficacy claim" ethics): non-invasive accuracy
trails invasive devices; EEG is much weaker than MEG; MEG needs a bulky shielded
scanner (not wearable in daily life); the task is *memorized typed* sentences, not
free thought; per-subject training on small cohorts (V1 ~20/modality, V2 only 9).

### 1.2 SpanishBCBL dataset (`bcbl190626/SpanishBCBL`, "DECOMEG")

Non-invasive MEG + EEG recordings of healthy adults typing memorized **Spanish**
sentences — the **Brain2Qwerty V1 dataset**, owned by BCBL (Basque Center on
Cognition, Brain and Language).

| Property | MEG | EEG |
|---|---|---|
| System | Megin (Elekta Neuromag) | BrainVision actiCAP slim |
| Channels | **306** (102 mag + 204 grad) | **64** |
| Sampling rate | **1 kHz** | **1 kHz** |
| Acquisition filter | 0.1–330 Hz | — |
| Sentences / chars | ~5.1K / ~193K | ~4K / ~146K |
| Recording time | ~21.5 h | ~17.7 h |
| File format | raw `.fif` + `.mat` logs | `.eeg`/`.vhdr`/`.vmrk` + `.mat` |

- **Subjects:** 35 volunteers (19 unique MEG; 5 in both modalities); mean age 31.6,
  native Spanish, right-handed skilled typists.
- **Paradigm:** read (RSVP) → wait (1.5 s fixation) → type from memory, no feedback.
- **Size:** **≈262 GB**. **License: CC BY-NC 4.0.** De-identified; MRI/video/
  eye-tracking excluded.
- **Load:** via the repo's `neuralset`/`neuralfetch` (`Study(name="Pinet2024Meg",
  path=...)` → per-event dataframes) **or** raw:
  `hf download bcbl190626/SpanishBCBL --repo-type dataset --local-dir SpanishBCBL`
  (optionally `--include "MEG/*"` / `"EEG/*"`).

### 1.3 The ruvnet optimization tooling

All three are **real, published npm packages** by ruvnet (verified against the npm
registry). The names are slightly misleading, so the report uses them precisely:

| Term in the request | Real package | Latest | What it actually is |
|---|---|---|---|
| "npm metaharness" | `metaharness` (lib: `@ruvnet/agent-harness-generator`) | 0.2.7 | CLI + browser "Studio" that **scaffolds** a custom, governed AI-agent harness from any repo (own `npx` CLI, MCP server, memory, Ed25519-signed releases). |
| "Darwin mode" | `@metaharness/darwin` | 0.7.1 | The **actual optimizer**. Evolutionary self-improvement modeled on the **Darwin Gödel Machine** (Sakana AI, arXiv:2505.22954). Slogan: *"freeze the model, evolve the harness."* |
| "npm agenticow" | `agenticow` | 0.2.3 | **Copy-on-write vector DB** for multi-agent memory ("Git for vectors"). **Not** an optimizer — it's experiment storage/versioning. |

**Darwin Mode — how it works (confirmed).** The foundation model stays fixed; Darwin
mutates the *scaffolding* across **seven policy surfaces** — `planner`,
`contextBuilder`, `reviewer`, `retryPolicy`, `toolPolicy`, `memoryPolicy`,
`scorePolicy` — sandbox-scores each variant, and promotes only measured wins, keeping
a DGM-style archive. Benchmarked (per registry) at 51.3% SWE-bench Lite / 55.6%
Verified with a GLM→Opus cascade. Commands: `npm run evolve` inside a scaffold, or
`npx ruflo metaharness evolve --repo . --confirm --generations N --children N`.

**agenticow — how it works (confirmed).** `npm install agenticow`; CLI
`agenticow {init,ingest,branch,query,diff,checkpoint,rollback,promote,bench}`; Node
API `open()/ingest()/branch()/query()/checkpoint()/rollback()/promote()/diff()`.
Branch a base memory in ~0.5 ms / 162 bytes regardless of size.

> **Important framing:** these tools optimize the **agentic workflow** that drives
> training/tuning/experiment-tracking — not the gradient descent inside the model.
> The mapping to a neural-decoding fitness function is **plausible but undocumented**;
> no published neural-decoding example exists for either. Treat §4 as a design
> proposal, not a reproduction of a known result.

---

## 2. How this maps onto the existing rUv Neural architecture

rUv Neural is a Rust workspace with a clear linear pipeline
(`sensor → signal → graph → mincut → embed → memory → decoder`), plus `stim` / `loop`
/ `biosense` for the closed-loop stimulation side. Two facts make a Brain2Qwerty
bridge natural rather than alien:

1. **The sensor model already knows MEG and EEG.**
   `ruv-neural-core::sensor::SensorType` already has `SquidMeg`, `Eeg`, `Opm`,
   `NvDiamond`, `AtomInterferometer`, with per-type sensitivities and a
   `SensorChannel { position, orientation, sample_rate_hz, label, … }` model. The
   SpanishBCBL 306-ch Megin array and 64-ch BrainVision cap fit this directly.
2. **The signal layer already has the right DSP.** `ruv-neural-signal` has Butterworth
   IIR (SOS) filters, Welch PSD, Hilbert phase, artifact rejection, and connectivity
   (PLV/coherence/AEC). Brain2Qwerty V1's preprocessing (bandpass 0.1–20 Hz →
   resample 50 Hz → baseline correct) is a small, well-bounded addition on top.

**Where the gap is.** The existing `ruv-neural-decoder` is a *cognitive-state*
classifier — KNN / threshold / transition / clinical ensembles over **graph topology
embeddings** (`NeuralEmbedding` + `TopologyMetrics`), producing a `CognitiveState`. It
is **not** a sequence-to-text model. Brain2Qwerty is a deep Conv+Transformer+LM that
emits **character sequences**. So a brain-to-text capability is a genuinely new decoder
*type*, not a tweak to the current one.

```
                 EXISTING rUv NEURAL                         NEW (this proposal)
   sensor ─► signal ─► graph ─► mincut ─► embed ─► memory ─► decoder (CognitiveState)
     │          │                                                   ▲
     │          │                                                   │
     └──────────┴───────────────────────────────┐                  │ (shared types)
                                                 ▼                  │
                                  ruv-neural-brain2text  ───────────┘
                                  ├─ dataset loader (.fif / BrainVision → MultiChannelTimeSeries)
                                  ├─ B2Q preprocessing (0.1–20 Hz, 50 Hz, baseline)
                                  ├─ event/keystroke windowing (read→wait→type epochs)
                                  ├─ CharSeq decoder trait + outputs (per-char logits, CER/WER)
                                  └─ inference backend: { native(stub) | python-sidecar }
```

---

## 3. Proposed integration design

### 3.1 New crate: `ruv-neural-brain2text`

A new workspace member, depending on `ruv-neural-core` and `ruv-neural-signal`. Modules:

- **`dataset`** — read SpanishBCBL into core types.
  - MEG `.fif`: parse via a Rust reader or (pragmatically) a thin Python/`mne`
    exporter that emits RVF / `.npy`; fill `SensorArray { SensorType::SquidMeg, 306 ch }`
    and `MultiChannelTimeSeries`.
  - EEG BrainVision (`.vhdr`/`.eeg`/`.vmrk`): straightforward to parse natively;
    `SensorType::Eeg`, 64 ch, 10-20-style labels.
  - `.mat` behavioral logs → `KeystrokeEvent { char, t_start, t_dur }` timelines.
- **`preprocess`** — Brain2Qwerty V1 pipeline reusing `ruv-neural-signal`: bandpass
  0.1–20 Hz, resample to 50 Hz, per-channel baseline subtraction over (−0.2, 0) s.
- **`epoch`** — windowing: extract −0.2…+0.3 s epochs around each keystroke (V1), and
  whole-sentence continuous windows (V2-style), with train/val/test splits.
- **`decode`** — a `CharSequenceDecoder` trait + `CharDecodeOutput` (per-char logits,
  decoded string, CER, WER, confidence). This mirrors the existing decoder crate's
  shape so outputs can flow through the same evidence/witness layer.
- **`metrics`** — CER / WER (Levenshtein) and a per-subject report, matching the
  upstream `extract_predictions.py` semantics.
- **`backend`** — pluggable inference:
  - `python-sidecar` (default, realistic): shell out to the upstream
    `python -m brain2qwerty_v1.main eval --ckpt …` or a small FastAPI/stdio service,
    exchanging RVF/JSON. Fastest route to a working demo.
  - `native` (stretch): a Rust forward pass (e.g. via `candle`/`burn`) for the
    Conv+Transformer; KenLM rescoring via FFI or a Rust n-gram. Large effort —
    park behind a feature flag.

### 3.2 CLI surface (`ruv-neural-cli`)

```bash
ruv-neural brain2text fetch   --modality meg --out data/SpanishBCBL
ruv-neural brain2text prep    --in data/SpanishBCBL --out data/epochs.rvf
ruv-neural brain2text decode  --epochs data/epochs.rvf --backend python-sidecar
ruv-neural brain2text report  --predictions out/pred.csv   # per-subject CER/WER
```

### 3.3 Evidence-layer fit

Decoded sentences + CER/WER per subject become a natural **signed evidence bundle**
(the existing Ed25519 witness system): hash-chain the dataset version, preprocessing
params, model checkpoint hash, and results so a third party can re-verify the
reproducibility claim locally — exactly the project's existing value proposition,
extended to brain-to-text.

---

## 4. Optimization with metaharness / Darwin Mode / agenticow

This is where the request's tooling fits — **around** the pipeline, optimizing the
*agentic workflow*, not the model's weights.

### 4.1 `metaharness` — scaffold the optimization agent

Mint a governed harness over the `ruv-neural` repo so an agent can operate the
training/tuning loop with an auditable CLI + MCP server + memory:

```bash
npx metaharness ruv-neural-b2t --template vertical:coding --host claude-code
harness validate && harness score .      # baseline 0–100 scorecard
harness sign && harness verify           # Ed25519 — mirrors rUv Neural's witness ethos
```

This gives the optimizer a repo-aware, default-deny-MCP, signed-release wrapper. It
does no optimization itself; it's the factory.

### 4.2 Darwin Mode (`@metaharness/darwin`) — the actual optimizer

"Freeze the model, evolve the harness." Point Darwin's **`scorePolicy`** at the
decoder's validation metric and let it evolve the agent strategy that runs prep +
training + eval:

```bash
npx ruflo metaharness evolve --repo . --confirm --generations 5 --children 4
```

- **Fitness function:** negative validation **CER** (or WER) from
  `ruv-neural brain2text report`, parsed per generation.
- **Mutated surfaces:** `contextBuilder` (which preprocessing / epoch params the agent
  tries), `retryPolicy` / `toolPolicy` (how it reruns failed training), `scorePolicy`
  (how candidates are ranked), `planner` (search order over the config space).
- **Loop:** each generation spawns `--children` variant configs, sandbox-runs `prep`
  → `decode` → `report`, keeps only measured CER improvements in the DGM-style archive.

> **Honest caveat (from research):** Darwin evolves the *harness/agent policies*, not
> gradient-descent hyperparameters directly. To make hyperparameters (filter band,
> resample rate, epoch window, beam size, LM weight α) part of the search, express them
> as knobs the harness controls (e.g. CLI flags Darwin mutates). This mapping is
> reasonable but **not documented upstream** — validate empirically before claiming
> gains.

### 4.3 `agenticow` — experiment memory / cheap rollback

Use agenticow as the **copy-on-write vector store** backing the search so each Darwin
child is an isolated, diffable branch:

```bash
agenticow init   b2t-mem.rvf --dim 2048          # h=2048 matches B2Q conv output dim
agenticow branch b2t-mem.rvf --as gen3-childB     # ~0.5 ms fork per candidate
agenticow query  b2t-mem.rvf.gen3-childB.rvf --k 10
agenticow diff   b2t-mem.rvf.gen3-childB.rvf      # compare candidate vs base
# promote the winner, rollback the rest
```

Store per-trial neural embeddings / candidate-config feature vectors here; branch per
Darwin child, `diff` to compare, `promote` the winner, `rollback` losers. It versions
experiments — it does not tune anything by itself.

### 4.4 Optimization data flow

```
  metaharness ── scaffolds ──► governed agent (CLI + MCP + signed releases)
        │
        ▼
  Darwin Mode ── evolves ──► {planner, contextBuilder, retryPolicy, scorePolicy,…}
        │  each generation: spawn N children → run prep/decode/report → score = −CER
        ▼
  agenticow  ── stores ──► per-child vector memory (branch/diff/promote/rollback)
        │
        ▼
  best config + checkpoint  ──►  signed evidence bundle (rUv Neural witness layer)
```

---

## 5. Licensing & ethics (read before any product use)

- **Both the Brain2Qwerty code and the SpanishBCBL data are CC BY-NC 4.0
  (non-commercial).** rUv Neural is MIT/Apache-2.0. Bundling CC BY-NC code/data into a
  commercial offering, or vendoring it into the MIT/Apache crates, is **not
  compatible**. Keep any B2Q/BCBL-derived code and data **out of the published crates**
  — isolate behind an optional, clearly-licensed `examples/` or sidecar that users opt
  into and download themselves, with attribution.
- **De-identified human neural data.** Even de-identified MEG/EEG is sensitive; honor
  the dataset's terms, keep it out of the repo (gitignore the ~262 GB), and document
  provenance.
- **No efficacy / medical claims** — consistent with the project's existing stance.
  Brain-to-text here is a *research decoding* capability, not a clinical device.
- **MEG reality check.** The strong numbers need a shielded MEG scanner. The EEG path
  (which fits cheaper hardware and rUv Neural's edge story) currently yields ~67% CER —
  far from usable. Be explicit about this in any README framing.

---

## 6. Phased roadmap

| Phase | Deliverable | Effort | License-safe? |
|---|---|---|---|
| **0. This report** | Feasibility + design (this doc) | done | ✅ |
| **1. Dataset bridge** | `ruv-neural-brain2text::dataset` — load EEG (BrainVision) natively + MEG via `mne` export → `MultiChannelTimeSeries`; keystroke timelines from `.mat`. Gitignore data. | S–M | ✅ (loader is original) |
| **2. Preprocessing parity** | V1 pipeline on `ruv-neural-signal` (0.1–20 Hz, 50 Hz, baseline); epoch windows; CER/WER metrics. | M | ✅ |
| **3. Sidecar inference** | `backend::python-sidecar` driving upstream V1 `eval`; reproduce per-subject CER on a few subjects. | M | ⚠️ keep B2Q deps in opt-in sidecar |
| **4. Evidence bundle** | Wrap results in the Ed25519 witness layer (dataset hash + params + checkpoint hash + CER). | S | ✅ |
| **5. Optimization harness** | `metaharness` scaffold + Darwin `evolve` over prep/epoch/decoding knobs, agenticow experiment memory. | M | ✅ (tooling is MIT) |
| **6. (Stretch) Native decoder** | Rust Conv+Transformer via `candle`/`burn` + n-gram/KenLM-FFI rescoring. | L | ✅ if clean-room |

S ≈ days, M ≈ 1–2 weeks, L ≈ multi-week.

## 7. Risks & open questions

- **Disk/compute:** ~262 GB dataset + CUDA GPU(s) for upstream training. Plan storage
  and a non-repo data dir.
- **`.fif` in Rust:** no mature native reader; Phase 1 leans on an `mne` export step.
  Acceptable as a documented preprocessing dependency.
- **V2 is not reproducible** (EnglishBCBL embargoed) — target **V1** only for now.
- **Darwin↔ML mapping unproven** for neural decoding — measure before claiming wins.
- **Cross-language maintenance:** a Python sidecar adds a second toolchain; keep it
  optional and pinned (`requirements.lock`, `torch==2.6.0`, etc.).

## 8. Recommendation

Proceed with **Phases 1–4** as a research bridge: a new opt-in `ruv-neural-brain2text`
crate that loads SpanishBCBL into the existing sensor/signal types, reproduces V1
preprocessing + metrics, runs inference through an opt-in Python sidecar, and emits a
signed evidence bundle. Add the **metaharness + Darwin + agenticow** optimization
harness in Phase 5 as the agentic workflow optimizer (with the documented caveat that
it tunes the *workflow*, not weights). Keep all CC BY-NC material isolated from the
MIT/Apache crates, and keep the project's "no medical/efficacy claim" framing.

---

## 9. Implementation status

A working, tested crate — [`ruv-neural-brain2text`](../../ruv-neural-brain2text) —
now implements the bridge and the optimizer natively in Rust (no Python required to
run it; the deep model remains an opt-in sidecar). **30 unit tests pass; the full
workspace remains green.**

**Built and tested:**

| Stage | Module | Notes |
|---|---|---|
| Dataset loader | `dataset::brainvision` | Real, zero-dep `.vhdr/.vmrk/.eeg` reader (INT_16 + IEEE_FLOAT_32, multiplexed + vectorized), round-trip tested. |
| Synthetic data | `dataset::synthetic` | SpanishBCBL-*structured*, *learnable* generator (real 262 GB CC BY-NC data stays out of the repo). |
| Events | `events` | Keystroke / sentence / timeline model. |
| Preprocessing | `preprocess` | V1 recipe (bandpass 0.1–20 Hz, resample 50 Hz) on `ruv-neural-signal`; out-of-band attenuation tested. |
| Epoching | `epoch` | −0.2…+0.3 s windows, baseline correction, feature extraction, deterministic train/val/test split. |
| Decoder | `decode` | `CharSequenceDecoder` trait, prototype acoustic model, char n-gram LM, beam-search fusion (`score = acoustic + α·LM`), CER/WER. |
| Optimizer | `evolve` | **Darwin mode**: genetic search over `Brain2TextConfig`, fitness `1 − val_CER`, tournament selection, crossover, mutation, elitism, archive. |
| End-to-end | `evaluate()` + `examples/demo.rs` | One call runs the whole pipeline; the demo evolves a config. |

**Now also built — trainable models, a composable harness, and a benchmark:**

| Capability | Module | Notes |
|---|---|---|
| Trainable acoustic models | `model::{prototype, linear, mlp}` | `Prototype` (nearest-centroid), `Linear` (multinomial logistic regression, SGD), `Mlp` (1 hidden layer, SGD/backprop). All **serializable** — the weights are the artifact. Training is numerically guarded (logit clamp + finite-sanitize) so any config stays serializable. |
| Composable harness | `harness` | Fluent `Harness` builder → self-contained, serializable `TrainedPipeline` (config + weights + LM) with `to_json`/`from_json`. The optimizer drives the same code path. |
| Optimize across models | `evolve` | Search space extended to the model family *and* training hyperparameters (lr, epochs, hidden size, L2). |
| Benchmark | `examples/benchmark.rs`, `benches/decode.rs` | Per-model CER/WER/timing matrix, baseline-vs-evolved, decode throughput; criterion micro-benches. |
| Distribution licensing | `WEIGHTS_LICENSE`, `MODEL_CARD.md` | Code is MIT/Apache; weights trained on SpanishBCBL are CC BY-NC 4.0 (see §5). |

**Demonstrated result** (synthetic, learnable data — validates the machinery and the
optimizer, *not* the published accuracy). The optimizer searches across model families
and hyperparameters and selects `Linear`:

```
cargo run -p ruv-neural-brain2text --example brain2text_demo
cargo run --release -p ruv-neural-brain2text --example benchmark
```

| | test CER | test WER |
|---|---|---|
| Prototype (V1 defaults) | 0.596 | 1.167 |
| Linear (V1 defaults) | 0.188 | 0.611 |
| MLP (V1 defaults) | 0.921 | 1.222 (overfits tiny data; the optimizer avoids it) |
| **Evolved best (Linear)** | **0.037** | **0.194** |

~7,000 keystrokes/sec decoded; serialized pipeline artifact ~50 KB.

**Can we train and distribute our own model on the data?** Yes — *non-commercially*.
The dataset card is **CC BY-NC 4.0, not gated, no separate Data Use Agreement**, so
training is permitted; the conservative, defensible posture is to treat the resulting
**weights as CC BY-NC 4.0** too (release as a separate, research-licensed artifact —
*not* inside the MIT/Apache crates — with the two required citations and a model card).
Commercial distribution requires training on your own / commercially-licensed data.
The crate's clean-room code stays MIT/Apache regardless; only weights trained on NC
data inherit the restriction. See `WEIGHTS_LICENSE` and `MODEL_CARD.md`.

**Deliberately a stand-in (per §5 licensing):** the native acoustic models are
clean-room classifiers, *not* a port of the CC BY-NC Conv+Transformer — the real model
is reached via the opt-in `python-sidecar` backend. The `evolve` module is the native
fitness/search loop that `@metaharness/darwin` would orchestrate externally; it runs
with zero extra dependencies here.

**Not yet built** (Phases 3–4, 6 of the roadmap): the Python sidecar backend, the
Ed25519 evidence-bundle wrapper for decoding runs, native MEG `.fif` ingest, and a
clean-room native deep decoder.

## References

**Brain2Qwerty / Meta AI**
- Blog: https://ai.meta.com/blog/brain2qwerty-brain-ai-human-communication/
- Code: https://github.com/facebookresearch/brain2qwerty
- V1 paper (arXiv:2502.17480): https://arxiv.org/abs/2502.17480 ·
  Nature Neuroscience: https://www.nature.com/articles/s41593-026-02303-2
- V2 paper: https://ai.meta.com/research/publications/accurate-decoding-of-natural-sentences-from-non-invasive-brain-recordings/
- Infra (`neuralset`/`neuraltrain`): https://github.com/facebookresearch/neuroai

**Dataset**
- SpanishBCBL (DECOMEG): https://huggingface.co/datasets/bcbl190626/SpanishBCBL

**Optimization tooling (ruvnet)**
- metaharness: https://www.npmjs.com/package/metaharness ·
  https://github.com/ruvnet/agent-harness-generator
- Darwin Mode: https://www.npmjs.com/package/@metaharness/darwin ·
  https://github.com/ruvnet/ruflo
- agenticow: https://www.npmjs.com/package/agenticow ·
  https://github.com/ruvnet/agenticow
- Darwin Gödel Machine background: https://sakana.ai/dgm/ ·
  https://arxiv.org/abs/2505.22954

**Related ruvnet ecosystem:** `ruv-swarm` (neural/WASM swarm, closest to ML),
`claude-flow`/`ruflo`, `agentic-flow`, `flow-nexus`.
