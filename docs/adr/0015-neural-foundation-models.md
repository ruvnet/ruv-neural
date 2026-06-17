# ADR-0015 — Neural foundation-model embeddings as an optional, pluggable backend

## Status

Proposed — research direction; backend **seam implemented**. The default
embedding path stays the lightweight, dependency-free methods already in
`ruv-neural-embed`; foundation-model embeddings are an *opt-in* backend behind
the `FoundationEmbedder` trait (`ruv-neural-embed/src/foundation.rs`), which
produces a standard method-tagged `NeuralEmbedding`. A dependency-free
`ReferenceFoundationEmbedder` exercises the seam end-to-end; real model backends
(LaBraM/REVE via ONNX, behind the `fm` feature) remain **Proposed** until they
beat the lightweight baselines out-of-sample (point 4).

## Context

Self-supervised "foundation models" (FMs) for EEG/biosignals are the dominant
2023–2026 research thrust. The landscape:

- **Established models:** BENDR (contrastive, TUEG); BrainBERT; LaBraM (*ICLR*
  2024 spotlight; neural-codebook tokenizer + masked modeling; ~2,500 h
  pretraining); EEGPT; Brant/Brant-2; CBraMod; **NeuroLM** (*ICLR* 2025;
  text-aligned tokenizer + LLM, ~1.7B params, ~25,000 h — treats EEG as a
  "foreign language").
- **2025–2026 scale leaders:** **REVE** (~60,000 h, 92 datasets, 25,000
  subjects — claimed largest EEG pretraining; masked-autoencoder with positional
  encoding for arbitrary montages); BrainPro (brain-state-aware). Pretraining
  still leans heavily on the **TUH EEG corpus (~21,600 h)**.
- **The sobering findings (critical literature):** the field is **data-starved
  relative to NLP/vision** (thousands of hours vs internet-scale text); has **no
  standardized benchmark** (EEG-FM-Bench, AdaBrain-Bench, EEG-Bench all measure
  different things); suffers **in-sample evaluation** (downstream sets reused in
  pretraining); and, most importantly, **EEG-FMs do not clearly follow scaling
  laws** — *compact, domain-specific models repeatedly match or beat much larger
  FMs*, simple baselines (e.g. LDA) stay competitive under clinical distribution
  shift, and general-purpose time-series FMs (MOMENT, TimesNet) sometimes win.
- **Edge & generalization reality:** full FMs (100M–1B params, e.g. LaBraM
  ranges 5.8M–369M; NeuroLM ~1.7B) are not MCU-deployable as-is — edge use needs
  tiny variants plus quantization-*aware* training (FEMBA-Tiny ~7.8M at ~2 MB,
  ~3× real-time on a RISC-V MCU; naïve post-training quantization can drop ~30%
  accuracy). Cross-dataset EEG accuracy can also collapse toward chance without
  per-subject alignment — domain shift is unsolved even for FMs.

This project's embeddings are deliberately small and transparent (spectral,
topology, node2vec, combined, temporal → `NeuralEmbedding`; and the 9-dim
ruVector, ADR-0006). FMs offer richer representations but are heavy, often
edge-infeasible, and — per the evidence above — not a guaranteed win.

## Decision

1. **Keep the lightweight, deterministic embeddings as the default.** They are
   edge/WASM-friendly, auditable, and competitive — exactly what the critical
   literature recommends, and what the closed-loop controller (ADR-0003) and
   ESP32/WASM targets require.
2. **Add FM embeddings as an optional backend** behind the existing `Embedder`
   trait, producing a standard `NeuralEmbedding` tagged with its method (e.g.
   `foundation:labram`, `foundation:reve`). This keeps downstream code
   (distance, RVF export, the controller) modality- and method-agnostic.
3. **Inference-only, behind a feature flag.** No training in-tree; the project
   consumes exported model outputs/weights through an adapter. FM dependencies
   are gated so the core stays dependency-light.
4. **Benchmark honestly before promotion.** An FM backend graduates from
   Proposed only with **out-of-sample** evaluation against the lightweight
   baselines on the project's own pipeline — no in-sample or cherry-picked
   numbers, consistent with ADR-0019 claims discipline.
5. **Prefer compact, open-licensed models** with permissive licenses; record the
   license per backend.

## Consequences

- The project can ride the FM wave without betting its core on it, and without
  breaking edge deployability.
- Method-tagged `NeuralEmbedding` means an FM backend is a drop-in for analysis
  and storage; the controller is unaffected.
- We commit to the discipline that an FM must *beat the baseline out-of-sample*
  to ship — protecting against the field's documented evaluation pitfalls.

## Evidence

- `ruv-neural-embed/src/lib.rs` — the `Embedder` interface and the existing
  methods (spectral/topology/node2vec/combined/temporal) an FM backend slots
  beside.
- `ruv-neural-core/src/embedding.rs` — `NeuralEmbedding` + `EmbeddingMetadata`
  carry the method tag a foundation backend would set.
- `docs/adr/0006-personal-state-embedding.md` — ruVector exports via the same
  `NeuralEmbedding` seam (`method = personal-state-fusion`).

## References

1. Jiang et al., "LaBraM: Large Brain Model," *ICLR* 2024 — arxiv.org/abs/2405.18765
2. Kostas et al., "BENDR," *Front. Hum. Neurosci.* 2021 — frontiersin.org/articles/10.3389/fnhum.2021.653659
3. Jiang et al., "NeuroLM: bridging language and EEG," *ICLR* 2025 — arxiv.org/abs/2409.00101
4. El Ouahidi et al., "REVE: a foundation model for EEG (25,000 subjects)," 2025 — arxiv.org/abs/2510.21585
5. Kuruppu et al., "EEG Foundation Models: A Critical Review," *J. Neural Eng.* 2025 (arXiv:2507.11783) — arxiv.org/abs/2507.11783
6. Xiong et al., "EEG-FM-Bench" (compact > large finding), 2025 — arxiv.org/abs/2508.17742
7. Kastrati et al., "EEG-Bench: clinical applications" (simple baselines competitive), *NeurIPS* 2025 — arxiv.org/abs/2512.08959
8. Temple University Hospital EEG Corpus (~21,600 h) — isip.piconepress.com/projects/tuh_eeg
9. FEMBA — EEG foundation model on a microcontroller (QAT, edge feasibility), 2026 — arxiv.org/html/2603.26716
10. Xu et al., cross-dataset EEG generalization collapse, *Front. Hum. Neurosci.* 2020 — pmc.ncbi.nlm.nih.gov/articles/PMC7188358
