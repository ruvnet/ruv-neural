# ADR-0023 — RuVector as the embedding store, ANN retrieval & inference backend

## Status

Proposed — integration direction. The RVF export seam exists today
(`ruv-neural-core`, `ruv-neural-embed`); adopting RuVector as the downstream
store/search/inference layer is a roadmap, gated on the format reconciliation
below. Does **not** alter the ADR-0001 wellness scope — RuVector is
infrastructure, not a new sensing or stimulation capability.

## Context

ruv-neural already *produces* vectors: `NeuralEmbedding` (ADR-0006/0015) and
brain-graph/topology snapshots are serialized to the **RVF (RuVector File)**
format expressly "for interoperability with the RuVector ecosystem." What the
project lacks is the *consumer* side — durable vector storage, approximate
nearest-neighbour (ANN) retrieval, and a hosted inference runtime. That is
exactly what **`ruvnet/ruvector`** provides (MIT-licensed Rust + WASM):

- **Vector DB & ANN:** HNSW index with a GNN re-ranking layer; metrics include
  cosine, Euclidean, dot product, and (notably for connectivity work) hyperbolic
  / Poincaré and Wasserstein/Sinkhorn distances; sub-ms query latency.
- **Quantization:** f16, PQ8/PQ4, binary, INT8 — directly relevant to the
  edge-storage reality ADR-0015 leans on.
- **RVF as a binary substrate:** the upstream RVF is far richer than this repo's
  current JSON profile — a container with `VEC`, `INDEX`, `WASM`, `WITNESS`,
  `CRYPTO`, and `FEDERATED_MANIFEST` segments, **post-quantum + Ed25519
  signatures**, copy-on-write branching, and tamper-evident witness chains.
- **Local/edge:** a ~5.5 KB WASM runtime for `.rvf` queries and 8+ embedded
  ONNX embedding models — local, no external API.

These capabilities line up with decisions already made here, which is the reason
to record the integration rather than improvise it.

## Decision

Adopt RuVector as the **optional downstream backend** for embeddings, behind a
feature flag, mapping its capabilities onto existing ADRs:

1. **Store & retrieve embeddings.** `NeuralEmbedding` / ruVector (ADR-0006) and
   any foundation-model embeddings (ADR-0015) are stored in RuVector's `VEC`
   segment and queried via HNSW ANN — enabling cross-session, cross-target
   similarity search ("right person, right time"). The hyperbolic/Wasserstein
   metrics suit connectivity/topology vectors specifically.
2. **Reconcile the RVF format (prerequisite).** This repo's RVF is a **JSON
   profile, version 1**, with five `RvfDataType`s; upstream RVF is a **binary,
   25-segment container**. Treat the current JSON form as a debug/interchange
   profile and define a documented mapping from `NeuralEmbedding` → upstream
   `VEC` (and topology/mincut snapshots → appropriate segments). **Do not** let
   the two "RVF"s silently diverge under the same acronym.
3. **Unify the audit story.** ADR-0009's hash-chained, Ed25519-signable audit and
   ADR-0014's signed evidence export map onto RuVector's `WITNESS`/`CRYPTO`
   segments; prefer reusing that substrate (incl. post-quantum signatures) over
   maintaining a parallel mechanism.
4. **Federation & edge reuse.** RuVector's `FEDERATED_MANIFEST` is the concrete
   substrate for ADR-0021's federated-personalization roadmap; its WASM runtime
   complements ADR-0014's local-first web console.
5. **Inference & interop.** RuVector's ONNX runtime is a candidate host for the
   ADR-0015 foundation-model backend and an additional interop bridge alongside
   NWB/LSL (ADR-0016).
6. **Investigate mincut consolidation.** Both projects implement mincut
   (`ruv-neural-mincut`; RuVector's mincut-gated transformer) — evaluate shared
   code rather than two implementations.

## Consequences

- ruv-neural gets a production-grade store/search/inference layer "for free,"
  permissively licensed, instead of growing its own.
- The audit, federation, and edge decisions converge on one substrate, reducing
  parallel mechanisms.
- **Coupling & maturity risk:** RuVector is a separate, currently out-of-scope
  repository; its capabilities here are sourced from its README (an ambitious
  one — e.g. "a `.rvf` file boots a Linux kernel in 125 ms"). The integration
  must be gated behind a feature flag and validated against the actual upstream
  before any hard dependency, and the RVF format divergence (point 2) resolved
  first.

## Evidence

- `ruv-neural-core/src/rvf.rs` — `RVF_MAGIC = [R,V,F,0x01]`, `RVF_VERSION = 1`,
  and `RvfDataType { BrainGraph, NeuralEmbedding, TopologyMetrics, MincutResult,
  TimeSeriesChunk }` — this repo's current RVF profile.
- `ruv-neural-embed/src/rvf_export.rs` — `RvfDocument`/`RvfHeader`/`RvfRecord`
  round-trip; the export seam an upstream `VEC` mapping would extend.
- `ruv-neural-embed/src/distance.rs` — cosine/euclidean/manhattan, the local
  analogue of RuVector's metric set.
- `docs/adr/0006-personal-state-embedding.md`, `0009-audit-trail.md`,
  `0014-web-console.md`, `0015-neural-foundation-models.md`,
  `0021-privacy-preserving-personalization.md` — the decisions this backend
  unifies.

## References

1. RuVector — vector DB + GNN, RVF format, WASM/ONNX (MIT) — github.com/ruvnet/ruvector
2. HNSW — Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using HNSW graphs," *IEEE TPAMI* 2018 — arxiv.org/abs/1603.09320
3. Product Quantization — Jégou et al., "Product quantization for nearest neighbor search," *IEEE TPAMI* 2011 — inria.hal.science/inria-00514462
4. Poincaré embeddings (hyperbolic representation) — Nickel & Kiela, *NeurIPS* 2017 — arxiv.org/abs/1705.08039
