# Deep-Research Report — Specs for the Next ruv-neural Increment

> **Status:** Adversarially verified deep-research sweep → **implemented**
> (see [§5](#5-implementation-status)).
> **Date:** 2026-08-09 · **Author:** rUv (ruv@ruv.net)
> **Method:** deep-research workflow — 5 search angles, **23 primary sources**
> fetched, **115 falsifiable claims** extracted, 25 put through **3-vote
> adversarial verification** (23 confirmed, **2 killed**), synthesized to 13
> findings. 105 agents, 759 tool calls. Numbers were re-verified against
> extracted PDF text after web-fetch summaries hallucinated table values twice.
> **Scope:** implementable specifications for (1) streaming neural
> speech/intent decoding, (2) EEG foundation-model evaluation gates, and
> (3) copy-on-write branchable vector memory — the three items planned in
> [`brain-ai-sota-2026.md`](brain-ai-sota-2026.md) §9 (rows 5, 1, 17).

---

## 0. What survived, and what did not

| | Verdict |
|---|---|
| **Streaming decoding** | Well-sourced. Two peer-reviewed *Nature*-family papers with public code and data pin the cadence, causality constraints, and latency budgets. |
| **Evaluation methodology** | Well-sourced but unrefereed. A mid-2026 arXiv benchmark supplies a transcribable negative-control protocol; its key numbers were verified from extracted PDF text. |
| **CoW vector memory** | **Unsupported.** Zero claims survived verification. The only material is vendor self-published, with both sources sharing one author — not independent corroboration. |

**Two claims were refuted (0-3) and must not propagate:**

1. That the 80-ms streaming brain-to-voice system uses an **RNN-T**. The
   architecture behind that increment is *not* publicly established. Any spec
   citing "RNN-T at 80 ms" is repeating an unverified attribution.
2. An EEG-to-mel **Pearson r = 0.069** SNR-ceiling figure for SparrKULee. The
   argument for preferring keyword-spotting formulations still stands, but on
   the chance-level cross-subject results below — not on this number.

---

## 1. Streaming decoding (→ `ruv-neural-brain2text/src/stream.rs`)

**80 ms is a validated cadence, and a *soft* budget — not a floor.** The
streaming brain-to-voice system decodes in 80-ms increments and meets that
per-step budget in **99.3% of steps** (99% CI [97.60, 99.73]); streaming
roughly doubles throughput over delayed decoding (**46.5 vs 24.8 WPM**,
non-overlapping CIs). Caveat carried into the implementation: that latency
distribution was measured by **offline replay**, not live logging, and the GPU
is unstated — so an edge implementation must *report* overruns, not assume
`RTF < 1`.
— Littlejohn, Cho et al., *Nat. Neurosci.* 28:902–912 (2025),
[10.1038/s41593-025-01905-6](https://www.nature.com/articles/s41593-025-01905-6)

**A causal decoder runs an order of magnitude faster.** A fully causal
Transformer produces speech at a **10-ms hop** with ~10 ms neural-to-sample
inference (~25 ms end-to-end including audio playback). Its front end is the
concrete constraint to copy: 10-ms non-overlapping bins, features **causally
smoothed with a 1.5 s past-only sigmoid kernel**, a 600-ms sliding input
window, "causally normalized and smoothed on a rolling basis before being
decoded."
— Wairagkar et al., *Nature* 644:145–152 (2025),
[10.1038/s41586-025-09127-3](https://www.nature.com/articles/s41586-025-09127-3);
code [Neuroprosthetics-Lab/brain-to-voice-2025](https://github.com/Neuroprosthetics-Lab/brain-to-voice-2025)

**The 2026 direction for non-invasive streaming is a causal SSM with a
persistent hidden state** — 62.5-ms patches, strictly left-to-right, no
sliding-window recomputation, linear time complexity.
— CaMBRAIN, [arXiv:2605.28792](https://arxiv.org/abs/2605.28792) *(unrefereed
preprint; medium confidence)*

**Porting constraint for SSM keyword spotters:** the headline 98.01% Keyword
Mamba result is **bidirectional**; the causal ablation collapses to **78.29%**.
A streaming port cannot inherit the bidirectional number.
— [arXiv:2508.07363](https://arxiv.org/abs/2508.07363) (Table 4, verified from
PDF text after a fetch summary reported different values)

**Keyword spotting is the implementable low-SNR formulation.** Event-referenced
binary detection over word onsets: 306 channels at 250 Hz, windows around the
onset (the offset sweep peaks at ~0.1 s pre / 0.3 s post), session-disjoint
splits. Held-out test set: n=4660 with **24 positives (base rate 0.00515)**.
**AUPRC is the primary metric**, reported against a permutation null, with
false-alarms-per-hour as the operational complement — a raw AUPRC of 0.069
looks like failure but is **13.4× chance** at that base rate. Thresholds must
be chosen on validation, never on the reported test set.
— LibriBrain KWS, [arXiv:2510.21038](https://arxiv.org/pdf/2510.21038)

> **Modality caveat, load-bearing:** every latency and accuracy number above
> comes from **invasive** recordings (ECoG, 256-electrode Utah arrays; n=1) or
> MEG. The *cadence and causal structure* transfer to an edge EEG pipeline.
> The SNR and accuracy do not.

## 2. Evaluation gates (→ `ruv-neural-embed/src/promotion.rs`)

A directly transcribable negative-control protocol, verified verbatim from the
source: baselines "are trained from random initialisation, whereas the
foundation models are fine-tuned from their public pre-trained checkpoints";
metrics are "Balanced Accuracy, Cohen's Kappa, and Weighted F1 … as mean±SD
across splits/seeds"; significance by **Wilcoxon signed-rank**; LOSO defined so
"no trial from the test subject is ever seen during training"; and an explicit
leakage check that the evaluation set "is absent from the pre-training data."

The empirical result that motivates a *strict* gate: a **16K-parameter EEGNet
reaches 61.3%**, beating LaBraM (5.8M, 57.3%) and EEGMamba (8.3M, 56.6%) on
3-class speech-mode LOSO. And under LOSO on BCIC2020-T3 **all five models sat
at chance** (balanced accuracy 0.189–0.202 against 0.20 chance, κ ≈ 0, all
p > 0.18).
— Khanday et al., [arXiv:2607.27268](https://arxiv.org/html/2607.27268)

Corroborating negative-control literature:
[arXiv:2607.24519](https://arxiv.org/abs/2607.24519) (random-init comparators,
dataset-identity leakage), [arXiv:2607.24834](https://arxiv.org/html/2607.24834v1)
(no FM recovers long-range temporal correlations),
[arXiv:2606.06647](https://arxiv.org/abs/2606.06647) ("Identity Trap" audit).

**Two gaps the sources leave open, filled by us and labelled as ours:**

- **No verified source specifies a dataset-identity probe.** The motivation is
  replicated, the protocol is not published. Our implementation
  (`dataset_identity_probe`) is a deterministic leave-one-out nearest-centroid
  probe — ruv-neural's own design, not a citation.
- **No source specifies a minimum seed count** beyond LibriBrain's 3, which is
  thin. We set 6, for a reason discovered during implementation rather than
  read from a paper: under the **exact** Wilcoxon signed-rank test the smallest
  attainable two-sided p-value is `2/2ⁿ`, so n = 5 tops out at **0.0625** and
  can never reach α = 0.05 however large the effect.

## 3. Copy-on-write branchable vector memory (→ `ruv-neural-memory/src/branch.rs`)

**Zero claims survived adversarial verification.** The retrievable material is
vendor self-published — the RuVector RVF README and the `agenticow` explainer —
and both share a single author (@ruvnet), so they are one design expressed
twice, not independent corroboration. Their stated numbers (2.6 ms branch on
10K vectors, 162-byte child files, 1348 vs 1442 ns/vector local vs inherited
reads) have **no independent benchmark, no peer review, no replication**, and
are therefore **not cited as support** anywhere in our implementation.

Undocumented in those sources — and load-bearing for any implementation:
merge/conflict semantics, garbage collection of unreferenced clusters, and
read-through behaviour at **branch depth > 1**.

The underlying pattern is nonetheless conventional and sound (cluster-granularity
CoW with refcounts is the ZFS/btrfs/overlayfs approach), and the adjacent
versioned-data-format space is real — Lance's
[branching and shallow clone](https://www.lancedb.com/blog/branching-and-shallow-clone)
and its [format](https://github.com/lance-format/lance) are the closest
comparable systems surfaced, though neither is a CoW-branched *embedding* store
of the kind specified.

**Consequence for the implementation:** build only the well-understood part —
overlay deltas with read-through resolution — define the semantics ourselves,
test them (including at depth > 1, which no source defines), and leave merge
and GC explicitly out of scope.

## 4. Open questions

- What *is* the architecture behind the 80-ms streaming increment? (RNN-T
  refuted; unknown.)
- Does a dataset-identity probe belong in a published protocol, and what form?
- What is the causal-port penalty for a streaming SSM keyword spotter once the
  mid-sequence class-token confound is removed?
- Are there any independently benchmarked CoW-branched embedding stores outside
  the ruvnet ecosystem?
- How does RVF read-through degrade at branch depth > 1, and what are its merge
  and GC semantics?

## 5. Implementation status

| Spec | Module | Tests |
|---|---|---|
| Streaming decoder, causal, gated; KWS windows + AUPRC/null/FA-h | `ruv-neural-brain2text/src/stream.rs` | 11 |
| Promotion gates: random-init comparator, LOSO, leakage, Wilcoxon, AUPRC lift, identity probe | `ruv-neural-embed/src/promotion.rs` | 13 |
| CoW branch store: deltas, tombstones, deep read-through, checkpoint/rollback | `ruv-neural-memory/src/branch.rs` | 13 |

Benchmarks (`ruv-neural-brain2text/benches/stream.rs`): **66.9 µs per frame at
306 channels** — a **1,196× margin** against the 80-ms budget — and **29.4 µs**
at the 10-ms cadence (340× margin), on x86. Per the caveat in §1, these are
reported as measured margins, not as a guarantee that overruns cannot occur.

## 6. Process note

Web-fetch summarization **hallucinated table values twice** during
verification: a fabricated "9.4% PER" with a fabricated supporting quote, and
wrong Table 4 numbers. Both were caught only by direct `pdftotext` extraction.
Any number carried into a ruv-neural spec should be re-checked against
extracted PDF text, never against a fetch summary.
