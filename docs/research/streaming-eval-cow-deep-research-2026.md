# Deep-Research Report — Specs for the Next ruv-neural Increment

> **Status:** Adversarially verified deep-research sweep → **implemented** (see
> §5). · **Date:** 2026-08-09 · **Author:** rUv (ruv@ruv.net)
> **Method:** deep-research workflow — 5 search angles, 23 primary sources
> fetched, 115 falsifiable claims extracted, 25 claims put through 3-vote
> adversarial verification (23 confirmed, 2 killed), synthesized to 13
> findings. 105 agents, 759 tool calls. Numbers were re-verified against
> extracted PDF text after web-fetch summaries hallucinated table values twice.
> **Scope:** implementable specifications for (1) streaming neural
> speech/intent decoding, (2) EEG foundation-model evaluation gates,
> (3) copy-on-write branchable vector memory.

## 1. Streaming decoding (→ `ruv-neural-brain2text/src/stream.rs`)

**Verified findings:**

- **80 ms is a validated frame-hop cadence but a *