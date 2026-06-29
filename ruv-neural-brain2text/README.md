# ruv-neural-brain2text

Non-invasive **brain-to-text** decoding bridge for the rUv Neural workspace,
inspired by Meta AI's [Brain2Qwerty](https://ai.meta.com/blog/brain2qwerty-brain-ai-human-communication/)
and the [SpanishBCBL (DECOMEG)](https://huggingface.co/datasets/bcbl190626/SpanishBCBL)
MEG/EEG dataset — with an **in-crate evolutionary optimizer** ("Darwin mode":
*freeze the model, evolve the harness*).

> Research harness, not a medical device. No efficacy or clinical claim.
> See [`docs/research/brain2qwerty-integration.md`](../docs/research/brain2qwerty-integration.md)
> for the full design study, licensing notes (the upstream model and dataset are
> **CC BY-NC 4.0**), and the MEG-dependence caveat.

## Pipeline

```text
Recording (MEG/EEG signal + keystroke timeline)
  -> preprocess  bandpass 0.1-20 Hz, resample 50 Hz        (mirrors Brain2Qwerty V1)
  -> epoch       -0.2..+0.3 s window, baseline, features
  -> decode      prototype acoustic model + char n-gram LM, beam search
  -> metrics     CER / WER (Levenshtein)
```

`evaluate()` ties it together; `evolve::evolve()` searches the
`Brain2TextConfig` space with a genetic algorithm whose fitness is
`1 - validation_CER`.

## What's real vs. stand-in

| Piece | Status |
|---|---|
| BrainVision `.vhdr/.vmrk/.eeg` reader | **real**, native, zero-dep (`dataset::brainvision`) |
| V1 preprocessing (bandpass/resample/baseline) | **real**, on `ruv-neural-signal` |
| Keystroke epoching, CER/WER metrics, n-gram LM, beam search | **real**, native |
| Evolutionary optimizer (Darwin mode) | **real**, native |
| Acoustic model | **prototype/nearest-centroid stand-in** for the deep Conv+Transformer (which is PyTorch + CC BY-NC) — swap in the `python-sidecar` backend for the real model |
| MEG `.fif` loading | requires an external `mne` export step (documented) |
| Test/demo data | **synthetic** SpanishBCBL-like generator (real data is ~262 GB, CC BY-NC, not in repo) |

## Demo

```bash
cargo run -p ruv-neural-brain2text --example brain2text_demo
```

Generates synthetic data, evaluates the Brain2Qwerty V1 default config, then
evolves it. Typical run (synthetic, learnable data):

```text
baseline (V1 defaults):   test CER = 0.594  WER = 1.167
evolved config:           test CER = 0.025  WER = 0.167
```

The optimizer recovers a configuration that decodes the held-out test split
almost perfectly — demonstrating the implement → test → optimize → evolve loop
end to end. (Numbers are on synthetic data; they validate the machinery, not the
published Brain2Qwerty accuracy.)

## Relationship to the ruvnet optimization tooling

The `evolve` module is the native, dependency-free fitness/search loop that the
external [`@metaharness/darwin`](https://www.npmjs.com/package/@metaharness/darwin)
tooling would orchestrate (`metaharness` scaffolds the agent; `agenticow`
versions per-experiment vector memory). See the research report for how they map
onto this crate.

## License

MIT OR Apache-2.0 (this crate). The upstream Brain2Qwerty code and SpanishBCBL
data are **CC BY-NC 4.0** and are deliberately **not** vendored here.
