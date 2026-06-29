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
  -> model       trainable acoustic model: Prototype | Linear | MLP
  -> decode      + char n-gram LM, beam search (score = acoustic + alpha*LM)
  -> metrics     CER / WER (Levenshtein)
```

`evaluate()` ties it together; the **composable `Harness`** builder fits a
self-contained, serializable `TrainedPipeline` (config + trained weights + LM)
you can `to_json()` / `from_json()` — that artifact *is* the distributable model.
`evolve::evolve()` ("Darwin mode") searches the whole `Brain2TextConfig` space —
**including the model family and training hyperparameters** — with a genetic
algorithm whose fitness is `1 - validation_CER`.

```rust
use ruv_neural_brain2text::{harness::Harness, Brain2TextConfig, EvalSplit};
let pipeline = Harness::new()
    .with_config(Brain2TextConfig::default())
    .with_split(0.7, 0.15)
    .fit(&recording)?;                       // -> TrainedPipeline (serializable)
let json = pipeline.to_json()?;              // distributable artifact
let report = pipeline.evaluate(&recording, EvalSplit::Test)?;
```

## Trainable models

Three native, dependency-free, **serializable** acoustic models (the weights are
the artifact), all implementing `AcousticModel`:

| Model | Training | Notes |
|---|---|---|
| `Prototype` | nearest-centroid | fast baseline, no gradient |
| `Linear` | multinomial logistic regression (SGD) | default; robust |
| `Mlp` | 1 hidden layer ReLU + softmax (SGD/backprop) | nonlinear capacity |

The deep Conv+Transformer remains the opt-in `python-sidecar` (CC BY-NC).

## What's real vs. stand-in

| Piece | Status |
|---|---|
| BrainVision `.vhdr/.vmrk/.eeg` reader | **real**, native, zero-dep (`dataset::brainvision`) |
| V1 preprocessing (bandpass/resample/baseline) | **real**, on `ruv-neural-signal` |
| Keystroke epoching, CER/WER metrics, n-gram LM, beam search | **real**, native |
| Evolutionary optimizer (Darwin mode) | **real**, native |
| Trainable acoustic models (Prototype / Linear / MLP) | **real**, native SGD, serializable weights — clean-room stand-ins for the deep Conv+Transformer (PyTorch + CC BY-NC); swap in the `python-sidecar` for that |
| Composable harness + serializable `TrainedPipeline` | **real**, native (`harness`) |
| Evolutionary optimizer over model + hyperparams | **real**, native (`evolve`) |
| MEG `.fif` loading | requires an external `mne` export step (documented) |
| Test/demo data | **synthetic** SpanishBCBL-like generator (real data is ~262 GB, CC BY-NC, not in repo) |

## Demo & benchmark

```bash
cargo run -p ruv-neural-brain2text --example brain2text_demo        # evolve loop
cargo run --release -p ruv-neural-brain2text --example benchmark    # model matrix + timing
cargo run -p ruv-neural-brain2text --example train -- out/model.json  # export artifact
cargo bench -p ruv-neural-brain2text                                # criterion micro-benches
```

Representative benchmark (synthetic, learnable data — validates the machinery and
the optimizer, **not** the published Brain2Qwerty accuracy):

```text
1) Per-model, V1 default config (test split)
   model            CER      WER    train(ms)
   Prototype      0.596    1.167        ...
   Linear         0.188    0.611        ...
   Mlp            0.921    1.222        ...      # overfits tiny data; optimizer avoids it
2) Evolved best (Darwin mode, across model families)
   model=Linear  ->  test CER=0.037  WER=0.194
3) ~7,000 keystrokes/sec decoded
```

The optimizer searches across model families *and* hyperparameters, selects
Linear, and cuts test CER from 0.188 to ~0.037 — the implement → test → optimize
→ evolve loop, end to end.

## Relationship to the ruvnet optimization tooling

The `evolve` module is the native, dependency-free fitness/search loop that the
external [`@metaharness/darwin`](https://www.npmjs.com/package/@metaharness/darwin)
tooling would orchestrate (`metaharness` scaffolds the agent; `agenticow`
versions per-experiment vector memory). See the research report for how they map
onto this crate.

## License

MIT OR Apache-2.0 (this crate). The upstream Brain2Qwerty code and SpanishBCBL
data are **CC BY-NC 4.0** and are deliberately **not** vendored here.
