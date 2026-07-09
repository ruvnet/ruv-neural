# ADR-0015: ThinkGear Connector Ingestion and RuVector Memory Bridge

- **Status**: proposed
- **Date**: 2026-07-09
- **Deciders**:
- **Tags**: thinkgear, neurosky, eeg, ruvector, ingestion, memory

## Context

`ruv-neural` already has a clean separation between sensor acquisition,
signal processing, graph construction, embeddings, memory, decoding, and the
Ruflo closed-loop controller. The existing acquisition boundary is the
`SensorSource` trait in `ruv-neural-core`; simulated, EEG, OPM, NV diamond, and
ESP32 sources all feed `MultiChannelTimeSeries` into the same downstream
pipeline.

NeuroSky's ThinkGear Connector (TGC) is a host-local background process for
MindWave and compatible ThinkGear devices. TGC owns the Bluetooth/serial
device connection and exposes headset packets over a TCP socket. The documented
default endpoint is `127.0.0.1:13854`. Clients authorize with an application
name and 40-character hex application key, configure output with JSON such as
`{"enableRawOutput":true,"format":"Json"}`, then read carriage-return-delimited
JSON objects from a streaming socket.

TGC emits several packet classes:

- `rawEeg`: single-channel raw forehead EEG, up to 512 Hz when raw output is
  enabled.
- `rawEegMulti`: optional multi-channel raw EEG.
- `poorSignalLevel`: signal quality indicator where 0 is good and 200 is
  commonly off-head.
- `eSense`: low-rate attention and meditation scores.
- `eegPower`: low-rate band-power values.
- `blinkStrength`: event-like blink detection.

The ruv-neural topology pipeline expects at least two channels to construct a
connectivity graph. A standard MindWave raw stream may be single-channel, so a
direct feed into PLV graph construction is insufficient. At the same time, the
raw acquisition layer should remain honest: it should report the physical
stream shape instead of pretending that the headset has more channels.

The wider ruvnet ecosystem also uses RuVector for vector indexing, HNSW search,
and RVF-style interchange. `ruv-neural-memory` currently has its own in-memory
store and simplified HNSW implementation, while `ruv-neural-embed` has JSON RVF
export. A ThinkGear integration should make live embeddings available to
RuVector without forcing all users to depend on RuVector's storage, SIMD, or
network embedding-provider features.

## Decision

Add ThinkGear Connector support and direct ThinkGear binary support as opt-in
acquisition backends in `ruv-neural-sensor`, and add a feature-gated RuVector
memory bridge in `ruv-neural-memory`.

### Sensor Boundary

Create `ruv-neural-sensor::thinkgear` behind the `thinkgear` feature. It
contains:

- `ThinkGearConfig`: host, port, app name, app key, raw-output flag, sample
  rate, and read timeout.
- `ThinkGearSource`: a blocking TCP client that implements `SensorSource`.
- `ThinkGearStatus`: auxiliary state for `poorSignalLevel`, eSense,
  `blinkStrength`, and `eegPower`.

`ThinkGearSource` performs the TGC handshake, requests JSON raw output, ignores
non-JSON startup bytes, parses CR/LF-delimited JSON packets, filters non-finite
raw EEG samples, buffers raw values by channel, and returns fixed-size
`MultiChannelTimeSeries` chunks.

The source reports `SensorType::Eeg`. If TGC emits `rawEeg`, it reports one
channel. If TGC emits `rawEegMulti`, it reports the available multi-channel
stream. This keeps acquisition metadata truthful.

Create `ruv-neural-sensor::thinkgear_binary` behind the `mindwave-binary`
feature. This is the Rust equivalent of the protocol-parsing role served by
`akloster/python-mindwave`, but designed for ruv-neural:

- `ThinkGearBinaryParser` consumes one byte at a time and reconstructs packets
  framed by `0xAA 0xAA`, payload length, payload, and checksum.
- `ThinkGearBinarySource` owns a serial/RFCOMM byte stream and implements
  `SensorSource`.
- `open_serial(port, baud, sample_rate)` supports paired Bluetooth SPP devices
  exposed as `COMx` on Windows or `/dev/rfcomm*` on Linux.

The binary backend parses raw EEG (`0x80`), poor-signal (`0x02`), attention
(`0x04`), meditation (`0x05`), blink strength (`0x16`), and ASIC EEG power
(`0x83`) packets. Like the TGC backend, it exposes physical raw EEG as a
single-channel stream unless the hardware protocol provides more.

### Pipeline Boundary

Extend the CLI `pipeline` command with:

- `--source simulated|thinkgear`
- `--tgc-host`
- `--tgc-port`
- `--tgc-app-name`
- `--tgc-app-key` with `THINKGEAR_APP_KEY` environment fallback
- `--mindwave-port` with `MINDWAVE_PORT` environment fallback
- `--mindwave-baud`
- `--ruvector-index`

For `--source thinkgear`, the CLI reads a live raw chunk from TGC. For
`--source mindwave-binary`, it reads direct ThinkGear binary packets from the
configured serial/RFCOMM port. If the chunk has a single physical channel, the
CLI constructs a lag-derived virtual montage before graph construction. The
virtual montage is a downstream analysis view, not a sensor-level claim. It
exists only to let PLV/mincut/topology code operate on single-channel consumer
EEG while preserving the original source contract.

The virtual montage is intentionally simple and deterministic: channels are
lagged/scaled views of the single raw signal. It is suitable for pipeline
smoke-testing, live telemetry, and feature extraction experiments, but it is not
equivalent to a true multi-electrode montage. Any clinical or research analysis
that needs spatial claims must use a real multi-channel source.

### RuVector Boundary

Create `ruv-neural-memory::ruvector_store` behind the `ruvector` feature. It
wraps `ruvector-core::VectorDB` and exposes:

- `RuvectorMemoryStore::new(dimension, capacity)`
- `insert(&NeuralEmbedding) -> String`
- `search_ids(&NeuralEmbedding, k) -> Vec<(String, f32)>`
- `NeuralMemory` implementation for nearest-neighbor search

The bridge uses `ruvector-core` with `default-features = false`, matching the
repository pattern used by `homecore-recorder`. This gives in-memory vector
search without requiring file storage, API embeddings, AVX-512 defaults, or
other production RuVector features.

Embeddings are validated before indexing:

- dimension must match the store dimension
- timestamp must be finite
- vector values must be finite
- vector values must fit in `f32`

Metadata is copied into the RuVector entry as JSON values:

- timestamp
- embedding method
- source atlas
- subject ID
- session ID
- cognitive state

### Non-Decisions

This ADR does not make TGC a default source. TGC remains opt-in.

This ADR does not claim MindWave single-channel EEG can support true spatial
connectivity inference. The CLI virtual montage is an analysis compatibility
layer for existing graph tooling.

This ADR does not replace existing `NeuralMemoryStore`, `HnswIndex`, or RVF
export. RuVector is an additional bridge for ruvnet interoperability.

This ADR does not add long-term persistent RuVector storage. Persistence can be
added later by enabling and configuring RuVector storage features deliberately.

## Consequences

### Positive

- Real NeuroSky/MindWave EEG data can enter the ruv-neural pipeline through the
  same `SensorSource` boundary as simulated and other hardware sources.
- TGC serial/Bluetooth handling remains outside ruv-neural; ruv-neural only
  consumes a local TCP stream.
- The sensor crate stays honest about physical channel count.
- The CLI can run the full pipeline from either simulated data or live TGC
  data.
- Live topology embeddings can be inserted into RuVector for nearest-neighbor
  search and downstream ruvnet interoperability.
- RuVector remains feature-gated for library consumers.
- The bridge avoids non-finite and out-of-range embedding values before they
  cross into RuVector.

### Negative

- TGC is a local desktop dependency and must be installed/running separately.
- The authorization app key must be supplied by users or deployment scripts.
- Single-channel ThinkGear data needs a virtual montage for topology code,
  which can be misunderstood if not labeled clearly.
- The blocking TCP client is simple and suitable for CLI use, but a future GUI
  or long-running service may want an async source wrapper.
- The initial RuVector bridge reconstructs nearest-neighbor embeddings from
  returned vectors and query metadata; rich metadata-filtered reconstruction is
  deferred.

### Neutral

- `ruv-neural-sensor` gains a `thinkgear` feature.
- `ruv-neural-sensor` gains a `mindwave-binary` feature.
- `ruv-neural-memory` gains a `ruvector` feature.
- `ruv-neural-cli` enables both features so users can run the integrated path
  directly from the binary.
- The pipeline output is changed to ASCII-only terminal text while preserving
  the prior stage order and summaries.

## Safety and Privacy

ThinkGear data can be biometric. The implementation keeps TGC access local by
default and does not transmit raw EEG off-host. The app key is accepted through
CLI/env configuration and is not hardcoded with a real credential.

The CLI should treat `poorSignalLevel == 200` as an off-head or invalid-contact
state in future closed-loop integrations. This ADR's implementation captures
that status in `ThinkGearStatus`; closed-loop gating can consume it in a later
controller change.

No stimulation decision should be made solely from a low-cost single-channel
consumer EEG stream without a separate safety envelope. The existing
`ruv-neural-loop` safety model remains the control boundary.

## Verification

Minimum verification for this ADR:

- `cargo test -p ruv-neural-sensor --features thinkgear`
- `cargo test -p ruv-neural-sensor --features mindwave-binary`
- `cargo test -p ruv-neural-memory --features ruvector`
- `cargo test -p ruv-neural-cli`
- Manual live test with TGC running:

```powershell
$env:THINKGEAR_APP_KEY = "<40 hex chars>"
cargo run -p ruv-neural-cli -- pipeline --source thinkgear --duration 5 --ruvector-index
cargo run -p ruv-neural-cli -- pipeline --source mindwave-binary --mindwave-port COM5 --duration 5 --ruvector-index
```

## Links

- NeuroSky ThinkGear Connector documentation:
  https://developer.neurosky.com/docs/doku.php?id=thinkgear_connector_tgc
- NeuroSky ThinkGear Connector Development Guide:
  https://developer.neurosky.com/docs/doku.php?id=thinkgear_connector_development_guide
- NeuroSky ThinkGear Socket Protocol:
  https://developer.neurosky.com/docs/doku.php?id=thinkgear_socket_protocol
- RuVector repository:
  https://github.com/ruvnet/ruvector
- ruv-neural repository:
  https://github.com/ruvnet/ruv-neural
- python-mindwave repository:
  https://github.com/akloster/python-mindwave
