# ruv-neural-io

Bounded native recording adapters for NeuroSleep research ingestion.

`EdfEpochSource` streams one complete EDF/EDF+ data record at a time. It checks
source bytes, channel count, sample rates, duration, metadata, temporary
allocation, arithmetic overflow, numeric metadata, sample finiteness, truncation,
annotation count, chronology, and trailing bytes before returning atomic epochs.
It performs no writes. Allocation admission conservatively accounts for the raw
record, decoded `f64` samples, epoch/channel/annotation structs, and cloned EDF
labels; the fuzz target provides an additional bounded malformed-input surface.

The `brainvision-compat` default feature reexports the existing BrainVision API
without changing its path or behavior. That compatibility reader is not the
bounded NeuroSleep ingestion boundary.
