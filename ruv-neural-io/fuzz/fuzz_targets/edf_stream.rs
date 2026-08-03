#![no_main]

use libfuzzer_sys::fuzz_target;
use ruv_neural_io::{EdfEpochSource, EpochSource, IoLimits};
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    let limits = IoLimits {
        max_bytes: 1_048_576,
        max_channels: 16,
        max_sample_rate_hz: 4_096,
        max_duration_seconds: 86_400,
        max_data_records: 4_096,
        max_metadata_bytes: 65_536,
        max_epoch_allocation_bytes: 1_048_576,
        max_annotations_per_epoch: 128,
    };
    if let Ok(mut source) = EdfEpochSource::new(Cursor::new(data), limits) {
        while let Ok(Some(_)) = source.next_epoch() {}
    }
});
