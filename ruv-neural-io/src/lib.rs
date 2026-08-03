//! Bounded, native EEG input adapters.
//!
//! The primary interface is [`EpochSource`]. Parsers produce one complete data
//! record at a time and never expose partially decoded output. Limits are
//! checked before allocation and all size arithmetic is checked.

pub mod edf;

pub use edf::{open_edf, EdfEpochSource};

#[cfg(feature = "brainvision-compat")]
/// Compatibility reexport of the existing BrainVision reader. This legacy API
/// is not the bounded NeuroSleep ingestion boundary and remains owned by
/// `ruv-neural-brain2text` until a dedicated bounded adapter replaces it.
pub mod brainvision {
    pub use ruv_neural_brain2text::dataset::brainvision::*;
}

use serde::{Deserialize, Serialize};
use std::io;
use thiserror::Error;

/// Explicit parser and allocation limits applied before output is produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IoLimits {
    /// Maximum bytes admitted for the complete source artifact.
    pub max_bytes: u64,
    /// Maximum signal count in the fixed header.
    pub max_channels: usize,
    /// Maximum per-channel sample rate.
    pub max_sample_rate_hz: u32,
    /// Maximum declared recording duration.
    pub max_duration_seconds: u64,
    /// Maximum number of EDF data records to iterate.
    pub max_data_records: u64,
    /// Maximum fixed-header and signal-metadata bytes.
    pub max_metadata_bytes: usize,
    /// Maximum temporary raw plus decoded allocation for one epoch.
    pub max_epoch_allocation_bytes: usize,
    /// Maximum EDF+ annotations emitted from one epoch.
    pub max_annotations_per_epoch: usize,
}

impl Default for IoLimits {
    fn default() -> Self {
        Self {
            max_bytes: 1_073_741_824,
            max_channels: 64,
            max_sample_rate_hz: 4_096,
            max_duration_seconds: 172_800,
            max_data_records: 1_000_000,
            max_metadata_bytes: 1_048_576,
            max_epoch_allocation_bytes: 16_777_216,
            max_annotations_per_epoch: 1_024,
        }
    }
}

impl IoLimits {
    /// Reject nonsensical policies before parsing untrusted bytes.
    pub fn validate(&self) -> Result<(), IoError> {
        if self.max_bytes == 0 {
            return Err(IoError::InvalidLimits("max_bytes must be positive"));
        }
        if self.max_channels == 0 {
            return Err(IoError::InvalidLimits("max_channels must be positive"));
        }
        if !(64..=4_096).contains(&self.max_sample_rate_hz) {
            return Err(IoError::InvalidLimits(
                "max_sample_rate_hz must be between 64 and 4096",
            ));
        }
        if self.max_duration_seconds == 0 {
            return Err(IoError::InvalidLimits(
                "max_duration_seconds must be positive",
            ));
        }
        if self.max_data_records == 0 {
            return Err(IoError::InvalidLimits("max_data_records must be positive"));
        }
        if self.max_metadata_bytes < 256 {
            return Err(IoError::InvalidLimits(
                "max_metadata_bytes must admit the EDF fixed header",
            ));
        }
        if self.max_epoch_allocation_bytes == 0 {
            return Err(IoError::InvalidLimits(
                "max_epoch_allocation_bytes must be positive",
            ));
        }
        if self.max_annotations_per_epoch == 0 {
            return Err(IoError::InvalidLimits(
                "max_annotations_per_epoch must be positive",
            ));
        }
        Ok(())
    }
}

/// Stable limit identifiers used in rejection reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitKind {
    SourceBytes,
    Channels,
    SampleRate,
    Duration,
    DataRecords,
    MetadataBytes,
    EpochAllocationBytes,
    Annotations,
}

/// Stable malformed-input reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectionReason {
    InvalidVersion,
    InvalidHeaderLength,
    InvalidSignalCount,
    InvalidRecordCount,
    UnknownRecordCount,
    InvalidRecordDuration,
    InvalidSamplesPerRecord,
    InvalidSampleRate,
    InvalidPhysicalRange,
    InvalidDigitalRange,
    InvalidNumericField,
    InvalidAnnotation,
    InvalidChronology,
    TrailingData,
}

/// Typed, deterministic ingestion failure.
#[derive(Debug, Error)]
pub enum IoError {
    #[error("invalid I/O limits: {0}")]
    InvalidLimits(&'static str),
    #[error("{limit:?} limit exceeded: actual {actual}, maximum {maximum}")]
    LimitExceeded {
        limit: LimitKind,
        actual: u64,
        maximum: u64,
    },
    #[error("malformed input: {0:?}")]
    Malformed(RejectionReason),
    #[error("truncated {section}: expected {expected} bytes, received {actual}")]
    Truncated {
        section: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("checked arithmetic overflow while computing {0}")]
    ArithmeticOverflow(&'static str),
    #[error("non-finite decoded sample at record {record}, channel {channel}, sample {sample}")]
    NonFiniteSample {
        record: u64,
        channel: usize,
        sample: usize,
    },
    #[error("I/O failure while reading {context}: {kind:?}")]
    Io {
        context: &'static str,
        kind: io::ErrorKind,
    },
    #[error("epoch source is terminal after a previous rejection")]
    SourceFailed,
}

impl IoError {
    /// Stable local-observability reason code containing no neural values.
    pub fn reason_code(&self) -> &'static str {
        match self {
            Self::InvalidLimits(_) => "invalid_limits",
            Self::LimitExceeded { .. } => "limit_exceeded",
            Self::Malformed(RejectionReason::TrailingData) => "trailing_data",
            Self::Malformed(_) => "malformed_input",
            Self::Truncated { .. } => "truncated_input",
            Self::ArithmeticOverflow(_) => "arithmetic_overflow",
            Self::NonFiniteSample { .. } => "nonfinite_sample",
            Self::Io { .. } => "io_failure",
            Self::SourceFailed => "source_failed",
        }
    }
}

/// Recording container recognized by the adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordingFormat {
    Edf,
    EdfPlusContinuous,
    EdfPlusDiscontinuous,
}

/// One signal's immutable acquisition metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalMetadata {
    pub label: String,
    pub transducer: String,
    pub physical_dimension: String,
    pub physical_minimum: f64,
    pub physical_maximum: f64,
    pub digital_minimum: i32,
    pub digital_maximum: i32,
    pub prefiltering: String,
    pub samples_per_record: usize,
    pub sample_rate_hz: f64,
    pub is_annotation: bool,
}

/// Immutable metadata available before reading the first epoch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordingMetadata {
    pub format: RecordingFormat,
    pub patient_id: String,
    pub recording_id: String,
    pub start_date: String,
    pub start_time: String,
    pub data_records: u64,
    pub record_duration_seconds: f64,
    pub total_duration_seconds: f64,
    pub signals: Vec<SignalMetadata>,
}

/// One completely decoded channel for a single data record.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochChannel {
    pub signal_index: usize,
    pub label: String,
    pub sample_rate_hz: f64,
    pub samples: Vec<f64>,
}

/// One EDF+ time-annotation-list item.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochAnnotation {
    pub onset_seconds: f64,
    pub duration_seconds: Option<f64>,
    pub text: String,
}

/// Atomic parser output. Implementations construct this only after every
/// channel and annotation in the record passes validation.
#[derive(Debug, Clone, PartialEq)]
pub struct Epoch {
    pub index: u64,
    pub start_offset_seconds: f64,
    pub duration_seconds: f64,
    pub channels: Vec<EpochChannel>,
    pub annotations: Vec<EpochAnnotation>,
}

/// Native pull-based source for bounded epoch processing.
pub trait EpochSource {
    fn metadata(&self) -> &RecordingMetadata;
    fn next_epoch(&mut self) -> Result<Option<Epoch>, IoError>;
}
