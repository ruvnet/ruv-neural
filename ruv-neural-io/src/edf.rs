//! Streaming EDF and EDF+ adapter.

use crate::{
    Epoch, EpochAnnotation, EpochChannel, EpochSource, IoError, IoLimits, LimitKind,
    RecordingFormat, RecordingMetadata, RejectionReason, SignalMetadata,
};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

const FIXED_HEADER_BYTES: usize = 256;
const SIGNAL_HEADER_BYTES: usize = 256;
const DIGITAL_SAMPLE_BYTES: usize = 2;

#[derive(Debug, Clone)]
struct SignalDecoder {
    metadata: SignalMetadata,
    scale: f64,
    offset: f64,
}

/// Bounded, sequential EDF source. One complete EDF data record is one epoch.
pub struct EdfEpochSource<R> {
    reader: R,
    limits: IoLimits,
    metadata: RecordingMetadata,
    decoders: Vec<SignalDecoder>,
    record_bytes: usize,
    records_read: u64,
    bytes_read: u64,
    last_epoch_start: Option<f64>,
    terminal_checked: bool,
    failed: bool,
}

/// Open an EDF/EDF+ path read-only after checking its file size. The parser
/// never derives an output path and performs no writes.
pub fn open_edf(path: impl AsRef<Path>, limits: IoLimits) -> Result<EdfEpochSource<File>, IoError> {
    limits.validate()?;
    let file = File::open(path.as_ref()).map_err(|error| IoError::Io {
        context: "source file",
        kind: error.kind(),
    })?;
    let length = file
        .metadata()
        .map_err(|error| IoError::Io {
            context: "source metadata",
            kind: error.kind(),
        })?
        .len();
    check_limit(LimitKind::SourceBytes, length, limits.max_bytes)?;
    EdfEpochSource::with_known_length(file, limits, Some(length))
}

impl<R: Read> EdfEpochSource<R> {
    /// Parse a header from a stream. Declared total bytes are bounded even when
    /// the underlying stream length is unavailable.
    pub fn new(reader: R, limits: IoLimits) -> Result<Self, IoError> {
        Self::with_known_length(reader, limits, None)
    }

    fn with_known_length(
        mut reader: R,
        limits: IoLimits,
        known_length: Option<u64>,
    ) -> Result<Self, IoError> {
        limits.validate()?;
        check_limit(
            LimitKind::SourceBytes,
            FIXED_HEADER_BYTES as u64,
            limits.max_bytes,
        )?;
        let fixed = read_exact(&mut reader, FIXED_HEADER_BYTES, "fixed header")?;
        let version = text(&fixed[0..8]);
        if version != "0" {
            return Err(IoError::Malformed(RejectionReason::InvalidVersion));
        }
        let header_bytes = parse_usize(&fixed[184..192])?;
        let signal_count = parse_usize(&fixed[252..256])?;
        if signal_count == 0 {
            return Err(IoError::Malformed(RejectionReason::InvalidSignalCount));
        }
        check_limit(
            LimitKind::Channels,
            to_u64(signal_count, "signal count")?,
            to_u64(limits.max_channels, "channel limit")?,
        )?;
        let signal_header_bytes = signal_count
            .checked_mul(SIGNAL_HEADER_BYTES)
            .ok_or(IoError::ArithmeticOverflow("signal header bytes"))?;
        let expected_header_bytes = FIXED_HEADER_BYTES
            .checked_add(signal_header_bytes)
            .ok_or(IoError::ArithmeticOverflow("total header bytes"))?;
        if header_bytes != expected_header_bytes {
            return Err(IoError::Malformed(RejectionReason::InvalidHeaderLength));
        }
        check_limit(
            LimitKind::MetadataBytes,
            to_u64(header_bytes, "header bytes")?,
            to_u64(limits.max_metadata_bytes, "metadata limit")?,
        )?;
        check_limit(
            LimitKind::SourceBytes,
            to_u64(header_bytes, "header bytes")?,
            limits.max_bytes,
        )?;

        let records_field = text(&fixed[236..244]);
        if records_field == "-1" {
            return Err(IoError::Malformed(RejectionReason::UnknownRecordCount));
        }
        let data_records = records_field
            .parse::<u64>()
            .map_err(|_| IoError::Malformed(RejectionReason::InvalidRecordCount))?;
        if data_records == 0 {
            return Err(IoError::Malformed(RejectionReason::InvalidRecordCount));
        }
        check_limit(
            LimitKind::DataRecords,
            data_records,
            limits.max_data_records,
        )?;
        let record_duration_seconds = parse_f64(&fixed[244..252])?;
        if record_duration_seconds <= 0.0 {
            return Err(IoError::Malformed(RejectionReason::InvalidRecordDuration));
        }
        let total_duration_seconds = record_duration_seconds * data_records as f64;
        if !total_duration_seconds.is_finite() {
            return Err(IoError::Malformed(RejectionReason::InvalidRecordDuration));
        }
        check_limit(
            LimitKind::Duration,
            total_duration_seconds.ceil() as u64,
            limits.max_duration_seconds,
        )?;

        let variable = read_exact(&mut reader, signal_header_bytes, "signal headers")?;
        let format = parse_format(&fixed[192..236]);
        let decoders =
            parse_signal_headers(&variable, signal_count, record_duration_seconds, limits)?;
        if format != RecordingFormat::Edf
            && !decoders
                .iter()
                .any(|decoder| decoder.metadata.is_annotation)
        {
            return Err(IoError::Malformed(RejectionReason::InvalidAnnotation));
        }

        let samples_per_record = decoders.iter().try_fold(0usize, |total, decoder| {
            total
                .checked_add(decoder.metadata.samples_per_record)
                .ok_or(IoError::ArithmeticOverflow("samples per data record"))
        })?;
        let record_bytes = samples_per_record
            .checked_mul(DIGITAL_SAMPLE_BYTES)
            .ok_or(IoError::ArithmeticOverflow("data record bytes"))?;
        let decoded_bytes = samples_per_record
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(IoError::ArithmeticOverflow("decoded epoch bytes"))?;
        let channel_struct_bytes = signal_count
            .checked_mul(std::mem::size_of::<EpochChannel>())
            .ok_or(IoError::ArithmeticOverflow("epoch channel metadata bytes"))?;
        let annotation_struct_bytes = limits
            .max_annotations_per_epoch
            .checked_mul(std::mem::size_of::<EpochAnnotation>())
            .ok_or(IoError::ArithmeticOverflow(
                "epoch annotation metadata bytes",
            ))?;
        let label_bytes = signal_count
            .checked_mul(16)
            .ok_or(IoError::ArithmeticOverflow("epoch label bytes"))?;
        let epoch_allocation = record_bytes
            .checked_add(decoded_bytes)
            .and_then(|bytes| bytes.checked_add(channel_struct_bytes))
            .and_then(|bytes| bytes.checked_add(annotation_struct_bytes))
            .and_then(|bytes| bytes.checked_add(label_bytes))
            .ok_or(IoError::ArithmeticOverflow("epoch allocation bytes"))?;
        check_limit(
            LimitKind::EpochAllocationBytes,
            to_u64(epoch_allocation, "epoch allocation")?,
            to_u64(limits.max_epoch_allocation_bytes, "epoch allocation limit")?,
        )?;
        let data_bytes = to_u64(record_bytes, "record bytes")?
            .checked_mul(data_records)
            .ok_or(IoError::ArithmeticOverflow("declared data bytes"))?;
        let expected_total_bytes = to_u64(header_bytes, "header bytes")?
            .checked_add(data_bytes)
            .ok_or(IoError::ArithmeticOverflow("declared source bytes"))?;
        check_limit(
            LimitKind::SourceBytes,
            expected_total_bytes,
            limits.max_bytes,
        )?;
        if let Some(actual) = known_length {
            if actual < expected_total_bytes {
                return Err(IoError::Truncated {
                    section: "source file",
                    expected: usize_from_u64(expected_total_bytes)?,
                    actual: usize_from_u64(actual)?,
                });
            }
            if actual > expected_total_bytes {
                return Err(IoError::Malformed(RejectionReason::TrailingData));
            }
        }

        Ok(Self {
            reader,
            limits,
            metadata: RecordingMetadata {
                format,
                patient_id: text(&fixed[8..88]),
                recording_id: text(&fixed[88..168]),
                start_date: text(&fixed[168..176]),
                start_time: text(&fixed[176..184]),
                data_records,
                record_duration_seconds,
                total_duration_seconds,
                signals: decoders
                    .iter()
                    .map(|decoder| decoder.metadata.clone())
                    .collect(),
            },
            decoders,
            record_bytes,
            records_read: 0,
            bytes_read: to_u64(header_bytes, "header bytes")?,
            last_epoch_start: None,
            terminal_checked: false,
            failed: false,
        })
    }

    fn read_epoch(&mut self) -> Result<Epoch, IoError> {
        let next_bytes = self
            .bytes_read
            .checked_add(to_u64(self.record_bytes, "record bytes")?)
            .ok_or(IoError::ArithmeticOverflow("stream byte count"))?;
        check_limit(LimitKind::SourceBytes, next_bytes, self.limits.max_bytes)?;
        let raw = read_exact(&mut self.reader, self.record_bytes, "data record")?;
        let epoch = decode_epoch(
            &raw,
            &self.decoders,
            self.records_read,
            self.metadata.record_duration_seconds,
            self.limits.max_annotations_per_epoch,
        )?;
        self.validate_chronology(&epoch)?;
        self.bytes_read = next_bytes;
        self.records_read += 1;
        self.last_epoch_start = Some(epoch.start_offset_seconds);
        Ok(epoch)
    }

    fn validate_chronology(&self, epoch: &Epoch) -> Result<(), IoError> {
        if epoch.start_offset_seconds < 0.0 {
            return Err(IoError::Malformed(RejectionReason::InvalidChronology));
        }
        check_limit(
            LimitKind::Duration,
            epoch.start_offset_seconds.ceil() as u64,
            self.limits.max_duration_seconds,
        )?;
        let Some(previous) = self.last_epoch_start else {
            return Ok(());
        };
        if epoch.start_offset_seconds <= previous {
            return Err(IoError::Malformed(RejectionReason::InvalidChronology));
        }
        if self.metadata.format == RecordingFormat::EdfPlusContinuous {
            let expected = previous + self.metadata.record_duration_seconds;
            if (epoch.start_offset_seconds - expected).abs() > 1e-6 {
                return Err(IoError::Malformed(RejectionReason::InvalidChronology));
            }
        }
        Ok(())
    }

    fn check_terminal(&mut self) -> Result<(), IoError> {
        if self.terminal_checked {
            return Ok(());
        }
        let mut byte = [0u8; 1];
        match self.reader.read(&mut byte) {
            Ok(0) => {
                self.terminal_checked = true;
                Ok(())
            }
            Ok(_) => Err(IoError::Malformed(RejectionReason::TrailingData)),
            Err(error) => Err(IoError::Io {
                context: "source terminator",
                kind: error.kind(),
            }),
        }
    }
}

impl<R: Read> EpochSource for EdfEpochSource<R> {
    fn metadata(&self) -> &RecordingMetadata {
        &self.metadata
    }

    fn next_epoch(&mut self) -> Result<Option<Epoch>, IoError> {
        if self.failed {
            return Err(IoError::SourceFailed);
        }
        let result = if self.records_read < self.metadata.data_records {
            self.read_epoch().map(Some)
        } else {
            self.check_terminal().map(|()| None)
        };
        if result.is_err() {
            self.failed = true;
        }
        result
    }
}

fn parse_format(reserved: &[u8]) -> RecordingFormat {
    let reserved = text(reserved);
    if reserved.starts_with("EDF+C") {
        RecordingFormat::EdfPlusContinuous
    } else if reserved.starts_with("EDF+D") {
        RecordingFormat::EdfPlusDiscontinuous
    } else {
        RecordingFormat::Edf
    }
}

fn parse_signal_headers(
    bytes: &[u8],
    count: usize,
    record_duration_seconds: f64,
    limits: IoLimits,
) -> Result<Vec<SignalDecoder>, IoError> {
    let labels = field_group(bytes, count, 0, 16)?;
    let transducers = field_group(bytes, count, 16, 80)?;
    let dimensions = field_group(bytes, count, 96, 8)?;
    let physical_minimums = field_group(bytes, count, 104, 8)?;
    let physical_maximums = field_group(bytes, count, 112, 8)?;
    let digital_minimums = field_group(bytes, count, 120, 8)?;
    let digital_maximums = field_group(bytes, count, 128, 8)?;
    let prefilters = field_group(bytes, count, 136, 80)?;
    let samples = field_group(bytes, count, 216, 8)?;

    let mut decoders = Vec::with_capacity(count);
    for index in 0..count {
        let label = text(labels[index]);
        let is_annotation = label.eq_ignore_ascii_case("EDF Annotations");
        let physical_minimum = parse_f64(physical_minimums[index])?;
        let physical_maximum = parse_f64(physical_maximums[index])?;
        if physical_maximum <= physical_minimum {
            return Err(IoError::Malformed(RejectionReason::InvalidPhysicalRange));
        }
        let digital_minimum = parse_i32(digital_minimums[index])?;
        let digital_maximum = parse_i32(digital_maximums[index])?;
        if digital_maximum <= digital_minimum
            || digital_minimum < i16::MIN as i32
            || digital_maximum > i16::MAX as i32
        {
            return Err(IoError::Malformed(RejectionReason::InvalidDigitalRange));
        }
        let samples_per_record = parse_usize(samples[index])?;
        if samples_per_record == 0 {
            return Err(IoError::Malformed(RejectionReason::InvalidSamplesPerRecord));
        }
        let sample_rate_hz = samples_per_record as f64 / record_duration_seconds;
        if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
            return Err(IoError::Malformed(RejectionReason::InvalidSamplesPerRecord));
        }
        if !is_annotation {
            if sample_rate_hz < 64.0 {
                return Err(IoError::Malformed(RejectionReason::InvalidSampleRate));
            }
            check_limit(
                LimitKind::SampleRate,
                sample_rate_hz.ceil() as u64,
                u64::from(limits.max_sample_rate_hz),
            )?;
        }
        let scale =
            (physical_maximum - physical_minimum) / f64::from(digital_maximum - digital_minimum);
        let offset = physical_minimum - f64::from(digital_minimum) * scale;
        if !scale.is_finite() || !offset.is_finite() {
            return Err(IoError::Malformed(RejectionReason::InvalidPhysicalRange));
        }
        decoders.push(SignalDecoder {
            metadata: SignalMetadata {
                label,
                transducer: text(transducers[index]),
                physical_dimension: text(dimensions[index]),
                physical_minimum,
                physical_maximum,
                digital_minimum,
                digital_maximum,
                prefiltering: text(prefilters[index]),
                samples_per_record,
                sample_rate_hz,
                is_annotation,
            },
            scale,
            offset,
        });
    }
    Ok(decoders)
}

fn field_group(
    bytes: &[u8],
    count: usize,
    preceding_width_per_signal: usize,
    width: usize,
) -> Result<Vec<&[u8]>, IoError> {
    let start = preceding_width_per_signal
        .checked_mul(count)
        .ok_or(IoError::ArithmeticOverflow("signal field offset"))?;
    let mut fields = Vec::with_capacity(count);
    for index in 0..count {
        let offset = index
            .checked_mul(width)
            .and_then(|value| start.checked_add(value))
            .ok_or(IoError::ArithmeticOverflow("signal field index"))?;
        let end = offset
            .checked_add(width)
            .ok_or(IoError::ArithmeticOverflow("signal field end"))?;
        fields.push(
            bytes
                .get(offset..end)
                .ok_or(IoError::Malformed(RejectionReason::InvalidHeaderLength))?,
        );
    }
    Ok(fields)
}

fn decode_epoch(
    raw: &[u8],
    decoders: &[SignalDecoder],
    record: u64,
    duration_seconds: f64,
    max_annotations: usize,
) -> Result<Epoch, IoError> {
    let mut offset = 0usize;
    let mut channels = Vec::with_capacity(decoders.len());
    let mut annotations = Vec::new();
    let mut annotation_onset = None;
    for (channel, decoder) in decoders.iter().enumerate() {
        let byte_count = decoder
            .metadata
            .samples_per_record
            .checked_mul(DIGITAL_SAMPLE_BYTES)
            .ok_or(IoError::ArithmeticOverflow("signal data bytes"))?;
        let end = offset
            .checked_add(byte_count)
            .ok_or(IoError::ArithmeticOverflow("signal data end"))?;
        let signal_bytes = raw.get(offset..end).ok_or(IoError::Truncated {
            section: "data record signal",
            expected: end,
            actual: raw.len(),
        })?;
        if decoder.metadata.is_annotation {
            let (parsed, onset) =
                parse_annotations(signal_bytes, annotations.len(), max_annotations)?;
            if let Some(onset) = onset {
                if annotation_onset
                    .is_some_and(|existing: f64| (existing - onset).abs() > f64::EPSILON)
                {
                    return Err(IoError::Malformed(RejectionReason::InvalidAnnotation));
                }
                annotation_onset = Some(onset);
            }
            annotations.extend(parsed);
        } else {
            let mut samples = Vec::with_capacity(decoder.metadata.samples_per_record);
            for (sample, pair) in signal_bytes.chunks_exact(2).enumerate() {
                let digital = f64::from(i16::from_le_bytes([pair[0], pair[1]]));
                let value = digital * decoder.scale + decoder.offset;
                if !value.is_finite() {
                    return Err(IoError::NonFiniteSample {
                        record,
                        channel,
                        sample,
                    });
                }
                samples.push(value);
            }
            channels.push(EpochChannel {
                signal_index: channel,
                label: decoder.metadata.label.clone(),
                sample_rate_hz: decoder.metadata.sample_rate_hz,
                samples,
            });
        }
        offset = end;
    }
    if offset != raw.len() {
        return Err(IoError::Malformed(RejectionReason::TrailingData));
    }
    Ok(Epoch {
        index: record,
        start_offset_seconds: annotation_onset.unwrap_or(record as f64 * duration_seconds),
        duration_seconds,
        channels,
        annotations,
    })
}

fn parse_annotations(
    bytes: &[u8],
    already_parsed: usize,
    maximum: usize,
) -> Result<(Vec<EpochAnnotation>, Option<f64>), IoError> {
    let mut annotations = Vec::new();
    let mut first_onset = None;
    for tal in bytes.split(|byte| *byte == 0) {
        if tal.is_empty() {
            continue;
        }
        let mut fields = tal.split(|byte| *byte == 0x14);
        let timing = fields
            .next()
            .ok_or(IoError::Malformed(RejectionReason::InvalidAnnotation))?;
        let mut timing = timing.split(|byte| *byte == 0x15);
        let onset = parse_annotation_f64(timing.next().unwrap_or_default())?;
        if onset < 0.0 {
            return Err(IoError::Malformed(RejectionReason::InvalidAnnotation));
        }
        first_onset.get_or_insert(onset);
        let duration = timing
            .next()
            .filter(|field| !field.is_empty())
            .map(parse_annotation_f64)
            .transpose()?;
        if duration.is_some_and(|duration| duration < 0.0) {
            return Err(IoError::Malformed(RejectionReason::InvalidAnnotation));
        }
        for annotation in fields.filter(|field| !field.is_empty()) {
            let text = latin1(annotation).trim().to_string();
            if !text.is_empty() {
                let total = already_parsed
                    .checked_add(annotations.len())
                    .ok_or(IoError::ArithmeticOverflow("annotation count"))?;
                if total >= maximum {
                    return Err(IoError::LimitExceeded {
                        limit: LimitKind::Annotations,
                        actual: to_u64(
                            total
                                .checked_add(1)
                                .ok_or(IoError::ArithmeticOverflow("annotation count"))?,
                            "annotation count",
                        )?,
                        maximum: to_u64(maximum, "annotation limit")?,
                    });
                }
                annotations.push(EpochAnnotation {
                    onset_seconds: onset,
                    duration_seconds: duration,
                    text,
                });
            }
        }
    }
    if first_onset.is_none() {
        return Err(IoError::Malformed(RejectionReason::InvalidAnnotation));
    }
    Ok((annotations, first_onset))
}

fn parse_annotation_f64(bytes: &[u8]) -> Result<f64, IoError> {
    let value = latin1(bytes)
        .parse::<f64>()
        .map_err(|_| IoError::Malformed(RejectionReason::InvalidAnnotation))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(IoError::Malformed(RejectionReason::InvalidAnnotation))
    }
}

fn parse_usize(bytes: &[u8]) -> Result<usize, IoError> {
    text(bytes)
        .parse::<usize>()
        .map_err(|_| IoError::Malformed(RejectionReason::InvalidNumericField))
}

fn parse_i32(bytes: &[u8]) -> Result<i32, IoError> {
    text(bytes)
        .parse::<i32>()
        .map_err(|_| IoError::Malformed(RejectionReason::InvalidNumericField))
}

fn parse_f64(bytes: &[u8]) -> Result<f64, IoError> {
    let value = text(bytes)
        .parse::<f64>()
        .map_err(|_| IoError::Malformed(RejectionReason::InvalidNumericField))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(IoError::Malformed(RejectionReason::InvalidNumericField))
    }
}

fn text(bytes: &[u8]) -> String {
    latin1(bytes).trim_matches([' ', '\0']).to_string()
}

fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| char::from(*byte)).collect()
}

fn read_exact(
    reader: &mut impl Read,
    expected: usize,
    section: &'static str,
) -> Result<Vec<u8>, IoError> {
    let mut bytes = vec![0u8; expected];
    let mut actual = 0usize;
    while actual < expected {
        match reader.read(&mut bytes[actual..]) {
            Ok(0) => {
                return Err(IoError::Truncated {
                    section,
                    expected,
                    actual,
                })
            }
            Ok(read) => {
                actual = actual
                    .checked_add(read)
                    .ok_or(IoError::ArithmeticOverflow("read byte count"))?;
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => {
                return Err(IoError::Io {
                    context: section,
                    kind: error.kind(),
                })
            }
        }
    }
    Ok(bytes)
}

fn check_limit(limit: LimitKind, actual: u64, maximum: u64) -> Result<(), IoError> {
    if actual > maximum {
        Err(IoError::LimitExceeded {
            limit,
            actual,
            maximum,
        })
    } else {
        Ok(())
    }
}

fn to_u64(value: usize, context: &'static str) -> Result<u64, IoError> {
    value
        .try_into()
        .map_err(|_| IoError::ArithmeticOverflow(context))
}

fn usize_from_u64(value: u64) -> Result<usize, IoError> {
    value
        .try_into()
        .map_err(|_| IoError::ArithmeticOverflow("platform source length"))
}
