use ruv_neural_io::{
    EdfEpochSource, EpochSource, IoError, IoLimits, LimitKind, RecordingFormat, RejectionReason,
};
use std::io::Cursor;

#[derive(Clone)]
struct Signal<'a> {
    label: &'a str,
    samples: usize,
    physical_minimum: &'a str,
    physical_maximum: &'a str,
    digital_minimum: &'a str,
    digital_maximum: &'a str,
}

fn signal(label: &str, samples: usize) -> Signal<'_> {
    Signal {
        label,
        samples,
        physical_minimum: "-100",
        physical_maximum: "100",
        digital_minimum: "-32768",
        digital_maximum: "32767",
    }
}

fn put(target: &mut [u8], value: &str) {
    target.fill(b' ');
    let length = value.len().min(target.len());
    target[..length].copy_from_slice(&value.as_bytes()[..length]);
}

fn group(output: &mut Vec<u8>, values: &[String], width: usize) {
    for value in values {
        let start = output.len();
        output.resize(start + width, b' ');
        put(&mut output[start..start + width], value);
    }
}

fn edf_bytes(
    signals: &[Signal<'_>],
    records: &str,
    duration: &str,
    reserved: &str,
    record_data: &[Vec<u8>],
) -> Vec<u8> {
    let header_bytes = 256 + signals.len() * 256;
    let mut fixed = vec![b' '; 256];
    put(&mut fixed[0..8], "0");
    put(&mut fixed[8..88], "study-subject");
    put(&mut fixed[88..168], "fixture-recording");
    put(&mut fixed[168..176], "03.08.26");
    put(&mut fixed[176..184], "12.00.00");
    put(&mut fixed[184..192], &header_bytes.to_string());
    put(&mut fixed[192..236], reserved);
    put(&mut fixed[236..244], records);
    put(&mut fixed[244..252], duration);
    put(&mut fixed[252..256], &signals.len().to_string());

    let strings = |value: &str| vec![value.to_string(); signals.len()];
    let mut variable = Vec::new();
    group(
        &mut variable,
        &signals
            .iter()
            .map(|s| s.label.to_string())
            .collect::<Vec<_>>(),
        16,
    );
    group(&mut variable, &strings("fixture-transducer"), 80);
    group(&mut variable, &strings("uV"), 8);
    group(
        &mut variable,
        &signals
            .iter()
            .map(|s| s.physical_minimum.to_string())
            .collect::<Vec<_>>(),
        8,
    );
    group(
        &mut variable,
        &signals
            .iter()
            .map(|s| s.physical_maximum.to_string())
            .collect::<Vec<_>>(),
        8,
    );
    group(
        &mut variable,
        &signals
            .iter()
            .map(|s| s.digital_minimum.to_string())
            .collect::<Vec<_>>(),
        8,
    );
    group(
        &mut variable,
        &signals
            .iter()
            .map(|s| s.digital_maximum.to_string())
            .collect::<Vec<_>>(),
        8,
    );
    group(&mut variable, &strings("HP:0.5Hz LP:40Hz"), 80);
    group(
        &mut variable,
        &signals
            .iter()
            .map(|s| s.samples.to_string())
            .collect::<Vec<_>>(),
        8,
    );
    group(&mut variable, &strings(""), 32);
    assert_eq!(variable.len(), signals.len() * 256);

    fixed.extend(variable);
    for data in record_data {
        fixed.extend_from_slice(data);
    }
    fixed
}

fn i16_data(values: &[i16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn annotation_data(tal: &[u8], byte_count: usize) -> Vec<u8> {
    let mut annotation = tal.to_vec();
    annotation.resize(byte_count, 0);
    annotation
}

#[test]
fn streams_complete_edf_records_without_partial_output() {
    let signals = [signal("F3", 4), signal("P3", 4)];
    let data = vec![i16_data(&[-32768, -1, 0, 32767]), i16_data(&[1, 2, 3, 4])];
    let bytes = edf_bytes(&signals, "1", "0.0625", "", &data);
    let mut source = EdfEpochSource::new(Cursor::new(bytes), IoLimits::default()).unwrap();
    assert_eq!(source.metadata().format, RecordingFormat::Edf);
    assert_eq!(source.metadata().signals.len(), 2);
    assert_eq!(source.metadata().signals[0].sample_rate_hz, 64.0);

    let epoch = source.next_epoch().unwrap().unwrap();
    assert_eq!(epoch.index, 0);
    assert_eq!(epoch.channels.len(), 2);
    assert_eq!(epoch.channels[0].samples.len(), 4);
    assert!(epoch
        .channels
        .iter()
        .flat_map(|channel| channel.samples.iter().copied())
        .all(f64::is_finite));
    assert!(source.next_epoch().unwrap().is_none());
}

#[test]
fn parses_edf_plus_annotations_separately_from_waveforms() {
    let signals = [signal("F3", 2), signal("EDF Annotations", 16)];
    let annotation = annotation_data(b"+0\x14Sleep stage N2\x14\0", 32);
    let data = vec![i16_data(&[0, 1]), annotation];
    let bytes = edf_bytes(&signals, "1", "0.03125", "EDF+C", &data);
    let mut source = EdfEpochSource::new(Cursor::new(bytes), IoLimits::default()).unwrap();
    assert_eq!(source.metadata().format, RecordingFormat::EdfPlusContinuous);
    let epoch = source.next_epoch().unwrap().unwrap();
    assert_eq!(epoch.channels.len(), 1);
    assert_eq!(epoch.annotations.len(), 1);
    assert_eq!(epoch.annotations[0].text, "Sleep stage N2");
    assert_eq!(epoch.annotations[0].onset_seconds, 0.0);
}

#[test]
fn edf_plus_discontinuous_uses_signed_annotation_onset() {
    let signals = [signal("F3", 2), signal("EDF Annotations", 16)];
    let annotation = annotation_data(b"+12.5\x14\x14\0", 32);
    let data = vec![i16_data(&[0, 1]), annotation];
    let bytes = edf_bytes(&signals, "1", "0.03125", "EDF+D", &data);
    let mut source = EdfEpochSource::new(Cursor::new(bytes), IoLimits::default()).unwrap();
    let epoch = source.next_epoch().unwrap().unwrap();
    assert_eq!(
        source.metadata().format,
        RecordingFormat::EdfPlusDiscontinuous
    );
    assert_eq!(epoch.start_offset_seconds, 12.5);
}

#[test]
fn rejects_negative_annotation_duration_and_invalid_record_chronology() {
    let signals = [signal("F3", 2), signal("EDF Annotations", 16)];
    let negative = annotation_data(b"+0\x15-1\x14N2\x14\0", 32);
    let data = vec![i16_data(&[0, 1]), negative];
    let bytes = edf_bytes(&signals, "1", "0.03125", "EDF+C", &data);
    let mut source = EdfEpochSource::new(Cursor::new(bytes), IoLimits::default()).unwrap();
    assert!(matches!(
        source.next_epoch(),
        Err(IoError::Malformed(RejectionReason::InvalidAnnotation))
    ));

    let event = annotation_data(b"+0\x14\x14\0-0.065\x14Stimulus\x14\0", 32);
    let data = vec![i16_data(&[0, 1]), event];
    let bytes = edf_bytes(&signals, "1", "0.03125", "EDF+C", &data);
    let mut source = EdfEpochSource::new(Cursor::new(bytes), IoLimits::default()).unwrap();
    let epoch = source.next_epoch().unwrap().unwrap();
    assert_eq!(epoch.start_offset_seconds, 0.0);
    assert_eq!(epoch.annotations[0].onset_seconds, -0.065);

    let data = vec![
        i16_data(&[0, 1]),
        annotation_data(b"+0\x14\x14\0", 32),
        i16_data(&[2, 3]),
        annotation_data(b"+0.03125\x14\x14\0", 32),
    ];
    let bytes = edf_bytes(&signals, "2", "0.03125", "EDF+C", &data);
    let mut source = EdfEpochSource::new(Cursor::new(bytes), IoLimits::default()).unwrap();
    assert!(source.next_epoch().unwrap().is_some());
    assert!(source.next_epoch().unwrap().is_some());
    assert!(source.next_epoch().unwrap().is_none());

    let data = vec![
        i16_data(&[0, 1]),
        annotation_data(b"+0\x14\x14\0", 32),
        i16_data(&[2, 3]),
        annotation_data(b"+1\x14\x14\0", 32),
    ];
    let bytes = edf_bytes(&signals, "2", "0.03125", "EDF+C", &data);
    let mut source = EdfEpochSource::new(Cursor::new(bytes), IoLimits::default()).unwrap();
    assert!(source.next_epoch().unwrap().is_some());
    assert!(matches!(
        source.next_epoch(),
        Err(IoError::Malformed(RejectionReason::InvalidChronology))
    ));

    let data = vec![
        i16_data(&[0, 1]),
        annotation_data(b"+1\x14\x14\0", 32),
        i16_data(&[2, 3]),
        annotation_data(b"+0\x14\x14\0", 32),
    ];
    let bytes = edf_bytes(&signals, "2", "0.03125", "EDF+D", &data);
    let mut source = EdfEpochSource::new(Cursor::new(bytes), IoLimits::default()).unwrap();
    assert!(source.next_epoch().unwrap().is_some());
    assert!(matches!(
        source.next_epoch(),
        Err(IoError::Malformed(RejectionReason::InvalidChronology))
    ));
}

#[test]
fn accepts_negative_amplifier_gain_and_enforces_tal_lexical_grammar() {
    let mut inverted = signal("F3", 2);
    inverted.physical_minimum = "100";
    inverted.physical_maximum = "-100";
    let bytes = edf_bytes(
        &[inverted],
        "1",
        "0.03125",
        "",
        &[i16_data(&[-32768, 32767])],
    );
    let mut source = EdfEpochSource::new(Cursor::new(bytes), IoLimits::default()).unwrap();
    let epoch = source.next_epoch().unwrap().unwrap();
    assert!((epoch.channels[0].samples[0] - 100.0).abs() < 1e-9);
    assert!((epoch.channels[0].samples[1] + 100.0).abs() < 1e-9);

    let signals = [signal("F3", 2), signal("EDF Annotations", 16)];
    for malformed in [
        b"0\x14\x14\0".as_slice(),
        b"+1e3\x14\x14\0".as_slice(),
        b"+.5\x14\x14\0".as_slice(),
        b"+0\x15+1\x14N2\x14\0".as_slice(),
        b"+0\x151e3\x14N2\x14\0".as_slice(),
    ] {
        let data = vec![i16_data(&[0, 1]), annotation_data(malformed, 32)];
        let bytes = edf_bytes(&signals, "1", "0.03125", "EDF+C", &data);
        let mut source = EdfEpochSource::new(Cursor::new(bytes), IoLimits::default()).unwrap();
        assert!(matches!(
            source.next_epoch(),
            Err(IoError::Malformed(RejectionReason::InvalidAnnotation))
        ));
    }
}

#[test]
fn discontinuous_epoch_end_must_fit_duration_limit() {
    let signals = [signal("F3", 2), signal("EDF Annotations", 16)];
    let data = vec![i16_data(&[0, 1]), annotation_data(b"+1\x14\x14\0", 32)];
    let bytes = edf_bytes(&signals, "1", "0.03125", "EDF+D", &data);
    let limits = IoLimits {
        max_duration_seconds: 1,
        ..IoLimits::default()
    };
    let mut source = EdfEpochSource::new(Cursor::new(bytes), limits).unwrap();
    assert!(matches!(
        source.next_epoch(),
        Err(IoError::LimitExceeded {
            limit: LimitKind::Duration,
            ..
        })
    ));
}

#[test]
fn rejects_truncated_headers_and_records_then_becomes_terminal() {
    let error = EdfEpochSource::new(Cursor::new(vec![b' '; 100]), IoLimits::default())
        .err()
        .unwrap();
    assert!(matches!(
        error,
        IoError::Truncated {
            section: "fixed header",
            expected: 256,
            actual: 100
        }
    ));

    let signals = [signal("F3", 4)];
    let bytes = edf_bytes(&signals, "1", "0.0625", "", &[i16_data(&[1, 2])]);
    let mut source = EdfEpochSource::new(Cursor::new(bytes), IoLimits::default()).unwrap();
    assert!(matches!(
        source.next_epoch(),
        Err(IoError::Truncated {
            section: "data record",
            expected: 8,
            actual: 4
        })
    ));
    assert!(matches!(source.next_epoch(), Err(IoError::SourceFailed)));
}

#[test]
fn rejects_invalid_and_unknown_header_fields() {
    let signals = [signal("F3", 1)];
    let mut bad_length = edf_bytes(&signals, "1", "1", "", &[i16_data(&[0])]);
    put(&mut bad_length[184..192], "999");
    assert!(matches!(
        EdfEpochSource::new(Cursor::new(bad_length), IoLimits::default()),
        Err(IoError::Malformed(RejectionReason::InvalidHeaderLength))
    ));

    let unknown = edf_bytes(&signals, "-1", "1", "", &[]);
    assert!(matches!(
        EdfEpochSource::new(Cursor::new(unknown), IoLimits::default()),
        Err(IoError::Malformed(RejectionReason::UnknownRecordCount))
    ));

    let mut nonfinite = signal("F3", 1);
    nonfinite.physical_minimum = "NaN";
    let bytes = edf_bytes(&[nonfinite], "1", "1", "", &[i16_data(&[0])]);
    assert!(matches!(
        EdfEpochSource::new(Cursor::new(bytes), IoLimits::default()),
        Err(IoError::Malformed(RejectionReason::InvalidNumericField))
    ));

    let bytes = edf_bytes(&[signal("F3", 63)], "1", "1", "", &[]);
    assert!(matches!(
        EdfEpochSource::new(Cursor::new(bytes), IoLimits::default()),
        Err(IoError::Malformed(RejectionReason::InvalidSampleRate))
    ));

    let signals = [signal("F3", 2), signal("EDF Annotations", 16)];
    let data = vec![i16_data(&[0, 1]), vec![0; 32]];
    let bytes = edf_bytes(&signals, "1", "0.03125", "EDF+C", &data);
    let mut source = EdfEpochSource::new(Cursor::new(bytes), IoLimits::default()).unwrap();
    assert!(matches!(
        source.next_epoch(),
        Err(IoError::Malformed(RejectionReason::InvalidAnnotation))
    ));
}

#[test]
fn enforces_channel_rate_duration_metadata_allocation_and_byte_limits() {
    let limits = IoLimits {
        max_bytes: 255,
        ..IoLimits::default()
    };
    assert_limit(Vec::new(), limits, LimitKind::SourceBytes);

    let two_signals = [signal("F3", 4), signal("P3", 4)];
    let bytes = edf_bytes(&two_signals, "1", "1", "", &[]);
    let limits = IoLimits {
        max_channels: 1,
        ..IoLimits::default()
    };
    assert_limit(bytes, limits, LimitKind::Channels);

    let bytes = edf_bytes(&[signal("F3", 4097)], "1", "1", "", &[]);
    assert_limit(bytes, IoLimits::default(), LimitKind::SampleRate);

    let bytes = edf_bytes(&[signal("F3", 1)], "3", "1", "", &[]);
    let limits = IoLimits {
        max_duration_seconds: 2,
        ..IoLimits::default()
    };
    assert_limit(bytes, limits, LimitKind::Duration);

    let bytes = edf_bytes(&[signal("F3", 64)], "1001", "1", "", &[]);
    let limits = IoLimits {
        max_data_records: 1_000,
        ..IoLimits::default()
    };
    assert_limit(bytes, limits, LimitKind::DataRecords);

    let bytes = edf_bytes(&two_signals, "1", "1", "", &[]);
    let limits = IoLimits {
        max_metadata_bytes: 512,
        ..IoLimits::default()
    };
    assert_limit(bytes, limits, LimitKind::MetadataBytes);

    let bytes = edf_bytes(&[signal("F3", 4)], "1", "0.0625", "", &[]);
    let limits = IoLimits {
        max_epoch_allocation_bytes: 39,
        ..IoLimits::default()
    };
    assert_limit(bytes, limits, LimitKind::EpochAllocationBytes);

    let bytes = edf_bytes(&[signal("F3", 4)], "1", "0.0625", "", &[]);
    let limits = IoLimits {
        max_bytes: 519,
        ..IoLimits::default()
    };
    assert_limit(bytes, limits, LimitKind::SourceBytes);

    let signals = [signal("F3", 2), signal("EDF Annotations", 16)];
    let annotation = annotation_data(b"+0\x14A\x14B\x14\0", 32);
    let data = vec![i16_data(&[0, 1]), annotation];
    let bytes = edf_bytes(&signals, "1", "0.03125", "EDF+C", &data);
    let limits = IoLimits {
        max_annotations_per_epoch: 1,
        ..IoLimits::default()
    };
    let mut source = EdfEpochSource::new(Cursor::new(bytes), limits).unwrap();
    assert!(matches!(
        source.next_epoch(),
        Err(IoError::LimitExceeded {
            limit: LimitKind::Annotations,
            actual: 2,
            maximum: 1
        })
    ));
}

fn assert_limit(bytes: Vec<u8>, limits: IoLimits, expected: LimitKind) {
    assert!(matches!(
        EdfEpochSource::new(Cursor::new(bytes), limits),
        Err(IoError::LimitExceeded { limit, .. }) if limit == expected
    ));
}

#[test]
fn trailing_data_is_rejected_after_declared_records() {
    let signals = [signal("F3", 1)];
    let mut bytes = edf_bytes(&signals, "1", "0.015625", "", &[i16_data(&[0])]);
    bytes.push(9);
    let mut source = EdfEpochSource::new(Cursor::new(bytes), IoLimits::default()).unwrap();
    assert!(source.next_epoch().unwrap().is_some());
    assert!(matches!(
        source.next_epoch(),
        Err(IoError::Malformed(RejectionReason::TrailingData))
    ));
}

#[test]
#[cfg(feature = "brainvision-compat")]
fn brainvision_compatibility_surface_remains_available() {
    let result = ruv_neural_io::brainvision::read_vhdr(std::path::Path::new(
        "ruv-neural-io-test-path-that-does-not-exist.vhdr",
    ));
    assert!(result.is_err());
}
