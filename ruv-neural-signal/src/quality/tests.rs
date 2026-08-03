//! Gate-by-gate tests for epoch admission.

use super::*;

const RATE: f64 = 250.0;
const SAMPLES: usize = 2_500;

fn rails() -> AdcRails {
    AdcRails {
        minimum_uv: -500.0,
        maximum_uv: 500.0,
        tolerance_uv: 0.5,
    }
}

fn config() -> EpochQualityConfig {
    EpochQualityConfig {
        adc_rails: rails(),
        maximum_clipped_fraction: 0.001,
        flatline_seconds: 1.0,
        flatline_epsilon_uv: 1e-6,
        maximum_gap_seconds: 1.0,
        maximum_artifact_fraction: 0.20,
    }
}

fn wave() -> Vec<f64> {
    (0..SAMPLES)
        .map(|index| 40.0 * (index as f64 * 0.05).sin())
        .collect()
}

fn assess(channel: &[f64], mask: &[bool]) -> EpochQuality {
    assess_epoch(&[channel], mask, RATE, config()).unwrap()
}

#[test]
fn a_clean_epoch_is_admitted_with_its_measurements_reported() {
    let quality = assess(&wave(), &vec![true; SAMPLES]);
    assert!(quality.accepted);
    assert_eq!(quality.rejection, None);
    assert_eq!(quality.artifact_fraction, 0.0);
    assert_eq!(quality.clipped_fraction, 0.0);
    assert_eq!(quality.longest_gap_seconds, 0.0);
}

#[test]
fn a_quiet_epoch_is_not_clipped_by_its_own_extrema() {
    // The decisive case: this epoch's own min and max are 2 uV apart, nowhere
    // near the declared +/-500 uV rails. Inferring rails from the data would
    // reject it; declared rails must not.
    let quiet: Vec<f64> = (0..SAMPLES)
        .map(|index| if index % 2 == 0 { -1.0 } else { 1.0 })
        .collect();
    let quality = assess(&quiet, &vec![true; SAMPLES]);
    assert!(quality.accepted, "{quality:?}");
    assert_eq!(quality.clipped_fraction, 0.0);
}

#[test]
fn samples_at_a_declared_rail_trip_the_clipping_gate() {
    let mut signal = wave();
    for value in signal.iter_mut().take(50) {
        *value = 499.75; // inside maximum_uv - tolerance_uv
    }
    let quality = assess(&signal, &vec![true; SAMPLES]);
    assert_eq!(quality.rejection, Some(EpochRejection::Clipped));
    assert!(quality.clipped_fraction > config().maximum_clipped_fraction);

    // A single railed sample stays under the fraction gate.
    let mut barely = wave();
    barely[0] = -500.0;
    assert!(assess(&barely, &vec![true; SAMPLES]).accepted);
}

#[test]
fn a_long_contiguous_invalid_run_trips_the_gap_gate_before_the_coverage_gate() {
    let mut mask = vec![true; SAMPLES];
    // 300 samples = 1.2 s of gap, but only 12% coverage: under the 20% gate.
    for valid in mask.iter_mut().skip(100).take(300) {
        *valid = false;
    }
    let quality = assess(&wave(), &mask);
    assert_eq!(quality.rejection, Some(EpochRejection::Gap));
    assert!((quality.longest_gap_seconds - 1.2).abs() < 1e-9);
    assert!(quality.artifact_fraction < config().maximum_artifact_fraction);
}

#[test]
fn scattered_invalid_samples_trip_coverage_without_a_long_gap() {
    let mut mask = vec![true; SAMPLES];
    for (index, valid) in mask.iter_mut().enumerate() {
        if index % 4 == 0 {
            *valid = false;
        }
    }
    let quality = assess(&wave(), &mask);
    assert_eq!(quality.rejection, Some(EpochRejection::ArtifactCoverage));
    assert!((quality.artifact_fraction - 0.25).abs() < 1e-9);
    assert!((quality.longest_gap_seconds - 1.0 / RATE).abs() < 1e-9);
}

#[test]
fn a_flatline_span_is_rejected_and_masked_gaps_neither_hide_nor_fake_one() {
    let mut flat = wave();
    for value in flat.iter_mut().skip(500).take(300) {
        *value = 7.0;
    }
    assert_eq!(
        assess(&flat, &vec![true; SAMPLES]).rejection,
        Some(EpochRejection::Flatline)
    );

    // An all-NaN masked span must not be read as an unchanging span.
    let mut gapped = wave();
    let mut mask = vec![true; SAMPLES];
    for index in 500..800 {
        gapped[index] = f64::NAN;
        mask[index] = false;
    }
    let quality = assess(&gapped, &mask);
    assert_eq!(quality.rejection, Some(EpochRejection::Gap));
}

#[test]
fn a_non_finite_sample_the_mask_calls_valid_is_rejected() {
    let mut signal = wave();
    signal[10] = f64::NAN;
    assert_eq!(
        assess(&signal, &vec![true; SAMPLES]).rejection,
        Some(EpochRejection::NonFinite)
    );

    let mut mask = vec![true; SAMPLES];
    mask[10] = false;
    assert!(assess(&signal, &mask).accepted);
}

#[test]
fn shorter_epochs_than_the_flatline_window_do_not_panic() {
    // flatline_seconds * rate = 250 samples, longer than this 100-sample epoch.
    let short = vec![1.0; 100];
    let quality = assess_epoch(&[&short], &[true; 100], RATE, config()).unwrap();
    assert!(quality.accepted);
}

#[test]
fn channel_and_mask_length_disagreement_is_a_typed_rejection() {
    let signal = wave();
    let short = vec![0.0; 10];
    assert_eq!(
        assess_epoch(&[&signal, &short], &[true; SAMPLES], RATE, config())
            .unwrap()
            .rejection,
        Some(EpochRejection::MaskMismatch)
    );
    assert_eq!(
        assess(&signal, &[true; 10]).rejection,
        Some(EpochRejection::MaskMismatch)
    );
}

#[test]
fn configuration_and_input_faults_are_errors_not_rejections() {
    let signal = wave();
    let mask = vec![true; SAMPLES];
    assert_eq!(
        assess_epoch(&[], &mask, RATE, config()),
        Err(QualityError::NoChannels)
    );
    assert!(assess_epoch(&[&signal], &mask, 0.0, config()).is_err());
    assert!(assess_epoch(&[&signal], &mask, f64::NAN, config()).is_err());

    let broken = [
        EpochQualityConfig {
            maximum_clipped_fraction: 1.5,
            ..config()
        },
        EpochQualityConfig {
            maximum_artifact_fraction: -0.1,
            ..config()
        },
        EpochQualityConfig {
            flatline_seconds: 0.0,
            ..config()
        },
        EpochQualityConfig {
            maximum_gap_seconds: -1.0,
            ..config()
        },
        EpochQualityConfig {
            flatline_epsilon_uv: f64::NAN,
            ..config()
        },
        EpochQualityConfig {
            adc_rails: AdcRails {
                minimum_uv: 500.0,
                maximum_uv: -500.0,
                tolerance_uv: 0.5,
            },
            ..config()
        },
        EpochQualityConfig {
            adc_rails: AdcRails {
                tolerance_uv: 600.0,
                ..rails()
            },
            ..config()
        },
        EpochQualityConfig {
            adc_rails: AdcRails {
                tolerance_uv: f64::INFINITY,
                ..rails()
            },
            ..config()
        },
    ];
    for config in broken {
        assert!(
            matches!(config.validate(), Err(QualityError::InvalidConfig(_))),
            "accepted {config:?}"
        );
    }
    assert!(config().validate().is_ok());
}
