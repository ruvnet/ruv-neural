//! End-to-end tests over a synthetic expert-scored night.
//!
//! The night is generated from the same aperiodic model the analysis fits, so
//! recovered parameters have a known truth. These are self-consistency tests:
//! no external polysomnography reference is available in-repo, so no claim of
//! parity with a published pipeline is made or tested here.

use super::*;
use ruv_neural_core::attestation::{verify_neurosleep_bundle, PersistentEd25519Signer};
use ruv_neural_core::neurosleep::{
    compatibility_fingerprint_v1, AcquisitionChannel, AcquisitionMetadata, AcquisitionModality,
    AlgorithmManifest, AlgorithmParameter, FeatureValue, NullReason, ResearchCitation,
    SourceFormat, Species, StageSource,
};
use std::f64::consts::PI;

const RATE: f64 = 250.0;
const EPOCH_SAMPLES: usize = 2_500;
const THETA_CENTER_HZ: f64 = 6.35;

/// Synthesise one epoch whose PSD follows `offset - log10(knee + f^exponent)`
/// with a Gaussian theta bump, by summing sinusoids on a comb.
///
/// The comb step matches the 0.2 Hz Welch grid so every component lands on a bin
/// centre and contributes no leakage. The theta bump centre deliberately does
/// *not* land on a bin, so recovering it exercises the sub-bin interpolation
/// rather than the grid.
fn epoch_waveform(exponent: f64, phase_shift: f64) -> Vec<f64> {
    let step_hz = 0.2;
    let mut samples = vec![0.0; EPOCH_SAMPLES];
    let mut harmonic = 1;
    while (harmonic as f64) * step_hz <= 45.0 {
        let frequency = harmonic as f64 * step_hz;
        let background = 10f64.powf(3.0 - (5.0 + frequency.powf(exponent)).log10());
        let bump = 10f64.powf(0.6 * (-0.5 * ((frequency - THETA_CENTER_HZ) / 0.8).powi(2)).exp());
        let amplitude = (2.0 * background * bump * step_hz).sqrt();
        // A fixed irrational phase ramp keeps the comb from summing into a
        // periodic spike train without needing a random number generator.
        let phase = (harmonic as f64 * 2.399_963_2) % (2.0 * PI) + phase_shift;
        for (index, value) in samples.iter_mut().enumerate() {
            *value += amplitude * (2.0 * PI * frequency * index as f64 / RATE + phase).sin();
        }
        harmonic += 1;
    }
    samples
}

/// All epochs of a stage share one waveform, so the stage-averaged spectrum is
/// exactly the single-epoch spectrum and the expected values stay analytic.
fn stage_epochs(state: SleepState, exponent: f64, count: usize) -> Vec<ExpertEpoch> {
    let epoch = ExpertEpoch {
        state,
        channels_uv: vec![epoch_waveform(exponent, 0.0), epoch_waveform(exponent, 0.4)],
        shared_valid_mask: vec![true; EPOCH_SAMPLES],
    };
    vec![epoch; count]
}

/// 180 wake + 180 NREM + 60 REM epochs: exactly the sufficiency thresholds.
fn night() -> Vec<ExpertEpoch> {
    let mut epochs = stage_epochs(SleepState::Wake, 1.4, 180);
    epochs.extend(stage_epochs(SleepState::Nrem, 2.1, 180));
    epochs.extend(stage_epochs(SleepState::Rem, 1.7, 60));
    epochs
}

fn profile() -> NeuroSleepProfile {
    NeuroSleepProfile::constantino_250hz()
}

fn value(feature: &FeatureValue) -> f64 {
    match feature {
        FeatureValue::Observed { value, .. } => *value,
        FeatureValue::Null { reason } => panic!("expected an observation, got {reason:?}"),
    }
}

fn unit(feature: &FeatureValue) -> &str {
    match feature {
        FeatureValue::Observed { unit, .. } => unit,
        FeatureValue::Null { reason } => panic!("expected an observation, got {reason:?}"),
    }
}

fn reason(feature: &FeatureValue) -> NullReason {
    match feature {
        FeatureValue::Null { reason } => *reason,
        FeatureValue::Observed { value, unit } => panic!("expected a null, got {value} {unit}"),
    }
}

#[test]
fn the_paper_profile_is_exactly_ten_seconds_and_2500_samples() {
    let profile = profile();
    assert_eq!(profile.sample_rate_hz, 250.0);
    assert_eq!(profile.epoch_seconds, 10.0);
    assert_eq!(profile.epoch_samples(), EPOCH_SAMPLES);
    assert!(profile.validate().is_ok());
}

#[test]
fn a_sufficient_night_produces_contract_units_for_every_stage() {
    let analysis = analyze_night(&night(), &profile()).unwrap();
    assert!(analysis.quality.accepted, "{:?}", analysis.quality);
    assert_eq!(analysis.quality.valid_coverage_fraction, 1.0);
    assert_eq!(analysis.quality.artifact_fraction, 0.0);
    assert_eq!(analysis.qeeg_by_stage.len(), 3);

    for stage in &analysis.qeeg_by_stage {
        assert_eq!(unit(&stage.delta_absolute_power), "uV2");
        assert_eq!(unit(&stage.theta_absolute_power), "uV2");
        assert_eq!(unit(&stage.alpha_absolute_power), "uV2");
        assert_eq!(unit(&stage.delta_relative_power), "ratio");
        assert_eq!(unit(&stage.theta_relative_power), "ratio");
        assert_eq!(unit(&stage.theta_peak_frequency), "Hz");
        assert_eq!(unit(&stage.theta_peak_power), "log10_uV2_per_hz");
        assert_eq!(unit(&stage.frontal_parietal_theta_coherence), "ratio");
        assert_eq!(unit(&stage.frontal_parietal_full_band_coherence), "ratio");

        let delta = value(&stage.delta_relative_power);
        let theta = value(&stage.theta_relative_power);
        assert!((0.0..=1.0).contains(&delta), "delta ratio {delta}");
        assert!((0.0..=1.0).contains(&theta), "theta ratio {theta}");
        assert!(value(&stage.delta_absolute_power) > 0.0);

        for coherence in [
            &stage.frontal_parietal_theta_coherence,
            &stage.frontal_parietal_full_band_coherence,
        ] {
            assert!((0.0..=1.0).contains(&value(coherence)));
        }
    }
}

#[test]
fn the_theta_peak_is_recovered_off_grid_and_the_wake_exponent_is_reported() {
    let analysis = analyze_night(&night(), &profile()).unwrap();
    let wake = &analysis.qeeg_by_stage[0];
    assert_eq!(wake.state, SleepState::Wake);

    let peak = value(&wake.theta_peak_frequency);
    assert!(
        (peak - THETA_CENTER_HZ).abs() <= 0.1,
        "theta peak {peak} vs {THETA_CENTER_HZ}"
    );

    let exponent = value(&wake.aperiodic_exponent);
    assert_eq!(unit(&wake.aperiodic_exponent), "dimensionless");
    assert!((exponent - 1.4).abs() < 0.2, "wake exponent {exponent}");
    assert_eq!(unit(&wake.aperiodic_offset), "log10_uV2_per_hz");
    assert_eq!(unit(&wake.spectral_fit_error), "log10_power");
    assert!(value(&wake.spectral_fit_error) <= 0.10);
}

#[test]
fn stages_outside_the_aperiodic_policy_report_an_explicit_not_applicable() {
    let analysis = analyze_night(&night(), &profile()).unwrap();
    for stage in &analysis.qeeg_by_stage[1..] {
        for feature in [
            &stage.aperiodic_exponent,
            &stage.aperiodic_offset,
            &stage.spectral_fit_error,
        ] {
            assert_eq!(reason(feature), NullReason::NotApplicable);
        }
        // Spectral features are still reported for these stages.
        assert!(value(&stage.delta_absolute_power) > 0.0);
    }
}

#[test]
fn a_stage_below_its_sufficiency_threshold_is_nulled_not_estimated() {
    // 59 REM epochs is 590 s, one epoch short of the 600 s threshold.
    let mut epochs = night();
    epochs.truncate(360 + 59);
    let analysis = analyze_night(&epochs, &profile()).unwrap();

    let rem = &analysis.qeeg_by_stage[2];
    assert_eq!(rem.state, SleepState::Rem);
    for feature in [
        &rem.delta_absolute_power,
        &rem.theta_relative_power,
        &rem.theta_peak_frequency,
        &rem.frontal_parietal_theta_coherence,
    ] {
        assert_eq!(reason(feature), NullReason::InsufficientStageDuration);
    }
    assert_eq!(
        reason(&analysis.stage_summary.rem_duration),
        NullReason::InsufficientStageDuration
    );
    assert_eq!(
        reason(&analysis.stage_summary.rem_bout_count),
        NullReason::InsufficientStageDuration
    );
    // The sufficient stages are unaffected.
    assert_eq!(value(&analysis.stage_summary.nrem_duration), 1800.0);
}

#[test]
fn stage_durations_and_bouts_count_only_admitted_epochs() {
    let analysis = analyze_night(&night(), &profile()).unwrap();
    assert_eq!(value(&analysis.stage_summary.wake_duration), 1800.0);
    assert_eq!(value(&analysis.stage_summary.nrem_duration), 1800.0);
    assert_eq!(value(&analysis.stage_summary.rem_duration), 600.0);
    assert_eq!(value(&analysis.stage_summary.rem_bout_count), 1.0);
    assert_eq!(
        value(&analysis.stage_summary.nrem_mean_bout_duration),
        1800.0
    );

    // A rejected epoch splits the run it sits in rather than bridging it.
    let mut epochs = night();
    epochs[270].channels_uv[0] = vec![0.0; EPOCH_SAMPLES];
    let split = analyze_night(&epochs, &profile()).unwrap();
    assert_eq!(
        split.rejections[270],
        Some(ruv_neural_signal::quality::EpochRejection::Flatline)
    );
    let aggregate = StageAggregate::new(&epochs, &split.accepted, 10.0);
    assert_eq!(aggregate.bouts(SleepState::Nrem), [900.0, 890.0]);
    assert_eq!(aggregate.duration(SleepState::Nrem), 1790.0);
}

#[test]
fn insufficient_coverage_is_reported_without_suppressing_the_measurements() {
    let mut epochs = night();
    for epoch in epochs.iter_mut().take(60) {
        epoch.channels_uv[0] = vec![0.0; EPOCH_SAMPLES];
    }
    let analysis = analyze_night(&epochs, &profile()).unwrap();
    assert!(!analysis.quality.accepted);
    assert_eq!(
        analysis.quality.reason_codes,
        ["insufficient_valid_coverage"]
    );
    assert!((analysis.quality.valid_coverage_fraction - 360.0 / 420.0).abs() < 1e-12);
    // Wake drops to 1200 s, below its 1800 s threshold, so it nulls out.
    assert_eq!(
        reason(&analysis.qeeg_by_stage[0].delta_absolute_power),
        NullReason::InsufficientStageDuration
    );
}

#[test]
fn epochs_that_disagree_with_the_profile_are_rejected_before_any_dsp_runs() {
    let profile = profile();
    let mut short = night();
    short[0].channels_uv[0].truncate(100);
    assert!(matches!(
        analyze_night(&short, &profile),
        Err(NeuroSleepError::InvalidEpoch(_))
    ));

    let mut maskless = night();
    maskless[3].shared_valid_mask.truncate(10);
    assert!(matches!(
        analyze_night(&maskless, &profile),
        Err(NeuroSleepError::InvalidEpoch(_))
    ));

    let mut channelless = night();
    channelless[5].channels_uv.clear();
    assert!(matches!(
        analyze_night(&channelless, &profile),
        Err(NeuroSleepError::InvalidEpoch(_))
    ));

    assert!(matches!(
        analyze_night(&[], &profile),
        Err(NeuroSleepError::InvalidEpoch(_))
    ));
}

#[test]
fn an_out_of_domain_profile_is_refused_before_any_epoch_is_touched() {
    let epochs = night();
    let broken: Vec<(&str, NeuroSleepProfile)> = vec![
        (
            "sample rate",
            NeuroSleepProfile {
                sample_rate_hz: 0.0,
                ..profile()
            },
        ),
        (
            "epoch length",
            NeuroSleepProfile {
                epoch_seconds: f64::NAN,
                ..profile()
            },
        ),
        (
            "coverage",
            NeuroSleepProfile {
                minimum_valid_coverage: 1.5,
                ..profile()
            },
        ),
        (
            "window longer than epoch",
            NeuroSleepProfile {
                welch: WelchWindow::oversized(),
                ..profile()
            },
        ),
        (
            "resolution vs theta profile",
            NeuroSleepProfile {
                theta_peak: ruv_neural_signal::neurosleep::ThetaPeakConfig {
                    maximum_bin_spacing_hz: 0.01,
                    ..profile().theta_peak
                },
                ..profile()
            },
        ),
        (
            "band above Nyquist",
            NeuroSleepProfile {
                bands: BandProfile {
                    relative_denominator:
                        ruv_neural_signal::neurosleep::FrequencyBandConfig::half_open(0.5, 200.0),
                    ..profile().bands
                },
                ..profile()
            },
        ),
    ];
    for (label, profile) in broken {
        assert!(
            matches!(profile.validate(), Err(NeuroSleepError::InvalidProfile(_))),
            "accepted an invalid profile: {label}"
        );
        assert!(
            analyze_night(&epochs, &profile).is_err(),
            "analysed with an invalid profile: {label}"
        );
    }
}

/// Helper so the oversized-window case stays readable above.
struct WelchWindow;
impl WelchWindow {
    fn oversized() -> ruv_neural_signal::neurosleep::WelchConfig {
        ruv_neural_signal::neurosleep::WelchConfig {
            window_samples: 5_000,
            overlap_samples: 2_500,
            detrend_mean: true,
        }
    }
}

fn acquisition() -> AcquisitionMetadata {
    AcquisitionMetadata {
        device_model: "synthetic-rig".into(),
        hardware_version: Some("1".into()),
        firmware_version: Some("1.0.0".into()),
        sampling_rate_hz: RATE,
        channels: vec![
            AcquisitionChannel {
                name: "frontal".into(),
                modality: AcquisitionModality::Eeg,
                reference: "cerebellar".into(),
            },
            AcquisitionChannel {
                name: "parietal".into(),
                modality: AcquisitionModality::Eeg,
                reference: "cerebellar".into(),
            },
        ],
    }
}

fn algorithm() -> AlgorithmManifest {
    AlgorithmManifest {
        pipeline_commit: "0".repeat(40),
        crate_versions: vec!["ruv-neural-neurosleep@0.1.0".into()],
        extractor_sha256: "22".repeat(32),
        configuration_sha256: "33".repeat(32),
        stage_source: StageSource::ExpertHypnogram {
            scorer_type: "human_scored_10_second".into(),
        },
        dsp_parameters: vec![AlgorithmParameter {
            name: "epoch_duration".into(),
            value: "10".into(),
            unit: "s".into(),
        }],
    }
}

fn context() -> NightBundleContext {
    NightBundleContext {
        bundle_id: "bundle-synthetic-001".into(),
        species: Species::Mouse,
        study_id: "synthetic-night".into(),
        subject_pseudonym: "subject-synthetic-001".into(),
        recording_id: "recording-synthetic-001".into(),
        night_start_ms: 1_700_000_000_000,
        night_end_ms: 1_700_004_200_000,
        nonce: "nonce-synthetic-001".into(),
        consent_scope: vec!["local_neurosleep_research_v1".into()],
        source_artifact_sha256: "11".repeat(32),
        source_format: SourceFormat::EdfPlus,
        source_byte_count: 4_096,
        acquisition: acquisition(),
        algorithm: algorithm(),
        literature_context: vec![ResearchCitation {
            identifier: "PMID:42252510".into(),
            title: "Synthetic NeuroSleep night".into(),
            evidence_maturity: "preclinical_mouse_model".into(),
        }],
    }
}

#[test]
fn a_signed_bundle_validates_verifies_and_carries_a_derived_fingerprint() {
    let signer = PersistentEd25519Signer::from_bytes("study-key-2026-01", &[7; 32]).unwrap();
    let bundle = analyze_and_sign(context(), &night(), &profile(), &signer).unwrap();

    // The contract validates every unit and every field; reaching here means the
    // emitted units match the registry the core crate enforces.
    bundle.payload.validate().unwrap();
    verify_neurosleep_bundle(&bundle, &signer.verifying_key_bytes()).unwrap();

    assert_eq!(
        bundle.payload.compatibility_fingerprint,
        compatibility_fingerprint_v1(&acquisition(), &algorithm()).unwrap()
    );
    assert_eq!(bundle.signer_key_id, "study-key-2026-01");
    assert_eq!(
        bundle.payload.payload_sha256().unwrap(),
        bundle.payload_sha256
    );

    // A different enrolled key must not verify the same bundle.
    let other = PersistentEd25519Signer::from_bytes("other-key", &[9; 32]).unwrap();
    assert!(verify_neurosleep_bundle(&bundle, &other.verifying_key_bytes()).is_err());
}

#[test]
fn signing_is_deterministic_and_depends_only_on_the_injected_signer() {
    let signer = PersistentEd25519Signer::from_bytes("study-key-2026-01", &[7; 32]).unwrap();
    let first = analyze_and_sign(context(), &night(), &profile(), &signer).unwrap();
    let second = analyze_and_sign(context(), &night(), &profile(), &signer).unwrap();
    assert_eq!(first, second);

    let other = PersistentEd25519Signer::from_bytes("study-key-2026-02", &[7; 32]).unwrap();
    let third = analyze_and_sign(context(), &night(), &profile(), &other).unwrap();
    assert_eq!(third.payload_sha256, first.payload_sha256);
    assert_ne!(third.signature_ed25519, first.signature_ed25519);
}
