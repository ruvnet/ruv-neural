//! Criterion benchmarks over a synthetic expert-scored night.
//!
//! The night is 420 epochs — 180 Wake, 180 NREM, 60 REM — which is 70 minutes at
//! the paper profile's 10-second epochs, not the 8 hours (2880 epochs) a real
//! recording spans. 420 is the smallest night that clears every sufficiency
//! threshold in [`NeuroSleepProfile::constantino_250hz`], so it exercises the
//! full measurement path (Welch, coherence, aperiodic fit, theta peak, staging)
//! for all three stages while keeping one criterion iteration under a second
//! (~0.65 s here). Cost scales linearly in epoch count, so an 8-hour night is
//! about 7x these numbers.
//!
//! Signals are generated here rather than loaded: a sinusoid comb on the 0.2 Hz
//! Welch grid whose amplitudes follow the aperiodic model the fitter recovers.
//! No fixture file, no RNG, and no network access is involved.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::f64::consts::PI;
use std::time::Duration;

use ruv_neural_core::attestation::PersistentEd25519Signer;
use ruv_neural_core::neurosleep::{
    AcquisitionChannel, AcquisitionMetadata, AcquisitionModality, AlgorithmManifest,
    AlgorithmParameter, ResearchCitation, SleepState, SourceFormat, Species, StageSource,
};
use ruv_neural_neurosleep::{
    analyze_and_sign, analyze_night, ExpertEpoch, NeuroSleepProfile, NightBundleContext,
    StageAggregate,
};

const RATE_HZ: f64 = 250.0;
const EPOCH_SAMPLES: usize = 2_500;
const EPOCH_SECONDS: f64 = 10.0;
const COMB_STEP_HZ: f64 = 0.2;
const COMB_TOP_HZ: f64 = 45.0;
const THETA_CENTER_HZ: f64 = 6.35;

/// One 100-sample artefact burst per epoch: 4% of the epoch and a 0.4 s gap,
/// both inside the profile's admission gates, so the epoch is still accepted and
/// every spectral estimate runs the masked segment-dropping path.
const BURST: std::ops::Range<usize> = 1_000..1_100;

/// One epoch whose PSD follows `10^(3 - log10(5 + f^exponent))` scaled by a
/// Gaussian theta bump, built by summing sinusoids on the 0.2 Hz comb so every
/// component lands on a Welch bin centre. The phase ramp is a fixed irrational
/// multiple, which keeps the comb from collapsing into a periodic spike train
/// without any randomness.
fn epoch_waveform(exponent: f64, phase_shift: f64) -> Vec<f64> {
    let mut samples = vec![0.0; EPOCH_SAMPLES];
    let mut harmonic = 1;
    while harmonic as f64 * COMB_STEP_HZ <= COMB_TOP_HZ {
        let frequency = harmonic as f64 * COMB_STEP_HZ;
        let background = 10f64.powf(3.0 - (5.0 + frequency.powf(exponent)).log10());
        let bump = 10f64.powf(0.6 * (-0.5 * ((frequency - THETA_CENTER_HZ) / 0.8).powi(2)).exp());
        let amplitude = (2.0 * background * bump * COMB_STEP_HZ).sqrt();
        let phase = (harmonic as f64 * 2.399_963_2) % (2.0 * PI) + phase_shift;
        for (index, value) in samples.iter_mut().enumerate() {
            *value += amplitude * (2.0 * PI * frequency * index as f64 / RATE_HZ + phase).sin();
        }
        harmonic += 1;
    }
    samples
}

/// Every epoch of a stage shares one waveform, so the stage-averaged spectrum
/// equals the single-epoch spectrum and the analysis stays deterministic.
fn stage_epochs(state: SleepState, exponent: f64, count: usize, masked: bool) -> Vec<ExpertEpoch> {
    let mut frontal = epoch_waveform(exponent, 0.0);
    let parietal = epoch_waveform(exponent, 0.4);
    let mut mask = vec![true; EPOCH_SAMPLES];
    if masked {
        for index in BURST {
            frontal[index] = f64::NAN;
            mask[index] = false;
        }
    }
    vec![
        ExpertEpoch {
            state,
            channels_uv: vec![frontal, parietal],
            shared_valid_mask: mask,
        };
        count
    ]
}

/// 180 Wake + 180 NREM + 60 REM epochs: exactly the sufficiency thresholds.
fn night(masked: bool) -> Vec<ExpertEpoch> {
    let mut epochs = stage_epochs(SleepState::Wake, 1.4, 180, masked);
    epochs.extend(stage_epochs(SleepState::Nrem, 2.1, 180, masked));
    epochs.extend(stage_epochs(SleepState::Rem, 1.7, 60, masked));
    epochs
}

fn acquisition() -> AcquisitionMetadata {
    AcquisitionMetadata {
        device_model: "synthetic-bench-rig".into(),
        hardware_version: Some("1".into()),
        firmware_version: Some("1.0.0".into()),
        sampling_rate_hz: RATE_HZ,
        channels: ["frontal", "parietal"]
            .into_iter()
            .map(|name| AcquisitionChannel {
                name: name.into(),
                modality: AcquisitionModality::Eeg,
                reference: "cerebellar".into(),
            })
            .collect(),
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
        bundle_id: "bundle-bench-001".into(),
        species: Species::Mouse,
        study_id: "bench-night".into(),
        subject_pseudonym: "subject-bench-001".into(),
        recording_id: "recording-bench-001".into(),
        night_start_ms: 1_700_000_000_000,
        night_end_ms: 1_700_004_200_000,
        nonce: "nonce-bench-001".into(),
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

/// The key material is a fixed byte pattern held only in this file. Signing is
/// deterministic, so it contributes a constant to every iteration.
fn signer() -> PersistentEd25519Signer {
    PersistentEd25519Signer::from_bytes("bench-key-0001", &[7; 32]).expect("valid bench key id")
}

fn full_night(criterion: &mut Criterion) {
    let profile = NeuroSleepProfile::constantino_250hz();
    let clean = night(false);
    let masked = night(true);
    let context = context();
    let signer = signer();

    // Fail loudly at setup rather than benchmarking a degraded path: a night that
    // silently nulled its stages would measure staging alone, not the DSP.
    let baseline = analyze_night(&clean, &profile).expect("synthetic night analyses");
    assert!(baseline.quality.accepted, "{:?}", baseline.quality);
    assert!(
        analyze_night(&masked, &profile)
            .expect("masked night analyses")
            .quality
            .accepted,
        "the per-epoch burst must stay inside the admission gates"
    );

    // One iteration is ~0.65 s, so criterion's default 100 samples would run for
    // over a minute per benchmark.
    let mut group = criterion.benchmark_group("neurosleep_full_night");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(15));

    group.bench_function("analyze_night_420_epochs", |bencher| {
        bencher.iter(|| analyze_night(black_box(&clean), black_box(&profile)).unwrap());
    });
    // Against the clean night above, the delta is what per-epoch masking costs:
    // the burst invalidates two of each epoch's three Welch and coherence
    // segments, so fewer FFTs run but every segment is still scanned.
    group.bench_function("analyze_night_420_epochs_masked", |bencher| {
        bencher.iter(|| analyze_night(black_box(&masked), black_box(&profile)).unwrap());
    });
    // Against the clean night, the delta is canonicalisation, the compatibility
    // fingerprint, contract validation, and one Ed25519 signature.
    group.bench_function("analyze_and_sign_420_epochs", |bencher| {
        bencher.iter(|| {
            analyze_and_sign(
                context.clone(),
                black_box(&clean),
                black_box(&profile),
                &signer,
            )
            .unwrap()
        });
    });
    group.finish();

    // Staging is cheap enough to keep criterion's defaults.
    criterion.bench_function("neurosleep_stage_aggregate_420_epochs", |bencher| {
        bencher.iter(|| {
            StageAggregate::new(
                black_box(&clean),
                black_box(&baseline.accepted),
                EPOCH_SECONDS,
            )
        });
    });
}

criterion_group!(night_benches, full_night);
criterion_main!(night_benches);
