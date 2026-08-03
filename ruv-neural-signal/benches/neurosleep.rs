//! Criterion benchmarks for the NeuroSleep DSP path.
//!
//! Everything runs at the paper profile: 250 Hz, 10-second (2500-sample) epochs,
//! a 1250-sample Welch window with 625 samples of overlap. Signals are generated
//! here rather than loaded: a sinusoid comb on the 0.2 Hz Welch grid whose
//! amplitudes follow the same `offset - log10(knee + f^exponent)` model the
//! aperiodic fitter recovers, plus a Gaussian theta bump. No fixture file, no
//! RNG, and no network access is involved, so every run measures the same work.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::f64::consts::PI;
use std::time::Duration;

use ruv_neural_signal::aperiodic::{fit_aperiodic, AperiodicConfig, AperiodicMode};
use ruv_neural_signal::connectivity::{coherence_masked, CoherenceConfig};
use ruv_neural_signal::neurosleep::{
    integrate_band_trapezoidal, periodic_theta_peak, relative_band_power, welch_psd,
    welch_psd_masked, BandEdgePolicy, FrequencyBandConfig, PowerSpectrum, RelativePowerConfig,
    ThetaPeakConfig, WelchConfig,
};
use ruv_neural_signal::quality::{assess_epoch, AdcRails, EpochQualityConfig};

const RATE_HZ: f64 = 250.0;
const EPOCH_SAMPLES: usize = 2_500;
const COMB_STEP_HZ: f64 = 0.2;
const COMB_TOP_HZ: f64 = 45.0;
const THETA_CENTER_HZ: f64 = 6.35;
const APERIODIC_OFFSET: f64 = 3.0;
const APERIODIC_KNEE: f64 = 5.0;
const APERIODIC_EXPONENT: f64 = 1.4;

/// The artefact burst used by every partially-masked benchmark: 100 samples,
/// 4% of the epoch, well inside the admission gates.
const BURST: std::ops::Range<usize> = 1_000..1_100;

const WELCH: WelchConfig = WelchConfig {
    window_samples: 1_250,
    overlap_samples: 625,
    detrend_mean: true,
};

const COHERENCE: CoherenceConfig = CoherenceConfig {
    window_samples: 1_250,
    overlap_samples: 625,
    detrend_mean: true,
};

const THETA_PEAK: ThetaPeakConfig = ThetaPeakConfig {
    low_hz: 4.0,
    high_hz: 8.0,
    minimum_log10_prominence: 0.05,
    maximum_bin_spacing_hz: 0.25,
};

const QUALITY: EpochQualityConfig = EpochQualityConfig {
    adc_rails: AdcRails {
        minimum_uv: -1_000.0,
        maximum_uv: 1_000.0,
        tolerance_uv: 0.5,
    },
    maximum_clipped_fraction: 0.001,
    flatline_seconds: 1.0,
    flatline_epsilon_uv: 1e-6,
    maximum_gap_seconds: 1.0,
    maximum_artifact_fraction: 0.20,
};

const DELTA: FrequencyBandConfig = FrequencyBandConfig::half_open(0.5, 4.0);
const THETA: FrequencyBandConfig = FrequencyBandConfig::half_open(4.0, 8.0);
const FULL_BAND: FrequencyBandConfig = FrequencyBandConfig::half_open(0.5, 40.0);

fn aperiodic_config(mode: AperiodicMode) -> AperiodicConfig {
    AperiodicConfig {
        low_hz: 2.0,
        high_hz: 40.0,
        mode,
        minimum_exponent: 0.1,
        maximum_exponent: 4.0,
        maximum_knee: 30.0,
        grid_points: 13,
        refinement_rounds: 8,
        peak_removal_iterations: 2,
        peak_removal_percentile: 0.025,
        peak_removal_tolerance_log10: 0.01,
        minimum_r_squared: 0.90,
        maximum_rmse_log10: 0.10,
    }
}

/// The aperiodic background in log10(uV^2/Hz), i.e. the model the fitter targets.
fn aperiodic_log10(frequency_hz: f64) -> f64 {
    APERIODIC_OFFSET - (APERIODIC_KNEE + frequency_hz.powf(APERIODIC_EXPONENT)).log10()
}

/// That background scaled by a Gaussian theta bump, in uV^2/Hz.
fn target_density(frequency_hz: f64) -> f64 {
    let bump = 10f64.powf(0.6 * (-0.5 * ((frequency_hz - THETA_CENTER_HZ) / 0.8).powi(2)).exp());
    10f64.powf(aperiodic_log10(frequency_hz)) * bump
}

/// One epoch whose PSD follows [`target_density`], built by summing sinusoids on
/// the 0.2 Hz comb so every component lands on a Welch bin centre and leaks into
/// no other bin. The phase ramp is a fixed irrational multiple, which keeps the
/// comb from collapsing into a periodic spike train without any randomness.
fn epoch_waveform(phase_shift: f64) -> Vec<f64> {
    let mut samples = vec![0.0; EPOCH_SAMPLES];
    let mut harmonic = 1;
    while harmonic as f64 * COMB_STEP_HZ <= COMB_TOP_HZ {
        let frequency = harmonic as f64 * COMB_STEP_HZ;
        let amplitude = (2.0 * target_density(frequency) * COMB_STEP_HZ).sqrt();
        let phase = (harmonic as f64 * 2.399_963_2) % (2.0 * PI) + phase_shift;
        for (index, value) in samples.iter_mut().enumerate() {
            *value += amplitude * (2.0 * PI * frequency * index as f64 / RATE_HZ + phase).sin();
        }
        harmonic += 1;
    }
    samples
}

/// The same epoch carrying one artefact burst, with NaN left in the gap to prove
/// the masked path never reads it.
///
/// A 2500-sample epoch holds three 1250-sample segments, at offsets 0, 625, and
/// 1250. A burst over `[1000, 1100)` touches the first two, so the masked path
/// walks and rejects two segments before averaging the one that survives.
fn burst_epoch(phase_shift: f64) -> (Vec<f64>, Vec<bool>) {
    let mut signal = epoch_waveform(phase_shift);
    let mut mask = vec![true; EPOCH_SAMPLES];
    for index in BURST {
        signal[index] = f64::NAN;
        mask[index] = false;
    }
    (signal, mask)
}

fn epoch_spectrum() -> PowerSpectrum {
    welch_psd(&epoch_waveform(0.0), RATE_HZ, WELCH).expect("paper-profile Welch PSD")
}

fn welch(criterion: &mut Criterion) {
    let signal = epoch_waveform(0.0);
    let all_valid = vec![true; EPOCH_SAMPLES];
    let (burst, burst_mask) = burst_epoch(0.0);

    let mut group = criterion.benchmark_group("neurosleep_welch");
    group.bench_function("welch_psd_epoch", |bencher| {
        bencher.iter(|| welch_psd(black_box(&signal), RATE_HZ, WELCH).unwrap());
    });
    group.bench_function("welch_psd_masked_all_valid", |bencher| {
        bencher.iter(|| welch_psd_masked(black_box(&signal), &all_valid, RATE_HZ, WELCH).unwrap());
    });
    // Against the all-valid case above, the delta is what the burst costs: two of
    // the three segments are scanned and then dropped instead of transformed.
    group.bench_function("welch_psd_masked_artifact_burst", |bencher| {
        bencher.iter(|| welch_psd_masked(black_box(&burst), &burst_mask, RATE_HZ, WELCH).unwrap());
    });
    group.finish();
}

fn band_power(criterion: &mut Criterion) {
    let spectrum = epoch_spectrum();
    let (grid, density) = (&spectrum.frequencies_hz, &spectrum.density);
    let integrate = |policy| integrate_band_trapezoidal(grid, black_box(density), DELTA, policy);
    let relative = RelativePowerConfig {
        numerator: THETA,
        denominator: FULL_BAND,
        edge_policy: BandEdgePolicy::InterpolatedEdges,
    };

    let mut group = criterion.benchmark_group("neurosleep_band_power");
    group.bench_function("integrate_delta_interpolated_edges", |bencher| {
        bencher.iter(|| integrate(BandEdgePolicy::InterpolatedEdges).unwrap());
    });
    group.bench_function("integrate_delta_bin_inclusive", |bencher| {
        bencher.iter(|| integrate(BandEdgePolicy::BinInclusive).unwrap());
    });
    group.bench_function("relative_theta_over_full_band", |bencher| {
        bencher.iter(|| relative_band_power(grid, black_box(density), relative).unwrap());
    });
    group.finish();
}

fn theta_peak(criterion: &mut Criterion) {
    let spectrum = epoch_spectrum();
    // The generating aperiodic model on the Welch grid. Using it rather than a
    // fit keeps this benchmark measuring the peak search alone.
    let background: Vec<f64> = spectrum
        .frequencies_hz
        .iter()
        .map(|frequency| {
            if *frequency > 0.0 {
                aperiodic_log10(*frequency)
            } else {
                f64::NAN
            }
        })
        .collect();
    let peak = || {
        periodic_theta_peak(
            &spectrum.frequencies_hz,
            &spectrum.density,
            &background,
            THETA_PEAK,
        )
    };
    peak().expect("the synthetic theta bump clears the acceptance profile");

    criterion.bench_function("neurosleep_periodic_theta_peak", |bencher| {
        bencher.iter(|| black_box(peak()).unwrap());
    });
}

fn aperiodic(criterion: &mut Criterion) {
    let spectrum = epoch_spectrum();
    let (grid, density) = (&spectrum.frequencies_hz, &spectrum.density);
    fit_aperiodic(grid, density, aperiodic_config(AperiodicMode::Knee))
        .expect("the synthetic spectrum clears the knee-fit quality gates");

    // A knee fit runs three bounded grid searches of ~2M model evaluations each,
    // so criterion's default 100 samples would dominate the whole suite.
    let mut group = criterion.benchmark_group("neurosleep_aperiodic");
    group.sample_size(10).warm_up_time(Duration::from_secs(1));
    for (label, mode) in [
        ("knee", AperiodicMode::Knee),
        ("fixed", AperiodicMode::Fixed),
    ] {
        group.bench_function(label, |bencher| {
            bencher.iter(|| fit_aperiodic(grid, black_box(density), aperiodic_config(mode)));
        });
    }
    group.finish();
}

fn coherence(criterion: &mut Criterion) {
    let frontal = epoch_waveform(0.0);
    let parietal = epoch_waveform(0.4);
    let all_valid = vec![true; EPOCH_SAMPLES];
    let (burst, burst_mask) = burst_epoch(0.0);

    let mut group = criterion.benchmark_group("neurosleep_coherence");
    group.bench_function("coherence_masked_all_valid", |bencher| {
        bencher.iter(|| {
            coherence_masked(
                black_box(&frontal),
                &parietal,
                &all_valid,
                RATE_HZ,
                COHERENCE,
            )
            .unwrap()
        });
    });
    group.bench_function("coherence_masked_artifact_burst", |bencher| {
        bencher.iter(|| {
            coherence_masked(
                black_box(&burst),
                &parietal,
                &burst_mask,
                RATE_HZ,
                COHERENCE,
            )
            .unwrap()
        });
    });
    group.finish();
}

fn quality(criterion: &mut Criterion) {
    let frontal = epoch_waveform(0.0);
    let parietal = epoch_waveform(0.4);
    let all_valid = vec![true; EPOCH_SAMPLES];
    let (burst, burst_mask) = burst_epoch(0.0);
    let clean: Vec<&[f64]> = vec![&frontal, &parietal];
    let bursty: Vec<&[f64]> = vec![&burst, &parietal];

    // The flatline gate scans a 250-sample window at every offset of every
    // channel, so one assessment is ~1 ms and criterion's default 5 s budget
    // falls short of its default 100 samples.
    let mut group = criterion.benchmark_group("neurosleep_quality");
    group.measurement_time(Duration::from_secs(8));
    group.bench_function("assess_epoch_two_channels", |bencher| {
        bencher.iter(|| assess_epoch(black_box(&clean), &all_valid, RATE_HZ, QUALITY).unwrap());
    });
    group.bench_function("assess_epoch_artifact_burst", |bencher| {
        bencher.iter(|| assess_epoch(black_box(&bursty), &burst_mask, RATE_HZ, QUALITY).unwrap());
    });
    group.finish();
}

criterion_group!(neurosleep, welch, band_power, theta_peak, aperiodic, coherence, quality);
criterion_main!(neurosleep);
