//! Synthetic-signal tests for the NeuroSleep spectral primitives.
//!
//! These are self-consistency and analytic-recovery tests. They do not assert
//! numeric parity against any external toolbox; no pinned reference vectors are
//! available in-repo.

use super::*;
use approx::assert_abs_diff_eq;

const RATE: f64 = 250.0;

fn welch() -> WelchConfig {
    WelchConfig {
        window_samples: 1250,
        overlap_samples: 625,
        detrend_mean: true,
    }
}

fn sine(frequency_hz: f64, amplitude_uv: f64, samples: usize) -> Vec<f64> {
    (0..samples)
        .map(|index| amplitude_uv * (2.0 * PI * frequency_hz * index as f64 / RATE).sin())
        .collect()
}

#[test]
fn welch_recovers_sine_power_as_absolute_micro_volts_squared() {
    // A pure sine of amplitude A has mean-square power A^2/2, which is what the
    // integrated PSD over a band containing the tone must return, in uV^2.
    let amplitude = 20.0;
    let signal = sine(10.0, amplitude, 10_000);
    let spectrum = welch_psd(&signal, RATE, welch()).unwrap();
    let power = integrate_band_trapezoidal(
        &spectrum.frequencies_hz,
        &spectrum.density,
        FrequencyBandConfig::half_open(8.0, 13.0),
        BandEdgePolicy::InterpolatedEdges,
    )
    .unwrap();
    assert_eq!(AbsoluteBandPower::UNIT, "uV2");
    assert_abs_diff_eq!(
        power.micro_volts_squared,
        amplitude * amplitude / 2.0,
        epsilon = 4.0
    );
}

#[test]
fn masked_welch_drops_segments_touching_invalid_samples() {
    let clean = sine(10.0, 20.0, 10_000);
    let mut corrupted = clean.clone();
    let mut mask = vec![true; clean.len()];
    // Destroy the tail with values a naive estimator would happily average in.
    for index in 5_000..clean.len() {
        corrupted[index] = f64::NAN;
        mask[index] = false;
    }
    let masked = welch_psd_masked(&corrupted, &mask, RATE, welch()).unwrap();
    let reference = welch_psd(&clean[..5_000], RATE, welch()).unwrap();
    assert_eq!(masked.segments, reference.segments);
    for (a, b) in masked.density.iter().zip(&reference.density) {
        assert_abs_diff_eq!(a, b, epsilon = 1e-12);
    }
    // Unmasked NaN must be a typed error, never a silently NaN spectrum.
    assert!(matches!(
        welch_psd(&corrupted, RATE, welch()),
        Err(DspError::NonFiniteSample(5_000))
    ));
}

#[test]
fn masked_welch_reports_no_valid_segments_rather_than_an_empty_spectrum() {
    let signal = sine(10.0, 20.0, 10_000);
    let mut mask = vec![true; signal.len()];
    // One invalid sample every 600 samples leaves no fully valid 1250-sample window.
    for index in (0..signal.len()).step_by(600) {
        mask[index] = false;
    }
    assert_eq!(
        welch_psd_masked(&signal, &mask, RATE, welch()),
        Err(DspError::NoValidSegments)
    );
}

#[test]
fn welch_and_band_configs_reject_out_of_domain_values() {
    assert!(WelchConfig {
        window_samples: 3,
        overlap_samples: 0,
        detrend_mean: true
    }
    .validate()
    .is_err());
    assert!(WelchConfig {
        window_samples: 64,
        overlap_samples: 64,
        detrend_mean: true
    }
    .validate()
    .is_err());
    assert!(FrequencyBandConfig::half_open(4.0, 4.0).validate().is_err());
    assert!(FrequencyBandConfig::half_open(f64::NAN, 4.0)
        .validate()
        .is_err());
    let signal = sine(10.0, 1.0, 4_000);
    assert!(welch_psd(&signal, f64::INFINITY, welch()).is_err());
    assert!(welch_psd(&signal, 0.0, welch()).is_err());
    assert!(matches!(
        welch_psd(&signal[..10], RATE, welch()),
        Err(DspError::InsufficientSamples { needed: 1250, .. })
    ));
}

#[test]
fn band_edge_policies_are_distinguishable_and_explicit() {
    // Flat unit density on a 1 Hz grid: interpolated edges integrate exactly
    // [0.5, 4.0) width 3.5; bin-inclusive covers only bins 1..=3, width 2.
    let frequencies: Vec<f64> = (0..10).map(f64::from).collect();
    let density = vec![1.0; 10];
    let band = FrequencyBandConfig::half_open(0.5, 4.0);
    let interpolated = integrate_band_trapezoidal(
        &frequencies,
        &density,
        band,
        BandEdgePolicy::InterpolatedEdges,
    )
    .unwrap();
    let bin_inclusive =
        integrate_band_trapezoidal(&frequencies, &density, band, BandEdgePolicy::BinInclusive)
            .unwrap();
    assert_abs_diff_eq!(interpolated.micro_volts_squared, 3.5, epsilon = 1e-12);
    assert_abs_diff_eq!(bin_inclusive.micro_volts_squared, 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(interpolated.mean_density(), 1.0, epsilon = 1e-12);
    assert_eq!(interpolated.edge_policy, BandEdgePolicy::InterpolatedEdges);
    assert_eq!(interpolated.band, band);
}

#[test]
fn relative_power_carries_its_denominator_and_rejects_an_escaping_numerator() {
    let frequencies: Vec<f64> = (0..50).map(f64::from).collect();
    let density = vec![2.0; 50];
    let config = RelativePowerConfig {
        numerator: FrequencyBandConfig::half_open(0.5, 4.0),
        denominator: FrequencyBandConfig::half_open(0.5, 40.0),
        edge_policy: BandEdgePolicy::InterpolatedEdges,
    };
    let relative = relative_band_power(&frequencies, &density, config).unwrap();
    assert_eq!(RelativeBandPower::UNIT, "ratio");
    assert_abs_diff_eq!(relative.ratio, 3.5 / 39.5, epsilon = 1e-12);
    assert_eq!(relative.denominator.band, config.denominator);
    assert_abs_diff_eq!(
        relative.denominator.micro_volts_squared,
        2.0 * 39.5,
        epsilon = 1e-12
    );

    let escaping = RelativePowerConfig {
        numerator: FrequencyBandConfig::half_open(0.5, 45.0),
        ..config
    };
    assert!(relative_band_power(&frequencies, &density, escaping).is_err());

    let zero = vec![0.0; 50];
    assert_eq!(
        relative_band_power(&frequencies, &zero, config),
        Err(DspError::NonPositiveDenominator)
    );
}

fn theta_peak_config() -> ThetaPeakConfig {
    ThetaPeakConfig {
        low_hz: 4.0,
        high_hz: 8.0,
        minimum_log10_prominence: 0.05,
        maximum_bin_spacing_hz: 0.25,
    }
}

#[test]
fn theta_peak_is_interpolated_to_within_a_tenth_of_a_hertz() {
    // Sweep true peak frequencies deliberately off the 0.2 Hz Welch grid.
    for step in 0..21 {
        let truth = 5.0 + 0.05 * f64::from(step);
        let mut signal = sine(truth, 25.0, 25_000);
        // Pink-ish background so the peak sits above a sloped aperiodic floor.
        for (index, value) in signal.iter_mut().enumerate() {
            *value += 12.0 * (2.0 * PI * 1.0 * index as f64 / RATE).sin();
        }
        let spectrum = welch_psd(&signal, RATE, welch()).unwrap();
        assert_abs_diff_eq!(spectrum.bin_spacing_hz().unwrap(), 0.2, epsilon = 1e-12);
        // Flat background: the residual is then just the log spectrum shape.
        let background = vec![-6.0; spectrum.density.len()];
        let peak = periodic_theta_peak(
            &spectrum.frequencies_hz,
            &spectrum.density,
            &background,
            theta_peak_config(),
        )
        .unwrap();
        assert!(
            (peak.center_frequency_hz - truth).abs() <= 0.1,
            "peak {} vs truth {truth}",
            peak.center_frequency_hz
        );
        assert_abs_diff_eq!(peak.bin_spacing_hz, 0.2, epsilon = 1e-12);
    }
}

#[test]
fn theta_peak_rejects_coarse_grids_and_unprominent_peaks() {
    let coarse: Vec<f64> = (0..20).map(f64::from).collect();
    let density = vec![1.0; 20];
    let background = vec![0.0; 20];
    assert!(matches!(
        periodic_theta_peak(&coarse, &density, &background, theta_peak_config()),
        Err(DspError::InvalidConfig(_))
    ));

    let fine: Vec<f64> = (0..100).map(|bin| f64::from(bin) * 0.2).collect();
    let flat = vec![1.0; 100];
    let matched = vec![0.0; 100];
    assert_eq!(
        periodic_theta_peak(&fine, &flat, &matched, theta_peak_config()),
        Err(DspError::NoAcceptedPeak)
    );
    assert!(ThetaPeakConfig {
        maximum_bin_spacing_hz: 0.0,
        ..theta_peak_config()
    }
    .validate()
    .is_err());
    assert!(ThetaPeakConfig {
        low_hz: 8.0,
        ..theta_peak_config()
    }
    .validate()
    .is_err());
}
