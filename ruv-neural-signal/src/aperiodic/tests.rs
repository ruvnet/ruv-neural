//! Recovery tests against spectra generated from the FOOOF model itself.
//!
//! These verify that the implementation inverts its own documented forward
//! model and that peaks do not drag the exponent. They are not parity tests
//! against a FOOOF reference run; see the module docs.

use super::*;

fn knee_config() -> AperiodicConfig {
    AperiodicConfig {
        low_hz: 2.0,
        high_hz: 40.0,
        mode: AperiodicMode::Knee,
        minimum_exponent: 0.1,
        maximum_exponent: 4.0,
        maximum_knee: 30.0,
        grid_points: 17,
        refinement_rounds: 14,
        peak_removal_iterations: 2,
        peak_removal_percentile: 0.025,
        peak_removal_tolerance_log10: 0.01,
        minimum_r_squared: 0.90,
        maximum_rmse_log10: 0.10,
    }
}

fn fixed_config() -> AperiodicConfig {
    AperiodicConfig {
        mode: AperiodicMode::Fixed,
        ..knee_config()
    }
}

/// Forward model: `10^(offset - log10(knee + f^exponent))`, i.e. linear power.
fn synthetic(offset: f64, knee: f64, exponent: f64) -> (Vec<f64>, Vec<f64>) {
    let frequencies: Vec<f64> = (0..401).map(|bin| f64::from(bin) * 0.2).collect();
    let density = frequencies
        .iter()
        .map(|frequency| {
            if *frequency > 0.0 {
                10f64.powf(offset - (knee + frequency.powf(exponent)).log10())
            } else {
                0.0
            }
        })
        .collect();
    (frequencies, density)
}

fn add_gaussian_peak(
    frequencies: &[f64],
    density: &mut [f64],
    center_hz: f64,
    log10_height: f64,
    width_hz: f64,
) {
    for (frequency, power) in frequencies.iter().zip(density.iter_mut()) {
        let bump = log10_height * (-0.5 * ((frequency - center_hz) / width_hz).powi(2)).exp();
        *power *= 10f64.powf(bump);
    }
}

#[test]
fn knee_model_recovers_its_own_parameters() {
    for (offset, knee, exponent) in [
        (1.5, 5.0, 2.0),
        (-0.5, 0.5, 1.2),
        (2.0, 20.0, 3.0),
        (0.0, 0.0, 1.0),
    ] {
        let (frequencies, density) = synthetic(offset, knee, exponent);
        let fit = fit_aperiodic(&frequencies, &density, knee_config()).unwrap();
        assert!(
            (fit.exponent - exponent).abs() < 0.05,
            "exponent {} vs {exponent}",
            fit.exponent
        );
        assert!(
            (fit.offset - offset).abs() < 0.05,
            "offset {} vs {offset}",
            fit.offset
        );
        assert!(
            (fit.knee.unwrap() - knee).abs() < 0.01,
            "knee {:?}",
            fit.knee
        );
        assert!(fit.r_squared > 0.999, "r2 {}", fit.r_squared);
        // A spectrum generated from the model must be reproduced by the model,
        // so anything above round-off means the search stalled short of the
        // optimum rather than the parameters being unidentifiable.
        assert!(fit.rmse_log10 < 1e-6, "rmse {}", fit.rmse_log10);
    }
}

#[test]
fn knee_model_is_not_a_straight_line_in_log_knee_plus_frequency() {
    // Guards against reintroducing the linear-in-log10(knee + f) shortcut: that
    // model cannot reproduce a spectrum generated with the exponent inside the
    // logarithm, so a fit that recovers the true exponent must be the real one.
    let (frequencies, density) = synthetic(1.5, 8.0, 2.5);
    let fit = fit_aperiodic(&frequencies, &density, knee_config()).unwrap();
    assert!((fit.exponent - 2.5).abs() < 0.05, "{}", fit.exponent);

    // The shortcut's best straight line through (log10(knee + f), log10 power)
    // has slope -1 by construction, regardless of the true exponent.
    let shortcut_predicts_unit_exponent = 1.0;
    assert!((fit.exponent - shortcut_predicts_unit_exponent).abs() > 1.0);
}

#[test]
fn fixed_model_recovers_a_pure_power_law() {
    let (frequencies, density) = synthetic(2.0, 0.0, 1.7);
    let fit = fit_aperiodic(&frequencies, &density, fixed_config()).unwrap();
    assert!((fit.exponent - 1.7).abs() < 0.02, "{}", fit.exponent);
    assert!((fit.offset - 2.0).abs() < 0.02, "{}", fit.offset);
    assert_eq!(fit.knee, None);
}

#[test]
fn robust_peak_removal_keeps_a_large_theta_peak_out_of_the_exponent() {
    let (frequencies, mut density) = synthetic(1.5, 0.0, 1.5);
    add_gaussian_peak(&frequencies, &mut density, 6.0, 0.8, 1.0);

    // The naive fit needs a slack quality gate purely so it returns a fit to
    // compare against; a peak that large genuinely breaks the frozen gate.
    let naive = AperiodicConfig {
        peak_removal_iterations: 0,
        maximum_rmse_log10: 1.0,
        ..fixed_config()
    };
    let naive_fit = fit_aperiodic(&frequencies, &density, naive).unwrap();
    let robust_fit = fit_aperiodic(&frequencies, &density, fixed_config()).unwrap();

    assert!(
        (robust_fit.exponent - 1.5).abs() < (naive_fit.exponent - 1.5).abs(),
        "robust {} should beat naive {}",
        robust_fit.exponent,
        naive_fit.exponent
    );
    assert!(
        (robust_fit.exponent - 1.5).abs() < 0.1,
        "{}",
        robust_fit.exponent
    );
    // Peaks inflate the whole-band error but not the fit's own support error.
    assert!(robust_fit.full_band_rmse_log10 > robust_fit.rmse_log10);
    assert!(robust_fit.retained_bins < robust_fit.selected_bins);
}

#[test]
fn protrusion_tolerance_stops_peak_removal_splitting_a_peak_free_spectrum() {
    // With the guard disabled, the zero-clamp threshold discards every bin that
    // happens to land above the fit by a floating-point margin, and the refit
    // drifts. With the guard on, a peak-free spectrum keeps all of its bins.
    let (frequencies, density) = synthetic(1.5, 5.0, 2.0);
    let unguarded = AperiodicConfig {
        peak_removal_tolerance_log10: 0.0,
        ..knee_config()
    };
    let guarded = fit_aperiodic(&frequencies, &density, knee_config()).unwrap();
    let split = fit_aperiodic(&frequencies, &density, unguarded).unwrap();

    assert_eq!(guarded.retained_bins, guarded.selected_bins);
    assert!(split.retained_bins < split.selected_bins);

    // The guard must not blunt real peak removal: a genuine peak still goes.
    let (peaked_frequencies, mut peaked) = synthetic(1.5, 5.0, 2.0);
    add_gaussian_peak(&peaked_frequencies, &mut peaked, 6.0, 0.8, 1.0);
    let fit = fit_aperiodic(&peaked_frequencies, &peaked, knee_config()).unwrap();
    assert!(fit.retained_bins < fit.selected_bins);
}

#[test]
fn quality_gate_failure_is_typed_and_carries_its_numbers() {
    let frequencies: Vec<f64> = (1..200).map(|bin| f64::from(bin) * 0.2).collect();
    // Noise-like spectrum with no 1/f structure at all.
    let density: Vec<f64> = frequencies
        .iter()
        .enumerate()
        .map(|(index, _)| if index % 2 == 0 { 1.0 } else { 100.0 })
        .collect();
    let strict = AperiodicConfig {
        minimum_r_squared: 0.99,
        maximum_rmse_log10: 0.01,
        ..fixed_config()
    };
    match fit_aperiodic(&frequencies, &density, strict) {
        Err(AperiodicError::FitQualityFailed {
            r_squared,
            rmse_log10,
        }) => {
            assert!(r_squared.is_finite() && rmse_log10.is_finite());
        }
        other => panic!("expected a typed quality failure, got {other:?}"),
    }
}

#[test]
fn config_validation_rejects_every_out_of_domain_field() {
    let base = knee_config();
    let broken = [
        AperiodicConfig {
            low_hz: 0.0,
            ..base
        },
        AperiodicConfig {
            high_hz: 1.0,
            ..base
        },
        AperiodicConfig {
            low_hz: f64::NAN,
            ..base
        },
        AperiodicConfig {
            maximum_exponent: 0.1,
            ..base
        },
        AperiodicConfig {
            maximum_knee: -1.0,
            ..base
        },
        AperiodicConfig {
            grid_points: 2,
            ..base
        },
        AperiodicConfig {
            refinement_rounds: 0,
            ..base
        },
        AperiodicConfig {
            peak_removal_percentile: 0.0,
            ..base
        },
        AperiodicConfig {
            peak_removal_percentile: 100.5,
            ..base
        },
        AperiodicConfig {
            peak_removal_tolerance_log10: f64::NAN,
            ..base
        },
        AperiodicConfig {
            minimum_r_squared: 1.5,
            ..base
        },
        AperiodicConfig {
            maximum_rmse_log10: 0.0,
            ..base
        },
    ];
    for config in broken {
        assert!(
            matches!(config.validate(), Err(AperiodicError::InvalidConfig(_))),
            "accepted {config:?}"
        );
    }
    assert!(base.validate().is_ok());
}

#[test]
fn insufficient_and_mismatched_inputs_are_typed() {
    let (frequencies, density) = synthetic(1.0, 0.0, 1.0);
    assert_eq!(
        fit_aperiodic(&frequencies, &density[..10], knee_config()),
        Err(AperiodicError::LengthMismatch)
    );
    let narrow = AperiodicConfig {
        low_hz: 39.5,
        high_hz: 39.9,
        ..knee_config()
    };
    assert!(matches!(
        fit_aperiodic(&frequencies, &density, narrow),
        Err(AperiodicError::InsufficientBins { needed: 4, .. })
    ));
}

#[test]
fn predicted_log10_is_defined_on_the_callers_grid_and_nan_at_dc() {
    let (frequencies, density) = synthetic(1.0, 2.0, 1.5);
    let fit = fit_aperiodic(&frequencies, &density, knee_config()).unwrap();
    assert_eq!(fit.predicted_log10.len(), frequencies.len());
    assert!(fit.predicted_log10[0].is_nan());
    assert!(fit.predicted_log10[1..].iter().all(|v| v.is_finite()));
}
