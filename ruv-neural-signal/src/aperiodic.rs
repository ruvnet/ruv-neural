//! Aperiodic (1/f) spectral fitting for NeuroSleep research profiles.
//!
//! The model is the one used by FOOOF/specparam 1.1.1, in log10 power:
//!
//! ```text
//! knee mode:  L(f) = offset - log10(knee + f^exponent)
//! fixed mode: L(f) = offset - log10(f^exponent) = offset - exponent * log10(f)
//! ```
//!
//! Note the exponent sits *inside* the logarithm in knee mode, so the knee model
//! is not linear in any transform of `f`. Fitting it with a straight line in
//! `log10(knee + f)` — a common shortcut — is a different model and yields a
//! different exponent.
//!
//! # What is and is not claimed
//!
//! The **model form** is FOOOF 1.1.1-compatible and the **robust peak-removal
//! strategy** follows FOOOF's flatten-and-threshold approach. The **optimiser is
//! not** FOOOF's: this crate uses a deterministic bounded grid search with
//! successive refinement and a closed-form offset, where FOOOF calls SciPy's
//! Levenberg-Marquardt `curve_fit`. Numerical parity with a FOOOF reference run
//! is therefore *not* claimed and is not covered by any test here; doing so
//! would require pinned reference spectra that are not in this repository.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Typed aperiodic fitting failures. A failed fit is a reportable outcome, not
/// a reason to substitute a default exponent.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum AperiodicError {
    /// A configuration field was non-finite or outside its declared domain.
    #[error("invalid aperiodic configuration: {0}")]
    InvalidConfig(&'static str),
    /// Frequency and density vectors had differing lengths.
    #[error("frequency and density lengths differ")]
    LengthMismatch,
    /// Too few usable in-band bins to identify the model.
    #[error("insufficient usable bins: need {needed}, have {actual}")]
    InsufficientBins { needed: usize, actual: usize },
    /// Robust peak removal discarded so many bins the model is unidentifiable.
    #[error("peak removal left only {retained} of {selected} bins")]
    OverPruned { retained: usize, selected: usize },
    /// The fit converged but did not meet the frozen quality gates.
    #[error("aperiodic fit quality failed: r_squared {r_squared}, rmse {rmse_log10}")]
    FitQualityFailed { r_squared: f64, rmse_log10: f64 },
}

/// Which aperiodic model form to fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AperiodicMode {
    /// No knee: `offset - exponent * log10(f)`.
    Fixed,
    /// With knee: `offset - log10(knee + f^exponent)`.
    Knee,
}

/// Frozen aperiodic fitting profile. Every field is validated before use.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AperiodicConfig {
    /// Lower fitting bound in Hz (must be > 0; log10 of DC is undefined).
    pub low_hz: f64,
    /// Upper fitting bound in Hz.
    pub high_hz: f64,
    /// Model form.
    pub mode: AperiodicMode,
    /// Inclusive lower bound of the exponent search.
    pub minimum_exponent: f64,
    /// Inclusive upper bound of the exponent search.
    pub maximum_exponent: f64,
    /// Inclusive upper bound of the knee search. Ignored in fixed mode.
    pub maximum_knee: f64,
    /// Samples per parameter axis in each grid-search round.
    pub grid_points: usize,
    /// Successive bracket-halving rounds after the coarse grid.
    pub refinement_rounds: usize,
    /// Flatten-and-threshold passes. Zero disables robust peak removal.
    pub peak_removal_iterations: usize,
    /// Percentile of the zero-clamped flattened spectrum used as the retention
    /// threshold. Because the flattened spectrum is clamped at zero, any value
    /// small enough to land on that clamp retains exactly the bins at or below
    /// the current fit — which is what FOOOF 1.1.1's `_ap_percentile_thresh`
    /// (0.025) does. Larger values retain progressively more of the peaks.
    pub peak_removal_percentile: f64,
    /// Minimum protrusion above the current fit, in log10 power, before a bin is
    /// treated as peak-contaminated. Without it the zero-clamp threshold splits
    /// a peak-free spectrum on floating-point noise and refits the aperiodic
    /// model to whichever half happened to land below the line. Setting it to a
    /// value at or below zero reproduces FOOOF's unguarded behaviour exactly.
    pub peak_removal_tolerance_log10: f64,
    /// Minimum coefficient of determination over the retained bins.
    pub minimum_r_squared: f64,
    /// Maximum RMSE in log10 power over the retained bins.
    pub maximum_rmse_log10: f64,
}

impl AperiodicConfig {
    /// Reject non-finite fields, inverted ranges, and degenerate search grids.
    pub fn validate(self) -> Result<Self, AperiodicError> {
        let finite = [
            self.low_hz,
            self.high_hz,
            self.minimum_exponent,
            self.maximum_exponent,
            self.maximum_knee,
            self.peak_removal_percentile,
            self.peak_removal_tolerance_log10,
            self.minimum_r_squared,
            self.maximum_rmse_log10,
        ];
        if finite.iter().any(|value| !value.is_finite()) {
            return Err(AperiodicError::InvalidConfig(
                "every numeric field must be finite",
            ));
        }
        if self.low_hz <= 0.0 || self.high_hz <= self.low_hz {
            return Err(AperiodicError::InvalidConfig(
                "fitting band must satisfy 0 < low_hz < high_hz",
            ));
        }
        if self.maximum_exponent <= self.minimum_exponent {
            return Err(AperiodicError::InvalidConfig(
                "exponent search range must be non-empty",
            ));
        }
        if self.maximum_knee < 0.0 {
            return Err(AperiodicError::InvalidConfig("knee bound must be >= 0"));
        }
        if self.grid_points < 3 || self.refinement_rounds == 0 {
            return Err(AperiodicError::InvalidConfig(
                "search needs >= 3 grid points and >= 1 refinement round",
            ));
        }
        if self.peak_removal_percentile <= 0.0 || self.peak_removal_percentile > 100.0 {
            return Err(AperiodicError::InvalidConfig(
                "peak removal percentile must lie in (0, 100]",
            ));
        }
        if !(0.0..=1.0).contains(&self.minimum_r_squared) || self.maximum_rmse_log10 <= 0.0 {
            return Err(AperiodicError::InvalidConfig("invalid fit quality gates"));
        }
        Ok(self)
    }
}

/// A converged aperiodic fit and the diagnostics needed to judge it.
#[derive(Debug, Clone, PartialEq)]
pub struct AperiodicFit {
    /// Aperiodic exponent (dimensionless, positive for a falling spectrum).
    pub exponent: f64,
    /// Aperiodic offset in log10(uV^2/Hz).
    pub offset: f64,
    /// Fitted knee, or `None` in fixed mode.
    pub knee: Option<f64>,
    /// Coefficient of determination over the retained (peak-removed) bins.
    pub r_squared: f64,
    /// RMSE in log10 power over the retained bins.
    pub rmse_log10: f64,
    /// RMSE in log10 power over *all* in-band bins, peaks included. Always
    /// >= `rmse_log10`; large values indicate strong periodic activity.
    pub full_band_rmse_log10: f64,
    /// Bins the robust passes kept, out of the in-band selection.
    pub retained_bins: usize,
    /// In-band bins before peak removal.
    pub selected_bins: usize,
    /// Model evaluated on the caller's full frequency grid, in log10 power.
    /// Entries at `f <= 0` are `NaN` because the model is undefined there.
    pub predicted_log10: Vec<f64>,
}

impl AperiodicFit {
    /// Unit tag for [`AperiodicFit::offset`] and the RMSE fields.
    pub const OFFSET_UNIT: &'static str = "log10_uV2_per_hz";
    /// Unit tag for [`AperiodicFit::exponent`].
    pub const EXPONENT_UNIT: &'static str = "dimensionless";
}

/// Fit the aperiodic component of a PSD with robust iterative peak removal.
pub fn fit_aperiodic(
    frequencies: &[f64],
    density: &[f64],
    config: AperiodicConfig,
) -> Result<AperiodicFit, AperiodicError> {
    let config = config.validate()?;
    if frequencies.len() != density.len() {
        return Err(AperiodicError::LengthMismatch);
    }
    let selected: Vec<(f64, f64)> = frequencies
        .iter()
        .copied()
        .zip(density.iter().copied())
        .filter(|(frequency, power)| {
            frequency.is_finite()
                && power.is_finite()
                && *frequency >= config.low_hz
                && *frequency <= config.high_hz
                && *frequency > 0.0
                && *power > 0.0
        })
        .map(|(frequency, power)| (frequency, power.log10()))
        .collect();
    // Three parameters (offset, knee, exponent) need at least four constraints
    // to be more than an interpolation.
    if selected.len() < 4 {
        return Err(AperiodicError::InsufficientBins {
            needed: 4,
            actual: selected.len(),
        });
    }

    let mut retained = selected.clone();
    let mut parameters = solve(&retained, config);
    for _ in 0..config.peak_removal_iterations {
        let next = remove_peaks(&selected, parameters, config);
        if next.len() < 4 {
            return Err(AperiodicError::OverPruned {
                retained: next.len(),
                selected: selected.len(),
            });
        }
        parameters = solve(&next, config);
        retained = next;
    }

    let (r_squared, rmse_log10) = goodness(&retained, parameters);
    let (_, full_band_rmse_log10) = goodness(&selected, parameters);
    if r_squared < config.minimum_r_squared || rmse_log10 > config.maximum_rmse_log10 {
        return Err(AperiodicError::FitQualityFailed {
            r_squared,
            rmse_log10,
        });
    }
    Ok(AperiodicFit {
        exponent: parameters.exponent,
        offset: parameters.offset,
        knee: matches!(config.mode, AperiodicMode::Knee).then_some(parameters.knee),
        r_squared,
        rmse_log10,
        full_band_rmse_log10,
        retained_bins: retained.len(),
        selected_bins: selected.len(),
        predicted_log10: frequencies
            .iter()
            .map(|frequency| {
                if frequency.is_finite() && *frequency > 0.0 {
                    parameters.evaluate(*frequency)
                } else {
                    f64::NAN
                }
            })
            .collect(),
    })
}

/// The three FOOOF aperiodic parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Parameters {
    offset: f64,
    knee: f64,
    exponent: f64,
}

impl Parameters {
    /// `offset - log10(knee + f^exponent)`.
    fn evaluate(self, frequency: f64) -> f64 {
        self.offset - (self.knee + frequency.powf(self.exponent)).log10()
    }
}

/// Sum of squared residuals for a (knee, exponent) pair, with the offset solved
/// in closed form: for fixed shape, the least-squares offset is the mean of
/// `y + log10(knee + f^exponent)`.
fn best_offset(points: &[(f64, f64)], knee: f64, exponent: f64) -> (f64, f64) {
    let shape: Vec<f64> = points
        .iter()
        .map(|(frequency, _)| (knee + frequency.powf(exponent)).log10())
        .collect();
    let offset = points
        .iter()
        .zip(&shape)
        .map(|((_, y), s)| y + s)
        .sum::<f64>()
        / points.len() as f64;
    let residual_sum_squares = points
        .iter()
        .zip(&shape)
        .map(|((_, y), s)| (y - (offset - s)).powi(2))
        .sum::<f64>();
    (offset, residual_sum_squares)
}

/// Deterministic bounded profile search.
///
/// Knee and exponent trade off along a narrow curved valley, so searching both
/// on one joint grid and then shrinking around the winner pins the pair to
/// whichever point of the valley the coarse grid happened to sample. Instead the
/// exponent is driven to convergence *inside* every knee candidate, so the outer
/// knee search sees a properly minimised profile error rather than a slice
/// through the valley wall. In fixed mode the knee axis is the single point 0
/// and this reduces to the 1-D exponent search.
fn solve(points: &[(f64, f64)], config: AperiodicConfig) -> Parameters {
    let knee_bounds = match config.mode {
        AperiodicMode::Fixed => (0.0, 0.0),
        AperiodicMode::Knee => (0.0, config.maximum_knee),
    };
    let mut knee_range = knee_bounds;
    let mut best = Parameters {
        offset: 0.0,
        knee: knee_bounds.0,
        exponent: config.minimum_exponent,
    };
    let mut best_error = f64::INFINITY;

    for _ in 0..config.refinement_rounds {
        for knee in axis(knee_range, config.grid_points) {
            let (exponent, offset, error) = profile_exponent(points, knee, config);
            if error < best_error {
                best_error = error;
                best = Parameters {
                    offset,
                    knee,
                    exponent,
                };
            }
        }
        if knee_bounds.1 <= knee_bounds.0 {
            break;
        }
        knee_range = shrink(knee_range, best.knee, config.grid_points, knee_bounds);
    }
    best
}

/// Minimise the residual over the exponent for one fixed knee, returning the
/// winning exponent, its closed-form offset, and the residual sum of squares.
fn profile_exponent(points: &[(f64, f64)], knee: f64, config: AperiodicConfig) -> (f64, f64, f64) {
    let bounds = (config.minimum_exponent, config.maximum_exponent);
    let mut range = bounds;
    let mut best = (bounds.0, 0.0, f64::INFINITY);
    for _ in 0..config.refinement_rounds {
        for exponent in axis(range, config.grid_points) {
            let (offset, error) = best_offset(points, knee, exponent);
            if error < best.2 {
                best = (exponent, offset, error);
            }
        }
        range = shrink(range, best.0, config.grid_points, bounds);
    }
    best
}

fn axis((low, high): (f64, f64), points: usize) -> Vec<f64> {
    if high <= low {
        return vec![low];
    }
    (0..points)
        .map(|index| low + (high - low) * index as f64 / (points - 1) as f64)
        .collect()
}

/// Re-centre a search range on the winner, keeping two grid steps either side and
/// clamping to the original outer bounds.
///
/// Two steps rather than one matters: knee and exponent trade off along a narrow
/// curved valley, so a bracket that collapses to a single step around the round's
/// winner pins one coordinate before the other can follow the valley, and the
/// search stalls short of the optimum.
fn shrink(
    (low, high): (f64, f64),
    center: f64,
    points: usize,
    (outer_low, outer_high): (f64, f64),
) -> (f64, f64) {
    if high <= low {
        return (low, high);
    }
    let reach = 2.0 * (high - low) / (points - 1) as f64;
    (
        (center - reach).max(outer_low),
        (center + reach).min(outer_high),
    )
}

/// FOOOF-style robust peak removal: flatten the spectrum by the current fit,
/// clamp negative residuals to zero, and keep only bins at or below the
/// configured percentile of the flattened spectrum.
///
/// The clamp is what makes this robust. Every bin sitting below the fit collapses
/// to exactly zero, so a small percentile threshold selects that whole set and
/// discards everything protruding above the fit — i.e. the periodic peaks.
///
/// Each pass re-derives its retained set from the full in-band selection rather
/// than from the previous pass's survivors, so iterating converges on a fixed
/// point instead of eroding the support set.
fn remove_peaks(
    selected: &[(f64, f64)],
    parameters: Parameters,
    config: AperiodicConfig,
) -> Vec<(f64, f64)> {
    let flattened: Vec<f64> = selected
        .iter()
        .map(|(frequency, y)| (y - parameters.evaluate(*frequency)).max(0.0))
        .collect();
    let mut sorted = flattened.clone();
    sorted.sort_by(f64::total_cmp);
    // Nearest-rank percentile: deterministic and free of interpolation choices.
    let rank = ((config.peak_removal_percentile / 100.0) * sorted.len() as f64).ceil() as usize;
    let threshold =
        sorted[rank.clamp(1, sorted.len()) - 1].max(config.peak_removal_tolerance_log10);
    selected
        .iter()
        .zip(&flattened)
        .filter(|(_, value)| **value <= threshold)
        .map(|(point, _)| *point)
        .collect()
}

fn goodness(points: &[(f64, f64)], parameters: Parameters) -> (f64, f64) {
    let mean = points.iter().map(|(_, y)| *y).sum::<f64>() / points.len() as f64;
    let residual_sum_squares = points
        .iter()
        .map(|(frequency, y)| (y - parameters.evaluate(*frequency)).powi(2))
        .sum::<f64>();
    let total_sum_squares = points.iter().map(|(_, y)| (y - mean).powi(2)).sum::<f64>();
    let r_squared = if total_sum_squares > 0.0 {
        1.0 - residual_sum_squares / total_sum_squares
    } else {
        1.0
    };
    (
        r_squared,
        (residual_sum_squares / points.len() as f64).sqrt(),
    )
}

#[cfg(test)]
mod tests;
